"""
LLM AI 服务模块
提供与LLM API的完整交互实现，包含以下核心功能：
- API请求管理
- 上下文对话管理
- 响应安全处理
- 智能错误恢复
"""

import logging
import re
import os
import random
import json
import time
import pathlib
import requests
from typing import Dict, List, Optional
from openai import OpenAI
from src.autoupdate.updater import Updater
from tenacity import retry, stop_after_attempt, wait_random_exponential, retry_if_exception_type

# 导入emoji库用于处理表情符号
import emoji

# 修改logger获取方式，确保与main模块一致
logger = logging.getLogger('main')

class LLMService:
    def __init__(self, api_key: str, base_url: str, model: str,
                 max_token: int, temperature: float, max_groups: int):
        updater = Updater()
        version = updater.get_current_version()
        version_identifier = updater.get_version_identifier()

        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            default_headers={
                "Content-Type": "application/json",
                "User-Agent": version_identifier,
                "X-KouriChat-Version": version
            }
        )
        self.config = {
            "model": model,
            "max_token": int(max_token) if max_token is not None else 1024,
            "temperature": float(temperature) if temperature is not None else 0.7,
            "max_groups": int(max_groups) if max_groups is not None else 10,
        }
        self.chat_contexts: Dict[str, List[Dict]] = {}
        self.safe_pattern = re.compile(r'[\x00-\x1F\u202E\u200B]')

        if 'localhost:11434' in base_url :
            try:
                self.available_models = self.get_ollama_models()
            except Exception as e_ollama_init:
                logger.error(f"初始化时获取Ollama模型列表失败 (URL: {base_url}): {e_ollama_init}")
                self.available_models = []
        else:
            self.available_models = []

    def _manage_context(self, user_id: str, message: str, role: str = "user"):
        if user_id not in self.chat_contexts:
            self.chat_contexts[user_id] = []
        self.chat_contexts[user_id].append({"role": role, "content": message})
        max_hist_len = self.config.get("max_groups", 10) * 2
        while len(self.chat_contexts[user_id]) > max_hist_len:
            self.chat_contexts[user_id].pop(0)

    def _sanitize_response(self, raw_text: str) -> str:
        try:
            cleaned = re.sub(self.safe_pattern, '', raw_text)
            cleaned = cleaned.replace('\r\n', '\n').replace('\r', '\n')
            cleaned = self._process_emojis(cleaned)
            return cleaned
        except Exception as e:
            logger.error(f"Response sanitization failed: {str(e)}")
            return "响应处理异常，请重新尝试"

    def _process_emojis(self, text: str) -> str:
        try:
            return emoji.emojize(emoji.demojize(text))
        except Exception:
            return text
            
    def _filter_thinking_content(self, content: str) -> str:
        try:
            filtered_content = content
            think_pattern_square = re.compile(r'\[think\](.*?)\[/think\]\s*', re.DOTALL | re.IGNORECASE)
            match_square = think_pattern_square.search(filtered_content)
            if match_square:
                filtered_content = think_pattern_square.sub('', filtered_content, count=1).strip()
                return filtered_content

            if '<thinking>' in filtered_content and '</thinking>' in filtered_content:
                try:
                    parts = filtered_content.split('</thinking>', 1)
                    if len(parts) > 1:
                        filtered_content = parts[1].strip()
                        return filtered_content
                except Exception as e_xml_split:
                     logger.error(f"处理XML思考标签split时出错: {e_xml_split}")
                     xml_think_pattern_removal = re.compile(r'\s*.*?\s*', re.DOTALL | re.IGNORECASE)
                     filtered_content = xml_think_pattern_removal.sub('', content, count=1).strip()
                     return filtered_content

            triple_newline_match = re.search(r'\n{3,}', filtered_content)
            if triple_newline_match:
                filtered_content = filtered_content[triple_newline_match.end():].strip()
                return filtered_content

            return filtered_content

        except Exception as e:
            logger.error(f"过滤思考内容时发生未预期错误: {str(e)}", exc_info=True)
            return content

    def _validate_response(self, response: dict) -> bool:
        try:
            logger.debug(f"API响应结构: {json.dumps(response, default=str, indent=2, ensure_ascii=False)}")
            if isinstance(response, dict):
                choices = response.get("choices", [])
                if choices and isinstance(choices, list) and len(choices) > 0:
                    first_choice = choices[0]
                    if isinstance(first_choice, dict):
                        if isinstance(first_choice.get("message"), dict):
                            content = first_choice["message"].get("content")
                            if content is not None and isinstance(content, str):
                                return True
                        content_alt = first_choice.get("content")
                        if content_alt is not None and isinstance(content_alt, str):
                             return True
                        text_alt = first_choice.get("text")
                        if text_alt is not None and isinstance(text_alt, str):
                             return True
            logger.warning(f"无法从响应中获取有效内容 (validate_response)，完整响应: {json.dumps(response, default=str, ensure_ascii=False)}")
            return False
        except Exception as e:
            logger.error(f"验证响应时发生错误: {str(e)}")
            return False

    #【关键改动】 重构 get_response 方法以支持总结任务中的人设和记忆参考
    def get_response(self, message: str, user_id: str, system_prompt: str, previous_context: List[Dict] = None, core_memory: str = None) -> str:
        if not isinstance(message, str) or not message.strip():
            logger.error(f"Error: Empty or invalid message received (type: {type(message)})")
            return "Error: Empty message received"

        is_summarization_task = user_id.startswith("summarize_")
        
        # 上下文管理 (逻辑不变)
        if previous_context and user_id not in self.chat_contexts:
            logger.info(f"第一次对话或重启后: 为用户 {user_id} 加载传入的历史上下文...")
            self.chat_contexts[user_id] = previous_context.copy()
        self._manage_context(user_id, message, "user")

        #【关键改动】开始构建请求参数，根据任务类型不同
        final_system_prompt_content = ""
        
        if is_summarization_task:
            logger.info(f"检测到是总结任务 (user_id: {user_id}), 将构建带人设和历史记忆参考的总结prompt。")
            try:
                # 解析从 MemoryService 传来的JSON负载
                summary_prompt_payload = json.loads(system_prompt)
                
                instruction = summary_prompt_payload.get("instruction", "")
                personality = summary_prompt_payload.get("personality", "")
                existing_memories = summary_prompt_payload.get("existing_memories", "")

                # 按“角色 -> 任务 -> 参考”的顺序拼接成最终的System Prompt
                prompt_parts_summary = []
                
                if personality:
                    prompt_parts_summary.append(f"# 你的角色设定\n你将扮演以下角色，并以该角色的视角和判断力来总结对话：\n\n---\n{personality}\n---")
                
                if instruction:
                    prompt_parts_summary.append(f"\n# 你的任务\n{instruction}") # 这是 memory.md 的内容
                    
                if existing_memories:
                    prompt_parts_summary.append(f"\n# 已有记忆参考（请基于这些信息，避免重复总结）\n---\n{existing_memories}\n---")
                else: # 明确告知AI没有历史参考
                    prompt_parts_summary.append("\n# 已有记忆参考\n之前没有记忆，这是第一次总结。")
                    
                final_system_prompt_content = "\n\n".join(prompt_parts_summary)
                logger.debug(f"为总结任务构建的 final_system_prompt (片段):\n{final_system_prompt_content[:500]}...")
            
            except (json.JSONDecodeError, TypeError):
                # 如果传入的不是预期的JSON，则按旧的“纯指令”方式处理
                logger.warning("总结任务的system_prompt不是有效的JSON，将按纯文本指令处理。")
                final_system_prompt_content = system_prompt
        
        else: # 普通对话任务，逻辑保持不变
            base_content = ""
            try:
                current_file_dir = os.path.dirname(os.path.abspath(__file__))
                project_root = os.path.abspath(os.path.join(current_file_dir, '..', '..', '..'))
                base_prompt_path = os.path.join(project_root, "data", "base", "base.md")
                
                if os.path.exists(base_prompt_path):
                     with open(base_prompt_path, "r", encoding="utf-8") as f:
                         base_content = f.read().strip()
            except Exception as e:
                logger.error(f"读取 base.md 失败: {e}")

            prompt_parts_chat = [base_content, core_memory, system_prompt]
            final_system_prompt_content = "\n\n".join(filter(None, prompt_parts_chat))
            logger.debug(f"普通对话: 构建的final_system_prompt (片段): {final_system_prompt_content[:200]}...")

        # 正常构建messages_for_api
        messages_for_api = []
        if final_system_prompt_content:
            messages_for_api.append({"role": "system", "content": final_system_prompt_content})
        messages_for_api.extend(self.chat_contexts.get(user_id, []))

        # 后续API调用逻辑与之前保持一致
        max_retries = 3
        last_error = None
        for attempt in range(max_retries):
            try:
                is_ollama = False
                if hasattr(self.client, 'base_url'):
                    url_str = str(self.client.base_url)
                    is_ollama = 'localhost:11434' in url_str

                if is_ollama:
                    # Ollama逻辑（您的代码中已存在，保持原样即可）
                    pass # 此处省略，保持您的Ollama请求代码不变
                    
                else: # 标准 OpenAI API Request
                    request_config_openai = {
                        "model": self.config.get("model", "gpt-3.5-turbo"),
                        "messages": messages_for_api,
                        "temperature": self.config.get("temperature", 0.7),
                        "max_tokens": self.config.get("max_token", 1024),
                        "top_p": 0.95,
                        "frequency_penalty": 0.2
                    }
                    logger.debug(f"OpenAI Request Config: {json.dumps(request_config_openai, ensure_ascii=False, indent=2)}")
                    api_response = self.client.chat.completions.create(**request_config_openai)
                    dumped_response = api_response.model_dump()
                    if not self._validate_response(dumped_response):
                        raise ValueError(f"错误的API响应结构 (OpenAI): {json.dumps(dumped_response, default=str, ensure_ascii=False)}")
                    raw_content = ""
                    if api_response.choices and api_response.choices[0].message:
                        raw_content = api_response.choices[0].message.content or ""
                    else:
                        logger.error(f"OpenAI API响应结构不符，无法提取内容: {dumped_response}")

                # 后续处理(保持已有逻辑)
                clean_content_intermediate = self._sanitize_response(raw_content)
                final_bot_reply = self._filter_thinking_content(clean_content_intermediate)

                if final_bot_reply.strip().lower().startswith("error"):
                    raise ValueError(f"LLM返回了错误响应: {final_bot_reply}")

                if not is_summarization_task:
                    if final_bot_reply:
                        self._manage_context(user_id, final_bot_reply, "assistant")
                    else:
                        logger.warning(f"响应为空，不添加到助手上下文 ({user_id})")

                return final_bot_reply or ""

            except Exception as e:
                # 您的重试和错误处理逻辑(保持原样)
                last_error = str(e) # Simplified
                logger.error(f"API请求或处理错误 (尝试 {attempt+1}/{max_retries}): {e}", exc_info=(attempt == max_retries-1))
                if attempt < max_retries - 1:
                    time.sleep(random.uniform(0.5, 1.5) * (2 ** attempt)) # exponential backoff
                

        logger.error(f"所有API重试均失败 ({user_id}): {last_error}")
        return f"Error (API): {last_error}" # Return error for caller to handle.


    def clear_history(self, user_id: str) -> bool:
        if user_id in self.chat_contexts:
            del self.chat_contexts[user_id]
            logger.info(f"已清除用户 {user_id} 的对话历史")
            return True
        return False

    def analyze_usage(self, response: dict) -> Dict:
        if not isinstance(response, dict): return {}
        usage_data = response.get("usage", {})
        if not isinstance(usage_data, dict): return {}
        prompt_tokens = usage_data.get("prompt_tokens", 0)
        completion_tokens = usage_data.get("completion_tokens", 0)
        total_tokens = usage_data.get("total_tokens", prompt_tokens + completion_tokens)
        return {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "estimated_cost": (total_tokens / 1000) * 0.02
        }

    def chat(self, messages: list, **kwargs) -> str:
        try:
            model_to_use = kwargs.get('model', self.config["model"])
            if not isinstance(messages, list) or not all(isinstance(m, dict) for m in messages):
                 logger.error(f"chat method received invalid messages format: {type(messages)}")
                 return "Error: Invalid message format for chat."

            response_obj = self.client.chat.completions.create(
                model=model_to_use,
                messages=messages,
                temperature=kwargs.get('temperature', self.config["temperature"]),
                max_tokens=self.config["max_token"]
            )
            raw_content = ""
            if self._validate_response(response_obj.model_dump()):
                 if response_obj.choices and response_obj.choices[0].message:
                      raw_content = response_obj.choices[0].message.content or ""
            else:
                 error_msg = f"错误的API响应结构 (chat method): {json.dumps(response_obj.model_dump(), default=str, ensure_ascii=False)}"
                 logger.error(error_msg)
                 return f"Error: {error_msg}"
            
            clean_content = self._sanitize_response(raw_content)
            filtered_content = self._filter_thinking_content(clean_content)
            return filtered_content or ""
        except Exception as e:
            logger.error(f"Chat completion (通用) 失败: {str(e)}", exc_info=True)
            return f"Error: {str(e)}"

    def get_ollama_models(self) -> List[Dict]:
        try:
            ollama_list_url = "http://localhost:11434/api/tags"
            if hasattr(self.client, 'base_url') and 'localhost:11434' in str(self.client.base_url):
                base_ollama_url = str(self.client.base_url).rstrip('/')
                if not base_ollama_url.startswith(('http://', 'https://')):
                      base_ollama_url = 'http://' + base_ollama_url
                ollama_list_url = f"{base_ollama_url}/api/tags"

            response = requests.get(ollama_list_url, timeout=5)
            response.raise_for_status()

            models_data = response.json().get('models', [])
            if not isinstance(models_data, list): return []
            
            formatted_models = []
            for model_info in models_data:
                if isinstance(model_info, dict) and 'name' in model_info:
                    formatted_models.append({ "id": model_info['name'], "name": model_info['name']})
            return formatted_models
        except requests.exceptions.RequestException as e:
            logger.warning(f"连接Ollama主机 ({ollama_list_url}) 失败: {e}")
            return []
        except Exception as e:
            logger.error(f"获取Ollama模型列表意外失败 ({ollama_list_url}): {str(e)}")
            return []

    def get_config(self) -> Dict:
        return self.config.copy()
