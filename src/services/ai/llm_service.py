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
import pathlib # 虽然没直接用，但保留导入
import requests
from typing import Dict, List, Optional, Tuple, Union
from openai import OpenAI
from src.autoupdate.updater import Updater # 假设路径正确
from tenacity import ( # 假设这些也正确导入
    retry,
    stop_after_attempt,
    wait_random_exponential,
    retry_if_exception_type
)

# 导入emoji库用于处理表情符号
import emoji

# 修改logger获取方式，确保与main模块一致
logger = logging.getLogger('main')

class LLMService:
    def __init__(self, api_key: str, base_url: str, model: str,
                 max_token: int, temperature: float, max_groups: int):
        """
        强化版AI服务初始化

        :param api_key: API认证密钥
        :param base_url: API基础URL
        :param model: 使用的模型名称
        :param max_token: 最大token限制
        :param temperature: 创造性参数(0~2)
        :param max_groups: 最大对话轮次记忆
        """
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
            # 确保 max_token 是整数，如果原始配置可能是字符串
            "max_token": int(max_token) if max_token is not None else 1024, # Default if None
            "temperature": float(temperature) if temperature is not None else 0.7, # Default if None
            "max_groups": int(max_groups) if max_groups is not None else 10, # Default if None
        }
        self.chat_contexts: Dict[str, List[Dict]] = {}
        self.safe_pattern = re.compile(r'[\x00-\x1F\u202E\u200B]')

        if 'localhost:11434' in base_url : # 更安全的检查
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
        # 确保 self.config["max_groups"] 是有效的
        max_hist_len = self.config.get("max_groups", 10) * 2
        while len(self.chat_contexts[user_id]) > max_hist_len:
            self.chat_contexts[user_id].pop(0) # 从最前面移除

    def _sanitize_response(self, raw_text: str) -> str:
        # ... (代码无变化) ...
        try:
            cleaned = re.sub(self.safe_pattern, '', raw_text)
            cleaned = cleaned.replace('\r\n', '\n').replace('\r', '\n')
            cleaned = self._process_emojis(cleaned)
            return cleaned
        except Exception as e:
            logger.error(f"Response sanitization failed: {str(e)}")
            return "响应处理异常，请重新尝试"

    def _process_emojis(self, text: str) -> str:
        # ... (代码无变化) ...
        try:
            return emoji.emojize(emoji.demojize(text))
        except Exception:
            return text

    def _filter_thinking_content(self, content: str) -> str:
        """
        过滤思考内容，支持多种格式 (修复版)。
        1. [think]思考过程[/think]
        2. ...思考过程
        3. 
思考过程
"""
        try:
            filtered_content = content

            # 1. 处理 [think]...[/think] 格式
            think_pattern_square = re.compile(r'\[think\](.*?)\[/think\]\s*', re.DOTALL | re.IGNORECASE)
            match_square = think_pattern_square.search(filtered_content)
            if match_square:
                # 思考内容日志记录 (可选, 可取消注释)
                # think_text_square = match_square.group(1).strip()
                # logger.info(f"提取到 AI 思考内容 (方括号): {think_text_square}")
                filtered_content = think_pattern_square.sub('', filtered_content, count=1).strip()
                # 优先处理方括号格式后直接返回，避免重复处理
                return filtered_content

            # 2. 处理 


            #    简单检查是否存在标签，然后用 split 分割
            if '<thinking>' in filtered_content and '</thinking>' in filtered_content:
                try:
                    # 按第一个 '</thinking>' 分割
                    parts = filtered_content.split('</thinking>', 1)
                    
                    # parts[0] 包含 '<thinking>' 和思考内容
                    # parts[1] 包含 '</thinking>' 之后的目标回复内容
                    
                    if len(parts) > 1:
                         # 提取思考内容 (可选, 可取消注释)
                         # think_part = parts[0]
                         # start_think_tag = think_part.find('<thinking>')
                         # if start_think_tag != -1:
                         #     think_content_xml = think_part[start_think_tag + len('<thinking>'):].strip()
                         #     logger.info(f"提取到 AI 思考内容 (XML标签): {think_content_xml}")

                        # 获取目标回复内容并去除前后空白
                        filtered_content = parts[1].strip()
                        # 找到XML格式并处理后直接返回
                        return filtered_content

                except Exception as e_xml_split:
                     logger.error(f"处理XML思考标签split时出错: {e_xml_split}")
                     # 如果split出错，尝试回退到简单的正则替换（但通常split更可靠）
                     xml_think_pattern_removal = re.compile(r'\s*.*?\s*', re.DOTALL | re.IGNORECASE)
                     filtered_content = xml_think_pattern_removal.sub('', content, count=1).strip()
                     # 已经尝试过补救，可以返回了
                     return filtered_content

            # 3. 处理 R1 格式 (思考过程...\n\n\n最终回复)
            #    这个检查应该在确定没有上述标签格式后再进行
            triple_newline_match = re.search(r'\n{3,}', filtered_content) # 匹配3个或更多连续换行符
            if triple_newline_match:
                # 只保留三个或更多连续换行符后面的内容（最终回复）
                filtered_content = filtered_content[triple_newline_match.end():].strip()
                # R1格式处理完返回
                return filtered_content

            # 如果以上格式都未匹配，则返回原始内容
            return filtered_content

        except Exception as e:
            logger.error(f"过滤思考内容时发生未预期错误: {str(e)}", exc_info=True)
            # 出错时返回原始内容作为保险
            return content


    def _validate_response(self, response: dict) -> bool:
        # ... (代码无变化) ...
        try:
            logger.debug(f"API响应结构: {json.dumps(response, default=str, indent=2, ensure_ascii=False)}")
            if isinstance(response, dict):
                choices = response.get("choices", [])
                if choices and isinstance(choices, list) and len(choices) > 0:
                    first_choice = choices[0]
                    if isinstance(first_choice, dict):
                        if isinstance(first_choice.get("message"), dict):
                            content = first_choice["message"].get("content")
                            if content is not None and isinstance(content, str): # Check for None explicitly
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

    # <<< --- 方法开始处 (在 LLMService 类内部) --- >>>
    def get_response(self, message: str, user_id: str, system_prompt: str, previous_context: List[Dict] = None, core_memory: str = None) -> str:
        """
        完整请求处理流程。已修正 '_config' NameError。
        Args:
            message: 用户消息
            user_id: 用户ID
            system_prompt: 系统提示词（人设 或 针对特定任务的指令如总结提示）
            previous_context: 历史上下文（可选）
            core_memory: 核心记忆（可选）
        """
        if not isinstance(message, str) or not message.strip():
            logger.error(f"Error: Empty or invalid message received (type: {type(message)})")
            return "Error: Empty message received"

        is_summarization_task = user_id.startswith("summarize_")
        if is_summarization_task:
            logger.info(f"检测到是总结任务 (user_id: {user_id}), 将仅使用传入的system_prompt (memory.md) 作为主要指令。")

        # Context Management
        if previous_context and user_id not in self.chat_contexts:
            logger.info(f"第一次对话或重启后: 为用户 {user_id} 加载传入的历史上下文，共 {len(previous_context)} 条消息")
            self.chat_contexts[user_id] = previous_context.copy()

        self._manage_context(user_id, message, "user")

        # Build Request Parameters
        base_content = ""
        if not is_summarization_task:
            # Try reading base.md only for non-summarization tasks
            try:
                # Determine project root dynamically (ensure this works in your deployment)
                current_file_dir = os.path.dirname(os.path.abspath(__file__))
                # Assuming structure is src/services/ai/llm_service.py
                project_root = os.path.abspath(os.path.join(current_file_dir, '..', '..', '..')) # Navigate up 3 levels
                base_prompt_path = os.path.join(project_root, "data", "base", "base.md")
                
                if not os.path.exists(base_prompt_path):
                     logger.warning(f"基础Prompt文件 (base.md) 未找到于推测路径: {base_prompt_path}")
                else:
                     with open(base_prompt_path, "r", encoding="utf-8") as f:
                         base_content = f.read().strip()
                     logger.debug(f"普通对话任务: 已加载 base.md, 长度: {len(base_content)}")

            except Exception as e:
                logger.error(f"普通对话任务中读取基础Prompt文件 (base.md) 失败: {str(e)}")

        # Construct Final System Prompt Content
        if is_summarization_task:
            final_system_prompt_content = system_prompt
            logger.debug(f"总结任务: final_system_prompt 使用传入的 memory.md 内容 (片段): {final_system_prompt_content[:200]}...")
        else:
            prompt_parts = []
            if base_content: prompt_parts.append(base_content)
            if core_memory: prompt_parts.append(core_memory)
            if system_prompt: prompt_parts.append(system_prompt)
            final_system_prompt_content = "\n\n".join(filter(None, prompt_parts))
            logger.debug(f"普通对话: 构建的final_system_prompt (片段): {final_system_prompt_content[:200]}...")

        # Construct message list for API
        current_user_chat_history = self.chat_contexts.get(user_id, [])
        messages_for_api = []
        if final_system_prompt_content:
            messages_for_api.append({"role": "system", "content": final_system_prompt_content})
        messages_for_api.extend(current_user_chat_history)

        # Check for Ollama
        is_ollama = False
        if hasattr(self.client, 'base_url') and isinstance(self.client.base_url, str):
             is_ollama = 'localhost:11434' in self.client.base_url
        elif isinstance(self.client.base_url, object) and hasattr(self.client.base_url, 'host') and isinstance(self.client.base_url.host, str):
            # Handle cases where base_url is an object (like httpx._urls.URL)
             is_ollama = 'localhost' ==self.client.base_url.host.lower() and self.client.base_url.port == 11434


        max_retries = 3
        last_error = None

        # --- API Call Loop ---
        for attempt in range(max_retries):
            try:
                request_body_log = {"model": self.config.get("model", "unknown"), "messages_count": len(messages_for_api)} # Use .get for safety
                logger.debug(f"准备API请求 (尝试 {attempt+1}): {request_body_log}, is_ollama: {is_ollama}")

                if is_ollama:
                    # --- Ollama API Request ---
                    model_name_for_ollama = self.config.get("model", "")
                    if '/' in model_name_for_ollama:
                        model_name_for_ollama = model_name_for_ollama.split('/')[-1]

                    ollama_max_tokens = self.config.get("max_token", -1)
                    if not isinstance(ollama_max_tokens, int) or ollama_max_tokens <= 0:
                        ollama_max_tokens = -1

                    request_config_ollama = {
                        "model": model_name_for_ollama,
                        "messages": messages_for_api,
                        "stream": False,
                        "options": {
                            "temperature": self.config.get("temperature", 0.7), # Use .get with default
                            "num_predict": ollama_max_tokens
                        }
                    }

                    # Construct the API URL robustly
                    ollama_api_url = "http://localhost:11434/api/chat" # Default
                    if hasattr(self.client, 'base_url') and self.client.base_url:
                        try:
                           # Preferred way: use base_url from the client if available
                           base_url_str = str(self.client.base_url).rstrip('/')
                           if not base_url_str.startswith(('http://', 'https://')):
                                base_url_str = 'http://' + base_url_str
                           ollama_api_url = f"{base_url_str}/api/chat"
                        except Exception as e_url:
                            logger.warning(f"无法从self.client.base_url构造Ollama URL ({self.client.base_url}): {e_url}, 使用默认值。")

                    # Get headers (need updater instance or move init)
                    updater_instance = Updater() # Re-init updater for each request - consider Class level init
                    version = updater_instance.get_current_version()
                    version_identifier = updater_instance.get_version_identifier()
                    headers = {
                        "Content-Type": "application/json",
                        "User-Agent": version_identifier,
                        "X-KouriChat-Version": version
                    }

                    logger.debug(f"Ollama Request to {ollama_api_url} Config: {json.dumps(request_config_ollama, ensure_ascii=False, indent=2)}")
                    api_response = requests.post(ollama_api_url, json=request_config_ollama, headers=headers, timeout=60) # Added timeout
                    api_response.raise_for_status()
                    response_data = api_response.json()
                    logger.debug(f"Ollama Raw Response Data: {json.dumps(response_data, ensure_ascii=False, indent=2)}")

                    # Extract content
                    if response_data and isinstance(response_data.get("message"), dict) and "content" in response_data["message"]:
                        raw_content = response_data["message"]["content"]
                    elif response_data and "choices" in response_data and response_data["choices"] and "message" in response_data["choices"][0] and "content" in response_data["choices"][0]["message"]:
                        raw_content = response_data["choices"][0]["message"]["content"]
                    else:
                        raise ValueError(f"Ollama响应中未找到预期的 'message.content' 或 'choices[0].message.content', 响应: {json.dumps(response_data, default=str, ensure_ascii=False)}")

                else:
                    # --- Standard OpenAI API Request ---
                    request_config_openai = {
                        "model": self.config.get("model", "gpt-3.5-turbo"), # Default Model
                        "messages": messages_for_api,
                        # --- VVVV THE FIX IS HERE VVVV ---
                        "temperature": self.config.get("temperature", 0.7), # Corrected: used self.config
                        # --- ^^^^ THE FIX IS HERE ^^^^ ---
                        "max_tokens": self.config.get("max_token", 1024),   # Use .get w/ default
                        "top_p": 0.95,                       # Consider config
                        "frequency_penalty": 0.2               # Consider config
                    }
                    logger.debug(f"OpenAI Request Config: {json.dumps(request_config_openai, ensure_ascii=False, indent=2)}")
                    start_time = time.time()
                    api_response = self.client.chat.completions.create(**request_config_openai)
                    end_time = time.time()
                    logger.debug(f"OpenAI API call duration: {end_time - start_time:.2f}s")

                    # Validate OpenAI Response
                    # model_dump() is preferred over dict() for pydantic v2+
                    dumped_response = api_response.model_dump()
                    if not self._validate_response(dumped_response):
                        raise ValueError(f"错误的API响应结构 (OpenAI): {json.dumps(dumped_response, default=str, ensure_ascii=False)}")
                    
                    # Extract content from OpenAI response
                    if api_response.choices and api_response.choices[0].message:
                        raw_content = api_response.choices[0].message.content
                        if raw_content is None: # Handle case where content is explicitly null
                            logger.warning("OpenAI API返回的content为None。")
                            raw_content = ""
                    else: # Handle case where structure is unexpected
                        logger.error(f"OpenAI API响应结构不符合预期，无法提取内容: {dumped_response}")
                        raw_content = ""

                # --- Common Post-API steps ---
                if not isinstance(raw_content, str):
                    logger.error(f"API返回的raw_content不是字符串: {type(raw_content)}, 内容: {raw_content}")
                    raw_content = str(raw_content)

                clean_content_intermediate = self._sanitize_response(raw_content)
                final_bot_reply = self._filter_thinking_content(clean_content_intermediate)

                if final_bot_reply.strip().lower().startswith("error"):
                     # Check if the error message is from our code or the LLM itself
                    if "Error:" in final_bot_reply[:10]: # Basic check
                         logger.warning(f"LLM似乎返回了错误指示: {final_bot_reply[:100]}...")
                         # Raise custom exception maybe? Or just ValueError works.
                    raise ValueError(f"LLM返回了错误响应: {final_bot_reply}")

                # Update assistant context ONLY for non-summary tasks
                if not is_summarization_task:
                      if final_bot_reply: # Ensure we don't add empty replies to context
                           self._manage_context(user_id, final_bot_reply, "assistant")
                      else:
                           logger.warning(f"成功获取API响应但处理后为空，不添加到助手上下文 ({user_id})")


                return final_bot_reply or "" # Return empty string if None/empty

            # --- Exception Handling for the attempt ---
            except requests.exceptions.RequestException as e_req:
                last_error = f"Error (Network): {str(e_req)}"
                logger.warning(f"API网络请求失败 (尝试 {attempt+1}/{max_retries}): {str(e_req)}")
            except Exception as e:
                last_error = f"Error (API or processing): {str(e)}"
                if attempt == max_retries - 1:
                    logger.error(f"API请求或处理失败 (尝试 {attempt+1}/{max_retries}): {str(e)}", exc_info=True) # Log full traceback on last attempt
                else:
                    logger.warning(f"API请求或处理失败 (尝试 {attempt+1}/{max_retries}): {str(e)}") # No traceback for intermediate fails


            # Exponential backoff before next retry
            if attempt < max_retries - 1:
                wait_seconds = random.uniform(0.5, 1.5) * (2 ** attempt)
                logger.info(f"等待 {wait_seconds:.2f} 秒后重试...")
                time.sleep(wait_seconds)

        # -- End of Rery Loop --- #

        logger.error(f"所有API重试尝试均失败 ({user_id}): {last_error}")
        # Return the specific error for internal handling, instead of generic message
        return last_error if last_error else "Error: API Unreachable after retries."


    def clear_history(self, user_id: str) -> bool:
        # ... (代码无变化) ...
        if user_id in self.chat_contexts:
            del self.chat_contexts[user_id]
            logger.info(f"已清除用户 {user_id} 的对话历史")
            return True
        logger.info(f"尝试清除历史但用户 {user_id} 不在上下文中。")
        return False

    def analyze_usage(self, response: dict) -> Dict:
        # ... (代码无变化) ...
        # Ensure response is a dict and usage is a dict for safety
        if not isinstance(response, dict): return {}
        usage_data = response.get("usage", {})
        if not isinstance(usage_data, dict): return {}

        prompt_tokens = usage_data.get("prompt_tokens", 0)
        completion_tokens = usage_data.get("completion_tokens", 0)
        total_tokens = usage_data.get("total_tokens", prompt_tokens + completion_tokens) # Calculate if not present

        return {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "estimated_cost": (total_tokens / 1000) * 0.02  # Example cost
        }

    def chat(self, messages: list, **kwargs) -> str:
        # ... (代码无变化, 但也要注意CoT处理 если这个口将来也用于希望无CoT的场景) ...
        try:
            model_to_use = kwargs.get('model', self.config["model"])
            temperature_to_use = kwargs.get('temperature', self.config["temperature"])
            # max_tokens_to_use = kwargs.get('max_tokens', self.config["max_token"]) # this wasn't used
            
            # Ensure messages is a list of dicts
            if not isinstance(messages, list) or not all(isinstance(m, dict) for m in messages):
                 logger.error(f"chat method received invalid messages format: {type(messages)}")
                 return "Error: Invalid message format for chat."

            logger.info(f"使用模型: {model_to_use} 发送通用聊天请求 (messages: {len(messages)})")

            response_obj = self.client.chat.completions.create(
                model=model_to_use,
                messages=messages,
                temperature=temperature_to_use,
                max_tokens=self.config["max_token"] # Using configured maxtoken
            )

            if not self._validate_response(response_obj.model_dump()):
                error_msg = f"错误的API响应结构 (chat method): {json.dumps(response_obj.model_dump(), default=str, ensure_ascii=False)}"
                logger.error(error_msg)
                return f"Error: {error_msg}"

            raw_content = response_obj.choices[0].message.content
            clean_content = self._sanitize_response(raw_content)
            # Potentially filter CoT here too if this 'chat' method could be used by tasks wanting raw output.
            # For now Matches get_response behavior by filtering.
            filtered_content = self._filter_thinking_content(clean_content)

            return filtered_content or ""

        except Exception as e:
            logger.error(f"Chat completion (通用) 失败: {str(e)}", exc_info=True)
            return f"Error: {str(e)}"


    def get_ollama_models(self) -> List[Dict]:
        # ... (代码无变化) ...
        try:
            # It's crucial base_url is valid if this is called even for non-Ollama setups
            # Using fixed localhost for safety if not configured clearly
            ollama_list_url = "http://localhost:11434/api/tags"
            if hasattr(self.client, 'base_url') and 'localhost:11434' in str(self.client.base_url):
                base_ollama_url = str(self.client.base_url).rstrip('/')
                if not base_ollama_url.startswith(('http://', 'https://')): # Ensure scheme presented
                      base_ollama_url = 'http://' + base_ollama_url
                ollama_list_url = f"{base_ollama_url}/api/tags"

            logger.debug(f"正在尝试从 {ollama_list_url} 获取Ollama模型列表")
            response = requests.get(ollama_list_url, timeout=5) # Added timeout
            response.raise_for_status() # Will raise HTTPError for bad responses (4xx or 5xx)
            
            models_data = response.json().get('models', [])
            if not isinstance(models_data, list):
                logger.error(f"Ollama返回的模型数据格式不正确: {type(models_data)}")
                return []
            
            formatted_models = []
            for model_info in models_data:
                if isinstance(model_info, dict) and 'name' in model_info:
                    formatted_models.append({
                        "id": model_info['name'],
                        "name": model_info['name'],
                        "status": "active",  # Placeholder
                        "type": "chat",     # Placeholder
                        "context_length": 16000  # Placeholder default
                    })
                else:
                     logger.warning(f"Ollama返回的单个模型信息格式不正确或缺少'name': {model_info}")
            return formatted_models
        except requests.exceptions.ConnectionError:
            logger.warning(f"连接Ollama主机 ({ollama_list_url}) 失败, Ollama模型列表将为空。")
            return []
        except requests.exceptions.Timeout:
            logger.warning(f"请求Ollama模型列表 ({ollama_list_url}) 超时, 模型列表将为空。")
            return []
        except Exception as e:
            logger.error(f"获取Ollama模型列表意外失败 ({ollama_list_url}): {str(e)}")
            return []


    def get_config(self) -> Dict:
        # ... (代码无变化) ...
        return self.config.copy()

