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
import pathlib # pathlib is imported but not used, consider removing if not needed elsewhere.
import requests
from typing import Dict, List, Optional, Tuple, Union # Tuple, Union not used in this version's public methods.
from openai import OpenAI
from openai import APIConnectionError, RateLimitError, APIStatusError # Import specific OpenAI errors

# tenacity is imported but not used in this core logic. If retry logic is needed, it should be applied.
# from tenacity import (
#     retry,
#     stop_after_attempt,
#     wait_random_exponential,
#     retry_if_exception_type
# )

logger = logging.getLogger('main')

class LLMService:
    def __init__(self, api_key: str, base_url: str, model: str,
                 max_token: int, temperature: float, max_groups: int):
        """
        强化版AI服务初始化
        """
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            default_headers={
                "Content-Type": "application/json",
                "User-Agent": "KouriChatClient/1.0" # Use a more specific user agent if desired
            },
            timeout=60.0, # Add a default timeout
            max_retries=2 # Add default max_retries
        )
        self.config = {
            "model": model,
            "max_token": max_token,
            "temperature": temperature,
            "max_groups": max_groups,
        }
        self.chat_contexts: Dict[str, List[Dict]] = {}
        self.safe_pattern = re.compile(r'[\x00-\x1F\u202E\u200B]')

        # Ollama model fetching initialization (assuming it's still relevant)
        if base_url and 'localhost:11434' in base_url:
            try:
                self.available_models = self.get_ollama_models()
            except Exception as e_ollama_init:
                logger.error(f"Failed to initialize Ollama models during LLMService setup: {e_ollama_init}")
                self.available_models = []
        else:
            self.available_models = []

    def _manage_context(self, user_id: str, message: str, role: str = "user"):
        """
        上下文管理器
        """
        if user_id not in self.chat_contexts:
            self.chat_contexts[user_id] = []
        self.chat_contexts[user_id].append({"role": role, "content": message})
        while len(self.chat_contexts[user_id]) > self.config["max_groups"] * 2:
            self.chat_contexts[user_id].pop(0) # Pop from beginning instead of re-slicing

    def _sanitize_response(self, raw_text: Optional[str]) -> str:
        """
        响应安全处理器
        """
        if raw_text is None: # Handle None input
            return ""
        if not isinstance(raw_text, str): # Ensure it's a string
            logger.warning(f"Sanitize_response received non-string input: {type(raw_text)}, converting to string.")
            raw_text = str(raw_text)

        try:
            cleaned = re.sub(self.safe_pattern, '', raw_text)
            return cleaned.replace('\r\n', '\n').replace('\r', '\n')
        except Exception as e:
            logger.error(f"Response sanitization failed: {str(e)}", exc_info=True)
            return "响应处理异常" # Keep it simple

    def _validate_response(self, response: dict) -> bool:
        """
        API响应校验 (已【极度】宽松化，主要目标是防止程序因非核心字段类型或缺失而崩溃)
        它只做最基本的检查，确保响应是一个字典，并且如果存在 'choices' 或 'candidates'，它们是列表。
        不强制要求所有 OpenAI 标准字段都存在或类型完全匹配。
        """
        logger.debug("API响应调试信息 (进行[极度宽松]验证)：\n%s", json.dumps(response, indent=2, ensure_ascii=False))

        if not isinstance(response, dict):
            logger.error("[宽松 Validation] 响应根本不是一个字典。类型: %s", type(response))
            return False

        has_choices = "choices" in response and isinstance(response.get("choices"), list)
        has_candidates = "candidates" in response and isinstance(response.get("candidates"), list)

        if not has_choices and not has_candidates:
            logger.warning("[宽松 Validation] 响应中既没有 'choices' 列表，也没有 'candidates' 列表。可能无法提取回复。")
            # Return True to allow content extraction logic to try its best.

        if has_choices and response.get("choices") is not None and not response["choices"]: # Check if it is not None before checking length
            logger.warning("[宽松 Validation] 'choices' 列表存在但为空。")

        if has_candidates and response.get("candidates") is not None and not response["candidates"]: # Check if it is not None
            logger.warning("[宽松 Validation] 'candidates' 列表存在但为空。")

        logger.info("[宽松 Validation] 基本结构检查通过。将尝试提取内容。")
        return True

    def get_response(self, message: str, user_id: str, system_prompt: str, previous_context: List[Dict] = None, core_memory: str = None) -> str:
        """
        完整请求处理流程
        Args:
            message: 用户消息
            user_id: 用户ID
            system_prompt: 系统提示词（人设）
            previous_context: 历史上下文（可选）
            core_memory: 核心记忆（可选）
        """
        try:
            if not message.strip():
                logger.warning("收到空消息请求")
                return "嗯...我好像收到了空白消息呢（歪头）"

            if previous_context and user_id not in self.chat_contexts:
                logger.info(f"程序启动初始化：加载历史上下文，共 {len(previous_context)} 条消息 for {user_id}")
                self.chat_contexts[user_id] = list(previous_context) # Use list() to ensure it's a mutable copy

            self._manage_context(user_id, message, "user") # Add current user message to context

            try:
                current_dir = os.path.dirname(os.path.abspath(__file__))
                project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
                base_prompt_path = os.path.join(project_root, "data", "base", "base.md")
                with open(base_prompt_path, "r", encoding="utf-8") as f:
                    base_content = f.read()
            except Exception as e_prompt:
                logger.error(f"基础Prompt文件读取失败: {str(e_prompt)}")
                base_content = ""

            final_system_prompt_content = f"{base_content}\n\n{system_prompt}"
            if core_memory and core_memory.strip():
                final_system_prompt_content = f"{base_content}\n\n核心记忆:\n{core_memory}\n\n当前人设:\n{system_prompt}"
                logger.debug("提示词顺序：base.md + 核心记忆 + 人设")
            else:
                logger.debug("提示词顺序：base.md + 人设 (无核心记忆)")


            # Construct messages for the API call
            # The context for user_id in self.chat_contexts already includes the latest user message
            api_messages = [
                {"role": "system", "content": final_system_prompt_content.strip()}
            ]
            # Add chat history, ensuring not to exceed max_groups
            # self.chat_contexts[user_id] at this point includes the current final_prompt as 'user' role
            # via self._manage_context before. The last one is the user, so we need context BEFORE that potentially.
            # The logic for self.chat_contexts has user, assistant, user, assistant...
            # So we take the last N pairs for context.
            # Correct approach: system prompt is separate, then the conversation history.
            api_messages.extend(self.chat_contexts.get(user_id, [])[-(self.config["max_groups"] * 2):])


            # Check if it is Ollama API (this logic was present earlier)
            is_ollama = self.client.base_url and 'localhost:11434' in str(self.client.base_url)

            if is_ollama:
                # Ollama specific logic, ensure it's up-to-date or tested separately
                # For simplicity, we'll assume Ollama is either not used or handled correctly elsewhere for now
                # This simplified version focuses on the main OpenAI-compatible path.
                logger.warning("Ollama specific path in get_response is simplified in this version. Full Ollama support may need review.")
                # Fall-through to OpenAI compatible path or implement specific Ollama logic here if crucial.
                # Assuming it might still use the OpenAI client object with a different base_url.
                request_config = {
                    "model": self.config["model"].split('/')[-1] if '/' in self.config.get("model","") else self.config.get("model",""),
                    "messages": api_messages, # Ollama usually prefers this format too
                    "temperature": self.config["temperature"],
                    "max_tokens": self.config["max_token"],
                    "stream": False, # Explicitly false for non-streaming
                }
                 # Using requests for Ollama as per original code
                try:
                    ollama_api_url = str(self.client.base_url).rstrip('/') + "/api/chat"  # Common Ollama chat endpoint
                    logger.info(f"Sending request to Ollama: {ollama_api_url} with model {request_config['model']}")
                    http_response = requests.post(
                        ollama_api_url,
                        json=request_config,
                        headers={"Content-Type": "application/json"},
                        timeout=60
                    )
                    http_response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
                    response_data_dict = http_response.json()
                except requests.exceptions.RequestException as e_ollama_req:
                    logger.error(f"Ollama API request failed (requests lib): {str(e_ollama_req)}", exc_info=True)
                    raise ValueError(f"Ollama API request error: {e_ollama_req}") from e_ollama_req


            else: # Standard OpenAI compatible API path
                request_config = {
                    "model": self.config["model"],
                    "messages": api_messages,
                    "temperature": self.config["temperature"],
                    "max_tokens": self.config["max_token"],
                    "top_p": self.config.get("top_p", 0.95), # Use .get for optional params
                    "frequency_penalty": self.config.get("frequency_penalty", 0.2)
                }
                logger.info(
                    f"向LLM发送请求: Model='{request_config['model']}', BaseURL='{self.client.base_url}', Msg_Count={len(api_messages)}"
                )
                if api_messages:
                    logger.debug(f"Last message sent: role='{api_messages[-1]['role']}', content='{str(api_messages[-1]['content'])[:100]}...'")


                try:
                    response_obj = self.client.chat.completions.create(**request_config)
                    response_data_dict = response_obj.model_dump(exclude_none=True) # Get dict, exclude_none can be helpful
                except (APIConnectionError, RateLimitError, APIStatusError) as e_openai_api:
                    logger.error(f"OpenAI Library API Error: {type(e_openai_api).__name__} - {str(e_openai_api)}", exc_info=True)
                    raise # Re-raise to be caught by the outer try-except
                except Exception as e_apicall:
                    logger.error(f"LLM API 调用 self.client.chat.completions.create 发生未知错误: {str(e_apicall)}", exc_info=True)
                    raise

            # ===============================================================
            # || START: 打印API响应 (无论Ollama还是OpenAI兼容)              ||
            # ===============================================================
            logger.info(">>>>>>>>>> RAW API RESPONSE START >>>>>>>>>>")
            logger.info(json.dumps(response_data_dict, indent=2, ensure_ascii=False)) # This will now print for Ollama too
            logger.info("<<<<<<<<<< RAW API RESPONSE END   <<<<<<<<<<")
            # ===============================================================

            if not self._validate_response(response_data_dict):
                logger.error(
                    f"宽松验证失败。原始响应导致宽松验证也失败了。Response: {json.dumps(response_data_dict, indent=2, ensure_ascii=False)}"
                )
                # Even if loose validation fails, content extraction will be attempted.
                # raise ValueError("错误的API响应结构 (即使是宽松验证也失败了)") # Commented out to allow content extraction to try.

            raw_content = None
            try:
                # Try OpenAI path first
                if "choices" in response_data_dict and \
                   isinstance(response_data_dict.get("choices"), list) and \
                   response_data_dict["choices"] and \
                   isinstance(response_data_dict["choices"][0], dict) and \
                   response_data_dict["choices"][0].get("message") and \
                   isinstance(response_data_dict["choices"][0]["message"], dict) and \
                   "content" in response_data_dict["choices"][0]["message"]:
                    raw_content = response_data_dict["choices"][0]["message"]["content"]
                    logger.info("成功从 OpenAI 路径 '.choices[0].message.content' 提取内容。")

                # Try possible Gemini path (if using a proxy that returns Gemini-like structure)
                elif "candidates" in response_data_dict and \
                     isinstance(response_data_dict.get("candidates"), list) and \
                     response_data_dict["candidates"] and \
                     isinstance(response_data_dict["candidates"][0], dict) and \
                     response_data_dict["candidates"][0].get("content") and \
                     isinstance(response_data_dict["candidates"][0]["content"], dict) and \
                     response_data_dict["candidates"][0]["content"].get("parts") and \
                     isinstance(response_data_dict["candidates"][0]["content"]["parts"], list) and \
                     response_data_dict["candidates"][0]["content"]["parts"] and \
                     isinstance(response_data_dict["candidates"][0]["content"]["parts"][0], dict) and \
                     "text" in response_data_dict["candidates"][0]["content"]["parts"][0]:
                    raw_content = response_data_dict["candidates"][0]["content"]["parts"][0]["text"]
                    logger.info("成功从 Gemini 路径 '.candidates[0].content.parts[0].text' 提取内容。")
                
                # Try Ollama direct 'message' path if others fail
                elif is_ollama and response_data_dict.get("message") and \
                    isinstance(response_data_dict["message"], dict) and \
                    "content" in response_data_dict["message"]:
                    raw_content = response_data_dict["message"]["content"]
                    logger.info("成功从 Ollama 直接 '.message.content' 路径提取内容。")                    
                
                else:
                    logger.warning("无法从已知的 OpenAI, Gemini, 或 Ollama(direct) 路径中提取聊天回复内容。请检查上面打印的原始API响应。")

            except Exception as e_extract:
                logger.error(f"提取内容时发生意外错误: {str(e_extract)}", exc_info=True)
                raw_content = None

            if raw_content is None:
                logger.error("最终未能提取到任何聊天回复内容 (raw_content is None)。将返回通用错误。")
                raise ValueError("未能从API响应中提取有效的聊天内容")

            clean_content = self._sanitize_response(raw_content) # sanitize_response now handles None
            self._manage_context(user_id, clean_content, "assistant")
            logger.info(f"AI回复 ({user_id}): {clean_content[:100]}...") # Log a snippet of the reply
            return clean_content

        except ValueError as ve: # Catch our specific ValueErrors for content/structure issues
            logger.error(f"值错误导致LLM服务失败: {str(ve)}", exc_info=False) # No need for full stack trace if it's our own ValueError
            return random.choice([
                "唔...好像API返回的数据有点奇怪，我没看懂呢。",
                "API响应的格式不太对哦，能帮我反馈一下吗？",
                "数据解析出错了，内容提取失败了~"
            ])
        except Exception as e: # Catch-all for other unexpected errors
            logger.error("大语言模型服务调用时发生未知类型错误: %s", str(e), exc_info=True)
            return random.choice([
                "好像有些小状况，请再试一次吧～",
                "信号好像不太稳定呢（皱眉）",
                "思考被打断了，请再说一次好吗？"
            ])

    def clear_history(self, user_id: str) -> bool:
        """
        清空指定用户的对话历史
        """
        if user_id in self.chat_contexts:
            del self.chat_contexts[user_id]
            logger.info("已清除用户 %s 的对话历史", user_id)
            return True
        logger.info("尝试清除用户 %s 的对话历史，但历史不存在。", user_id)
        return False

    def analyze_usage(self, response: dict) -> Dict: # This method isn't currently called by get_response
        """
        用量分析工具
        Note: This needs to be adapted if the 'usage' field structure changes or is absent.
        """
        usage_data = {}
        if response and isinstance(response, dict):
            # OpenAI style
            if "usage" in response and isinstance(response["usage"], dict):
                usage = response["usage"]
                usage_data = {
                    "prompt_tokens": usage.get("prompt_tokens", 0),
                    "completion_tokens": usage.get("completion_tokens", 0),
                    "total_tokens": usage.get("total_tokens", 0),
                }
            # Gemini style
            elif "usageMetadata" in response and isinstance(response["usageMetadata"], dict):
                usage_meta = response["usageMetadata"]
                usage_data = {
                    "prompt_tokens": usage_meta.get("promptTokenCount", 0),
                    "completion_tokens": usage_meta.get("candidatesTokenCount", 0), # Sum if multiple candidates?
                    "total_tokens": usage_meta.get("totalTokenCount", 0),
                }
            else:
                logger.warning("analyze_usage: No 'usage' or 'usageMetadata' found in response.")
        else:
            logger.warning("analyze_usage: Invalid response object passed.")

        # Example costing, adjust as needed
        usage_data["estimated_cost"] = (usage_data.get("total_tokens", 0) / 1000) * 0.002 # Generic cost
        return usage_data


    def chat(self, messages: list, **kwargs) -> str: # This seems like a more direct chat method
        """
        发送聊天请求并获取更直接的回复 (可能用于非上下文的单一轮次查询)
        """
        try:
            request_params = {
                "model": self.config["model"],
                "messages": messages,
                "temperature": kwargs.get('temperature', self.config["temperature"]),
                "max_tokens": kwargs.get('max_tokens', self.config["max_token"]) # Allow override
            }
            logger.info(f"Direct chat call with model: {request_params['model']}")

            response_obj = self.client.chat.completions.create(**request_params)
            response_data = response_obj.model_dump(exclude_none=True)

            # ===============================================================
            logger.info(">>>>>>>>>> RAW API RESPONSE START (chatmethod) >>>>>>>>>>")
            logger.info(json.dumps(response_data, indent=2, ensure_ascii=False))
            logger.info("<<<<<<<<<< RAW API RESPONSE END   (chatmethod) <<<<<<<<<<")
            # ===============================================================

            if not self._validate_response(response_data): # Use loose validation
                logger.error("Invalid API response structure in chat method (after loose validation).")
                 # Even if validation fails, try to extract
                # raise ValueError("Invalid API response structure in chat method")

            # Robust content extraction for chat method as well
            content = None
            try:
                if "choices" in response_data and response_data["choices"] and \
                   response_data["choices"][0].get("message") and \
                   "content" in response_data["choices"][0]["message"]:
                    content = response_data["choices"][0]["message"]["content"]
                elif "candidates" in response_data and response_data["candidates"] and \
                     response_data["candidates"][0].get("content") and \
                     response_data["candidates"][0]["content"].get("parts") and \
                     response_data["candidates"][0]["content"]["parts"] and \
                     "text" in response_data["candidates"][0]["content"]["parts"][0]:
                    content = response_data["candidates"][0]["content"]["parts"][0]["text"]
                # Ollama direct path could also be added here if chat() might use it
                else:
                    logger.warning("chat method: Could not extract content from known paths.")
            except Exception as e_chat_extract:
                 logger.error(f"chat method: Error extracting content: {e_chat_extract}")
                 content = None

            return self._sanitize_response(content) if content is not None else "" # Sanitize and return, or empty string

        except Exception as e:
            logger.error(f"Chat completion failed: {str(e)}", exc_info=True)
            return "" # Return empty string on failure for direct chat


    def get_ollama_models(self) -> List[Dict]: # Make sure this method is robust
        """获取本地 Ollama 可用的模型列表"""
        ollama_url = str(self.client.base_url).rstrip('/') # Assuming base_url is for ollama here
        if not 'localhost:11434' in ollama_url: # Double check if this is indeed for ollama
           ollama_url = 'http://localhost:11434' # Default if base_url not specific

        api_tags_url = f"{ollama_url}/api/tags"
        logger.info(f"Fetching Ollama models from: {api_tags_url}")
        try:
            response = requests.get(api_tags_url, timeout=10) # Add timeout
            response.raise_for_status() # Raise HTTPError for bad responses
            models_data = response.json().get('models', [])
            formatted_models = [
                { # This structure might be for a specific UI, adjust if needed
                    "id": model.get('name', 'unknown_model'),
                    "name": model.get('name', 'Unnamed Model'),
                    "status": "active", # Assume active
                    "type": "chat",    # Assume chat
                    "context_length": model.get('details',{}).get('parameter_size',0) * 1024 if model.get('details',{}).get('parameter_size') else 16000# Attempt better context length (example)
                }
                for model in models_data if model.get('name') # Ensure name exists
            ]
            logger.info(f"Found {len(formatted_models)} Ollama models.")
            return formatted_models
        except requests.exceptions.RequestException as e_ollama:
            logger.error(f"获取Ollama模型列表失败 (RequestException): {api_tags_url}, Error: {str(e_ollama)}")
            return []
        except json.JSONDecodeError as e_json:
            logger.error(f"获取Ollama模型列表失败 (JSONDecodeError): Body was '{response.text if 'response' in locals() else 'N/A'}', Error: {str(e_json)}")
            return []
        except Exception as e_generic:
            logger.error(f"获取Ollama模型列表中的未知错误: {str(e_generic)}", exc_info=True)
            return []

