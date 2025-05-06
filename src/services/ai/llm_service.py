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
import json  # 新增导入
import time  # 新增导入
import pathlib
import requests
from typing import Dict, List, Optional, Tuple, Union
from openai import OpenAI
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
    retry_if_exception_type
)

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
        :param system_prompt: 系统级提示词
        """
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            default_headers={
                "Content-Type": "application/json",
                "User-Agent": "MyDreamBot/1.0"
            }
        )
        self.config = {
            "model": model,
            "max_token": max_token,
            "temperature": temperature,
            "max_groups": max_groups,
        }
        self.chat_contexts: Dict[str, List[Dict]] = {}

        # 安全字符白名单（可根据需要扩展）
        self.safe_pattern = re.compile(r'[\x00-\x1F\u202E\u200B]')

        # 如果是 Ollama，获取可用模型列表
        if 'localhost:11434' in base_url:
            self.available_models = self.get_ollama_models()
        else:
            self.available_models = []

    def _manage_context(self, user_id: str, message: str, role: str = "user"):
        """
        上下文管理器（支持动态记忆窗口）

        :param user_id: 用户唯一标识
        :param message: 消息内容
        :param role: 角色类型(user/assistant)
        """
        if user_id not in self.chat_contexts:
            self.chat_contexts[user_id] = []

        # 添加新消息
        self.chat_contexts[user_id].append({"role": role, "content": message})

        # 维护上下文窗口
        while len(self.chat_contexts[user_id]) > self.config["max_groups"] * 2:
            # 优先保留最近的对话组
            self.chat_contexts[user_id] = self.chat_contexts[user_id][-self.config["max_groups"]*2:]

    def _sanitize_response(self, raw_text: str) -> str:
        """
        响应安全处理器
        1. 移除控制字符
        2. 标准化换行符
        3. 防止字符串截断异常
        """
        try:
            cleaned = re.sub(self.safe_pattern, '', raw_text)
            return cleaned.replace('\r\n', '\n').replace('\r', '\n')
        except Exception as e:
            logger.error(f"Response sanitization failed: {str(e)}")
            return "响应处理异常，请重新尝试"

    def _validate_response(self, response: dict) -> bool:
    """
        API响应校验 (已宽松化，以兼容第三方API)
        主要放松token计数一致性检查和finish_reason检查。
        """
        logger.debug("API响应调试信息 (进行验证)：\n%s", json.dumps(response, indent=2, ensure_ascii=False))

        # —— 校验层级1：基础结构 (保持) ——
        required_root_keys = {"id", "object", "created", "model", "choices", "usage"}
        if missing := required_root_keys - response.keys():
            logger.error("[Validation] 根层级缺少必需字段：%s", missing)
            return False # 核心结构缺失，验证失败

        # —— 校验层级2：字段类型校验 (保持) ——
        type_checks = [
            ("id", str, "字段应为字符串"),
            ("object", str, "字段应为字符串"),
            ("created", int, "字段应为时间戳整数"),
            ("model", str, "字段应为模型名称字符串"),
            ("choices", list, "字段应为列表类型"),
            ("usage", dict, "字段应为使用量字典")
        ]
        for field, expected_type, error_msg in type_checks:
            if not isinstance(response.get(field), expected_type):
                logger.error("[Validation] 字段[%s]类型错误：%s", field, error_msg)
                return False # 基本类型错误，验证失败

        # —— 校验层级3：choices数组结构 (基本保持) ——
        if len(response.get("choices", [])) == 0:
            # 允许空 choices 数组吗？如果允许，这里应该改成 warning 并 return True
            logger.warning("[Validation] 响应 choices 数组为空")
            # return False # 如果空choices代表错误，则保留此行
            # 暂时假设空choices可能是合法的（比如被过滤），或内容在别处。如果确定空choices是错误，取消这行的注释

        for index, choice in enumerate(response.get("choices",[])): # 使用 .get 以防 choices 不存在
            if not isinstance(choice, dict):
                logger.error("[Validation] 第%d个choice类型错误", index)
                return False # Choice 结构错，验证失败

            if missing := {"message", "finish_reason"} - choice.keys(): # 简化检查，只看必备的
                logger.error("[Validation] choice[%d]缺少关键字段：%s", index, missing)
                return False # Choice 结构不完整，验证失败

            # 校验message结构
            message = choice.get("message", {}) # 使用 .get
            if not isinstance(message, dict):
                 logger.error("[Validation] 'message' 字段类型错误，应为字典: %s", type(message))
                 return False

            if missing := {"role", "content"} - message.keys():
                logger.error("[Validation] message结构异常：缺少%s", missing)
                # Content 可能为 null 或空，但 role 必须有
                if "role" not in message:
                   return False # Role 必须有！
                # Content 先不做严格检查

            if message.get("role") != "assistant": # 使用 .get
                logger.warning("[Validation] 非标准的角色类型：%s，但将继续处理...", message.get("role"))
                # 对于 role 不为 assistant 的情况，不再直接报错，而是发出警告

            # 检查 content 是否存在且是字符串 （放松对空内容的检查）
            if "content" in message and not (isinstance(message["content"], str) or message["content"] is None):
                 logger.error("[Validation] 'content' 字段类型错误，应为字符串或None: %s", type(message["content"]))
                 return False
            
            # ================ 放松 finish_reason 检查 ================
            allowed_finish_reasons = ("stop", "length", "content_filter", "tool_calls", None) # 加入 tool_calls
            finish_reason = choice.get("finish_reason") # 使用 .get
            if finish_reason not in allowed_finish_reasons:
                logger.warning("[Validation] 未知或非预期的对话终止原因：%s (将继续处理)", finish_reason)
                # 不再因为 finish_reason 不符合预设列表而返回 False

        # —— 校验层级4：使用量统计 (usage) ——
        usage = response.get("usage", {}) # 使用 .get
        if not isinstance(usage, dict):
             logger.error("[Validation] 'usage' 字段类型错误，应为字典: %s", type(usage))
             return False # usage 必须是字典

        usage_checks = [
            ("prompt_tokens", int),
            ("completion_tokens", int),
            ("total_tokens", int)
        ]
        # 只检查字段是否存在且类型是数字即可，不再检查非负
        for field, expected_type in usage_checks:
             if field not in usage or not isinstance(usage[field], expected_type):
                 logger.warning("[Validation] 使用量字段[%s]缺失或类型非整数。尝试继续...", field)
                 # 不再因此返回 False，最多给个警告

        # ================ 放松 Token 总数一致性检查 ================
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        total_tokens = usage.get("total_tokens", 0)
        
        # 确保提取的值是整数，如果不是或者不存在，则我们无法进行一致性检查
        if isinstance(prompt_tokens, int) and isinstance(completion_tokens, int) and isinstance(total_tokens, int):
            if total_tokens != (prompt_tokens + completion_tokens):
                logger.warning("[Validation] Token总数可能不一致(兼容模式): prompt(%d) + completion(%d) = %d ≠ total(%d)。这在某些API代理上可能发生，尝试继续...",
                             prompt_tokens, completion_tokens, prompt_tokens + completion_tokens, total_tokens)
                # 不再因为 token 计数不完全一致而返回 False

        # 如果所有关键检查都通过了
        logger.info("[Validation] 数据校验通过 (兼容模式)")
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
            # —— 阶段1：输入验证 ——
            if not message.strip():
                logger.warning("收到空消息请求")
                return "嗯...我好像收到了空白消息呢（歪头）"

            # —— 阶段2：上下文更新 ——
            # 只在程序刚启动时（上下文为空时）加载外部历史上下文
            if previous_context and user_id not in self.chat_contexts:
                logger.info(f"程序启动初始化：加载历史上下文，共 {len(previous_context)} 条消息")
                self.chat_contexts[user_id] = previous_context.copy()
            
            # 添加当前消息到上下文
            self._manage_context(user_id, message)

            # —— 阶段3：构建请求参数 ——
            # 读取基础Prompt
            try:
                # 从当前文件位置(llm_service.py)向上导航到项目根目录
                current_dir = os.path.dirname(os.path.abspath(__file__))  # src/services/ai
                project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))  # 项目根目录
                base_prompt_path = os.path.join(project_root, "data", "base", "base.md")
                
                with open(base_prompt_path, "r", encoding="utf-8") as f:
                    base_content = f.read()
            except Exception as e:
                logger.error(f"基础Prompt文件读取失败: {str(e)}")
                base_content = ""
            
            # 构建完整提示词: base + 核心记忆 + 人设
            if core_memory:
                final_prompt = f"{base_content}\n\n{core_memory}\n\n{system_prompt}"
                logger.debug("提示词顺序：base.md + 核心记忆 + 人设")
            else:
                final_prompt = f"{base_content}\n\n{system_prompt}"
                logger.debug("提示词顺序：base.md + 人设")
            
            # 构建消息列表
            messages = [
                {"role": "system", "content": final_prompt},
                *self.chat_contexts.get(user_id, [])[-self.config["max_groups"] * 2:]
            ]

            # 为 Ollama 构建消息内容
            chat_history = self.chat_contexts.get(user_id, [])[-self.config["max_groups"] * 2:]
            history_text = "\n".join([
                f"{msg['role']}: {msg['content']}" 
                for msg in chat_history
            ])
            ollama_message = {
                "role": "user",
                "content": f"{final_prompt}\n\n对话历史：\n{history_text}\n\n用户问题：{message}"
            }

            # 检查是否是 Ollama API
            is_ollama = 'localhost:11434' in str(self.client.base_url)

            if is_ollama:
                # Ollama API 格式
                request_config = {
                    "model": self.config["model"].split('/')[-1],  # 移除路径前缀
                    "messages": [ollama_message],  # 将消息包装在列表中
                    "stream": False,
                    "options": {
                        "temperature": self.config["temperature"],
                        "max_tokens": self.config["max_token"]
                    }
                }
                
                # 使用 requests 库向 Ollama API 发送 POST 请求
                try:
                    response = requests.post(
                        f"{str(self.client.base_url)}",
                        json=request_config,
                        headers={"Content-Type": "application/json"}
                    )
                    response.raise_for_status()
                    response_data = response.json()
                    
                    # 检查响应中是否包含 message 字段
                    if response_data and "message" in response_data:
                        raw_content = response_data["message"]["content"]
                        logger.debug("Ollama API响应内容: %s", raw_content)
                    else:
                        raise ValueError("错误的API响应结构")
                        
                    clean_content = self._sanitize_response(raw_content)
                    self._manage_context(user_id, clean_content, "assistant")
                    return clean_content
                    
                except Exception as e:
                    logger.error(f"Ollama API请求失败: {str(e)}")
                    raise

            else:
                # 主要 api 请求（重要）
                # 标准 OpenAI 格式
                request_config = {
                    "model": self.config["model"],  # 模型名称
                    "messages": messages,  # 消息列表
                    "temperature": self.config["temperature"],  # 温度参数
                    "max_tokens": self.config["max_token"],  # 最大 token 数
                    "top_p": 0.95,  # top_p 参数
                    "frequency_penalty": 0.2  # 频率惩罚参数
                }
                
                # 使用 OpenAI 客户端发送请求
                response = self.client.chat.completions.create(**request_config)
                # 验证 API 响应结构
                if not self._validate_response(response.model_dump()):
                    raise ValueError("错误的API响应结构")
                    
                # 获取原始内容
                raw_content = response.choices[0].message.content
                # 清理响应内容
                clean_content = self._sanitize_response(raw_content)
                # 管理上下文
                self._manage_context(user_id, clean_content, "assistant")
                # 返回清理后的内容
                return clean_content or ""

        except Exception as e:
            logger.error("大语言模型服务调用失败: %s", str(e), exc_info=True)
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
        return False

    def analyze_usage(self, response: dict) -> Dict:
        """
        用量分析工具
        """
        usage = response.get("usage", {})
        return {
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
            "total_tokens": usage.get("total_tokens", 0),
            "estimated_cost": (usage.get("total_tokens", 0) / 1000) * 0.02  # 示例计价
        }

    def chat(self, messages: list, **kwargs) -> str:
        """
        发送聊天请求并获取回复
        
        Args:
            messages: 消息列表，每个消息是包含 role 和 content 的字典
            **kwargs: 额外的参数配置
            
        Returns:
            str: AI的回复内容
        """
        try:
            response = self.client.chat.completions.create(
                model=self.config["model"],
                messages=messages,
                temperature=kwargs.get('temperature', self.config["temperature"]),
                max_tokens=self.config["max_token"]
            )
            
            if not self._validate_response(response.model_dump()):
                raise ValueError("Invalid API response structure")
                
            return response.choices[0].message.content or ""
            
        except Exception as e:
            logger.error(f"Chat completion failed: {str(e)}")
            return ""

    def get_ollama_models(self) -> List[Dict]:
        """获取本地 Ollama 可用的模型列表"""
        try:
            response = requests.get('http://localhost:11434/api/tags')
            if response.status_code == 200:
                models = response.json().get('models', [])
                return [
                    {
                        "id": model['name'],
                        "name": model['name'],
                        "status": "active",
                        "type": "chat",
                        "context_length": 16000  # 默认上下文长度
                    }
                    for model in models
                ]
            return []
        except Exception as e:
            logger.error(f"获取Ollama模型列表失败: {str(e)}")
            return []
