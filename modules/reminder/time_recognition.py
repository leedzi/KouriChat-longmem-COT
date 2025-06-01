"""
时间和搜索识别服务
负责识别消息中的时间信息、提醒意图和联网搜索需求
"""

import json
import os
import logging
from datetime import datetime
from typing import Optional, List, Tuple, Dict, Any

# 使用main日志器
logger = logging.getLogger('main')

class TimeRecognitionService:
    def __init__(self, llm_service):
        """
        初始化时间和搜索识别服务
        Args:
            llm_service: LLM服务实例，用于时间和搜索识别
        """
        self.llm_service = llm_service

        # 从文件读取提示词
        current_dir = os.path.dirname(os.path.abspath(__file__))
        root_dir = os.path.dirname(os.path.dirname(current_dir))

        # 读取时间识别提示词（仅时间）
        time_only_prompt_path = os.path.join(root_dir, "data", "base", "time_only.md")
        with open(time_only_prompt_path, "r", encoding="utf-8") as f:
            self.time_only_system_prompt = f.read().strip()

        # 读取时间和搜索识别提示词（时间+搜索）
        time_search_prompt_path = os.path.join(root_dir, "data", "base", "time_and_search.md")
        with open(time_search_prompt_path, "r", encoding="utf-8") as f:
            self.time_search_system_prompt = f.read().strip()

        # 兼容旧版提示词
        reminder_prompt_path = os.path.join(root_dir, "data", "base", "reminder.md")
        if os.path.exists(reminder_prompt_path):
            with open(reminder_prompt_path, "r", encoding="utf-8") as f:
                self.reminder_system_prompt = f.read().strip()
        else:
            self.reminder_system_prompt = self.time_only_system_prompt

    def recognize_time(self, message: str) -> Optional[List[Tuple[datetime, str]]]:
        """
        识别消息中的时间信息，支持多个提醒
        Args:
            message: 用户消息
        Returns:
            Optional[list]: [(目标时间, 提醒内容), ...] 或 None
        """
        current_time = datetime.now()
        user_prompt = f"""当前时间是：{current_time.strftime('%Y-%m-%d %H:%M:%S')}
请严格按照JSON格式分析这条消息中的提醒请求：{message}"""

        response = self.llm_service.get_response(
            message=user_prompt,
            system_prompt=self.reminder_system_prompt,
            user_id="time_recognition_system"
        )

        # 如果没有有效响应或明确不是时间相关
        if not response or response == "NOT_TIME_RELATED":
            return None

        # 提取和解析JSON
        try:
            # 清理响应
            response = ' '.join(response.split())
            start = response.find('{')
            end = response.rfind('}')

            # 检查是否找到了有效的JSON边界
            if start == -1 or end == -1 or start >= end:
                logger.debug(f"响应中未找到有效的JSON: {response[:100]}...")
                return None

            json_str = response[start:end + 1]

            # 解析JSON
            result = json.loads(json_str)

            # 提取提醒信息
            if "reminders" not in result or not isinstance(result["reminders"], list):
                return None

            reminders = []
            for reminder in result["reminders"]:
                if not isinstance(reminder, dict):
                    continue

                if "target_time" not in reminder or "reminder_content" not in reminder:
                    continue

                target_time = datetime.strptime(
                    reminder["target_time"],
                    "%Y-%m-%d %H:%M:%S"
                )
                reminders.append((target_time, reminder["reminder_content"]))

            return reminders if reminders else None

        except Exception as e:
            logger.error(f"处理时间识别响应失败: {str(e)}")
            logger.debug(f"错误的响应内容: {response[:200]}...")
            return None

    def recognize_time_and_search(self, message: str, user_name: str = None, avatar_name: str = None, network_search_enabled: bool = True) -> Dict[str, Any]:
        """
        同时识别消息中的时间信息和联网搜索需求
        Args:
            message: 用户消息
            user_name: 用户名称，用于替换提示词中的{user_name}
            avatar_name: 角色名称，用于替换提示词中的{avatar_name}
            network_search_enabled: 是否启用联网搜索
        Returns:
            Dict: {
                'reminders': [(目标时间, 提醒内容), ...] 或 [],
                'search_required': True/False,
                'search_query': 搜索查询内容 或 ''
            }
        """
        result = {
            'reminders': [],
            'search_required': False,
            'search_query': ''
        }

        # 根据是否启用联网搜索选择不同的提示词
        if network_search_enabled:
            # 使用时间和搜索提示词
            prompt_template = self.time_search_system_prompt
            # 替换变量
            if user_name:
                prompt_template = prompt_template.replace("{user_name}", user_name)
            else:
                prompt_template = prompt_template.replace("{user_name}", "用户")

            if avatar_name:
                prompt_template = prompt_template.replace("{avatar_name}", avatar_name)
            else:
                prompt_template = prompt_template.replace("{avatar_name}", "助手")
        else:
            # 使用仅时间提示词
            prompt_template = self.time_only_system_prompt

        current_time = datetime.now()
        user_prompt = f"""当前时间是：{current_time.strftime('%Y-%m-%d %H:%M:%S')}
请严格按照JSON格式分析这条消息中的提醒请求{' 和联网搜索需求' if network_search_enabled else ''}：{message}"""

        response = self.llm_service.get_response(
            message=user_prompt,
            system_prompt=prompt_template,
            user_id="time_search_recognition_system"
        )

        # 如果没有有效响应或明确不需要处理
        if not response or response == "NOT_REQUIRED":
            return result

        # 提取和解析JSON
        try:
            # 清理响应
            response = ' '.join(response.split())
            start = response.find('{')
            end = response.rfind('}')

            # 检查是否找到了有效的JSON边界
            if start == -1 or end == -1 or start >= end:
                logger.debug(f"响应中未找到有效的JSON: {response[:100]}...")
                return result

            json_str = response[start:end + 1]

            # 解析JSON
            parsed_result = json.loads(json_str)

            # 提取提醒信息
            if "reminders" in parsed_result and isinstance(parsed_result["reminders"], list):
                reminders = []
                for reminder in parsed_result["reminders"]:
                    if not isinstance(reminder, dict):
                        continue

                    if "target_time" not in reminder or "reminder_content" not in reminder:
                        continue

                    target_time = datetime.strptime(
                        reminder["target_time"],
                        "%Y-%m-%d %H:%M:%S"
                    )
                    reminders.append((target_time, reminder["reminder_content"]))

                result['reminders'] = reminders

            # 如果启用了联网搜索，则提取搜索需求
            if network_search_enabled:
                if "search_required" in parsed_result:
                    result['search_required'] = bool(parsed_result["search_required"])

                if "search_query" in parsed_result and parsed_result["search_query"]:
                    result['search_query'] = parsed_result["search_query"]

            return result

        except Exception as e:
            logger.error(f"处理时间和搜索识别响应失败: {str(e)}")
            logger.debug(f"错误的响应内容: {response[:200]}...")
            return result
