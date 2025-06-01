"""
提醒服务
负责管理和执行提醒任务
"""

import logging
import time
import random
from datetime import datetime
from typing import Dict
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.date import DateTrigger
from src.utils.console import print_status

logger = logging.getLogger('main')

class ReminderService:
   
    def __init__(self, message_handler):
        # 添加调度器配置，容忍任务错过时间
        job_defaults = {
            'misfire_grace_time': 60,  # 允许任务最多延迟60秒执行
            'coalesce': True           # 合并错过的执行
        }
        
        # 使用线程池而不限制最大工作线程
        executors = {
            'default': {'type': 'threadpool'}  # 使用默认值
        }
        
        # 创建调度器时应用上述配置
        self.scheduler = BackgroundScheduler(
            job_defaults=job_defaults,
            executors=executors
        )
        
        # 禁用APScheduler的警告日志
        logging.getLogger('apscheduler').setLevel(logging.ERROR)
        
        self.message_handler = message_handler
        self.active_reminders: Dict[str, dict] = {}
        self.scheduler.start()
        logger.info("统一提醒服务已启动")

    def add_reminder(self, chat_id: str, target_time: datetime, 
                    content: str, sender_name: str, silent: bool = True) -> bool:
        """
        添加提醒任务
        Args:
            chat_id: 聊天ID
            target_time: 目标时间
            content: 提醒内容
            sender_name: 发送者名称
            silent: 是否静默添加
        Returns:
            bool: 是否添加成功
        """
        try:
            task_id = f"reminder_{chat_id}_{datetime.now().timestamp()}"
            
            job = self.scheduler.add_job(
                self.send_reminder,
                trigger=DateTrigger(run_date=target_time),
                args=[chat_id, content, sender_name],
                id=task_id
            )
            
            self.active_reminders[task_id] = {
                'chat_id': chat_id,
                'time': target_time,
                'content': content,
                'sender_name': sender_name
            }
            
            self._print_task_info(task_id, "新建", sender_name, target_time, content)
            logger.info(f"已添加提醒任务: {task_id}")
            return True
            
        except Exception as e:
            logger.error(f"添加提醒任务失败: {str(e)}")
            print_status(f"添加提醒任务失败: {str(e)}", "error", "CROSS")
            return False

    def send_reminder(self, chat_id: str, content: str, sender_name: str):
        """发送提醒消息"""
        try:
            # 生成提醒提示词
            prompt = self._get_reminder_prompt(content)
            logger.info(f"生成提醒消息 - 用户: {sender_name}, 提示词: {prompt}")
            
            # 直接使用提示词作为消息发送
            self.message_handler.handle_user_message(
                content=prompt,  # 直接使用提示词，而不是让AI再次生成回复
                chat_id=chat_id,
                sender_name="System",
                username="System",
                is_group=False
            )
            
            # 记录提醒已发送
            print_status(
                f"提醒已发送 - 接收者: {sender_name}, 内容: {content}",
                "success",
                "BELL"
            )
            logger.info(f"已发送提醒消息给 {sender_name}")
            
            # 移除已完成的提醒
            self._remove_reminder(chat_id, content)
            
        except Exception as e:
            logger.error(f"发送提醒消息失败: {str(e)}")

    def _get_reminder_prompt(self, content: str) -> str:
        """
        生成提醒提示词
        Args:
            content: 提醒内容
        Returns:
            str: 提示词
        """
        return f"""现在时间到了，用户之前让你提醒他{content}。请以你的人设中的身份主动找用户聊天。保持角色设定的一致性和上下文的连贯性"""

    def _remove_reminder(self, chat_id: str, content: str):
        """
        移除已完成的提醒
        Args:
            chat_id: 聊天ID
            content: 提醒内容
        """
        task_id = next(
            (tid for tid, task in self.active_reminders.items() 
             if task['chat_id'] == chat_id and task['content'] == content),
            None
        )
        if task_id:
            del self.active_reminders[task_id]

    def _print_task_info(self, task_id: str, action: str, 
                        sender_name: str, target_time: datetime, content: str):
        """
        打印任务信息
        Args:
            task_id: 任务ID
            action: 动作类型
            sender_name: 发送者名称
            target_time: 目标时间
            content: 提醒内容
        """
        time_str = target_time.strftime("%Y-%m-%d %H:%M:%S")
        time_diff = target_time - datetime.now()
        minutes = int(time_diff.total_seconds() / 60)
        
        
        logger.info(f"提醒任务[{action}]: ID={task_id[:8]}..., 接收者={sender_name}, 时间={time_str}({minutes}分钟后), 内容={content[:30]}{'...' if len(content) > 30 else ''}")