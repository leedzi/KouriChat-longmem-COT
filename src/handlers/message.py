"""
消息处理模块
负责处理聊天消息，包括:
- 消息队列管理
- 消息分发处理
- API响应处理
- 多媒体消息处理
"""
import logging
import threading
import time
import re  # <--- 确保导入 re
import tempfile
from datetime import datetime
from typing import Dict, List, Optional
from openai import OpenAI
from wxauto import WeChat
from src.services.database import Session, ChatMessage
import random
import os
# 确保路径正确，根据你的实际项目结构调整
from src.services.ai.llm_service import LLMService
from src.services.ai.network_search_service import NetworkSearchService
from src.config import config, WEBLENS_ENABLED, NETWORK_SEARCH_ENABLED
# 确保路径正确
from modules.memory.memory_service import MemoryService
from modules.memory.content_generator import ContentGenerator
from modules.reminder.time_recognition import TimeRecognitionService
from modules.reminder import ReminderService
from .debug import DebugCommandHandler

# 导入emoji库用于处理表情符号
import emoji

# 修改logger获取方式，确保与main模块一致
logger = logging.getLogger('main')

class MessageHandler:
    def __init__(self, root_dir, api_key, base_url, model, max_token, temperature,
                 max_groups, robot_name, prompt_content, image_handler, emoji_handler, voice_handler, memory_service, content_generator=None):
        self.root_dir = root_dir
        self.api_key = api_key
        self.model = model
        self.max_token = max_token
        self.temperature = temperature
        self.max_groups = max_groups
        self.robot_name = robot_name
        self.prompt_content = prompt_content

        # 使用 DeepSeekAI 替换直接的 OpenAI 客户端
        self.deepseek = LLMService(
            api_key=api_key,
            base_url=base_url,
            model=model,
            max_token=max_token,
            temperature=temperature,
            max_groups=max_groups
        )

        # 消息队列相关
        self.message_queues = {}
        self.queue_timers = {}
        from src.config import config
        self.QUEUE_TIMEOUT = config.behavior.message_queue.timeout
        self.queue_lock = threading.Lock()
        self.chat_contexts = {} # 这个可能与LLMService内部的重复，注意使用

        # 微信实例
        self.wx = WeChat()

        # 添加 handlers
        self.image_handler = image_handler
        self.emoji_handler = emoji_handler
        self.voice_handler = voice_handler
        self.memory_service = memory_service

        # 保存当前角色名
        # 确保 config.behavior.context.avatar_dir 是相对路径或绝对路径
        try:
            avatar_full_path = os.path.join(self.root_dir, config.behavior.context.avatar_dir)
            self.current_avatar = os.path.basename(avatar_full_path)
            logger.info(f"当前使用角色: {self.current_avatar}")
        except AttributeError:
             logger.error("config.behavior.context.avatar_dir 未找到或配置错误！")
             self.current_avatar = "default_avatar" # 或者抛出异常
             logger.warning(f"将使用默认角色名: {self.current_avatar}")


        self.content_generator = content_generator
        if self.content_generator is None:
            try:
                # from modules.memory.content_generator import ContentGenerator # 已在顶部导入
                # from src.config import config # 已在顶部导入
                if hasattr(config, 'llm'): # 先检查config里是否有llm配置
                    self.content_generator = ContentGenerator(
                        root_dir=root_dir,
                        api_key=config.llm.api_key,
                        base_url=config.llm.base_url,
                        model=config.llm.model,
                        max_token=config.llm.max_tokens,
                        temperature=config.llm.temperature
                    )
                    logger.info("已创建内容生成器实例（从config）")
                else:
                     logger.warning("Config中未找到LLM详细配置，无法自动创建ContentGenerator。")
                     self.content_generator = None
            except Exception as e:
                logger.error(f"创建内容生成器实例失败: {str(e)}")
                self.content_generator = None

        # 初始化调试命令处理器
        self.debug_handler = DebugCommandHandler(
            root_dir=root_dir,
            memory_service=memory_service,
            llm_service=self.deepseek,
            content_generator=self.content_generator
        )

        self.preserve_format_commands = [None, '/diary', '/state', '/letter', '/list', '/pyq', '/gift', '/shopping']
        logger.info("调试命令处理器已初始化")

        self.time_recognition = TimeRecognitionService(self.deepseek)
        logger.info("时间识别服务已初始化")
        self.reminder_service = ReminderService(self)
        logger.info("提醒服务已初始化")
        self.network_search_service = NetworkSearchService(self.deepseek)
        logger.info("网络搜索服务已初始化")


    # +++ 新增方法：处理CoT回复 +++
    def _process_cot_reply(self, raw_reply: str) -> tuple[str, Optional[str]]:
        """
        处理包含CoT思考链 ([think]...[/think]) 的原始LLM回复。
        提取思考内容，记录日志(INFO级别)，并返回清理后的回复。

        Args:
            raw_reply: 从LLM获取的原始回复字符串。

        Returns:
            tuple: (cleaned_reply, think_content)
                   cleaned_reply: 移除思考块后的干净回复字符串。
                   think_content: 提取出的思考内容字符串，如果没有则为None。
        """
        think_content = None
        cleaned_reply = raw_reply
        think_pattern = re.compile(r'\[think\](.*?)\[/think\]\s*', re.DOTALL | re.IGNORECASE)

        match = think_pattern.search(raw_reply)
        if match:
            think_content = match.group(1).strip()
            cleaned_reply = think_pattern.sub('', raw_reply, count=1).strip()
            # --- 使用 INFO 级别记录思考内容 ---
            logger.info(f"--- AI Thought Process ---")
            logger.info(f"{think_content}")
            # -------------------------------
        else:
            # 如果没有找到[think]...[/think]，再尝试XML风格的 <thinking>...</thinking>
            # (之前这里的正则有点问题，修正一下)
            xml_think_pattern = re.compile(r'\s*(.*?)\s*\</thinking\>', re.DOTALL | re.IGNORECASE) # 修正正则
            xml_match = xml_think_pattern.search(raw_reply)
            if xml_match:
                think_content = xml_match.group(1).strip()
                cleaned_reply = xml_think_pattern.sub('', raw_reply, count=1).strip()
                # --- 使用 INFO 级别记录思考内容 ---
                logger.info(f"--- AI Thought Process (XML format) ---")
                logger.info(f"{think_content}")
                # ---------------------------------
            # else: # 这个 debug log 通常不需要，保持注释
            #     logger.debug("No [think]...[/think] or <thinking>...</thinking> block found in reply.")

        return cleaned_reply, think_content
    # +++ 结束新增方法 +++


    def _get_queue_key(self, chat_id: str, sender_name: str, is_group: bool) -> str:
        """生成队列键值...""" # (代码不变)
        return f"{chat_id}_{sender_name}" if is_group else chat_id

    # ------ VVVVVV 修改 save_message 定义和内部逻辑 VVVVVV ------
    def save_message(self, sender_id: str, sender_name: str, message: str, reply: str, is_group: bool, is_system_message: bool = False):
        """
        保存聊天记录到数据库和短期记忆。(已接收 is_group 参数)
        Args:
            sender_id (str): 发送者ID (通常是chat_id)
            sender_name (str): 发送者昵称
            message (str): 用户发送的消息 (可能是格式化后的)
            reply (str): AI生成的干净回复 (不含思考块，不含 @ 前缀)
            is_group (bool): 是否为群聊消息
            is_system_message (bool): 是否为系统内部消息
        """
        try:
             # Reply传入时应已是 cleaned_reply (不含思考块)
             # 这里 clean_reply 的目标是统一格式，确保不含可能误加的群聊@前缀
            clean_reply = reply
            if clean_reply and reply.startswith(f"@{sender_name} ") and is_group: # 确保群聊时的 @ 前缀不会意外存入
                # This case theoretically shouldn't happen often if we pass cleaned_reply correctly,
                # but acts as a safeguard.
                logger.warning(f"save_message received reply for group starting with @{sender_name}, stripping.")
                clean_reply = reply[len(f"@{sender_name} "):]

            session = Session()
            chat_message = ChatMessage(
                sender_id=sender_id,
                sender_name=sender_name,
                message=message, # 保存用户发送的消息
                # --- 统一保存清理后的回复到数据库 ---
                reply=clean_reply
            )
            session.add(chat_message)
            session.commit()
            session.close()
            logger.debug(f"消息已保存至数据库 (User: {sender_name}, Reply snippet: {clean_reply[:30]}...)")

            avatar_name = self.current_avatar
            # --- 传递清理后的回复给 MemoryService ---
            self.memory_service.add_conversation(avatar_name, message, clean_reply, sender_id, is_system_message) # 使用 clean_reply
            logger.debug(f"消息已添加至短期记忆服务 (User: {sender_name})")

        except Exception as e:
             # 记录更详细的错误日志
            logger.error(f"保存消息失败 (User: {sender_name}): {str(e)}", exc_info=True)
    # ------ ^^^^^^ 修改 save_message 定义和内部逻辑完成 ^^^^^^ ------


    def get_api_response(self, message: str, user_id: str, is_group: bool = False) -> str:
        """获取 API 回复 (返回的是包含或不包含 [think] 块的原始回复)"""
        # --- 这部分代码与你之前能工作的版本一致 ---
        avatar_dir = os.path.join(self.root_dir, config.behavior.context.avatar_dir)
        prompt_path = os.path.join(avatar_dir, "avatar.md")
        avatar_name = self.current_avatar

        try:
            with open(prompt_path, "r", encoding="utf-8") as f:
                avatar_content = f.read()
                logger.debug(f"角色提示文件大小: {len(avatar_content)} bytes")

            core_memory = self.memory_service.get_core_memory(avatar_name, user_id=user_id)
            # 如果你的日志级别现在可能是INFO，这条核心记忆的日志还是建议用 DEBUG
            logger.debug(f"核心记忆长度: {len(core_memory)}") # Keep DEBUG level for this potentially long output
            core_memory_prompt = f"# 核心记忆\n{core_memory}" if core_memory else ""

            recent_context = None
            # 检查 LLMService 实例内部是否有 chat_contexts (更健壮的方式)
            if hasattr(self.deepseek, 'chat_contexts') and user_id not in self.deepseek.chat_contexts:
                recent_context = self.memory_service.get_recent_context(avatar_name, user_id)
                if recent_context:
                    logger.info(f"加载历史上下文 (用户: {user_id}, 轮数: {len(recent_context)//2})") # 简化 INFO 日志
                    logger.debug(f"用户 {user_id} 的历史上下文 (片段): {str(recent_context)[:200]}...") # DEBUG 输出片段
            elif not hasattr(self.deepseek, 'chat_contexts'):
                logger.warning("LLMService(self.deepseek) 没有 'chat_contexts' 属性，无法检查是否已加载，尝试加载。")
                recent_context = self.memory_service.get_recent_context(avatar_name, user_id)
                if recent_context:
                    logger.info(f"加载历史上下文 (用户: {user_id}, 轮数: {len(recent_context)//2}) (无法确认是否重复加载)") # 简化 INFO 日志

            if is_group:
                group_prompt_path = os.path.join(self.root_dir, "data", "base", "group.md")
                try:
                    with open(group_prompt_path, "r", encoding="utf-8") as f:
                        group_chat_prompt = f.read().strip()
                    combined_system_prompt = f"{group_chat_prompt}\n\n{avatar_content}"
                except FileNotFoundError:
                    logger.warning(f"群聊提示模板未找到: {group_prompt_path}")
                    combined_system_prompt = avatar_content # Fallback
                except Exception as e_read_group:
                     logger.error(f"读取群聊模板失败: {e_read_group}")
                     combined_system_prompt = avatar_content # Fallback safer
            else:
                combined_system_prompt = avatar_content

            if hasattr(self, 'system_prompts') and user_id in self.system_prompts and self.system_prompts[user_id]:
                additional_prompt = "\n\n".join(self.system_prompts[user_id])
                # INFO 级别只打片段
                logger.info(f"使用附加系统提示 (片段): {additional_prompt[:100]}...")
                combined_system_prompt = f"{combined_system_prompt}\n\n参考信息:\n{additional_prompt}"
                self.system_prompts[user_id] = []

            # 调用 LLMService, 获取可能包含 [think] 块的原始回复
            response = self.deepseek.get_response(
                message=message,
                user_id=user_id, # 此 user_id 重要
                system_prompt=combined_system_prompt,
                previous_context=recent_context,
                core_memory=core_memory_prompt
            )
            logger.debug(f"LLM原始响应 (片段): {response[:200]}...") # DEBUG 看原始返回
            return response

        except FileNotFoundError as e_fnf:
             logger.error(f"提示词文件未找到: {e_fnf}", exc_info=True)
             return "糟糕，好像用于思考的笔记找不到了……你刚刚说什么？"
        except Exception as e:
            logger.error(f"获取API响应失败 ({user_id}): {str(e)}", exc_info=True)
            # 使用更安全的基础回复
            return "嗯……抱歉，我好像有点走神了，你能再说一下吗？"


    def handle_user_message(self, content: str, chat_id: str, sender_name: str,
                     username: str, is_group: bool = False, is_image_recognition: bool = False):
        """统一的消息处理入口"""
        try:
            # INFO 更简洁，详细内容用 DEBUG
            log_prefix = f"[消息处理 in] {sender_name}" + (" (群聊）" if is_group else f" (私聊 {username})")
            detailed_content = f" 内容: {content}"
            logger.info(log_prefix)
            logger.debug(log_prefix + detailed_content) # 完整内容放 DEBUG

            # --- 处理调试命令 (逻辑不变) ---
            if self.debug_handler.is_debug_command(content):
                logger.info(f"{log_prefix}: 检测到调试命令 '{content}'")
                def command_callback(command, reply, chat_id):
                    try:
                        # cleaned_reply, _ = self._process_cot_reply(reply) # 也清理一下命令的回复？取决于命令逻辑
                        # 暂时不清理，按原样发送
                        if is_group:
                            reply = f"@{sender_name} {reply}"
                        self._send_command_response(command, reply, chat_id)
                        logger.info(f"异步处理调试命令完成: {command}")
                    except Exception as e:
                         logger.error(f"异步命令回调失败: {str(e)}")

                intercept, response = self.debug_handler.process_command(
                    command=content, current_avatar=self.current_avatar, user_id=chat_id, chat_id=chat_id, callback=command_callback
                )
                if intercept:
                    if response:
                        # cleaned_response, _ = self._process_cot_reply(response) # 清理同步命令回复？
                        # 按原样送出，如果debug指令输出[think]也方便看
                        if is_group:
                            response = f"@{sender_name} {response}"
                        self._send_raw_message(response, chat_id) # 或_send_command_response? 看格式需求
                    logger.info(f"{log_prefix}: 调试命令 '{content}' 已处理并拦截后续流程。")
                    return None # 直接返回

            # --- 添加到消息队列 (逻辑不变) ---
            self._add_to_message_queue(content, chat_id, sender_name, username, is_group, is_image_recognition)

        except Exception as e:
            logger.error(f"处理用户消息入口失败: {str(e)}", exc_info=True)
            return None


    def _add_to_message_queue(self, content: str, chat_id: str, sender_name: str,
                            username: str, is_group: bool, is_image_recognition: bool):
        """添加消息到队列并设置定时器 (逻辑不变)"""
        # ... 逻辑不变 ...
        has_link = False
        if WEBLENS_ENABLED: # 假设这是个全局开关或配置项
            try:
                urls = self.network_search_service.detect_urls(content)
                if urls:
                    has_link = True
                    # Log only the first detected URL for brevity in info
                    logger.info(f"[消息队列] 检测到链接: {urls[0]}, 将在队列处理时提取内容")
            except AttributeError:
                 logger.error("network_search_service 或其 detect_urls 方法不可用。")
            except Exception as e_detect:
                 logger.error(f"检测URL时出错: {e_detect}")


        with self.queue_lock:
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            queue_key = self._get_queue_key(chat_id, sender_name, is_group)

            if queue_key not in self.message_queues:
                logger.info(f"[消息队列] 创建新队列 for key: {queue_key}")
                self.message_queues[queue_key] = {
                    'messages': [f"[{current_time}]\n{content}"],
                    'chat_id': chat_id,
                    'sender_name': sender_name,
                    'username': username,
                    'is_group': is_group,
                    'is_image_recognition': is_image_recognition,
                    'last_update': time.time(),
                    'has_link': has_link,
                    'urls': urls if has_link else []
                }
                logger.debug(f"[消息队列] 首条消息 for {queue_key}: {content[:50]}...")
            else:
                # Append message, update timestamp, merge link status/URLs
                self.message_queues[queue_key]['messages'].append(content)
                self.message_queues[queue_key]['last_update'] = time.time()
                self.message_queues[queue_key]['has_link'] |= has_link # Use |= for bool OR
                if has_link:
                    # Ensure 'urls' exists and is a list before extending
                    if 'urls' not in self.message_queues[queue_key] or not isinstance(self.message_queues[queue_key]['urls'], list):
                        self.message_queues[queue_key]['urls'] = []
                    self.message_queues[queue_key]['urls'].extend(urls) # Append all newly detected URLs
                msg_count = len(self.message_queues[queue_key]['messages'])
                logger.info(f"[消息队列] 追加消息 to {queue_key}, 当前队列数: {msg_count}")
                logger.debug(f"[消息队列] 新增消息: {content[:50]}...")

            # Reset timer
            if queue_key in self.queue_timers and self.queue_timers[queue_key]:
                try:
                    self.queue_timers[queue_key].cancel()
                    # logger.debug(f"[消息队列] 重置定时器 for key: {queue_key}") # Too verbose maybe
                except Exception as e_cancel: logger.error(f"[消息队列] 取消旧定时器失败 for {queue_key}: {str(e_cancel)}")
                self.queue_timers[queue_key] = None # Clear ref

            # Start new timer
            timer = threading.Timer(self.QUEUE_TIMEOUT, self._process_message_queue, args=[queue_key])
            timer.daemon = True # Ensure threads don't block exit
            timer.start()
            self.queue_timers[queue_key] = timer
            logger.info(f"[消息队列] 设置/更新 {self.QUEUE_TIMEOUT}s 定时器 for key: {queue_key}")


    def _process_message_queue(self, queue_key: str):
        """处理消息队列"""
        # Initialize variables outside lock where possible
        queue_data = None
        log_prefix = f"[消息队列处理 for {queue_key}]" # Log using key initially

        try:
            with self.queue_lock:
                if queue_key not in self.message_queues:
                    logger.debug(f"{log_prefix} 队列不再存在，跳过。")
                    return

                current_time = time.time()
                queue_data = self.message_queues[queue_key] # Get data ref
                last_update = queue_data['last_update']

                if current_time - last_update < self.QUEUE_TIMEOUT - 0.1: # Check threshold
                    logger.debug(f"{log_prefix} 定时器触发但未到超时，重新等待。")
                    # No need to reschedule usually, Timer thread handles this
                    return

                # Timeout reached, pop the data to process
                logger.info(f"{log_prefix} 队列超时，开始处理...")
                # Now pop the actual data from the dict
                queue_data = self.message_queues.pop(queue_key)
                if queue_key in self.queue_timers:
                    # Clean up timer reference
                    self.queue_timers.pop(queue_key, None)
                else:
                     logger.warning(f"{log_prefix} 处理时未找到定时器记录。")


            # ---- Processing starts outside the lock ----
            # Extracted queue data vars
            messages = queue_data['messages']
            chat_id = queue_data['chat_id']
            username = queue_data['username']
            sender_name = queue_data['sender_name']
            is_group = queue_data['is_group']
            is_image_recognition = queue_data['is_image_recognition']
            has_link = queue_data.get('has_link', False)
            urls = queue_data.get('urls', [])

            # Rebuild log_prefix with sender_name for clarity
            log_prefix = f"[消息队列处理 for {sender_name}]"

            combined_message = "\n".join(messages)

            logger.info(f"{log_prefix} 消息数: {len(messages)}")
            logger.debug(f"{log_prefix} 合并内容 (片段): {combined_message[:100]}...")

            # --- 处理 WebLens 链接 ---
            processed_message = combined_message
            web_content_added = False
            if has_link and WEBLENS_ENABLED:
                unique_urls = list(set(urls)) # Process unique URLs
                if unique_urls:
                    logger.info(f"{log_prefix} 处理 {len(unique_urls)} 个URL (WebLens)...")
                    all_web_results = []
                    # --- Potential IO, should be outside lock ---
                    for url in unique_urls:
                         logger.info(f"{log_prefix} extracting from URL: {url}")
                         try:
                             web_results = self.network_search_service.extract_web_content(url) # Network IO
                             if web_results and web_results.get('original'):
                                 all_web_results.append(web_results['original'])
                             else:
                                 logger.warning(f"{log_prefix} 无法从 {url} 提取内容 (WebLens)")
                         except Exception as e_weblens:
                              logger.error(f"{log_prefix} 提取URL内容失败 ({url}): {e_weblens}", exc_info=True)

                    if all_web_results:
                        formatted_web_content = "\n\n".join(all_web_results)
                        processed_message = f"{combined_message}\n\n# 参考网页内容\n{formatted_web_content}"
                        web_content_added = True
                        logger.info(f"{log_prefix} 已添加网页内容到后续prompt (WebLens)")
                        logger.debug(f"{log_prefix} 带网页内容的消息 (片段): {processed_message[:200]}...")

             # --- 处理网络搜索 ---
            network_search_results_added = False # RENAME var if search results added to distinguish
            if NETWORK_SEARCH_ENABLED and not web_content_added: # Only search if WebLens didn't run/succeed
                logger.debug(f"{log_prefix} 检查网络搜索需求...")
                try:
                    search_intent = self.time_recognition.recognize_time_and_search(
                        message=combined_message, # Use original msg for intent recognition
                        user_name=sender_name,
                        avatar_name=self.current_avatar,
                        network_search_enabled=NETWORK_SEARCH_ENABLED
                    )

                    if search_intent.get('search_required') and search_intent.get('search_query'):
                        query = search_intent['search_query']
                        logger.info(f"{log_prefix} 检测到网络搜索需求, 查询: '{query}'")
                        # --- Sending notification needs IO, outside lock ---
                        searching_msg = f"[{self.robot_name}] 正在为你搜索相关信息：{query}"
                        try: self.wx.SendMsg(msg=searching_msg, who=chat_id)
                        except Exception as send_err: logger.error(f"发送'正在搜索'提示失败: {send_err}")

                        # User ID context is important for good search results
                        user_id = chat_id # !!! Assuming user_id is chat_id for search context !!!
                                          # Might need fixing for groups if you need group-member specific context
                        conv_context = self.memory_service.get_recent_context(self.current_avatar, user_id)

                        # --- Network IO for search ---
                        search_results = self.network_search_service.search_internet(
                            query=query,
                            conversation_context=conv_context # Pass context for potentially better search
                        )

                        if search_results and search_results.get('original'):
                             search_result_text = search_results['original']
                             # Prepend header to results for clarity in prompt
                             processed_message += f"\n\n# 网络搜索结果 (查询: {query})\n{search_result_text}"
                             network_search_results_added = True # Mark search result addedness
                             logger.info(f"{log_prefix} 已添加网络搜索结果到后续prompt")
                             logger.debug(f"{log_prefix} 带搜索结果的消息 (片段): {processed_message[:200]}...")
                        else:
                             logger.warning(f"{log_prefix} 网络搜索 '{query}' 失败或结果为空。")
                             # --- Send failure message ---
                             failed_msg = f"[{self.robot_name}] 抱歉，关于“{query[:20]}...”的信息未能找到。"
                             try: self.wx.SendMsg(msg=failed_msg, who=chat_id)
                             except Exception as send_err: logger.error(f"发送搜索失败提示失败: {send_err}")
                    else:
                        logger.debug(f"{log_prefix} 未检测到明确的网络搜索需求。")
                except AttributeError:
                     logger.error("time_recognition 或 network_search_service 不可用，无法执行搜索检查。")
                except Exception as e_search_check:
                     logger.error(f"{log_prefix} 检查或执行网络搜索时出错: {e_search_check}", exc_info=True)


            # --- 最终分发 ---
            # 使用 processed_message (可能包含WebLens或搜索结果)
            logger.info(f"{log_prefix} 开始路由处理...")
            # --- 分发时确定要传递给具体 handler 的 user_id ---
            # For simplicity, using chat_id as general user_id, needs careful review for complex group logic
            user_id_for_handler = chat_id

            if self.voice_handler.is_voice_request(processed_message): # Check original for voice keyword
                logger.info(f"{log_prefix} 路由到语音处理...")
                self._handle_voice_request(processed_message, chat_id, sender_name, username, is_group)
            elif self.image_handler.is_random_image_request(combined_message): # Check original for image keyword
                logger.info(f"{log_prefix} 路由到随机图片处理...")
                self._handle_random_image_request(combined_message, chat_id, sender_name, username, is_group) # Pass original
            # elif not is_image_recognition and self.image_handler.is_image_generation_request(combined_message): # Generation based on original
                 # logger.info(f"{log_prefix} Routing to image generation...")
                 # self._handle_image_generation_request(combined_message, chat_id, sender_name, username, is_group)
            else:
                # Default to text handler using the potentially augmented message
                logger.info(f"{log_prefix} 路由到文本处理...")
                self._handle_text_message(processed_message, chat_id, sender_name, username, is_group) # Pass final processed


        except Exception as e:
            logger.error(f"处理消息队列失败 KEY({queue_key}): {str(e)}", exc_info=True)
            # Attempt cleanup, maybe try to re-queue or notify admin/user?
            with self.queue_lock: # Acquire lock again for cleanup
                 # Remove possibly leftover data just in case error was after lock release but before finish
                 self.message_queues.pop(queue_key, None)
                 self.queue_timers.pop(queue_key, None)


    def _process_text_for_display(self, text: str) -> str:
        """处理文本以确保表情符号正确显示 -- Basic implementation"""
        # (代码不变)
        try:
            # Normalize emoji representation
            return emoji.emojize(emoji.demojize(text))
        except Exception as e_emoji: # Be specific about expected errors if possible
            logger.error(f"Emoji processing failed: {e_emoji}")
            return text # Return original on error

    def _send_message_with_dollar(self, reply, chat_id):
        """以$为分隔符分批发送回复 (逻辑不变)"""
        # ... 逻辑不变 ...
        send_prefix = f"[消息发送 to {chat_id}]"
        try:
            display_reply = self._process_text_for_display(reply)
        except Exception as e_disp:
             logger.error(f"{send_prefix} Error processing text for display: {e_disp}. Sending raw reply.")
             display_reply = reply # Fallback to raw reply

        if not display_reply:
             logger.warning(f"{send_prefix} Attempted to send an empty reply.")
             return

        # 使用更健壮的分隔符查找
        # Split by '$' but keep empty parts if necessary (e.g., "$hello$$world$")?
        # Current logic removes empty parts: `if p.strip()`
        # Let's stick with that for now.
        pattern = re.compile(r"[$＄]") # Pattern for both half/full width dollar
        if pattern.search(display_reply):
             parts = [p.strip() for p in pattern.split(display_reply) if p.strip()]
             if not parts and display_reply.strip(): # Handle cases like "$$" -> parts is empty
                  parts = [display_reply.replace("$", "").replace("＄","").strip()] # Send without dollars
        else:
            parts = [display_reply.strip()] # No delimiter

        if not parts: # If split results in nothing (e.g. empty string or only delimiters)
             logger.warning(f"{send_prefix} Message became empty after splitting by '$'. Not sending.")
             return


        logger.info(f"{send_prefix} 发送分段消息 ({len(parts)}段)...")
        part_num = 0
        for part in parts:
            part_num += 1
            logger.debug(f"{send_prefix} 发送段 {part_num}/{len(parts)}: '{part[:50]}...'")
            try:
                emotion_tags = self.emoji_handler.extract_emotion_tags(part)
                clean_part = part
                if emotion_tags:
                    logger.debug(f"{send_prefix} 段 {part_num} 含表情标签: {emotion_tags}")
                    for tag in emotion_tags:
                        clean_part = clean_part.replace(f'[{tag}]', '')

                if clean_part.strip():
                    self.wx.SendMsg(msg=clean_part.strip(), who=chat_id)
                    logger.debug(f"{send_prefix} Sent text part {part_num}.")

                # 发送该片段关联的表情
                if emotion_tags:
                    for emotion_type in emotion_tags:
                        logger.debug(f"{send_prefix} 发送表情 '{emotion_type}'...")
                        try:
                            emoji_path = self.emoji_handler.get_emoji_for_emotion(emotion_type)
                            if emoji_path:
                                self.wx.SendFiles(filepath=emoji_path, who=chat_id)
                                logger.debug(f"{send_prefix} 表情文件发送成功: {emotion_type}")
                                time.sleep(1) # 短暂延迟
                            else:
                                 logger.warning(f"{send_prefix} 未找到表情文件: {emotion_type}")
                        except FileNotFoundError:
                             logger.error(f"{send_prefix} 表情图像文件丢失或路径错误 for {emotion_type} at: {emoji_path}")
                        except Exception as e_emoji_send:
                            logger.error(f"{send_prefix} 发送表情失败 ({emotion_type}): {str(e_emoji_send)}")

                # Segment delay逻辑保持不变
                if len(parts) > 1 and part_num < len(parts) :
                     delay = random.randint(2, 4)
                     logger.debug(f"{send_prefix} 段间延迟 {delay}s...")
                     time.sleep(delay)

            except Exception as e_send_part:
                 logger.error(f"{send_prefix} 发送段 {part_num} 时出错: {e_send_part}", exc_info=True)
                 # Consider adding a max retry or skipping the part? For now, continue.

    def _send_raw_message(self, text: str, chat_id: str):
        """直接发送原始文本消息，保留主要格式 (逻辑不变)"""
           # ... 逻辑不变 ...
        send_prefix = f"[发送原始消息 to {chat_id}]"
        if not text: logger.warning(f"{send_prefix} Attempted to send empty raw message."); return
        try:
            # Process for display (mainly fixing emojis)
            display_text = self._process_text_for_display(text)

            emotion_tags = self.emoji_handler.extract_emotion_tags(display_text)
            clean_text = display_text
            if emotion_tags:
                 logger.debug(f"{send_prefix} Detected emotion tags: {emotion_tags}")
                 for tag in emotion_tags: clean_text = clean_text.replace(f'[{tag}]', '')

            # Remove potential splitters $ just in case they leak?
            final_text = clean_text.replace('$', '').replace('＄', '')

             # Check if any non-whitespace content remains
            if final_text.strip():
                logger.info(f"{send_prefix} 发送 raw: '{final_text[:100]}'... ")
                # Send the text - assuming wx.SendMsg handles newline characters ('\n') correctly
                self.wx.SendMsg(msg=final_text, who=chat_id)
            else:
                 logger.warning(f"{send_prefix} Raw message became empty after cleaning. Not sending.")


            # 发送该消息关联的表情
            if emotion_tags:
                 logger.debug(f"{send_prefix} Sending associated emojis for raw message...")
                 for emotion_type in emotion_tags:
                       logger.debug(f"{send_prefix} Trying to send assoc. emoji: {emotion_type}")
                       try:
                           emoji_path = self.emoji_handler.get_emoji_for_emotion(emotion_type)
                           if emoji_path:
                               self.wx.SendFiles(filepath=emoji_path, who=chat_id)
                               logger.debug(f"{send_prefix} Sent emoji file: {emotion_type}")
                               time.sleep(1)
                           else:
                               logger.warning(f"{send_prefix} No emoji file for tag: {emotion_type}")
                       except FileNotFoundError:
                             logger.error(f"{send_prefix} 表情图像文件丢失或路径错误 for {emotion_type} at: {emoji_path}")
                       except Exception as e_send_emoji:
                            logger.error(f"{send_prefix} 发送表情失败 ({emotion_type}): {e_send_emoji}")

        except Exception as e:
            logger.error(f"{send_prefix} 发送原始格式消息失败: {str(e)}", exc_info=True)


    def _clear_tts_text(self, text: str) -> str:
        """用于清洗回复,使得其适合进行TTS(逻辑不变)"""
           # ... 逻辑不变 ...
        try:
            # Remove emojis using replace_emoji
            cleaned = emoji.replace_emoji(text, replace='')
            # Replace $ with space/comma or remove for TTS clarity
            cleaned = cleaned.replace('$', ', ').replace('＄', ', ')
            # Remove bracketed tags like [emotion]
            cleaned = re.sub(r'\[.*?\]', '', cleaned)
             # Remove XML tags if they weren't caught earlier (like <thinking>)
            cleaned = re.sub(r'<.*?>', '', cleaned)
            cleaned = cleaned.strip() # Strip leading/trailing whitespace
            logger.debug(f"Cleaned text for TTS: '{cleaned[:100]}...'")
            return cleaned
        except Exception as e:
            logger.error(f"Error cleaning text for TTS: {e}")
            # Basic fallback
            fallback_cleaned = text.replace('$',' ').replace('＄',' ')
            fallback_cleaned = re.sub(r'\[.*?\]','', fallback_cleaned)
            return fallback_cleaned.strip()


    # ------ VVVVVV 修改 _handle_voice_request 调用 save_message VVVVVV ------
    def _handle_voice_request(self, content, chat_id, sender_name, username, is_group):
        """处理语音请求 (已集成 CoT 处理, 已修正 save_message 调用)"""
        log_prefix = f"[语音处理 for {sender_name}]"
        logger.info(log_prefix + f" (Group: {is_group})") # Add group status
        if is_group:
            logger.warning(f"{log_prefix} 暂不支持群聊语音功能，将以文本回复。")
            api_content = f"<用户 {sender_name}>\n{content}\n</用户>"
        else:
            api_content = content

        # 1. 获取原始回复 (可能含 [think])
        raw_reply = self.get_api_response(api_content, chat_id, is_group)
        logger.debug(f"{log_prefix} 原始LLM回复 (片段): {raw_reply[:150]}...")

        # 2. 处理 CoT，获取干净回复
        cleaned_reply, thought_content = self._process_cot_reply(raw_reply)
        logger.info(f"{log_prefix} 清理后回复 (片段): {cleaned_reply[:150]}...")


        # 3. 使用 cleaned_reply 进行后续操作
        if is_group:
            display_reply = f"@{sender_name} {cleaned_reply}"
            logger.debug(f"{log_prefix} 已添加群聊@")
            self._send_message_with_dollar(display_reply, chat_id)
        else:
            tts_text = self._clear_tts_text(cleaned_reply)
            logger.info(f"{log_prefix} 待合成TTS内容: {tts_text}")
            voice_path = self.voice_handler.generate_voice(tts_text)
            if voice_path:
                try:
                     logger.info(f"{log_prefix} 发送生成的语音文件: {voice_path}")
                     self.wx.SendFiles(filepath=voice_path, who=chat_id)
                     logger.info(f"{log_prefix} 语音文件发送成功。")
                except Exception as e:
                    logger.error(f"{log_prefix} 发送语音文件失败，将发送文本代替: {str(e)}", exc_info=True)
                    self._send_message_with_dollar(cleaned_reply, chat_id) # Fallback
                finally: # Ensure cleanup runs
                    if voice_path and os.path.exists(voice_path): # Check path exists before removing
                        try: os.remove(voice_path); logger.debug(f"{log_prefix} Removed temp voice file: {voice_path}")
                        except Exception as e_rem: logger.error(f"{log_prefix} 删除临时语音文件失败: {str(e_rem)}")
            else:
                logger.warning(f"{log_prefix} 语音生成失败，发送文本回复。")
                self._send_message_with_dollar(cleaned_reply, chat_id) # Fallback

        # --- 异步保存消息记录 ---
        is_system_message = sender_name == "System" or username == "System"
        # Save original user content and the FINAL cleaned AI reply
        save_content = content
        threading.Thread(
            target=self.save_message,
            # ------ VVV 传递 is_group VVV ------
            args=(chat_id, sender_name, save_content, cleaned_reply, is_group, is_system_message),
            daemon=True
        ).start()
        logger.debug(f"{log_prefix} 已启动消息保存线程 (使用清理后回复, 传递 is_group={is_group})")

        return cleaned_reply
   # ------ ^^^^^^ 修改 _handle_voice_request 调用 save_message 完成 ^^^^^^ ------


    # ------ VVVVVV 修改 _handle_text_message 调用 save_message VVVVVV ------
    def _handle_text_message(self, content, chat_id, sender_name, username, is_group):
        """处理普通文本消息 (已集成 CoT 处理, 已修正 save_message 调用)"""
        log_prefix = f"[文本处理 for {sender_name}]"
        logger.info(log_prefix + f" (Group: {is_group})") # Add group info

        command = None
        if content.startswith('/'):
            command = content.split(' ')[0].lower()
            logger.debug(f"{log_prefix} 检测到潜在命令: {command}")

        # 准备 API 内容
        if is_group:
            api_content = f"<用户 {sender_name}>\n{content}\n</用户>" # Group format
        else:
            api_content = content

        # 1. 获取 LLM 原始回复
        raw_reply = self.get_api_response(api_content, chat_id, is_group) # Use chat_id as user_id
        logger.debug(f"{log_prefix} 原始LLM回复 (片段): {raw_reply[:150]}...")

        # 2. 处理 CoT
        cleaned_reply, thought_content = self._process_cot_reply(raw_reply)
        logger.info(f"{log_prefix} 清理后回复 (片段): {cleaned_reply[:150]}...")

        # 3. 准备发送
        if is_group:
            display_reply = f"@{sender_name} {cleaned_reply}"
            logger.debug(f"{log_prefix} 已添加群聊@")
        else:
            display_reply = cleaned_reply

        # 4. 发送消息 (根据命令)
        if command and command in self.preserve_format_commands:
            logger.info(f"{log_prefix} 使用命令的原始格式发送: {command}")
            self._send_command_response(command, display_reply, chat_id)
        else:
            logger.info(f"{log_prefix} 使用分段($)模式发送")
            self._send_message_with_dollar(display_reply, chat_id)


        # --- 异步保存消息记录 ---
        is_system_message = sender_name == "System" or username == "System"
        # Save original user content, and the FINAL cleaned AI reply
        save_content = content # Pass the original user content
        threading.Thread(
            target=self.save_message,
            # ------ VVV 传递 is_group VVV ------
            args=(chat_id, sender_name, save_content, cleaned_reply, is_group, is_system_message),
            daemon=True
        ).start()
        logger.debug(f"{log_prefix} 已启动消息保存线程 (使用清理后回复, 传递 is_group={is_group})")

        return cleaned_reply
    # ------ ^^^^^^ 修改 _handle_text_message 调用 save_message 完成 ^^^^^^ ------


    # ------ VVVVVV 修改 _handle_random_image_request 调用 save_message VVVVVV ------
    def _handle_random_image_request(self, content, chat_id, sender_name, username, is_group):
        """处理随机图片请求 (已修正 save_message 调用)"""
        log_prefix = f"[随机图片处理 for {sender_name}]"
        logger.info(log_prefix + f" (Group: {is_group})") # Add group info
        
        # No LLM text reply gen here, so no CoT processing needed for 'reply'
        image_path = self.image_handler.get_random_image()
        if image_path:
            # Use a fixed or simple reply text
            reply_text = "给你找了一张好看的图片哦~"
            display_reply_text = reply_text # Keep unformatted for saving
            if is_group: display_reply_text = f"@{sender_name} {reply_text}" # Format for sending

            try:
                self.wx.SendFiles(filepath=image_path, who=chat_id)
                logger.info(f"{log_prefix} 发送随机图片成功: {image_path}")
                # Send the accompanying text *after* the image
                self.wx.SendMsg(msg=display_reply_text, who=chat_id)
                logger.info(f"{log_prefix} 发送图片伴随文本成功")
            except Exception as e:
                logger.error(f"{log_prefix} 发送随机图片或其文本失败: {str(e)}", exc_info=True)
                reply_text = "抱歉主人，图片发送失败了..." # Update fixed reply on error
                display_reply_text = reply_text
                if is_group: display_reply_text = f"@{sender_name} {reply_text}"
                 # Send error message text
                try: self.wx.SendMsg(msg=display_reply_text, who=chat_id)
                except: pass # Ignore secondary error
            finally: # Cleanup
                if os.path.exists(image_path):
                    try: os.remove(image_path); logger.debug(f"{log_prefix} Removed temp image: {image_path}")
                    except Exception as e_rem: logger.error(f"{log_prefix} 删除临时图片失败: {str(e_rem)}")


            # --- 异步保存消息记录 ---
            is_system_message = sender_name == "System" or username == "System"
            # Use original user 'content' and the fixed 'reply_text' (not display_reply_text with @)
            save_content = content
            threading.Thread(
                target=self.save_message,
                # ------ VVV 传递 is_group VVV ------
                args=(chat_id, sender_name, save_content, reply_text, is_group, is_system_message),
                daemon=True
            ).start()
            logger.debug(f"{log_prefix} 已启动图片请求消息保存线程 (传递 is_group={is_group})")
            return reply_text # Return the base reply text
        else:
              logger.error(f"{log_prefix} 获取随机图片路径失败。")
              return None
    # ------ ^^^^^^ _handle_random_image_request 调用 save_message 完成 ^^^^^^ ------


    # ------ VVVVVV 修改 _handle_image_generation_request 调用 save_message VVVVVV ------
    def _handle_image_generation_request(self, content, chat_id, sender_name, username, is_group):
        """处理图像生成请求 (已修正 save_message 调用)"""
        # Assuming LLM is *not* used here for text reply generation.
        # If LLM *is* used for reply, _process_cot_reply must be used.
        log_prefix = f"[图像生成处理 for {sender_name}]"
        logger.info(log_prefix + f" (Group: {is_group})") # Add group info


        image_path = self.image_handler.generate_image(content) # Assumes content is the prompt
        if image_path:
            # Fixed reply text
            reply_text = "这是按照主人您的要求生成的图片\\(^o^)/~"
            display_reply_text = reply_text
            if is_group: display_reply_text = f"@{sender_name} {reply_text}"

            try:
                self.wx.SendFiles(filepath=image_path, who=chat_id)
                logger.info(f"{log_prefix} 发送生成图片成功: {image_path}")
                 # Send the accompanying text *after* the image
                self.wx.SendMsg(msg=display_reply_text, who=chat_id)
                logger.info(f"{log_prefix} 发送图片伴随文本成功")
            except Exception as e:
                logger.error(f"{log_prefix} 发送生成图片或其文本失败: {str(e)}")
                reply_text = "抱歉主人，图片生成失败了..." # Update reply text on error
                display_reply_text = reply_text
                if is_group: display_reply_text = f"@{sender_name} {reply_text}"
                 # Send error message text
                try: self.wx.SendMsg(msg=display_reply_text, who=chat_id)
                except: pass
            finally: # Cleanup
                if os.path.exists(image_path):
                    try: os.remove(image_path); logger.debug(f"{log_prefix} 删除临时生成图片: {image_path}")
                    except Exception as e: logger.error(f"{log_prefix} 删除临时生成图片失败: {str(e)}")

            # --- 异步保存消息记录 ---
            is_system_message = sender_name == "System" or username == "System"
            # Use original user 'content' (the prompt) and the fixed 'reply_text'
            save_content = content
            threading.Thread(
                target=self.save_message,
                # ------ VVV 传递 is_group VVV ------
                args=(chat_id, sender_name, save_content, reply_text, is_group, is_system_message),
                daemon=True
            ).start()
            logger.debug(f"{log_prefix} 已启动图像生成消息保存线程 (传递 is_group={is_group})")
            return reply_text # Return base reply text
        else:
             logger.error(f"{log_prefix} 图像生成失败 DALL-E/etc 调用出错")
             # Maybe send error text?
             error_reply = "抱歉主人，生成图片似乎遇到了问题..."
             if is_group: error_reply = f"@{sender_name} {error_reply}"
             try: self.wx.SendMsg(msg=error_reply, who=chat_id)
             except: pass
             return None # Indicate failure
     # ------ ^^^^^^ _handle_image_generation_request 调用 save_message 完成 ^^^^^^ ------


    def _send_command_response(self, command: str, reply: str, chat_id: str):
        """发送命令响应，根据命令类型决定格式 (逻辑不变)"""
        # ... 逻辑不变 ...
        log_prefix = f"[命令响应 for {command} to {chat_id}]"
        if not reply: logger.warning(f"{log_prefix} 命令回复为空，不发送。"); return

        cleaned_reply, _ = self._process_cot_reply(reply) # Clean CoT just in case

        # Check command against list for preservation needs
        if command in self.preserve_format_commands:
            logger.info(f"{log_prefix} 使用原始格式发送。")
            # Pass the cleaned reply to raw sender
            self._send_raw_message(cleaned_reply, chat_id)
        else:
            logger.info(f"{log_prefix} 使用标准分段模式发送。")
            # Pass the cleaned reply to dollar sender
            self._send_message_with_dollar(cleaned_reply, chat_id)


    # --- 其他辅助函数保持不变 ---
    def _add_to_system_prompt(self, chat_id: str, content: str) -> None:
        # (代码不变)
         try:
             if not hasattr(self, 'system_prompts'): self.system_prompts = {}
             if chat_id not in self.system_prompts: self.system_prompts[chat_id] = []
             self.system_prompts[chat_id].append(content)
             max_prompts = 5 # Consider making this configurable via config file
             if len(self.system_prompts[chat_id]) > max_prompts:
                   logger.debug(f"Trimming system prompts for {chat_id}, exceeded {max_prompts}")
                   self.system_prompts[chat_id] = self.system_prompts[chat_id][-max_prompts:]
             logger.info(f"Added/Updated system prompts for {chat_id}. Count: {len(self.system_prompts[chat_id])}.")
             logger.debug(f"Current system prompts for {chat_id}: {self.system_prompts[chat_id]}")
         except Exception as e: logger.error(f"Failed to add to system prompt for {chat_id}: {e}")

    def _remove_search_content_from_context(self, chat_id: str, content: str) -> None:
        # (代码不变 - 实验性)
        logger.warning(f"Function _remove_search_content_from_context is likely experimental and may not function correctly.")
        pass

    def _async_generate_summary(self, chat_id: str, url: str, content: str, model: str = None) -> None:
        # (代码不变)
        log_prefix=f"[异步总结 for {chat_id}] Query/URL: {url[:50]}..."
        try:
            wait_time = 30 # Configurable?
            logger.info(f"{log_prefix} Waiting {wait_time}s before generating summary.")
            time.sleep(wait_time)

            logger.info(f"{log_prefix} Starting summary generation.")

            # Determine model for summary
            summary_model = model
            if not summary_model :
                try:
                     # Safely access nested config attributes
                     summary_model = config.llm.model # Primary config model
                except AttributeError:
                     logger.error(f"{log_prefix} config.llm.model is missing! Falling back.")
                     # Fallback: Use the handler's main model? Requires passing base_url, api_key?
                     # Or define a default hard coded model?
                     if hasattr(self, 'model'): summary_model = self.model # Use handler's model if exists
                     else: logger.error(f"{log_prefix} No fallback model found! Cannot generate summary."); return

            logger.info(f"{log_prefix} Using model for summary: {summary_model}")

            summary_prompt = f"请将以下关于'{url[:30]}...'的内容总结为简洁的要点:\n\n{content}"
            summary_messages = [{"role": "user", "content": summary_prompt }]

            llm_client_for_summary = None
            if hasattr(self, 'network_search_service') and hasattr(self.network_search_service, 'llm_service'):
                 llm_client_for_summary = self.network_search_service.llm_service
            elif hasattr(self, 'deepseek'): # Fallback to handler's main LLM if network one isn't available
                 llm_client_for_summary = self.deepseek
                 logger.warning(f"{log_prefix} Using main LLM service for summary as network LLM service unavailable.")
            else:
                logger.error(f"{log_prefix} No suitable LLM client available for summary generation.")
                return

            summary_result = None
            try:
                # Assuming get_response is the standard method
                 summary_result = llm_client_for_summary.get_response(
                      message=summary_prompt, # Use prompt as message
                      user_id=f"summary_{chat_id}", # Task-specific user ID
                      system_prompt="You are a summarization bot.",
                      model=summary_model,
                      temperature=0.5 # Optional reduced temperature for summary
                 )
            except Exception as e_llm_call:
                 logger.error(f"{log_prefix} 调用LLM进行总结时出错: {e_llm_call}", exc_info=True)


            if summary_result:
                 # Ensure result is not empty/null
                if isinstance(summary_result, str) and summary_result.strip():
                     final_summary = f"参考信息 关于 '{url[:30]}...': {summary_result.strip()}"
                     # Attempt removal (experimental)
                     # self._remove_search_content_from_context(chat_id, content) # Keep commented out unless needed
                     self._add_to_system_prompt(chat_id, final_summary)
                     logger.info(f"{log_prefix} 异步总结已添加到系统提示词。")
                     logger.debug(f"Summary added: {final_summary}")
                else:
                    logger.warning(f"{log_prefix} LLM返回的总结为空或无效。")

            else:
                 logger.warning(f"{log_prefix} 无法生成有效的异步总结。")

        except Exception as e: # Catch wider exceptions in the thread
            logger.error(f"{log_prefix} 异步总结线程意外失败: {str(e)}", exc_info=True)


    def _check_time_reminder_and_search(self, content: str, sender_name: str) -> bool:
        # (代码不变 - 保持禁用复杂逻辑)
        if sender_name == "System" or sender_name.lower() == "system" : return False
        return False # Temporarily disabled


    def _check_time_reminder(self, content: str, chat_id: str, sender_name: str):
        # (代码不变)
        if sender_name == "System" or sender_name.lower() == "system" : return
        log_prefix=f"[时间提醒检查 for {sender_name}]"
        try:
            time_infos = self.time_recognition.recognize_time(content)
            if time_infos:
                logger.info(f"{log_prefix} 检测到 {len(time_infos)} 个提醒请求。")
                for target_time, reminder_content in time_infos:
                     logger.info(f"{log_prefix} - 时间: {target_time}, 内容: {reminder_content[:50]}...")
                     # Use ReminderService Instance
                     success = self.reminder_service.add_reminder(
                           chat_id=chat_id, target_time=target_time, content=reminder_content,
                           sender_name=sender_name, silent=True # Assuming silent adds without user confirmation msg now
                     )
                     if success: logger.info(f"{log_prefix} 提醒任务 for {target_time} 创建成功.")
                     else: logger.error(f"{log_prefix} 提醒任务 for {target_time} 创建失败！")
        except AttributeError:
             logger.error(f"{log_prefix} time_recognition 或 reminder_service 不可用。")
        except Exception as e:
            logger.error(f"{log_prefix} 时间提醒处理失败: {str(e)}", exc_info=True)


    def add_to_queue(self, chat_id: str, content: str, sender_name: str,
                    username: str, is_group: bool = False):
        # 调用新的主入口
        # Old method name was potentially confusing, handle_user_message is better
        self.handle_user_message(content, chat_id, sender_name, username, is_group)

    def process_messages(self, chat_id: str):
        # 保持废弃状态
        logger.warning("process_messages方法已废弃，队列通过定时器自动处理。")
        pass

# End of MessageHandler Class
