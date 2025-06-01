import logging
import random
from datetime import datetime, timedelta
import threading
import time
import os
import shutil
from src.config import config, DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, MODEL, MAX_TOKEN, TEMPERATURE, MAX_GROUPS
from wxauto import WeChat
import re
from src.handlers.emoji import EmojiHandler
from src.handlers.image import ImageHandler
from src.handlers.message import MessageHandler
from src.handlers.voice import VoiceHandler
from src.services.ai.llm_service import LLMService
from src.services.ai.image_recognition_service import ImageRecognitionService
from modules.memory.memory_service import MemoryService
from modules.memory.content_generator import ContentGenerator
from src.utils.logger import LoggerConfig
from src.utils.console import print_status
from colorama import init, Style
from src.AutoTasker.autoTasker import AutoTasker
from src.handlers.autosend import AutoSendHandler

# 创建一个事件对象来控制线程的终止
stop_event = threading.Event()

# 获取项目根目录
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 检查并初始化配置文件
config_path = os.path.join(root_dir, 'src', 'config', 'config.json')
config_template_path = os.path.join(root_dir, 'src', 'config', 'config.json.template')

if not os.path.exists(config_path) and os.path.exists(config_template_path):
    logger = logging.getLogger('main')
    logger.info("配置文件不存在，正在从模板创建...")
    shutil.copy2(config_template_path, config_path)
    logger.info(f"已从模板创建配置文件: {config_path}")

# 初始化colorama
init()

# 全局变量
logger = None
listen_list = []

def initialize_logging():
    """初始化日志系统"""
    global logger, listen_list

    # 清除所有现有日志处理器
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logger_config = LoggerConfig(root_dir)
    logger = logger_config.setup_logger('main')
    listen_list = config.user.listen_list

# 消息队列接受消息时间间隔
wait = 1

class ChatBot:
    def __init__(self, message_handler, image_recognition_service, auto_sender, emoji_handler):
        self.message_handler = message_handler
        self.image_recognition_service = image_recognition_service
        self.auto_sender = auto_sender
        self.emoji_handler = emoji_handler

        # 获取机器人的微信名称
        self.wx = WeChat()
        self.robot_name = self.wx.A_MyIcon.Name  # 使用Name属性而非方法
        logger.info(f"机器人名称: {self.robot_name}")

    def handle_wxauto_message(self, msg, chatName, is_group=False):
        try:
            username = msg.sender
            content = getattr(msg, 'content', None) or getattr(msg, 'text', None)

            # 重置倒计时
            self.auto_sender.start_countdown()

            # 简化日志输出
            logger.info(f"收到消息 - 来自: {username}" + (" (群聊)" if is_group else ""))
            logger.debug(f"消息内容: {content}")

            img_path = None
            is_emoji = False
            is_image_recognition = False

            # 处理群聊@消息
            if is_group and self.robot_name and content:
                content = re.sub(f'@{self.robot_name}\u2005', '', content).strip()

            if content and content.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                img_path = content
                is_emoji = False
                content = None

            # 检查动画表情
            if content and "[动画表情]" in content:
                img_path = self.emoji_handler.capture_and_save_screenshot(username)
                is_emoji = True
                content = None

            if img_path:
                recognized_text = self.image_recognition_service.recognize_image(img_path, is_emoji)
                content = recognized_text if content is None else f"{content} {recognized_text}"
                is_image_recognition = True

            # 处理消息
            if content:
                sender_name = username
                # 直接添加到消息队列
                self.message_handler.handle_user_message(
                    content=content,
                    chat_id=chatName,
                    sender_name=sender_name,
                    username=username,
                    is_group=is_group,
                    is_image_recognition=is_image_recognition
                )

        except Exception as e:
            logger.error(f"消息处理失败: {str(e)}")

# 全局变量
prompt_content = ""
emoji_handler = None
image_handler = None
voice_handler = None
memory_service = None
content_generator = None
message_handler = None
image_recognition_service = None
auto_sender = None
chat_bot = None
ROBOT_WX_NAME = ""

def initialize_services():
    """初始化服务实例"""
    global prompt_content, emoji_handler, image_handler, voice_handler, memory_service, content_generator
    global message_handler, image_recognition_service, auto_sender, chat_bot, ROBOT_WX_NAME

    # 读取提示文件
    avatar_dir = os.path.join(root_dir, config.behavior.context.avatar_dir)
    prompt_path = os.path.join(avatar_dir, "avatar.md")
    if os.path.exists(prompt_path):
        with open(prompt_path, "r", encoding="utf-8") as file:
            prompt_content = file.read()

        # 处理无法读取文件的情况
    else:
        raise FileNotFoundError(f"avatar.md 文件不存在: {prompt_path}")

    # 创建服务实例
    emoji_handler = EmojiHandler(root_dir)
    image_handler = ImageHandler(
        root_dir=root_dir,
        api_key=config.llm.api_key,
        base_url=config.llm.base_url,
        image_model=config.media.image_generation.model
    )
    voice_handler = VoiceHandler(
        root_dir=root_dir,
        tts_api_key=config.media.text_to_speech.tts_api_key
    )
    memory_service = MemoryService(
        root_dir=root_dir,
        api_key=DEEPSEEK_API_KEY,
        base_url=DEEPSEEK_BASE_URL,
        model=MODEL,
        max_token=MAX_TOKEN,
        temperature=TEMPERATURE,
        max_groups=MAX_GROUPS
    )

    content_generator = ContentGenerator(
        root_dir=root_dir,
        api_key=DEEPSEEK_API_KEY,
        base_url=DEEPSEEK_BASE_URL,
        model=MODEL,
        max_token=MAX_TOKEN,
        temperature=TEMPERATURE
    )
    # 创建图像识别服务
    image_recognition_service = ImageRecognitionService(
        api_key=config.media.image_recognition.api_key,
        base_url=config.media.image_recognition.base_url,
        temperature=config.media.image_recognition.temperature,
        model=config.media.image_recognition.model
    )

    # 获取机器人名称
    try:
        wx = WeChat()
        ROBOT_WX_NAME = wx.A_MyIcon.Name  # 使用Name属性而非方法
        logger.info(f"获取到机器人名称: {ROBOT_WX_NAME}")
    except Exception as e:
        logger.warning(f"获取机器人名称失败: {str(e)}")
        ROBOT_WX_NAME = ""

    # 创建消息处理器
    message_handler = MessageHandler(
        root_dir=root_dir,
        api_key=config.llm.api_key,
        base_url=config.llm.base_url,
        model=config.llm.model,
        max_token=config.llm.max_tokens,
        temperature=config.llm.temperature,
        max_groups=config.behavior.context.max_groups,
        robot_name=ROBOT_WX_NAME,  # 使用动态获取的机器人名称
        prompt_content=prompt_content,
        image_handler=image_handler,
        emoji_handler=emoji_handler,
        voice_handler=voice_handler,
        memory_service=memory_service,  # 使用新的记忆服务
        content_generator=content_generator  # 直接传递内容生成器实例
    )

    # 创建主动消息处理器
    auto_sender = AutoSendHandler(message_handler, config, listen_list)

    # 创建聊天机器人实例
    chat_bot = ChatBot(message_handler, image_recognition_service, auto_sender, emoji_handler)

    # 启动主动消息倒计时
    auto_sender.start_countdown()

def message_listener():
    # 使用全局变量
    global chat_bot, ROBOT_WX_NAME, logger, wait

    wx = None
    last_window_check = 0
    check_interval = 600

    while not stop_event.is_set():
        try:
            current_time = time.time()

            if wx is None or (current_time - last_window_check > check_interval):
                wx = WeChat()
                if not wx.GetSessionList():
                    time.sleep(5)
                    continue
                last_window_check = current_time

            msgs = wx.GetListenMessage()
            if not msgs:
                time.sleep(wait)
                continue

            for chat in msgs:
                who = chat.who
                if not who:
                    continue

                one_msgs = msgs.get(chat)
                if not one_msgs:
                    continue

                for msg in one_msgs:
                    try:
                        msgtype = msg.type
                        content = msg.content
                        if not content:
                            continue
                        if msgtype != 'friend':
                            logger.debug(f"非好友消息，忽略! 消息类型: {msgtype}")
                            continue
                            # 接收窗口名跟发送人一样，代表是私聊，否则是群聊
                        if who == msg.sender:
                            # 确保chat_bot已经初始化
                            if chat_bot:
                                chat_bot.handle_wxauto_message(msg, msg.sender) # 处理私聊信息
                            else:
                                logger.warning("聊天机器人实例尚未初始化")
                        elif ROBOT_WX_NAME != '' and (bool(re.search(f'@{ROBOT_WX_NAME}\u2005', msg.content)) or bool(re.search(f'{ROBOT_WX_NAME}\u2005', msg.content))):
                            # 修改：在群聊被@时或者被叫名字，传入群聊ID(who)作为回复目标
                            if chat_bot:
                                chat_bot.handle_wxauto_message(msg, who, is_group=True)
                            else:
                                logger.warning("聊天机器人实例尚未初始化")
                        else:
                            logger.debug(f"非需要处理消息，可能是群聊非@消息: {content}")
                    except Exception as e:
                        logger.debug(f"处理单条消息失败: {str(e)}")
                        continue

        except Exception as e:
            logger.debug(f"消息监听出错: {str(e)}")
            wx = None
        time.sleep(wait)

def initialize_wx_listener():
    """
    初始化微信监听，包含重试机制
    """
    # 使用全局变量
    global listen_list, logger

    max_retries = 3
    retry_delay = 2  # 秒

    for attempt in range(max_retries):
        try:
            wx = WeChat()
            if not wx.GetSessionList():
                logger.error("未检测到微信会话列表，请确保微信已登录")
                time.sleep(retry_delay)
                continue

            # 循环添加监听对象，修改savepic参数为False
            for chat_name in listen_list:
                try:
                    # 先检查会话是否存在
                    if not wx.ChatWith(chat_name):
                        logger.error(f"找不到会话: {chat_name}")
                        continue

                    # 尝试添加监听，设置savepic=False
                    wx.AddListenChat(who=chat_name, savepic=True)
                    logger.info(f"成功添加监听: {chat_name}")
                    time.sleep(0.5)  # 添加短暂延迟，避免操作过快
                except Exception as e:
                    logger.error(f"添加监听失败 {chat_name}: {str(e)}")
                    continue

            return wx

        except Exception as e:
            logger.error(f"初始化微信失败 (尝试 {attempt + 1}/{max_retries}): {str(e)}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
            else:
                raise Exception("微信初始化失败，请检查微信是否正常运行")

    return None

def initialize_auto_tasks(message_handler):
    """初始化自动任务系统"""
    print_status("初始化自动任务系统...", "info", "CLOCK")

    try:
        # 导入config变量
        from src.config import config

        # 创建AutoTasker实例
        auto_tasker = AutoTasker(message_handler)
        print_status("创建AutoTasker实例成功", "success", "CHECK")

        # 清空现有任务
        auto_tasker.scheduler.remove_all_jobs()
        print_status("清空现有任务", "info", "CLEAN")

        # 从配置文件读取任务信息
        if hasattr(config, 'behavior') and hasattr(config.behavior, 'schedule_settings'):
            schedule_settings = config.behavior.schedule_settings
            if schedule_settings and schedule_settings.tasks:  # 直接检查 tasks 列表
                tasks = schedule_settings.tasks
                if tasks:
                    print_status(f"从配置文件读取到 {len(tasks)} 个任务", "info", "TASK")
                    tasks_added = 0

                    # 遍历所有任务并添加
                    for task in tasks:
                        try:
                            # 添加定时任务
                            auto_tasker.add_task(
                                task_id=task.task_id,
                                chat_id=listen_list[0],  # 使用 listen_list 中的第一个聊天ID
                                content=task.content,
                                schedule_type=task.schedule_type,
                                schedule_time=task.schedule_time
                            )
                            tasks_added += 1
                            print_status(f"成功添加任务 {task.task_id}: {task.content}", "success", "CHECK")
                        except Exception as e:
                            print_status(f"添加任务 {task.task_id} 失败: {str(e)}", "error", "ERROR")

                    print_status(f"成功添加 {tasks_added}/{len(tasks)} 个任务", "info", "TASK")
                else:
                    print_status("配置文件中没有找到任务", "warning", "WARNING")
        else:
            print_status("未找到任务配置信息", "warning", "WARNING")
            print_status(f"当前 behavior 属性: {dir(config.behavior)}", "info", "INFO")

        return auto_tasker

    except Exception as e:
        print_status(f"初始化自动任务系统失败: {str(e)}", "error", "ERROR")
        logger.error(f"初始化自动任务系统失败: {str(e)}")
        return None

def switch_avatar(new_avatar_name):
    # 使用全局变量
    global emoji_handler, message_handler, root_dir

    # 导入config变量
    from src.config import config

    # 更新配置
    config.behavior.context.avatar_dir = f"avatars/{new_avatar_name}"

    # 重新初始化 emoji_handler
    emoji_handler = EmojiHandler(root_dir)

    # 更新 message_handler 中的 emoji_handler
    if message_handler:
        message_handler.emoji_handler = emoji_handler
    else:
        logger.warning("消息处理器实例尚未初始化")

def main():
    # 初始化变量
    listener_thread = None

    try:
        # 初始化日志系统
        initialize_logging()

        # 初始化服务实例
        initialize_services()

        # 设置wxauto日志路径
        automation_log_dir = os.path.join(root_dir, "logs", "automation")
        if not os.path.exists(automation_log_dir):
            os.makedirs(automation_log_dir)
        os.environ["WXAUTO_LOG_PATH"] = os.path.join(automation_log_dir, "AutomationLog.txt")

        # 初始化微信监听
        print_status("初始化微信监听...", "info", "BOT")
        wx = initialize_wx_listener()
        if not wx:
            print_status("微信初始化失败，请确保微信已登录并保持在前台运行!", "error", "CROSS")
            return
        print_status("微信监听初始化完成", "success", "CHECK")

        # 验证记忆目录
        print_status("验证角色记忆存储路径...", "info", "FILE")
        avatar_dir = os.path.join(root_dir, config.behavior.context.avatar_dir)
        avatar_name = os.path.basename(avatar_dir)
        memory_dir = os.path.join(avatar_dir, "memory")
        if not os.path.exists(memory_dir):
            os.makedirs(memory_dir)
            print_status(f"创建角色记忆目录: {memory_dir}", "success", "CHECK")

        # 初始化记忆文件 - 为每个监听用户创建独立的记忆文件
        print_status("初始化记忆文件...", "info", "FILE")

        # 为每个监听的用户创建独立记忆
        for user_name in listen_list:
            print_status(f"为用户 '{user_name}' 创建独立记忆...", "info", "USER")
            # 使用用户名作为用户ID
            memory_service.initialize_memory_files(avatar_name, user_id=user_name)
            print_status(f"用户 '{user_name}' 记忆初始化完成", "success", "CHECK")

        avatar_dir = os.path.join(root_dir, config.behavior.context.avatar_dir)
        prompt_path = os.path.join(avatar_dir, "avatar.md")
        if not os.path.exists(prompt_path):
            with open(prompt_path, "w", encoding="utf-8") as f:
                f.write("# 核心人格\n[默认内容]")
            print_status(f"创建人设提示文件", "warning", "WARNING")
        # 启动消息监听线程
        print_status("启动消息监听线程...", "info", "ANTENNA")
        listener_thread = threading.Thread(target=message_listener)
        listener_thread.daemon = True  # 确保线程是守护线程
        listener_thread.start()
        print_status("消息监听已启动", "success", "CHECK")

        # 初始化主动消息系统
        print_status("初始化主动消息系统...", "info", "CLOCK")
        print_status("主动消息系统已启动", "success", "CHECK")

        print("-" * 50)
        print_status("系统初始化完成", "success", "STAR_2")
        print("=" * 50)

        # 初始化自动任务系统
        auto_tasker = initialize_auto_tasks(message_handler)
        if not auto_tasker:
            print_status("自动任务系统初始化失败", "error", "ERROR")
            return

        # 主循环
        while True:
            time.sleep(1)
            if not listener_thread.is_alive():
                print_status("监听线程已断开，尝试重新连接...", "warning", "SYNC")
                try:
                    wx = initialize_wx_listener()
                    if wx:
                        listener_thread = threading.Thread(target=message_listener)
                        listener_thread.daemon = True
                        listener_thread.start()
                        print_status("重新连接成功", "success", "CHECK")
                except Exception as e:
                    print_status(f"重新连接失败: {str(e)}", "error", "CROSS")
                    time.sleep(5)

    except Exception as e:
        print_status(f"主程序异常: {str(e)}", "error", "ERROR")
        logger.error(f"主程序异常: {str(e)}", exc_info=True)
    finally:
        # 清理资源
        if 'auto_sender' in locals():
            auto_sender.stop()

        # 设置事件以停止线程
        stop_event.set()

        # 关闭监听线程
        if listener_thread and listener_thread.is_alive():
            print_status("正在关闭监听线程...", "info", "SYNC")
            listener_thread.join(timeout=2)
            if listener_thread.is_alive():
                print_status("监听线程未能正常关闭", "warning", "WARNING")

        print_status("正在关闭系统...", "warning", "STOP")
        print_status("系统已退出", "info", "BYE")
        print("\n")

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n")
        print_status("用户终止程序", "warning", "STOP")
        print_status("感谢使用，再见！", "info", "BYE")
        print("\n")
    except Exception as e:
        print_status(f"程序异常退出: {str(e)}", "error", "ERROR")
