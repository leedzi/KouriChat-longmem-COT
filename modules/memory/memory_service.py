import os
# import json # 不再需要 json
import logging
from typing import List, Dict, Optional # 保留 Dict for get_recent_context
from datetime import datetime
from src.services.ai.llm_service import LLMService

logger = logging.getLogger('main')

class MemoryService:
    """
    记忆服务模块 - 已修改以模拟旧版长时记忆行为，并调整核心记忆获取方式。
    1. 短期记忆 (short_memory.txt): 保存对话，达到阈值后调用总结。
    2. '核心记忆' (`long_memory_buffer.txt`):
       - `get_core_memory()` 被修改为读取并返回此文件的【全部内容】。
       - `update_core_memory()` 执行总结并将结果【追加】到此文件。
    3. `get_relevant_memories()` 已被禁用，总是返回空列表。
    """
    def __init__(self, root_dir: str, api_key: str, base_url: str, model: str, max_token: int, temperature: float):
        self.root_dir = root_dir
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.max_token = max_token
        self.temperature = temperature
        self.llm_client = None
        # self.conversation_count = {} # 已移除

    # --- 内部帮助方法 ---
    def _get_avatar_memory_dir(self, avatar_name: str) -> str:
        """获取角色记忆目录，如果不存在则创建"""
        avatar_memory_dir = os.path.join(self.root_dir, "data", "avatars", avatar_name, "memory")
        os.makedirs(avatar_memory_dir, exist_ok=True)
        return avatar_memory_dir

    def _get_short_memory_path(self, avatar_name: str) -> str:
        """获取短期记忆文件路径 (TXT格式)"""
        memory_dir = self._get_avatar_memory_dir(avatar_name)
        return os.path.join(memory_dir, "short_memory.txt")

    def _get_core_memory_path(self, avatar_name: str) -> str:
        """
        获取『核心』记忆文件路径 (实际为长期记忆缓冲区 TXT格式)
        注意：此方法名保留兼容性，但指向的是长期记忆累积文件。
        """
        memory_dir = self._get_avatar_memory_dir(avatar_name)
        return os.path.join(memory_dir, "long_memory_buffer.txt")

    def _get_llm_client(self):
        """获取或创建LLM客户端"""
        if not self.llm_client:
            self.llm_client = LLMService(
                api_key=self.api_key,
                base_url=self.base_url,
                model=self.model,
                max_token=self.max_token, # 用于总结的 token 可能需要调整
                temperature=self.temperature, # 用于总结的 temperature 可能需要调整
                max_groups=10 # 此 max_groups 参数是给 LLMService 的上下文控制，总结时可能不需要那么多
            )
        return self.llm_client

    def _get_timestamp(self) -> str:
        """获取当前时间戳"""
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def initialize_memory_files(self, avatar_name: str):
        """初始化角色的记忆文件 (TXT格式)，确保文件存在"""
        try:
            memory_dir = self._get_avatar_memory_dir(avatar_name) # 确保目录存在
            short_memory_path = self._get_short_memory_path(avatar_name)
            long_memory_buffer_path = self._get_core_memory_path(avatar_name) # 即 long_memory_buffer.txt

            if not os.path.exists(short_memory_path):
                with open(short_memory_path, "w", encoding="utf-8"):
                    pass # 创建空文件
                logger.info(f"创建短期记忆文件: {short_memory_path}")

            if not os.path.exists(long_memory_buffer_path):
                with open(long_memory_buffer_path, "w", encoding="utf-8"):
                     pass # 创建空文件
                logger.info(f"创建长期记忆缓冲区文件: {long_memory_buffer_path}")

        except Exception as e:
            logger.error(f"初始化记忆文件失败 ({ avatar_name }): {str(e)}")

    def add_conversation(self, avatar_name: str, user_message: str, bot_reply: str, is_system_message: bool = False):
        """
        添加对话到短期记忆 (.txt)，并检查是否达到总结阈值以调用 update_core_memory。
        """
        if is_system_message:
            logger.debug(f"跳过记录系统消息: {user_message[:30]}...")
            return

        try:
            short_memory_path = self._get_short_memory_path(avatar_name)
            if not os.path.exists(short_memory_path): # 文件检查
                logger.warning(f"短期记忆文件丢失，尝试重新初始化: {short_memory_path}")
                self.initialize_memory_files(avatar_name)
                if not os.path.exists(short_memory_path): # 再次检查
                    logger.error(f"无法创建短期记忆文件，无法记录对话: {short_memory_path}")
                    return

            with open(short_memory_path, "a", encoding="utf-8") as f:
                f.write(f"用户: {user_message}\n")
                f.write(f"bot: {bot_reply}\n\n")

            line_count = 0
            with open(short_memory_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                         line_count += 1
            
            # 阈值: 30行 (指非空行，大约等于15轮对话)
            if line_count >= 30:
                logger.info(f"角色 {avatar_name} 短期记忆达到 {line_count} 行，调用 update_core_memory (实际为总结到长期记忆)...")
                self.update_core_memory(avatar_name)

        except Exception as e:
            logger.error(f"添加对话到短期记忆失败 ({ avatar_name }): {str(e)}")

    def update_core_memory(self, avatar_name: str):
        """
        !! 功能重写：执行短期记忆的总结，并将结果【追加】到长期记忆缓冲区，然后清空短期记忆 !!
        (保留原始函数签名，但现在它的目标是累积长期记忆)
        """
        logger.info(f"开始为角色 {avatar_name} 执行记忆总结并追加到长期记忆...")
        short_memory_path = self._get_short_memory_path(avatar_name)
        long_memory_buffer_path = self._get_core_memory_path(avatar_name) # 指向 long_memory_buffer.txt

        if not os.path.exists(short_memory_path):
            logger.warning(f"短期记忆文件不存在，无法总结: {avatar_name}")
            return

        try:
            dialog_to_summarize = ""
            with open(short_memory_path, "r", encoding="utf-8") as f:
                dialog_to_summarize = f.read()

            if not dialog_to_summarize.strip():
                 logger.info(f"短期记忆内容为空，跳过总结: {avatar_name}")
                 return

            max_retries = 3
            retries = 0
            summary = None
            while retries < max_retries:
                try:
                    system_prompt_summarize = "请将以下对话记录总结为最重要的几条长期记忆，总结内容要求：1. 严格控制字数在50-100字内2. 仅保留对未来对话至关重要的信息3. 按优先级提取：用户个人信息 > 用户偏好/喜好 > 重要约定 > 特殊事件 > 常去地点4. 使用第一人称视角撰写，仿佛是你自己在记录对话记忆5. 使用极简句式，省略不必要的修饰词,禁止使用颜文字和括号描述动作6. 不保留日期、时间等临时性信息，除非是周期性的重要约定7. 信息应当是从你的角度了解到的用户信息8. 格式为简洁的要点，可用分号分隔不同信息9. 如果约定的时间已经过去，或者用户改变了约定，则更改相关的约定记忆"
                    llm = self._get_llm_client()
                    summary = llm.get_response(
                        message=dialog_to_summarize,
                        user_id=f"summarize_{avatar_name}",
                        system_prompt=system_prompt_summarize
                    )
                    logger.debug(f"角色 {avatar_name} 的记忆总结结果（准备追加）:\n{summary}")
                    retry_sentences = [
                        "好像有些小状况，请再试一次吧～", "信号好像不太稳定呢（皱眉）", "思考被打断了，请再说一次好吗？"
                    ]
                    if summary and summary.strip() in retry_sentences: # 确保 summary 不是 None
                        logger.warning(f"收到需重试的总结结果({avatar_name}): {summary.strip()}, 重试次数 {retries + 1}")
                        retries += 1
                        summary = None
                        continue
                    elif not summary or not summary.strip(): # 处理LLM返回空或无效的情况
                        logger.warning(f"LLM 返回空或无效的总结结果，重试次数 {retries + 1} ({avatar_name})")
                        retries += 1
                        summary = None
                        continue
                    else:
                        break # 成功获取有效总结
                except Exception as e:
                    logger.error(f"记忆总结LLM调用失败 (尝试 {retries + 1}/{max_retries}): {str(e)} ({avatar_name})")
                    retries += 1
                    summary = None

            if summary and summary.strip(): # 再次确认 summary 有效
                sanitized_summary = summary.strip()
                try:
                    with open(long_memory_buffer_path, "a", encoding="utf-8") as f:
                        f.write(f"总结时间: {self._get_timestamp()}\n")
                        f.write(sanitized_summary + "\n\n")
                    logger.info(f"已将新总结追加到长期记忆缓冲区: {avatar_name}")
                    try:
                        with open(short_memory_path, "w", encoding="utf-8") as f:
                            f.truncate(0)
                        logger.info(f"已清空短期记忆（因总结成功）: {avatar_name}")
                    except Exception as e_clear:
                       logger.error(f"清空短期记忆失败，但总结已写入: {str(e_clear)} ({avatar_name})")
                except Exception as e_write:
                    logger.error(f"写入长期记忆缓冲区失败: {str(e_write)} ({avatar_name})，短期记忆未清空")
            else:
                 logger.error(f"无法获取有效的记忆总结，本次不更新长期记忆且不清空短期记忆: {avatar_name}")
        except Exception as e:
            logger.error(f"执行记忆总结 (update_core_memory) 时发生意外错误: {str(e)} ({avatar_name})")

    # ===========================================================
    # || START: 修改后的 get_core_memory 方法                   ||
    # ===========================================================
    def get_core_memory(self, avatar_name: str) -> str:
        """
        !! 功能修改：读取并返回指定角色长期记忆缓冲区 (`long_memory_buffer.txt`) 的【全部内容】!!
        将所有条目拼接成一个单一的字符串。
        """
        long_memory_buffer_path = self._get_core_memory_path(avatar_name) # 实际指向 long_memory_buffer.txt

        if not os.path.exists(long_memory_buffer_path):
            logger.info(f"长期记忆缓冲区不存在 (get_core_memory 读取全部时) for {avatar_name}")
            return ""

        try:
            with open(long_memory_buffer_path, "r", encoding="utf-8") as f:
                full_long_memory_content = f.read().strip() # 读取整个文件并移除首尾空白

            if full_long_memory_content:
                logger.info(f"get_core_memory 返回了角色 {avatar_name} 的全部长期记忆内容，长度: {len(full_long_memory_content)}")
            else:
                logger.debug(f"长期记忆缓冲区 ({avatar_name}) 为空，get_core_memory 返回空字符串。")

            return full_long_memory_content

        except Exception as e:
            logger.error(f"获取全部长期记忆内容失败 ({avatar_name}) (源: get_core_memory): {str(e)}")
            return ""
    # ===========================================================
    # || END: get_core_memory 方法                             ||
    # ===========================================================

    def get_recent_context(self, avatar_name: str, context_size: int = 5) -> List[Dict]:
        """
        获取最近的对话上下文 (从 short_memory.txt)。
        (保留原始函数行为)
        """
        context = []
        short_memory_path = self._get_short_memory_path(avatar_name)
        if not os.path.exists(short_memory_path):
            logger.info(f"短期记忆文件不存在 (get_recent_context): {avatar_name}")
            return []
        try:
            with open(short_memory_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
            conversation_pairs = []
            user_msg_buffer = None
            for i in range(len(lines) - 1, -1, -1):
                line = lines[i].strip()
                if line.startswith("bot:"):
                    bot_msg = line[len("bot:"):].strip()
                    if user_msg_buffer:
                       conversation_pairs.append({"user": user_msg_buffer, "bot": bot_msg})
                       user_msg_buffer = None
                       if len(conversation_pairs) >= context_size:
                           break
                elif line.startswith("用户:"):
                     user_msg_buffer = line[len("用户:"):].strip()
            conversation_pairs.reverse()
            for conv in conversation_pairs:
                context.append({"role": "user", "content": conv["user"]})
                context.append({"role": "assistant", "content": conv["bot"]})
            logger.debug(f"get_recent_context 返回 {len(context)//2} 轮对话 for {avatar_name}")
            return context
        except Exception as e:
            logger.error(f"获取最近上下文失败 (get_recent_context) ({avatar_name}): {str(e)}")
            return []

    # ===========================================================
    # || START: 修改后的 get_relevant_memories 方法             ||
    # ===========================================================
    def get_relevant_memories(self, avatar_name: str, query: str, top_n: int = 3) -> List[str]:
        """
        !! 功能修改：此方法不再执行相关性检索，总是返回空列表。!!
        因为现在的策略是全局读取长期记忆 (通过 get_core_memory)。
        (保留函数签名以防有调用点，但禁用其功能)
        """
        logger.info(f"get_relevant_memories 方法已被调用 (但功能已禁用)。Avatar: {avatar_name}, Query: '{query[:50]}...'")
        return [] # 总是返回一个空列表
    # ===========================================================
    # || END: get_relevant_memories 方法                       ||
    # ===========================================================

# <<< --- MemoryService 类结束 --- >>>
