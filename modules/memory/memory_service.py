import os
# import json # 不再需要 json
import logging
from typing import List, Dict, Optional # 保留 Dict for get_recent_context
from datetime import datetime
from src.services.ai.llm_service import LLMService

logger = logging.getLogger('main')

class MemoryService:
    """
    新版记忆服务模块 - 已修改以模拟旧版长时记忆行为.
    保留了原始方法签名，但功能已大幅调整。
    1. 短期记忆 (short_memory.txt): 保存对话，达到阈值后调用总结。
    2. '核心记忆' (实际为 long_memory_buffer.txt): 存储【累积】的总结，每次总结后追加。
       - get_core_memory() 被修改为返回【最近的总结】。
       - update_core_memory() 被修改为【执行总结并追加到长期记忆】。
    3. 添加了 get_relevant_memories() 方法（旧版核心检索逻辑），但默认未被项目其他地方调用。
    """
    def __init__(self, root_dir: str, api_key: str, base_url: str, model: str, max_token: int, temperature: float):
        self.root_dir = root_dir
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.max_token = max_token
        self.temperature = temperature
        self.llm_client = None
        # self.conversation_count = {} # 移除对话计数器

    # --- 内部帮助方法 ---
    def _get_avatar_memory_dir(self, avatar_name: str) -> str:
        """获取角色记忆目录，如果不存在则创建"""
        avatar_memory_dir = os.path.join(self.root_dir, "data", "avatars", avatar_name, "memory")
        os.makedirs(avatar_memory_dir, exist_ok=True)
        return avatar_memory_dir

    def _get_short_memory_path(self, avatar_name: str) -> str:
        """获取短期记忆文件路径 (TXT格式)"""
        memory_dir = self._get_avatar_memory_dir(avatar_name)
        return os.path.join(memory_dir, "short_memory.txt") # 改为 .txt

    # !! 注意：这里仍然叫 _get_core_memory_path，但实际是指向长期记忆缓冲文件 !!
    # 这是为了兼容名字，但它现在代表了 "长期记忆缓冲区" 文件
    def _get_core_memory_path(self, avatar_name: str) -> str:
        """获取『核心』记忆文件路径 (实际为长期记忆缓冲区 TXT格式)"""
        memory_dir = self._get_avatar_memory_dir(avatar_name)
        # 指向这个文件，用于累积总结
        return os.path.join(memory_dir, "long_memory_buffer.txt")

    def _get_llm_client(self):
        """获取或创建LLM客户端"""
        if not self.llm_client:
            # 稍微增加 max_groups 以容纳可能更长的 prompt (总结/检索)
            self.llm_client = LLMService(
                api_key=self.api_key,
                base_url=self.base_url,
                model=self.model,
                max_token=self.max_token,
                temperature=self.temperature,
                max_groups=10 # 可调整
            )
        return self.llm_client

    def _get_timestamp(self) -> str:
        """获取当前时间戳"""
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def initialize_memory_files(self, avatar_name: str):
        """初始化角色的记忆文件 (TXT格式)，确保文件存在"""
        try:
            memory_dir = self._get_avatar_memory_dir(avatar_name)
            short_memory_path = self._get_short_memory_path(avatar_name)
            # !! 使用 _get_core_memory_path 但它指向 long_memory_buffer !!
            long_memory_buffer_path = self._get_core_memory_path(avatar_name)

            # 初始化短期记忆文件（如果不存在）
            if not os.path.exists(short_memory_path):
                with open(short_memory_path, "w", encoding="utf-8") as f:
                    pass # 创建空文件
                logger.info(f"创建短期记忆文件: {short_memory_path}")

            # 初始化长期记忆缓冲区文件（如果不存在）
            if not os.path.exists(long_memory_buffer_path):
                with open(long_memory_buffer_path, "w", encoding="utf-8") as f:
                     pass # 创建空文件
                logger.info(f"创建长期记忆缓冲区文件（路径名仍用 core 但内容是 long buffer）: {long_memory_buffer_path}")

        except Exception as e:
            logger.error(f"初始化记忆文件失败 ({ avatar_name }): {str(e)}")


    def add_conversation(self, avatar_name: str, user_message: str, bot_reply: str, is_system_message: bool = False):
        """
        添加对话到短期记忆 (.txt)，并检查是否达到总结阈值。
        (保留原始函数签名)
        """
        if is_system_message:
            logger.debug(f"跳过记录系统消息: {user_message[:30]}...")
            return

        try:
            short_memory_path = self._get_short_memory_path(avatar_name)

            # 确保文件存在 (如果不存在，尝试初始化一次)
            if not os.path.exists(short_memory_path):
                logger.warning(f"短期记忆文件丢失，尝试重新初始化: {short_memory_path}")
                self.initialize_memory_files(avatar_name)
                # 再次检查路径确保初始化成功
                if not os.path.exists(short_memory_path):
                    logger.error(f"无法创建短期记忆文件，无法记录对话: {short_memory_path}")
                    return

            # 添加新对话到短期记忆 (使用旧格式)
            with open(short_memory_path, "a", encoding="utf-8") as f:
                f.write(f"用户: {user_message}\n")
                f.write(f"bot: {bot_reply}\n\n")

            # --- 检查是否需要总结 (如果需要则调用 update_core_memory) ---
            line_count = 0
            with open(short_memory_path, "r", encoding="utf-8") as f:
                # 更可靠地计数，防止空行影响
                for line in f:
                    if line.strip(): # 只计算非空行
                         line_count += 1

            # ---> 阈值: 30行 (指非空行，大约等于15轮对话) <--- 你可以调整这个数字
            if line_count >= 30:
                logger.info(f"角色 {avatar_name} 短期记忆达到 {line_count} 行，调用核心记忆更新（实际为总结）...")
                # !! 调用 update_core_memory 来执行总结 !!
                self.update_core_memory(avatar_name)
            #-----------------------------------------------------

        except Exception as e:
            logger.error(f"添加对话到短期记忆失败 ({ avatar_name }): {str(e)}")


    def update_core_memory(self, avatar_name: str):
        """
        !! 功能重写：执行短期记忆的总结，并将结果追加到长期记忆缓冲区，然后清空短期记忆 !!
        (保留原始函数签名)
        """
        logger.info(f"开始为角色 {avatar_name} 执行记忆总结 (替代 update_core_memory)...")
        short_memory_path = self._get_short_memory_path(avatar_name)
        # !! 获取长期记忆缓冲区的文件路径 (虽然方法名是 _get_core_memory_path) !!
        long_memory_buffer_path = self._get_core_memory_path(avatar_name)

        if not os.path.exists(short_memory_path):
            logger.warning(f"短期记忆文件不存在，无法总结: {avatar_name}")
            return

        try:
            # 1. 读取短期记忆内容
            dialog_to_summarize = ""
            with open(short_memory_path, "r", encoding="utf-8") as f:
                dialog_to_summarize = f.read()

            if not dialog_to_summarize.strip():
                 logger.info(f"短期记忆内容为空，跳过总结: {avatar_name}")
                 #可以选择在这里也清空短期记忆，避免下次立即触发总结
                 #with open(short_memory_path, "w", encoding="utf-8") as f: f.truncate(0)
                 return

            # 2. 调用 LLM 进行总结 (使用旧版 Prompt 和重试逻辑)
            max_retries = 3
            retries = 0
            summary = None # 初始化 summary 变量
            while retries < max_retries:
                try:
                    # --- 使用旧版的总结 Prompt ---
                    system_prompt_summarize = "请将以下对话记录总结为最重要的几条长期记忆，总结内容应包含地点，事件，人物（如果对话记录中有的话）用中文简要表述："
                    llm = self._get_llm_client()
                    summary = llm.get_response(
                        message=dialog_to_summarize,
                        user_id=f"summarize_{avatar_name}", # 添加标识
                        system_prompt=system_prompt_summarize
                    )
                    logger.debug(f"角色 {avatar_name} 的记忆总结结果（准备写入长期记忆）:\n{summary}")

                    # --- 旧版的重试检查 ---
                    retry_sentences = [
                        "好像有些小状况，请再试一次吧～",
                        "信号好像不太稳定呢（皱眉）",
                        "思考被打断了，请再说一次好吗？"
                    ]
                    if summary.strip() in retry_sentences:
                        logger.warning(f"收到需重试的总结结果({avatar_name}): {summary.strip()}, 重试次数 {retries + 1}")
                        retries += 1
                        summary = None # 重置 summary，避免错误写入
                        continue
                    else:
                        # 获得有效总结，可以退出重试循环
                        break

                except Exception as e:
                    # API 调用本身失败等异常
                    logger.error(f"记忆总结LLM调用失败 (尝试 {retries + 1}/{max_retries}): {str(e)} ({avatar_name})")
                    retries += 1
                    summary = None # 重置 summary

            # 3. 如果总结成功且有效，追加到长期记忆缓冲 (`long_memory_buffer.txt`)
            if summary and summary.strip() and summary.strip() not in retry_sentences:
                sanitized_summary = summary.strip() # 确保移除首尾空白
                try:
                    with open(long_memory_buffer_path, "a", encoding="utf-8") as f:
                        f.write(f"总结时间: {self._get_timestamp()}\n")
                        f.write(sanitized_summary + "\n\n") # 每个总结后加双换行符分隔
                    logger.info(f"已将新总结追加到长期记忆缓冲区: {avatar_name}")

                    # 4. 总结成功【并写入】长期记忆后，才清空短期记忆
                    try:
                        with open(short_memory_path, "w", encoding="utf-8") as f:
                            f.truncate(0) # 清空文件内容
                        logger.info(f"已清空短期记忆（因总结成功）: {avatar_name}")
                    except Exception as e_clear:
                       logger.error(f"清空短期记忆失败，但总结已写入: {str(e_clear)} ({avatar_name})")

                except Exception as e_write:
                    # 写入失败，则短期记忆不应被清空
                    logger.error(f"写入长期记忆缓冲区失败: {str(e_write)} ({avatar_name})，保留短期记忆供下次尝试")
            else:
                 # 总结失败或LLM返回重试提示达到最大次数
                 logger.error(f"无法获取有效的记忆总结，本次不更新长期记忆且不清空短期记忆: {avatar_name}")


        except Exception as e:
            # 读取短期记忆等步骤的异常
            logger.error(f"执行记忆总结 (update_core_memory 模拟逻辑) 时发生意外错误: {str(e)} ({avatar_name})")


    def get_core_memory(self, avatar_name: str) -> str:
        """
        !! 功能修改：获取长期记忆缓冲区中的【最后一次总结】的内容 !!
        这只是一个兼容性层，不提供完整的长期记忆能力。
        (保留原始函数签名，返回值类型仍为 str)
        """
        # !! 操作的是长期记忆缓冲区 (long_memory_buffer.txt) !!
        long_memory_buffer_path = self._get_core_memory_path(avatar_name)

        if not os.path.exists(long_memory_buffer_path):
            logger.info(f"长期记忆缓冲区不存在 (当调用 get_core_memory 时): {avatar_name}")
            return "" # 返回空字符串，兼容旧接口预期

        try:
            with open(long_memory_buffer_path, "r", encoding="utf-8") as f:
                all_lines = f.readlines()

            last_summary_content = ""
            current_summary_lines = [] # 收集属于一个总结的行
            for line in all_lines:
                stripped_line = line.strip()
                if stripped_line.startswith("总结时间:"):
                    # 遇到新的时间戳，结算上一个总结（如果存在）
                    if current_summary_lines:
                        last_summary_content = "".join(current_summary_lines).strip()
                    current_summary_lines = [] # 开始收集新的总结内容
                elif stripped_line: # 非时间戳的非空行是总结内容
                     current_summary_lines.append(line) # 保留原始的换行

            # 处理文件中最后一个总结块
            if current_summary_lines:
                last_summary_content = "".join(current_summary_lines).strip()

            if last_summary_content:
                logger.debug(f"get_core_memory 返回最后一个总结内容 ({avatar_name}), 长度: {len(last_summary_content)}")
            else:
                logger.debug(f"长期记忆缓冲区存在但未解析出最后总结或为空 ({avatar_name})，get_core_memory 返回空")

            # 返回最后一个找到的总结内容，否则返回空字符串
            return last_summary_content

        except Exception as e:
            logger.error(f"获取最后一次总结失败 (在 get_core_memory 模拟逻辑中): {str(e)} ({avatar_name})")
            return "" # 出错也返回空字符串

    def get_recent_context(self, avatar_name: str, context_size: int = 5) -> List[Dict]:
        """
        获取最近的对话上下文 (从 short_memory.txt)。
        注意：由于short_memory.txt会在总结后清空，这里最多包含触发总结前的对话。
        返回格式为LLM使用的消息列表格式 (List[Dict])
        (保留原始函数签名)
        """
        context = []
        short_memory_path = self._get_short_memory_path(avatar_name)

        if not os.path.exists(short_memory_path):
            logger.info(f"短期记忆文件不存在 (get_recent_context): {avatar_name}")
            return []

        try:
            with open(short_memory_path, "r", encoding="utf-8") as f:
                lines = f.readlines()

            # --- 解析 .txt 对话格式（用户+bot+空行） ---
            conversation_pairs = []
            # 从后往前读取lines，更容易凑齐最后 N 轮
            user_msg_buffer = None
            for i in range(len(lines) - 1, -1, -1):
                line = lines[i].strip()

                if line.startswith("bot:"):
                    bot_msg = line[len("bot:"):].strip()
                    # 期望上一行 (i-1) 是 user 消息
                    if user_msg_buffer: # 如果buffer里有等待的user消息
                       conversation_pairs.append({"user": user_msg_buffer, "bot": bot_msg})
                       user_msg_buffer = None # 清空buffer
                       if len(conversation_pairs) >= context_size:
                           break # 找到足够数量了
                    # else: 可能格式错误，忽略这个 bot 消息

                elif line.startswith("用户:"):
                     # 读到了一条 User 消息，存入buffer等待匹配它的 bot 消息
                     user_msg_buffer = line[len("用户:"):].strip()
                # else: 空行或其他内容，忽略

            # 因为是从后往前添加的，需要反转列表得到正确的对话顺序
            conversation_pairs.reverse()

            # 转换为 LLM 接口要求的消息格式
            for conv in conversation_pairs:
                context.append({"role": "user", "content": conv["user"]})
                context.append({"role": "assistant", "content": conv["bot"]})

            logger.debug(f"get_recent_context 返回 {len(context)} 条消息 ({len(conversation_pairs)} 轮对话) for {avatar_name}")
            return context

        except Exception as e:
            logger.error(f"获取最近上下文失败 (get_recent_context): {str(e)} ({avatar_name})")
            return []


    ## --- 旧版核心检索逻辑 (推荐保留，供未来使用) ---
    ## --- 这个方法目前没有被项目其他地方调用 (除非你手动修改其他文件来调用它) ---
    def get_relevant_memories(self, avatar_name: str, query: str, top_n: int = 3) -> List[str]:
        """
        根据查询 从长期记忆缓冲区 获取最相关的N条【总结】。
        这是旧版 (v1.3.7) 的核心检索逻辑。
        """
        # !! 操作的是长期记忆缓冲区 (long_memory_buffer.txt) !!
        long_memory_buffer_path = self._get_core_memory_path(avatar_name)

        if not os.path.exists(long_memory_buffer_path):
            logger.info(f"长期记忆缓冲区不存在，无法检索相关记忆: {avatar_name}")
            return []

        max_retries = 3
        for retry_count in range(max_retries):
            try:
                # 1. 读取并解析长期记忆缓冲区中的所有总结条目
                with open(long_memory_buffer_path, "r", encoding="utf-8") as f:
                    all_lines = f.readlines()

                memories = [] # 存储解析出的每个总结条目 (字符串)
                current_summary_lines = []
                for line in all_lines:
                    stripped_line = line.strip()
                    if stripped_line.startswith("总结时间:"):
                        if current_summary_lines: # 结算上一个条目
                            memories.append("".join(current_summary_lines).strip())
                        current_summary_lines = [] # 开始收集新的
                    elif stripped_line: # 当前条目的内容行
                        current_summary_lines.append(line) # 保留原始换行

                if current_summary_lines: # 处理文件末尾的最后一个条目
                    memories.append("".join(current_summary_lines).strip())

                if not memories:
                    logger.debug(f"长期记忆缓冲区为空或未能解析出总结条目用于检索 ({avatar_name})")
                    return []

                # 2. 准备检索所需的数据和Prompt
                # 限制最多检索最近的N条总结 (例如20条) 来控制API成本和延迟
                memories_to_search_from = memories[-20:]
                # 用分隔符让LLM更好地区分条目
                memories_text = "\n\n---\n\n".join(memories_to_search_from)

                system_prompt_retrieve = f"请从以下的多条历史记忆总结中，找出与用户当前的问话或陈述 '{query}' 最相关的条目。请按相关性排序，返回最多 {top_n} 条。只返回找到的记忆条目的原始内容，不要添加任何额外的解释、介绍文字、列表编号或标记。"

                # 3. 调用LLM进行检索
                llm = self._get_llm_client()
                response = llm.get_response(
                    message=memories_text, # 将待检索的记忆作为message传给LLM
                    user_id=f"retrieve_{avatar_name}",
                    system_prompt=system_prompt_retrieve
                )

                # 4. 处理LLM的响应 (包括重试逻辑)
                retry_sentences = [
                    "...", #根据你的LLM可能返回的错误/重试提示补充
                    "好像有些小状况，请再试一次吧～",
                    "信号好像不太稳定呢（皱眉）",
                    "思考被打断了，请再说一次好吗？"
                ]
                # 做更鲁棒的检查，比如检查是否包含特定错误信息
                needs_retry = False
                if response.strip() in retry_sentences:
                     needs_retry = True
                # 你可以添加更多检查，例如 if not response.strip() or response.startswith("抱歉") etc.

                if needs_retry:
                     if retry_count < max_retries - 1:
                         logger.warning(f"相关记忆检索重试({avatar_name}): LLM 返回需重试内容 (尝试 {retry_count + 1})")
                         continue # 进入下一次重试
                     else:
                         logger.error(f"相关记忆检索达到最大重试次数，仍收到需重试响应 ({avatar_name})")
                         return [] # 返回空列表表示失败
                else:
                    # 假设成功返回相关记忆，每行一条
                    relevant_memories = [line.strip() for line in response.split("\n") if line.strip()]
                    logger.info(f"为查询 '{query[:50]}...' 找到 {len(relevant_memories)} 条相关记忆 ({avatar_name})")
                    return relevant_memories # 返回结果

            except Exception as e:
                 logger.error(f"检索相关记忆时发生异常 ({avatar_name}) (尝试 {retry_count + 1}/{max_retries}): {str(e)}")
                 if retry_count < max_retries - 1:
                     continue
                 else:
                      logger.error(f"尝试检索记忆达到最大次数或发生致命错误 ({avatar_name}): {str(e)}")
                      return [] # 最终失败

        return [] # 所有重试都失败后返回空列表

# <<< --- MemoryService 类结束 --- >>>
