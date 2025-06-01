import os
import json
import logging
from typing import List, Dict, Optional
from datetime import datetime
from src.services.ai.llm_service import LLMService

# 获取日志记录器
logger = logging.getLogger('memory')

class MemoryService:
    """
    记忆服务模块 - 修改为模拟旧版长时记忆行为（追加总结），并保留新版滚动短期记忆。
    总结提示词从 'data/base/memory.md' 加载。
    长期记忆的每一条总结前会附加时间戳。
    """
    def __init__(self, root_dir: str, api_key: str, base_url: str, model: str, max_token: int, temperature: float, max_groups: int = 10):
        self.root_dir = root_dir
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.max_token = max_token
        self.temperature = temperature
        self.max_groups = max_groups
        self.llm_client = None
        self.conversation_count = {}

    def _get_llm_client(self):
        """获取或创建LLM客户端"""
        if not self.llm_client:
            self.llm_client = LLMService(
                api_key=self.api_key,
                base_url=self.base_url,
                model=self.model,
                max_token=self.max_token, # 用于总结的 token 可能需要调整
                temperature=self.temperature, # 用于总结的 temperature 可能需要调整
                max_groups=self.max_groups
            )
            logger.info(f"创建LLM客户端，API: {self.base_url}, Model: {self.model}, MaxGroups: {self.max_groups}")
        return self.llm_client

    def _get_avatar_memory_dir(self, avatar_name: str, user_id: str) -> str:
        """获取角色与用户组合的记忆目录，如果不存在则创建"""
        avatar_memory_dir = os.path.join(self.root_dir, "data", "avatars", avatar_name, "memory", user_id)
        os.makedirs(avatar_memory_dir, exist_ok=True)
        return avatar_memory_dir

    def _get_short_memory_path(self, avatar_name: str, user_id: str) -> str:
        """获取短期记忆文件路径 (JSON格式)"""
        memory_dir = self._get_avatar_memory_dir(avatar_name, user_id)
        return os.path.join(memory_dir, "short_memory.json")

    def _get_core_memory_path(self, avatar_name: str, user_id: str) -> str:
        """
        获取『核心』记忆文件路径 (实际为长期记忆缓冲区 long_memory_buffer.txt)。
        """
        memory_dir = self._get_avatar_memory_dir(avatar_name, user_id)
        return os.path.join(memory_dir, "long_memory_buffer.txt")

    def _get_timestamp(self) -> str:
        """获取当前时间戳"""
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def initialize_memory_files(self, avatar_name: str, user_id: str):
        """初始化角色的记忆文件，确保文件存在"""
        try:
            memory_dir = self._get_avatar_memory_dir(avatar_name, user_id)
            short_memory_path = self._get_short_memory_path(avatar_name, user_id)
            long_term_memory_path = self._get_core_memory_path(avatar_name, user_id)

            if not os.path.exists(short_memory_path):
                with open(short_memory_path, "w", encoding="utf-8") as f:
                    json.dump([], f, ensure_ascii=False, indent=2)
                logger.info(f"创建短期记忆文件 (JSON): {short_memory_path}")

            if not os.path.exists(long_term_memory_path):
                with open(long_term_memory_path, "w", encoding="utf-8"):
                    pass # 创建空文件
                logger.info(f"创建长期记忆缓冲区文件 (TXT): {long_term_memory_path}")

        except Exception as e:
            logger.error(f"初始化记忆文件失败 ({avatar_name}/{user_id}): {str(e)}")

    def add_conversation(self, avatar_name: str, user_message: str, bot_reply: str, user_id: str, is_system_message: bool = False):
        """
        添加对话到短期记忆 (short_memory.json)，并检查是否达到总结阈值以调用 update_core_memory。
        短期记忆是滚动的，只保留最近N条。 测试配置：1轮对话触发总结。
        """
        conversation_key = f"{avatar_name}_{user_id}"
        if conversation_key not in self.conversation_count:
            self.conversation_count[conversation_key] = 0

        if is_system_message or (isinstance(bot_reply, str) and bot_reply.startswith("Error:")):
            logger.debug(f"跳过记录系统或错误消息: User: {user_message[:30]}..., Bot: {str(bot_reply)[:30]}")
            return

        try:
            short_memory_path = self._get_short_memory_path(avatar_name, user_id)
            if not os.path.exists(short_memory_path) :
                 logger.warning(f"短期记忆文件丢失，尝试重新初始化: {short_memory_path} for {avatar_name}/{user_id}")
                 self.initialize_memory_files(avatar_name,user_id)
                 if not os.path.exists(short_memory_path):
                     logger.error(f"无法创建短期记忆文件，无法记录对话: {short_memory_path}")
                     return

            logger.debug(f"保存对话到用户短期记忆: 角色={avatar_name}, 用户ID={user_id}, 路径: {short_memory_path}")

            short_memory = []
            if os.path.exists(short_memory_path):
                try:
                    with open(short_memory_path, "r", encoding="utf-8") as f:
                        content = f.read()
                        if content.strip():
                            short_memory = json.loads(content)
                        else:
                            short_memory = []
                except json.JSONDecodeError:
                    logger.warning(f"短期记忆文件 ({short_memory_path}) 损坏或为空，重置为空列表并继续。")
                    short_memory = []
            
            new_conversation = {
                "timestamp": self._get_timestamp(),
                "user": user_message,
                "bot": bot_reply
            }
            short_memory.append(new_conversation)
            
            if len(short_memory) > 50: # 保持滚动短期记忆大小为50
                short_memory = short_memory[-50:]
            
            with open(short_memory_path, "w", encoding="utf-8") as f:
                json.dump(short_memory, f, ensure_ascii=False, indent=2)
            
            self.conversation_count[conversation_key] += 1
            current_count = self.conversation_count[conversation_key]
            
            # -------- 测试配置：1轮对话触发总结 --------
            summary_trigger_threshold = 30 
            context_size_for_summary = 30 
            # ------------------------------------------
            
            logger.debug(f"当前对话计数 (总结阈值为{summary_trigger_threshold}): {current_count}/{summary_trigger_threshold} (角色={avatar_name}, 用户ID={user_id})")
            
            if self.conversation_count[conversation_key] >= summary_trigger_threshold:
                logger.info(f"角色 {avatar_name} 用户 {user_id} 达到 {summary_trigger_threshold} 轮对话，调用 update_core_memory (追加总结到长期记忆)...")
                context_for_summary = self.get_recent_context(avatar_name, user_id, context_size=context_size_for_summary)

                self.update_core_memory(avatar_name, user_id, context_for_summary)
                self.conversation_count[conversation_key] = 0 # 重置计数器
                
        except Exception as e:
            logger.error(f"添加对话到短期记忆失败 ({avatar_name}/{user_id}): {str(e)}")

    def _build_memory_prompt(self, filepath: str, relative_to_root: bool = True) -> str:
        """
        从指定路径读取 md 文件作为提示词。
        Args:
            filepath: md 文件的相对路径 (如 'data/base/memory.md')。
            relative_to_root: 如果为True, filepath是相对于self.root_dir的路径。否则是绝对路径或相对于当前工作目录。
        Returns:
            文件内容的字符串，或在出错时返回空字符串。
        """
        if relative_to_root:
            # 确保 self.root_dir 存在
            if not hasattr(self, 'root_dir') or not self.root_dir:
                logger.error("MemoryService 的 root_dir 未设置，无法解析相对路径。")
                # 可以尝试使用脚本的当前工作目录作为备选，但这可能不准确
                # resolved_filepath = os.path.join(os.getcwd(), filepath) 
                return "" # 或者抛出异常
            resolved_filepath = os.path.join(self.root_dir, filepath)
        else:
            resolved_filepath = filepath
            
        try:
            with open(resolved_filepath, 'r', encoding='utf-8') as f:
                prompt_content = f.read()
            # 使用 debug级别，因为这个操作会比较频繁
            logger.debug(f"成功从 {resolved_filepath} 加载记忆提示词。")
            return prompt_content
        except FileNotFoundError:
            logger.error(f"记忆提示词模板 {resolved_filepath} (原始路径: {filepath}) 未找到。")
            return ""
        except Exception as e:
            logger.error(f"读取记忆提示词模板 {resolved_filepath} (原始路径: {filepath}) 时出错: {e}")
            return ""

    def update_core_memory(self, avatar_name: str, user_id: str, context_to_summarize: List[Dict]):
        """
        执行短期对话的总结，并将结果【追加】到长期记忆缓冲区 (TXT)，短期记忆 (JSON) 不清空。
        总结的每条记忆前会附加时间戳。
        """
        logger.info(f"开始为角色 {avatar_name} 用户 {user_id} 执行记忆总结并追加到长期记忆...")
        
        long_term_memory_path = self._get_core_memory_path(avatar_name, user_id)

        if not context_to_summarize:
            logger.info(f"用于总结的上下文为空，跳过总结: {avatar_name}/{user_id}")
            return

        dialog_to_summarize_str = ""
        for entry in context_to_summarize:
            role = entry.get("role", "").lower()
            content = entry.get("content", "")
            if role == "user":
                dialog_to_summarize_str += f"用户: {content}\n"
            elif role == "assistant" or role == "bot":
                dialog_to_summarize_str += f"bot: {content}\n"
        dialog_to_summarize_str += "\n" 

        if not dialog_to_summarize_str.strip():
            logger.info(f"转换后的对话内容为空，跳过总结: {avatar_name}/{user_id}")
            return

        summarization_prompt = self._build_memory_prompt('data/base/memory.md')
        if not summarization_prompt:
            logger.error(f"无法加载总结提示词 (data/base/memory.md)，跳过为 {avatar_name}/{user_id} 更新核心记忆。")
            return

        max_retries = 3
        retries = 0
        summary_text_from_llm = None
        
        llm_message_for_summary = dialog_to_summarize_str 

        while retries < max_retries:
            try:
                llm = self._get_llm_client()
                summary_text_from_llm = llm.get_response(
                    message=llm_message_for_summary,
                    user_id=f"summarize_{avatar_name}_{user_id}",
                    system_prompt=summarization_prompt,
                    core_memory=None, 
                    previous_context=None
                )
                logger.debug(f"角色 {avatar_name} 用户 {user_id} 的原始记忆总结结果:\n{summary_text_from_llm}")
                
                retry_sentences = [ # 这些可以根据你的LLM的特定回复进行调整
                    "好像有些小状况，请再试一次吧～", "信号好像不太稳定呢（皱眉）", "思考被打断了，请再说一次好吗？"
                ]
                if summary_text_from_llm and summary_text_from_llm.strip() in retry_sentences:
                    logger.warning(f"收到需重试的总结结果({avatar_name}/{user_id}): {summary_text_from_llm.strip()}, 重试次数 {retries + 1}")
                    retries += 1
                    summary_text_from_llm = None
                    continue
                elif not summary_text_from_llm or not summary_text_from_llm.strip():
                    logger.warning(f"LLM 返回空或无效的总结结果 ({avatar_name}/{user_id})，重试次数 {retries + 1}")
                    retries += 1
                    summary_text_from_llm = None
                    continue
                else:
                    break 
            except Exception as e:
                logger.error(f"记忆总结LLM调用失败 ({avatar_name}/{user_id}, 尝试 {retries + 1}/{max_retries}): {str(e)}")
                retries += 1
                summary_text_from_llm = None

        if summary_text_from_llm and summary_text_from_llm.strip():
            sanitized_summary_block = summary_text_from_llm.strip()
            item_timestamp_str = self._get_timestamp()
            
            summary_lines_with_timestamp = []
            for line in sanitized_summary_block.splitlines():
                trimmed_line = line.strip()
                if trimmed_line:
                    summary_lines_with_timestamp.append(f"[{item_timestamp_str}] {trimmed_line}")
            
            final_summary_to_write = "\n".join(summary_lines_with_timestamp)

            if not final_summary_to_write:
                logger.warning(f"处理后的总结内容为空，虽然LLM返回了内容 ({avatar_name}/{user_id})。原始: {sanitized_summary_block}")
                return

            try:
                os.makedirs(os.path.dirname(long_term_memory_path), exist_ok=True)
                with open(long_term_memory_path, "a", encoding="utf-8") as f:
                    f.write(f"总结时间: {item_timestamp_str}\n") 
                    f.write(final_summary_to_write + "\n\n")
                logger.info(f"已将新总结（每条带时间戳）追加到长期记忆缓冲区: {avatar_name}/{user_id}")
                logger.debug(f"追加到 {long_term_memory_path} 的内容:\n总结时间: {item_timestamp_str}\n{final_summary_to_write}\n")
            except Exception as e_write:
                logger.error(f"写入长期记忆缓冲区失败: {str(e_write)} ({avatar_name}/{user_id})。")
        else:
            logger.error(f"无法获取有效的记忆总结，本次不更新长期记忆: {avatar_name}/{user_id}")
            
    def get_core_memory(self, avatar_name: str, user_id: str) -> str:
        """
        读取并返回指定角色与用户组合的长期记忆缓冲区 (`long_memory_buffer.txt`) 的【全部内容】。
        """
        long_term_memory_path = self._get_core_memory_path(avatar_name, user_id)

        if not os.path.exists(long_term_memory_path):
            logger.debug(f"核心记忆（长期记忆）文件不存在: {long_term_memory_path} for {avatar_name}/{user_id}")
            return ""
        try:
            with open(long_term_memory_path, "r", encoding="utf-8") as f:
                full_long_memory_content = f.read().strip()
            logger.debug(f"get_core_memory 返回 {avatar_name}/{user_id} 长期记忆, 长度: {len(full_long_memory_content)}")
            return full_long_memory_content
        except Exception as e:
            logger.error(f"获取全部长期记忆内容失败 for {avatar_name}/{user_id} (源: get_core_memory): {str(e)}")
            return ""

    def get_recent_context(self, avatar_name: str, user_id: str, context_size: Optional[int] = None) -> List[Dict]:
        """
        获取最近的对话上下文 (从 short_memory.json)。
        如果 context_size 未指定，则使用 LLM 配置的 max_groups。
        """
        try:
            num_dialogues_to_fetch = context_size
            if num_dialogues_to_fetch is None:
                llm_client = self._get_llm_client()
                num_dialogues_to_fetch = llm_client.config.get("max_groups", self.max_groups) if hasattr(llm_client, 'config') else self.max_groups

            logger.debug(f"为 {avatar_name}/{user_id} 获取最近上下文，目标轮数: {num_dialogues_to_fetch}")
            
            short_memory_path = self._get_short_memory_path(avatar_name, user_id)
            
            if not os.path.exists(short_memory_path):
                logger.debug(f"短期记忆文件不存在 (get_recent_context): {short_memory_path}")
                return []
            
            short_memory_entries = []
            with open(short_memory_path, "r", encoding="utf-8") as f:
                content = f.read()
                if content.strip():
                    try:
                        short_memory_entries = json.loads(content)
                    except json.JSONDecodeError:
                        logger.warning(f"短期记忆文件损坏 ({short_memory_path})，返回空上下文。")
                        return []
                else: # empty file
                    return []

            context = []
            # 获取最后 N 轮对话
            for conv in short_memory_entries[-num_dialogues_to_fetch:]: 
                context.append({"role": "user", "content": conv["user"]})
                context.append({"role": "assistant", "content": conv["bot"]})
            
            logger.debug(f"已加载 {len(context)//2} 轮对话作为上下文 for {avatar_name}/{user_id}")
            return context
            
        except Exception as e:
            logger.error(f"获取最近上下文失败 (get_recent_context) ({avatar_name}/{user_id}): {str(e)}")
            return []

    def get_relevant_memories(self, avatar_name: str, user_id: str, query: str, top_n: int = 3) -> List[str]:
        """
        !! 功能修改：此方法不再执行相关性检索，总是返回空列表 (模拟旧版行为)。!!
        """
        logger.info(f"get_relevant_memories 方法已被调用 (但功能已禁用)。 Avatar: {avatar_name}, User: {user_id}")
        return []

# <<< --- MemoryService 类结束 --- >>>
