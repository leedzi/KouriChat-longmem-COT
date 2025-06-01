"""
语音处理模块
负责处理语音相关功能，包括:
- 语音请求识别
- TTS语音生成
- 语音文件管理
- 清理临时文件
"""

import os
import logging
import requests
from datetime import datetime
from typing import Optional
from fish_audio_sdk import Session, TTSRequest

# 修改logger获取方式，确保与main模块一致
logger = logging.getLogger('main')

class VoiceHandler:
    def __init__(self, root_dir, tts_api_key):
        self.root_dir = root_dir
        self.voice_dir = os.path.join(root_dir, "data", "voices")
        self.tts_api_key = tts_api_key
        
        # 确保语音目录存在
        os.makedirs(self.voice_dir, exist_ok=True)

    def is_voice_request(self, text: str) -> bool:
        """判断是否为语音请求"""
        voice_keywords = ["语音"]
        return any(keyword in text for keyword in voice_keywords)

    def generate_voice(self, text: str) -> Optional[str]:
        """调用TTS API生成语音"""
        try:
            # 确保语音目录存在
            if not os.path.exists(self.voice_dir):
                os.makedirs(self.voice_dir)
                
            # 生成唯一的文件名
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            voice_path = os.path.join(self.voice_dir, f"voice_{timestamp}.mp3")
            
            # 调用TTS API
            with open(voice_path, "wb") as f:
                for chunk in Session(self.tts_api_key).tts(TTSRequest(
                    reference_id="7f92f8afb8ec43bf81429cc1c9199cb1",
                    text=text
                )):
                    f.write(chunk)
                
        except Exception as e:
            logger.error(f"语音生成失败: {str(e)}")
            return None
        
        return voice_path

    def cleanup_voice_dir(self):
        """清理语音目录中的旧文件"""
        try:
            if os.path.exists(self.voice_dir):
                for file in os.listdir(self.voice_dir):
                    file_path = os.path.join(self.voice_dir, file)
                    try:
                        if os.path.isfile(file_path):
                            os.remove(file_path)
                            logger.info(f"清理旧语音文件: {file_path}")
                    except Exception as e:
                        logger.error(f"清理语音文件失败 {file_path}: {str(e)}")
        except Exception as e:
            logger.error(f"清理语音目录失败: {str(e)}") 