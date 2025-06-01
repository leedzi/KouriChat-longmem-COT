"""
自动更新模块
提供程序自动更新功能，包括:
- 更新包下载
- 文件更新
- 备份和恢复
- 更新回滚
"""

import os
import requests
import zipfile
import shutil
import json
import logging
import datetime
import fnmatch
import hashlib
from typing import Optional, Dict, Any, List

logger = logging.getLogger(__name__)


class Updater:

    # 云端公告、版本信息、模型列表和更新包 URLs
    CLOUD_ANNOUNCEMENT_URL = "https://git.kourichat.com/KouriChat-Main/cloud-delivery-repo/raw/branch/main/config/announcement.json"
    CLOUD_VERSION_URL = "https://git.kourichat.com/KouriChat-Main/cloud-delivery-repo/raw/branch/main/config/version.json"
    CLOUD_MODELS_URL = "https://git.kourichat.com/KouriChat-Main/cloud-delivery-repo/raw/branch/main/config/models.json"
    CLOUD_RELEASE_URL = "https://git.kourichat.com/KouriChat-Main/cloud-delivery-repo/raw/branch/main/releases/releases.zip"

    # 默认需要跳过的文件和文件夹（不会被更新）
    DEFAULT_IGNORE_PATTERNS = [
        # 用户数据目录，排除base.md
        "data/**",
        "!data/base/base.md",  # 允许更新base.md文件
        "**/data/database/**",
        "**/data/images/**",
        "**/data/voices/**",
        "**/data/avatars/**",

        # 日志和临时文件
        "logs/**",
        "tmp/**",
        "screenshot/**",
        "wxauto文件/**",
        "__pycache__/**",
        "**/*.pyc",
        "**/*.pyo",
        "**/*.pyd",
        "**/*.so",
        "**/*.dll",

        # 配置和环境文件
        ".env",
        "**/*.env",
        "src/config/config.json",

        # 版本控制
        ".git/**",
        ".gitignore",
        ".gitattributes",

        # 更新相关临时目录
        "temp_update/**",
        "backup/**",

        # 其他不需要更新的目录
        ".venv/**",
        "venv/**"
    ]



    def __init__(self):
        self.root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.temp_dir = os.path.join(self.root_dir, 'temp_update')

        # 云端同步的配置文件路径
        self.cloud_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cloud')
        os.makedirs(self.cloud_dir, exist_ok=True)  # 确保目录存在

        self.version_file = os.path.join(self.cloud_dir, 'version.json')
        self.announcement_file = os.path.join(self.cloud_dir, 'announcement.json')
        self.models_file = os.path.join(self.cloud_dir, 'models.json')


        self.ignore_patterns = self._load_ignore_patterns()  # 加载忽略模式

    def _load_ignore_patterns(self) -> List[str]:
        """加载忽略模式，优先使用.updateignore文件，如果存在的话"""
        # 首先使用内置的默认规则
        patterns = self.DEFAULT_IGNORE_PATTERNS.copy()

        # 如果存在.updateignore文件，读取其中的模式（可选步骤）
        ignore_file = os.path.join(self.root_dir, '.updateignore')
        if os.path.exists(ignore_file):
            try:
                with open(ignore_file, 'r', encoding='utf-8') as f:
                    file_patterns = []
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith('#'):
                            file_patterns.append(line)

                    if file_patterns:
                        # 如果文件中有规则，则替换默认规则
                        patterns = file_patterns
                        logger.info(f"使用.updateignore文件中的规则，共{len(patterns)}条")
                    else:
                        logger.info("发现.updateignore文件但没有有效规则，使用默认规则")
            except Exception as e:
                logger.error(f"读取.updateignore文件失败: {str(e)}")
                logger.info("使用默认忽略规则")
        else:
            logger.info("使用内置默认忽略规则")

        return patterns



    def get_current_version(self) -> str:
        """获取当前版本号"""
        try:
            if os.path.exists(self.version_file):
                with open(self.version_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    return data.get('version', '0.0.0')
        except Exception as e:
            logger.error(f"读取版本文件失败: {str(e)}")
        return '0.0.0'

    def get_version_identifier(self) -> str:
        """获取版本标识符，用于 User-Agent"""
        try:
            if os.path.exists(self.version_file):
                with open(self.version_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    return data.get('version_identifier', 'KouriChat/1.0.0')
        except Exception as e:
            logger.error(f"读取版本标识符失败: {str(e)}")
        return 'KouriChat/1.0.0'

    def format_version_info(self, current_version: str, update_info: dict = None) -> str:
        """格式化版本信息输出"""
        output = (
            "\n" + "="*50 + "\n"
            f"当前版本: {current_version}\n"
        )

        if update_info:
            output += (
                f"最新版本: {update_info['version']}\n\n"
                f"更新时间: {update_info.get('last_update', '未知')}\n\n"
                "更新内容:\n"
                f"  {update_info.get('description', '无更新说明')}\n"
                + "="*50 + "\n\n"
                "是否现在更新? (y/n): "  # 添加更新提示
            )
        else:
            output += (
                "检查结果: 当前已是最新版本\n"
                + "="*50 + "\n"
            )

        return output

    def format_update_progress(self, step: str, success: bool = True, details: str = "") -> str:
        """格式化更新进度输出"""
        status = "✓" if success else "✗"
        output = f"[{status}] {step}"
        if details:
            output += f": {details}"
        return output

    def fetch_cloud_announcement(self) -> dict:
        """从云端获取公告信息"""
        current_version = self.get_current_version()
        version_identifier = self.get_version_identifier()
        headers = {
            'User-Agent': version_identifier,
            'X-KouriChat-Version': current_version
        }

        try:
            logger.info(f"正在从 {self.CLOUD_ANNOUNCEMENT_URL} 获取公告信息...")
            response = requests.get(
                self.CLOUD_ANNOUNCEMENT_URL,
                headers=headers,
                timeout=10,
                verify=True
            )
            response.raise_for_status()

            cloud_announcement = response.json()
            logger.info("从云端成功获取公告信息")

            # 将云端公告信息保存到本地
            os.makedirs(os.path.dirname(self.announcement_file), exist_ok=True)
            with open(self.announcement_file, 'w', encoding='utf-8') as f:
                json.dump(cloud_announcement, f, ensure_ascii=False, indent=4)
            logger.info(f"已将云端公告信息保存到: {self.announcement_file}")

            return cloud_announcement
        except Exception as e:
            logger.error(f"从云端获取公告信息失败: {str(e)}")
            return None

    def fetch_cloud_version(self) -> dict:
        """从云端获取版本信息"""
        current_version = self.get_current_version()
        version_identifier = self.get_version_identifier()
        headers = {
            'User-Agent': version_identifier,
            'X-KouriChat-Version': current_version
        }

        try:
            logger.info(f"正在从 {self.CLOUD_VERSION_URL} 获取版本信息...")
            response = requests.get(
                self.CLOUD_VERSION_URL,
                headers=headers,
                timeout=10,
                verify=True
            )
            response.raise_for_status()

            cloud_version = response.json()
            logger.info("从云端成功获取版本信息")

            # 将云端版本信息保存到本地
            os.makedirs(os.path.dirname(self.version_file), exist_ok=True)
            with open(self.version_file, 'w', encoding='utf-8') as f:
                json.dump(cloud_version, f, ensure_ascii=False, indent=4)
            logger.info(f"已将云端版本信息保存到: {self.version_file}")

            return cloud_version
        except Exception as e:
            logger.error(f"从云端获取版本信息失败: {str(e)}")
            return None

    def fetch_cloud_models(self) -> dict:
        """从云端获取模型列表"""
        current_version = self.get_current_version()
        version_identifier = self.get_version_identifier()
        headers = {
            'User-Agent': version_identifier,
            'X-KouriChat-Version': current_version
        }

        try:
            logger.info(f"正在从 {self.CLOUD_MODELS_URL} 获取模型列表...")
            response = requests.get(
                self.CLOUD_MODELS_URL,
                headers=headers,
                timeout=10,
                verify=True
            )
            response.raise_for_status()

            cloud_models = response.json()
            logger.info("从云端成功获取模型列表")

            # 将云端模型列表保存到本地
            os.makedirs(os.path.dirname(self.models_file), exist_ok=True)
            with open(self.models_file, 'w', encoding='utf-8') as f:
                json.dump(cloud_models, f, ensure_ascii=False, indent=4)
            logger.info(f"已将云端模型列表保存到: {self.models_file}")

            return cloud_models
        except Exception as e:
            logger.error(f"从云端获取模型列表失败: {str(e)}")
            return None

    def check_for_updates(self) -> dict:
        """检查更新"""
        # 从云端获取版本信息
        remote_version_info = self.fetch_cloud_version()

        # 如果云端获取失败
        if not remote_version_info:
            logger.error("从云端获取版本信息失败")
            return {
                'has_update': False,
                'error': "检查更新失败：无法连接到更新服务器",
                'output': "检查更新失败：无法连接到更新服务器"
            }

        current_version = self.get_current_version()
        latest_version = remote_version_info.get('version', '0.0.0')

        # 版本比较逻辑
        def parse_version(version: str) -> tuple:
            # 移除版本号中的 'v' 前缀（如果有）
            version = version.lower().strip('v')
            try:
                # 尝试将版本号分割为数字列表
                parts = version.split('.')
                # 确保至少有三个部分（主版本号.次版本号.修订号）
                while len(parts) < 3:
                    parts.append('0')
                # 转换为整数元组
                return tuple(map(int, parts[:3]))
            except (ValueError, AttributeError):
                # 如果是 commit hash 或无法解析的版本号，返回 (0, 0, 0)
                return (0, 0, 0)

        current_ver_tuple = parse_version(current_version)
        latest_ver_tuple = parse_version(latest_version)

        # 只有当最新版本大于当前版本时才返回更新信息
        if latest_ver_tuple > current_ver_tuple:
            # 使用固定的云端更新包URL
            download_url = self.CLOUD_RELEASE_URL

            return {
                'has_update': True,
                'version': latest_version,
                'download_url': download_url,
                'description': remote_version_info.get('description', '无更新说明'),
                'last_update': remote_version_info.get('last_update', ''),
                'output': self.format_version_info(current_version, remote_version_info)
            }

        return {
            'has_update': False,
            'output': self.format_version_info(current_version)
        }

    def download_update(self, download_url: str = None) -> bool:
        """从固定的云端 URL 下载更新包"""
        try:
            # 使用固定的云端更新包 URL
            download_url = self.CLOUD_RELEASE_URL
            logger.info(f"正在从 {download_url} 下载更新...")

            current_version = self.get_current_version()
            version_identifier = self.get_version_identifier()
            headers = {
                'User-Agent': version_identifier,
                'X-KouriChat-Version': current_version
            }

            response = requests.get(
                download_url,
                headers=headers,
                timeout=30,
                stream=True
            )
            response.raise_for_status()

            os.makedirs(self.temp_dir, exist_ok=True)
            zip_path = os.path.join(self.temp_dir, 'update.zip')

            with open(zip_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            logger.info("更新包下载成功")
            return True

        except Exception as e:
            logger.error(f"下载更新包失败: {str(e)}")
            return False

    def should_skip_file(self, file_path: str) -> bool:
        """检查是否应该跳过更新某个文件
        file_path: 文件的相对路径（相对于仓库根目录）
        """
        # 规范化路径，使用正斜杠
        file_path = file_path.replace('\\', '/')

        # 特殊处理 data/base/base.md
        if file_path == "data/base/base.md":
            return False

        # 使用fnmatch检查是否匹配任何忽略模式
        for pattern in self.ignore_patterns:
            if pattern.startswith('!'):  # 排除模式
                continue
            if fnmatch.fnmatch(file_path, pattern):
                logger.debug(f"跳过文件: {file_path} (匹配模式: {pattern})")
                return True

        return False

    def backup_current_version(self) -> bool:
        """备份当前版本"""
        try:
            backup_dir = os.path.join(self.root_dir, 'backup')
            if os.path.exists(backup_dir):
                shutil.rmtree(backup_dir)

            # 创建备份目录
            os.makedirs(backup_dir, exist_ok=True)

            # 复制所有不应该被忽略的文件
            for root, dirs, files in os.walk(self.root_dir):
                # 跳过backup和temp_update目录
                if 'backup' in root or 'temp_update' in root:
                    continue

                # 获取相对路径
                rel_path = os.path.relpath(root, self.root_dir)
                if rel_path == '.':
                    rel_path = ''

                # 处理文件
                for file in files:
                    file_rel_path = os.path.join(rel_path, file).replace('\\', '/')
                    if not self.should_skip_file(file_rel_path):
                        src_file = os.path.join(root, file)
                        dst_file = os.path.join(backup_dir, file_rel_path)
                        os.makedirs(os.path.dirname(dst_file), exist_ok=True)
                        shutil.copy2(src_file, dst_file)

            return True
        except Exception as e:
            logger.error(f"备份失败: {str(e)}")
            return False

    def restore_from_backup(self) -> bool:
        """从备份恢复"""
        try:
            backup_dir = os.path.join(self.root_dir, 'backup')
            if not os.path.exists(backup_dir):
                logger.error("备份目录不存在")
                return False

            # 恢复所有备份的文件
            for root, dirs, files in os.walk(backup_dir):
                # 获取相对路径
                rel_path = os.path.relpath(root, backup_dir)
                if rel_path == '.':
                    rel_path = ''

                # 处理文件
                for file in files:
                    file_rel_path = os.path.join(rel_path, file).replace('\\', '/')
                    if not self.should_skip_file(file_rel_path):
                        src_file = os.path.join(root, file)
                        dst_file = os.path.join(self.root_dir, file_rel_path)
                        os.makedirs(os.path.dirname(dst_file), exist_ok=True)
                        shutil.copy2(src_file, dst_file)

            return True
        except Exception as e:
            logger.error(f"恢复失败: {str(e)}")
            return False

    def apply_update(self) -> bool:
        """应用更新"""
        try:
            # 解压更新包
            zip_path = os.path.join(self.temp_dir, 'update.zip')
            extract_dir = os.path.join(self.temp_dir, 'extracted')

            if os.path.exists(extract_dir):
                shutil.rmtree(extract_dir)
            os.makedirs(extract_dir, exist_ok=True)

            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)

            # 直接使用kourichat作为解压后的目录名
            extracted_root = os.path.join(extract_dir, 'kourichat')
            if not os.path.exists(extracted_root):
                raise Exception("无法找到解压后的项目目录")

            # 记录更新的文件
            updated_files = []

            # 复制新文件
            for root, dirs, files in os.walk(extracted_root):
                # 获取相对路径
                rel_path = os.path.relpath(root, extracted_root)
                if rel_path == '.':
                    rel_path = ''

                # 处理目录 - 跳过应该忽略的目录
                dirs_to_remove = []
                for dir_name in dirs:
                    dir_rel_path = os.path.join(rel_path, dir_name).replace('\\', '/')
                    if self.should_skip_file(dir_rel_path):
                        dirs_to_remove.append(dir_name)

                # 从将被处理的目录列表中移除需要忽略的目录
                for dir_name in dirs_to_remove:
                    dirs.remove(dir_name)

                # 处理文件
                for file in files:
                    file_rel_path = os.path.join(rel_path, file).replace('\\', '/')

                    # 判断是否应该忽略这个文件
                    if self.should_skip_file(file_rel_path):
                        continue

                    # 源文件和目标文件路径
                    src_file = os.path.join(root, file)
                    dst_file = os.path.join(self.root_dir, file_rel_path)

                    # 直接复制文件，不进行哈希值比较
                    os.makedirs(os.path.dirname(dst_file), exist_ok=True)
                    shutil.copy2(src_file, dst_file)
                    updated_files.append(file_rel_path)

            # 记录更新的文件列表
            if updated_files:
                logger.info(f"更新了 {len(updated_files)} 个文件")
                logger.debug("更新的文件列表: " + ", ".join(updated_files[:10]) +
                           ("..." if len(updated_files) > 10 else ""))
            else:
                logger.info("未发现需要更新的文件")

            return True
        except Exception as e:
            logger.error(f"更新失败: {str(e)}")
            return False

    def cleanup(self):
        """清理临时文件"""
        try:
            # 清理临时更新目录
            if os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir)
                logger.info(f"已清理临时更新目录: {self.temp_dir}")

            # 清理备份目录
            backup_dir = os.path.join(self.root_dir, 'backup')
            self._force_remove_directory(backup_dir, "备份目录")

            # 清理可能的解压目录 (kourichat 和 KouriChat)
            possible_repo_dirs = [
                os.path.join(self.root_dir, "kourichat"),
                os.path.join(self.root_dir, "KouriChat"),
            ]
            for repo_dir in possible_repo_dirs:
                self._force_remove_directory(repo_dir, "解压目录")

        except Exception as e:
            logger.error(f"清理临时文件失败: {str(e)}")
            # 继续进行强制系统命令删除
            self._system_force_remove(possible_repo_dirs, backup_dir)

    def _force_remove_directory(self, directory, dir_type="目录"):
        """使用多种方法强制删除目录"""
        if not os.path.exists(directory):
            return

        # 尝试修改权限后删除
        try:
            for root, dirs, files in os.walk(directory, topdown=False):
                for file in files:
                    file_path = os.path.join(root, file)
                    try:
                        os.chmod(file_path, 0o777)
                    except:
                        pass
                for dir in dirs:
                    dir_path = os.path.join(root, dir)
                    try:
                        os.chmod(dir_path, 0o777)
                    except:
                        pass

            # 尝试常规删除
            shutil.rmtree(directory)
            logger.info(f"已清理{dir_type}: {directory}")
            return
        except Exception as e:
            logger.warning(f"常规方法删除{dir_type}失败: {str(e)}")

        # 尝试使用系统命令删除
        try:
            import subprocess
            if os.name == 'nt':  # Windows
                subprocess.run(['rd', '/s', '/q', directory], shell=True, timeout=10)
            else:  # Linux/Mac
                subprocess.run(['rm', '-rf', directory], timeout=10)

            if not os.path.exists(directory):
                logger.info(f"已使用系统命令清理{dir_type}: {directory}")
            else:
                logger.warning(f"系统命令无法完全清理{dir_type}: {directory}")
        except Exception as e:
            logger.error(f"使用系统命令清理{dir_type}失败: {str(e)}")

    def _system_force_remove(self, repo_dirs, backup_dir):
        """最后的系统命令强制删除尝试"""
        try:
            import subprocess
            import time

            if os.name == 'nt':  # Windows
                # 尝试使用强制删除命令
                for repo_dir in repo_dirs:
                    if os.path.exists(repo_dir):
                        try:
                            # 先尝试常规删除
                            subprocess.run(['rd', '/s', '/q', repo_dir], shell=True, timeout=5)
                            time.sleep(1)

                            # 如果仍然存在，尝试使用del命令
                            if os.path.exists(repo_dir):
                                subprocess.run(['del', '/f', '/s', '/q', repo_dir], shell=True, timeout=5)
                                time.sleep(1)

                                # 如果仍然存在，尝试使用robocopy技巧清空后删除
                                if os.path.exists(repo_dir):
                                    empty_dir = os.path.join(self.temp_dir, 'empty')
                                    os.makedirs(empty_dir, exist_ok=True)
                                    subprocess.run(['robocopy', empty_dir, repo_dir, '/mir'], timeout=10)
                                    os.rmdir(repo_dir)
                        except:
                            pass

                if os.path.exists(backup_dir):
                    try:
                        subprocess.run(['rd', '/s', '/q', backup_dir], shell=True, timeout=5)
                    except:
                        pass
            else:  # Linux/Mac
                for repo_dir in repo_dirs:
                    if os.path.exists(repo_dir):
                        subprocess.run(['rm', '-rf', '--no-preserve-root', repo_dir], timeout=5)
                if os.path.exists(backup_dir):
                    subprocess.run(['rm', '-rf', '--no-preserve-root', backup_dir], timeout=5)
        except Exception as e:
            logger.error(f"最终强制系统命令清理失败: {str(e)}")

    def prompt_update(self, update_info: dict) -> bool:
        """提示用户是否更新"""
        print(self.format_version_info(self.get_current_version(), update_info))

        # 在WebUI模式下，由前端提供输入确认
        # 这里为了兼容命令行模式，保留原有代码
        while True:
            choice = input("\n是否现在更新? (y/n): ").lower().strip()
            if choice in ('y', 'yes'):
                return True
            elif choice in ('n', 'no'):
                return False
            print("请输入 y 或 n")

    def update(self, callback=None) -> dict:
        """执行更新 - 使用固定的云端 URL 下载更新包"""
        try:
            progress = []
            def log_progress(step, success=True, details=""):
                msg = self.format_update_progress(step, success, details)
                logger.info(msg)  # 确保记录到日志
                progress.append(msg)
                if callback:
                    callback(msg)

            # 检查更新
            log_progress("开始检查云端更新...")
            update_info = self.check_for_updates()
            if not update_info.get('has_update', False):
                log_progress("检查更新完成", True, "当前已是最新版本")
                return {
                    'success': True,
                    'output': '\n'.join(progress)
                }

            # 显示有新版本可用
            log_progress(f"发现新版本: {update_info['version']}", True, "开始下载更新")

            # 下载更新
            log_progress("开始下载更新...")
            if not self.download_update():
                log_progress("下载更新", False, "下载失败")
                return {
                    'success': False,
                    'output': '\n'.join(progress)
                }
            log_progress("下载更新", True, "下载完成")

            # 备份当前版本
            log_progress("开始备份当前版本...")
            if not self.backup_current_version():
                log_progress("备份当前版本", False, "备份失败")
                return {
                    'success': False,
                    'output': '\n'.join(progress)
                }
            log_progress("备份当前版本", True, "备份完成")

            # 应用更新
            log_progress("开始应用更新...")
            if not self.apply_update():
                log_progress("应用更新", False, "更新失败")
                # 尝试恢复
                log_progress("正在恢复之前的版本...")
                if not self.restore_from_backup():
                    log_progress("恢复备份", False, "恢复失败！请手动处理")
                return {
                    'success': False,
                    'output': '\n'.join(progress)
                }
            log_progress("应用更新", True, "更新成功")

            # 更新版本文件
            with open(self.version_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'version': update_info['version'],
                    'last_update': update_info.get('last_update', ''),
                    'description': update_info.get('description', '')
                }, f, indent=4, ensure_ascii=False)

            # 清理
            self.cleanup()
            log_progress("清理临时文件", True)
            log_progress("更新完成", True, "请重启程序以应用更新")

            return {
                'success': True,
                'has_update': True,
                'version': update_info.get('version', ''),
                'description': update_info.get('description', ''),
                'last_update': update_info.get('last_update', ''),
                'output': '\n'.join(progress)
            }

        except Exception as e:
            logger.error(f"更新失败: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'output': f"更新失败: {str(e)}"
            }

def check_cloud_info():
    """检查云端公告、版本信息和模型列表"""
    logger.info("开始检查云端信息...")
    try:
        updater = Updater()
        announcement = updater.fetch_cloud_announcement()
        version = updater.fetch_cloud_version()
        models = updater.fetch_cloud_models()

        # 检查是否所有请求都失败了
        if announcement is None and version is None and models is None:
            logger.error("所有云端信息获取失败，可能是网络问题")
            return None

        return {
            'announcement': announcement,
            'version': version,
            'models': models
        }
    except Exception as e:
        logger.error(f"检查云端信息时发生错误: {str(e)}")
        return None

def check_and_update():
    """检查并执行更新"""
    logger.info("开始检查云端更新...")
    updater = Updater()
    return updater.update()

if __name__ == "__main__":
    # 设置日志格式
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    try:
        result = check_and_update()
        if result['success']:
            input("\n按回车键退出...")  # 等待用户确认后退出
        else:
            input("\n更新失败，按回车键退出...")
    except KeyboardInterrupt:
        print("\n用户取消更新")
    except Exception as e:
        print(f"\n发生错误: {str(e)}")
        input("按回车键退出...")
