# video_evaluation_client.py - Qwen3VL版媒体评估客户端
from gradio_client import Client
import json
import os

class MediaEvaluationClient:
    def __init__(self, server_url="http://localhost:16660", logger=None):
        """初始化媒体评估客户端"""
        self.server_url = server_url
        self.client = None
        self.logger = logger or self._get_default_logger()
        self.connect()

    def _get_default_logger(self):
        import logging
        logger = logging.getLogger("MediaEvaluationClient")
        if not logger.handlers:
            logger.addHandler(logging.StreamHandler())
        logger.setLevel(logging.INFO)
        return logger

    def connect(self):
        """连接到服务器"""
        try:
            self.client = Client(self.server_url)
            self.logger.info(f"成功连接到媒体评估服务器: {self.server_url}")
        except Exception as e:
            self.logger.error(f"连接失败: {e}")
            self.client = None

    def _ensure_connected(self):
        """确保客户端已连接"""
        if self.client is None:
            self.connect()
        return self.client is not None

    def analyze_mismatches(self, media_path, original_prompt, media_type="video", return_ids=False):
        """
        分析媒体与原始提示的不匹配

        Args:
            media_path (str): 媒体文件路径
            original_prompt (str): 原始提示词
            media_type (str): 媒体类型 ("video" 或 "image")
            return_ids (bool): 兼容性参数，暂未使用

        Returns:
            str: 不匹配分析结果
        """
        if not self._ensure_connected():
            return "客户端未连接"

        if not os.path.exists(media_path):
            return f"文件不存在: {media_path}"

        try:
            media_name = "视频" if media_type == "video" else "图片"
            self.logger.info(f"分析{media_name}不匹配: {os.path.basename(media_path)}")

            result = self.client.predict(
                media_path,
                original_prompt,
                media_type,
                api_name="/analyze_mismatches"
            )

            self.logger.info("不匹配分析完成")
            return result

        except Exception as e:
            error_msg = f"不匹配分析失败: {str(e)}"
            self.logger.error(error_msg)
            return error_msg

    def get_enhanced_prompt(self, media_path, original_prompt, diagnosis, media_type="video", return_ids=False):
        """
        获取增强的提示词

        Args:
            media_path (str): 媒体文件路径
            original_prompt (str): 原始提示词
            diagnosis (str): 诊断结果
            media_type (str): 媒体类型 ("video" 或 "image")
            return_ids (bool): 兼容性参数，暂未使用

        Returns:
            str: 增强的提示词
        """
        if not self._ensure_connected():
            return "客户端未连接"

        if not os.path.exists(media_path):
            return f"文件不存在: {media_path}"

        try:
            media_name = "视频" if media_type == "video" else "图片"
            self.logger.info(f"生成增强提示词: {os.path.basename(media_path)}")

            result = self.client.predict(
                media_path,
                original_prompt,
                diagnosis,
                media_type,
                api_name="/get_enhanced_prompt"
            )

            self.logger.info("增强提示词生成完成")
            return result

        except Exception as e:
            error_msg = f"增强提示词生成失败: {str(e)}"
            self.logger.error(error_msg)
            return error_msg

    def get_negative_prompt(self, media_path, original_prompt, diagnosis, media_type="video", return_ids=False):
        """
        获取负面提示词

        Args:
            media_path (str): 媒体文件路径
            original_prompt (str): 原始提示词
            diagnosis (str): 诊断结果
            media_type (str): 媒体类型 ("video" 或 "image")
            return_ids (bool): 兼容性参数，暂未使用

        Returns:
            str: 负面提示词
        """
        if not self._ensure_connected():
            return "客户端未连接"

        if not os.path.exists(media_path):
            return f"文件不存在: {media_path}"

        try:
            media_name = "视频" if media_type == "video" else "图片"
            self.logger.info(f"生成负面提示词: {os.path.basename(media_path)}")

            result = self.client.predict(
                media_path,
                original_prompt,
                diagnosis,
                media_type,
                api_name="/get_negative_prompt"
            )

            self.logger.info("负面提示词生成完成")
            return result

        except Exception as e:
            error_msg = f"负面提示词生成失败: {str(e)}"
            self.logger.error(error_msg)
            return error_msg

    def evaluate_media(self, media_path, original_prompt, media_type="video", return_ids=False):
        """
        完整的媒体评估

        Args:
            media_path (str): 媒体文件路径
            original_prompt (str): 原始提示词
            media_type (str): 媒体类型 ("video" 或 "image")
            return_ids (bool): 兼容性参数，暂未使用

        Returns:
            dict: 包含enhanced_prompt, negative_prompt, diagnosis的字典
        """
        if not self._ensure_connected():
            return {
                "enhanced_prompt": "客户端未连接",
                "negative_prompt": "客户端未连接",
                "diagnosis": "客户端未连接"
            }

        if not os.path.exists(media_path):
            error_msg = f"文件不存在: {media_path}"
            return {
                "enhanced_prompt": error_msg,
                "negative_prompt": error_msg,
                "diagnosis": error_msg
            }

        try:
            media_name = "视频" if media_type == "video" else "图片"
            self.logger.info(f"开始完整评估{media_name}: {os.path.basename(media_path)}")
            self.logger.info(f"原始提示词: {original_prompt}")
            self.logger.info(f"媒体类型: {media_type}")

            # 调用完整评估API
            result_json = self.client.predict(
                media_path,
                original_prompt,
                media_type,
                api_name="/evaluate_media_complete"
            )

            # 解析JSON结果
            try:
                result_dict = json.loads(result_json)
                if result_dict.get("success", False):
                    self.logger.info("完整评估成功完成")
                    return {
                        "enhanced_prompt": result_dict.get("enhanced_prompt", ""),
                        "negative_prompt": result_dict.get("negative_prompt", ""),
                        "diagnosis": result_dict.get("diagnosis", "")
                    }
                else:
                    error_msg = result_dict.get("error", "未知错误")
                    self.logger.error(f"评估失败: {error_msg}")
                    return {
                        "enhanced_prompt": error_msg,
                        "negative_prompt": error_msg,
                        "diagnosis": error_msg
                    }
            except json.JSONDecodeError:
                # 如果不是JSON格式，可能是直接返回的文本
                self.logger.error("返回结果不是JSON格式，使用分步评估")
                return self._evaluate_step_by_step(media_path, original_prompt, media_type)

        except Exception as e:
            error_msg = f"完整评估失败: {str(e)}"
            self.logger.error(error_msg)
            return {
                "enhanced_prompt": error_msg,
                "negative_prompt": error_msg,
                "diagnosis": error_msg
            }

    def _evaluate_step_by_step(self, media_path, original_prompt, media_type):
        """分步评估（备用方案）"""
        try:
            self.logger.info("使用分步评估方案...")

            # 1. 获取诊断
            diagnosis = self.analyze_mismatches(media_path, original_prompt, media_type)

            # 2. 获取增强提示
            enhanced = self.get_enhanced_prompt(media_path, original_prompt, diagnosis, media_type)

            # 3. 获取负面提示
            negative = self.get_negative_prompt(media_path, original_prompt, diagnosis, media_type)

            return {
                "enhanced_prompt": enhanced,
                "negative_prompt": negative,
                "diagnosis": diagnosis
            }

        except Exception as e:
            error_msg = f"分步评估失败: {str(e)}"
            return {
                "enhanced_prompt": error_msg,
                "negative_prompt": error_msg,
                "diagnosis": error_msg
            }

    # 保持向后兼容的别名
    def evaluate_video(self, video_path, original_prompt, return_ids=False):
        """
        评估视频（向后兼容的方法）

        Args:
            video_path (str): 视频文件路径
            original_prompt (str): 原始提示词
            return_ids (bool): 兼容性参数，暂未使用

        Returns:
            dict: 评估结果字典
        """
        return self.evaluate_media(video_path, original_prompt, media_type="video", return_ids=return_ids)

    def evaluate_image(self, image_path, original_prompt, return_ids=False):
        """
        评估图片

        Args:
            image_path (str): 图片文件路径
            original_prompt (str): 原始提示词
            return_ids (bool): 兼容性参数，暂未使用

        Returns:
            dict: 评估结果字典
        """
        return self.evaluate_media(image_path, original_prompt, media_type="image", return_ids=return_ids)

def quick_evaluate_video(video_path, original_prompt, server_url="http://localhost:16660", media_type="video", logger=None):
    """
    快速评估单个媒体文件的便捷函数

    Args:
        video_path (str): 媒体文件路径（为了向后兼容保留video_path参数名）
        original_prompt (str): 原始提示词
        server_url (str): 服务器地址
        media_type (str): 媒体类型 ("video" 或 "image")
        logger: 日志记录器（可选）

    Returns:
        dict: 评估结果字典
    """
    client = MediaEvaluationClient(server_url, logger=logger)
    return client.evaluate_media(video_path, original_prompt, media_type=media_type)

def quick_evaluate_media(media_path, original_prompt, media_type="video", server_url="http://localhost:16660", logger=None):
    """
    快速评估单个媒体文件的便捷函数

    Args:
        media_path (str): 媒体文件路径
        original_prompt (str): 原始提示词
        media_type (str): 媒体类型 ("video" 或 "image")
        server_url (str): 服务器地址
        logger: 日志记录器（可选）

    Returns:
        dict: 评估结果字典
    """
    client = MediaEvaluationClient(server_url, logger=logger)
    return client.evaluate_media(media_path, original_prompt, media_type=media_type)