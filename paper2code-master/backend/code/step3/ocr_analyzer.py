"""
ocr_analyzer.py

使用 EasyOCR 提取图片中的文字

Author: Paper2Code
"""

import re
import threading

import cv2
import numpy as np

from logger import logger


class OCRAnalyzer:

    def __init__(self):
        self._reader = None
        self._reader_lock = threading.Lock()

    @property
    def reader(self):
        """懒加载 EasyOCR reader（首次使用时才加载模型）。

        用双重检查锁保证多线程并行分析图片时，模型只加载一次，
        避免多个线程同时触发懒加载导致重复初始化。
        """
        if self._reader is None:
            with self._reader_lock:
                if self._reader is None:
                    import easyocr
                    logger.info("Loading EasyOCR...")
                    self._reader = easyocr.Reader(['en'], gpu=False)
                    logger.info("EasyOCR loaded")
        return self._reader

    def analyze(self, image_path: str) -> list:
        """
        对图片执行 OCR，返回关键词列表。

        Args:
            image_path: 图片文件路径

        Returns:
            去重后的 OCR 关键词列表（最多30个）
        """
        logger.info(f"OCR analyze: {image_path}")

        try:
            # 使用 numpy 读取，支持中文路径
            data = np.fromfile(image_path, dtype=np.uint8)
            img = cv2.imdecode(data, cv2.IMREAD_COLOR)

            results = self.reader.readtext(img)

            keywords = []
            for item in results:
                text = item[1].strip()

                # 过滤纯数字
                if re.fullmatch(r"\d+", text):
                    continue

                # 过滤太短的
                if len(text) < 2:
                    continue

                if text not in keywords:
                    keywords.append(text)

            return keywords[:30]

        except Exception as e:
            logger.error(f"OCR failed: {e}")
            return []
