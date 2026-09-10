"""
figure_analyzer.py

图片分析流水线: OCR → FigureCodeGenerator (生成代码) → CodeExecutor (执行代码)

Author: Paper2Code
"""

from typing import List, Dict, Optional, Any, Tuple, Union

import json

from logger import logger
from step3.figure_code_generator import FigureCodeGenerator
from step3.code_executor import CodeExecutor
from step3.ocr_analyzer import OCRAnalyzer


class FigureAnalyzer:

    def __init__(self, code_gen_prompt: str, generated_dir: Optional[str] = None):
        self.ocr_analyzer = OCRAnalyzer()
        self.code_generator = FigureCodeGenerator(code_gen_prompt, generated_dir)
        self.executor = CodeExecutor()

    def analyze(self, image_path: str) -> dict:
        """
        分析单张图片。

        Args:
            image_path: 图片文件路径

        Returns:
            分析结果 dict，格式:
            {"type": "...", "description": "...", "important_components": [...],
             "code_relevant_information": "..."}
        """
        logger.info(f"分析图片: {image_path}")

        # 1. OCR
        ocr_text = self.ocr_analyzer.analyze(image_path)
        logger.info(f"OCR结果 ({len(ocr_text)} words): {ocr_text[:5]}...")

        # 2. 生成图片分析代码
        code = self.code_generator.generate_code(image_path, ocr_text)

        # 3. 执行代码
        output = self.executor.run(code, image_path)

        # 4. 解析结果
        if output is None:
            return {
                "type": "other",
                "description": "analysis failed (no output)",
                "important_components": [],
                "code_relevant_information": "",
            }

        try:
            result = json.loads(output)
        except json.JSONDecodeError:
            result = None

        # LLM 生成的代码可能输出数组/标量等非 dict，下游会直接按 dict 取字段，
        # 这里统一兜底成合法的 dict，避免 TypeError。
        if not isinstance(result, dict):
            result = {
                "type": "other",
                "description": "analysis failed (invalid JSON output)",
                "important_components": [],
                "code_relevant_information": output[:500],
            }

        return result
