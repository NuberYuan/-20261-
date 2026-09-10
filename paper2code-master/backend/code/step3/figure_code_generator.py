"""
figure_code_generator.py

AI 生成图片分析代码

Author: Paper2Code
"""

from typing import List, Dict, Optional, Any, Tuple, Union

import os
import re
from pathlib import Path

from llm_utils import chat_with_text
from logger import logger


class FigureCodeGenerator:

    def __init__(self, prompt: str, generated_dir: Optional[str] = None):
        self.prompt = prompt
        # 生成代码输出目录。下游传入 step3 输出目录下的 generated/；
        # 未传入时回退到本模块下的 generated/，兼容独立调用。
        self.generated_dir = (
            Path(generated_dir) if generated_dir
            else Path(__file__).resolve().parent / "generated"
        )

    def generate_code(self, image_path: str, ocr_text: Optional[list] = None) -> str:
        """
        调用 LLM 生成图片分析代码。

        Args:
            image_path: 图片文件路径
            ocr_text: OCR 提取的文字列表

        Returns:
            Python 代码字符串
        """
        logger.info(f"Figure Code Generator analyze: {image_path}")

        # 填充模板中的 {image_path} 与 {ocr_text} 两个占位符。
        # 之前只替换了 {image_path}，导致模板里 {ocr_text} 的字面量被原样发给 LLM。
        ocr_str = chr(10).join(ocr_text) if ocr_text else "（无）"
        final_prompt = (
            self.prompt
            .replace("{image_path}", image_path)
            .replace("{ocr_text}", ocr_str)
        )

        code = chat_with_text(final_prompt)
        code = self._clean_code(code)

        # 保存生成的代码
        self._save_code(image_path, code)

        return code

    # =========================
    # 代码清理
    # =========================

    @staticmethod
    def _clean_code(code: str) -> str:
        """移除 markdown 代码块包裹"""
        if "```python" in code:
            code = code.replace("```python", "").replace("```", "")
        elif "```" in code:
            code = code.replace("```", "")
        return code.strip()

    # =========================
    # 持久化
    # =========================

    def _save_code(self, image_path: str, code: str) -> Optional[str]:
        """将生成的代码保存到 Step3 输出目录的 generated/ 子目录"""
        try:
            generated_dir = self.generated_dir
            generated_dir.mkdir(parents=True, exist_ok=True)

            name = os.path.splitext(os.path.basename(image_path))[0]
            save_path = generated_dir / f"generated_figure_code_{name}.py"

            with open(save_path, "w", encoding="utf-8") as f:
                f.write(code)

            logger.info(f"Saved generated code: {save_path}")
            return str(save_path)
        except OSError as e:
            logger.warning(f"Failed to save generated code: {e}")
            return None
