"""
text_utils.py

文本处理工具

Author: Paper2Code
"""

import re


# ==========================================================
# 文本清洗
# ==========================================================

def clean_text(text: str) -> str:
    """
    清理PDF提取出的文本
    """

    if not text:
        return ""

    # 统一换行
    text = text.replace("\r", "\n")

    # 去掉连续空格
    text = re.sub(r"[ \t]+", " ", text)

    # 去掉连续空行
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()