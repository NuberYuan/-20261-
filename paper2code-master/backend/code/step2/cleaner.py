"""
cleaner.py

JSON 清洗：去除噪音文本、日期、页码、重复段落、断词合并。

Author: Paper2Code
"""

import re
from logger import logger


class JSONCleaner:

    def clean(self, paper: dict) -> dict:
        """
        清洗论文 JSON。

        Args:
            paper: {"paragraphs": [...], "images": [...]}

        Returns:
            清洗后的 paper dict
        """
        logger.info("Cleaning JSON...")

        new_paragraphs = []
        seen = set()

        for item in paper.get("paragraphs", []):
            text = item.get("text", "")

            # 基础清洗
            text = self._clean_text(text)

            # 合并单字母断裂 (M + odern → Modern)
            text = self._merge_single_letter(text)

            if not text:
                continue

            # 删除日期噪音
            if self._is_date_noise(text):
                continue

            # 删除页码噪音
            if self._is_page_number_noise(text):
                continue

            # 删除其他噪音
            if self._is_noise(text):
                continue

            # 去重（取前80字符做 key）
            key = text[:80]
            if key in seen:
                continue
            seen.add(key)

            item["text"] = text
            new_paragraphs.append(item)

        paper["paragraphs"] = new_paragraphs
        logger.info(f"After cleaning: {len(new_paragraphs)} paragraphs")
        return paper

    # =========================
    # 文本清洗
    # =========================

    def _clean_text(self, text: str) -> str:
        """替换换行为空格，压缩连续空白"""
        text = text.replace("\n", " ")
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def _merge_single_letter(self, text: str) -> str:
        """
        修复 PDF 提取造成的字母间空格。

        只合并大写单字母 + 小写词段（真正的断词），
        不合并小写单字母（如冠词 "a" 后面跟正常单词）。

        例如: "M odern" → "Modern"
              "A pproach" → 不合并（保留 "A pproach"，依赖后续清洗）
        """
        return re.sub(r"\b([A-Z]) ([a-z]+)", r"\1\2", text)

    # =========================
    # 噪音检测
    # =========================

    def _is_date_noise(self, text: str) -> bool:
        """检测是否为日期/投稿信息等噪音"""
        patterns = [
            # April 15, 2016
            r"\b(January|February|March|April|May|June|July|August|"
            r"September|October|November|December)\s+\d{1,2},\s+\d{4}\b",
            # 15 April 2016
            r"\b\d{1,2}\s+(January|February|March|April|May|June|July|"
            r"August|September|October|November|December)\s+\d{4}\b",
            # compiled on / received / accepted
            r"(compiled|received|accepted|published|submitted).*\d{4}",
        ]

        for p in patterns:
            if re.search(p, text, re.I):
                return True
        return False

    def _is_page_number_noise(self, text: str) -> bool:
        """检测是否为页码等噪音"""
        patterns = [
            r"^\d+$",              # 纯数字
            r"^page\s*\d+$",       # Page 1
            r"^[-–]\s*\d+\s*[-–]$",  # - 1 -
            r"^\d+\s*/\s*\d+$",    # 1 / 10
        ]

        for p in patterns:
            if re.match(p, text, re.I):
                return True
        return False

    def _is_noise(self, text: str) -> bool:
        """检测其他噪音"""
        if not text:
            return True

        # 单字符非字母 → 噪音
        if len(text) == 1 and not text.isalpha():
            return True

        return False
