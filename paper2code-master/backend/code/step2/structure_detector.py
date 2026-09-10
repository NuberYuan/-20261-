"""
structure_detector.py

论文结构检测：将每个段落分类为 title / author / abstract / keywords / heading / paragraph。

检测策略（按优先级）：
  1. 位置 → 第一个段落 = title
  2. 关键词 → 包含 university/department 等 = author
  3. 关键词 → 以 "Keywords" 开头或含 "|" = keywords
  4. 位置+长度 → 首页前 5 段且 >500 字 = abstract
  5. 格式 → 全大写或匹配典型标题词 = heading
  6. 默认 → paragraph

Author: Paper2Code
"""

import re


class StructureDetector:

    # 常见论文章节标题词
    HEADING_KEYWORDS = [
        "introduction",
        "related work",
        "background",
        "method",
        "methodology",
        "approach",
        "experiment",
        "evaluation",
        "results",
        "discussion",
        "conclusion",
        "future work",
        "acknowledgment",
        "acknowledgements",
        "references",
        "appendix",
        "implementation",
        "dataset",
        "training",
        "architecture",
        "model",
        "data",
        "analysis",
        "overview",
    ]

    # 作者机构关键词
    AFFILIATION_KEYWORDS = [
        "university",
        "department",
        "laboratory",
        "lab",
        "institute",
        "school",
        "college",
        "research center",
        "research centre",
    ]

    def detect(self, paper: dict) -> dict:
        """对论文 JSON 中每个段落标注 type 和 confidence"""
        if "paragraphs" not in paper:
            return paper

        paragraphs = paper["paragraphs"]

        for i, para in enumerate(paragraphs):
            text = (para.get("text") or "").strip()
            page = para.get("page", 0)
            bbox = para.get("bbox", [])

            para_type, confidence = self._classify(text, i, page, bbox)
            para["type"] = para_type
            para["confidence"] = confidence

            # 删除旧字段
            para.pop("section_hint", None)

        return paper

    # =========================
    # 分类主逻辑
    # =========================

    def _classify(self, text: str, index: int, page: int, bbox: list) -> tuple:
        """返回 (type, confidence)"""
        length = len(text)

        # title: 第一个段落
        if index == 0:
            return ("title", 0.95)

        # author: 作者/机构信息
        if self._is_affiliation(text):
            return ("author", 0.90)

        # keywords: 关键词段
        if self._is_keywords(text):
            return ("keywords", 0.90)

        # abstract: 首页靠前的大段文本
        if self._is_abstract(text, index, page):
            return ("abstract", 0.85)

        # heading: 章节标题
        if self._is_heading(text, length):
            return ("heading", 0.75)

        # 正文段落
        return ("paragraph", 0.50)

    # =========================
    # 分类规则
    # =========================

    # 技术/学术关键词——短文本包含这些词说明是内容而非作者
    NOT_AUTHOR_KEYWORDS = [
        "batch", "learning", "epoch", "layer", "dropout",
        "weight", "decay", "momentum", "optimizer", "loss",
        "accuracy", "split", "ratio", "size", "model",
        "figure", "table", "experiment", "dataset", "training",
        "comparison", "progression", "classification",
    ]

    # 常见英文虚词/代词——作者行不应该包含这些
    FUNCTION_WORDS = [
        "which", "that", "this", "these", "those", "they", "their",
        "also", "very", "note", "show", "found", "result",
        "make", "made", "will", "can", "may", "has", "have",
        "with", "for", "from", "into", "over", "than",
        "total", "number", "sample", "case", "data",
    ]

    def _is_affiliation(self, text: str) -> bool:
        """检测作者/机构段落"""
        lower = text.lower()

        # 机构关键词 ≥2 → 确定是作者机构
        count = sum(1 for k in self.AFFILIATION_KEYWORDS if k in lower)
        if count >= 2:
            return True

        # 含逗号、长度适中、不含日期 → 可能是作者列表
        if len(text) < 200 and "," in text and not self._contains_date(text):
            if self._is_not_author(text):
                return False
            return True

        return False

    def _is_not_author(self, text: str) -> bool:
        """排除看起来像作者但实际是技术内容的短文本"""
        lower = text.lower()

        # 以子弹点/括号/数字编号开头 → 内容
        if re.match(r'^[\•\-\*\·\>\►\✓\✔\☐\(\[\d]', text):
            return True

        # 以 "Figure" / "Fig" 开头 → 图注
        if re.match(r'^(?:Figure|Fig)[\s\.]', text, re.I):
            return True

        # 包含 ≥1 个技术关键词 → 技术内容
        tech_count = sum(1 for k in self.NOT_AUTHOR_KEYWORDS if k in lower)
        if tech_count >= 1:
            return True

        # 包含英文虚词（which/that/this/also 等）→ 自然语言而非作者名
        func_count = sum(1 for w in self.FUNCTION_WORDS if w in lower)
        if func_count >= 1:
            return True

        return False

    def _is_keywords(self, text: str) -> bool:
        """检测关键词段落"""
        if "|" in text:
            return True

        if text.lower().startswith("keywords"):
            return True

        return False

    def _is_abstract(self, text: str, index: int, page: int) -> bool:
        """检测摘要段落"""
        # 首页前 5 段，且 >500 字符
        if page == 1 and index <= 5 and len(text) > 500:
            return True
        return False

    def _is_heading(self, text: str, length: int) -> bool:
        """检测章节标题"""
        # 标题通常较短
        if length > 80:
            return False

        # 全大写格式
        if text.isupper():
            return True

        # 以数字开头 (如 "1. Introduction", "3.1 Method")
        if re.match(r"^[\d]+[\.\)]\s", text):
            return True

        # 匹配常见的标题词
        lower = text.lower().strip(" .:")
        for kw in self.HEADING_KEYWORDS:
            if lower == kw or lower.startswith(kw):
                return True

        return False

    # =========================
    # 辅助方法
    # =========================

    @staticmethod
    def _contains_date(text: str) -> bool:
        """检测文本是否包含日期"""
        patterns = [
            r"\b\d{1,2},\s\d{4}\b",
            r"\b\d{4}\b",
        ]
        for p in patterns:
            if re.search(p, text):
                return True
        return False
