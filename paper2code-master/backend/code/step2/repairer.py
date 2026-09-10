"""
repairer.py

JSON 修复：连字替换、断词修复、单字母合并。

Author: Paper2Code
"""

import re


class JSONRepairer:

    def repair(self, paper: dict) -> dict:
        """修复论文 JSON"""
        if not paper:
            return paper

        if "paragraphs" in paper:
            paper["paragraphs"] = self._repair_blocks(paper["paragraphs"])
            return paper

        if isinstance(paper, list):
            return self._repair_blocks(paper)

        if "blocks" in paper:
            paper["blocks"] = self._repair_blocks(paper["blocks"])
            return paper

        return paper

    # =========================
    # 块级修复
    # =========================

    def _repair_blocks(self, blocks: list) -> list:
        """对段落列表进行修复"""
        repaired = []
        i = 0

        while i < len(blocks):
            current = blocks[i]
            text = self._get_text(current)

            # Step 1: 基础文本修复
            text = self._fix_ligature(text)
            text = self._fix_internal_hyphen(text)
            current["text"] = text

            if not text:
                i += 1
                continue

            # Step 2: block 级断裂修复
            #   情况 A: M + odern → Modern (单字母断裂)
            #   情况 B: threat- + ened → threatened (单词断裂)
            if i + 1 < len(blocks):
                nxt = blocks[i + 1]
                next_text = self._get_text(nxt)

                # 单字母断裂: "M" + "odern"
                if self._is_single_letter(text):
                    merged = self._merge_blocks(current, nxt, join_direct=True)
                    repaired.append(merged)
                    i += 2
                    continue

                # 单词断裂: "threat-" + "ened"
                if self._is_word_break(text, next_text):
                    merged = self._merge_blocks(current, nxt, remove_hyphen=True)
                    repaired.append(merged)
                    i += 2
                    continue

            repaired.append(current)
            i += 1

        return repaired

    # =========================
    # 文本修复方法
    # =========================

    def _fix_ligature(self, text: str) -> str:
        """修复 PDF 连字 (ligature)

        ﬁ → fi, ﬀ → ff, ﬂ → fl, ﬃ → ffi, ﬄ → ffl
        """
        replacements = {
            "ﬃ": "ffi",
            "ﬄ": "ffl",
            "ﬁ": "fi",
            "ﬂ": "fl",
            "ﬀ": "ff",
        }
        for old, new in replacements.items():
            text = text.replace(old, new)
        return text

    def _fix_internal_hyphen(self, text: str) -> str:
        """修复 PDF 内部断词连字符

        threat-\nen ed → threatened
        """
        return re.sub(
            r"-[\s  -​]*([a-z]{2,})",
            r"\1",
            text,
        )

    # =========================
    # 辅助方法
    # =========================

    @staticmethod
    def _get_text(block) -> str:
        if isinstance(block, dict):
            return (block.get("text") or "").strip()
        return ""

    @staticmethod
    def _is_single_letter(text: str) -> bool:
        return bool(re.fullmatch(r"[A-Za-z]", text))

    @staticmethod
    def _is_word_break(text1: str, text2: str) -> bool:
        """判断 text1 和 text2 是否为断裂的单词"""
        if not text1.endswith("-"):
            return False
        if len(text1) < 3:
            return False
        if not text2:
            return False
        return bool(re.match(r"^[a-z]+", text2))

    def _merge_blocks(self, block1: dict, block2: dict,
                      remove_hyphen: bool = False,
                      join_direct: bool = False) -> dict:
        """合并两个 block"""
        new_block = block1.copy()
        text1 = self._get_text(block1)
        text2 = self._get_text(block2)

        if remove_hyphen:
            text1 = text1[:-1]
            new_text = text1 + text2
        elif join_direct:
            new_text = text1 + text2
        else:
            new_text = text1 + " " + text2

        new_block["text"] = new_text
        return new_block
