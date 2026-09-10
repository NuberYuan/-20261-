"""
table_extractor.py

从 PDF 中提取表格数据。

两阶段策略:
  1) PyMuPDF find_tables() — 真正的网格表格
  2) 结构化文本检测 — 识别子弹点列表、编号列表、键值对组

输出: 结构化 JSON，包含 header + rows

Author: Paper2Code
"""

import re
import json
from pathlib import Path
from typing import List, Optional, Dict, Any

import fitz

from logger import logger


class TableExtractor:
    """从 PDF 文档中提取表格"""

    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # =========================
    # 主入口
    # =========================

    def extract(self, doc: fitz.Document) -> List[Dict[str, Any]]:
        """
        从 PDF 文档中提取所有表格。

        Returns:
            [
              {
                "id": "table_1",
                "page": 5,
                "method": "grid",      # grid | bullet_list | numbered_list | kv_pairs
                "caption": "...",       # 表格上方的标题文本（如果有）
                "header": ["col1", "col2"],
                "rows": [["val1", "val2"], ...],
                "bbox": [x0, y0, x1, y1],
              },
              ...
            ]
        """
        tables = []
        tid = 1

        for page_no in range(doc.page_count):
            page = doc[page_no]

            # ---- 阶段 1: 网格表格 ----
            grid_tables = self._extract_grid_tables(page, page_no + 1)
            for t in grid_tables:
                t["id"] = f"table_{tid}"
                tid += 1
                tables.append(t)

            # ---- 阶段 2: 结构化文本 ----
            grouped = self._extract_structured_text(page, page_no + 1)
            for t in grouped:
                t["id"] = f"table_{tid}"
                tid += 1
                tables.append(t)

        # ---- 保存 ----
        if tables:
            self._save(tables)

        logger.info(f"Tables extracted: {len(tables)}")
        return tables

    # =========================
    # 网格表格 (PyMuPDF)
    # =========================

    @staticmethod
    def _extract_grid_tables(page: fitz.Page, page_no: int) -> List[Dict[str, Any]]:
        """使用 PyMuPDF find_tables 提取带网格线的表格"""
        results = []
        try:
            found = page.find_tables()
            if not found or not found.tables:
                return results
        except Exception:
            return results

        for t in found.tables:
            try:
                rows = t.extract()
                if not rows:
                    continue

                # 第一行作为 header
                header = [str(c) if c else "" for c in rows[0]]
                data_rows = [[str(c) if c else "" for c in r] for r in rows[1:]]

                # 过滤全空行
                data_rows = [r for r in data_rows if any(c.strip() for c in r)]

                if not data_rows:
                    continue

                results.append({
                    "page": page_no,
                    "method": "grid",
                    "caption": TableExtractor._find_caption(page, t.bbox),
                    "header": header,
                    "rows": data_rows,
                    "bbox": list(t.bbox),
                    "row_count": len(data_rows),
                    "col_count": len(header),
                })
            except Exception as e:
                logger.warning(f"Grid table extraction error on page {page_no}: {e}")
                continue

        return results

    # =========================
    # 辅助
    # =========================

    @staticmethod
    def _find_caption(page: fitz.Page, table_bbox: tuple) -> str:
        """在表格上方寻找可能的标题文本"""
        blocks = page.get_text("blocks")
        x0, y0, x1, y1 = table_bbox
        caption_candidates = []

        for b in blocks:
            bx0, by0, bx1, by1 = b[:4]
            text = b[4].strip() if len(b) > 4 else ""
            # 表格上方 50pt 范围内的文本
            if (
                abs(bx0 - x0) < 50
                and by1 < y0
                and y0 - by1 < 50
                and bx1 - bx0 > 50
            ):
                if text and len(text) > 5:
                    # 以 Table / Tab 开头的是标题
                    if re.match(r'^(Table|Tab\.?)\s', text, re.I):
                        caption_candidates.append(text)

        return " | ".join(caption_candidates) if caption_candidates else ""

    # =========================
    # 结构化文本检测
    # =========================

    @staticmethod
    def _extract_structured_text(page: fitz.Page, page_no: int) -> List[Dict[str, Any]]:
        """
        检测非网格的"类表格"文本:
          - 子弹点列表（每行格式一致，如 "• Key: value"）
          - 编号列表（如 "1) ItemA, 2) ItemB"）
          - 键值对组（连续多行 "Key: value"）
        """
        blocks = page.get_text("blocks")
        results = []

        # 按 y 排序
        text_blocks = []
        for b in blocks:
            text = b[4].strip() if len(b) > 4 else ""
            if text and len(text) > 5:
                text_blocks.append({
                    "text": text,
                    "bbox": list(b[:4]),
                    "y": b[1],  # y0
                })
        text_blocks.sort(key=lambda b: b["y"])

        # ---- 检测子弹点组 ----
        bullet_group = TableExtractor._find_bullet_group(text_blocks)
        if bullet_group:
            results.append(bullet_group)

        # ---- 检测编号列表 ----
        numbered = TableExtractor._find_numbered_list(text_blocks)
        if numbered:
            results.append(numbered)

        return results

    @staticmethod
    def _find_bullet_group(blocks: list) -> Optional[Dict[str, Any]]:
        """检测连续的子弹点行"""
        bullet_pattern = re.compile(r'^[\•\-\*\·\>\►\✓\✔\☐\d+\.\)]\s*(.+)')
        group = []
        bboxes = []

        for b in blocks:
            text = b["text"]
            m = bullet_pattern.match(text)
            if m:
                group.append(m.group(1).strip())
                bboxes.append(b["bbox"])
            else:
                # 连续中断 → 如果已经收集了≥3行，认为是一个表格
                if len(group) >= 3:
                    break
                group = []
                bboxes = []

        if len(group) < 3:
            return None

        # 尝试解析 key: value 格式
        key_value_pairs = []
        for item in group:
            kv = re.split(r'\s*[:：=]\s*', item, maxsplit=1)
            if len(kv) == 2:
                key_value_pairs.append([kv[0].strip(), kv[1].strip()])
            else:
                key_value_pairs.append([item, ""])

        return {
            "page": None,  # 跨 block 不标记单个 page
            "method": "bullet_list",
            "caption": "",
            "header": ["Key", "Value"] if all(len(kv) == 2 and kv[1] for kv in key_value_pairs) else ["Item"],
            "rows": key_value_pairs,
            "bbox": [
                min(b[0] for b in bboxes),
                bboxes[0][1],
                max(b[2] for b in bboxes),
                bboxes[-1][3],
            ],
            "row_count": len(key_value_pairs),
            "col_count": len(key_value_pairs[0]) if key_value_pairs else 1,
        }

    @staticmethod
    def _find_numbered_list(blocks: list) -> Optional[Dict[str, Any]]:
        """检测编号列表 1) ... 2) ..."""
        numbered_pattern = re.compile(r'^(\d+)[\.\)]\s*(.+)')
        group = []
        numbers = []
        bboxes = []

        for b in blocks:
            text = b["text"]
            m = numbered_pattern.match(text)
            if m:
                num = int(m.group(1))
                if not numbers or num == numbers[-1] + 1:
                    group.append(m.group(2).strip())
                    numbers.append(num)
                    bboxes.append(b["bbox"])
                else:
                    if len(group) >= 5:
                        break
                    group = []
                    numbers = []
            else:
                if len(group) >= 5:
                    break
                group = []
                numbers = []

        if len(group) < 5:
            return None

        return {
            "page": None,
            "method": "numbered_list",
            "caption": "",
            "header": ["No.", "Content"],
            "rows": [[str(n), c] for n, c in zip(numbers, group)],
            "bbox": [
                min(b[0] for b in bboxes),
                bboxes[0][1],
                max(b[2] for b in bboxes),
                bboxes[-1][3],
            ],
            "row_count": len(group),
            "col_count": 2,
        }

    # =========================
    # 持久化
    # =========================

    def _save(self, tables: List[Dict[str, Any]]) -> None:
        """保存表格为 JSON"""
        save_path = self.output_dir / "extracted_tables.json"

        # 不保存 bbox（太大了）
        clean = []
        for t in tables:
            clean.append({
                k: v for k, v in t.items()
                if k != "bbox"
            })

        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(clean, f, ensure_ascii=False, indent=2)

        # 同时为每个表格保存 CSV
        for t in tables:
            csv_path = self.output_dir / f"{t['id']}.csv"
            try:
                with open(csv_path, "w", encoding="utf-8-sig") as f:
                    if t.get("header"):
                        f.write(",".join(f'"{c}"' for c in t["header"]) + "\n")
                    for row in t.get("rows", []):
                        f.write(",".join(f'"{c}"' for c in row) + "\n")
            except Exception as e:
                logger.warning(f"Failed to save CSV {csv_path}: {e}")

        logger.info(f"Saved {len(tables)} tables to {self.output_dir}")
