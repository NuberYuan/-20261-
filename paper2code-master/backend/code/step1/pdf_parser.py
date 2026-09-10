"""
pdf_parser.py

使用 PyMuPDF 解析论文，生成 paragraph 级数据。

Author: Paper2Code
"""

import fitz

from config import IMAGE_OUTPUT_DIR, TABLE_OUTPUT_DIR
from logger import logger
from step1.text_utils import clean_text
from step1.image_extractor import ImageExtractor
from step1.table_extractor import TableExtractor


class PDFParser:

    def __init__(self, pdf_path: str, images_dir: str = None, tables_dir: str = None):
        self.pdf_path = pdf_path
        # 默认回退到 config 中的全局目录，支持指定论文时按输出目录就近存放
        self.images_dir = images_dir or IMAGE_OUTPUT_DIR
        self.tables_dir = tables_dir or TABLE_OUTPUT_DIR

    def parse(self) -> dict:
        """
        解析 PDF，返回:
          {
            "paragraphs": [...],
            "images": [...],
            "tables": [...]      # 新增: 结构化表格数据
          }
        """
        logger.info(f"Reading PDF: {self.pdf_path}")

        doc = fitz.open(self.pdf_path)

        paragraphs = []
        pid = 1

        # ------------------------------------
        # 逐页解析文本块
        # ------------------------------------
        for page_no, page in enumerate(doc):
            blocks = page.get_text("blocks")

            for block in blocks:
                x0, y0, x1, y1, text, block_no, block_type = block

                text = clean_text(text)

                if not text:
                    continue

                paragraphs.append({
                    "id": pid,
                    "block_id": block_no,
                    "page": page_no + 1,
                    "bbox": [x0, y0, x1, y1],
                    "type": "unknown",
                    "text": text,
                })

                pid += 1

        logger.info(f"Paragraph Count: {len(paragraphs)}")

        # ==========================
        # 提取 PDF 图片
        # ==========================
        extractor = ImageExtractor(self.images_dir)
        images = extractor.extract(doc)

        logger.info(f"Image Count: {len(images)}")

        # ==========================
        # 提取 PDF 表格
        # ==========================
        table_extractor = TableExtractor(self.tables_dir)
        tables = table_extractor.extract(doc)

        logger.info(f"Table Count: {len(tables)}")

        doc.close()

        paper = {
            "paragraphs": paragraphs,
            "images": images,
            "tables": tables,
        }

        return paper
