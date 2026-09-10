"""
image_extractor.py

负责提取PDF中的图片

Author: Paper2Code
"""

import os
from pathlib import Path
import fitz

from logger import logger


class ImageExtractor:

    def __init__(self, output_dir: str):
        """
        Args:
            output_dir: 图片输出目录，图片将直接存入此目录
        """
        self.output_dir = Path(output_dir)

    def extract(self, doc: fitz.Document) -> list:
        """
        从 PDF 文档中提取所有图片。

        Args:
            doc: PyMuPDF Document 对象

        Returns:
            [{"id": "image_1", "page": 1, "path": "/abs/path/image_1.png"}, ...]
        """
        images = []
        image_dir = self.output_dir
        image_dir.mkdir(parents=True, exist_ok=True)

        image_id = 1

        for page_no, page in enumerate(doc):
            img_list = page.get_images(full=True)
            logger.info(f"Page {page_no + 1} image count: {len(img_list)}")

            for img in img_list:
                xref = img[0]
                pix = fitz.Pixmap(doc, xref)

                image_path = image_dir / f"image_{image_id}.png"
                pix.save(str(image_path))

                images.append({
                    "id": f"image_{image_id}",
                    "page": page_no + 1,
                    "path": str(image_path),
                })

                image_id += 1

        logger.info(f"Image Count: {len(images)}")
        return images
