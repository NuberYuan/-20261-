"""
Step1

PDF -> paper.json

Usage:
    python step1/step1_pdf_to_json.py                           # 使用默认配置
    python step1/step1_pdf_to_json.py --pdf path/to/paper.pdf   # 指定 PDF
    python step1/step1_pdf_to_json.py --pdf paper.pdf --out out.json

Author: Paper2Code
"""

import argparse
from pathlib import Path

from config import INPUT_PDF, OUTPUT_JSON
from logger import logger

from step1.pdf_parser import PDFParser
from step1.json_writer import JsonWriter


def main():
    parser = argparse.ArgumentParser(
        description="Step1: PDF → raw paper JSON"
    )
    parser.add_argument("--pdf", type=str, default=INPUT_PDF,
                        help="输入 PDF 路径")
    parser.add_argument("--out", type=str, default=OUTPUT_JSON,
                        help="输出 JSON 路径")
    parser.add_argument("--images-dir", type=str, default=None,
                        help="图片输出目录（默认 <out>/images）")
    parser.add_argument("--tables-dir", type=str, default=None,
                        help="表格输出目录（默认 <out>/tables）")
    args = parser.parse_args()

    # 图片/表格默认就近放在输出 JSON 同级的 images/ tables/ 目录下
    out_parent = Path(args.out).parent
    images_dir = args.images_dir or str(out_parent / "images")
    tables_dir = args.tables_dir or str(out_parent / "tables")

    logger.info("=" * 60)
    logger.info("Step1 Started")
    logger.info(f"  Input : {args.pdf}")
    logger.info(f"  Output: {args.out}")
    logger.info("=" * 60)

    # 1. 解析 PDF
    pdf_parser = PDFParser(args.pdf, images_dir=images_dir, tables_dir=tables_dir)
    paper = pdf_parser.parse()

    # 2. 保存 JSON
    writer = JsonWriter()
    writer.save(paper, args.out)

    logger.info("paper.json saved.")
    logger.info("=" * 60)
    logger.info("Step1 Finished")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
