"""
Step2.1

raw_paper.json → cleaned_paper.json

Usage:
    python step2/step2_1_clean_json.py
    python step2/step2_1_clean_json.py --in raw.json --out cleaned.json

Author: Paper2Code
"""

import json
import argparse
import os

from config import OUTPUT_JSON, CLEANED_JSON
from logger import logger
from step2.cleaner import JSONCleaner


def main():
    parser = argparse.ArgumentParser(
        description="Step2.1: 清洗 raw JSON → cleaned JSON"
    )
    parser.add_argument("--in", dest="input_path", type=str, default=OUTPUT_JSON,
                        help="输入 JSON 路径")
    parser.add_argument("--out", dest="output_path", type=str, default=CLEANED_JSON,
                        help="输出 JSON 路径")
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("Step2.1: JSON Cleaning")
    logger.info(f"  Input : {args.input_path}")
    logger.info(f"  Output: {args.output_path}")
    logger.info("=" * 60)

    with open(args.input_path, "r", encoding="utf-8") as f:
        paper = json.load(f)

    cleaner = JSONCleaner()
    paper = cleaner.clean(paper)

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(paper, f, indent=4, ensure_ascii=False)

    logger.info("Step2.1 finished")


if __name__ == "__main__":
    main()
