"""
Step2.2

cleaned_paper.json → repaired_paper.json

Usage:
    python step2/step2_2_repair_json.py
    python step2/step2_2_repair_json.py --in cleaned.json --out repaired.json

Author: Paper2Code
"""

import json
import argparse
import os

from step2.repairer import JSONRepairer
from config import CLEANED_JSON, REPAIRED_JSON
from logger import logger


def main():
    parser = argparse.ArgumentParser(
        description="Step2.2: 修复 JSON (连字、断词)"
    )
    parser.add_argument("--in", dest="input_path", type=str, default=CLEANED_JSON,
                        help="输入 JSON 路径")
    parser.add_argument("--out", dest="output_path", type=str, default=REPAIRED_JSON,
                        help="输出 JSON 路径")
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("Step2.2: JSON Repair")
    logger.info(f"  Input : {args.input_path}")
    logger.info(f"  Output: {args.output_path}")
    logger.info("=" * 60)

    with open(args.input_path, "r", encoding="utf-8") as f:
        paper = json.load(f)

    logger.info(f"Input paragraphs: {len(paper.get('paragraphs', []))}")

    repairer = JSONRepairer()
    repaired = repairer.repair(paper)

    logger.info(f"Output paragraphs: {len(repaired.get('paragraphs', []))}")

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(repaired, f, ensure_ascii=False, indent=2)

    logger.info(f"Saved: {args.output_path}")
    logger.info("Step2.2 completed.")


if __name__ == "__main__":
    main()
