"""
Step2.3

repaired_paper.json → structure_detect_paper.json

Usage:
    python step2/step2_3_detect_structure.py
    python step2/step2_3_detect_structure.py --in repaired.json --out structure.json

Author: Paper2Code
"""

import json
import argparse
import os

from step2.structure_detector import StructureDetector
from config import REPAIRED_JSON, STRUCTURE_DETECT
from logger import logger


def main():
    parser = argparse.ArgumentParser(
        description="Step2.3: 论文结构检测"
    )
    parser.add_argument("--in", dest="input_path", type=str, default=REPAIRED_JSON,
                        help="输入 JSON 路径")
    parser.add_argument("--out", dest="output_path", type=str, default=STRUCTURE_DETECT,
                        help="输出 JSON 路径")
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("Step2.3: Structure Detection")
    logger.info(f"  Input : {args.input_path}")
    logger.info(f"  Output: {args.output_path}")
    logger.info("=" * 60)

    with open(args.input_path, "r", encoding="utf-8") as f:
        paper = json.load(f)

    logger.info(f"Loaded paragraphs: {len(paper.get('paragraphs', []))}")

    detector = StructureDetector()
    result = detector.detect(paper)

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    # 统计各类型数量
    type_counts = {}
    for p in result.get("paragraphs", []):
        t = p.get("type", "unknown")
        type_counts[t] = type_counts.get(t, 0) + 1
    logger.info(f"Structure distribution: {type_counts}")

    logger.info(f"Saved: {args.output_path}")
    logger.info("=" * 60)
    logger.info("Step2.3 Finished")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
