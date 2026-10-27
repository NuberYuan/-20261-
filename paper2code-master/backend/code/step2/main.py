"""
Step2 主入口：一步完成 JSON 清洗 + 修复 + 结构检测

raw_paper.json → structure_detect_paper.json

Usage:
    python step2/main.py
    python step2/main.py --in raw.json --out structure.json

Author: Paper2Code
"""

import json
import argparse
import os

from config import OUTPUT_JSON, STRUCTURE_DETECT
from logger import logger
from step2.cleaner import JSONCleaner
from step2.repairer import JSONRepairer
from step2.structure_detector import StructureDetector


def main():
    parser = argparse.ArgumentParser(
        description="Step2: JSON 清洗 + 修复 + 结构检测"
    )
    parser.add_argument("--in", dest="input_path", type=str, default=OUTPUT_JSON,
                        help="输入 JSON 路径 (默认: raw_paper.json)")
    parser.add_argument("--out", dest="output_path", type=str, default=STRUCTURE_DETECT,
                        help="输出 JSON 路径 (默认: structure_detect_paper.json)")
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("Step2: Clean + Repair + Structure Detection")
    logger.info(f"  Input : {args.input_path}")
    logger.info(f"  Output: {args.output_path}")
    logger.info("=" * 60)

    # 加载原始 JSON
    with open(args.input_path, "r", encoding="utf-8") as f:
        paper = json.load(f)

    logger.info(f"Loaded {len(paper.get('paragraphs', []))} paragraphs")

    # Step 2.1: 清洗
    logger.info("--- Step2.1: Cleaning ---")
    cleaner = JSONCleaner()
    paper = cleaner.clean(paper)

    # Step 2.2: 修复连字、断词
    logger.info("--- Step2.2: Repairing ---")
    repairer = JSONRepairer()
    paper = repairer.repair(paper)
    logger.info(f"After repair: {len(paper.get('paragraphs', []))} paragraphs")

    # Step 2.3: 结构检测
    logger.info("--- Step2.3: Structure Detection ---")
    detector = StructureDetector()
    paper = detector.detect(paper)

    # 统计各类型数量
    type_counts = {}
    for p in paper.get("paragraphs", []):
        t = p.get("type", "unknown")
        type_counts[t] = type_counts.get(t, 0) + 1
    logger.info(f"Structure distribution: {type_counts}")

    # 保存结果
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(paper, f, ensure_ascii=False, indent=2)

    logger.info(f"Saved: {args.output_path}")
    logger.info("=" * 60)
    logger.info("Step2 Finished")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
