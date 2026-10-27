"""
Step3

structure_detect_paper.json → key_information.json

整合:
  1. 文本信息提取 (InformationExtractor + LLM)
  2. 表格分析 (TableAnalyzer: LLM 解析表格内容)
  3. 图片分析 (FigureAnalyzer: base64 → 视觉 LLM 直接分析)

三路互不依赖，由 execute_step3 并行调度后合并。各分析器内部逻辑不变。

Usage:
    python step3/step3_extract_information.py
    python step3/step3_extract_information.py --in structure.json --out key_info.json
    python step3/step3_extract_information.py --skip-tables --skip-figures

Author: Paper2Code
"""

import json
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, wait
from typing import Callable, Optional

from step3.information_extractor import InformationExtractor
from step3.figure_analyzer import FigureAnalyzer
from step3.table_analyzer import TableAnalyzer
from config import STRUCTURE_DETECT, KEY_INFO_JSON
from logger import logger


# Step3 的 prompts 目录
PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"


def execute_step3(
    paper: dict,
    *,
    skip_tables: bool = False,
    skip_figures: bool = False,
    extract_text: Callable[[dict], dict],
    analyze_tables: Callable[[dict], dict],
    analyze_figures: Callable[[dict], Optional[list]],
) -> dict:
    """并行调度文本 / 表格 / 图片三路，合并为与串行版相同的 result dict。

    skip_tables / skip_figures 时不调用对应分析器，也不写入对应键。
    analyze_figures 返回 None 表示 prompt 缺失等，不写入 image_analysis。
    """
    with ThreadPoolExecutor(max_workers=3) as ex:
        text_f = ex.submit(extract_text, paper)
        table_f = None if skip_tables else ex.submit(analyze_tables, paper)
        figure_f = None if skip_figures else ex.submit(analyze_figures, paper)

        pending = [text_f]
        if table_f is not None:
            pending.append(table_f)
        if figure_f is not None:
            pending.append(figure_f)
        wait(pending)

        result = text_f.result()
        paper_figures = result.get("figures", [])

        if skip_tables:
            logger.info("跳过表格分析 (--skip-tables)")
        else:
            result["table_analysis"] = table_f.result()

        if skip_figures:
            logger.info("跳过图片分析 (--skip-figures)")
        else:
            figure_result = figure_f.result()
            if figure_result is not None:
                result["image_analysis"] = figure_result

        result["paper_figures"] = paper_figures
        return result


def _extract_text(paper: dict, prompt: str) -> dict:
    extractor = InformationExtractor(prompt)
    result = extractor.extract(paper)

    logger.info("=" * 40)
    logger.info("InformationExtractor Result:")
    logger.info(json.dumps(result, ensure_ascii=False, indent=2)[:2000])
    logger.info("=" * 40)
    return result


def _analyze_tables(paper: dict) -> dict:
    tables = paper.get("tables", [])
    if tables:
        title = paper.get("paragraphs", [{}])[0].get("text", "") if paper.get("paragraphs") else ""
        paper_context = title[:500]

        table_analyzer = TableAnalyzer()
        table_result = table_analyzer.analyze(tables, paper_context)

        logger.info(f"表格分析: {table_result.get('tables_analyzed', 0)} 个表格")
        return table_result

    logger.info("无表格数据，跳过表格分析")
    return {"tables_analyzed": 0, "table_results": []}


def _analyze_figures(paper: dict, output_path: str) -> Optional[list]:
    figure_analyzer = FigureAnalyzer()

    images = paper.get("images", [])
    valid_images = [img for img in images if img.get("path")]

    if valid_images:
        def _analyze_one(image: dict) -> dict:
            image_path = image.get("path")
            logger.info(f"分析图片: {image_path}")
            try:
                output = figure_analyzer.analyze(image_path)
            except Exception as e:
                logger.error(f"图片分析失败: {image_path} — {e}")
                output = {
                    "type": "other",
                    "description": f"analysis failed ({e})",
                    "important_components": [],
                    "code_relevant_information": "",
                }
            output["id"] = image.get("id")
            output["page"] = image.get("page")
            return output

        max_workers = min(len(valid_images), 4)
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            return list(ex.map(_analyze_one, valid_images))

    return []


def main():
    parser = argparse.ArgumentParser(
        description="Step3: 论文关键信息提取"
    )
    parser.add_argument("--in", dest="input_path", type=str, default=STRUCTURE_DETECT,
                        help="输入 JSON 路径 (structure_detect_paper.json)")
    parser.add_argument("--out", dest="output_path", type=str, default=KEY_INFO_JSON,
                        help="输出 JSON 路径 (key_information.json)")
    parser.add_argument("--skip-figures", action="store_true",
                        help="跳过图片分析")
    parser.add_argument("--skip-tables", action="store_true",
                        help="跳过表格分析")
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("Step3: Information Extraction")
    logger.info(f"  Input : {args.input_path}")
    logger.info(f"  Output: {args.output_path}")
    logger.info("=" * 60)

    with open(args.input_path, "r", encoding="utf-8") as f:
        paper = json.load(f)

    step3_prompt_path = PROMPTS_DIR / "step3_prompt.txt"
    if not step3_prompt_path.exists():
        logger.error(f"Prompt file not found: {step3_prompt_path}")
        return

    with open(step3_prompt_path, "r", encoding="utf-8") as f:
        prompt = f.read()

    result = execute_step3(
        paper,
        skip_tables=args.skip_tables,
        skip_figures=args.skip_figures,
        extract_text=lambda p: _extract_text(p, prompt),
        analyze_tables=_analyze_tables,
        analyze_figures=lambda p: _analyze_figures(p, args.output_path),
    )

    Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    logger.info(f"Saved: {args.output_path}")
    logger.info("Step3 finished")


if __name__ == "__main__":
    main()
