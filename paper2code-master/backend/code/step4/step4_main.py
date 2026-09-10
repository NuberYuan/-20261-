"""
Step4 总入口: 串联 4.1 → 4.2 → 4.3 → 4.4

功能: 将 Step3 的论文事实转化为系统需求定义
输入: data/output/{paper}/step3/key_information.json
输出: data/output/{paper}/step4/*.yaml

Usage:
    cd code
    python -m step4.step4_main --paper AgriPath
    python -m step4.step4_main --all
    python -m step4.step4_main --all --skip-llm    # 仅规则提取
    python -m step4.step4_main --paper AgriPath --force   # 忽略已生成的子 yaml，强制重跑
"""

import sys
import os
import argparse
import yaml
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from step4.base_requirement_analyzer import OUTPUT_ROOT
from logger import logger

from step4.step4_1_functional import FunctionalAnalyzer
from step4.step4_2_dataflow import DataFlowAnalyzer
from step4.step4_3_acceptance import AcceptanceAnalyzer
from step4.step4_4_gap import GapAnalyzer

# 4.1 先串行执行，产出 functional_requirements.yaml；
# 4.2/4.3/4.4 并行，但需读取 4.1 的 FR id（用于 related_fr 引用），
# 因此把 4.1 单独放到并行块之前。
FUNCTIONAL_SUBSTEP = (FunctionalAnalyzer, "functional_requirements.yaml")
PARALLEL_SUBSTEPS = [
    (DataFlowAnalyzer, "data_flow.yaml"),
    (AcceptanceAnalyzer, "acceptance_criteria.yaml"),
    (GapAnalyzer, "gap_analysis.yaml"),
]


def _run_one(paper_name, skip_llm, force, analyzer_cls, filename):
    """运行单个子步；输出已存在且非 force 时直接复用（子产物级断点续跑）。"""
    analyzer = analyzer_cls(paper_name)
    out_path = analyzer.output_dir / filename
    if out_path.exists() and not force:
        logger.info(f"  [跳过] {filename} 已存在")
        return analyzer.load_yaml(filename)
    result = analyzer.analyze(skip_llm=skip_llm)
    analyzer.save_yaml(result, filename)
    return result


def run_paper(paper_name: str, skip_llm: bool = False, force: bool = False):
    """对单篇论文运行完整 Step4 需求分析"""
    logger.info(f"{'=' * 50}\n  Step4: 需求分析 — {paper_name}\n{'=' * 50}")

    # 检查前置依赖: key_information.json 必须存在
    ki_path = OUTPUT_ROOT / paper_name / "step3" / "key_information.json"
    if not ki_path.exists():
        logger.error(f"  前置文件不存在: {ki_path}")
        logger.error(f"  请先运行 Step3 生成 key_information.json")
        return False

    # 创建输出目录
    out_dir = OUTPUT_ROOT / paper_name / "step4"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 4.1 先串行（后续子步依赖其 FR id）
    r1 = _run_one(paper_name, skip_llm, force, *FUNCTIONAL_SUBSTEP)

    max_workers = min(len(PARALLEL_SUBSTEPS), 3)
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [
            ex.submit(_run_one, paper_name, skip_llm, force, cls, fn)
            for cls, fn in PARALLEL_SUBSTEPS
        ]
        results = [f.result() for f in futures]
    r2, r3, r4 = results

    # ---- 汇总: requirements_summary.yaml ----
    summary = _build_summary(paper_name, r1, r2, r3, r4)
    summary_path = out_dir / "requirements_summary.yaml"
    with open(summary_path, "w", encoding="utf-8") as f:
        yaml.dump(summary, f, allow_unicode=True, default_flow_style=False,
                  sort_keys=False, indent=2)
    logger.info("  requirements_summary.yaml saved")

    # ---- 输出清单 ----
    yaml_files = sorted(out_dir.glob("*.yaml"))
    logger.info(f"  Step4 完成: {paper_name}")
    logger.info(f"  输出目录: {out_dir}")
    logger.info(f"  生成文件: {[f.name for f in yaml_files]}")

    return True


def _build_summary(paper_name: str, r1: dict, r2: dict, r3: dict,
                   r4: dict) -> dict:
    """聚合 4.1~4.4 为人类可读的摘要"""
    return {
        "paper_name": paper_name,
        "generated_at": datetime.now().isoformat(),
        "functional_requirements": {
            "total": r1.get("summary", {}).get("total", 0),
            "categories": r1.get("summary", {}).get("categories", {}),
        },
        "data_flow": {
            "pipeline_stages": len(r2.get("pipeline", [])),
            "interfaces": len(r2.get("interfaces", [])),
        },
        "acceptance_criteria": {
            "accuracy_targets": (
                r3.get("accuracy_targets", {}).get("primary", {}).get("metric",
                                                                      "N/A")
            ),
            "functional_tests": len(r3.get("functional_tests", [])),
            "edge_cases": len(r3.get("edge_cases", [])),
            "suggested_validation_runs": (
                r3.get("reproducibility", {}).get("suggested_validation_runs", 3)
            ),
        },
        "gap_analysis": {
            "total_gaps": r4.get("summary", {}).get("total", 0),
            "by_severity": r4.get("summary", {}).get("by_severity", {}),
            "resolved_assumptions": len(r4.get("resolved_assumptions", [])),
        },
    }


def main():
    parser = argparse.ArgumentParser(
        description="Step4: 论文需求分析 —— 将论文事实转化为系统需求定义"
    )
    parser.add_argument("--paper", type=str, default="plant",
                        help="单篇论文名称 (data/output/ 下的目录名)")
    parser.add_argument("--all", action="store_true",
                        help="处理所有论文")
    parser.add_argument("--skip-llm", action="store_true",
                        help="跳过 LLM 补全（仅规则提取）")
    parser.add_argument("--force", action="store_true",
                        help="忽略已生成的子 yaml，强制重跑（默认跳过已存在的）")
    args = parser.parse_args()

    if args.all:
        papers = []
        for d in OUTPUT_ROOT.iterdir():
            if d.is_dir() and (d / "step3" / "key_information.json").exists():
                papers.append(d.name)
        logger.info(f"发现 {len(papers)} 篇论文: {papers}")
    elif args.paper:
        papers = [args.paper]
    else:
        parser.print_help()
        return

    success = 0
    failed = 0
    for p in papers:
        try:
            if run_paper(p, skip_llm=args.skip_llm, force=args.force):
                success += 1
            else:
                failed += 1
        except Exception as e:
            logger.error(f"{p} 失败: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    logger.info(f"Step4 全部完成: {success} 成功, {failed} 失败")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
