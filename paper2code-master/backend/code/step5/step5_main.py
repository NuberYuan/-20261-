"""
Step5 总入口: 串联 5.1 → 5.8

Usage:
    cd code
    python -m step5.step5_main --paper AgriPath
    python -m step5.step5_main --all
    python -m step5.step5_main --all --skip-llm    # 跳过所有 LLM 调用（仅规则提取）
    python -m step5.step5_main --paper AgriPath --force   # 忽略已生成的子 yaml，强制重跑
"""

import sys
import os
import argparse
import yaml
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from step5.base_analyzer import OUTPUT_ROOT
from logger import logger

from step5.step5_1_goal import GoalAnalyzer
from step5.step5_2_tech import TechAnalyzer
from step5.step5_3_data import DatasetAnalyzer
from step5.step5_4_model import ModelAnalyzer
from step5.step5_5_exp import ExperimentAnalyzer
from step5.step5_6_struct import StructureAnalyzer
from step5.step5_7_train import TrainConfigAnalyzer

# 5.1~5.5、5.7 彼此独立（都只读 key_information.json，各自产出独立 yaml），可并行；
# 5.6 需读取 5.1 的 reproduction_goal.yaml 与 5.4 的 model_design.yaml，存在数据依赖，
#     必须放到并行块之后串行执行（否则会读到不存在的文件或与写入竞态）；
# 5.8 依赖前 7 步结果，同样串行。
# (分析器类, 输出文件名)
PARALLEL_SUBSTEPS = [
    (GoalAnalyzer, "reproduction_goal.yaml"),
    (TechAnalyzer, "technology.yaml"),
    (DatasetAnalyzer, "dataset.yaml"),
    (ModelAnalyzer, "model_design.yaml"),
    (ExperimentAnalyzer, "experiment_design.yaml"),
    (TrainConfigAnalyzer, "training_config.yaml"),
]

# 5.6 依赖 5.1/5.4 的输出，串行执行
STRUCT_SUBSTEP = (StructureAnalyzer, "project_structure.yaml")


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
    """对单篇论文运行完整 Step5"""
    logger.info(f"{'=' * 50}\n  Step5: {paper_name}\n{'=' * 50}")

    max_workers = min(len(PARALLEL_SUBSTEPS), 6)
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [
            ex.submit(_run_one, paper_name, skip_llm, force, cls, fn)
            for cls, fn in PARALLEL_SUBSTEPS
        ]
        results = [f.result() for f in futures]
    r1, r2, r3, r4, r5, r7 = results

    # 5.6 串行：依赖 5.1(reproduction_goal.yaml) 与 5.4(model_design.yaml) 的输出
    r6 = _run_one(paper_name, skip_llm, force, *STRUCT_SUBSTEP)

    # ---- project_analysis.yaml 聚合 ----
    analysis = {
        "paper_name": paper_name,
        "generated_at": datetime.now().isoformat(),
        "reproduction_goal": {
            k: v for k, v in r1.items() if not k.startswith("_")
        },
        "technology": {
            k: v for k, v in r2.items() if not k.startswith("_")
        },
        "dataset": {
            k: v for k, v in r3.items() if not k.startswith("_")
        },
        "model_design": {
            k: v for k, v in r4.items() if not k.startswith("_")
        },
        "experiment": {
            k: v for k, v in r5.items() if not k.startswith("_")
        },
        "project_structure": {
            k: v for k, v in r6.items() if not k.startswith("_")
        },
        "training_config": {
            k: v for k, v in r7.items() if not k.startswith("_")
        },
    }
    out_dir = OUTPUT_ROOT / paper_name / "step5"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "project_analysis.yaml", "w", encoding="utf-8") as f:
        yaml.dump(analysis, f, allow_unicode=True, default_flow_style=False,
                  sort_keys=False, indent=2)
    logger.info("  project_analysis.yaml saved")

    # 5.8 LLM 规范细化（依赖 5.1~5.7 结果，串行）
    if not skip_llm:
        from step5.step5_8_spec import SpecAnalyzer
        spec = SpecAnalyzer(paper_name)
        spec_file = spec.output_dir / "optimized_project_spec.yaml"
        if spec_file.exists() and not force:
            logger.info("  [跳过] optimized_project_spec.yaml 已存在")
        else:
            r8 = spec.analyze()
            spec.save_yaml(r8, "optimized_project_spec.yaml")
    else:
        logger.info("  跳过 5.8 (--skip-llm)")

    # 汇总输出文件清单
    yaml_files = sorted(out_dir.glob("*.yaml"))
    logger.info(f"  Step5 完成: {paper_name}")
    logger.info(f"  输出目录: {out_dir}")
    logger.info(f"  生成文件: {[f.name for f in yaml_files]}")

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Step5: 论文复现项目分析"
    )
    parser.add_argument("--paper", type=str, default="plant",
                        help="单篇论文名称 (data/output/ 下的目录名)")
    parser.add_argument("--all", action="store_true",
                        help="处理所有论文")
    parser.add_argument("--skip-llm", action="store_true",
                        help="跳过所有 LLM 调用（5.1~5.7 Layer2 补全 + 5.8 细化，仅规则提取）")
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
            run_paper(p, skip_llm=args.skip_llm, force=args.force)
            success += 1
        except Exception as e:
            logger.error(f"{p} 失败: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    logger.info(f"Step5 全部完成: {success} 成功, {failed} 失败")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
