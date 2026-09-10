"""
Step6 总入口: 项目整合与代码生成准备

将 Step4（需求分析）与 Step5（工程设计）横向对齐，规则驱动地生成
代码生成 Agent 可直接消费的 project_brief.yaml。

流程:
    Step4 YAML + Step5 YAML
        ↓ 一致性检查（规则驱动）
    冲突 / 缺失 / 假设
        ↓ 项目整合
    project_brief.yaml

Usage:
    cd code
    python -m step6.step6_main --paper plant
    python -m step6.step6_main --all
"""

import sys
import os
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from step6.step6_summarizer import ProjectBriefIntegrator, OUTPUT_ROOT
from logger import logger


def run_paper(paper_name: str) -> bool:
    """对单篇论文运行 Step6"""
    logger.info(f"{'=' * 50}\n  Step6: 项目整合 — {paper_name}\n{'=' * 50}")

    integrator = ProjectBriefIntegrator(paper_name)

    if not integrator.step4_yamls and not integrator.step5_yamls:
        logger.error(f"  Step4 和 Step5 输出均缺失，请先运行 Step4 和 Step5")
        return False

    try:
        brief = integrator.integrate()

        integrator.save_yaml(brief, "project_brief.yaml")

        logger.info(f"  Step6 完成: {paper_name}")
        logger.info(f"  输出: {integrator.output_dir / 'project_brief.yaml'}")

        # 简单统计
        report = brief.get("consistency_report", {}).get("summary", {})
        req_count = len(brief.get("requirements_digest", {}).get("core_requirements", []))
        trace_count = len(brief.get("traceability", []))
        plan_files = sum(
            len(phase.get("files", []))
            for phase in brief.get("code_generation_plan", {}).get("generation_order", [])
        )
        logger.info(f"  摘要: {req_count} 条核心需求, "
                    f"{trace_count} 条追溯, {plan_files} 个文件生成计划")
        logger.info(f"  一致性: {report.get('checks_run', 0)} 项检查, "
                    f"{report.get('conflicts', 0)} 冲突, "
                    f"{report.get('gaps', 0)} 缺失, "
                    f"{report.get('assumptions', 0)} 假设")

        return True

    except Exception as e:
        logger.error(f"  {paper_name} 失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Step6: 项目整合与代码生成准备 —— 规则驱动对齐需求与设计"
    )
    parser.add_argument("--paper", type=str, default="plant",
                        help="单篇论文名称")
    parser.add_argument("--all", action="store_true",
                        help="处理所有论文")
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
        if run_paper(p):
            success += 1
        else:
            failed += 1

    logger.info(f"Step6 全部完成: {success} 成功, {failed} 失败")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
