"""
Step8 总入口: 实验执行管理

读取 Step7 生成的 generated_project，先做实验可行性分析（哪些能跑、哪些不能跑、为什么），
再对 decision=RUN 的实验实际执行并保存结果。Step8 是管线终点，不做结果分析
（结果准确性、指标对比、论文 vs 复现对比均不在本管线内）。

流程:
    generated_project/config/config.yaml
        ↓ 实验列表解析 (ExperimentAnalyzer)
        ↓ 资源检查 (ResourceChecker, 程序检测真实环境)
        ↓ 调度决策 (ExperimentScheduler: RUN/SKIP)
    execution_plan.yaml
        ↓ 只执行 decision=RUN 的实验 (ExperimentRunner)
    execution_report.yaml

Usage:
    cd code
    python -m step8.step8_main --paper plant
    python -m step8.step8_main --paper plant --plan-only   # 只生成计划，不实际执行
    python -m step8.step8_main --all
"""

import sys
import os
import re
import argparse
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml

from config import get_paper_output_dir
from logger import logger

from step8.experiment_analyzer import ExperimentAnalyzer
from step8.resource_checker import ResourceChecker
from step8.experiment_scheduler import ExperimentScheduler
from step8.experiment_runner import ExperimentRunner
from step8.result_collector import ResultCollector


class Step8Runner:
    """Step8 总控制器。"""

    def __init__(self, paper_name: str, plan_only: bool = False, timeout: int = 3600):
        self.paper_name = paper_name
        self.plan_only = plan_only
        self.timeout = timeout
        self.paper_dir = get_paper_output_dir(paper_name)
        self.generated_project = self.paper_dir / "step7" / "generated_project"
        self.output_dir = self.paper_dir / "step8"
        self.experiments_dir = self.output_dir / "experiments"

    def run(self) -> bool:
        logger.info(f"{'=' * 50}\n  Step8: 实验执行管理 — {self.paper_name}\n{'=' * 50}")

        # 0. 前置检查：generated_project 必须存在
        analyzer = ExperimentAnalyzer(self.generated_project)
        try:
            config_path = analyzer.find_config()
        except FileNotFoundError as e:
            logger.error(f"  {e}，请先运行 Step7")
            return False

        self.output_dir.mkdir(parents=True, exist_ok=True)
        config = analyzer.load_config()

        # 1. 解析实验列表
        data_source, acquisition = self._read_data_fields()
        experiments = analyzer.analyze(config, data_source=data_source, acquisition=acquisition)
        if not experiments:
            logger.error("  experiment.runs 为空")
            return False
        logger.info(f"  解析到 {len(experiments)} 个实验")

        # 1.5 把 requirements.txt 声明的依赖合并进每个实验，让依赖检查真实生效。
        # 否则 config.system.dependencies 未声明时，缺依赖会到运行期才崩（ModuleNotFoundError）。
        req_deps = self._read_requirements_deps()
        if req_deps:
            for e in experiments:
                declared = set(e.get("dependencies") or [])
                e["dependencies"] = sorted(declared | req_deps)
            logger.info(f"  依赖检查: 合并 requirements.txt {len(req_deps)} 个依赖")

        # 2. 资源检查（程序检测真实环境）
        checker = ResourceChecker()
        env = checker.environment
        logger.info(
            f"  环境: Python={env['python_version']}, CPU={env['cpu_count']} 核, "
            f"RAM={env['ram_total_gb']}GB, Disk可用={env['disk_free_gb']}GB, "
            f"GPU={env['gpu_available']} (显存 {env['gpu_memory_gb']}GB), "
            f"torch={env['torch_version']}"
        )

        # 3. 读取执行配置（Step7 产出）后调度决策：计划里预填每个 RUN 实验的入口+命令
        exec_spec = self._read_execution_spec()
        plan = ExperimentScheduler().build_plan(
            experiments, checker,
            spec=exec_spec, config_path=config_path, project=self.generated_project,
        )
        self._save_yaml(plan, "execution_plan.yaml")
        s = plan["summary"]
        logger.info(
            f"  调度: {s['run']} RUN, {s['skip']} SKIP, "
            f"{s.get('warnings', 0)} 警告 (共 {s['total']})"
        )

        # 4. 执行（仅 decision=RUN；--plan-only 时跳过）
        if self.plan_only:
            logger.info("  --plan-only：只生成执行计划，不实际执行实验")
            run_results = []
        else:
            run_results = self._execute_run_experiments(plan, config_path, exec_spec)

        # 5. 汇总报告（--plan-only 不产生 execution_report）
        if not self.plan_only:
            report = ResultCollector().collect(plan, run_results)
            self._save_yaml(report, "execution_report.yaml")
            ex = report["execution"]
            logger.info(
                f"  Step8 完成: {ex['success']} 成功, {ex['failed']} 失败, "
                f"{ex['skipped']} 跳过 (共 {ex['total']})"
            )

        logger.info(f"  输出: {self.output_dir}")
        return True

    def _execute_run_experiments(self, plan: dict, config_path: Path, spec: dict) -> list:
        runner = ExperimentRunner(
            self.generated_project, self.experiments_dir, self.timeout, spec=spec
        )
        results = []
        for e in plan["experiments"]:
            if e["decision"] != "RUN":
                continue
            logger.info(f"  执行 {e['experiment_id']} (run_index={e['run_index']}) ...")
            rr = runner.run(e, config_path)
            results.append(rr)
            logger.info(f"    -> {rr['execution']['status']}")
        return results

    def _read_execution_spec(self) -> dict:
        """读取 Step7 产出的 execution_spec.yaml（入口 + 命令模板）。

        缺失或解析失败时返回 {}，ExperimentRunner 回退默认入口/命令（main.py/run.py + --run/--config）。
        """
        spec_path = self.paper_dir / "step7" / "execution_spec.yaml"
        if not spec_path.exists():
            logger.warning("  未找到 execution_spec.yaml，回退默认入口/命令")
            return {}
        try:
            spec = yaml.safe_load(spec_path.read_text(encoding="utf-8")) or {}
            return spec if isinstance(spec, dict) else {}
        except Exception:
            logger.warning("  execution_spec.yaml 解析失败，回退默认入口/命令")
            return {}

    def _read_requirements_deps(self) -> set:
        """读取 generated_project/requirements.txt 的依赖包名（发行名，未去重解析导入名）。

        行格式兼容 `numpy`、`torch>=2.0`、`scikit-learn==1.5`、`pillow` 等；
        注释（#）与空行跳过。解析失败返回空集合（不阻断调度）。
        """
        req_path = self.generated_project / "requirements.txt"
        if not req_path.exists():
            logger.warning("  未找到 requirements.txt，跳过依赖检查")
            return set()
        deps = set()
        try:
            for line in req_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                # 去掉版本/环境标记：numpy>=2.0、scikit-learn==1.5、foo[extra] 等
                pkg = re.split(r"[<>=!~\s\[;]", line, maxsplit=1)[0].strip()
                if pkg:
                    deps.add(pkg)
        except Exception as e:  # noqa: BLE001 - 解析失败仅告警，不中断
            logger.warning(f"  requirements.txt 解析失败，跳过依赖检查: {e}")
            return set()
        return deps

    def _read_data_fields(self) -> tuple:
        """从 Step5 dataset.yaml 读 data_source + acquisition，缺省/非法回退保守默认。

        data_source  回退 external_dataset（保守，避免漏检数据集）
        acquisition  回退 local_path（保守，视为需用户预置本地路径）
        """
        ds_path = self.paper_dir / "step5" / "dataset.yaml"
        if not ds_path.exists():
            return "external_dataset", "local_path"
        try:
            ds = yaml.safe_load(ds_path.read_text(encoding="utf-8")) or {}
            data_source = str(ds.get("data_source") or "external_dataset").strip().lower()
            if data_source not in {"external_dataset", "synthetic", "environment", "none"}:
                data_source = "external_dataset"
            acquisition = str(ds.get("acquisition") or "").strip().lower()
            if data_source != "external_dataset":
                acquisition = "n/a"
            elif acquisition not in {"local_path", "auto"}:
                acquisition = "local_path"
            return data_source, acquisition
        except Exception:
            return "external_dataset", "local_path"

    def _save_yaml(self, obj: dict, name: str):
        path = self.output_dir / name
        path.write_text(
            yaml.safe_dump(obj, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )
        logger.info(f"  已写 {path}")


def main():
    parser = argparse.ArgumentParser(
        description="Step8: 实验执行管理 —— 判断哪些实验能跑并执行 RUN 实验"
    )
    parser.add_argument("--paper", type=str, default="plant", help="单篇论文名称")
    parser.add_argument("--all", action="store_true", help="处理所有论文")
    parser.add_argument("--plan-only", action="store_true",
                        help="只生成执行计划，不实际运行实验")
    parser.add_argument("--timeout", type=int, default=3600,
                        help="单个实验超时（秒）")
    args = parser.parse_args()

    if args.all:
        root = get_paper_output_dir("")
        papers = []
        for d in Path(root).iterdir():
            cfg_dir = d / "step7" / "generated_project" / "config"
            if d.is_dir() and (
                (cfg_dir / "config.yaml").exists() or (cfg_dir / "config.yml").exists()
            ):
                papers.append(d.name)
        logger.info(f"发现 {len(papers)} 篇可执行论文: {papers}")
    elif args.paper:
        papers = [args.paper]
    else:
        parser.print_help()
        return

    success = 0
    failed = 0
    for p in papers:
        if Step8Runner(p, plan_only=args.plan_only, timeout=args.timeout).run():
            success += 1
        else:
            failed += 1

    logger.info(f"Step8 全部完成: {success} 成功, {failed} 失败")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
