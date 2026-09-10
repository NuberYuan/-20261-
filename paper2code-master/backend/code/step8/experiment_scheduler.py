"""
Step8.3: 实验调度器 (Experiment Scheduler)

只负责"决定哪些实验能跑、哪些不能跑、为什么"，不负责执行（Step8.md 第七节）。

输入：解析后的实验列表 + 资源检查结果
输出：execution_plan（每个实验 decision=RUN/SKIP 及原因）

状态语义（Step8.md 第九节）：
    PLANNED   已计划，尚未执行
    RUN       决策为可运行（执行前的决策态）
    SKIP      决策为跳过（未执行，不是代码错误）
    RUNNING   正在执行
    SUCCESS   执行成功
    FAILED    执行失败（代码/环境问题）

跳过原因的优先级（ResourceChecker.check_all 的顺序）：
    dataset_unresolved > dataset_unavailable > gpu_unavailable
    > insufficient_cpu > insufficient_ram > insufficient_disk
    > python_version_too_low > missing_dependencies

软性警告（不 SKIP，仍 RUN，记入 warnings）：
    large_dataset            数据集总容量过大（容量不代表不可跑）
    insufficient_gpu_memory  粗略显存估算不足（估算非精确，不硬跳）

注：除 dataset 外，其余维度（GPU/CPU/RAM/Disk/Python/Dependencies）仅在实验声明了
对应需求时才可能不满足；GPU 非必需，未声明 GPU 需求时无 GPU 也能 RUN。

除决策外，本模块还把 Step7 的 execution_spec 解析成"入口 + 命令"预填进每个 RUN 实验
的 execution 段（build_command），使 execution_plan.yaml 成为 Runner 的单一执行来源
（Runner 不再自己猜入口/命令）。
"""

from step8.experiment_runner import build_command


class ExperimentScheduler:
    """根据资源检查结果，为每个实验做 RUN/SKIP 决策。"""

    def build_plan(self, experiments: list, checker, spec: dict = None,
                   config_path=None, project=None) -> dict:
        plan = []
        for exp in experiments:
            checks = checker.check_all(exp)
            reasons = [
                {"reason": r["reason"], "detail": r.get("detail")}
                for r in checks.values()
                if not r.get("sufficient", True) and r.get("reason")
            ]
            warnings = [
                {"reason": w, "check": name}
                for name, r in checks.items()
                for w in (r.get("warnings") or [])
            ]

            decision = "SKIP" if reasons else "RUN"

            # 仅 RUN 实验需要入口+命令；解析失败记入 error，Runner 执行时据此 FAILED
            execution = None
            if decision == "RUN":
                try:
                    entrypoint, command = build_command(spec, exp["run_index"], config_path, project)
                    execution = {"entrypoint": entrypoint, "command": command}
                except Exception as e:  # noqa: BLE001 - 入口解析失败不阻断调度，交由执行阶段处理
                    execution = {"error": str(e)}

            plan.append({
                "experiment_id": exp["id"],
                "run_index": exp["run_index"],
                "model": exp["model"],
                "mode": exp["mode"],
                "dataset_version": exp["dataset_version"],
                "data_source": exp.get("data_source", "external_dataset"),
                "acquisition": exp.get("acquisition", "local_path"),
                "train_ratio": exp["train_ratio"],
                "decision": decision,
                "status": "PLANNED",
                "skip_reasons": reasons,
                "warnings": warnings,
                "resource_check": checks,
                "execution": execution,
            })
        return {
            "environment": checker.environment,
            "summary": self._summarize(plan),
            "experiments": plan,
        }

    @staticmethod
    def _summarize(plan: list) -> dict:
        run = sum(1 for e in plan if e["decision"] == "RUN")
        skip = sum(1 for e in plan if e["decision"] == "SKIP")
        warnings = sum(1 for e in plan if e.get("warnings"))
        return {"total": len(plan), "run": run, "skip": skip, "warnings": warnings}
