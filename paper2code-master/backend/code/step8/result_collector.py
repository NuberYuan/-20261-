"""
Step8.5: 结果汇总器 (Result Collector)

把执行计划与运行结果聚合成 execution_report.yaml（Step8.md 第十二节）。

只收集原始执行事实，不做结果分析（指标提取/对比不在此步，留给后续步骤）：
    - 是否执行 / 是否成功（SUCCESS / FAILED / SKIP）
    - exit_code / duration_seconds
    - 日志路径（stdout / stderr）
    - 实验生成的结果文件（result_files）

关键：SKIP（根本没执行）与 FAILED（执行了但失败）严格区分（第十节）。
"""


class ResultCollector:
    """聚合执行结果，产出最终报告结构。"""

    def collect(self, plan: dict, run_results: list) -> dict:
        """聚合执行结果。状态语义（严格区分，绝不混用）：
            SKIP    根本没执行（资源/环境不满足，reason=dataset_*/insufficient_*）
            SUCCESS 已执行且成功
            FAILED  已执行但失败（reason=runtime_error/exit_code/timeout）
            FAILED + reason=not_executed  决定执行却未产生结果（编排异常，非 SKIP）
        """
        experiments = plan["experiments"]
        by_id = {r["experiment_id"]: r for r in run_results}

        success, failed, skipped = [], [], []
        for e in experiments:
            if e["decision"] == "SKIP":
                e["status"] = "SKIP"
                primary = e["skip_reasons"][0]["reason"] if e["skip_reasons"] else None
                skipped.append({
                    "experiment_id": e["experiment_id"],
                    "status": "SKIP",
                    "reason": primary,
                })
                continue

            rr = by_id.get(e["experiment_id"])
            if rr is None:
                # 决定执行却未产生结果记录（编排异常）：归入 failed，reason 明确 not_executed，
                # 绝不与 SKIP（资源/环境不满足）混为一谈。
                e["status"] = "FAILED"
                failed.append({
                    "experiment_id": e["experiment_id"],
                    "status": "FAILED",
                    "reason": "not_executed",
                })
                continue

            exec_ = rr.get("execution") or {}
            status = exec_.get("status") or rr.get("status")
            if status == "SUCCESS":
                e["status"] = "SUCCESS"
                success.append({
                    "experiment_id": e["experiment_id"],
                    "status": "SUCCESS",
                    "warnings": e.get("warnings", []),
                    "exit_code": exec_.get("exit_code"),
                    "duration_seconds": exec_.get("duration_seconds"),
                    "logs": rr.get("logs", {}),
                    "result_files": rr.get("result_files", []),
                    "output_dir": rr.get("output_dir"),
                })
            else:
                e["status"] = "FAILED"
                failed.append({
                    "experiment_id": e["experiment_id"],
                    "status": "FAILED",
                    "reason": exec_.get("error") or f"exit_code={exec_.get('exit_code')}",
                    "exit_code": exec_.get("exit_code"),
                    "duration_seconds": exec_.get("duration_seconds"),
                    "logs": rr.get("logs", {}),
                    "result_files": rr.get("result_files", []),
                })

        total = len(experiments)
        return {
            "execution": {
                "total": total,
                "planned": total,
                "success": len(success),
                "failed": len(failed),
                "skipped": len(skipped),
            },
            "success": success,
            "failed": failed,
            "skipped": skipped,
        }
