"""Step5.1: 复现目标分析 → reproduction_goal.yaml"""
import json
from step5.base_analyzer import BaseAnalyzer


class GoalAnalyzer(BaseAnalyzer):

    def extract_by_rules(self) -> dict:
        ki = self.key_info
        meth = ki.get("methodology", {})
        impl = ki.get("implementation_details", {})
        ds = ki.get("dataset", {})

        # 论文类型
        paper_type = self._classify(ki)

        # 任务类型
        task = self._task_type(ki)

        # 是否训练
        trainable = self._has_training(impl)

        # 复现目标
        targets = self._targets(paper_type, trainable)

        arch = meth.get("model_architecture", {})
        if not isinstance(arch, dict):
            arch = {}

        return {
            "paper_name": self.paper_name,
            # 论文核心信息（之前丢失的）
            "title": self._pick(ki.get("title")),
            "research_problem": self._pick(ki.get("research_problem")),
            "motivation": self._pick(ki.get("motivation")),
            "main_contribution": self._pick(
                " ".join(ki.get("main_contribution", []))
                if isinstance(ki.get("main_contribution"), list)
                else ki.get("main_contribution"),
            ),
            # 类型与任务
            "paper_type": paper_type,
            "task": task,
            "input": self._pick(arch.get("input")),
            "output": self._pick(arch.get("output")),
            "num_classes": self._pick(ds.get("number_of_classes")),
            "reproduction_targets": targets,
            "is_trainable": trainable,
        }

    def _classify(self, ki: dict) -> str:
        impl = ki.get("implementation_details", {})
        meth = ki.get("methodology", {})
        ds = ki.get("dataset", {})
        title = (ki.get("title") or "").lower()
        research = (ki.get("research_problem") or "").lower()
        contrib = ki.get("main_contribution", "")
        if isinstance(contrib, list):
            contrib = " ".join(contrib)
        contrib = contrib.lower()

        has_training = any(
            impl.get(k, "NOT_FOUND") != "NOT_FOUND"
            for k in ["optimizer", "learning_rate", "epoch", "loss_function"]
        )
        # 无数据集：dataset_name 为 NOT_FOUND（纯算法/理论推导论文）
        has_dataset = ds.get("dataset_name", "NOT_FOUND") not in ("NOT_FOUND", "", None)
        method_name = meth.get("method_name", "NOT_FOUND")
        has_new_model = method_name != "NOT_FOUND" and bool(method_name)

        code_kw = ["code", "software", "defect", "bug", "static analysis"]
        is_code = any(k in title + " " + research for k in code_kw)

        if is_code:
            return "empirical_comparison"
        if has_new_model and has_training:
            return "model_proposing"
        # 纯算法论文：既无训练也无数据集（如规则/图算法、理论推导）
        if not has_training and not has_dataset:
            return "algorithm"
        return "empirical_comparison"

    def _task_type(self, ki: dict) -> str:
        import re
        text = json.dumps(ki).lower()

        def has(w):
            return bool(re.search(r'\b' + re.escape(w) + r'\b', text))

        # 架构中实际使用了 LSTM/Transformer 的（值不是 NOT_FOUND）
        arch = ki.get("methodology", {}).get("model_architecture", {})
        if isinstance(arch, dict):
            lstm_val = str(arch.get("lstm", "NOT_FOUND")).lower()
            transformer_val = str(arch.get("transformer", "NOT_FOUND")).lower()
            actually_uses_lstm = lstm_val not in ("not_found", "") and len(lstm_val) > 5
            actually_uses_transformer = transformer_val not in ("not_found", "") and len(transformer_val) > 5
        else:
            actually_uses_lstm = False
            actually_uses_transformer = False

        # 视觉/图像任务（通用关键词，不绑定具体论文的 leaf/disease）
        if has("image") or has("vision") or has("crop"):
            return "image_classification"

        # 时间序列任务（实际使用 LSTM/Transformer 且出现时序信号，不绑定具体领域词）
        if (actually_uses_lstm or actually_uses_transformer) and any(
            has(k) for k in ("time", "temporal", "sequence", "series", "forecast")
        ):
            return "time_series_prediction"

        if has("code") and (has("defect") or has("bug")):
            return "code_analysis"
        if has("genomic") or has("snp"):
            return "genomic_prediction"
        return "other"

    def _has_training(self, impl: dict) -> bool:
        if impl.get("training_strategy", "NOT_FOUND") not in (
            "NOT_FOUND", "Not applicable", ""
        ):
            no_kw = ["not applicable", "no fine-tuning", "evaluation study"]
            ts = impl["training_strategy"].lower()
            if not any(k in ts for k in no_kw):
                return True
        return any(
            impl.get(k, "NOT_FOUND") != "NOT_FOUND"
            for k in ["optimizer", "learning_rate", "epoch", "loss_function"]
        )

    def _targets(self, ptype: str, trainable: bool) -> list:
        if ptype == "model_proposing":
            t = ["实现论文提出的模型架构"]
            if trainable:
                t.append("复现训练流程")
            t += ["复现评估指标计算", "验证与论文结果的一致性"]
        elif ptype == "algorithm":
            t = ["实现论文提出的算法/方法"]
            t += ["构造输入并验证算法正确性", "验证与论文结果的一致性"]
        else:
            t = ["实现统一的评估协议", "复现对比实验结果"]
            if trainable:
                t.insert(0, "复现训练/微调流程")
        return t
