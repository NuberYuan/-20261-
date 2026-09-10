"""Step5.5: 实验流程设计 → experiment_design.yaml"""
from step5.base_analyzer import BaseAnalyzer


class ExperimentAnalyzer(BaseAnalyzer):

    def extract_by_rules(self) -> dict:
        exp = self.key_info.get("experiment", {})
        meth = self.key_info.get("methodology", {})

        # --- pipeline ---
        pipeline = self._extract_pipeline(exp, meth)

        # --- primary metric ---
        primary = self._extract_primary_metric(exp)

        # --- important results (之前丢失的) ---
        important_results = self._pick(exp.get("important_results"))

        # --- comparison results (不截断) ---
        comparison = str(exp.get("comparison_results", "NOT_FOUND"))

        return {
            "paper_name": self.paper_name,
            "pipeline": pipeline,
            "experiments": [{
                "name": "main_experiment",
                "description": comparison,
            }],
            "evaluation": {
                "primary_metric": primary,
                "secondary_metrics": self._parse_list(
                    exp.get("evaluation_metrics", "")
                ),
            },
            "comparison": {
                "baselines": self._parse_list(
                    exp.get("baseline_methods", "")
                ),
                "ablation_factors": self._extract_ablation_factors(exp),
            },
            "experiment_matrix": self._build_experiment_matrix(),
            "important_results": important_results,
        }

    # =========================
    # Pipeline
    # =========================

    def _extract_pipeline(self, exp: dict, meth: dict) -> list:
        pipeline = exp.get("technical_pipeline", [])
        if pipeline:
            return pipeline

        p = meth.get("pipeline", "")
        if p and p != "NOT_FOUND":
            # 按句号或数字编号分割
            steps = []
            for s in p.replace(" 1.", "\n1.").replace(" 2.", "\n2.").replace(
                " 3.", "\n3.").replace(" 4.", "\n4.").replace(" 5.", "\n5."
            ).split("\n"):
                s = s.strip()
                if s and len(s) > 5:
                    steps.append(s)
            if steps:
                return steps

        # 默认模板：按是否有模型/训练决定，纯算法论文不套 ML 模板
        impl = self.key_info.get("implementation_details", {})
        arch = meth.get("model_architecture", {})
        if not isinstance(arch, dict):
            arch = {}
        trainable = any(
            impl.get(k, "NOT_FOUND") != "NOT_FOUND"
            for k in ["optimizer", "learning_rate", "epoch", "loss_function"]
        )
        has_model = any(
            arch.get(k, "NOT_FOUND") not in ("NOT_FOUND", "")
            for k in ["alexnet", "googlenet", "resnet", "vgg", "densenet",
                      "mobilenet", "efficientnet", "lstm", "gru",
                      "transformer", "attention"]
        )
        if trainable or has_model:
            return [
                "1. 数据准备与预处理",
                "2. 模型训练/推理",
                "3. 评估与对比",
            ]
        return [
            "1. 输入构造与准备",
            "2. 算法/方法执行",
            "3. 结果验证与对比",
        ]

    # =========================
    # Metrics
    # =========================

    def _extract_primary_metric(self, exp: dict) -> str:
        mt = str(exp.get("evaluation_metrics", "")).lower()
        priority = ["f1", "accuracy", "r2", "rmse", "precision", "recall", "auc"]
        for kw in priority:
            if kw in mt:
                return kw.upper() if kw in ("r2", "auc") else kw.title()
        return "NOT_FOUND"

    # =========================
    # Ablation
    # =========================

    def _extract_ablation_factors(self, exp: dict) -> list:
        """
        从 ablation_study 文本中提取实验因子的结构化列表。

        不使用逗号暴力分割，而是按自然语义拆分。
        """
        text = str(exp.get("ablation_study", ""))

        if not text or text == "NOT_FOUND":
            return []

        factors = []

        # 尝试按已知模式拆分
        for delimiter in [
            "different architectures",
            "training mechanisms",
            "dataset types",
            "train-test splits",
            "learning rate",
            "batch size",
            "optimizer",
        ]:
            idx = text.lower().find(delimiter)
            if idx >= 0:
                # 提取这个因子所在的子句
                end = min(idx + 200, len(text))
                chunk = text[idx:end]
                # 找到第一个句号或逗号结束处
                for sep in [". ", "), ", "). "]:
                    end2 = chunk.find(sep)
                    if 20 < end2 < 150:
                        chunk = chunk[:end2 + 1]
                        break
                factors.append(chunk.strip())

        # 如果没匹配到模式，用括号内的内容
        if not factors:
            import re
            parens = re.findall(r'\(([^)]+)\)', text)
            for p in parens:
                if len(p) > 5:
                    factors.append(p.strip())

        return factors if factors else [text[:200]]

    # =========================
    # 工具
    # =========================

    def _parse_list(self, text: str) -> list:
        """将逗号分隔的文本解析为列表"""
        if not text or text == "NOT_FOUND":
            return []
        return [s.strip() for s in str(text).split(",") if s.strip()][:15]

    # =========================
    # 实验矩阵
    # =========================

    def _build_experiment_matrix(self) -> dict:
        """
        将 ablation 因子展开为可执行的实验组合矩阵。

        Architectures × TrainingModes × DataTypes × TrainTestSplits

        矩阵让代码生成 Agent 知道需要跑哪些组合，而不是隐式依赖散文描述。
        """
        meth = self.key_info.get("methodology", {})
        arch = meth.get("model_architecture", {})
        if not isinstance(arch, dict):
            arch = {}

        # 从 model_architecture 提取模型列表（只保留非 NOT_FOUND 的）
        architectures = []
        for key in ["alexnet", "googlenet", "resnet", "vgg", "densenet",
                     "mobilenet", "efficientnet"]:
            if arch.get(key, "NOT_FOUND") != "NOT_FOUND":
                architectures.append(key.title())

        if not architectures:
            model_str = meth.get("model", "")
            if model_str and model_str != "NOT_FOUND":
                architectures = [n.strip() for n in model_str.split(",") if n.strip()]

        # 训练模式：无训练信号时不生成任何训练模式
        impl = self.key_info.get("implementation_details", {})
        trainable = any(
            impl.get(k, "NOT_FOUND") != "NOT_FOUND"
            for k in ["optimizer", "learning_rate", "epoch", "loss_function"]
        )
        if not trainable:
            training_modes = []
        elif arch.get("pretrained_model", "NOT_FOUND") == "NOT_FOUND":
            training_modes = ["Scratch"]
        else:
            training_modes = ["Scratch", "Transfer"]

        # 数据集类型 — 仅图像数据存在 Color/Grayscale/Segmented 版本概念；
        # 非图像数据（合成/RL/无数据）固定 default，避免 PlantVillage 模板泄漏。
        ds = self.key_info.get("dataset", {})
        if not self._is_image_data(ds):
            data_types = ["default"]
        else:
            pp = str(ds.get("preprocessing", "")).lower()
            data_types = [dt for dt in ["Color", "Grayscale", "Segmented"]
                          if dt.lower() in pp]
            if not data_types:
                data_types = ["default"]

        # Train-test split 比例（兼容 "80/20"、"80-20"、"80:20" 等写法）
        splits_text = str(ds.get("train_test_split", ""))
        import re
        split_matches = re.findall(r'(\d+)\s*[-/:]\s*(\d+)', splits_text)
        if split_matches:
            splits = [f"{a}-{b}" for a, b in split_matches]
        else:
            splits = ["NOT_FOUND"]

        total_configs = (
            len(architectures) * len(training_modes)
            * len(data_types) * len(splits)
        )

        # 从 comparison_results 提取已知的最佳/最差值
        exp = self.key_info.get("experiment", {})
        comparison = str(exp.get("comparison_results", ""))
        best_match = re.search(r'Best.*?(\d+\.?\d*%).*?\(([^)]+)\)',
                               comparison, re.IGNORECASE)
        worst_match = re.search(r'Worst.*?(\d+\.?\d*%).*?\(([^)]+)\)',
                                comparison, re.IGNORECASE)

        return {
            "dimensions": {
                "architectures": architectures,
                "training_modes": training_modes,
                "data_types": data_types,
                "train_test_splits": splits,
            },
            "total_configurations": total_configs,
            "known_best": (
                f"{best_match.group(1)} — {best_match.group(2)}"
                if best_match else "NOT_FOUND"
            ),
            "known_worst": (
                f"{worst_match.group(1)} — {worst_match.group(2)}"
                if worst_match else "NOT_FOUND"
            ),
            "run_all": 0 < total_configs <= 4,
            "recommended_subset": (
                []
                if total_configs <= 4
                else [
                    f"{arch}::{mode}::{data_types[0]}::{splits[0]}"
                    for arch in architectures[:2]
                    for mode in training_modes
                ][:4]
            ),
            "note": (
                "无实验组合（论文无模型或无训练实验，纯理论/无数据论文）"
                if total_configs == 0
                else (
                    f"共 {total_configs} 个实验组合。90%+ 场景下只需跑推荐子集确认趋势，"
                    f"CI/CD 中跑全部组合验证稳定性。"
                    if total_configs > 4
                    else "实验组合 ≤4，可全量跑。"
                )
            ),
        }
