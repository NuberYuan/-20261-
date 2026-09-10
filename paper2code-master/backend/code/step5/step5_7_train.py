"""Step5.7: 参数配置设计 → training_config.yaml"""
from step5.base_analyzer import BaseAnalyzer


class TrainConfigAnalyzer(BaseAnalyzer):

    def extract_by_rules(self) -> dict:
        impl = self.key_info.get("implementation_details", {})
        repro = self.key_info.get("reproduction_requirements", {})
        params = repro.get("important_parameters", {})

        # 无训练信号时，不生成训练配置（纯算法/纯推理/无数据论文）
        if not self._is_trainable(impl):
            return {
                "paper_name": self.paper_name,
                "training": {
                    "applies": False,
                    "note": "论文无训练流程（无 optimizer/learning_rate/epoch/loss 信号）",
                },
                "training_modes": {},
                "reproducibility_notes": self._extract_difficulties(repro),
            }

        comparison = str(
            self.key_info.get("experiment", {})
            .get("comparison_results", "NOT_FOUND")
        )

        opt = impl.get("optimizer", "NOT_FOUND")
        opt_clean = (
            "SGD" if "sgd" in opt.lower()
            else ("Adam" if "adam" in opt.lower() else opt)
        )

        # 学习率调度器——优先从 important_parameters 读取
        scheduler_name = self._find_param(params, "scheduler")
        if scheduler_name == "NOT_FOUND":
            ts = str(impl.get("training_strategy", "")).lower()
            if "step" in ts:
                scheduler_name = "StepLR"
            elif "cosine" in ts:
                scheduler_name = "CosineAnnealingLR"
            elif "plateau" in ts:
                scheduler_name = "ReduceLROnPlateau"
            else:
                scheduler_name = "NOT_FOUND"

        # 参数来源追踪：区分论文明确值 vs 推断值 vs 工程默认值
        paper_params = set()
        if impl.get("batch_size", "NOT_FOUND") != "NOT_FOUND":
            paper_params.update(["batch_size", "batch_size_note"])
        if impl.get("learning_rate", "NOT_FOUND") != "NOT_FOUND":
            paper_params.add("learning_rate")
        if impl.get("epoch", "NOT_FOUND") != "NOT_FOUND":
            paper_params.add("epochs")
        if impl.get("optimizer", "NOT_FOUND") != "NOT_FOUND":
            paper_params.add("optimizer_name")
        if impl.get("momentum", "NOT_FOUND") != "NOT_FOUND":
            paper_params.add("momentum")
        if impl.get("weight_decay", "NOT_FOUND") != "NOT_FOUND":
            paper_params.add("weight_decay")
        if impl.get("loss_function", "NOT_FOUND") != "NOT_FOUND":
            paper_params.add("loss_function")

        scheduler_step_size = self._clean_step_size(
            self._find_param(params, "step_size")
        )
        scheduler_gamma = self._find_param(params, "gamma")

        # transfer 模式仅在论文使用预训练模型时存在（非 CNN/从头训练论文不适用）
        arch = self.key_info.get("methodology", {}).get("model_architecture", {})
        has_pretrained = (
            isinstance(arch, dict)
            and arch.get("pretrained_model", "NOT_FOUND") != "NOT_FOUND"
        )

        return {
            "paper_name": self.paper_name,
            "parameter_provenance": {
                "paper_confirmed": sorted(paper_params),
                "inferred_from_paper": [
                    "scheduler_name", "scheduler_step_size", "scheduler_gamma",
                    "dropout_ratio", "image_size",
                ],
                "engineering_default": [
                    "checkpoint.save_best", "checkpoint.best_metric",
                    "checkpoint.save_final", "checkpoint.save_optimizer_state",
                    "inference.device",
                ],
                "NOT_FOUND_in_paper": [
                    "early_stopping", "checkpoint.save_frequency_epochs",
                    "inference.batch_size", "freeze_backbone",
                ],
                "note": (
                    "paper_confirmed: 值直接来自论文 implementation_details；"
                    "inferred_from_paper: 从论文描述推断但非逐字引用；"
                    "engineering_default: 我们选择的工程实践值；"
                    "NOT_FOUND_in_paper: 论文未提供，需自行决定"
                ),
            },
            "training_modes": self._build_training_modes(comparison, has_pretrained),
            "training": {
                "batch_size": impl.get("batch_size", "NOT_FOUND"),
                "batch_size_note": (
                    "论文显式指定 batch_size；若为多模型，逐模型数值见 batch_size 字段。"
                ),
                "epochs": self._pick(impl.get("epoch"), "30"),
                "image_size": self._extract_image_size(params, impl),
                "optimizer": {
                    "name": opt_clean,
                    "lr": self._pick(impl.get("learning_rate")),
                    "momentum": self._find_param(params, "momentum"),
                    "weight_decay": self._find_param(params, "weight_decay"),
                },
                "scheduler": {
                    "name": scheduler_name,
                    "step_size": scheduler_step_size,
                    "gamma": scheduler_gamma,
                    "note": self._build_scheduler_note(
                        scheduler_name, scheduler_step_size, scheduler_gamma
                    ),
                },
                "loss_function": self._pick(
                    impl.get("loss_function"),
                    "CrossEntropyLoss",
                ),
                "dropout_ratio": self._extract_dropout(impl, params),
                "early_stopping": self._find_param(params, "early_stopping"),
            },
            "checkpoint": {
                "save_best": True,
                "best_metric": "val_accuracy",
                "best_mode": "max",
                "save_final": True,
                "save_frequency_epochs": "NOT_FOUND",
                "save_optimizer_state": True,
                "resume_from": None,
                "note": (
                    "论文无独立验证集，best_metric 设为 val_accuracy 是工程偏离。"
                    "如严格复现论文，应从训练集二次划分验证集，"
                    "或直接用 test_accuracy（不推荐，会导致数据泄露）。"
                ),
            },
            "train_test_splits": self._extract_splits(params),
            "inference": {
                "device": (
                    "cuda" if "gpu" in str(impl.get("hardware", "")).lower()
                    else "NOT_FOUND"
                ),
                "batch_size": "NOT_FOUND",
            },
            "reproducibility_notes": self._extract_difficulties(repro),
        }

    # =========================
    # 参数提取
    # =========================

    def _is_trainable(self, impl: dict) -> bool:
        """是否存在训练信号（与 step5_1 的 _has_training 一致）。"""
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

    def _infer_reset_layers(self) -> list:
        """
        推断迁移学习时需要重置的层。

        AlexNet → fc8 (最后的全连接层)
        GoogLeNet → fc (classifier) + 两个辅助分类器的 fc
        """
        meth = self.key_info.get("methodology", {})
        arch = meth.get("model_architecture", {})
        if not isinstance(arch, dict):
            return ["fc8 或 classifier (NOT_FOUND: 无法确定模型)"]

        nc = self._extract_num_classes(self.key_info.get("dataset", {}))
        reset_map = {
            "alexnet": [f"classifier.6 (fc8, in_features=4096, out_features={nc})"],
            "googlenet": [
                f"fc (主分类器, out_features={nc})",
                f"aux1.fc2 (辅助分类器1, out_features={nc})",
                f"aux2.fc2 (辅助分类器2, out_features={nc})",
            ],
            "resnet": [f"fc (out_features={nc})"],
            "vgg": [f"classifier.6 (out_features={nc})"],
        }
        result = []
        for model_key, layers in reset_map.items():
            if arch.get(model_key, "NOT_FOUND") != "NOT_FOUND":
                result.extend(layers)
        return result if result else ["fc8 或 classifier (需根据模型确认)"]

    def _build_training_modes(self, comparison: str, has_pretrained: bool) -> dict:
        """
        构建训练模式。

        scratch 始终存在；transfer 仅在论文使用预训练模型（has_pretrained）
        时生成，避免对非 CNN / 从头训练的论文硬编码 ImageNet 迁移学习。
        """
        modes = {
            "scratch": {
                "description": "从头训练：所有层随机初始化",
                "pretrained": False,
                "weight_init": "kaiming_normal",
                "reset_layers": [],
                "freeze_backbone": False,
            },
        }
        if has_pretrained:
            modes["transfer"] = {
                "description": "迁移学习：加载ImageNet预训练权重，重置最后一层",
                "pretrained": True,
                "pretrained_source": "torchvision.models (ImageNet)",
                "reset_layers": self._infer_reset_layers(),
                "freeze_backbone": {
                    "value": "NOT_FOUND",
                    "decision_type": "engineering_decision",
                    "note": (
                        "⚠ 此项为纯工程决策（论文未提及，NOT_FOUND），不是论文要求。"
                        "默认不冻结（全量微调），因为目标域与 ImageNet 差异较大。"
                        "两种策略可选："
                        "(a) 不冻结 (fine-tuning)：所有层参与训练，收敛慢但效果好 — 推荐；"
                        "(b) 冻结 backbone：只训练重置的分类层，收敛快但可能欠拟合。"
                        "代码 Agent 可自由选择，但需在 config.yaml 中显式标注。"
                    ),
                    "engineering_default": False,
                },
                "note": (
                    f"论文两种模式的对比结果: {comparison}"
                    if comparison != "NOT_FOUND" else "NOT_FOUND"
                ),
            }
        return modes

    @staticmethod
    def _build_scheduler_note(name: str, step_size: str, gamma: str) -> str:
        """根据实际解析出的调度器生成说明，避免写死某篇论文的调度策略。"""
        if name == "StepLR":
            if step_size != "NOT_FOUND" and gamma != "NOT_FOUND":
                return (
                    f"每 {step_size} 个 epoch 将学习率乘以 {gamma}；"
                    "PyTorch StepLR 与此行为等价。"
                )
            return "StepLR：按固定步长衰减学习率；论文未给出具体 step_size/gamma，需自行设定。"
        if name == "CosineAnnealingLR":
            return "CosineAnnealingLR：余弦退火调度。"
        if name == "ReduceLROnPlateau":
            return "ReduceLROnPlateau：验证指标停滞时降低学习率。"
        return "论文未明确说明学习率调度器，训练时可选用 StepLR 等常见策略。"

    def _find_param(self, params, key) -> str:
        """在 important_parameters 中模糊匹配参数"""
        if not isinstance(params, dict):
            return "NOT_FOUND"
        # 统一将下划线替换为空格后匹配
        search = key.replace("_", " ").lower()
        for k, v in params.items():
            if search in str(k).replace("_", " ").lower():
                return str(v)
        return "NOT_FOUND"

    @staticmethod
    def _clean_step_size(raw: str) -> str:
        """
        从含解释文本的 step_size 中提取纯净值。
        "10 epochs (decrease by factor of 10 every 30/3 epochs)" → "10"
        "30/3 epochs" → "10"
        "30" → "30"
        """
        if raw in ("NOT_FOUND", ""):
            return "NOT_FOUND"
        import re
        # 评估 "30/3" 表达式
        expr_m = re.search(r'(\d+(?:\s*/\s*\d+)?)', raw)
        if expr_m:
            expr = expr_m.group(1).replace(" ", "")
            try:
                if "/" in expr:
                    a, b = expr.split("/")
                    result = int(a) / int(b)
                else:
                    result = int(expr)
                return str(int(result)) if result == int(result) else str(result)
            except (ValueError, ZeroDivisionError):
                pass
        return raw

    def _extract_image_size(self, params, impl: dict) -> str:
        """从 params 或 preprocessing 中提取图像尺寸"""
        import re
        # 先查 params（只匹配 image_size 或 image size，避免匹配 step_size）
        if isinstance(params, dict):
            for k, v in params.items():
                key_lower = str(k).lower()
                if "image" in key_lower and "size" in key_lower:
                    return str(v)
                # 如果值本身包含明确的尺寸格式如 256x256
                if isinstance(v, str) and re.search(r'\d+\s*x\s*\d+', v):
                    return str(v)

        # 再查 implementation 或 dataset
        ds = self.key_info.get("dataset", {})
        pp = str(ds.get("preprocessing", ""))
        for s in ["256", "224", "512", "299"]:
            if s in pp:
                return f"{s}x{s}"

        return "NOT_FOUND"

    def _extract_dropout(self, impl: dict, params) -> str:
        """提取 dropout ratio"""
        # 从 params
        if isinstance(params, dict):
            for k, v in params.items():
                if "dropout" in str(k).lower():
                    return str(v)

        # 从 model_architecture
        arch = self.key_info.get("methodology", {}).get("model_architecture", {})
        if isinstance(arch, dict):
            special = str(arch.get("special_modules", ""))
        else:
            special = str(arch) if arch else ""
        import re
        m = re.search(r"dropout.*?(\d+\.?\d*)", special.lower())
        if m:
            return m.group(1)

        return "NOT_FOUND"

    def _extract_splits(self, params) -> list:
        """提取 train-test 分割比例"""
        if isinstance(params, dict):
            for k, v in params.items():
                if "split" in str(k).lower():
                    if isinstance(v, list):
                        return [float(x) for x in v]
                    try:
                        return [float(v)]
                    except (ValueError, TypeError):
                        pass

        # Fallback: 从 dataset 字符串解析
        ds = self.key_info.get("dataset", {})
        ts = str(ds.get("train_test_split", ""))
        import re
        matches = re.findall(r'(\d+)[-–](\d+)', ts)
        if matches:
            return [int(a) / 100 for a, _ in matches][:5]

        return ["NOT_FOUND"]

    def _extract_difficulties(self, repro: dict) -> list:
        """提取复现困难/注意事项"""
        diff = repro.get("possible_difficulties", [])
        if isinstance(diff, list):
            return diff
        if isinstance(diff, str) and diff not in ("NOT_FOUND", ""):
            return [diff]
        return []
