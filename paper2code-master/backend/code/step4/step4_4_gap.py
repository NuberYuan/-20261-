"""
Step4.4: 歧义与缺失分析 → gap_analysis.yaml

标记论文未明确说明、但代码实现必须决策的点。

设计原则:
  - Rule 层: 从 key_information.json 中扫描 NOT_FOUND 字段
  - LLM 层: 根据论文上下文推断可能的解决方案
  - 每个 gap 包含 severity / affect_modules / suggested / alternatives
"""

from step4.base_requirement_analyzer import BaseRequirementAnalyzer


class GapAnalyzer(BaseRequirementAnalyzer):

    # 需要检查的关键路径
    CRITICAL_PATHS = [
        # (路径, 问题模板, 严重程度)
        ("dataset.train_test_split",
         "验证集划分比例及方法未明确说明", "high"),
        ("dataset.preprocessing",
         "数据预处理的具体参数未完全给出", "high"),
        ("implementation_details.batch_size",
         "batch_size 未在论文中明确标注", "high"),
        ("implementation_details.learning_rate",
         "学习率调度策略未完整说明", "medium"),
        ("implementation_details.hardware",
         "训练硬件环境未说明（影响复现精度对比）", "medium"),
        ("implementation_details.training_strategy",
         "训练策略细节（early stopping, warmup 等）缺失", "medium"),
        ("methodology.model_architecture.pretrained_model",
         "预训练权重的具体来源或版本未说明", "medium"),
        ("methodology.model_architecture.special_modules",
         "模型特殊模块（dropout, BN 等）位置和参数未详述", "medium"),
        ("implementation_details.optimizer",
         "优化器未指定（如 SGD 的 momentum/weight_decay）", "medium"),
        ("implementation_details.loss_function",
         "损失函数未明确说明", "medium"),   # 严重程度在运行时根据论文类型动态调整
        ("experiment.evaluation_metrics",
         "评估指标的实现细节未说明", "low"),
        ("reproduction_requirements.important_parameters",
         "超参数未完整列出", "medium"),
    ]

    # 建议方案映射
    SUGGESTIONS = {
        "dataset.train_test_split": "使用 80/10/10 分层随机划分（train/val/test），或按论文数据集惯例",
        "dataset.preprocessing": "对于使用预训练权重的模型，建议采用 ImageNet 标准预处理",
        "implementation_details.batch_size": "参考同类论文或根据 GPU 显存合理设置；图像分类任务常用 32/64/128",
        "implementation_details.learning_rate": "如使用 StepLR，建议 step_size=10, gamma=0.1（常见配置）",
        "implementation_details.hardware": "使用可用 GPU 运行，记录硬件型号和 CUDA 版本以便对比",
        "implementation_details.training_strategy": "如论文未提及 early stopping，训练到收敛或使用 patience=10",
        "methodology.model_architecture.pretrained_model": "优先使用 torchvision 官方提供的预训练权重",
        "methodology.model_architecture.special_modules": "参考原始模型的标准实现（如 dropout、激活函数等默认超参）",
        "implementation_details.optimizer": "SGD 默认 momentum=0.9, weight_decay=5e-4 是常见设置",
        "implementation_details.loss_function": "分类任务默认 CrossEntropyLoss，回归任务默认 MSELoss",
        "experiment.evaluation_metrics": "使用 scikit-learn 标准实现，确保与论文定义一致",
        "reproduction_requirements.important_parameters": "从实验章节和 ablation study 中提取所有可调参数",
    }

    def extract_by_rules(self) -> dict:
        self._reset_counter()

        ki = self.key_info
        gaps = []

        # 原型信号：非训练/无模型/无数据论文过滤掉 ML 专属 gap 路径
        is_training = self._is_training_paper()
        has_dataset = str(
            ki.get("dataset", {}).get("dataset_name", "NOT_FOUND")
        ) != "NOT_FOUND"
        meth = ki.get("methodology", {})
        arch = meth.get("model_architecture", {})
        if not isinstance(arch, dict):
            arch = {}
        has_model = any(
            arch.get(k, "NOT_FOUND") not in ("NOT_FOUND", "")
            for k in ["alexnet", "googlenet", "resnet", "vgg", "densenet",
                      "mobilenet", "efficientnet", "lstm", "transformer"]
        )
        if not has_model:
            mf = meth.get("model", "")
            has_model = bool(mf and mf != "NOT_FOUND")

        # 非图像数据（合成/RL/无数据）替换图像专属建议（ImageNet/torchvision/图像分类）
        is_image = self._is_image_data(ki.get("dataset", {}))
        suggestions = dict(self.SUGGESTIONS)
        if not is_image:
            suggestions["dataset.preprocessing"] = "按数据语义做标准化（z-score/归一化），无 RGB 通道概念"
            suggestions["implementation_details.batch_size"] = "参考同类论文或根据内存合理设置；常用 32/64/128"
            suggestions["methodology.model_architecture.pretrained_model"] = "优先使用官方提供的预训练权重"

        # ---- 扫描关键路径 ----
        paths = self.CRITICAL_PATHS
        if not is_training:
            paths = [p for p in paths
                     if not p[0].startswith("implementation_details.")]
        if not has_model:
            paths = [p for p in paths
                     if not p[0].startswith("methodology.model_architecture.")]
        if not has_dataset:
            paths = [p for p in paths
                     if not p[0].startswith("dataset.")]
        for path, question, severity in paths:
            value = self._get_nested(ki, path)
            if value in ("NOT_FOUND", "", None, "not mentioned"):
                gaps.append({
                    "id": self._next_id(self.GAP_PREFIX),
                    "path": path,
                    "question": question,
                    "severity": severity,
                    "current_value": "NOT_FOUND",
                    "affect_modules": self._infer_affected_modules(path),
                    "suggested": suggestions.get(path, "需要人工决策"),
                    "alternatives": self._get_alternatives(path),
                })

        # ---- 检查数值字段是否为占位符（仅训练论文） ----
        impl = ki.get("implementation_details", {})
        if is_training:
            for field in ["batch_size", "learning_rate", "epoch", "momentum",
                          "weight_decay"]:
                val = str(impl.get(field, "NOT_FOUND"))
                if val in ("NOT_FOUND", "not found", "not mentioned", ""):
                    if not any(g["path"] == f"implementation_details.{field}"
                               for g in gaps):
                        gaps.append({
                            "id": self._next_id(self.GAP_PREFIX),
                            "path": f"implementation_details.{field}",
                            "question": f"{field} 未在论文中指定",
                            "severity": "high" if field in ("batch_size", "learning_rate")
                                        else "medium",
                            "current_value": "NOT_FOUND",
                            "affect_modules": ["training/train.py"],
                            "suggested": suggestions.get(
                                f"implementation_details.{field}",
                                "从论文上下文推断或使用常见默认值"),
                            "alternatives": [],
                        })

        # ---- 纯算法论文（无训练、无数据集）：补算法正确性/复现验证 gap ----
        if not is_training and not has_dataset:
            gaps.append({
                "id": self._next_id(self.GAP_PREFIX),
                "path": "methodology.method_name",
                "question": "纯算法论文缺少标准化的正确性/复杂度验证基准",
                "severity": "high",
                "current_value": "NOT_FOUND",
                "affect_modules": ["src/"],
                "suggested": "构造代表性输入，与论文示例/定理/已知结果对照验证",
                "alternatives": ["与参考实现对比", "运行论文提供的样例"],
            })

        # ---- 检查文本字段中的模糊描述 ----
        gaps.extend(self._check_vague_descriptions(ki))

        # 按严重程度排序
        severity_order = {"high": 0, "medium": 1, "low": 2}
        gaps.sort(key=lambda g: severity_order.get(g["severity"], 99))

        # ---- 汇总已解决的假设 ----
        resolved = self._extract_resolved_assumptions(ki)

        return {
            "paper_name": self.paper_name,
            "gaps": gaps,
            "summary": {
                "total": len(gaps),
                "by_severity": {
                    "high": sum(1 for g in gaps if g["severity"] == "high"),
                    "medium": sum(1 for g in gaps if g["severity"] == "medium"),
                    "low": sum(1 for g in gaps if g["severity"] == "low"),
                },
            },
            "resolved_assumptions": resolved,
        }

    # ============================================================
    # 模糊描述检查
    # ============================================================

    def _check_vague_descriptions(self, ki: dict) -> list:
        """扫描包含模糊措辞的字段值"""
        gaps = []
        vague_keywords = [
            "similar to", "like", "such as", "etc", "approximately",
            "about", "around", "roughly",
            "类似", "约", "等等", "左右", "大概",
        ]

        def scan(obj, prefix=""):
            if isinstance(obj, dict):
                for k, v in obj.items():
                    path = f"{prefix}.{k}" if prefix else k
                    if isinstance(v, str) and v not in ("NOT_FOUND", ""):
                        v_lower = v.lower()
                        for kw in vague_keywords:
                            if kw in v_lower:
                                gaps.append({
                                    "id": self._next_id(self.GAP_PREFIX),
                                    "path": path,
                                    "question": f"'{v[:100]}' 使用了模糊措辞 '{kw}'，具体值待确认",
                                    "severity": "low",
                                    "current_value": v[:200],
                                    "affect_modules": self._infer_affected_modules(path),
                                    "suggested": "从论文其他章节交叉验证，或标记为可配置参数",
                                    "alternatives": [],
                                })
                                break
                    elif isinstance(v, (dict, list)):
                        scan(v, path)
            elif isinstance(obj, list):
                for i, item in enumerate(obj):
                    if isinstance(item, (dict, list)):
                        scan(item, f"{prefix}[{i}]")

        scan(ki)
        return gaps

    # ============================================================
    # 已解决的假设
    # ============================================================

    def _extract_resolved_assumptions(self, ki: dict) -> list:
        """收集已经基于代码常识做的推断"""
        assumptions = []
        meth = ki.get("methodology", {})
        arch = meth.get("model_architecture", {})
        if not isinstance(arch, dict):
            arch = {}

        pretrained = arch.get("pretrained_model", "NOT_FOUND")
        if pretrained != "NOT_FOUND" and "imagenet" in pretrained.lower():
            assumptions.append(
                "预训练权重来源假设为 torchvision 官方提供的 ImageNet 权重"
            )

        impl = ki.get("implementation_details", {})
        fw = str(impl.get("programming_framework", "")).lower()
        if fw in ("not_found", "") or "caffe" in fw:
            model_text = str(meth).lower()
            if any(k in model_text for k in
                   ["alexnet", "resnet", "googlenet", "vgg", "lstm"]):
                assumptions.append("框架推断为 PyTorch（论文可能使用其他框架，但代码采用 PyTorch 实现）")

        # 数据增强参数假设
        pp = str(ki.get("dataset", {}).get("preprocessing", "")).lower()
        if "flip" in pp and "p=" not in pp:
            assumptions.append("数据增强概率参数未明确，默认使用 p=0.5")

        return assumptions

    # ============================================================
    # 辅助方法
    # ============================================================

    @staticmethod
    def _infer_affected_modules(path: str) -> list:
        """根据路径推断受影响的代码模块"""
        mapping = {
            "dataset": ["data/dataset.py"],
            "implementation_details": ["training/train.py", "config/config.yaml"],
            "methodology.model_architecture": ["models/*.py"],
            "reproduction_requirements": ["training/train.py", "config/config.yaml"],
            "experiment": ["evaluation/evaluate.py", "utils/metrics.py"],
        }
        for prefix, modules in mapping.items():
            if path.startswith(prefix):
                return modules
        return ["config/config.yaml"]

    @staticmethod
    def _get_alternatives(path: str) -> list:
        """提供备选方案"""
        alts = {
            "dataset.train_test_split": [
                "80/20 简单划分（无验证集）",
                "5-fold 交叉验证（更严格）",
                "70/15/15 三路划分",
            ],
            "implementation_details.batch_size": [
                "32（通用默认）",
                "64（较大 GPU 内存）",
                "16（较小 GPU 内存）",
            ],
            "implementation_details.hardware": [
                "使用可用 GPU 并在报告中注明型号",
                "纯 CPU 训练（较慢）",
            ],
        }
        return alts.get(path, [])

    def _is_training_paper(self) -> bool:
        """判断论文是否涉及训练"""
        impl = self.key_info.get("implementation_details", {})
        return any(
            impl.get(k, "NOT_FOUND") != "NOT_FOUND"
            for k in ["optimizer", "learning_rate", "epoch"]
        )
