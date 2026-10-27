"""
Step4.4: 歧义与缺失分析 → gap_analysis.yaml

标记论文未明确说明、但代码实现必须决策的点。

设计原则:
  - Rule 层: 从 key_information.json 中扫描 NOT_FOUND 字段
  - LLM 层: 根据论文上下文推断可能的解决方案
  - 每个 gap 包含 severity / affect_modules / suggested / alternatives
  - 工程参数缺口与跨字段冲突与 Step6 一致性检查对齐
    （training_config.* 四项 + train/test vs val 冲突），保证 gap_analysis
    不漏报 Step6 才发现的问题
"""

import json
import re

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

        # ---- 工程参数缺口（与 Step6 的 training_config.* gap 对齐）----
        if is_training:
            gaps.extend(self._check_engineering_parameter_gaps(ki, arch))

        # ---- 跨字段冲突（与 Step6 的 CONFLICT-01 等对齐）----
        gaps.extend(self._check_cross_field_conflicts(ki, is_training))

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
    # 工程参数缺口（与 Step6 training_config.NOT_FOUND_in_paper 对齐）
    # ============================================================

    # (路径, 关键词(在论文全文找), 问题, 建议)
    # 对应 Step5 training_config.parameter_provenance.NOT_FOUND_in_paper 的四项：
    # early_stopping / checkpoint.save_frequency_epochs / inference.batch_size /
    # freeze_backbone。这些工程参数论文通常不写，但代码必须决策。
    ENGINEERING_PARAM_GAPS = [
        ("training.early_stopping",
         ("early stop", "early-stop", "patience", "早停"),
         "early_stopping 策略（patience/触发条件）未在论文中说明",
         "训练到固定 epochs，或工程默认 patience=10 做早停，并在 config.yaml 显式标注"),
        ("checkpoint.save_frequency_epochs",
         ("save every", "checkpoint every", "saved every", "saving every",
          "保存频率", "保存间隔", "save frequency"),
         "checkpoint 保存频率（save_frequency_epochs）未在论文中说明",
         "工程默认每个 epoch 保存一次（或只保存 best+final），并在 config.yaml 显式标注"),
        ("inference.batch_size",
         ("inference batch", "test batch", "推理 batch", "evaluation batch",
          "推理批", "测试批"),
         "inference.batch_size（推理批大小）未在论文中说明",
         "推理批大小与训练 batch_size 解耦，工程默认 32/64，config.yaml 显式标注"),
        ("training.freeze_backbone",
         ("freeze", "frozen", "冻结", "fixed feature", "feature extractor"),
         "迁移学习时骨干网络冻结策略（freeze_backbone）未在论文中说明",
         "默认不冻结全量微调（目标域与预训练域差异大时更稳），config.yaml 显式标注为工程决策"),
    ]

    def _check_engineering_parameter_gaps(self, ki: dict, arch: dict) -> list:
        """扫描论文缺失的工程参数（对应 Step6 的 training_config.* gap）。

        与 Step5 parameter_provenance.NOT_FOUND_in_paper 同口径：这些参数
        论文几乎不会写，但实现必须决策。关键词在 key_info 全文里找；
        freeze_backbone 仅对使用预训练权重的论文有意义。
        """
        blob = json.dumps(ki, ensure_ascii=False).lower()
        gaps = []
        has_pretrained = (
            isinstance(arch, dict)
            and arch.get("pretrained_model", "NOT_FOUND") not in ("NOT_FOUND", "")
        )
        for path, keywords, question, suggested in self.ENGINEERING_PARAM_GAPS:
            if path.endswith("freeze_backbone") and not has_pretrained:
                continue  # 无预训练就没有骨干冻结的决策点
            if any(kw in blob for kw in keywords):
                continue  # 论文已说明
            gaps.append({
                "id": self._next_id(self.GAP_PREFIX),
                "path": path,
                "question": question,
                "severity": "low",
                "current_value": "NOT_FOUND",
                "affect_modules": ["training/train.py", "config/config.yaml"],
                "suggested": suggested,
                "alternatives": [],
            })
        return gaps

    # ============================================================
    # 跨字段冲突（与 Step6 CONFLICT-01 对齐）
    # ============================================================

    def _check_cross_field_conflicts(self, ki: dict, is_training: bool) -> list:
        """检测 key_info 字段之间的口径冲突。

        目前覆盖 Step6 CONFLICT-01 的同款问题在需求侧的形态：
        论文只给 train/test 二路划分（无验证集），而训练流程保存最优模型
        需要模型选择口径。直接用 test 指标选模型构成数据泄露——这是
        「划分口径」与「模型选择」两字段间的冲突，需在需求侧就显式决策。
        """
        gaps = []
        if not is_training:
            return gaps

        split_text = str(ki.get("dataset", {}).get("train_test_split", ""))
        low = split_text.lower()
        has_val = bool(re.search(r"\bval(?:idation)?\b|验证集", low))
        has_train_test = bool(re.search(r"\btrain\b|\btest\b|训练|测试", low))

        if has_train_test and not has_val:
            gaps.append({
                "id": self._next_id(self.GAP_PREFIX),
                "path": "dataset.train_test_split",
                "question": (
                    "跨字段冲突：论文仅 train/test 二路划分（无验证集），但训练流程"
                    "保存最优模型需要模型选择口径。直接用 test 指标选模型会造成"
                    "数据泄露，需从训练集二次划分验证集（工程偏离）"
                ),
                "severity": "medium",
                "current_value": split_text[:200],
                "affect_modules": ["data/dataset.py", "training/train.py",
                                   "config/config.yaml"],
                "suggested": (
                    "方案A（推荐）: 从训练集按 80/20 二次划分出验证集用于早停/模型选择；"
                    "方案B: 严格复现论文，用 test 指标选最优模型（不推荐，有泄露风险）。"
                    "选定后在 config.yaml 的 split_strategy/checkpoint 中显式记录"
                ),
                "alternatives": [
                    "从训练集二次划分验证集（工程偏离）",
                    "严格 train/test 二路 + 固定最后 epoch 权重（不做模型选择）",
                ],
            })
        return gaps

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

        # 跳过纯描述性字段（image_analysis 的描述文字是给人看的，不是代码参数）
        skip_prefixes = ("image_analysis", "paper_figures", "figures")

        def scan(obj, prefix=""):
            if isinstance(obj, dict):
                for k, v in obj.items():
                    path = f"{prefix}.{k}" if prefix else k
                    # 跳过图片描述类字段
                    if any(path.startswith(p) for p in skip_prefixes):
                        continue
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
