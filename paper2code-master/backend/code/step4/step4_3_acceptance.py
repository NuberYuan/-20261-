"""
Step4.3: 验收标准分析 → acceptance_criteria.yaml

定义"复现成功"的衡量标准，包括:
  - 精度目标（从论文实验结果提取）
  - 功能测试用例（从 4.1 功能需求推导）
  - 边界情况处理
  - 可复现性约束

验收标准是 Step6 迭代修复的停止条件。
"""

from typing import List, Dict, Optional, Any, Tuple, Union

import re
from step4.base_requirement_analyzer import BaseRequirementAnalyzer


class AcceptanceAnalyzer(BaseRequirementAnalyzer):

    def extract_by_rules(self) -> dict:
        ki = self.key_info
        self._reset_counter()

        exp = ki.get("experiment", {})

        # ---- 精度目标 ----
        accuracy_targets = self._extract_accuracy_targets(exp)

        # ---- 功能测试用例 ----
        functional_tests = self._derive_functional_tests(ki)

        # ---- 边界情况 ----
        edge_cases = self._derive_edge_cases(ki)

        # ---- 可复现性约束 ----
        reproducibility = self._extract_reproducibility(ki)

        return {
            "paper_name": self.paper_name,
            "accuracy_targets": accuracy_targets,
            "functional_tests": functional_tests,
            "edge_cases": edge_cases,
            "reproducibility": reproducibility,
        }

    # ============================================================
    # 精度目标
    # ============================================================

    def _extract_accuracy_targets(self, exp: dict) -> dict:
        metrics = str(exp.get("evaluation_metrics", "")).lower()
        results = str(exp.get("important_results", ""))
        comparison = str(exp.get("comparison_results", ""))

        targets = {}

        # 主要指标
        primary = self._detect_primary_metric(metrics)
        primary_value = self._search_metric_value(primary, results + " " + comparison)

        targets["primary"] = {
            "metric": primary,
            "target": primary_value if primary_value else "NOT_FOUND",
            "source": "论文实验结果",
            "tolerance": self._suggest_tolerance(primary),
        }

        # 次要指标
        secondary = {}
        for m in ["precision", "recall", "f1", "accuracy", "rmse", "r2", "auc"]:
            if m in metrics and m != primary:
                val = self._search_metric_value(m, results + " " + comparison)
                secondary[m] = val if val else "NOT_FOUND"

        if secondary:
            targets["secondary"] = secondary

        return targets

    @staticmethod
    def _detect_primary_metric(metrics: str) -> str:
        priority = ["f1", "accuracy", "r²", "rmse", "precision", "recall", "auc"]
        for kw in priority:
            if kw in metrics:
                return kw.title() if kw == "auc" else (
                    "F1-Score" if kw == "f1" else kw.title())
        return "Accuracy"

    @staticmethod
    def _search_metric_value(metric: str, text: str) -> Optional[str]:
        """从论文结果文本中搜索指标数值"""
        if not text or text == "NOT_FOUND":
            return None

        # 构建搜索模式: "accuracy ... 99.24%" 或 "99.24% ... accuracy"
        patterns = [
            rf'{metric}[:\s]*(\d+\.?\d*)\s*%?',
            rf'(\d+\.?\d*)\s*%?\s*{metric}',
        ]
        for pat in patterns:
            match = re.search(pat, text, re.IGNORECASE)
            if match:
                val = match.group(1)
                return f"{val}%" if "%" in match.group(0) else val
        return None

    @staticmethod
    def _suggest_tolerance(metric: str) -> str:
        """根据指标类型建议容差范围"""
        m = metric.lower()
        if m in ("accuracy", "precision", "recall", "f1", "auc", "f1-score"):
            return "±1%（考虑硬件和随机性差异）"
        if m in ("rmse", "mae"):
            return "±5%（数值型指标的合理波动范围）"
        if m in ("r2", "r²"):
            return "±0.02"
        return "NOT_FOUND"

    # ============================================================
    # 功能测试用例
    # ============================================================

    def _derive_functional_tests(self, ki: dict) -> list:
        tests = []
        impl = ki.get("implementation_details", {})
        ds = ki.get("dataset", {})
        meth = ki.get("methodology", {})
        arch = meth.get("model_architecture", {})
        if not isinstance(arch, dict):
            arch = {}
        is_image = self._is_image_data(ds)

        # 是否存在 ML 模型（与 4.1/4.2 一致）；纯算法论文无模型，不生成模型相关测试
        model_names = []
        for key in ["alexnet", "googlenet", "resnet", "vgg", "densenet",
                    "mobilenet", "efficientnet", "lstm", "transformer"]:
            detail = arch.get(key, "NOT_FOUND")
            if detail and detail != "NOT_FOUND" and isinstance(detail, str):
                model_names.append(key.title())
        if not model_names:
            mf = meth.get("model", "")
            if mf and mf != "NOT_FOUND":
                model_names = [n.strip() for n in mf.split(",") if n.strip()]
        has_model = bool(model_names)
        method_name = meth.get("method_name", "")
        has_method = (
            not has_model and bool(method_name)
            and method_name != "NOT_FOUND"
        )

        # 测试 1: 单样本推理（仅当有模型）
        if has_model:
            tests.append({
                "id": self._next_id(self.AC_PREFIX),
                "name": "单样本推理可用性",
                "description": (
                    "加载训练好的模型，对单张图像进行推理，验证返回结果格式正确"
                    if is_image else
                    "加载训练好的模型，对单个样本进行推理，验证返回结果格式正确"
                ),
                "steps": [
                    "1. 加载模型权重",
                    "2. 读取一张测试图像" if is_image else "2. 读取一条测试样本",
                    "3. 执行预处理",
                    "4. 调用 model.forward()",
                    "5. 检查输出格式",
                ],
                "expected": "返回合法的概率分布，shape 符合预期，sum ≈ 1.0（softmax 输出）",
                "related_fr": ["FR-03"],
            })
        elif has_method:
            # 纯算法论文：验证算法能正确运行
            tests.append({
                "id": self._next_id(self.AC_PREFIX),
                "name": "算法运行可用性",
                "description": f"执行论文方法 {method_name}，验证输出格式与预期一致",
                "steps": [
                    "1. 构造代表性输入",
                    "2. 执行算法",
                    "3. 检查输出格式与数值合理性",
                ],
                "expected": "算法可运行并产出预期格式的输出",
                "related_fr": ["FR-01"],
            })

        # 测试 2: 训练 Loss 收敛（如果有训练）
        if self._has_training(impl):
            tests.append({
                "id": self._next_id(self.AC_PREFIX),
                "name": "训练 Loss 收敛",
                "description": f"运行完整训练流程（{impl.get('epoch', 'N')} epochs），验证 loss 趋势",
                "steps": [
                    "1. 从头初始化模型",
                    "2. 运行训练循环",
                    "3. 记录每个 epoch 的 training loss 和 validation loss",
                ],
                "expected": "Training loss 单调下降，Validation loss 不出现严重过拟合",
                "related_fr": ["FR-05"],
            })

        # 测试 3: 精度达标
        metrics = str(ki.get("experiment", {}).get("evaluation_metrics", ""))
        if metrics and metrics != "NOT_FOUND":
            tests.append({
                "id": self._next_id(self.AC_PREFIX),
                "name": "评估指标达标",
                "description": f"在测试集上计算 {metrics}，与论文报告值对比",
                "steps": [
                    "1. 加载最优模型权重",
                    "2. 在完整测试集上运行推理",
                    "3. 计算所有评估指标",
                    "4. 与论文 Table 中报告的值对比",
                ],
                "expected": "各指标在容差范围内与论文一致",
                "related_fr": ["FR-06"],
            })

        # 测试 4: 模型持久化往返（仅当有模型）
        if has_model:
            tests.append({
                "id": self._next_id(self.AC_PREFIX),
                "name": "模型序列化往返",
                "description": "保存模型 → 重新加载 → 推理，验证精度不下降",
                "steps": [
                    "1. 保存模型 state_dict 到磁盘",
                    "2. 重新初始化模型并加载权重",
                    "3. 对同一批数据分别推理",
                    "4. 对比两次输出是否完全一致",
                ],
                "expected": "加载后的模型输出与保存前完全一致（torch.allclose）",
                "related_fr": ["FR-06"],
            })

        return tests

    # ============================================================
    # 边界情况
    # ============================================================

    def _derive_edge_cases(self, ki: dict) -> list:
        edges = []
        ds = ki.get("dataset", {})
        res = self._extract_resize(ds)
        is_image = self._is_image_data(ds)

        if is_image:
            edges.append({
                "id": self._next_id(self.AC_PREFIX),
                "case": "输入图像尺寸不匹配",
                "handling": f"自动 resize 到 {res}x{res}" if res != "NOT_FOUND"
                            else "报错提示期望尺寸",
                "severity": "medium",
            })
            edges.append({
                "id": self._next_id(self.AC_PREFIX),
                "case": "损坏或无法读取的图像文件",
                "handling": "跳过并记录警告日志，不中断整体流程",
                "severity": "low",
            })
        else:
            edges.append({
                "id": self._next_id(self.AC_PREFIX),
                "case": "输入特征维度不匹配",
                "handling": "校验输入维度，不一致时报错提示期望维度",
                "severity": "medium",
            })
            edges.append({
                "id": self._next_id(self.AC_PREFIX),
                "case": "异常/缺失的数据样本",
                "handling": "跳过并记录警告日志，不中断整体流程",
                "severity": "low",
            })

        edges.append({
            "id": self._next_id(self.AC_PREFIX),
            "case": "GPU 不可用时降级到 CPU",
            "handling": "自动检测可用设备，GPU 不可用时使用 CPU 并提示性能下降",
            "severity": "medium",
        })

        edges.append({
            "id": self._next_id(self.AC_PREFIX),
            "case": "输入包含未知类别标签",
            "handling": "根据配置决定是报错还是归为 'unknown' 类",
            "severity": "low",
        })

        return edges

    # ============================================================
    # 可复现性约束
    # ============================================================

    def _extract_reproducibility(self, ki: dict) -> dict:
        impl = ki.get("implementation_details", {})
        repro = ki.get("reproduction_requirements", {})
        diff = repro.get("possible_difficulties", [])

        # 随机种子
        seed_mentioned = any("seed" in str(v).lower()
                             for v in [impl, repro]
                             if isinstance(v, (str, dict)))

        # 硬件
        hw = str(impl.get("hardware", "NOT_FOUND"))

        return {
            "random_seed": {
                "paper_specified": seed_mentioned,
                "recommendation": "论文未明确指定" if not seed_mentioned
                                  else "使用论文指定的种子值",
                "default": 42,
            },
            "hardware": {
                "paper_hardware": hw if hw != "NOT_FOUND" else "未提及",
                "note": "不同 GPU 架构可能导致微小精度差异",
            },
            "environment": {
                "python": "≥3.10",
                "key_dependencies": self._infer_dependencies(ki),
                "note": "建议使用 requirements.txt 或 conda environment.yml 锁定版本",
            },
            "known_difficulties": (
                [str(d) for d in diff] if isinstance(diff, list) and diff
                else ["论文未单独列出复现难点"]
            ),
            "suggested_validation_runs": (
                3 if seed_mentioned else 5
            ),
        }

    # ============================================================
    # 辅助
    # ============================================================

    def _has_training(self, impl: dict) -> bool:
        return any(
            impl.get(k, "NOT_FOUND") != "NOT_FOUND"
            for k in ["optimizer", "learning_rate", "epoch", "loss_function"]
        )

    @staticmethod
    def _extract_resize(ds: dict) -> str:
        t = str(ds.get("preprocessing", ""))
        for s in ["256", "224", "512", "299", "300", "227"]:
            if s in t:
                return s
        return "NOT_FOUND"

    def _infer_dependencies(self, ki: dict) -> list:
        """从论文推断关键依赖。非图像数据（合成/RL/无数据）不加 torchvision/pillow。"""
        deps = []
        impl = ki.get("implementation_details", {})
        fw = str(impl.get("programming_framework", "")).lower()
        is_image = self._is_image_data(ki.get("dataset", {}))

        if "pytorch" in fw or "torch" in fw:
            deps.append("torch")
            if is_image:
                deps.append("torchvision")
        elif "tensorflow" in fw or "tf" in fw:
            deps.append("tensorflow")
        elif "scikit" in fw or "sklearn" in fw:
            deps.append("scikit-learn")
        else:
            # 从真实模型名推断（排除 NOT_FOUND 占位键名，否则 str(meth) 里的
            # "alexnet"/"resnet" 等键名会误触发 torch 依赖）。
            meth = ki.get("methodology", {})
            arch = meth.get("model_architecture", {})
            if not isinstance(arch, dict):
                arch = {}
            model_names = []
            mf = str(meth.get("model", ""))
            if mf and mf != "NOT_FOUND":
                model_names += [p.strip() for p in mf.split(",") if p.strip()]
            for key in ["alexnet", "googlenet", "resnet", "vgg", "densenet",
                        "mobilenet", "efficientnet", "lstm", "transformer"]:
                v = arch.get(key, "NOT_FOUND")
                if v and v != "NOT_FOUND" and isinstance(v, str):
                    model_names.append(key)
            model_text = " ".join(model_names).lower()
            if any(k in model_text for k in
                   ["alexnet", "resnet", "googlenet", "vgg", "lstm"]):
                deps.append("torch")
                if is_image:
                    deps.append("torchvision")
            elif "caffe" in model_text:
                deps.append("caffe")

        # 无明确框架/模型信号时，不默认 torch（纯算法/无数据论文可能不需要）。
        # scikit-learn 仅在框架明确为 sklearn 时加入；torch/tensorflow 复现用各自评估
        # 函数，不需要 sklearn（否则 requirements.txt 无谓引入 sklearn，step8 依赖检查
        # 会因未安装 sklearn 而误跳过本可运行的 DL 项目）。

        deps.extend(["numpy", "pyyaml"])
        if is_image:
            deps.append("pillow")
        return sorted(set(deps))
