"""
Step4.1: 功能需求分析 → functional_requirements.yaml

从 key_information.json 推导系统必须实现的功能，
每条需求包含: id, name, description, input, output, source, verification

设计原则:
  - 每个需求必须有论文出处（source）
  - 输入/输出明确，方便 Step5 确定接口
  - 验证方式具体可执行
"""

import json
from step4.base_requirement_analyzer import BaseRequirementAnalyzer
from logger import logger


class FunctionalAnalyzer(BaseRequirementAnalyzer):

    def extract_by_rules(self) -> dict:
        ki = self.key_info
        self._reset_counter()

        requirements = []

        # ---- 1. 数据加载需求 ----
        requirements.extend(self._extract_data_requirements(ki))

        # ---- 2. 模型需求 ----
        requirements.extend(self._extract_model_requirements(ki))

        # ---- 3. 训练需求 ----
        requirements.extend(self._extract_training_requirements(ki))

        # ---- 4. 评估需求 ----
        requirements.extend(self._extract_evaluation_requirements(ki))

        return {
            "paper_name": self.paper_name,
            "requirements": requirements,
            "summary": {
                "total": len(requirements),
                "categories": self._categorize(requirements),
            },
        }

    # ============================================================
    # 数据加载需求
    # ============================================================

    def _extract_data_requirements(self, ki: dict) -> list:
        reqs = []
        ds = ki.get("dataset", {})

        ds_name = str(ds.get("dataset_name", "NOT_FOUND"))
        ds_size = str(ds.get("dataset_size", "NOT_FOUND"))
        preprocessing = str(ds.get("preprocessing", "NOT_FOUND"))
        split = str(ds.get("train_test_split", "NOT_FOUND"))
        num_classes = str(ds.get("number_of_classes", "NOT_FOUND"))

        # 图像 vs 非图像：非图像数据（合成向量/RL 状态/无数据）不套用"原始图像文件"模板
        is_image = self._is_image_data(ds)

        # FR: 数据加载
        if ds_name != "NOT_FOUND":
            if is_image:
                load_desc = f"加载 {ds_name} 数据集（{ds_size}），支持从本地路径读取原始图像文件"
                load_output = f"原始图像及其标签（{num_classes} 类）"
            else:
                load_desc = f"加载/构造 {ds_name} 数据集（{ds_size}）"
                load_output = f"数据样本及其标签（{num_classes} 类）"
            reqs.append({
                "id": self._next_id(self.FR_PREFIX),
                "name": "数据集加载",
                "description": load_desc,
                "input": "数据集根目录路径" if is_image else "数据集来源（本地路径或代码内生成）",
                "output": load_output,
                "source": self._source("dataset", "dataset_name"),
                "verification": f"加载后样本数应与 {ds_size} 一致",
            })

        # FR: 预处理管道
        if preprocessing != "NOT_FOUND":
            reqs.append({
                "id": self._next_id(self.FR_PREFIX),
                "name": "数据预处理",
                "description": f"预处理步骤: {preprocessing}",
                "input": "原始图像" if is_image else "原始数据样本",
                "output": ("归一化张量，尺寸符合模型输入要求" if is_image
                           else "标准化后的特征张量"),
                "source": self._source("dataset", "preprocessing"),
                "verification": "检查输出张量的 shape 和数值范围",
            })

        # FR: 数据集划分
        if split != "NOT_FOUND":
            reqs.append({
                "id": self._next_id(self.FR_PREFIX),
                "name": "数据集划分",
                "description": f"按 {split} 划分训练/测试集",
                "input": "全部样本列表",
                "output": "train_loader, test_loader",
                "source": self._source("dataset", "train_test_split"),
                "verification": "各集合样本数符合设定比例",
                "design_decision": (
                    "论文仅使用 train/test 二路划分，无独立验证集。"
                    "若需要 val_loader 用于早停(early stopping)或最优模型选择，"
                    "应从训练集中按 80/20 二次划分出验证集，"
                    "并标记为工程偏离(engineering deviation from paper)。"
                ),
            })

        return reqs

    # ============================================================
    # 模型需求
    # ============================================================

    def _extract_model_requirements(self, ki: dict) -> list:
        reqs = []
        meth = ki.get("methodology", {})
        arch = meth.get("model_architecture", {})
        if not isinstance(arch, dict):
            arch = {}
        ds = ki.get("dataset", {})
        is_image = self._is_image_data(ds)

        # FR: 模型定义
        model_names = []
        for model_key in ["alexnet", "googlenet", "resnet", "vgg", "densenet",
                          "mobilenet", "efficientnet", "lstm", "transformer"]:
            detail = arch.get(model_key, "NOT_FOUND")
            if detail and detail != "NOT_FOUND" and isinstance(detail, str):
                model_names.append(model_key.title())

        if not model_names:
            # 从 methodology.model 提取
            mf = meth.get("model", "")
            if mf and mf != "NOT_FOUND":
                model_names = [n.strip() for n in mf.split(",") if n.strip()]

        for name in model_names:
            reqs.append({
                "id": self._next_id(self.FR_PREFIX),
                "name": f"{name} 模型实现",
                "description": f"构建 {name} 模型，能够接收预处理后的输入并输出预测结果",
                "input": self._pick(
                    arch.get("input"),
                    "预处理后的图像张量" if is_image else "预处理后的特征张量"),
                "output": self._pick(
                    arch.get("output"),
                    ("分类概率分布" if self._extract_num_classes(ds) != "NOT_FOUND"
                     else "预测输出")),
                "source": self._source("methodology", "model_architecture",
                                       name.lower()),
                "verification": (
                    "模型可正常实例化并通过前向传播；参数量在合理数量级范围内 "
                    "（与参考实现同一量级），但不同框架实现可能导致 "
                    "±5% 差异（如归一化层、padding 策略差异），"
                    "不要求逐参数精确匹配。"
                ),
            })

        # FR: 无 ML 模型时（纯算法论文，method_name 是「方法」而非「模型」），
        # 补一条通用「算法/方法实现」需求，避免整篇需求为空。
        if not model_names:
            method_name = meth.get("method_name", "")
            if method_name and method_name != "NOT_FOUND":
                reqs.append({
                    "id": self._next_id(self.FR_PREFIX),
                    "name": "算法/方法实现",
                    "description": f"实现论文提出的方法: {method_name}",
                    "input": self._pick(arch.get("input"), "构造的输入数据"),
                    "output": self._pick(arch.get("output"), "算法输出结果"),
                    "source": self._source("methodology", "method_name"),
                    "verification": (
                        "算法可运行并产出预期格式的输出，与论文示例/结论一致"
                    ),
                })

        # FR: 训练模式支持（从头训练 + 迁移学习）
        # 两种模式必须显式分离，不能用一个 pretrained=True 全局开关混过去。
        # 迁移学习 FR 仅在论文确实使用预训练模型时出现；从头训练的 MLP/DQN、
        # 纯理论论文没有迁移学习，不生成，避免凭空捏造 ImageNet 预训练权重。
        pretrained = arch.get("pretrained_model", "NOT_FOUND")
        has_training = any(
            ki.get("implementation_details", {}).get(k, "NOT_FOUND") != "NOT_FOUND"
            for k in ["optimizer", "learning_rate", "epoch"]
        )
        if has_training and pretrained != "NOT_FOUND":
            transfer_desc = (
                f"(b) 迁移学习(Transfer): 加载 ImageNet 预训练权重，重置最后一层"
                f"({pretrained})，其余层可选冻结或微调。"
                if is_image
                else f"(b) 迁移学习(Transfer): 加载预训练权重（{pretrained}），"
                     "重置输出层，其余层可选冻结或微调。"
            )
            reqs.append({
                "id": self._next_id(self.FR_PREFIX),
                "name": "训练模式支持（从头训练 / 迁移学习）",
                "description": (
                    "支持两种独立的训练模式: "
                    "(a) 从头训练(Scratch): 所有层随机初始化（Kaiming/Xavier），不使用预训练权重；"
                    f"{transfer_desc}"
                ),
                "input": "training_mode 参数 (scratch|transfer), 模型定义, 预训练权重路径(仅transfer)",
                "output": "按指定模式初始化好的模型参数",
                "source": self._source("methodology", "model_architecture",
                                       "pretrained_model"),
                "verification": (
                    "scratch 模式下所有权重随机且不含预训练统计量；"
                    "transfer 模式下除最后一层外与预训练权重一致，"
                    "最后一层为随机初始化。"
                ),
                "implementation_notes": (
                    "⚠ 不能使用单一的 pretrained=True/False 布尔值，"
                    "因为两种模式的初始化逻辑完全不同。"
                    "建议使用 training_mode 枚举 (scratch/transfer) 而非 pretrained 布尔。"
                ),
            })

        return reqs

    # ============================================================
    # 训练需求
    # ============================================================

    def _extract_training_requirements(self, ki: dict) -> list:
        reqs = []
        impl = ki.get("implementation_details", {})

        has_training = any(
            impl.get(k, "NOT_FOUND") != "NOT_FOUND"
            for k in ["optimizer", "learning_rate", "epoch", "loss_function"]
        )

        if not has_training:
            return reqs

        # FR: 训练循环
        opt = impl.get("optimizer", "NOT_FOUND")
        lr = impl.get("learning_rate", "NOT_FOUND")
        epoch = impl.get("epoch", "NOT_FOUND")
        bs = impl.get("batch_size", "NOT_FOUND")
        loss_fn = impl.get("loss_function", "NOT_FOUND")

        reqs.append({
            "id": self._next_id(self.FR_PREFIX),
            "name": "模型训练",
            "description": (
                f"使用 {opt} 优化器（lr={lr}）训练模型 {epoch} 个 epoch，"
                f"batch_size={bs}，损失函数为 {loss_fn}。"
            ),
            "input": "训练数据加载器、初始化好的模型",
            "output": "训练好的模型权重 + 训练日志（loss/accuracy 曲线）",
            "source": self._source("implementation_details"),
            "verification": (
                f"训练 {epoch} epoch 能完整跑通（无中断、无显存溢出），"
                "loss 呈下降趋势。"
                "不要求最终 accuracy 接近论文报告值"
                "（本阶段只保证训练流程正确，不包含论文原始数据集的完整复现）。"
            ),
        })

        # FR: Checkpoint 保存
        reqs.append({
            "id": self._next_id(self.FR_PREFIX),
            "name": "模型持久化",
            "description": "保存训练过程中最优模型权重和最终模型权重",
            "input": "模型 state_dict、优化器状态、训练元信息",
            "output": ".pth 或 .pt 权重文件",
            "source": self._source("reproduction_requirements"),
            "verification": "加载保存的权重后可恢复推理，精度不下降",
        })

        return reqs

    # ============================================================
    # 评估需求
    # ============================================================

    def _extract_evaluation_requirements(self, ki: dict) -> list:
        reqs = []
        exp = ki.get("experiment", {})

        metrics = str(exp.get("evaluation_metrics", "NOT_FOUND"))
        results = str(exp.get("important_results", "NOT_FOUND"))

        if metrics != "NOT_FOUND":
            reqs.append({
                "id": self._next_id(self.FR_PREFIX),
                "name": "模型评估",
                "description": f"在测试集上计算评估指标: {metrics}",
                "input": "训练好的模型、测试数据加载器",
                "output": f"指标报告: {metrics}",
                "source": self._source("experiment", "evaluation_metrics"),
                "verification": f"输出 {metrics} 的值，与论文报告值对比",
            })

        if results != "NOT_FOUND":
            reqs.append({
                "id": self._next_id(self.FR_PREFIX),
                "name": "论文结果复现与报告",
                "description": (
                    f"[论文要求] 计算并报告: {metrics}。"
                    f"关键发现: {results[:300]}。"
                    "[工程辅助] 附加可视化（非论文要求，用于调试和验证）："
                    "confusion matrix、per-class accuracy bar chart、"
                    "training curve (loss/accuracy over epochs)。"
                    "——论文只要求数值指标表格，图表为工程附加。"
                ),
                "input": "评估指标数值、预测结果",
                "output": {
                    "paper_required": {
                        "format": "数值表格或文本报告",
                        "content": f"{metrics} 的数值",
                        "purpose": "与论文 Table 对比",
                    },
                    "engineering_auxiliary": {
                        "format": "PNG/SVG 图表",
                        "content": "混淆矩阵、各类别准确率柱状图、训练曲线",
                        "purpose": "调试验证和结果展示",
                    },
                },
                "source": self._source("experiment", "important_results"),
                "verification": (
                    "[论文要求] 评估指标值与论文报告值在容差范围内一致；"
                    "[工程辅助] 图表可正确生成，不要求与论文图表逐像素一致"
                ),
            })

        return reqs

    # ============================================================
    # 统计
    # ============================================================

    def _categorize(self, requirements: list) -> dict:
        cats = {}
        for r in requirements:
            # 从 source 提取一级分类
            src = r.get("source", "other")
            top = src.split(".")[0] if "." in src else src
            cats[top] = cats.get(top, 0) + 1
        return cats
