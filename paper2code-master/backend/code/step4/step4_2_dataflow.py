"""
Step4.2: 数据流与接口分析 → data_flow.yaml

定义数据在系统中的完整生命周期，以及模块间的接口契约。
每条接口引用 4.1 的功能需求 ID（如 IF-XX 连接 FR-01 和 FR-02）。

设计原则:
  - 数据流按 pipeline 阶段组织
  - 每个阶段明确输入数据格式、输出数据格式、关联的 FR
  - 接口契约定义模块间的数据传递规格
  - 阶段由「数据/模型/训练」三信号决定，纯算法论文不再套用 ML 模板
"""

from step4.base_requirement_analyzer import BaseRequirementAnalyzer


class DataFlowAnalyzer(BaseRequirementAnalyzer):

    def extract_by_rules(self) -> dict:
        ki = self.key_info
        self._reset_counter()

        meth = ki.get("methodology", {})
        ds = ki.get("dataset", {})
        impl = ki.get("implementation_details", {})
        arch = meth.get("model_architecture", {})
        if not isinstance(arch, dict):
            arch = {}

        # 图像 vs 非图像：非图像数据（合成/RL/无数据）不套用 PIL/ImageNet/3 通道模板
        is_image = self._is_image_data(ds)

        # 4.1 的 FR 列表 → {name: id}，用于 related_fr 引用（避免硬编码 FR-01..FR-10 悬空）
        fr_map = self._load_fr_map()

        # ---- 构建 pipeline 阶段 ----
        stages = self._build_stages(ki, meth, ds, impl, arch, is_image, fr_map)

        # ---- 构建接口契约 ----
        num_classes = self._extract_num_classes(ds)
        res = self._extract_resize(ds)
        interfaces = self._build_interfaces(stages, num_classes, res, is_image)

        return {
            "paper_name": self.paper_name,
            "pipeline": stages,
            "interfaces": interfaces,
            "data_formats": self._infer_data_formats(ds, arch, is_image),
        }

    # ============================================================
    # Pipeline 阶段
    # ============================================================

    def _load_fr_map(self) -> dict:
        """加载 4.1 产出的 FR 列表，构建 {FR name: FR id} 映射。"""
        fr_data = self.load_yaml("functional_requirements.yaml")
        reqs = fr_data.get("requirements", [])
        return {r.get("name", ""): r.get("id") for r in reqs if r.get("id")}

    @staticmethod
    def _detect_model_names(meth: dict, arch: dict) -> list:
        """与 4.1 相同的模型名检测逻辑，用于判断是否存在模型。"""
        names = []
        for key in ["alexnet", "googlenet", "resnet", "vgg", "densenet",
                    "mobilenet", "efficientnet", "lstm", "transformer"]:
            detail = arch.get(key, "NOT_FOUND")
            if detail and detail != "NOT_FOUND" and isinstance(detail, str):
                names.append(key.title())
        if not names:
            mf = meth.get("model", "")
            if mf and mf != "NOT_FOUND":
                names = [n.strip() for n in mf.split(",") if n.strip()]
        return names

    def _build_stages(self, ki: dict, meth: dict, ds: dict, impl: dict,
                      arch: dict, is_image: bool = False,
                      fr_map: dict = None) -> list:
        """按「数据 / 模型 / 训练」三信号拼装 pipeline 阶段。"""
        fr_map = fr_map or {}
        stages = []

        ds_name = str(ds.get("dataset_name", "NOT_FOUND"))
        has_data = ds_name != "NOT_FOUND"
        has_training = self._has_training(impl)

        model_names = self._detect_model_names(meth, arch)
        has_model = bool(model_names)
        method_name = meth.get("method_name", "")
        has_method = (
            not has_model and bool(method_name)
            and method_name != "NOT_FOUND"
        )

        # related_fr 引用：按名字映射到 4.1 实际生成的 FR id
        def fr(*names):
            return [fr_map[n] for n in names if fr_map.get(n)]

        def fr_like(suffix=None, contains=None):
            out = []
            for n, fid in fr_map.items():
                if suffix and n.endswith(suffix):
                    out.append(fid)
                elif contains and contains in n:
                    out.append(fid)
            return out

        stage_no = 1

        # ---- 数据加载 / 预处理（仅当有数据） ----
        if has_data:
            pp = str(ds.get("preprocessing", ""))
            res = self._extract_resize(ds)
            stages.append({
                "stage": stage_no,
                "name": "数据加载",
                "description": "从原始数据源读取数据到内存",
                "entry_data": self._pick(ds.get("dataset_name"),
                                         "原始图像文件" if is_image else "原始数据"),
                "exit_data": ("原始图像 (PIL.Image / numpy.ndarray)"
                              if is_image else "数据样本（张量或结构化数据）"),
                "related_fr": fr("数据集加载"),
                "constraints": self._extract_data_constraints(ds),
            })
            stage_no += 1

            stages.append({
                "stage": stage_no,
                "name": "数据预处理",
                "description": pp if pp and pp != "NOT_FOUND" else (
                    "图像预处理管道" if is_image else "标准化/归一化"),
                "entry_data": "原始图像" if is_image else "原始数据样本",
                "exit_data": (
                    f"归一化张量 (B, 3, {res}, {res})"
                    if is_image and res != "NOT_FOUND"
                    else ("归一化张量" if is_image
                          else "标准化后的特征张量 (B, F)")
                ),
                "related_fr": fr("数据预处理"),
                "detail": {
                    "resize": f"{res}x{res}" if is_image and res != "NOT_FOUND"
                              else "NOT_FOUND",
                    "augmentation": self._extract_augmentation(pp),
                    "normalization": (
                        "ImageNet 均值/标准差 (如使用预训练权重)"
                        if is_image and arch.get("pretrained_model", "NOT_FOUND") != "NOT_FOUND"
                        else ("按数据语义做标准化（z-score/归一化），无 RGB 通道概念"
                              if not is_image else "NOT_FOUND")
                    ),
                },
            })
            stage_no += 1

        # ---- 模型构建 / 算法执行 ----
        if has_model:
            stages.append({
                "stage": stage_no,
                "name": "模型构建（模型定义）",
                "description": "构建模型结构，接收预处理输入并输出预测结果",
                "entry_data": ("归一化张量 (B, C, H, W)" if is_image
                               else "标准化后的特征张量 (B, F)"),
                "exit_data": self._pick(arch.get("output"),
                                        "分类概率 (B, num_classes)" if is_image
                                        else "预测输出 (B, num_classes)"),
                "related_fr": fr_like(suffix="模型实现"),
            })
            stage_no += 1
        elif has_method:
            stages.append({
                "stage": stage_no,
                "name": "算法执行",
                "description": f"执行论文方法: {method_name}",
                "entry_data": "构造的输入数据",
                "exit_data": "算法输出结果",
                "related_fr": fr("算法/方法实现"),
            })
            stage_no += 1

        # ---- 模型训练（仅当有训练） ----
        if has_training:
            stages.append({
                "stage": stage_no,
                "name": "模型训练",
                "description": (
                    f"使用 {impl.get('optimizer', 'NOT_FOUND')} "
                    f"训练 {impl.get('epoch', 'NOT_FOUND')} 个 epoch"
                ),
                "entry_data": "训练集 DataLoader",
                "exit_data": "训练好的模型权重 + 训练日志",
                "related_fr": fr("模型训练", "模型持久化")
                              + fr_like(contains="训练模式支持"),
                "detail": {
                    "loss": self._get_loss_impl(impl),
                    "metrics": self._get_training_metrics(ki),
                },
                "training_loop": self._training_loop_detail(),
            })
            stage_no += 1

        # ---- 评估 / 结果验证（有模型或方法时） ----
        if has_model or has_method:
            stages.append({
                "stage": stage_no,
                "name": "模型评估",
                "description": "在测试集上计算最终指标" if has_model
                               else "验证算法结果与论文结论一致",
                "entry_data": ("测试集 DataLoader + 训练好的模型" if has_model
                               else "算法输出 + 预期结果"),
                "exit_data": "评估报告（指标数值 + 可视化）" if has_model
                             else "验证报告（结果一致性）",
                "related_fr": fr("模型评估", "论文结果复现与报告"),
            })

        return stages

    def _training_loop_detail(self) -> dict:
        """训练循环伪代码（图像/非图像共用，与 4.2 原逻辑一致）。"""
        return {
            "per_epoch": [
                {
                    "phase": "train",
                    "description": "遍历 train_loader，计算 loss 并反向传播",
                    "steps": [
                        "for batch_idx, (inputs, labels) in enumerate(train_loader):",
                        "  inputs, labels = inputs.to(device), labels.to(device)",
                        "  optimizer.zero_grad()",
                        "  outputs = model(inputs)",
                        "  loss = criterion(outputs, labels)",
                        "  loss.backward()",
                        "  optimizer.step()",
                        "  train_loss.update(loss.item())",
                    ],
                },
                {
                    "phase": "log",
                    "description": "记录每个 epoch 的 training loss 和 learning rate",
                    "steps": [
                        "logger.info(f'Epoch {epoch}: train_loss={avg_loss:.4f}, lr={current_lr}')",
                    ],
                },
            ],
            "post_epoch": [
                {
                    "phase": "validation",
                    "description": "可选验证阶段。若论文无独立验证集；如需早停或模型选择，应从训练集二次划分。",
                    "condition": "if val_loader is not None (engineering deviation)",
                    "steps": [
                        "model.eval()",
                        "with torch.no_grad():",
                        "  for inputs, labels in val_loader:",
                        "    outputs = model(inputs)",
                        "    val_metrics.update(outputs, labels)",
                        "model.train()",
                    ],
                },
                {
                    "phase": "scheduler_step",
                    "description": "每个 epoch 后更新学习率",
                    "steps": [
                        "scheduler.step()  # StepLR: 每 step_size 个 epoch 衰减 gamma 倍",
                    ],
                },
                {
                    "phase": "checkpoint",
                    "description": "保存最优模型和定期 checkpoint",
                    "steps": [
                        "if val_metric > best_metric:",
                        "  save_checkpoint(model, optimizer, epoch, 'best.pth')",
                        "  best_metric = val_metric",
                        "if epoch % save_frequency == 0:",
                        "  save_checkpoint(model, optimizer, epoch, f'epoch_{epoch}.pth')",
                    ],
                },
            ],
        }

    # ============================================================
    # 接口契约
    # ============================================================

    def _build_interfaces(self, stages: list, num_classes: str, res: str,
                          is_image: bool = False) -> list:
        """构建详细的接口契约，包含数据格式、shape、验证方式"""
        label_range = (
            f"0-{int(num_classes) - 1}" if num_classes != "NOT_FOUND" else "0-(N-1)"
        )
        # 非图像数据走通用接口模板（无 PIL/3 通道/灰度 repeat/GoogLeNet aux）
        if not is_image:
            return self._build_generic_interfaces(stages, num_classes, label_range)

        templates = {
            ("数据加载", "数据预处理"): {
                "data": "原始图像 (PIL.Image / numpy.ndarray)",
                "format": "PIL.Image (mode='RGB') 或 np.ndarray (H, W, 3), dtype=uint8",
                "shape": "(H, W, 3) — 可变尺寸，在预处理阶段统一 resize",
                "contract": (
                    "数据加载必须输出有效的图像对象（非 None，可正常解码）。"
                    "损坏文件应跳过并记录日志，不中断整个流程。"
                    f"标签保持原始格式（整数 {label_range}），在预处理阶段不做 one-hot 转换。"
                ),
                "validation": "assert isinstance(img, (PIL.Image.Image, np.ndarray)); assert img is not None",
            },
            ("数据预处理", "模型构建（模型定义）"): {
                "data": f"归一化张量 (B, 3, {res}, {res})",
                "format": "torch.Tensor, dtype=torch.float32, device 与模型一致",
                "shape": f"(B, 3, {res}, {res}) — 灰度版本需 repeat 为 3 通道",
                "contract": (
                    "预处理输出必须为归一化到 [0,1] 或 ImageNet 标准化后的张量。"
                    "灰度版本必须先 repeat 为 3 通道再输入模型。"
                    "分割版本沿目标轮廓裁剪后 resize，填充背景为均值。"
                ),
                "validation": (
                    f"assert tensor.shape == (B, 3, {res}, {res}); "
                    "assert tensor.dtype == torch.float32; "
                    "assert tensor.min() >= -3.0 and tensor.max() <= 3.0"
                ),
            },
            ("模型构建（模型定义）", "模型训练"): {
                "data": f"{num_classes}类概率分布 + (可选: 带辅助分类器模型的 aux1_logits, aux2_logits)",
                "format": (
                    f"torch.Tensor (B, {num_classes}) — 主分类器输出 "
                    "或 tuple(main, aux1, aux2) — 带辅助分类器模型的训练模式"
                ),
                "shape": f"主输出 (B, {num_classes})；辅助分类器输出同为 (B, {num_classes})×2",
                "contract": (
                    "前向传播返回 raw logits（非 softmax 概率），由 CrossEntropyLoss 内部处理。"
                    "带辅助分类器的模型（如 GoogLeNet）训练模式下，__init__ 设置 aux_logits=True，forward 返回三元组。"
                    "推理/评估模式下，aux_logits=False，forward 返回单一输出。"
                    "这是带辅助分类器模型与普通模型在代码层面的关键差异。"
                ),
                "validation": (
                    "训练: assert isinstance(output, tuple) and len(output) == 3 "
                    f"(带辅助分类器模型); assert output[0].shape == (B, {num_classes}) "
                    f"推理: assert output.shape == (B, {num_classes})"
                ),
            },
            ("模型训练", "模型评估"): {
                "data": "训练好的模型权重 + 训练日志（loss/accuracy 曲线数据）",
                "format": (
                    "模型权重: dict (state_dict) — torch.save(model.state_dict(), path)\n"
                    "训练日志: JSON/dict — {train_loss: [...], train_acc: [...], "
                    "(val_loss: [...], val_acc: [...])}"
                ),
                "shape": "N/A — 序列化格式",
                "contract": (
                    "保存的 state_dict 必须能用 model.load_state_dict() 正确恢复。"
                    "保存前确认 model 在 CPU 上 (model.cpu().state_dict())。"
                    "检查点需同时保存 optimizer.state_dict() 和当前 epoch，"
                    "以支持训练中断后恢复。"
                    "训练日志保存为 JSON 以支持 Step5 的可视化需求。"
                ),
                "validation": (
                    "restored = ModelClass(**config); "
                    "restored.load_state_dict(torch.load(path)); "
                    "assert torch.allclose(restored(input), original(input), atol=1e-6)"
                ),
            },
        }

        interfaces = []
        for i in range(len(stages) - 1):
            s_curr = stages[i]
            s_next = stages[i + 1]
            key = (s_curr["name"], s_next["name"])
            template = templates.get(key)

            if template:
                interfaces.append({
                    "id": self._next_id(self.IF_PREFIX),
                    "from_stage": s_curr["name"],
                    "to_stage": s_next["name"],
                    "data": template["data"],
                    "format": template["format"],
                    "shape": template.get("shape", "N/A"),
                    "contract": template["contract"],
                    "validation": template.get("validation", "N/A"),
                })
            else:
                # Fallback
                interfaces.append({
                    "id": self._next_id(self.IF_PREFIX),
                    "from_stage": s_curr["name"],
                    "to_stage": s_next["name"],
                    "data": s_curr["exit_data"],
                    "format": self._infer_format(s_curr["exit_data"]),
                    "contract": (
                        f"{s_curr['name']} 的输出必须满足 "
                        f"{s_next['name']} 的输入要求"
                    ),
                })

        return interfaces

    def _build_generic_interfaces(self, stages: list, num_classes: str,
                                  label_range: str) -> list:
        """非图像数据的通用接口契约（无 PIL/3 通道/灰度 repeat/GoogLeNet aux 概念）。"""
        templates = {
            ("数据加载", "数据预处理"): {
                "data": "数据样本（张量或结构化数据）",
                "format": "torch.Tensor 或 numpy.ndarray，dtype 与任务一致",
                "shape": "(B, F) — 特征维度 F 由数据决定，无图像通道概念",
                "contract": (
                    f"数据加载必须输出有效的样本（非 None）。标签保持原始格式（整数 {label_range}），"
                    "预处理阶段不做 one-hot 转换。"
                ),
                "validation": "assert sample is not None",
            },
            ("数据预处理", "模型构建（模型定义）"): {
                "data": "标准化后的特征张量 (B, F)",
                "format": "torch.Tensor, dtype=torch.float32, device 与模型一致",
                "shape": "(B, F)",
                "contract": "预处理输出必须为标准化/归一化后的特征张量。",
                "validation": (
                    "assert tensor.dtype == torch.float32; "
                    "assert tensor.ndim == 2"
                ),
            },
            ("模型构建（模型定义）", "模型训练"): {
                "data": f"预测输出 (B, {num_classes})",
                "format": f"torch.Tensor (B, {num_classes})",
                "shape": f"(B, {num_classes})",
                "contract": "前向传播返回预测输出，由损失函数内部处理。",
                "validation": f"assert output.shape == (B, {num_classes})",
            },
            ("模型训练", "模型评估"): {
                "data": "训练好的模型权重 + 训练日志（loss/accuracy 曲线数据）",
                "format": (
                    "模型权重: dict (state_dict) — torch.save(model.state_dict(), path)\n"
                    "训练日志: JSON/dict — {train_loss: [...], train_acc: [...], "
                    "(val_loss: [...], val_acc: [...])}"
                ),
                "shape": "N/A — 序列化格式",
                "contract": (
                    "保存的 state_dict 必须能用 model.load_state_dict() 正确恢复。"
                    "保存前确认 model 在 CPU 上 (model.cpu().state_dict())。"
                    "检查点需同时保存 optimizer.state_dict() 和当前 epoch，"
                    "以支持训练中断后恢复。"
                ),
                "validation": (
                    "restored = ModelClass(**config); "
                    "restored.load_state_dict(torch.load(path)); "
                    "assert torch.allclose(restored(input), original(input), atol=1e-6)"
                ),
            },
        }

        interfaces = []
        for i in range(len(stages) - 1):
            s_curr = stages[i]
            s_next = stages[i + 1]
            key = (s_curr["name"], s_next["name"])
            template = templates.get(key)
            if template:
                interfaces.append({
                    "id": self._next_id(self.IF_PREFIX),
                    "from_stage": s_curr["name"],
                    "to_stage": s_next["name"],
                    "data": template["data"],
                    "format": template["format"],
                    "shape": template.get("shape", "N/A"),
                    "contract": template["contract"],
                    "validation": template.get("validation", "N/A"),
                })
            else:
                interfaces.append({
                    "id": self._next_id(self.IF_PREFIX),
                    "from_stage": s_curr["name"],
                    "to_stage": s_next["name"],
                    "data": s_curr["exit_data"],
                    "format": self._infer_format(s_curr["exit_data"]),
                    "contract": (
                        f"{s_curr['name']} 的输出必须满足 "
                        f"{s_next['name']} 的输入要求"
                    ),
                })
        return interfaces

    # ============================================================
    # 数据格式推断
    # ============================================================

    def _infer_data_formats(self, ds: dict, arch: dict,
                            is_image: bool = False) -> dict:
        formats = {}

        num_classes = self._extract_num_classes(ds)
        res = self._extract_resize(ds)

        if is_image:
            formats["input_image"] = {
                "type": "JPEG/PNG 图像文件",
                "expected_size": f"{res}x{res}" if res != "NOT_FOUND" else "可变",
            }
            formats["tensor"] = {
                "type": "torch.Tensor (float32)",
                "shape": {
                    "single": f"(3, {res}, {res})" if res != "NOT_FOUND" else "待定",
                    "batch": f"(B, 3, {res}, {res})" if res != "NOT_FOUND" else "待定",
                },
            }
        else:
            formats["input_data"] = {
                "type": "结构化数据/特征向量（非图像）",
                "expected_size": "特征维度 F 由数据决定",
            }
            formats["tensor"] = {
                "type": "torch.Tensor (float32)",
                "shape": {
                    "single": "(F,)",
                    "batch": "(B, F)",
                },
            }

        if num_classes != "NOT_FOUND":
            formats["output"] = {
                "type": "torch.Tensor (float32)",
                "shape": f"(B, {num_classes})",
                "description": f"{num_classes} 类分类概率分布",
            }

        return formats

    # ============================================================
    # 辅助
    # ============================================================

    def _has_training(self, impl: dict) -> bool:
        return any(
            impl.get(k, "NOT_FOUND") != "NOT_FOUND"
            for k in ["optimizer", "learning_rate", "epoch", "loss_function"]
        )

    def _extract_resize(self, ds: dict) -> str:
        t = str(ds.get("preprocessing", ""))
        for s in ["256", "224", "512", "299", "300", "227"]:
            if s in t:
                return s
        return "NOT_FOUND"

    def _extract_data_constraints(self, ds: dict) -> list:
        constraints = []
        split = str(ds.get("train_test_split", "")).lower()
        if "same leaf" in split:
            constraints.append("同一叶子的多张图像必须保持在同一集合中")
        if "stratified" in split:
            constraints.append("分层抽样，保持类别分布一致")
        return constraints

    def _extract_augmentation(self, pp: str) -> list:
        aug = []
        pp_lower = pp.lower()
        if "flip" in pp_lower or "horizontal" in pp_lower:
            aug.append("RandomHorizontalFlip")
        if "rotation" in pp_lower or "rotate" in pp_lower:
            aug.append("RandomRotation")
        if "crop" in pp_lower:
            aug.append("RandomResizedCrop")
        if "color" in pp_lower or "brightness" in pp_lower:
            aug.append("ColorJitter")
        return aug

    def _get_loss_impl(self, impl: dict) -> str:
        loss = str(impl.get("loss_function", "NOT_FOUND"))
        if "cross" in loss.lower() and "entropy" in loss.lower():
            return "torch.nn.CrossEntropyLoss()"
        if "mse" in loss.lower():
            return "torch.nn.MSELoss()"
        return loss

    def _get_training_metrics(self, ki: dict) -> list:
        exp = ki.get("experiment", {})
        metrics = str(exp.get("evaluation_metrics", ""))
        if metrics and metrics != "NOT_FOUND":
            return [m.strip() for m in metrics.split(",") if m.strip()][:6]
        return ["Loss", "Accuracy"]

    @staticmethod
    def _infer_format(text: str) -> str:
        """从描述文本推断数据格式"""
        text_lower = text.lower()
        if "张量" in text or "tensor" in text_lower:
            return "torch.Tensor"
        if "图像" in text or "image" in text_lower:
            return "PIL.Image / np.ndarray"
        if "权重" in text or "weight" in text_lower:
            return "dict (state_dict)"
        if "json" in text_lower:
            return "JSON"
        if "csv" in text_lower:
            return "CSV"
        return "待定"
