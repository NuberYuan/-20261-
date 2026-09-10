"""Step5.4: 模型结构解析 → model_design.yaml"""
from step5.base_analyzer import BaseAnalyzer

CNN_KW = ["cnn", "convolution", "alexnet", "googlenet", "resnet", "vgg",
          "densenet", "mobilenet", "efficientnet", "convnext"]
RNN_KW = ["lstm", "gru", "rnn", "recurrent"]
TRANSFORMER_KW = ["transformer", "attention", "bert", "gpt", "vit",
                  "vision transformer", "llama"]
VLM_KW = ["vlm", "vision language", "llm", "qwen", "smolvlm"]
MLP_KW = ["mlp", "perceptron", "multi-layer", "multi layer",
          "fully connected", "fully-connected", "feedforward", "feed-forward"]


class ModelAnalyzer(BaseAnalyzer):

    def extract_by_rules(self) -> dict:
        meth = self.key_info.get("methodology", {})
        ds = self.key_info.get("dataset", {})
        impl = self.key_info.get("implementation_details", {})
        arch = meth.get("model_architecture", {})
        if not isinstance(arch, dict):
            arch = {}

        models = []

        # 从 model_architecture 的子 key 中提取具体模型
        arch_models = self._extract_arch_models(arch, ds)

        # 从 methodology.model 或 method_name 提取（仅当名字确实像 ML 模型时）
        mf = meth.get("model", "")
        method_name = meth.get("method_name", "")

        if arch_models:
            models = arch_models
        elif mf and mf != "NOT_FOUND" and self._looks_like_model(mf, arch, impl):
            for p in mf.replace(" and ", ", ").split(","):
                p = p.strip().strip("()")
                if p and len(p) > 2 and "not" not in p.lower():
                    models.append(self._build_model_entry(
                        p, arch, ds, meth
                    ))
        elif method_name and self._looks_like_model(method_name, arch, impl):
            models.append(self._build_model_entry(
                method_name, arch, ds, meth
            ))

        # 无法判定为 ML 模型时保持空列表（纯算法/无数据集论文的 method_name
        # 是「方法」而非「模型」），不再塞 "Unknown/Model" 占位，
        # 避免下游据此强制生成 models/ 文件。
        return {"paper_name": self.paper_name, "models": models}

    def _looks_like_model(self, name: str, arch: dict, impl: dict) -> bool:
        """判断一个名称是否确实指代 ML 模型（而非纯算法方法名）。

        任一命中即视为模型：
          1) 名字含已知 NN 类型关键词（CNN/RNN/Transformer/VLM/MLP）
          2) 架构里用了预训练权重
          3) 论文有训练信号（optimizer/lr/epoch/loss）
        """
        if self._infer_type(name.lower()) != "other":
            return True
        if self._pick(arch.get("pretrained_model")) != "NOT_FOUND":
            return True
        if any(
            impl.get(k, "NOT_FOUND") != "NOT_FOUND"
            for k in ["optimizer", "learning_rate", "epoch", "loss_function"]
        ):
            return True
        return False

    # =========================
    # 从 architecture 字段提取
    # =========================

    def _extract_arch_models(self, arch: dict, ds: dict) -> list:
        """从 model_architecture 的子 key 中提取模型列表，并结构化架构细节"""
        known_arch_keys = [
            "alexnet", "googlenet", "resnet", "vgg", "densenet",
            "mobilenet", "efficientnet", "convnext",
            "lstm", "gru", "transformer", "attention",
        ]
        models = []

        for key in known_arch_keys:
            detail = arch.get(key, "")
            if detail and detail != "NOT_FOUND" and isinstance(detail, str):
                name = key.replace("_", " ").title()
                mtype = self._infer_type(key)
                architecture = self._structure_architecture(detail)
                models.append({
                    "name": name,
                    "type": mtype,
                    "code_class_name": self._sanitize_class_name(name),
                    "input": self._pick(arch.get("input")),
                    "output": self._pick(arch.get("output")),
                    "num_classes": self._pick(ds.get("number_of_classes")),
                    "pretrained": self._pick(arch.get("pretrained_model")),
                    "architecture_summary": detail[:500],
                    "architecture": architecture,
                })

        return models

    def _build_model_entry(self, name: str, arch: dict,
                           ds: dict, meth: dict) -> dict:
        return {
            "name": name.strip(),
            "type": self._infer_type(name.lower()),
            "code_class_name": self._sanitize_class_name(name),
            "input": self._pick(arch.get("input")),
            "output": self._pick(arch.get("output")),
            "num_classes": self._pick(ds.get("number_of_classes")),
            "pretrained": self._pick(arch.get("pretrained_model")),
            "architecture_summary": "NOT_FOUND",
            "architecture": {
                "total_layers": "NOT_FOUND",
                "param_count": "NOT_FOUND",
                "special_components": [],
                "final_classifier": "NOT_FOUND",
                "implementation_notes": [],
            },
        }

    # =========================
    # 类型推断
    # =========================

    def _infer_type(self, text: str) -> str:
        if any(k in text for k in VLM_KW):
            return "VLM"
        if any(k in text for k in TRANSFORMER_KW):
            return "Transformer"
        if any(k in text for k in RNN_KW):
            return "RNN"
        if any(k in text for k in CNN_KW):
            return "CNN"
        if any(k in text for k in MLP_KW):
            return "MLP"
        return "other"

    # =========================
    # 架构结构化解析
    # =========================

    def _structure_architecture(self, detail: str) -> dict:
        """
        将文本描述的模型架构解析为结构化字段。

        提取: 层数、参数量、特殊组件、最终分类层名称。
        所有解析均基于 detail 文本内容，不针对具体模型名，对任意论文通用。
        """
        result = {
            "total_layers": "NOT_FOUND",
            "param_count": "NOT_FOUND",
            "special_components": [],
            "final_classifier": "NOT_FOUND",
            "implementation_notes": [],
        }

        # 通用解析：参数量
        import re
        param_m = re.search(r'(\d+)\s*million\s*parameters?', detail, re.IGNORECASE)
        if param_m:
            result["param_count"] = f"{param_m.group(1)}M"

        # 通用解析：层数
        layer_m = re.search(r'(\d+)\s*layers?', detail, re.IGNORECASE)
        if layer_m:
            result["total_layers"] = int(layer_m.group(1))

        # 通用解析：最后的分类层
        fc_m = re.search(r'(fc\d+|classifier|linear)', detail, re.IGNORECASE)
        if fc_m:
            result["final_classifier"] = fc_m.group(1)

        # 通用解析：结构特征（按内容触发，不按模型名）
        result.update(self._parse_special_components(detail))

        # 通用：Dropout
        if "dropout" in detail.lower():
            result["special_components"].append({
                "type": "Dropout",
                "details": self._extract_dropout_detail(detail),
            })

        return result

    def _parse_special_components(self, detail: str) -> dict:
        """
        从架构描述文本中通用地识别结构特征（不针对具体模型名）。

        触发条件全部来自 detail 文本内容，因此对任意论文通用：
          - Inception 模块（并行多尺度卷积 + 拼接）
          - 辅助分类器（侧分支，训练时 loss 加权）
          - 全局平均池化（替代全连接层）
          - 卷积/全连接层数量
          - 归一化层（Normalization / LocalResponseNorm）
        """
        info = {
            "special_components": [],
            "implementation_notes": [],
        }
        low = detail.lower()
        import re

        # 1) Inception 模块
        inc = re.search(r'(\d+)\s*inception\s*modules?', detail, re.IGNORECASE)
        if "inception" in low:
            info["special_components"].append({
                "type": "Inception",
                "count": int(inc.group(1)) if inc else None,
                "description": (
                    "Inception 模块：并行运行多个不同尺度的卷积与池化，"
                    "结果沿 channel 维度拼接；小尺度卷积用于降维。"
                ),
                "code_implementation": (
                    "可自实现 InceptionModule(nn.Module):\n"
                    "  branches = [conv1x1, conv3x3, conv5x5, pool_proj]\n"
                    "  output = torch.cat([b(x) for b in branches], dim=1)"
                ),
            })
            if inc:
                info["inception_module_count"] = int(inc.group(1))

        # 2) 辅助分类器（侧分支）
        aux = re.search(r'(\d+)\s*auxiliary\s*classifiers?', detail, re.IGNORECASE)
        if "auxiliary" in low or "aux classifier" in low:
            info["special_components"].append({
                "type": "AuxiliaryClassifier",
                "count": int(aux.group(1)) if aux else None,
                "location": "NOT_FOUND（论文未说明位置）",
                "description": (
                    "辅助分类器（侧分支）：训练时每个侧分支的 loss 按论文给定的权重 "
                    "加入总 loss，用于缓解梯度消失和提供额外正则化；推理时禁用。"
                ),
                "code_implementation": (
                    "class AuxClassifier(nn.Module):  # avgpool + conv + fc + dropout + fc(num_classes)\n"
                    "训练: total_loss = main_loss + weight_i * aux_i_loss\n"
                    "推理: model.eval() 时 aux 分支禁用"
                ),
                "training_impact": {
                    "loss_weight": "NOT_FOUND（按论文原文，不同模型可能不同）",
                    "enabled_during": "training only",
                },
            })

        # 3) 全局平均池化
        if "global average" in low or "global avg" in low or re.search(r'\bgap\b', low):
            info["special_components"].append({
                "type": "GlobalAveragePooling",
                "description": "用全局平均池化替代全连接层以减少参数量",
                "code_implementation": "nn.AdaptiveAvgPool2d((1, 1)) → flatten → fc(num_classes)",
            })

        # 4) 卷积/全连接层数量
        conv_m = re.search(r'(\d+)\s*convolution\s*layers?', detail, re.IGNORECASE)
        fc_m = re.search(r'(\d+)\s*fully\s*connected\s*layers?', detail, re.IGNORECASE)
        if conv_m:
            info["conv_layers"] = int(conv_m.group(1))
        if fc_m:
            info["fc_layers"] = int(fc_m.group(1))

        # 5) 归一化层
        if "local response" in low or re.search(r'\blrn\b', low):
            info["special_components"].append({
                "type": "LocalResponseNorm",
                "description": "卷积层后的局部响应归一化层",
                "code_implementation": (
                    "nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2) — "
                    "PyTorch 0.4+ 已弃用，可用 nn.BatchNorm2d 替代（非严格复现）"
                ),
            })
        elif "normalization" in low or "norm" in low:
            info["special_components"].append({
                "type": "Normalization",
                "description": (
                    "原文提及 normalization/norm 层；具体类型（BatchNorm / LayerNorm / "
                    "LocalResponseNorm / InstanceNorm）需结合模型与上下文确认。"
                ),
            })

        return info

    @staticmethod
    def _extract_dropout_detail(detail: str) -> str:
        """提取 Dropout 配置细节"""
        import re
        m = re.search(r'dropout.*?(ratio|rate|prob)[:\s]*(\d+\.?\d*)',
                      detail, re.IGNORECASE)
        if m:
            return f"rate={m.group(2)}"
        m = re.search(r'dropout.*?(\d+\.\d+)', detail, re.IGNORECASE)
        if m:
            return f"rate={m.group(1)}"
        return "rate 未指定"

    # =========================
    # 安全的 class 名生成
    # =========================

    @staticmethod
    def _sanitize_class_name(name: str) -> str:
        """
        生成合法的 Python 类名。

        规则:
          - 去除非字母数字的字符
          - 每个单词首字母大写 (PascalCase)
          - 不以数字开头
        """
        import re

        # 去掉括号和特殊字符，只保留字母数字和空格
        cleaned = re.sub(r"[^a-zA-Z0-9\s]", "", name)

        # 按空格分割，每个词首字母大写
        words = cleaned.split()
        words = [w for w in words if w.lower() not in ("for", "of", "in", "and", "the")]

        if not words:
            words = ["Model"]

        class_name = "".join(w[0].upper() + w[1:] for w in words)

        # 确保不以数字开头
        if class_name[0].isdigit():
            class_name = "M" + class_name

        return class_name
