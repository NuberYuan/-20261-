"""Step5.2: 技术栈分析 → technology.yaml"""
from step5.base_analyzer import BaseAnalyzer


class TechAnalyzer(BaseAnalyzer):

    def extract_by_rules(self) -> dict:
        ki = self.key_info
        meth = ki.get("methodology", {})
        impl = ki.get("implementation_details", {})
        arch = meth.get("model_architecture", {})
        if not isinstance(arch, dict):
            arch = {}

        # --- 框架推断 ---
        # ML 论文（有训练或有 ML 模型）→ 统一 PyTorch（step7 code_generator 为 PyTorch 工程师）。
        # 纯算法/方法论文（无训练、无 ML 模型）→ Python 标准库 + numpy/pyyaml，不引入 PyTorch。
        # 论文原文使用的框架是"事实"信息，单独记录在 programming_framework_mentioned，
        # 不污染 recommended，避免 Step6 decisions 与 Step7 config.yaml 的 framework 冲突。
        prog_fw = impl.get("programming_framework", "NOT_FOUND")

        has_training = any(
            impl.get(k, "NOT_FOUND") != "NOT_FOUND"
            for k in ["optimizer", "learning_rate", "epoch", "loss_function"]
        )
        has_model = self._has_ml_model(meth, arch)

        if has_training or has_model:
            rec = "PyTorch"
            fw_src = "复现代码统一使用 PyTorch；论文原文框架见 programming_framework_mentioned"
        else:
            rec = "Python 标准库"
            fw_src = "纯算法/方法复现（无训练、无 ML 模型），使用 Python 标准库 + numpy/pyyaml，不引入 PyTorch"

        # --- 模型列表 ---
        # 优先级: methodology.model > model_architecture 的 key > method_name
        models = self._extract_model_names(meth, arch)

        # --- 架构详情 ---
        arch_details = {}
        for arch_key in ["alexnet", "googlenet", "resnet", "vgg",
                          "lstm", "transformer", "attention"]:
            detail = arch.get(arch_key, "")
            if detail and detail != "NOT_FOUND":
                arch_details[arch_key] = str(detail)[:500]

        pretrained = arch.get("pretrained_model", "NOT_FOUND")
        special = arch.get("special_modules", "NOT_FOUND")

        return {
            "paper_name": self.paper_name,
            "framework": {
                "recommended": rec,
                "candidates": ["PyTorch"],
                "source": fw_src,
            },
            "programming_framework_mentioned": self._pick(prog_fw),
            "model_backbone": {
                "names": models if models else ["NOT_FOUND"],
                "source": "论文提到" if models else "NOT_FOUND",
            },
            "architecture_details": arch_details if arch_details else {"note": "NOT_FOUND"},
            "training": {
                "optimizer": {"type": self._pick(impl.get("optimizer"))},
                "loss": {"type": self._pick(impl.get("loss_function"))},
                "scheduler": {"type": self._infer_scheduler(impl)},
            },
            "pretrained_weights": {
                "used": pretrained != "NOT_FOUND",
                "source": str(pretrained),
            },
            "special_modules": self._pick(special),
            "hardware": {
                "mentioned": self._pick(impl.get("hardware")),
            },
        }

    def _extract_model_names(self, meth: dict, arch: dict) -> list:
        """从多个来源提取模型名称"""
        models = []

        # 来源1: methodology.model
        mf = meth.get("model", "")
        if mf and mf != "NOT_FOUND":
            for p in mf.replace(" and ", ", ").split(","):
                p = p.strip().strip("()")
                if p and len(p) > 2 and "not" not in p.lower():
                    models.append(p)

        # 来源2: model_architecture 的下级 key (如 alexnet, googlenet)
        # 只添加值为非 NOT_FOUND 的 key
        arch_keys = ["alexnet", "googlenet", "resnet", "vgg", "densenet",
                      "mobilenet", "efficientnet", "lstm", "transformer"]
        for ak in arch_keys:
            val = arch.get(ak, "NOT_FOUND")
            if val and val != "NOT_FOUND" and ak.lower() not in [m.lower() for m in models]:
                models.append(ak)

        # 来源3: methodology.method_name (fallback)
        if not models:
            mn = meth.get("method_name", "")
            if mn and mn != "NOT_FOUND":
                models.append(mn)

        return models[:10]

    def _has_ml_model(self, meth: dict, arch: dict) -> bool:
        """是否存在真正的 ML 模型（排除 NOT_FOUND 占位与纯算法 method_name）。

        architecture 键有非 NOT_FOUND 值，或 model/method_name 含 NN 类型关键词。
        """
        for key in ["alexnet", "googlenet", "resnet", "vgg", "densenet",
                    "mobilenet", "efficientnet", "lstm", "transformer"]:
            v = str(arch.get(key, "NOT_FOUND"))
            if v not in ("NOT_FOUND", ""):
                return True

        nn_kw = ("cnn", "convolution", "lstm", "gru", "rnn", "recurrent",
                 "transformer", "attention", "bert", "gpt", "vit", "mlp",
                 "perceptron", "fully connected", "feedforward")
        name_text = (str(meth.get("model", "")) + " " +
                     str(meth.get("method_name", ""))).lower()
        return any(kw in name_text for kw in nn_kw)

    def _infer_scheduler(self, impl: dict) -> str:
        """推断学习率调度器——从 training_strategy 和 important_parameters"""
        # 优先从 important_parameters 读取（由 Step3 兜底搜索填充）
        params = self.key_info.get("reproduction_requirements", {}).get("important_parameters", {})
        if isinstance(params, dict):
            sched = params.get("scheduler", "")
            if sched and sched != "NOT_FOUND":
                return sched

        ts = str(impl.get("training_strategy", "")).lower()
        if "step" in ts:
            return "StepLR"
        if "cosine" in ts:
            return "CosineAnnealingLR"
        if "plateau" in ts:
            return "ReduceLROnPlateau"
        return "NOT_FOUND"
