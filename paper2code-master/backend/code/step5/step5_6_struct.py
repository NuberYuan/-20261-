"""Step5.6: 项目文件架构设计 → project_structure.yaml

按「项目原型」拼装目录/文件，而非两个固定 ML 模板：
  - 纯算法论文（无外部数据/环境且不可训练）→ 不生成 data/（输入在 src/ 内程序化生成）
  - 无 ML 模型              → 不生成 models/（纯算法论文的 method_name 不是模型）
  - is_trainable == False   → 不生成 training/（无训练）
  - 纯算法论文             → 额外生成 src/<method>.py 承载算法实现
"""
from step5.base_analyzer import BaseAnalyzer


class StructureAnalyzer(BaseAnalyzer):

    def extract_by_rules(self) -> dict:
        goal = self.load_yaml("reproduction_goal.yaml")
        ptype = goal.get("paper_type", "empirical_comparison")
        model_d = self.load_yaml("model_design.yaml")
        dataset = self.load_yaml("dataset.yaml")

        is_trainable = bool(goal.get("is_trainable", False))
        data_source = dataset.get("data_source", "external_dataset")
        models = model_d.get("models", [])

        # data/ 仅在真正需要"加载/包装数据"时生成：
        #   external_dataset → 加载本地/下载数据集文件
        #   environment      → 包装交互式环境/仿真器
        #   synthetic+可训练  → 合成数据仍需 Dataset 供训练循环消费
        # 纯算法论文（synthetic/none 且不可训练）不生成 data/，输入在 src/ 内程序化生成。
        has_data = (
            data_source == "external_dataset"
            or data_source == "environment"
            or (data_source == "synthetic" and is_trainable)
        )
        has_models = bool(models)
        # 纯算法论文：无数据、无训练（method_name 是方法不是模型）
        is_algorithm = ptype == "algorithm" or (
            not has_data and not has_models and not is_trainable
        )

        # 从 model_design.yaml 生成模型文件名
        model_names = []
        for m in models:
            cn = m.get("code_class_name", m.get("name", "model"))
            safe_name = self._sanitize_filename(cn)
            if safe_name:
                model_names.append(safe_name)
        if not model_names:
            model_names = ["model"]

        # 项目名
        title = self.key_info.get("title", "project")
        proj = "".join(
            c for c in title.replace(" ", "_")
            if c.isalnum() or c == "_"
        )[:50]

        dirs = []
        files = {}
        mapping = {}

        if has_data:
            dirs.append("data")
            files["data"] = ["dataset.py"]
            mapping["data"] = "数据加载与预处理"

        if has_models:
            dirs.append("models")
            files["models"] = [f"{n}.py" for n in model_names] + ["__init__.py"]
            mapping["models"] = f"模型定义: {', '.join(model_names)}"

        if is_trainable:
            dirs.append("training")
            files["training"] = ["train.py"]
            mapping["training"] = "训练循环"

        # 评估/对比：有模型或有对比实验时生成
        if has_models or ptype == "empirical_comparison":
            dirs.append("evaluation")
            files["evaluation"] = ["evaluate.py"]
            if ptype == "empirical_comparison":
                files["evaluation"].append("compare.py")
            mapping["evaluation"] = "评估与对比"

        if has_models or is_trainable:
            dirs.append("utils")
            files["utils"] = ["metrics.py"]
            mapping["utils"] = "工具函数"

        # 纯算法论文：src 目录承载算法/方法实现
        if is_algorithm:
            dirs.append("src")
            files["src"] = [self._method_filename()]
            mapping["src"] = "算法/方法实现"

        dirs.append("config")
        files["config"] = ["config.yaml"]
        mapping["config"] = "YAML 配置"

        files["."] = ["main.py", "requirements.txt", "README.md"]
        mapping["entry_point"] = "main.py"

        return {
            "paper_name": self.paper_name,
            "project_name": proj,
            "root_structure": {
                "directories": dirs,
                "files": files,
            },
            "module_mapping": mapping,
        }

    def _method_filename(self) -> str:
        """纯算法论文的算法实现文件名，取自 method_name，回退 algorithm。"""
        meth = self.key_info.get("methodology", {})
        name = meth.get("method_name", "") or ""
        if not name or name == "NOT_FOUND":
            name = "algorithm"
        safe = self._sanitize_filename(name)
        if safe in ("", "model"):
            safe = "algorithm"
        return f"{safe}.py"

    @staticmethod
    def _sanitize_filename(name: str) -> str:
        """生成安全的 Python 文件名（小写+下划线）"""
        import re
        # PascalCase → snake_case
        s1 = re.sub(r'(.)([A-Z][a-z]+)', r'\1_\2', name)
        s2 = re.sub(r'([a-z0-9])([A-Z])', r'\1_\2', s1)
        # 只保留字母数字和下划线
        cleaned = re.sub(r'[^a-zA-Z0-9_]', '', s2)
        # 转为小写
        result = cleaned.lower().strip('_')
        return result or "model"
