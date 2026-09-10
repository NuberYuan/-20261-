"""
Step7.2: 代码上下文构建器

把 Step6 project_brief 中分散的信息，整理成
"现在要生成这个文件，Agent 需要知道什么"的上下文。

对模型 / 数据 / 训练文件，额外回读 Step5 的详细设计（model_design /
dataset / training_config），补充 project_brief 摘要里没有的架构细节，
避免生成 Agent 因为缺细节而自行脑补。
"""

from pathlib import Path
import yaml
from logger import logger
from step7.code_validator import CodeValidator


class CodeContextBuilder:
    """整理单文件生成上下文。"""

    def __init__(self, brief: dict, step5_dir: Path = None):
        self.brief = brief
        self.step5_dir = step5_dir
        self.specs = {
            s.get("file"): s
            for s in brief.get("code_generation_plan", {}).get("per_file_specs", [])
        }
        self._phase_map = self._build_phase_map()
        self._step5_cache = {}

    # ============================================================
    # 主入口
    # ============================================================

    def build(self, file: str, generated: dict) -> dict:
        """构建单个文件的生成上下文。generated = {已生成文件: 代码}。"""
        spec = self.specs.get(file, {})

        ctx = {
            "file": file,
            "phase": self._phase_map.get(file),
            "spec": spec,
            "project": self.brief.get("project", {}),
            "tech_stack": self.brief.get("design_digest", {}).get("tech_stack", {}),
            "models": self.brief.get("design_digest", {}).get("models", []),
            "training_config": self.brief.get("design_digest", {}).get(
                "training_config_summary", {}),
            "has_config": any(f.startswith("config/") for f in self.specs),
            "requirements": self._related_requirements(spec),
            "related_code": self._related_code(file, spec, generated),
            "step5_detail": self._step5_detail(file),
            "agent_notes": self.brief.get("risks_and_notes", {}).get("for_code_agent", []),
        }

        # 如果 config.yaml 已生成，注入其完整路径树供 LLM 参考
        config_key = next(
            (f for f in ("config/config.yaml", "config/config.yml") if f in generated),
            None,
        )
        if config_key:
            try:
                cfg = yaml.safe_load(generated[config_key]) or {}
                paths = CodeValidator._collect_config_paths(cfg, max_depth=3)
                if paths:
                    ctx["config_paths"] = sorted(paths)
            except Exception:
                pass

        return ctx

    # ============================================================
    # 需求 / 已生成代码
    # ============================================================

    def _related_requirements(self, spec: dict) -> list:
        """提取该文件满足的 FR 描述。"""
        frs = self.brief.get("requirements_digest", {}).get("core_requirements", [])
        fr_map = {fr.get("id"): fr for fr in frs}
        ids = spec.get("satisfies_requirements", []) or []
        return [fr_map.get(i, {"id": i, "what": "", "acceptance": ""}) for i in ids]

    def _related_code(self, file: str, spec: dict, generated: dict) -> dict:
        """提取该文件依赖的、已经生成的代码。"""
        related = {}
        for dep in spec.get("depends_on", []) or []:
            if dep in generated:
                related[dep] = generated[dep]
        return related

    # ============================================================
    # Step5 详细设计补充（模型架构等 project_brief 未展开的细节）
    # ============================================================

    def _step5_detail(self, file: str) -> dict:
        """按文件类型回读 Step5 详细设计。"""
        if self.step5_dir is None:
            return {}

        if file.startswith("models/") and file.endswith(".py"):
            return self._model_detail(file)
        if file == "data/dataset.py":
            return self._dataset_detail()
        if file in ("training/train.py",):
            return self._training_detail()
        if file in ("config/config.yaml", "config/config.yml"):
            return self._config_detail()
        if file.startswith("src/") and file.endswith(".py"):
            return self._method_detail()
        return {}

    def _load_step5(self, name: str) -> dict:
        """惰性加载 Step5 YAML，带缓存。"""
        if name not in self._step5_cache:
            path = self.step5_dir / f"{name}.yaml"
            try:
                self._step5_cache[name] = yaml.safe_load(
                    path.read_text(encoding="utf-8")) or {}
            except Exception:
                self._step5_cache[name] = {}
        return self._step5_cache[name]

    def _model_detail(self, file: str) -> dict:
        """取对应模型的架构详情（architecture_summary + 特殊组件）。"""
        stem = Path(file).stem  # alexnet / googlenet
        md = self._load_step5("model_design")
        for m in md.get("models", []):
            if (m.get("name") or "").lower() == stem:
                return {
                    "name": m.get("name"),
                    "type": m.get("type"),
                    "input": m.get("input"),
                    "output": m.get("output"),
                    "num_classes": m.get("num_classes"),
                    "pretrained": m.get("pretrained"),
                    "architecture_summary": m.get("architecture_summary"),
                    "architecture": m.get("architecture"),
                }
        return {}

    def _dataset_detail(self) -> dict:
        """取数据集预处理细节。"""
        ds = self._load_step5("dataset")
        return {
            "datasets": ds.get("datasets", []),
            "dataset_versions": ds.get("dataset_versions", []),
            "preprocessing": ds.get("preprocessing", {}),
            "split_strategy": ds.get("split_strategy", {}),
        }

    def _training_detail(self) -> dict:
        """取完整训练配置（含训练模式与 checkpoint）。"""
        tc = self._load_step5("training_config")
        return {
            "training_modes": tc.get("training_modes", {}),
            "training": tc.get("training", {}),
            "checkpoint": tc.get("checkpoint", {}),
            "train_test_splits": tc.get("train_test_splits", []),
            "inference": tc.get("inference", {}),
        }

    def _method_detail(self) -> dict:
        """纯算法论文（src/）回读复现目标，供生成 Agent 了解方法上下文。"""
        rg = self._load_step5("reproduction_goal")
        return {
            "paper_type": rg.get("paper_type"),
            "task": rg.get("task"),
        }

    def _config_detail(self) -> dict:
        """config/config.yaml 需要的完整参数（训练 + 数据版本 + 划分）。"""
        tc = self._load_step5("training_config")
        ds = self._load_step5("dataset")
        return {
            "training": tc.get("training", {}),
            "training_modes": tc.get("training_modes", {}),
            "checkpoint": tc.get("checkpoint", {}),
            "train_test_splits": tc.get("train_test_splits", []),
            "inference": tc.get("inference", {}),
            "dataset_versions": ds.get("dataset_versions", []),
            "version_selector": ds.get("preprocessing", {}).get("version_selector", {}),
            "split_strategy": ds.get("split_strategy", {}),
        }

    # ============================================================
    # 辅助
    # ============================================================

    def _build_phase_map(self) -> dict:
        """文件 → phase 编号。"""
        m = {}
        for phase in self.brief.get("code_generation_plan", {}).get("generation_order", []):
            for f in phase.get("files", []):
                m[f] = phase.get("phase")
        return m
