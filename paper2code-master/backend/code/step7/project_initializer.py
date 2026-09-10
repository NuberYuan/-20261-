"""
Step7.1: 项目初始化器

根据 Step6 的 code_generation_plan 创建项目目录骨架和空占位文件。
不调用 LLM —— 这是纯文件系统操作。
"""

from pathlib import Path
from logger import logger


class ProjectInitializer:
    """根据 project_brief 创建 generated_project 的目录与文件骨架。"""

    def __init__(self, project_root: Path):
        self.project_root = project_root  # .../step7/generated_project

    def initialize(self, brief: dict) -> list:
        """
        创建目录与空文件，返回所有文件路径（相对 project_root，按依赖数升序排好的生成顺序）。
        """
        per_file_specs = brief.get("code_generation_plan", {}).get("per_file_specs", [])
        if not per_file_specs:
            logger.warning("  code_generation_plan.per_file_specs 为空，无法初始化项目")
            return []

        # 1. 收集目录与文件
        dirs = set()
        files = []
        for spec in per_file_specs:
            f = spec.get("file", "")
            if not f:
                continue
            files.append(f)
            parent = str(Path(f).parent)
            if parent != ".":
                dirs.add(parent)

        # 2. 创建目录
        self.project_root.mkdir(parents=True, exist_ok=True)
        for d in sorted(dirs):
            (self.project_root / d).mkdir(parents=True, exist_ok=True)

        # 3. 创建空占位文件
        for f in files:
            path = self.project_root / f
            path.parent.mkdir(parents=True, exist_ok=True)
            if not path.exists():
                path.touch()

        logger.info(f"  项目骨架: {len(dirs)} 个目录, {len(files)} 个文件")

        return self._ordered_files(files, per_file_specs)

    # ============================================================
    # 生成顺序：被依赖的文件先生成（拓扑排序）
    # ============================================================

    @staticmethod
    def _ordered_files(files: list, specs: list) -> list:
        """
        对文件做拓扑排序（Kahn 算法），保证被依赖的文件先生成。

        依赖关系来自 spec.depends_on。规则:
          1. 无依赖 / 依赖已满足的文件先出队
          2. 同批就绪文件中，非 Python 文件（config.yaml 等）优先
          3. 其余按原始声明顺序（稳定）

        注意: 不能用「依赖数升序」近似 —— 例如 main.py 依赖 evaluation/evaluate.py，
        但 evaluate.py 的依赖数更多，按计数排序会导致 main.py 先于 evaluate.py 生成。
        """
        spec_map = {s.get("file"): s for s in specs}
        files_set = set(files)

        # 反向邻接表: 被依赖文件 -> 依赖它的文件
        dependents = {f: [] for f in files}
        indegree = {f: 0 for f in files}
        for f in files:
            for dep in spec_map.get(f, {}).get("depends_on", []) or []:
                if dep in files_set:  # 只统计项目内依赖
                    dependents[dep].append(f)
                    indegree[f] += 1

        # 就绪队列（入度为 0），按 (是否 py, 原始顺序) 排序保证稳定 + 非 py 优先
        original_order = {f: i for i, f in enumerate(files)}
        ready = [f for f in files if indegree[f] == 0]
        ready.sort(key=lambda f: (f.endswith(".py"), original_order[f]))

        ordered = []
        while ready:
            f = ready.pop(0)
            ordered.append(f)
            for dependent in dependents.get(f, []):
                indegree[dependent] -= 1
                if indegree[dependent] == 0:
                    ready.append(dependent)
            # 重新排序，非 py 优先 + 稳定
            ready.sort(key=lambda x: (x.endswith(".py"), original_order[x]))

        # 兜底: 若有环导致无法全部入队，补上剩余文件
        if len(ordered) < len(files):
            ordered += [f for f in files if f not in ordered]

        return ordered
