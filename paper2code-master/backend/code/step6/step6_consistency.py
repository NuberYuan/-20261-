"""
Step6.1: 一致性检查器 —— 规则驱动的跨来源对齐

将 Step4（需求分析）与 Step5（工程设计）横向交叉校验，产出三类结构化结论:

  - checks:      每项一致性检查的通过/告警/失败状态
  - conflicts:   Step4 与 Step5 相互矛盾的地方（必须决策）
  - gaps:        两边都缺失、或某侧 NOT_FOUND 的信息（代码 Agent 需自行决策）
  - assumptions: 为推进工程实现而做的显式假设

设计原则:
  - 纯规则驱动，不调用 LLM。一致性是确定性的事实比对，交给 LLM 反而引入偏差。
  - 每个结论可回溯到具体来源（Step4 的哪个 FR / Step5 的哪个字段）。
"""

import re
from logger import logger


class ConsistencyChecker:
    """Step4 ↔ Step5 一致性检查器。输入是两边的 YAML 字典。"""

    def __init__(self, step4_yamls: dict, step5_yamls: dict):
        self.step4 = step4_yamls
        self.step5 = step5_yamls
        self._counter = {"check": 0, "conflict": 0, "gap": 0, "asm": 0}

    # ============================================================
    # 主入口
    # ============================================================

    def check(self) -> dict:
        """运行全部检查，返回 consistency_report 字典。"""
        checks = []
        conflicts = []
        gaps = []
        assumptions = []

        # ---- 各项一致性检查（可能附带产出 conflicts / gaps）----
        checks += self._check_numerical(conflicts, gaps)
        checks += self._check_training_modes(conflicts)
        checks += self._check_fr_module_coverage(gaps)
        checks += self._check_split_consistency(conflicts)
        checks += self._check_metric_consistency(conflicts)
        checks += self._check_interface_consistency()

        # ---- 聚合 Step4/5 已有的缺失与假设信息 ----
        gaps += self._collect_gaps()
        assumptions += self._collect_assumptions()

        # ---- 汇总 ----
        summary = {
            "checks_run": len(checks),
            "checks_passed": sum(1 for c in checks if c["status"] == "pass"),
            "checks_warning": sum(1 for c in checks if c["status"] == "warning"),
            "checks_failed": sum(1 for c in checks if c["status"] == "fail"),
            "conflicts": len(conflicts),
            "gaps": len(gaps),
            "assumptions": len(assumptions),
        }

        logger.info(
            f"  [Step6 一致性] {summary['checks_run']} 项检查: "
            f"{summary['checks_passed']} 通过, {summary['checks_warning']} 告警, "
            f"{summary['checks_failed']} 失败 | {summary['conflicts']} 冲突, "
            f"{summary['gaps']} 缺失, {summary['assumptions']} 假设"
        )

        return {
            "summary": summary,
            "checks": checks,
            "conflicts": conflicts,
            "gaps": gaps,
            "assumptions": assumptions,
        }

    # ============================================================
    # CHECK-01: 数值一致性
    # ============================================================

    # ---- 原型信号（供各检查项按 is_trainable / has_dataset 门控）----

    def _is_trainable(self) -> bool:
        """是否存在训练流程（Step5 training_config）。"""
        tc = self.step5.get("training_config", {})
        training = tc.get("training", {})
        if isinstance(training, dict) and training.get("applies") is False:
            return False
        return bool(tc.get("training_modes"))

    def _has_dataset(self) -> bool:
        """是否依赖真实外部数据集（仅 external_dataset 才有规模/类别数等数值）。"""
        ds = self.step5.get("dataset", {})
        return str(ds.get("data_source", "external_dataset")) == "external_dataset"

    _IMAGE_SIGNALS = (
        "imagenet", "cifar", "mnist", "coco", "leaf", "叶片", "plantvillage",
        "植物", "pixel", "像素", "resize", "分辨率",
    )

    def _is_image_data(self) -> bool:
        """数据集是否为图像数据（决定是否检查图像尺寸）。"""
        ds = self.step5.get("dataset", {})
        if not isinstance(ds, dict):
            return False
        parts = [str(ds.get(k, "")) for k in ("dataset_source", "preprocessing")]
        for d in (ds.get("datasets") or []):
            if isinstance(d, dict):
                parts.append(str(d.get("data_type", "")))
                parts.append(str(d.get("name", "")))
        text = " ".join(parts).lower()
        return any(sig in text for sig in self._IMAGE_SIGNALS)

    # (名称, step4 侧文本提取器, step5 侧文本提取器)
    def _numerical_facts(self):
        """返回需要比对的关键数值事实，每项为 (名称, step4_text, step5_text)。

        按原型门控：数据集规模/类别数/图像尺寸仅对外部数据集有意义；
        学习率/训练轮数/批大小仅对训练论文有意义。纯算法/合成/无数据论文返回空列表，
        避免产生一堆 ML 专属的 warning/gap。
        """
        fr_desc = self._fr_field("description")
        fr_output = self._fr_field("output")

        ds = self.step5.get("dataset", {})
        datasets = ds.get("datasets", []) or [{}]
        tc = self.step5.get("training_config", {}).get("training", {})
        rg = self.step5.get("reproduction_goal", {})

        facts = []
        if self._has_dataset():
            facts.append(
                ("数据集规模", fr_desc("数据集加载"), str(datasets[0].get("size", "")))
            )
            facts.append(
                ("类别数", fr_output("数据集加载"),
                 str(datasets[0].get("num_classes", "")) + " " + str(rg.get("num_classes", "")))
            )
            if self._is_image_data():
                facts.append(
                    ("图像尺寸", fr_desc("数据预处理"), str(tc.get("image_size", "")))
                )
        if self._is_trainable():
            facts.append(
                ("学习率", fr_desc("模型训练"), str(tc.get("optimizer", {}).get("lr", "")))
            )
            facts.append(
                ("训练轮数", fr_desc("模型训练"), str(tc.get("epochs", "")))
            )
            facts.append(
                ("批大小", fr_desc("模型训练"),
                 str(tc.get("batch_size", "")) + " " + str(tc.get("batch_size_note", "")))
            )
        return facts

    def _check_numerical(self, conflicts: list, gaps: list) -> list:
        """比对 Step4 与 Step5 中的关键数值是否一致。"""
        checks = []
        for name, s4_text, s5_text in self._numerical_facts():
            s4_nums = self._extract_nums(s4_text)
            s5_nums = self._extract_nums(s5_text)

            check_id = self._next_id("check")
            if not s4_nums and not s5_nums:
                checks.append(self._check(
                    check_id, name, "warning",
                    "Step4 与 Step5 均未给出该数值，标记为缺失。"))
                gaps.append({
                    "id": self._next_id("gap"),
                    "field": name,
                    "description": f"{name} 在 Step4 与 Step5 中均缺失（NOT_FOUND）。",
                    "severity": "medium",
                    "mitigation": "由代码 Agent 采用合理工程默认值，并在 config.yaml 中显式标注为可配置项。",
                })
                continue

            # 关键数值是否有交集
            if s4_nums and s5_nums and s4_nums & s5_nums:
                checks.append(self._check(
                    check_id, name, "pass",
                    f"Step4 与 Step5 一致: {sorted(s4_nums & s5_nums)}"))
            elif s4_nums and s5_nums:
                checks.append(self._check(
                    check_id, name, "fail",
                    f"数值不一致: Step4={sorted(s4_nums)} vs Step5={sorted(s5_nums)}"))
                conflicts.append({
                    "id": self._next_id("conflict"),
                    "field": name,
                    "description": f"{name} 在需求与设计两侧数值不一致。",
                    "step4_says": str(s4_text),
                    "step5_says": str(s5_text),
                    "severity": "high",
                    "resolution": "以论文原文为准回查 PDF，修正不一致的一侧；无法判定时以 Step5（更接近工程实现）为准并在假设中记录。",
                })
            elif s4_nums:
                checks.append(self._check(
                    check_id, name, "warning",
                    f"Step4 有值 {sorted(s4_nums)}，Step5 缺失。"))
            else:
                checks.append(self._check(
                    check_id, name, "warning",
                    f"Step5 有值 {sorted(s5_nums)}，Step4 未在需求中体现。"))

        return checks

    # ============================================================
    # CHECK-02: 训练模式一致性
    # ============================================================

    def _check_training_modes(self, conflicts: list) -> list:
        """检查 scratch/transfer 两种训练模式是否在需求与设计中贯穿一致。"""
        # 非训练论文没有训练模式，跳过检查（否则误报 fail）。
        if not self._is_trainable():
            return []
        fr_desc = self._fr_field("description")("训练模式")
        modes = self.step5.get("training_config", {}).get("training_modes", {})

        # Step4 是否声明了两种模式
        has_scratch = bool(re.search(r"scratch|从头训练", fr_desc, re.I))
        has_transfer = bool(re.search(r"transfer|迁移学习|预训练", fr_desc, re.I))
        # Step5 是否落地了两种模式
        s5_scratch = "scratch" in modes
        s5_transfer = "transfer" in modes

        checks = []
        check_id = self._next_id("check")
        if has_scratch and has_transfer and s5_scratch and s5_transfer:
            checks.append(self._check(
                check_id, "训练模式（scratch/transfer）", "pass",
                "Step4 与 Step5 均显式区分两种训练模式。"))
        elif (has_scratch or has_transfer) and (s5_scratch or s5_transfer):
            checks.append(self._check(
                check_id, "训练模式（scratch/transfer）", "warning",
                "一侧只声明了单一模式。"))
            conflicts.append({
                "id": self._next_id("conflict"),
                "field": "training_modes",
                "description": "scratch 与 transfer 两种模式未在需求与设计两侧完全对齐。",
                "step4_says": f"scratch={has_scratch}, transfer={has_transfer}",
                "step5_says": f"scratch={s5_scratch}, transfer={s5_transfer}",
                "severity": "medium",
                "resolution": "补齐缺失的训练模式配置，避免代码只实现单一 pretrained 布尔开关。",
            })
        else:
            checks.append(self._check(
                check_id, "训练模式（scratch/transfer）", "fail",
                "训练模式信息缺失或两侧矛盾。"))

        return checks

    # ============================================================
    # CHECK-03: FR → 模块覆盖
    # ============================================================

    def _check_fr_module_coverage(self, gaps: list) -> list:
        """检查每个功能需求是否在 Step5 的项目结构中有对应模块。

        按原型映射：算法论文的"方法实现"落在 src（不是 models），
        且不把 data/models/training/evaluation 当作缺失模块（纯算法论文本就不应有）。
        """
        checks = []
        frs = self.step4.get("functional_requirements", {}).get("requirements", [])
        module_mapping = self.step5.get("project_structure", {}).get("module_mapping", {})

        if not frs:
            return []

        # 归一化模块名集合
        modules = set(module_mapping.keys())
        # 纯算法论文：有 src 无 models/training/data
        is_algorithm = ("src" in modules and "models" not in modules
                        and "training" not in modules and "data" not in modules)

        for fr in frs:
            name = fr.get("name", "")
            expected = self._expected_module(name, modules)
            if expected is None:
                continue

            check_id = self._next_id("check")
            if expected in modules:
                checks.append(self._check(
                    check_id, f"{fr.get('id')} → 模块覆盖", "pass",
                    f"{name} 映射到 Step5 模块 '{expected}'。"))
            elif is_algorithm:
                # 算法论文不把 data/models/training/evaluation 当缺失模块
                continue
            else:
                checks.append(self._check(
                    check_id, f"{fr.get('id')} → 模块覆盖", "warning",
                    f"{name} 期望模块 '{expected}' 未在 project_structure.module_mapping 中出现。"))
                gaps.append({
                    "id": self._next_id("gap"),
                    "field": f"project_structure.module_mapping.{expected}",
                    "description": f"需求 {fr.get('id')}（{name}）缺少对应的工程模块 '{expected}'。",
                    "severity": "medium",
                    "mitigation": f"在项目结构中补充 '{expected}' 模块，或在该 FR 中标注其由现有模块间接实现。",
                })

        return checks

    def _expected_module(self, name: str, modules: set) -> str:
        """根据 FR 名称与实际模块集合，推断该 FR 应由哪个模块承载。"""
        if any(kw in name for kw in ["加载", "预处理", "划分"]):
            return "data"
        if any(kw in name for kw in ["实现"]):
            # 算法论文的"方法实现"落在 src；ML 论文落在 models
            return "src" if "src" in modules else "models"
        if any(kw in name for kw in ["训练模式", "训练", "持久化", "保存"]):
            return "training"
        if any(kw in name for kw in ["评估", "报告", "复现"]):
            # 有 evaluation 模块 → evaluation；算法论文的验证/报告落在 src
            return "evaluation" if "evaluation" in modules else ("src" if "src" in modules else None)
        return None

    # ============================================================
    # CHECK-04: 数据划分一致性（train/test vs val）
    # ============================================================

    def _check_split_consistency(self, conflicts: list) -> list:
        """
        检查数据划分口径是否一致。

        重点: 论文是否声明了独立验证集。Step4 FR-03 明确 train/test 二路，
        Step5 若用 val_accuracy 做最优模型选择，则构成工程偏离，需显式记录。
        """
        checks = []
        fr_desc = self._fr_field("description")("数据集划分")
        fr_output = self._fr_field("output")("数据集划分")

        checkpoint = self.step5.get("training_config", {}).get("checkpoint", {})
        best_metric = str(checkpoint.get("best_metric", ""))

        # 论文是否只有 train/test（无 val）
        paper_no_val = ("val" not in fr_output.lower() and
                        "test_loader" in fr_output.lower())

        check_id = self._next_id("check")
        if paper_no_val and "val" in best_metric.lower():
            checks.append(self._check(
                check_id, "数据划分（train/test vs val）", "warning",
                "论文仅 train/test 二路划分，但 Step5 checkpoint.best_metric 使用 val_accuracy。"))
            conflicts.append({
                "id": self._next_id("conflict"),
                "field": "checkpoint.best_metric",
                "description": "需求侧声明无独立验证集，设计侧却用验证集指标选择最优模型。",
                "step4_says": f"FR-03 output: {fr_output}（论文仅 train/test，无验证集）",
                "step5_says": f"checkpoint.best_metric = {best_metric}",
                "severity": "medium",
                "resolution": (
                    "方案A: 从训练集按 80/20 二次划分出验证集用于早停/模型选择（工程偏离，推荐）；"
                    "方案B: 严格复现论文，直接用 test_accuracy 选择最优模型（不推荐，有数据泄露风险）。"
                ),
            })
        else:
            checks.append(self._check(
                check_id, "数据划分（train/test vs val）", "pass",
                "划分口径在需求与设计两侧一致。"))

        return checks

    # ============================================================
    # CHECK-05: 评估指标一致性
    # ============================================================

    def _check_metric_consistency(self, conflicts: list) -> list:
        """检查验收指标与实验设计指标是否对齐。"""
        checks = []
        fr_desc = self._fr_field("description")("模型评估")
        eval_cfg = self.step5.get("experiment_design", {}).get("evaluation", {})
        primary = str(eval_cfg.get("primary_metric", ""))
        secondary = [str(m) for m in eval_cfg.get("secondary_metrics", [])]

        # 归一化关键词
        norm = lambda s: s.lower().replace(" ", "").replace("-", "").replace("_", "")
        all_s5 = norm(primary + " " + " ".join(secondary))

        check_id = self._next_id("check")
        # 检查 Step4 提到的每个指标是否都能在 Step5 找到
        s4_metrics = [m for m in ["f1", "precision", "recall", "accuracy"]
                      if m in norm(fr_desc)]
        missing = [m for m in s4_metrics if m not in all_s5]

        if not missing:
            checks.append(self._check(
                check_id, "评估指标", "pass",
                f"Step4 评估指标与 Step5 实验设计对齐（primary={primary}）。"))
        else:
            checks.append(self._check(
                check_id, "评估指标", "warning",
                f"Step4 提到的指标 {missing} 未在 Step5 实验设计中体现。"))
            conflicts.append({
                "id": self._next_id("conflict"),
                "field": "evaluation.metrics",
                "description": f"需求侧要求评估 {missing}，设计侧未将其纳入实验设计。",
                "step4_says": fr_desc[:200],
                "step5_says": f"primary={primary}, secondary={secondary}",
                "severity": "medium",
                "resolution": "将缺失指标加入 evaluation.secondary_metrics，或在需求侧明确其不在本次复现范围。",
            })

        return checks

    # ============================================================
    # CHECK-06: 数据流接口 vs 模块
    # ============================================================

    def _check_interface_consistency(self) -> list:
        """
        检查 Step4 数据流接口的端到端链路是否在 Step5 项目结构中闭环。

        轻量检查：接口两端 stage 是否都能映射到项目模块。
        """
        checks = []
        interfaces = self.step4.get("data_flow", {}).get("interfaces", [])
        files = self.step5.get("project_structure", {}).get("root_structure", {}).get("files", {})

        if not interfaces:
            return []

        # 收集所有文件（用于检查是否有对应实现文件）
        all_files = set()
        for module, fs in files.items():
            for f in fs:
                all_files.add(f"{module}/{f}" if module != "." else f)

        check_id = self._next_id("check")
        # 接口端点 → 期望实现文件。仅当对应模块真实存在时才期待该文件，
        # 避免算法论文（无 data/training/evaluation）被误报"缺少实现文件"。
        has_data = "data" in files
        has_training = "training" in files
        has_eval = "evaluation" in files
        stage_to_file = {
            "数据加载": "data/dataset.py" if has_data else None,
            "数据预处理": "data/dataset.py" if has_data else None,
            "模型构建（模型定义）": None,  # 多模型，动态
            "模型训练": "training/train.py" if has_training else None,
            "模型评估": "evaluation/evaluate.py" if has_eval else None,
        }
        uncovered = []
        for itf in interfaces:
            for endpoint in [itf.get("from_stage", ""), itf.get("to_stage", "")]:
                expected = stage_to_file.get(endpoint)
                if expected and expected not in all_files:
                    uncovered.append(f"{endpoint} → {expected}")

        if not uncovered:
            checks.append(self._check(
                check_id, "数据流接口闭环", "pass",
                f"{len(interfaces)} 个接口端到端链路均有对应实现文件。"))
        else:
            checks.append(self._check(
                check_id, "数据流接口闭环", "warning",
                f"接口端点缺少实现文件: {uncovered}"))

        return checks

    # ============================================================
    # 聚合 gaps / assumptions（来自 Step4/5 已有信息）
    # ============================================================

    def _collect_gaps(self) -> list:
        """聚合两侧已记录的缺失项。"""
        gaps = []

        # 1. Step4 gap_analysis 的显式 GAP
        ga = self.step4.get("gap_analysis", {})
        for g in ga.get("gaps", []):
            gaps.append({
                "id": self._next_id("gap"),
                "field": g.get("path", "NOT_FOUND"),
                "description": g.get("question", g.get("path", "")),
                "severity": g.get("severity", "medium"),
                "mitigation": g.get("suggested", "标记为可配置项。"),
            })

        # 2. Step5 training_config 的 NOT_FOUND_in_paper
        provenance = self.step5.get("training_config", {}).get("parameter_provenance", {})
        for field in provenance.get("NOT_FOUND_in_paper", []):
            gaps.append({
                "id": self._next_id("gap"),
                "field": f"training_config.{field}",
                "description": f"论文未提供 {field} 的取值。",
                "severity": "low",
                "mitigation": "由代码 Agent 采用合理工程默认值，并在 config.yaml 中显式标注。",
            })

        return gaps

    def _collect_assumptions(self) -> list:
        """聚合为推进实现而做的显式假设。"""
        assumptions = []

        # 1. Step4 gap_analysis 的 resolved_assumptions
        ga = self.step4.get("gap_analysis", {})
        for a in ga.get("resolved_assumptions", []):
            assumptions.append({
                "id": self._next_id("asm"),
                "description": str(a),
                "source": "step4/gap_analysis.resolved_assumptions",
            })

        # 2. Step5 框架推断（如 Caffe 论文 → PyTorch 代码）
        tech = self.step5.get("technology", {})
        fw = tech.get("framework", {})
        recommended = fw.get("recommended", "")
        mentioned = tech.get("programming_framework_mentioned", "")
        if recommended and mentioned and recommended != mentioned:
            assumptions.append({
                "id": self._next_id("asm"),
                "description": (
                    f"论文原文使用 {mentioned}，代码工程改用 {recommended} 实现。"
                    "此为显式工程偏离，需在 README 中说明。"
                ),
                "source": "step5/technology.framework",
            })

        # 3. Step5 工程默认值
        provenance = self.step5.get("training_config", {}).get("parameter_provenance", {})
        for field in provenance.get("engineering_default", []):
            assumptions.append({
                "id": self._next_id("asm"),
                "description": f"{field} 采用工程默认值（论文未指定）。",
                "source": "step5/training_config.parameter_provenance.engineering_default",
            })

        return assumptions

    # ============================================================
    # 辅助
    # ============================================================

    def _fr_field(self, field: str):
        """返回按需求名称检索该字段的闭包。"""
        frs = self.step4.get("functional_requirements", {}).get("requirements", [])

        def getter(name_keyword: str) -> str:
            for r in frs:
                if name_keyword in str(r.get("name", "")):
                    v = r.get(field, "")
                    if isinstance(v, dict):
                        return " ".join(str(x) for x in v.values())
                    return str(v)
            return ""
        return getter

    @staticmethod
    def _extract_nums(text: str) -> set:
        """从文本提取所有数字（去掉千分位逗号），返回规范化字符串集合。"""
        if not text:
            return set()
        cleaned = re.sub(r"(\d),(\d)", r"\1\2", str(text))
        return set(re.findall(r"\d+\.?\d*", cleaned))

    def _check(self, cid: str, name: str, status: str, detail: str) -> dict:
        return {"id": cid, "name": name, "status": status, "detail": detail}

    def _next_id(self, prefix: str) -> str:
        """生成连续 ID: check/conflict/gap/asm → CHECK-01, CONFLICT-01, ..."""
        key = prefix
        self._counter[key] = self._counter.get(key, 0) + 1
        label = {
            "check": "CHECK", "conflict": "CONFLICT", "gap": "GAP", "asm": "ASM",
        }[key]
        return f"{label}-{self._counter[key]:02d}"
