"""
Step6: 项目整合与代码生成准备 —— 规则驱动的跨来源整合

读取 Step4（需求分析）和 Step5（工程设计）的全部 YAML，做两件事:

  1. 一致性检查（Step6.1）: 交叉比对需求与设计，识别冲突/缺失/假设
  2. 项目整合: 将两侧信息规整为代码生成 Agent 可直接消费的 project_brief.yaml

核心输出:
  - project:               项目元信息
  - consistency_report:    一致性检查结论（冲突/缺失/假设）★ 本轮新增
  - requirements_digest:   需求摘要（从 step4 规则抽取，不重写）
  - design_digest:         设计摘要（从 step5 规则抽取，不重写）
  - traceability:          FR → 代码文件 追溯矩阵（规则映射）
  - code_generation_plan:  有序的文件生成计划（规则生成）
  - risks_and_notes:       风险点 + 给代码 Agent 的注意事项

设计原则:
  - 与旧版"LLM 总结"不同，本版纯规则驱动。整合是确定性的搬运与映射，
    不需要 LLM 生成，避免引入事实偏差。LLM 只在 Step4/5 的语义提取阶段介入。
"""

import yaml
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
OUTPUT_ROOT = PROJECT_ROOT / "data" / "output"

from step6.step6_consistency import ConsistencyChecker
from logger import logger


class ProjectBriefIntegrator:
    """
    Step6 整合器。

    读取 step4/ 和 step5/ 的全部 YAML，规则驱动地整合为 project_brief.yaml。
    一致性检查由 ConsistencyChecker 完成。
    """

    def __init__(self, paper_name: str):
        self.paper_name = paper_name
        self.paper_dir = OUTPUT_ROOT / paper_name
        self.step4_dir = self.paper_dir / "step4"
        self.step5_dir = self.paper_dir / "step5"
        self.output_dir = self.paper_dir / "step6"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.step4_yamls = self._load_all_yaml(self.step4_dir)
        self.step5_yamls = self._load_all_yaml(self.step5_dir)
        self._traceability = []  # 缓存，供 code_generation_plan 反查

    # ============================================================
    # 主流程
    # ============================================================

    def integrate(self) -> dict:
        """规则驱动地整合 Step4 + Step5，生成项目简报。"""
        self._check_prerequisites()

        # 1. 一致性检查（规则驱动）
        checker = ConsistencyChecker(self.step4_yamls, self.step5_yamls)
        consistency_report = checker.check()
        self._consistency_report = consistency_report

        # 2. 各 section 规则整合
        brief = {
            "project": self._build_project(),
            "consistency_report": consistency_report,
            "requirements_digest": self._build_requirements_digest(),
            "design_digest": self._build_design_digest(),
            "decisions": self._build_decisions(),
            "data_dependency": self._build_data_dependency(),
            "unresolved_dataset_versions": self._build_unresolved_dataset_versions(),
            "traceability": self._build_traceability(),
            "code_generation_plan": self._build_code_generation_plan(),
            "risks_and_notes": self._build_risks_and_notes(),
        }

        # 3. 元信息
        brief["_meta"] = {
            "paper_name": self.paper_name,
            "generated_at": datetime.now().isoformat(),
            "method": "rule-driven (no LLM)",
            "sources": {
                "step4_files": list(self.step4_yamls.keys()),
                "step5_files": list(self.step5_yamls.keys()),
            },
        }

        return brief

    # ============================================================
    # Section 1: project
    # ============================================================

    def _build_project(self) -> dict:
        rg = self.step5_yamls.get("reproduction_goal", {})
        ps = self.step5_yamls.get("project_structure", {})
        models = [m.get("name", "?") for m in
                  self.step5_yamls.get("model_design", {}).get("models", [])]
        dataset = (self.step5_yamls.get("dataset", {}).get("datasets", [{}])[0]
                   .get("name", "数据集"))

        task = rg.get("task", "task")
        task_zh = self.TASK_ZH.get(task, task)

        data_source = self._data_dependency_value("data_source")
        data_phrase = {
            "external_dataset": f"在 {dataset} 上",
            "synthetic": "在代码生成的合成数据上",
            "environment": "在交互式环境（仿真器）上",
            "none": "（无数据集依赖）",
        }.get(data_source, f"在 {dataset} 上")

        ptype = rg.get("paper_type", "")
        if ptype == "algorithm":
            one_liner = "复现论文提出的算法/方法，构造输入验证正确性并与论文结论对齐。"
        else:
            one_liner = (
                f"复现论文中 {'、'.join(models) or '方法'} {data_phrase}的{task_zh}，"
                f"对比不同训练模式与数据划分下的性能。"
            )

        return {
            "name": ps.get("project_name", self.paper_name),
            "paper_title": rg.get("title", "NOT_FOUND"),
            "paper_type": rg.get("paper_type", "NOT_FOUND"),
            "task": task,
            "one_liner": one_liner,
        }

    TASK_ZH = {
        "image_classification": "图像分类",
        "object_detection": "目标检测",
        "semantic_segmentation": "语义分割",
        "instance_segmentation": "实例分割",
    }

    # ============================================================
    # Section 2: requirements_digest
    # ============================================================

    def _build_requirements_digest(self) -> dict:
        fr = self.step4_yamls.get("functional_requirements", {})
        reqs = fr.get("requirements", [])
        df = self.step4_yamls.get("data_flow", {})

        return {
            "summary": self._requirements_summary(reqs, fr.get("summary", {})),
            "core_requirements": [self._map_core_requirement(r) for r in reqs],
            "data_flow_overview": self._data_flow_overview(df),
            "key_interfaces": self._key_interfaces(df),
        }

    def _requirements_summary(self, reqs: list, summary: dict) -> str:
        categories = summary.get("categories", {})
        cat_names = list(categories.keys())
        ids = "、".join(r.get("id", "?") for r in reqs)
        return (
            f"本项目包含 {len(reqs)} 个功能需求（{ids}），"
            f"涉及 {', '.join(cat_names) if cat_names else '多个'} 维度。"
        )

    def _map_core_requirement(self, r: dict) -> dict:
        """将单个 FR 映射为 {id, what, why, acceptance}。"""
        return {
            "id": r.get("id", "?"),
            "what": self._clip(r.get("description", "")),
            "why": self.SOURCE_WHY.get(
                (r.get("source", "") or "").split(".")[0],
                "论文提出的功能要求"),
            "acceptance": self._clip(r.get("verification", "")),
        }

    SOURCE_WHY = {
        "dataset": "论文指定数据集与预处理要求",
        "methodology": "论文对比的模型与训练策略",
        "implementation_details": "论文指定的训练超参数",
        "reproduction_requirements": "复现实验所需",
        "experiment": "论文报告的主要结果",
    }

    def _data_flow_overview(self, df: dict) -> str:
        stages = df.get("pipeline", [])
        if not stages:
            return "NOT_FOUND"
        parts = []
        for s in stages:
            parts.append(f"{s.get('name', '?')}（{s.get('exit_data', '')}）")
        return "数据依次经过: " + " → ".join(parts) + "。"

    def _key_interfaces(self, df: dict) -> list:
        return [
            {
                "from": itf.get("from_stage", "?"),
                "to": itf.get("to_stage", "?"),
                "data": itf.get("data", "?"),
            }
            for itf in df.get("interfaces", [])
        ]

    # ============================================================
    # Section 3: design_digest
    # ============================================================

    def _build_design_digest(self) -> dict:
        tech = self.step5_yamls.get("technology", {})
        md = self.step5_yamls.get("model_design", {})
        ps = self.step5_yamls.get("project_structure", {})
        tc = self.step5_yamls.get("training_config", {})

        framework = tech.get("framework", {}).get("recommended", "NOT_FOUND")
        models = md.get("models", [])
        dirs = ps.get("root_structure", {}).get("directories", [])
        entry = ps.get("module_mapping", {}).get("entry_point", "main.py")

        return {
            "summary": (
                f"项目采用 {framework} 框架，实现 {', '.join(m.get('name', '?') for m in models)}，"
                f"目录结构包含 {', '.join(dirs)} 模块，入口为 {entry}。"
            ),
            "tech_stack": {
                "framework": framework,
                "key_dependencies": self._key_dependencies(),
                "hardware": tech.get("hardware", {}).get("mentioned", "NOT_FOUND"),
            },
            "models": [self._map_model(m) for m in models],
            "project_structure": {
                "root_directories": dirs,
                "entry_point": entry,
            },
            "training_config_summary": self._training_summary(tc),
        }

    def _key_dependencies(self) -> list:
        """从 Step4 环境依赖聚合，缺可视化时补 matplotlib。"""
        env = (self.step4_yamls.get("acceptance_criteria", {})
               .get("reproducibility", {}).get("environment", {}))
        deps = list(env.get("key_dependencies", []))
        # FR-10 若有图表，需要 matplotlib
        fr10 = next((r for r in
                     self.step4_yamls.get("functional_requirements", {})
                     .get("requirements", [])
                     if "报告" in r.get("name", "") or "可视化" in r.get("description", "")),
                    None)
        if fr10 and "matplotlib" not in deps:
            deps.append("matplotlib")
        return deps

    def _map_model(self, m: dict) -> dict:
        return {
            "name": m.get("code_class_name", m.get("name", "?")),
            "type": m.get("type", "NOT_FOUND"),
            "file": self._model_file(m),
            "key_params": f"num_classes={m.get('num_classes', '?')}, "
                          f"pretrained={'是' if 'pretrained' in str(m.get('pretrained', '')).lower() else '否'}",
        }

    def _model_file(self, m: dict) -> str:
        stem = self._sanitize_filename(
            m.get("code_class_name", m.get("name", ""))
        )
        return f"models/{stem}.py"

    @staticmethod
    def _sanitize_filename(name: str) -> str:
        """生成安全的 Python 文件名（小写+下划线），与 step5_6_struct 保持一致。"""
        import re
        s1 = re.sub(r'(.)([A-Z][a-z]+)', r'\1_\2', name)
        s2 = re.sub(r'([a-z0-9])([A-Z])', r'\1_\2', s1)
        cleaned = re.sub(r'[^a-zA-Z0-9_]', '', s2)
        return cleaned.lower().strip('_') or "model"

    def _training_summary(self, tc: dict) -> dict:
        training = tc.get("training", {})
        opt = training.get("optimizer", {})
        sched = training.get("scheduler", {})
        return {
            "optimizer": opt.get("name", "NOT_FOUND"),
            "learning_rate": opt.get("lr", "NOT_FOUND"),
            "batch_size": training.get("batch_size", "NOT_FOUND"),
            "epochs": training.get("epochs", "NOT_FOUND"),
            "loss": training.get("loss_function", "NOT_FOUND"),
            "scheduler": (
                f"{sched.get('name', 'NOT_FOUND')} "
                f"(step_size={sched.get('step_size', '?')}, gamma={sched.get('gamma', '?')})"
            ),
        }

    # ============================================================
    # Section 3.5: decisions（跨来源确认的关键决策，供 Validator 一致性检查）
    # ============================================================

    def _build_decisions(self) -> dict:
        """把跨来源确认过的关键决策集中记录，供 Step7 Validator 做一致性检查。

        这些值来自 Step4/Step5 的最终结论，避免在代码里各写一套。
        """
        tc = self.step5_yamls.get("training_config", {})
        framework = self._framework()
        dd = self._build_data_dependency()

        decisions = {
            "framework": framework,
            "data_source": dd["data_source"],
            "acquisition": dd["acquisition"],
        }
        # 训练相关决策只在可训练论文里给出；纯算法/无训练论文不硬塞 scratch/val_split 等 ML 语义。
        if self._is_trainable():
            decisions.update({
                # 80/20（val_split_ratio=0.2）：从训练集二次划分验证集用于早停/模型选择。
                # 工程默认值——Step5 只提取 train/test 二路划分、未提取验证集比例，故固定 80/20。
                "val_split_ratio": 0.2,
                "val_split_note": "80/20 从训练集二次划分出验证集（工程默认，非论文要求）",
                "training_modes": list((tc.get("training_modes", {}) or {}).keys()),
                "best_metric": (tc.get("checkpoint", {}) or {}).get("best_metric", "val_accuracy"),
            })
        else:
            decisions["training_modes"] = []
        return decisions

    def _build_data_dependency(self) -> dict:
        """透传 Step5 判定的数据依赖类型与获取方式，供 Step7/8 使用。

        data_source: external_dataset / synthetic / environment / none
        acquisition: local_path / auto / n/a（非 external_dataset 时固定 n/a）
        缺失/非法值回退保守默认 external_dataset + local_path。

        leaf_based_split 时额外登记 leaf_mapping 依赖：dataset.py 必须读取叶子
        映射文件做分组划分；映射文件缺失时必须显式告警并记录
        leaf_constraint_effective=false，禁止静默回退成随机划分。
        """
        ds = self.step5_yamls.get("dataset", {})
        data_source = str(ds.get("data_source") or "external_dataset").strip().lower()
        if data_source not in {"external_dataset", "synthetic", "environment", "none"}:
            data_source = "external_dataset"
        acquisition = str(ds.get("acquisition") or "").strip().lower()
        if data_source != "external_dataset":
            acquisition = "n/a"
        elif acquisition not in {"local_path", "auto"}:
            acquisition = "local_path"
        dd = {
            "data_source": data_source,
            "acquisition": acquisition,
            "note": ds.get("data_source_note", ""),
        }
        if self._is_leaf_split():
            dd["leaf_mapping"] = {
                "required": True,
                "file": "data/leaf_mappings.csv",
                "constraint": (
                    "同一片叶子的多张图片必须保持在同一个集合中（train/val/test 不得跨集）"
                ),
                "on_missing": (
                    "映射文件缺失时必须 logger.warning 显式告警并在结果/配置中记录 "
                    "leaf_constraint_effective=false，不得静默回退为随机划分"
                ),
            }
        return dd

    def _build_unresolved_dataset_versions(self) -> list:
        """检测 Step5 中数据来源/处理参数未定义（NOT_FOUND）的 dataset version。

        例如某版本：论文只提某种分割/预处理思路，未给完整脚本/参数，
        implementation_status 含 NOT_FOUND。Step7 据此在 config.yaml 中标记
        unresolved，而非让代码用 color 兜底伪装成 segmented。
        方法描述已知而参数未知（PARTIAL）时同样标记，reason 里保留已知部分。
        """
        versions = (self.step5_yamls.get("dataset", {})
                    .get("preprocessing", {}).get("versions", {}))
        unresolved = []
        for name, cfg in versions.items():
            if not isinstance(cfg, dict):
                continue
            seg = cfg.get("segmentation_algorithm", {})
            status = str(seg.get("implementation_status", ""))
            if seg and "NOT_FOUND" in status.upper():
                if status.upper().startswith("PARTIAL"):
                    reason = (
                        f"Segmented preprocessing only partially specified for '{name}': "
                        f"method description known ({seg.get('method', '')[:120]}), "
                        "but script thresholds/parameters are NOT_FOUND in the paper. "
                        "Implement from the method description or a documented fallback; "
                        "do not silently substitute another version."
                    )
                else:
                    reason = (
                        f"Dataset source/processing method not specified for '{name}' "
                        f"(Step5 implementation_status={status})"
                    )
                unresolved.append({"name": name, "reason": reason})
        return unresolved

    # ============================================================
    # Section 4: traceability（规则映射 FR → 实现）
    # ============================================================

    def _build_traceability(self) -> list:
        frs = self.step4_yamls.get("functional_requirements", {}).get("requirements", [])
        models = self.step5_yamls.get("model_design", {}).get("models", [])

        # 模型名 → 文件::类 映射
        model_refs = {
            m.get("name", "").lower():
                f"models/{self._sanitize_filename(m.get('code_class_name', m.get('name', '')))}.py::"
                f"{m.get('code_class_name', m.get('name', ''))}"
            for m in models
        }

        result = []
        for r in frs:
            name = r.get("name", "")
            refs, design_ref = self._map_fr_to_impl(name, model_refs)
            result.append({
                "requirement_id": r.get("id", "?"),
                "requirement_name": name,
                "planned_implementation": refs,
                "design_ref": design_ref,
            })

        self._traceability = result
        return result

    def _map_fr_to_impl(self, name: str, model_refs: dict):
        """根据 FR 名称关键词映射到实现文件与方法。返回 (refs, design_ref)。

        只引用真实存在于 project_structure 的模块，避免算法论文
        （无 data/training/evaluation）出现悬空引用。
        """
        n = name.lower()
        pfiles = self._project_files()
        modules = set(pfiles.keys())

        # 模型实现：动态匹配模型名
        for mk, ref in model_refs.items():
            if mk in n and "实现" in n:
                return [ref], "step5/model_design.yaml"

        # 算法/方法实现（纯算法论文，src/）
        if ("算法" in n or "方法实现" in n) and "src" in modules:
            src_files = [f for f in pfiles.get("src", []) if f.endswith(".py")]
            return [f"{f}::run" for f in src_files], "step5/reproduction_goal.yaml"

        # 静态规则：仅当对应模块真实存在才返回其 ref
        preprocess_method = "preprocess_image" if self._is_image_data() else "preprocess_data"
        rules = []
        if "data" in modules:
            rules += [
                (["加载"], [f"data/dataset.py::{self._dataset_class_name()}"], "step5/dataset.yaml"),
                (["预处理"], [f"data/dataset.py::{preprocess_method}"], "step5/dataset.yaml"),
                (["划分"], ["data/dataset.py::split_dataset"], "step5/dataset.yaml"),
            ]
        if "training" in modules:
            rules += [
                (["训练模式"], ["training/train.py::initialize_model"], "step5/training_config.yaml"),
                (["持久化", "保存"], ["training/train.py::save_model"], "step5/training_config.yaml"),
                (["训练"], ["training/train.py::train_model"], "step5/training_config.yaml"),
            ]
        if "evaluation" in modules:
            eval_refs = ["evaluation/evaluate.py::evaluate_model"]
            if "utils" in modules:
                eval_refs.append("utils/metrics.py::compute_metrics")
            rules += [
                (["评估"], eval_refs, "step5/experiment_design.yaml"),
                (["报告", "复现"], ["evaluation/evaluate.py::generate_report"], "step5/experiment_design.yaml"),
            ]
        for keywords, refs, design_ref in rules:
            if any(kw in n for kw in keywords):
                return refs, design_ref

        # 算法论文：未匹配到具体模块的 FR（数据预处理/评估/报告等）统一落到 src，
        # 避免产生 planned_implementation: [] 的悬空追溯。
        if ("src" in modules and "data" not in modules and "training" not in modules
                and "evaluation" not in modules):
            src_files = [f for f in pfiles.get("src", []) if f.endswith(".py")]
            if src_files:
                return [f"{src_files[0]}::run"], "step5/reproduction_goal.yaml"

        return [], None

    # ============================================================
    # Section 5: code_generation_plan
    # ============================================================

    def _build_code_generation_plan(self) -> dict:
        ps = self.step5_yamls.get("project_structure", {})
        files = ps.get("root_structure", {}).get("files", {})
        models = self.step5_yamls.get("model_design", {}).get("models", [])

        model_class_map = {
            self._sanitize_filename(m.get("code_class_name", m.get("name", ""))):
                m.get("code_class_name", m.get("name", ""))
            for m in models
        }

        # 展平文件列表 → (module, filename, full_path)
        all_files = []
        for module, fs in files.items():
            for f in fs:
                full = f if module == "." else f"{module}/{f}"
                all_files.append((module, f, full))

        # 实际存在的顶层模块 → 每个文件 spec 的 depends_on/notes 按此生成
        pfiles = self._project_files()

        return {
            "generation_order": self._build_generation_order(all_files),
            "per_file_specs": [
                self._build_file_spec(module, f, full, model_class_map, pfiles)
                for module, f, full in all_files
            ],
        }

    def _build_generation_order(self, all_files: list) -> list:
        """按依赖关系分阶段排序（被依赖的文件先生成），阶段由实际存在的模块推导。"""
        # 阶段 → (模块列表, 描述, 原因)；缺失的模块自动跳过
        module_phases = [
            (["config"], "基础配置",
             "配置是所有后续模块的基础，必须先定义。"),
            (["data"], "数据模块",
             "数据加载依赖配置。"),
            (["models", "src"], "核心逻辑（模型/算法）",
             "模型/算法是训练和评估的前提。"),
            (["utils"], "工具模块",
             "指标计算等工具函数。"),
            (["training", "evaluation"], "训练与评估",
             "训练依赖模型和数据，评估依赖训练好的模型。"),
            (["."], "入口与文档",
             "入口文件整合所有模块，文档和依赖列表最后生成。"),
        ]

        order = []
        phase = 0
        for modules, desc, reason in module_phases:
            files_in_phase = [full for m, _, full in all_files if m in modules]
            if not files_in_phase:
                continue
            phase += 1
            order.append({
                "phase": phase,
                "description": desc,
                "files": files_in_phase,
                "reason": reason,
            })
        return order

    def _build_file_spec(self, module: str, filename: str, full: str,
                         model_class_map: dict, pfiles: dict) -> dict:
        """按模块生成单个文件的 spec。

        pfiles: 实际存在的顶层模块 → 文件列表（来自 project_structure），
        用于让 depends_on/notes 只引用真实存在的模块（算法论文无 data/training）。
        """
        is_cnn = self._is_torchvision_cnn()
        modules_present = set(pfiles.keys())
        spec = {
            "file": full,
            "purpose": "",
            "classes": [],
            "key_methods": [],
            "depends_on": [],
            "satisfies_requirements": self._fr_satisfied_by(full),
            "notes": "",
        }

        if module == "config":
            trainable = self._is_trainable()
            has_data_module = "data" in modules_present
            # 顶层键按原型拼装：data 键仅在确有 data 模块时要求，训练键仅在可训练论文里要求。
            required_keys = [
                f"project（name/paper/framework: {self._framework()}）",
            ]
            extra_notes = []
            if has_data_module:
                data_desc = f"data（根目录/版本/{'图像尺寸/mean/std/normalize' if is_cnn else '特征维度/归一化方式'}）"
                required_keys.append(data_desc)
            if trainable:
                required_keys += [
                    "split_strategy（train_test_splits/seed/default_train_ratio，另加 val_split_ratio: 0.2 即 80/20 工程偏离）",
                    "reproduction_mode（'engineering' 或 'strict'）",
                    "model（architectures/num_classes/dropout_ratio/weight_init）",
                    f"training_modes（{'scratch/transfer' if is_cnn else 'scratch'}）",
                    "training（batch_size 按模型分表/epochs/optimizer/scheduler）",
                    "checkpoint、inference",
                ]
                extra_notes.append(
                    "【reproduction_mode】'engineering'=工程模式(train→val→best→test：划分 val、每 epoch 验证、按 val_accuracy 选最优)；"
                    "'strict'=严格复现论文(train→训练结束→test：不划分 val、不验证、只保存最终模型)。"
                )
            required_keys += ["experiment（runs 列表）", "logging、system（device/num_workers/seed）"]
            if trainable:
                base_note = "包含数据路径、预处理参数、划分比例、模型参数、训练超参数等。"
            elif has_data_module:
                base_note = "本论文无训练流程：不要生成 model/training_modes/training/checkpoint/inference 键，仅包含数据路径、预处理与算法参数。"
            else:
                base_note = "本论文无训练、无外部数据：不要生成 data/model/training_modes/training/checkpoint/inference/split_strategy 键，仅包含算法参数与运行配置。"
            notes = (
                base_note
                + "顶层键必须包含: " + "、".join(required_keys) + "。"
                + "".join(extra_notes)
            )
            if has_data_module:
                notes += (
                    f"data.versions 是版本名列表（如 {self._dataset_versions()}）。"
                    "每个版本的具体信息写在 data 段下（与 versions 平级）的 data.<version> 键："
                    "只有 unresolved_dataset_versions 列出的版本写 {status: unresolved, reason: ...}（不要编造 path）；"
                    "其余版本一律写 {status: resolved, path: ..., description: ...}，绝不要把 resolved 版本写成 unresolved。"
                )
            notes += "其余代码读取的键都必须在此列全，避免代码引用不存在的键。"
            notes += (
                "【NOT_FOUND 禁止入数值】凡代码会当数字消费的字段（num_classes/batch_size/"
                "epochs/learning_rate/feature_dim/dropout_ratio 等），论文能推导出具体值时必须写具体数字；"
                "确实未知时写 null 或省略该键（让代码用 .get(key, 默认值) 兜底），"
                "严禁写字符串 'NOT_FOUND'，否则代码里 np.zeros(('NOT_FOUND','NOT_FOUND')) 或 int('NOT_FOUND') 会崩溃。"
            )
            spec.update({
                "purpose": "集中管理所有配置参数",
                "notes": notes,
            })
            spec["depends_on"] = []
            spec["notes"] += self._config_data_dependency_note()

        elif module == "data":
            # 非外部数据集：数据在代码内生成/来自环境/无数据依赖，不加载本地图片路径。
            if self._data_dependency_value("data_source") != "external_dataset":
                spec.update(self._data_module_spec_non_external())
                return spec

            dataset_class = self._dataset_class_name()
            versions = self._dataset_versions()
            default_version = self._default_version()
            is_image = self._is_image_data()
            preprocess_method = "preprocess_image" if is_image else "preprocess_data"

            data_notes = [
                f"实现数据集加载、预处理和划分。"
                f"需处理 {'、'.join(versions) if versions else '论文指定'} 等版本。",
                "所有 open() 读文件必须显式 encoding='utf-8'（config.yaml 含中文，Windows 默认 GBK 会读崩）。",
                f"【version 数据流】get_dataloaders（或等价加载函数）必须接受 version 参数"
                f"（默认 '{default_version}'），并把它原样传给 {dataset_class}(version=...)；"
                f"绝不能从 config['data'] 读默认值写死，否则各版本实验无差别。",
            ]
            if is_image:
                data_notes.append(
                    "图像加载用 PIL（Pillow，已在依赖列表），不要引入 opencv（cv2）——cv2 不在依赖列表里。"
                )
                data_notes.append(
                    "【train/val/test 增强分离】训练集必须加论文指定的数据增强，验证集/测试集绝不能加。"
                    "因为 Subset 共享同一个底层 Dataset 对象，严禁写 splits['train'].dataset.transform = ... 这种改共享对象属性的写法"
                    "（后写的覆盖先写的，训练集会因此丢增强）。正确做法：底层 Dataset.__getitem__ 只返回原始 PIL 图 + label（不设 transform），"
                    "再用轻量包装类 TransformSubset(Dataset)（持有一个 Subset + 一个 transform，__getitem__ 先取 (x,y) 再套 transform）"
                    "分别包 train/test/val：train 用 get_transform(train=True)，test/val 用 get_transform(train=False)。"
                )
            else:
                data_notes.append(
                    "【非图像数据】数据是图/文本/序列/表格等非图像数据，不要引入 PIL/opencv 或图像增强；"
                    "按论文的数据格式加载（如解析文本/JSON 图文件、读取序列、读取 CSV 等），"
                    f"并把预处理逻辑放在 {preprocess_method} 方法里（按论文描述实现，无图像通道/尺寸概念）。"
                )

            if self._is_leaf_split():
                data_notes.append(
                    "【划分必须按对象分组】engineering 模式从训练集二次划分 val 时，也必须按对象 id 分组划分"
                    "（同一对象的图片绝不能分到 train 和 val 两边，否则数据泄漏）。"
                    "严禁用 val_rng.sample(range(len(train_indices)), val_size) 这种图片级随机抽样。"
                    "正确做法：先 train/test 按对象分组（split_dataset 已保证），再从 train_indices 里按对象二次划分——"
                    "写一个按对象分组的 helper（输入 indices，用 dataset 的对象 id 映射聚合对象 id，"
                    "shuffle 对象 id 后贪心分配直到达到 val_split_ratio 比例），返回 (val_indices, 剩余 train_indices)。"
                )

            unresolved = self._build_unresolved_dataset_versions()
            unresolved_names = [u.get("name") for u in unresolved if u.get("name")]
            if unresolved_names:
                data_notes.append(
                    f"【unresolved 版本】{', '.join(unresolved_names)} 的数据来源/处理方法在论文中 NOT_FOUND（无完整参数）。"
                    "初始化时必须读取 config['data'][version]['status']（顶层，如 "
                    "config['data'][<version>]['status']，不是 config['data']['versions'][...]）；"
                    "若 == 'unresolved'，立即 raise NotImplementedError(reason)，"
                    "严禁用其他版本兜底伪装。"
                )

            spec.update({
                "purpose": "实现数据集加载、预处理和划分",
                "classes": [dataset_class],
                "key_methods": ["__init__", "__len__", "__getitem__",
                                preprocess_method, "split_dataset"],
                "depends_on": ["config/config.yaml"],
                "notes": "".join(data_notes),
            })

        elif module == "models":
            # 模型文件依赖（供 training / evaluation 引用）
            model_files = [f"models/{stem}.py" for stem in model_class_map]

            if filename == "__init__.py":
                export_lines = "; ".join(
                    f"from .{stem} import {cls}"
                    for stem, cls in model_class_map.items()
                )
                spec.update({
                    "purpose": "模型模块的初始化文件，便于统一导入",
                    "notes": (f"导出各模型类，类名必须与各模型文件定义完全一致: "
                              f"{export_lines}。不要自行改名或改成其他拼写。"),
                })
            else:
                stem = filename.replace(".py", "")
                cls = model_class_map.get(stem, stem)
                spec.update({
                    "purpose": f"定义 {cls} 模型结构",
                    "classes": [cls],
                    "key_methods": ["__init__", "forward"],
                    "depends_on": ["config/config.yaml"],
                    "notes": self._models_module_note(cls, is_cnn),
                })

        elif module == "src":
            # 纯算法论文：实现论文方法/算法，无数据/训练/模型
            stem = filename.replace(".py", "")
            depends = ["config/config.yaml"] if "config" in modules_present else []
            spec.update({
                "purpose": f"实现论文方法 {stem}",
                "key_methods": ["run"],
                "depends_on": depends,
                "notes": (
                    "实现论文提出的算法/方法，构造输入并验证与论文结果的一致性。"
                    "不要引入 torch/torchvision（非训练论文），依赖 numpy/pyyaml。"
                    "所有 open() 读配置文件必须 encoding='utf-8'。"
                    + self._algorithm_target_note()
                ),
            })

        elif module == "training":
            model_files = [f"models/{stem}.py" for stem in model_class_map]
            depends = []
            if "config" in modules_present:
                depends.append("config/config.yaml")
            if "data" in modules_present:
                depends.append("data/dataset.py")
            depends += model_files
            spec.update({
                "purpose": "实现模型训练流程",
                "key_methods": ["initialize_model", "train_model", "save_model"],
                "depends_on": depends,
                "notes": self._training_module_note(is_cnn),
            })

        elif module == "evaluation":
            model_files = [f"models/{stem}.py" for stem in model_class_map]
            depends = []
            if "config" in modules_present:
                depends.append("config/config.yaml")
            if "training" in modules_present:
                depends.append("training/train.py")
            if "utils" in modules_present:
                depends.append("utils/metrics.py")
            depends += model_files
            spec.update({
                "purpose": "实现模型评估和报告生成",
                "key_methods": ["evaluate_model", "generate_report"],
                "depends_on": depends,
                "notes": "计算评估指标；生成论文要求数值报告和工程辅助图表。"
                         "【接口约定】generate_report 的签名是 (metrics, run_name, output_dir, config)，"
                         "返回单个 run 的报告路径（不要做成接收 results 字典并批量绘制对比图的函数）。"
                         "本文件只提供 evaluate_model / generate_report 等函数，"
                         "不要写 __main__ 入口或目录扫描逻辑（入口统一在 main.py）。"
                         "【绘图参数】_generate_metrics_bar_chart 接收完整的 metrics dict"
                         "（内部一次绘制 mean_f1/precision/recall/accuracy），调用时传 metrics 整个 dict，"
                         "严禁传 metrics['mean_f1'] 单标量；_generate_confusion_matrix_plot 则接收 "
                         "metrics['confusion_matrix'] 二维数组。"
                         "【matplotlib 可选】matplotlib 仅是可视化辅助依赖，目标环境可能未安装，不要假设它一定存在。"
                         "文件顶部必须用 try/except 包裹 matplotlib 导入（"
                         "try: import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt "
                         "except ImportError: plt = None），所有绘图函数在 plt is None 时直接跳过并返回 None，"
                         "generate_report 在无 matplotlib 时仍输出文本报告。"
                         "严禁因 matplotlib 缺失在 import 阶段或运行期崩溃——指标计算与文本报告必须无条件完成。",
            })

        elif module == "utils":
            metrics = self._eval_metrics()
            spec.update({
                "purpose": "实现评估指标计算函数",
                "key_methods": ["compute_metrics"],
                "depends_on": [],
                "notes": (f"计算 {'、'.join(metrics)}。" if metrics
                          else "计算论文要求的评估指标。"),
            })

        elif module == ".":
            if filename == "main.py":
                depends = []
                if "config" in modules_present:
                    depends.append("config/config.yaml")
                if "data" in modules_present:
                    depends.append("data/dataset.py")
                if "src" in modules_present:
                    depends.extend(f for f in pfiles.get("src", [])
                                   if f.endswith(".py"))
                if "training" in modules_present:
                    depends.append("training/train.py")
                if "evaluation" in modules_present:
                    depends.append("evaluation/evaluate.py")

                is_algorithm = (
                    "src" in modules_present
                    and "training" not in modules_present
                    and "data" not in modules_present
                )
                if is_algorithm:
                    notes = ("解析配置，执行论文方法（算法），验证与论文结果的一致性。"
                             "所有 open() 读配置文件必须 encoding='utf-8'。"
                             "【命令行参数】支持 --config <path>（配置文件路径，"
                             "默认 'config/config.yaml'）。"
                             "不要引入 torch/torchvision（非训练论文），依赖 numpy/pyyaml。")
                else:
                    notes = ("解析配置，执行数据加载、训练、评估和报告生成。"
                             "所有 open() 读 config.yaml 必须 encoding='utf-8'。"
                             "【命令行参数】必须支持两个可选参数："
                             "--run <int>（只运行 experiment.runs 中该下标的单个实验，省略则运行全部 runs）；"
                             "--config <path>（配置文件路径，默认 'config/config.yaml'）。"
                             "解析后若给了 --run，只执行对应下标的那一个 run 并输出其指标；"
                             "若给了 --config，从该路径加载配置（否则按默认相对路径）。"
                             "【接口约定】evaluation.evaluate.generate_report 的签名是 "
                             "(metrics, run_name, output_dir, config)，返回单个 run 的报告路径；"
                             "入口自行生成汇总报告，不要用错误的参数个数调用该函数。"
                             "【segmented 跳过】执行 experiment.runs 前，先读 config['data'][version]['status']，"
                             "若 dataset_version 的 status == 'unresolved'，跳过该 run 并记为 skipped（不是报错失败），不要尝试训练它。"
                             "【reproduction_mode】把 config['reproduction_mode'] 传给训练逻辑（strict 时不划 val、不验证）。"
                             "【version 数据流】从 experiment.runs[].dataset_version 读取版本，"
                             "调用 get_dataloaders 时显式传 version=dataset_version。"
                             "【报告目录前置初始化】report/output_dir 目录必须在遍历 runs 的循环之前就初始化"
                             "（如 output_dir = Path('reports'); output_dir.mkdir(exist_ok=True)），"
                             "汇总报告 summary_report.txt 也要能在一个 run 都没成功时正常生成——"
                             "严禁依赖循环体内才赋值的局部变量（否则模型初始化失败的 run continue 后，"
                             "汇总段会 UnboundLocalError）。")

                spec.update({
                    "purpose": ("项目入口，串联整个流程" if not is_algorithm
                                else "项目入口，执行论文方法"),
                    "key_methods": ["main"],
                    "depends_on": depends,
                    "satisfies_requirements": self._all_fr_ids(),
                    "notes": notes,
                })
            elif filename == "requirements.txt":
                spec.update({
                    "purpose": "列出项目依赖",
                    "notes": f"包含 {', '.join(self._key_dependencies())}。",
                })
            elif filename == "README.md":
                tree = self._file_tree(pfiles)
                spec.update({
                    "purpose": "项目说明文档",
                    "notes": (
                        "说明项目结构、运行方法和复现步骤。"
                        "【项目结构】以下为本项目实际的文件结构，README 中的目录树必须与之完全一致，"
                        "严禁凭空描述不存在的目录/文件（如 configs/、setup.py、experiments/、"
                        "results/、tests/、src/models/、src/training/ 等）：\n" + tree
                    ),
                })

        return spec

    def _fr_satisfied_by(self, file: str) -> list:
        """反查哪些 FR 的 planned_implementation 指向该文件。"""
        if not self._traceability:
            return []
        result = []
        for entry in self._traceability:
            for ref in entry.get("planned_implementation", []):
                if ref.split("::")[0] == file:
                    result.append(entry.get("requirement_id"))
                    break
        return result

    def _all_fr_ids(self) -> list:
        """返回全部功能需求 ID（用于入口文件 main.py 串联所有功能）。"""
        frs = self.step4_yamls.get("functional_requirements", {}).get("requirements", [])
        return [r.get("id") for r in frs if r.get("id")]

    # ============================================================
    # Section 6: risks_and_notes
    # ============================================================

    def _build_risks_and_notes(self) -> dict:
        report = self._build_consistency_ref()
        return {
            "gaps": report["gaps"],
            "conflicts": report["conflicts"],
            "assumptions": report["assumptions"],
            "for_code_agent": self._for_code_agent_notes(),
        }

    def _build_consistency_ref(self) -> dict:
        """从一致性检查结果中提取 gaps/conflicts/assumptions，供 risks_and_notes 复用。"""
        report = self._consistency_report
        return {
            "gaps": [
                {"issue": g.get("description", ""), "severity": g.get("severity", "medium"),
                 "mitigation": g.get("mitigation", "")}
                for g in report.get("gaps", [])
            ],
            "conflicts": [
                {"issue": c.get("description", ""), "severity": c.get("severity", "high"),
                 "step4_says": c.get("step4_says", ""), "step5_says": c.get("step5_says", ""),
                 "resolution": c.get("resolution", "")}
                for c in report.get("conflicts", [])
            ],
            "assumptions": [
                a.get("description", "") for a in report.get("assumptions", [])
            ],
        }

    def _for_code_agent_notes(self) -> list:
        """给代码生成 Agent 的注意事项（规则汇总）。"""
        notes = []

        # 从 FR 的 implementation_notes / design_decision 提取
        frs = self.step4_yamls.get("functional_requirements", {}).get("requirements", [])
        for r in frs:
            impl_note = r.get("implementation_notes")
            if impl_note:
                notes.append(f"[{r.get('id')}] {impl_note}")
            design_decision = r.get("design_decision")
            if design_decision:
                notes.append(f"[{r.get('id')}] {design_decision}")

        # 训练配置的关键偏离提示
        tc = self.step5_yamls.get("training_config", {})
        checkpoint_note = tc.get("checkpoint", {}).get("note")
        if checkpoint_note:
            notes.append(f"[checkpoint] {checkpoint_note}")

        # leaf_based_split：叶子级分组划分约束（防止数据泄漏 + 禁止静默回退）
        if self._is_leaf_split():
            notes.append(
                "[leaf_mapping] 数据划分必须按叶子分组（同一片叶子的多张图片不得跨 "
                "train/val/test 集）：dataset.py 读取叶子映射文件（config 的 "
                "split_strategy.leaf_mapping_file，缺省 data/leaf_mappings.csv）做分组划分；"
                "映射文件缺失时必须 logger.warning 显式告警，并在结果/配置中记录 "
                "leaf_constraint_effective=false，严禁静默回退为随机划分。"
            )

        # 论文结论不可写死（Step7 会做规则检查兜底）
        notes.append(
            "[结论] 论文的'关键发现'（如某模型优于另一模型、迁移学习优于从头训练）是论文结论，"
            "代码必须从实际评估结果动态计算比较，严禁在报告/README/注释中写死结论性措辞"
            "（如 'consistently outperforms' / 'always yields better results'）。"
        )

        return notes

    # ============================================================
    # 动态数据（从 Step5 YAML 读取，避免 plant 专属硬编码）
    # ============================================================

    def _project_files(self) -> dict:
        """project_structure 中实际存在的顶层模块 → 文件列表。

        例如 {"data": ["data/dataset.py"], "src": ["src/graph_program.py"],
              ".": ["main.py", "requirements.txt", "README.md"]}。
        Step6 的 depends_on/traceability 据此只引用真实存在的文件。
        """
        ps = self.step5_yamls.get("project_structure", {})
        files = ps.get("root_structure", {}).get("files", {})
        result = {}
        for module, fs in files.items():
            for f in fs:
                full = f if module == "." else f"{module}/{f}"
                result.setdefault(module, []).append(full)
        return result

    def _file_tree(self, pfiles: dict) -> str:
        """把实际文件结构渲染成逐行文本，注入 README 提示，避免 README 臆造目录。"""
        lines = []
        for module in sorted(pfiles.keys()):
            for f in sorted(pfiles[module]):
                lines.append(f if module == "." else f"{module}/{f}")
        return "\n".join("  " + ln for ln in lines)

    def _algorithm_target_note(self) -> str:
        """从 Step5 提取论文核心贡献与关键结果，注入 src 实现提示。

        让生成 Agent 复现论文的具体 case study（如 connectedness/acyclicity/
        Bellman-Ford），而不是自由发挥与论文无关的示例程序。
        """
        rg = self.step5_yamls.get("reproduction_goal", {})
        contrib = str(rg.get("main_contribution", "")).strip()
        results = str(self.step5_yamls.get("experiment_design", {})
                      .get("important_results", "")).strip()

        parts = []
        if contrib and contrib != "NOT_FOUND":
            parts.append(f"论文核心贡献：{self._clip(contrib, 600)}")
        if results and results != "NOT_FOUND":
            parts.append(f"论文关键结果/案例：{self._clip(results, 600)}")
        if not parts:
            return ""
        return (
            "\n【复现目标】" + "；".join(parts) + "。"
            "实现时必须覆盖这些具体案例/程序（而非无关示例），构造对应输入验证正确性，"
            "并对论文的时间复杂度等结论给出定性验证。"
        )

    def _dataset_class_name(self) -> str:
        """从数据集名生成数据集类名: "ImageNet dataset" → "ImageNetDataset".

        NOT_FOUND 哨兵绝不能拼进类名（否则生成 NOTFOUNDDataset）。
        """
        import re
        ds_list = self.step5_yamls.get("dataset", {}).get("datasets", [{}])
        name = str(ds_list[0].get("name", "Dataset")) if ds_list else "Dataset"
        if not name or name in ("NOT_FOUND", "NOTFOUND", "not_found", "未知"):
            name = "Dataset"
        words = re.findall(r"[A-Za-z0-9]+", name)
        base = "".join(w[0].upper() + w[1:] for w in words) if words else "Dataset"
        if base and base[0].isdigit():
            base = "D" + base
        return base if base.endswith("Dataset") else base + "Dataset"

    def _dataset_versions(self) -> list:
        """数据集版本列表，如 ["color", "grayscale"]."""
        return self.step5_yamls.get("dataset", {}).get("dataset_versions", []) or []

    def _default_version(self) -> str:
        """默认数据集版本。"""
        versions = self._dataset_versions()
        default = (self.step5_yamls.get("dataset", {})
                   .get("preprocessing", {}).get("version_selector", {})
                   .get("default", ""))
        return default or (versions[0] if versions else "default")

    def _is_leaf_split(self) -> bool:
        """是否要求 leaf 级划分（同一对象的多张图必须同集合）。"""
        method = (self.step5_yamls.get("dataset", {})
                  .get("split_strategy", {}).get("method", ""))
        return method == "leaf_based_split"

    def _eval_metrics(self) -> list:
        """评估指标列表（来自 Step5 experiment_design）。"""
        ev = self.step5_yamls.get("experiment_design", {}).get("evaluation", {})
        return ev.get("secondary_metrics", []) or []

    def _data_dependency_value(self, key: str, default: str = "external_dataset") -> str:
        """读取数据依赖字段（data_source/acquisition）的已校验值。"""
        dd = self._build_data_dependency()
        return dd.get(key, default)

    def _framework(self) -> str:
        """读取 Step5 推荐的代码框架（无推荐时 NOT_FOUND，不硬编码 pytorch）。"""
        tech = self.step5_yamls.get("technology", {})
        return tech.get("framework", {}).get("recommended", "NOT_FOUND")

    def _is_trainable(self) -> bool:
        """是否存在训练流程（Step5 training_config 是否有训练模式）。"""
        tc = self.step5_yamls.get("training_config", {})
        training = tc.get("training", {})
        if isinstance(training, dict) and training.get("applies") is False:
            return False
        return bool(tc.get("training_modes"))

    _IMAGE_DATA_SIGNALS = (
        "imagenet", "cifar", "mnist", "coco", "leaf", "叶片", "plantvillage",
        "植物", "pixel", "像素", "resize", "分辨率",
    )

    def _is_image_data(self) -> bool:
        """数据集是否为图像数据（决定 data 模块是否用 PIL/图像增强）。"""
        ds = self.step5_yamls.get("dataset", {})
        if not isinstance(ds, dict):
            return False
        parts = [str(ds.get(k, "")) for k in ("dataset_source", "preprocessing")]
        for d in (ds.get("datasets") or []):
            if isinstance(d, dict):
                parts.append(str(d.get("data_type", "")))
                parts.append(str(d.get("name", "")))
        text = " ".join(parts).lower()
        return any(sig in text for sig in self._IMAGE_DATA_SIGNALS)

    def _config_data_dependency_note(self) -> str:
        """config.yaml 顶层键需写入数据依赖字段的说明（Step7 会确定性注入兜底）。"""
        data_source = self._data_dependency_value("data_source")
        acquisition = self._data_dependency_value("acquisition", "local_path")
        if data_source == "none":
            data_note = "（data_source=none，无需 data 段，也不要生成 data.<version>.path）"
        elif data_source != "external_dataset":
            data_note = "（data_source 非 external_dataset，data 段无需 path，数据在代码内生成/来自环境）"
        else:
            data_note = ""
        return (
            "【data_source/acquisition】顶层必须写 data_source 和 acquisition 两个键，"
            f"本论文固定为 data_source: {data_source}、acquisition: {acquisition}{data_note}；"
            "同时每个 experiment.runs[] 条目也必须带 data_source 和 acquisition（值同上）。"
            "这两个键由 Step7 确定性注入兜底，LLM 写错也会被覆盖，但请保持一致，勿省略。"
        )

    def _data_module_spec_non_external(self) -> dict:
        """非外部数据集（synthetic/environment/none）的 data 模块 spec。

        保留 get_dataloaders(version=...) 接口与 <DatasetClass>(...) 类，使
        training/evaluation/main 的导入与调用无需改动；仅改变数据来源实现：
        不读本地图片路径、不写 PIL 加载。
        """
        dataset_class = self._dataset_class_name()
        data_source = self._data_dependency_value("data_source")
        source_desc = {
            "synthetic": (
                "数据在代码内生成/合成（随机张量、程序化生成、从分布采样等），"
                "绝不读本地文件路径或 os.listdir。"
            ),
            "environment": (
                "数据来自交互式环境/仿真器（如 gym/Atari/MuJoCo），"
                "<DatasetClass> 包装环境，__getitem__ 返回 (observation, label/action)。"
                "不读本地图片路径。"
            ),
            "none": (
                "无数据依赖（纯算法/理论）。<DatasetClass> 提供空实现或占位："
                "get_dataloaders 返回空 dataloader（或直接让训练逻辑跳过），不读任何本地路径。"
            ),
        }.get(data_source, "数据在代码内生成，不读本地文件路径。")

        return {
            "purpose": f"实现 {data_source} 数据来源（无外部数据集）",
            "classes": [dataset_class],
            "key_methods": ["__init__", "__len__", "__getitem__", "get_dataloaders"],
            "depends_on": ["config/config.yaml"],
            "notes": (
                f"【数据来源】data_source={data_source}（非 external_dataset）。{source_desc}"
                f"实现 <{dataset_class}>(...) 与 get_dataloaders(version=...) 函数，"
                f"保持与 training/evaluation/main 的调用接口一致（version 参数可忽略）。"
                "不要引入 opencv、不要 open() 读数据集文件。"
                "所有 open() 读 config.yaml 必须 encoding='utf-8'。"
            ),
        }

    # ============================================================
    # 模型类型分支（CNN → torchvision/transfer；非 CNN → 从头训练）
    # ============================================================

    def _is_torchvision_cnn(self) -> bool:
        """是否所有模型均为 CNN 类型（需继承 torchvision + transfer 预训练权重）。

        非 CNN（MLP/RNN/VLM/other，或纯理论/RL 论文）返回 False，
        避免把 PlantVillage/ImageNet 的 torchvision 迁移学习模板泄漏进去。
        """
        models = self.step5_yamls.get("model_design", {}).get("models", [])
        if not models:
            return False
        return all(str(m.get("type", "")).upper() == "CNN" for m in models)

    def _models_module_note(self, cls: str, is_cnn: bool) -> str:
        """models 模块 spec 的 notes，按模型类型分支。"""
        base = (
            "num_classes、dropout_ratio、pretrained 等参数一律从 config.yaml 读取，不要硬编码。"
            "【形状参数 null 兜底】若 config 里的 num_classes/dropout_ratio 等形状参数是 "
            "null（论文未给具体类别数/词汇表大小），严禁把 None 传给 nn.Linear / nn.Embedding "
            "等任何 torch 形状参数（会 empty(): argument 'size' 崩溃）；必须用从论文描述推导的"
            "具体默认值兜底（如「词汇表 = 数字位数 + 特殊符号」→ 一个具体 int），"
            "并在代码注释里写明该默认值来源。"
            "模型类只定义结构（随机初始化），不要写 training_mode/transfer 分支、"
            "不要在 __init__ 里 load_state_dict；权重初始化统一在 "
            "training/train.py 的 initialize_model 里处理。"
            "所有 open() 读 config.yaml 必须 encoding='utf-8'。"
        )
        if is_cnn:
            return (
                f"定义 {cls}（scratch 与 transfer 共用同一架构，二者只有权重初始化不同）。"
                f"直接继承 torchvision.models 里对应的模型类，不要手写自定义结构"
                f"（自定义结构与 torchvision 不一致会导致 transfer 预训练权重无法 1:1 加载、"
                f"形状冲突或静默丢失），也不要往模型里加 torchvision 默认结构没有的层。"
                + base
            )
        return (
            f"定义 {cls}（随机初始化结构，无迁移学习/预训练权重）。"
            f"按论文模型结构实现前向传播，不要引入 torchvision 继承或 transfer 分支。"
            + base
        )

    def _training_module_note(self, is_cnn: bool) -> str:
        """training 模块 spec 的 notes，按模型类型分支。"""
        notes = [
            "从本地 models 包导入模型类（类名见相关代码）；保存最终模型。",
            "所有 open() 读 config.yaml 必须 encoding='utf-8'。",
        ]
        if is_cnn:
            notes.append(
                "【transfer 预训练权重】scratch 与 transfer 必须用同一个模型类（本地 models 包的 torchvision 子类），"
                "只有权重初始化不同。transfer 实现：先用本地类实例化，"
                "再取 torchvision 预训练模型的 state_dict，pop 掉类别相关层后 load_state_dict(strict=False)，"
                "其余层 1:1 加载（子类与 torchvision 结构完全一致）。"
                "具体需要 pop 的类别相关层名见 training_config.yaml 的 training_modes.transfer.reset_layers 字段。"
                "严禁对结构不一致的自定义模型 load_state_dict(strict=False)（会静默丢权重）。"
            )
        else:
            notes.append(
                "【权重初始化】从头训练：initialize_model 里用模型类随机初始化，"
                "不要加载任何预训练权重、不要引入 torchvision 或 transfer 逻辑。"
            )
        notes.append(
            "【reproduction_mode】读 config['reproduction_mode']："
            "'engineering'=train→val→best→test：从训练集二次划分 val（config.split_strategy.val_split_ratio，80/20），"
            "每 epoch 验证、按 val_accuracy 选最优 checkpoint；"
            "'strict'=train→最终→test：不划分 val、不验证、只保存最终模型，train_model 里 val_loader 可为 None 且必须能跳过验证分支。"
            "train_model 的 validate 调用必须用 if val_loader is not None 守卫，不能无条件调用。"
            "【train/test 语义】论文仅 train/test 二路划分、无独立验证集，val 仅在 engineering 模式从训练集二次划分；"
            "test 集只用于最终评估，绝不能进入 train_loader 或 val_loader。"
            "【version 数据流】从 experiment.runs[].dataset_version 读取版本，"
            "调用 get_dataloaders 时显式传 version=dataset_version。"
        )
        return "".join(notes)

    # ============================================================
    # 辅助
    # ============================================================

    @staticmethod
    def _clip(text: str, limit: int = 160) -> str:
        """截断长文本，避免 YAML 过长。"""
        text = str(text)
        return text if len(text) <= limit else text[:limit] + "…"

    def _check_prerequisites(self):
        missing = []
        if not self.step4_yamls:
            missing.append("step4/ (无 YAML 文件)")
        if not self.step5_yamls:
            missing.append("step5/ (无 YAML 文件)")
        if missing:
            logger.warning(f"  缺少前置文件: {missing}，请先运行 Step4 和 Step5")

    @staticmethod
    def _load_all_yaml(directory: Path) -> dict:
        """加载目录下所有 YAML 文件，返回 {name: data}。"""
        yamls = {}
        if not directory.exists():
            return yamls
        for yf in sorted(directory.glob("*.yaml")):
            try:
                with open(yf, "r", encoding="utf-8") as f:
                    data = yaml.safe_load(f)
                    if data:
                        yamls[yf.stem] = data
            except Exception as e:
                logger.warning(f"  无法加载 {yf}: {e}")
        return yamls

    def save_yaml(self, data: dict, filename: str = "project_brief.yaml") -> str:
        """保存 YAML。"""
        path = self.output_dir / filename
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(data, f, allow_unicode=True, default_flow_style=False,
                      sort_keys=False, indent=2)
        logger.info(f"  -> {path}")
        return str(path)
