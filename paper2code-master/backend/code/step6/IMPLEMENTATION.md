# Step 6：项目整合与代码生成准备

## 思路

- 目标：把 Step4（需求分析）与 Step5（工程设计）两份多文件 YAML 横向对齐，产出代码生成 Agent 可直接消费的单一 `project_brief.yaml`。核心是**确定性地**完成「需求 ↔ 设计」的一致性核查，并把两侧信息规整为逐文件的代码生成计划、追溯矩阵与关键决策。本 Step 不调用 LLM，纯规则驱动，避免引入事实偏差（LLM 只在 Step4/5 的语义提取阶段介入）。

- 输入 / 输出：
  - 输入：
    - Step4：`step4/{functional_requirements, data_flow, gap_analysis, acceptance_criteria, requirements_summary}.yaml`
    - Step5：`step5/{dataset, model_design, training_config, experiment_design, project_structure, technology, reproduction_goal, project_analysis, optimized_project_spec}.yaml`
  - 输出：`step6/project_brief.yaml`

- 核心流程（一致性检查 + 项目整合两大块）：
  1. 加载 step4/ 与 step5/ 目录下全部 `*.yaml`，得到 `{文件名: dict}` 两个字典（`_load_all_yaml`）。
  2. 实例化 `ConsistencyChecker`，运行 6 组规则检查（数值、训练模式、FR→模块覆盖、数据划分、评估指标、接口闭环），并聚合两侧已有的 gaps/assumptions，得到 `consistency_report`。
  3. `ProjectBriefIntegrator.integrate()` 按顺序组装各 section：project → consistency_report → requirements_digest → design_digest → decisions → data_dependency → unresolved_dataset_versions → traceability → code_generation_plan → risks_and_notes，最后补 `_meta`。
  4. `save_yaml` 落盘 `project_brief.yaml`。

- 关键设计取舍与假设：
  - **纯规则、确定性搬运与映射**：整合阶段不重写需求/设计，只抽取、截断（`_clip` 160 字符）与映射；结论类信息（如“某模型优于另一模型”）只在 notes 中提示代码 Agent“从实际结果动态计算”，绝不下结论。
  - **一致性结论可回溯**：每条 check/conflict/gap/assumption 都带来源（Step4 哪个 FR / Step5 哪个字段），并带 `resolution`/`mitigation` 供代码 Agent 决策。
  - **模板分支用显式判定隔离**：模型类型（CNN vs 非 CNN）与数据来源（external_dataset / synthetic / environment / none）各有一个判定函数，试图避免把 PlantVillage/ImageNet 的图像+迁移学习模板泄漏进非图像/理论/RL 论文。
  - **工程默认值集中记录在 decisions**：80/20 二次划分验证集（`val_split_ratio=0.2`）、`best_metric=val_accuracy`、`framework`、`training_modes` 等作为「跨来源确认过的关键决策」集中落盘，供 Step7 Validator 校验 config.yaml 一致性。
  - **数据依赖（data_source/acquisition）透传给 Step7 确定性注入**：Step6 判定后写入 `data_dependency` 与 `decisions`，Step7 据此把这两个键写入 config.yaml 顶层与每个 `experiment.runs[]`，防止 LLM 脑补。

## 具体实现

### 文件清单

- `__init__.py` — 模块说明文档（流程、核心产出、特点），无实际逻辑。
- `step6_consistency.py` — 一致性检查器 `ConsistencyChecker`：规则驱动的 Step4↔Step5 交叉校验，产出 checks/conflicts/gaps/assumptions。
- `step6_summarizer.py` — 项目整合器 `ProjectBriefIntegrator`：加载 YAML、调用一致性检查、组装 project_brief 的 10 个 section 并保存。
- `step6_main.py` — 总入口：CLI（`--paper` / `--all`），对每篇论文跑整合并打印统计。

### 关键类 / 函数

`ConsistencyChecker`（step6_consistency.py）
- `check()` — 主入口，运行 6 组检查 + 聚合 gaps/assumptions，返回 `consistency_report` 字典。
- `_check_numerical()` — CHECK-01：比对「数据集规模/类别数/图像尺寸/学习率/训练轮数/批大小」两侧数值（数字交集判等）。
- `_check_training_modes()` — CHECK-02：scratch/transfer 两种训练模式是否两侧贯穿一致。
- `_check_fr_module_coverage()` — CHECK-03：每个 FR 按关键词（`FR_MODULE_MAP`）映射到期望模块，检查是否出现在 `project_structure.module_mapping`。
- `_check_split_consistency()` — CHECK-04：论文仅 train/test 无 val 而 `checkpoint.best_metric` 用 val_accuracy 时产生冲突。
- `_check_metric_consistency()` — CHECK-05：Step4 提到的 f1/precision/recall/accuracy 是否在 Step5 evaluation 中体现。
- `_check_interface_consistency()` — CHECK-06：数据流接口端点（`stage_to_file`）是否都有对应实现文件。
- `_collect_gaps()` / `_collect_assumptions()` — 聚合两侧已有的 `gap_analysis.gaps`、`parameter_provenance.NOT_FOUND_in_paper`、`resolved_assumptions`、框架迁移、`engineering_default`。
- `_fr_field(field)` — 返回「按 FR 名称关键词取某字段」的闭包（数值/文本比对的数据源）。
- `_extract_nums(text)` / `_check(...)` / `_next_id(prefix)` — 数值提取、构造检查项、生成 `CHECK-xx/CONFLICT-xx/GAP-xx/ASM-xx` 连续 ID。

`ProjectBriefIntegrator`（step6_summarizer.py）
- `__init__(paper_name)` — 定位 paper 目录、加载 step4/5 全部 YAML、初始化 `_traceability` 缓存。
- `integrate()` — 主流程：`_check_prerequisites` → 一致性检查 → 逐 section 组装 brief → 补 `_meta`。
- `_build_project()` / `_build_requirements_digest()` / `_build_design_digest()` — 项目元信息 / 需求摘要 / 设计摘要。
- `_build_decisions()` — 集中记录跨来源关键决策（val_split_ratio、framework、training_modes、best_metric、data_source、acquisition）。
- `_build_data_dependency()` — 透传并校验 `data_source`/`acquisition`（非法值回退 `external_dataset`/`local_path`）；leaf_based_split 时登记 `leaf_mapping`（`required`/`file`/`constraint`/`on_missing`，缺映射文件必须显式告警并记 `leaf_constraint_effective=false`，禁止静默回退随机划分），供 Step7 注入 config。
- `_build_unresolved_dataset_versions()` — 检测 `segmentation_algorithm.implementation_status == NOT_FOUND` 的版本供 Step7 标记 unresolved；`PARTIAL` 版本不整版 unresolved，reason 改为「方法已知、脚本参数未公开，按描述实现或文档化兜底，禁止悄悄替换成其他版本」。
- `_build_traceability()` / `_map_fr_to_impl()` — FR → 实现文件::符号 的规则映射，并缓存到 `self._traceability`。
- `_build_code_generation_plan()` / `_build_generation_order()` / `_build_file_spec()` — 有序的文件生成计划与逐文件 spec（purpose/classes/key_methods/depends_on/satisfies_requirements/notes）。
- `_fr_satisfied_by(file)` / `_all_fr_ids()` — 反查某文件满足哪些 FR / 全部 FR id（main.py 用）。
- `_build_risks_and_notes()` / `_build_consistency_ref()` / `_for_code_agent_notes()` — 风险与给代码 Agent 的注意事项；leaf_based_split 时 `for_code_agent` 追加 `[leaf_mapping]` 约束（分组划分、缺文件须告警并记 `leaf_constraint_effective=false`）。
- 动态数据与类型分支：`_dataset_class_name()`、`_dataset_versions()`、`_default_version()`、`_is_leaf_split()`、`_eval_metrics()`、`_data_dependency_value()`、`_is_torchvision_cnn()`、`_models_module_note()`、`_training_module_note()`、`_data_module_spec_non_external()`、`_config_data_dependency_note()`。
- 辅助：`_sanitize_filename()`、`_clip()`、`_load_all_yaml()`、`save_yaml()`。

### 数据流与相邻 Step 衔接

`project_brief.yaml` 顶层结构（`integrate()` 中字典顺序即写出顺序）：

- `project` — name / paper_title / paper_type / task / one_liner。
- `consistency_report` — summary（checks_run/passed/warning/failed/conflicts/gaps/assumptions）、checks、conflicts、gaps、assumptions。
- `requirements_digest` — summary、core_requirements（id/what/why/acceptance）、data_flow_overview、key_interfaces。
- `design_digest` — summary、tech_stack（framework/key_dependencies/hardware）、models（name/type/file/key_params）、project_structure、training_config_summary。
- `decisions` — val_split_ratio / val_split_note / framework / training_modes / best_metric / data_source / acquisition。
- `data_dependency` — data_source / acquisition / note（note 原样透传 Step5 的 `data_source_note`）/ leaf_mapping（仅 leaf_based_split：required/file/constraint/on_missing）。
- `unresolved_dataset_versions` — 数据来源/处理方法 NOT_FOUND 的版本列表（name/reason）。
- `traceability` — 每条：requirement_id / requirement_name / planned_implementation（文件::符号列表）/ design_ref（来源 Step5 文件名）。
- `code_generation_plan` — generation_order（分 4 个 phase 的有序文件列表）+ per_file_specs（逐文件 spec）。
- `risks_and_notes` — gaps / conflicts / assumptions / for_code_agent。
- `_meta` — paper_name / generated_at / method / sources。

Step7 依赖哪些字段（结合 step7/ 源码）：

- `code_generation_plan.per_file_specs`：`project_initializer.py` 据此创建骨架文件；`code_context_builder.py` 以 `spec["file"]` 为键建索引，并把整个 spec（含 `purpose`/`classes`/`key_methods`/`depends_on`/`satisfies_requirements`/`notes`）注入单文件生成上下文，`notes` 是给代码 Agent 的核心约束文本。
- `code_generation_plan.generation_order`：`code_context_builder` 建 phase 映射，决定生成顺序。
- `decisions`：`code_validator._check_decisions` 逐一校验 config.yaml 的 `val_split_ratio`、`framework`、`training_modes`（keys 集合）、`checkpoint.best_metric`、顶层 `data_source`、顶层 `acquisition` 与 Step6 决策一致。
- `data_dependency.data_source / acquisition`：`step7_main` 的确定性注入逻辑（顶层 + 每个 `experiment.runs[]`）直接读取这两个字段兜底。`data_dependency.leaf_mapping` 同样由注入逻辑写入 `split_strategy.leaf_mapping_file/constraint/on_missing`。
- `traceability`：`code_validator._check_traceability` 按 FR 分组校验生成代码中是否存在对应符号。
- `risks_and_notes.for_code_agent`：作为 `agent_notes` 注入每个文件的生成上下文。
- `design_digest.tech_stack / models / training_config_summary`、`requirements_digest.core_requirements`：供生成上下文补充技术栈、模型、训练配置、相关需求信息。
