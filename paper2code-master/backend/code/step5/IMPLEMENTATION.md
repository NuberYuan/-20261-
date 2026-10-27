# Step 5：工程设计

> 把 Step3 的论文自然语言信息（`key_information.json`）翻译成 Step6/7 代码 Agent 可消费的结构化工程规格（多份 YAML）。
> 本文档是 `STEP5_IMPLEMENTATION.md` 的补充，聚焦"实现细节与数据流"。

## 思路

### 目标

- 从论文信息中提取可执行、可验证、**不编造**的工程规格，作为 Step6（生成代码 spec）与 Step7（生成代码）的唯一事实来源。
- 明确论文类型、任务类型、数据来源、模型类型、训练模式、实验矩阵与项目文件结构。

### 输入 / 输出

**输入（只读，来自 Step1-3）：**

- `data/output/{paper}/step3/key_information.json` —— 论文结构化关键信息（title、methodology、dataset、implementation_details、experiment、reproduction_requirements 等）。
- `data/output/{paper}/step2/structure_detect_paper.json` —— 论文原始段落文本（用于 LLM 上下文与 Layer3 词汇验证）。

**输出（全部写入 `data/output/{paper}/step5/`）：**

| 文件 | 子步 | 受众 |
|------|------|------|
| `reproduction_goal.yaml` | 5.1 | 人+Agent |
| `technology.yaml` | 5.2 | 人+Agent |
| `dataset.yaml` | 5.3 | 人+Agent |
| `model_design.yaml` | 5.4 | Agent |
| `experiment_design.yaml` | 5.5 | Agent |
| `project_structure.yaml` | 5.6 | Agent |
| `training_config.yaml` | 5.7 | Agent |
| `project_analysis.yaml` | 汇总 | 人 |
| `optimized_project_spec.yaml` | 5.8 | Agent |

### 核心流程（8 个子步）

| 子步 | 类 | 职责 | 依赖 |
|------|-----|------|------|
| 5.1 | `GoalAnalyzer` | 论文类型（`paper_type`）、任务类型（`task`）、是否可训练（`is_trainable`）、复现目标 | 仅 key_info |
| 5.2 | `TechAnalyzer` | 框架推荐（固定 PyTorch）、模型列表、优化器/损失/调度器、预训练权重 | 仅 key_info |
| 5.3 | `DatasetAnalyzer` | 数据集名/来源/规模/类别、`data_source`/`acquisition`、版本、预处理、划分策略 | 仅 key_info（`data_source` 另需 LLM） |
| 5.4 | `ModelAnalyzer` | 每个模型的 class 名、类型（CNN/RNN/VLM/MLP/other）、输入/输出、结构解析 | 仅 key_info |
| 5.5 | `ExperimentAnalyzer` | 实验 pipeline、主指标、baseline/ablation、实验矩阵 | 仅 key_info |
| 5.6 | `StructureAnalyzer` | 项目目录树与文件列表（按 paper_type 选模板、按模型名生成文件名） | **读 5.1/5.4 的 yaml 文件** |
| 5.7 | `TrainConfigAnalyzer` | 超参数（batch/epochs/lr/optimizer/scheduler/loss）、`training_modes`（scratch/transfer）、checkpoint | 仅 key_info |
| 5.8 | `SpecAnalyzer` | 汇总 5.1~5.7，用 LLM 细化每个 `.py` 的 class/function/依赖/伪代码 | 前 7 步全部 yaml |

- **并行/串行**：`step5_main.py` 用 `ThreadPoolExecutor(max_workers=4)` 并行跑 5.1~5.7，之后串行跑 5.8。
  - 设计假设是"5.1~5.7 彼此独立、都只读 key_information.json"。**但 5.6 实际通过 `load_yaml` 读取 5.1 和 5.4 的产物文件**，与并行假设矛盾（见"关键设计取舍"与 bug 清单）。

- **三层防编造机制**（`BaseAnalyzer.analyze`）：
  1. **Layer1 规则提取**：`extract_by_rules()` 纯确定性代码，从 key_info 直接映射，不调 LLM。
  2. **Layer2 约束型 LLM 补全**：`_find_unknown()` 找出 `NOT_FOUND/""/None` 字段 → `_call_llm()` 用严格 prompt 补全 → `_merge()` 只填空值（规则优先）。
  3. **Layer3 后验证**：`_validate()` 审计关键字段是否存在 + 词汇验证（LLM 输出的概念词是否在原文中出现，防编造）。

- **data_source / acquisition 判定**（`step5_3_data.py`）：
  - `data_source` 合法值：`external_dataset` / `synthetic` / `environment` / `none`。
  - `acquisition` 合法值：`local_path` / `auto` / `n/a`（非 external_dataset 时强制 `n/a`）。
  - 规则层先给保守默认 `external_dataset + local_path`，`analyze()` 里再用一次专门 LLM 判定，最后 `_coerce_*` 做枚举校验（非法/缺失回退 `external_dataset`/`local_path`）。

- **图像/非图像分支**（`BaseAnalyzer._is_image_data`）：用信号词判断数据集是否图像类，决定是否套用 color/grayscale/segmented 图像预处理模板与迁移学习模板；非图像数据走单一 `default` 版本。

### 关键设计取舍与假设

- **框架固定 PyTorch**：`technology.framework.recommended` 恒为 `PyTorch`，论文原文框架作为事实记在 `programming_framework_mentioned`（避免 Step6/7 框架冲突）。
- **保守默认 fail-safe 方向**：`data_source` 判定失败回退 `external_dataset + local_path`，宁可多查数据集、不漏检（但代价是对 synthetic/none 论文可能误判，见 bug 清单）。
- **transfer 门控 = `pretrained_model` 是否存在**：5.5/5.7 用 `model_architecture.pretrained_model != NOT_FOUND` 决定是否生成 transfer 模式。
- **5.8 是唯一"规则结论 + LLM 细化"混合产物**：规则部分（project/modules/dependencies/flow）确定性生成，LLM 只补 `code_requirements`。
- **5.6 的文件间依赖被当作"独立"处理**：这是已知风险点，详见 bug 清单。

## 具体实现

### 文件清单

- `__init__.py` — 包说明（仅一行 docstring）。
- `base_analyzer.py` — `BaseAnalyzer` 基类：三层防编造主流程、key_info/structure 加载、`NOT_FOUND` 约定、YAML 读写、图像信号词判定 `_is_image_data`。
- `step5_1_goal.py` — `GoalAnalyzer`：论文类型/任务类型/可训练性/复现目标 → `reproduction_goal.yaml`。
- `step5_2_tech.py` — `TechAnalyzer`：框架、模型列表、优化器/损失/调度器、预训练 → `technology.yaml`。
- `step5_3_data.py` — `DatasetAnalyzer`：数据集、`data_source`/`acquisition`、版本、预处理、划分 → `dataset.yaml`。
- `step5_4_model.py` — `ModelAnalyzer`：模型列表与类型推断、架构结构化解析 → `model_design.yaml`。
- `step5_5_exp.py` — `ExperimentAnalyzer`：pipeline、指标、baseline/ablation、实验矩阵 → `experiment_design.yaml`。
- `step5_6_struct.py` — `StructureAnalyzer`：目录/文件模板 → `project_structure.yaml`。
- `step5_7_train.py` — `TrainConfigAnalyzer`：超参数、训练模式、checkpoint → `training_config.yaml`。
- `step5_8_spec.py` — `SpecAnalyzer`：汇总 + LLM 细化 `code_requirements` → `optimized_project_spec.yaml`。
- `step5_main.py` — 总入口：并行调度 5.1~5.7、聚合 `project_analysis.yaml`、串行 5.8。

### 关键类 / 函数

**`BaseAnalyzer`（base_analyzer.py）**

- `analyze(skip_llm=False)` — 三层主流程：规则 → LLM 补全 → 后验证 → 附 `_meta`。
- `extract_by_rules()` — Layer1 规则提取（子类实现）。
- `_call_llm(unknown_fields, current)` — Layer2：严格约束 prompt，仅补空值。
- `_find_unknown(data, prefix)` — 递归定位 `NOT_FOUND/""/None` 字段路径。
- `_merge(base, update)` — 规则优先的递归合并（LLM 只补空值）。
- `_validate(result)` — Layer3：关键字段审计日志 + 词汇验证（防编造）。
- `save_yaml(data, filename)` / `load_yaml(filename)` — YAML 写/读（写时剔除 `_` 前缀键）。
- `_is_image_data(ds)` — 图像数据判定（信号词匹配），驱动图像/非图像分支。
- `_get_nested` / `_safe_str` / `_pick` / `_extract_num_classes` — 通用取值工具。

**`GoalAnalyzer`（step5_1_goal.py）**

- `extract_by_rules()` — 产出 `paper_name`/`title`/`research_problem`/`paper_type`/`task`/`input`/`output`/`num_classes`/`reproduction_targets`/`is_trainable`。
- `_classify(ki)` — 论文类型：code 关键词 → `empirical_comparison`；有新模型+有训练 → `model_proposing`；否则 `empirical_comparison`。
- `_task_type(ki)` — 任务类型：image/vision/crop → `image_classification`；LSTM/Transformer + 时序词 → `time_series_prediction`；code+defect/bug → `code_analysis`；genomic/snp → `genomic_prediction`；否则 `other`。
- `_has_training(impl)` — 依据 `training_strategy` 与 optimizer/lr/epoch/loss 是否有值判断可训练性。
- `_targets(ptype, trainable)` — 生成复现目标列表。

**`TechAnalyzer`（step5_2_tech.py）**

- `extract_by_rules()` — `framework`（固定 PyTorch）、`model_backbone.names`、`architecture_details`、`training`（optimizer/loss/scheduler）、`pretrained_weights`、`special_modules`、`hardware`。
- `_extract_model_names(meth, arch)` — 从 `methodology.model` → `model_architecture` 子键 → `method_name`（fallback）提取模型名。
- `_infer_scheduler(impl)` — 从 `important_parameters.scheduler` 或 `training_strategy` 关键词推断调度器。

**`DatasetAnalyzer`（step5_3_data.py）**

- `extract_by_rules()` — 保守默认 `external_dataset`/`local_path` + datasets + versions + preprocessing + split_strategy。
- `analyze(skip_llm)` — 覆写：在基类三层后额外 LLM 判定 data_source/acquisition，再枚举校验。
- `_determine_data_source(result)` — 专用 LLM 判定数据来源与获取方式，失败回退默认。
- `_coerce_data_source(value)` / `_coerce_acquisition(value, data_source)` — 枚举校验与回退。
- `_determine_split_method(ds)` — 依据 `same leaf`/`stratified` 关键词推断 `leaf_based_split`/`stratified_split`/`random_split`。
- `_build_per_version_preprocessing(ds, versions)` — 图像数据生成 color/grayscale/segmented 三套预处理；非图像走 `_build_generic_preprocessing`。
- `_extract_versions(ds)` — 提取版本列表；非图像返回 `["default"]`。
- `_extract_resize(ds)` — 从 preprocessing 提取图像尺寸。

**`ModelAnalyzer`（step5_4_model.py）**

- `extract_by_rules()` — 从 `model_architecture` 子键 / `methodology.model` / `method_name` 提取模型列表，产出 `models[]`。
- `_extract_arch_models(arch, ds)` — 从已知架构子键（alexnet/resnet/vgg/lstm/transformer/...）提取结构化模型条目。
- `_build_model_entry(name, ...)` — 构造单个模型条目（type/class 名/input/output/num_classes/pretrained/architecture）。
- `_infer_type(text)` — 类型推断：VLM > RNN > CNN > MLP > other。
- `_structure_architecture(detail)` — 从文本解析层数/参数量/最终分类层/特殊组件。
- `_parse_special_components(detail)` — 识别 Inception/辅助分类器/GAP/归一化层等结构特征。
- `_sanitize_class_name(name)` — 生成合法 PascalCase 类名。

**`ExperimentAnalyzer`（step5_5_exp.py）**

- `extract_by_rules()` — pipeline/experiments/evaluation/comparison/experiment_matrix/important_results。
- `_extract_pipeline(exp, meth)` — 技术 pipeline 提取（technical_pipeline → methodology.pipeline → 默认模板）。
- `_extract_primary_metric(exp)` — 从 evaluation_metrics 匹配主指标（f1/accuracy/r2/rmse/...）。
- `_extract_ablation_factors(exp)` — 从 ablation_study 文本结构化提取消融因子。
- `_build_experiment_matrix()` — 展开 Architectures × TrainingModes × DataTypes × Splits 组合，计算 `total_configurations`/`run_all`/`recommended_subset`/`known_best`/`known_worst`/`known_real_world`（由 `_extract_known_extremes` 从 comparison 句子级提取：Best/Worst/real-world 各取本句主指标，配置名含 `-` 不截断，括号附注不混入）。
- `_parse_list(text)` — 逗号分隔列表解析。

**`StructureAnalyzer`（step5_6_struct.py）**

- `extract_by_rules()` — 读 `reproduction_goal.yaml`（取 paper_type）与 `model_design.yaml`（取模型文件名），套用模板生成目录树。
- `_sanitize_filename(name)` — PascalCase → snake_case 文件名。

**`TrainConfigAnalyzer`（step5_7_train.py）**

- `extract_by_rules()` — 超参数、`parameter_provenance`（参数来源追踪）、`training_modes`、`checkpoint`、`train_test_splits`、`inference`、`reproducibility_notes`。
- `_build_training_modes(comparison, has_pretrained)` — scratch 恒在；transfer 仅在 `has_pretrained` 时生成（含 reset_layers/freeze_backbone）。
- `_infer_reset_layers()` — 按模型名推断迁移学习需重置的分类层。
- `_find_param(params, key)` — 在 important_parameters 中模糊匹配参数。
- `_clean_step_size(raw)` — 提取/求值 step_size（含 "30/3" 表达式）。
- `_extract_image_size(params, impl)` / `_extract_dropout(impl, params)` / `_extract_splits(params)` / `_extract_difficulties(repro)` — 各参数提取。

**`SpecAnalyzer`（step5_8_spec.py）**

- `__init__` — 预加载 7 份 yaml。
- `extract_by_rules()` — 确定性汇总：project/architecture.modules/dependencies.files/execution_flow/reproducibility_notes。
- `analyze()` — 覆写：规则汇总后调 LLM 细化 `code_requirements`（每个 `.py` 的 class/function/伪代码）。
- `_format_training_config_for_prompt()` — 提取训练参数供 LLM 严格照抄。

**`step5_main.py`**

- `_run_one(...)` — 单子步执行 + 产物级断点续跑（已存在且非 `--force` 则复用）。
- `run_paper(...)` — 并行跑 5.1~5.7 → 聚合 `project_analysis.yaml` → 串行 5.8。
- `main()` — 参数解析（`--paper`/`--all`/`--skip-llm`/`--force`），逐论文容错执行。

### 数据流与相邻 Step 衔接

#### 各 YAML 顶层结构（关键字段）

- `reproduction_goal.yaml`：`paper_name` / `paper_type`（model_proposing|empirical_comparison）/ `task` / `input` / `output` / `num_classes` / `reproduction_targets` / `is_trainable`。
- `technology.yaml`：`framework`（recommended/candidates/source）/ `programming_framework_mentioned` / `model_backbone.names` / `architecture_details` / `training`（optimizer/loss/scheduler）/ `pretrained_weights` / `special_modules` / `hardware`。
- `dataset.yaml`：`data_source` / `acquisition` / `data_source_note` / `datasets[]`（name/source/size/num_classes/data_type）/ `dataset_versions[]` / `preprocessing`（versions + version_selector）/ `split_strategy`（method/ratios/leaf_mapping_constraint）。
- `model_design.yaml`：`models[]`（**name / type**（CNN|RNN|VLM|MLP|other）/ code_class_name / input / output / num_classes / pretrained / architecture_summary / architecture）。
- `experiment_design.yaml`：`pipeline` / `experiments[]` / `evaluation`（primary_metric/secondary_metrics）/ `comparison`（baselines/ablation_factors）/ `experiment_matrix`（dimensions/total_configurations/known_best/known_worst/run_all/recommended_subset）/ `important_results`。
- `project_structure.yaml`：`project_name` / `root_structure`（directories/files）/ `module_mapping`。
- `training_config.yaml`：`parameter_provenance` / **`training_modes`**（scratch/transfer，transfer 含 reset_layers/freeze_backbone）/ `training`（batch_size/epochs/image_size/optimizer/scheduler/loss_function/dropout_ratio/early_stopping）/ `checkpoint` / `train_test_splits` / `inference` / `reproducibility_notes`。
- `optimized_project_spec.yaml`：`project` / `architecture.modules`（含 `models.classes`）/ `dependencies`（files/external）/ `execution_flow` / `reproducibility_notes` / `code_requirements`（LLM 细化）。

#### Step6 依赖的字段（step6/step6_summarizer.py）

- `dataset.data_source` / `dataset.acquisition` / `dataset.data_source_note` —— 决定 data 模块走外部数据集加载还是代码内生成/环境/无数据（`_data_dependency_value`、`_data_module_spec_non_external`）。
- `dataset.dataset_versions` —— config 的 `data.versions` 列表（`_dataset_versions`）。
- `dataset.preprocessing.versions` 中 `segmentation_algorithm` 由 `_extract_segmentation_algorithm` 解析：方法已描述（color space/分量/color casts/mask 信号）但参数未公开时标 `PARTIAL`（保留 method/color_spaces/components，参数仍 NOT_FOUND），完全未描述才 `NOT_FOUND`；`NOT_FOUND` 识别 unresolved 版本（`_build_unresolved_dataset_versions`），`PARTIAL` 走「按方法描述实现或文档化兜底」而非整版缺失。
- `dataset.split_strategy.method` —— 判断是否 leaf 分组划分（`_is_leaf_split`）。
- `model_design.models[].type` —— `_is_torchvision_cnn()`：全部为 `CNN` 才走 torchvision 继承 + transfer 模板。
- `model_design.models[].num_classes` / `.pretrained` / `.code_class_name` / `.name` —— 模型上下文与文件映射。
- `training_config.training_modes`（键集合）—— Step6 decisions 的 `training_modes`（`_build_decisions`）。
- `reproduction_goal.paper_type` / `num_classes` —— 论文类型与类别数。

#### Step7 依赖的字段（step7/code_context_builder.py、code_generator.py、code_validator.py、step7_main.py）

- `model_design.models[].num_classes` / `.pretrained` / `.name` / `.type` / `.architecture_summary` / `.architecture`（`_model_detail`）。
- `dataset.dataset_versions` / `.preprocessing`（含 `version_selector`）/ `.split_strategy`（`_dataset_detail`/`_config_detail`）。
- `training_config.training_modes` / `.training` / `.checkpoint` / `.train_test_splits` / `.inference`（`_training_detail`/`_config_detail`）。
- `code_generator.py` 直接读取键：`num_classes`/`pretrained`/`preprocessing`/`split_strategy`/`dataset_versions`/`train_test_splits`/`training`/`training_modes`/`checkpoint`/`inference`。
- `code_validator.py` 校验：`split_strategy.val_split_ratio`、`training_modes` 键集合、顶层 `data_source`/`acquisition`（与 Step6 decisions 一致）、`unresolved_dataset_versions`。
- `step7_main.py` 把 Step6 判定的 `data_source`/`acquisition` 确定性注入 config 顶层与每个 `experiment.runs[]`（`_inject_data_dependency`，不让 LLM 猜数据依赖）。
