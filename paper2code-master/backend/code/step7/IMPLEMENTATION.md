# Step 7：代码生成

## 思路

- 目标：把 Step6 产出的 `project_brief.yaml`（项目结构 + 逐文件代码生成计划）落成一套可运行的 Python 项目骨架与源码，并对 LLM 生成的代码做规则化检查与自动修复，最终给出可复现入口与生成报告，交给 Step8 去实际执行实验。
- 输入 / 输出：
  - 输入：`step6/project_brief.yaml`（含 `code_generation_plan.per_file_specs`、`generation_order`、`design_digest`、`requirements_digest`、`traceability`、`decisions`、`data_dependency`、`unresolved_dataset_versions`、`risks_and_notes` 等）；可选回读 `step5/*.yaml`（`model_design.yaml` / `dataset.yaml` / `training_config.yaml`）补全架构细节。
  - 输出：`step7/generated_project/`（`config/config.yaml`、`data/`、`models/`、`training/`、`evaluation/`、`utils/`、`main.py`、`requirements.txt`、`README.md` 等）+ `step7/generation_report.yaml` + `step7/generation_log.json`。
- 核心流程（项目初始化 → 上下文构建 → LLM 逐文件生成 → 校验 → 修复循环）：
  1. `ProjectInitializer` 依据 `per_file_specs` 创建目录骨架与空占位文件，并按依赖关系做拓扑排序得到生成顺序（被依赖者先生成，非 Python 文件优先）。
  2. 按顺序逐文件：`CodeContextBuilder` 整理该文件的生成上下文（spec、需求、已生成依赖代码、Step5 详情、全局约束）→ `CodeGenerator` 调 DeepSeek 生成 → `CodeValidator` 规则校验。
  3. 校验失败则进入 `CodeRepairer` 修复循环（最多 `MAX_REPAIR_ATTEMPTS`），修复后重新校验。
  4. 全部文件生成后，对 `config.yaml` 做确定性注入（未定义来源的 dataset version 标记 `unresolved`、`data_source`/`acquisition` 顶层与 per-run），再跑项目级检查（文件完整性 / 第三方依赖 / 需求追踪 / Step6 决策一致性 / 未定义版本标记），最后写报告。
- 关键设计取舍与假设：
  - 校验优先用「规则」（`ast`/`yaml`/符号表/静态配置键链），不额外调 LLM 做 judge，降低开销且可复现。
  - 「不运行实验」是 Step7 的硬边界：只生成与静态检查，训练/评估留给 Step8。
  - 对论文中「未定义数据来源」的 dataset version，用「确定性注入」而非让 LLM 猜，避免 LLM 把 `segmented` 版本伪造成 `color` 兜底。
  - 假设项目是图像/CNN 风格的 PyTorch 工程：校验器与上下文构建器里硬编码了 `models/`、`data/dataset.py`、`training/train.py`、`config/config.yaml` 等路径与 `data/models/training/evaluation/utils` 五个包名。

## 具体实现

### 文件清单
- `__init__.py` — 模块职责说明与边界（Step7 生成代码、不运行实验）。
- `project_initializer.py` — 依据 Step6 计划创建目录/空文件骨架，并对文件做拓扑排序（不调 LLM）。
- `code_context_builder.py` — 把 project_brief 分散信息 + Step5 详情整理成「单文件生成上下文」。
- `code_generator.py` — 调 DeepSeek 文本模型，逐文件生成代码并剥离 markdown 代码块。
- `code_validator.py` — 规则化检查：语法/类/方法/import/配置引用/函数接口/结论写死 + 项目级检查。
- `code_repairer.py` — 校验失败后调 DeepSeek 生成一次修复结果。
- `step7_main.py` — 总控制器：编排初始化 → 生成 → 校验 → 修复 → 注入 → 项目级检查 → 报告。

### 关键类 / 函数
- `ProjectInitializer.initialize(brief)` — 创建目录与空文件，返回拓扑排序后的文件顺序。
- `ProjectInitializer._ordered_files(files, specs)` — Kahn 拓扑排序；无依赖/依赖已满足者先出队，非 `.py` 优先，稳定。
- `CodeContextBuilder.build(file, generated)` — 汇总 spec / 需求 / 已生成代码 / Step5 详情 / agent_notes，生成单个文件上下文。
- `CodeContextBuilder._step5_detail(file)` — 按文件路径回读 Step5 的 model/dataset/training/config 详情。
- `CodeGenerator.generate(context)` — 组装 prompt 调 `chat_with_text`，再用 `_extract_code` 剥离代码块。
- `CodeGenerator._output_requirements(file)` — 按 `.py`/`.yaml`/`.txt`/`.md` 给出针对性输出约束（如 py 要过 `ast.parse`、超参数不硬编码、config 键必须与 config.yaml 对齐）。
- `CodeValidator.validate(file, code, spec)` — 单文件七项检查，返回 `{status, errors, checks}`。
- `CodeValidator.register(file, code)` / `set_config(config_dict)` — 注入已生成文件的类/函数/签名符号表与 config.yaml 内容，供跨文件检查。
- `CodeValidator._check_imports(code)` — 本地模块存在性 + `from X import Y` 的符号存在性。
- `CodeValidator._check_config_refs(file, code)` — 静态解析 `config[...]`/`config.get(...)` 键链，与 config.yaml 真实键比对。
- `CodeValidator._check_function_interface(code)` — 校验 `from X import Y` 跨文件调用与同文件顶层函数调用的参数个数/关键字名/必填参数。
- `CodeValidator.validate_project(plan_files, generated, brief)` — 项目级：文件完整性 / 第三方依赖 / 函数接口汇总 / 需求追踪 / 决策一致性 / 未定义版本标记。
- `CodeRepairer.repair(file, code, errors, spec, context)` — 组装修复 prompt 并返回修复后代码。
- `Step7Runner.run()` — 主流程编排，返回 `report["generation"]["status"] == "success"`。
- `Step7Runner._generate_one_file(...)` — 单文件「生成 → 校验 → 修复循环」。
- `Step7Runner._inject_unresolved_versions(...)` / `_inject_data_dependency(...)` — 对 config.yaml 做确定性注入。

### 数据流与相邻 Step 衔接
- 生成项目的目录结构（由 Step6 `per_file_specs` 决定，典型 CNN 工程）：`config/config.yaml`、`data/dataset.py`、`models/<model>.py`、`training/train.py`、`evaluation/evaluate.py`、`utils/`、`main.py`、`requirements.txt`、`README.md`。
- `config.yaml` 顶层键（LLM 生成 + Step7 确定性注入补全）：`project`（含 `framework`）、`data_source`、`acquisition`、`experiment`（含 `runs[]`，每条带 `data_source`/`acquisition`）、`data`（按版本名，`{status: unresolved/resolved, reason}`）、`split_strategy`（含 `val_split_ratio`）、`training`、`training_modes`、`checkpoint`（含 `best_metric`）、`inference`、`dataset_versions`、`version_selector`。
- Step8 依赖的产物：
  - 可执行入口：`generated_project/main.py`（以及 `training/train.py`、`evaluation/evaluate.py` 等入口）。
  - `generated_project/requirements.txt`：第三方依赖清单，Step8 据此装环境。
  - `generated_project/config/config.yaml`：Step8 读配置跑训练/评估。
  - `step7/generation_report.yaml`：每文件生成状态、validation 分项、phases、traceability，Step8 判断代码是否可用。
  - `step7/generation_log.json`：逐文件尝试次数与状态，供排障。
