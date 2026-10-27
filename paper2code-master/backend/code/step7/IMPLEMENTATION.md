# Step 7：代码生成

## 思路

- 目标：把 Step6 产出的 `project_brief.yaml`（项目结构 + 逐文件代码生成计划）落成一套可运行的 Python 项目骨架与源码，并对 LLM 生成的代码做规则化检查与自动修复，最终给出可复现入口与生成报告，交给 Step8 去实际执行实验。
- 输入 / 输出：
  - 输入：`step6/project_brief.yaml`（含 `code_generation_plan.per_file_specs`、`generation_order`、`design_digest`、`requirements_digest`、`traceability`、`decisions`、`data_dependency`、`unresolved_dataset_versions`、`risks_and_notes` 等）；可选回读 `step5/*.yaml`（`model_design.yaml` / `dataset.yaml` / `training_config.yaml`）补全架构细节。
  - 输出：`step7/generated_project/`（`config/config.yaml`、`data/`、`models/`、`training/`、`evaluation/`、`utils/`、`main.py`、`requirements.txt`、`README.md` 等）+ `step7/generation_report.yaml` + `step7/generation_log.json`。
- 核心流程（项目初始化 → 模板/LLM 生成 → 校验 → 修复 → 异步沙箱）：
  1. `ProjectInitializer` 依据 `per_file_specs` 创建目录骨架与空占位文件，并按依赖关系做拓扑排序得到生成顺序（被依赖者先生成，非 Python 文件优先）。
  2. 平凡文件（`requirements.txt` / `README.md` / `__init__.py`）走确定性模板产出，不调 LLM；模板未过校验时回退 LLM。
  3. 其余文件按拓扑层级逐层生成（层内互不依赖可并行）：`CodeContextBuilder` 整理上下文 → `CodeGenerator` 调 LLM 生成 → `CodeValidator` 规则校验。生成/修复 prompt 里依赖文件只给「接口摘要」（类/函数签名 + 文档首行 / YAML 键树），不给完整源码，显著缩短 prompt。
  4. 校验失败则进入 `CodeRepairer` 修复循环（最多 `MAX_REPAIR_ATTEMPTS`），修复后重新校验。LLM 档位按文件分层（`task_type_for_file`）：核心算法文件（`models/`、`training/`、`data/`、`main.py`、`run.py`、`config/config.yaml`）走 heavy，辅助文件（`utils/`、`evaluation/`、`__init__.py`、文档）走 light。
  5. 每层静态通过后，沙箱冒烟测试提交到后台线程池（与下一层 LLM 生成重叠），失败再做最多 1 次运行时修复；全部生成结束后统一 join。
  6. 全部文件生成后，对 `config.yaml` 做确定性注入（未定义来源的 dataset version 标记 `unresolved`、`data_source`/`acquisition` 顶层与 per-run、`split_strategy.leaf_mapping_*` 叶子映射依赖、`experiment.runs` 实验矩阵、`execution` 执行规格、requirements 同步），再跑项目级检查（文件完整性 / 第三方依赖 / 需求追踪 / Step6 决策一致性 / 未定义版本标记），最后写报告。
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
- `code_generator.py` — 调 LLM 文本模型，逐文件生成代码并剥离 markdown 代码块。
- `code_validator.py` — 规则化检查：语法/类/方法/import/配置引用/函数接口/结论写死 + 项目级检查。
- `code_repairer.py` — 校验失败后调 LLM 生成一次修复结果。
- `step7_main.py` — 总控制器：编排初始化 → 生成 → 校验 → 修复 → 注入 → 项目级检查 → 报告。

### 关键类 / 函数
- `ProjectInitializer.initialize(brief)` — 创建目录与空文件，返回拓扑排序后的文件顺序。
- `ProjectInitializer._ordered_files(files, specs)` — Kahn 拓扑排序；无依赖/依赖已满足者先出队，非 `.py` 优先，稳定。
- `CodeContextBuilder.build(file, generated)` — 汇总 spec / 需求 / 依赖接口摘要 / Step5 详情 / agent_notes，生成单个文件上下文。
- `CodeContextBuilder._related_code(file, spec, generated)` — 依赖文件的接口摘要：`.py` → `CodeValidator.interface_digest`，`.yaml` → `CodeValidator.config_digest`，文档类跳过。
- `CodeContextBuilder._step5_detail(file)` — 按文件路径回读 Step5 的 model/dataset/training/config 详情。
- `CodeValidator.interface_digest(code)` / `CodeValidator.config_digest(code)` — 提取 .py 公开接口摘要（类/方法/函数签名含默认值 + 文档首行 + 大写常量，不含实现）/ YAML 键树摘要（点号路径 + 标量取值，长列表折叠为 `[N 项]`），供 prompt 瘦身。
- `task_type_for_file(file)`（`code_generator.py`）— 文件 → LLM 档位路由（heavy/light），`CodeGenerator`/`CodeRepairer` 的 `llm_call` 共用；light 档可在 settings.json `task_models.light` 配更便宜的模型。
- `CodeGenerator.generate(context)` — 组装 prompt 调 `chat_with_text`，再用 `_extract_code` 剥离代码块。
- `CodeGenerator._output_requirements(file)` — 按 `.py`/`.yaml`/`.txt`/`.md` 给出针对性输出约束（如 py 要过 `ast.parse`、超参数不硬编码、config 键必须与 config.yaml 对齐）；入口文件（`main.py`/`run.py`）额外要求支持 `--run N`（argparse），按 `config['experiment']['runs'][N]` 逐组合执行（Step8 实验矩阵依赖）。
- `CodeValidator.validate(file, code, spec)` — 单文件八项检查（语法/类/方法/import/配置引用/函数接口/结论写死/入口 `--run`），返回 `{status, errors, checks}`。
- `CodeValidator.supports_run_arg(code)` / `set_expected_runs(n)` / `_check_entry_run_arg(...)` — 入口 `--run` 确定性检查：实验矩阵组合数 > 1 时，`main.py`/`run.py` 必须有 argparse `add_argument("--run", ...)`，否则判失败进入修复循环（组合数取自 Step5 `experiment_design.yaml` 的 `runs`，`Step7Runner._expected_run_count` 读入）。
- `CodeValidator.register(file, code)` / `set_config(config_dict)` — 注入已生成文件的类/函数/签名符号表与 config.yaml 内容，供跨文件检查。
- `CodeValidator._check_imports(code)` — 本地模块存在性 + `from X import Y` 的符号存在性。
- `CodeValidator._check_config_refs(file, code)` — 静态解析 `config[...]`/`config.get(...)` 键链，与 config.yaml 真实键比对。
- `CodeValidator._check_function_interface(code)` — 校验 `from X import Y` 跨文件调用与同文件顶层函数调用的参数个数/关键字名/必填参数。
- `CodeValidator.validate_project(plan_files, generated, brief)` — 项目级：文件完整性 / 第三方依赖 / 函数接口汇总 / 需求追踪 / 决策一致性 / 未定义版本标记。
- `CodeRepairer.repair(file, code, errors, spec, context)` — 组装修复 prompt 并返回修复后代码。
- `Step7Runner.run()` — 主流程编排，返回 `report["generation"]["status"] == "success"`。
- `Step7Runner._generate_one_file(...)` — 单文件「生成 → 校验 → 修复循环」（静态阶段；沙箱在后台）。
- `Step7Runner._template_trivial_file(...)` / `_template_package_init(...)` / `_template_requirements(...)` / `_template_readme(...)` — 平凡文件确定性模板，不调 LLM；`__init__.py` 的 re-export 类名取自兄弟模块 spec 的 `classes`（校验器保证与代码一致）。
- `Step7Runner._refresh_package_inits(...)` — 全部生成后按兄弟模块 AST 再对齐一次包 re-export。
- `Step7Runner._submit_sandbox(...)` / `_join_sandbox()` / `_sandbox_and_repair(...)` — 沙箱冒烟测试后台执行（`SANDBOX_TIMEOUT`（120s）/`SANDBOX_WORKERS` 控制超时与并发），失败做最多 1 次运行时修复，与后续层的 LLM 生成重叠。
- `Step7Runner._inject_unresolved_versions(...)` / `_inject_data_dependency(...)` / `_inject_experiment_runs(...)` / `_write_execution_spec(...)` — 对 config.yaml 做确定性注入。`_inject_data_dependency` 还注入 `split_strategy.leaf_mapping_file/constraint/on_missing`（Step6 登记的叶子映射依赖，防静默回退随机划分）；`_write_execution_spec` 依 `_entry_supports_run_arg`（与 `CodeValidator.supports_run_arg` 共用检测）决定 `execution.template` 是否带 `--run {run_index}`。

### 数据流与相邻 Step 衔接
- 生成项目的目录结构（由 Step6 `per_file_specs` 决定，典型 CNN 工程）：`config/config.yaml`、`data/dataset.py`、`models/<model>.py`、`training/train.py`、`evaluation/evaluate.py`、`utils/`、`main.py`、`requirements.txt`、`README.md`。
- `config.yaml` 顶层键（LLM 生成 + Step7 确定性注入补全）：`project`（含 `framework`）、`data_source`、`acquisition`、`experiment`（含 `runs[]`，每条带 `data_source`/`acquisition`）、`data`（按版本名，`{status: unresolved/resolved, reason}`）、`split_strategy`（含 `val_split_ratio`；leaf 划分时含 `leaf_mapping_file`/`leaf_mapping_constraint`/`leaf_mapping_on_missing`）、`training`、`training_modes`、`checkpoint`（含 `best_metric`）、`inference`、`dataset_versions`、`version_selector`。
- Step8 依赖的产物：
  - 可执行入口：`generated_project/main.py`（以及 `training/train.py`、`evaluation/evaluate.py` 等入口）。
  - `generated_project/requirements.txt`：第三方依赖清单，Step8 据此装环境。
  - `generated_project/config/config.yaml`：Step8 读配置跑训练/评估。
  - `step7/generation_report.yaml`：每文件生成状态、validation 分项、phases、traceability，Step8 判断代码是否可用。
  - `step7/generation_log.json`：逐文件尝试次数与状态，供排障。
