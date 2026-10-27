# Step 3：关键信息提取

## 思路

- **目标**：把 Step2 的结构化论文 JSON（`step2/structure_detect_paper.json`）中的正文文本、表格、图片，转成一份「复现导向」的关键信息 JSON（`step3/key_information.json`），作为后续 Step4（需求分析）与 Step5（工程设计）的核心数据源。本质是让 LLM 逐字从论文原文中抽取结构化字段，而不是写摘要。

- **输入 / 输出**：
  - 输入：`step2/structure_detect_paper.json`（含 `paragraphs`、`tables`、`images` 等，由 Step1/Step2 产出）。
  - 输出：`step3/key_information.json`（本文件即核心产物）。

- **核心流程**（三路并行，最终合并进同一个 result dict）：
  1. **正文信息提取**（`InformationExtractor`）：把 `paragraphs[].text` 拼成整段 `paper_text`，塞进 `prompts/step3_prompt.txt` 的 `{paper_text}` 占位符，调用 LLM 文本模型，解析 JSON，再做「结构校验 + 修复」。
  2. **表格分析**（`TableAnalyzer`）：读 `paper["tables"]`，逐表格式化后调 LLM 归类（experiment_results / ablation_study / comparison / dataset_statistics / hyperparameter_table / model_architecture / other），提取数值与 best_result，最后跨表汇总。
  3. **图片分析**（`FigureAnalyzer`）：对每张图片执行 `OCR → FigureCodeGenerator 生成 Python 分析代码 → CodeExecutor 子进程执行 → 解析 JSON` 的流水线，输出图片分类与复现相关组件。

- **字段分组**：正文提取的字段按 5 组组织 —— `methodology`（方法/模型/架构）、`dataset`（数据集）、`implementation_details`（框架/优化器/lr/batch/epoch/loss 等超参）、`experiment`（指标/基线/消融/结果）、`reproduction_requirements`（依赖模块/关键参数/复现难点）；外加 `figures`（正文里提到的图）与 `title`/`research_problem`/`motivation`/`main_contribution` 等元信息。

- **关键设计取舍与假设**：
  - **`NOT_FOUND` 约定**：论文未明确提到的字段统一填 `NOT_FOUND`，绝不编造；下游 Step4/5 用 `!= "NOT_FOUND"` 判断字段是否可用，缺失项由 Step5 用工程决策补齐。
  - **结构修复优于重新生成**：LLM 输出不稳定，因此 `InformationExtractor` 内置了「拍平字符串→dict、游离子字段归位、数组→对象、正则兜底搜索原文」四类修复，而不是简单地重试 LLM。
  - **「AI 生成代码再执行」的图片分析方式**：不直接让 LLM 看图，而是让 LLM 生成一段只依赖 OCR 文本 + 基础视觉信息（`numpy`/`cv2`/`PIL`）的 Python 代码，在受限子进程里执行后取 JSON。这样把「看图」拆成「OCR 给文字」+「代码给结构化判断」，规避多模态模型，也便于控制代码行为。
  - **模型上下文假设**：`InformationExtractor` 假设 LLM 64K token ≈ 48K 字符，正文预留 `max_text_len=44000`，超长时保留前 75% + 后 25%（超参数多在末尾）。

## 具体实现

### 文件清单

- `information_extractor.py` — 正文信息提取：拼 prompt → 调 LLM → 解析 JSON → 结构校验/修复 + 正则兜底搜索。
- `figure_analyzer.py` — 图片分析编排：OCR → 生成代码 → 执行代码 → 解析 JSON，并兜底为合法 dict。
- `figure_code_generator.py` — 调 LLM 生成图片分析 Python 代码，清理 markdown 包裹，落盘到 `generated/`。
- `table_analyzer.py` — 逐表调 LLM 归类并提取数值，含 JSON 解析、规则兜底、跨表汇总。
- `ocr_analyzer.py` — EasyOCR 懒加载（双重检查锁）图片文字识别，返回去重关键词列表（≤30）。
- `code_executor.py` — 在隔离子进程中执行 AI 生成的代码（导入黑名单、禁用 os.system/popen/subprocess、300s 超时、临时文件即用即删）。
- `step3_extract_information.py` — 主入口：读取输入 → 串起三路分析（表格/图片各用线程池并行）→ 写 `key_information.json`。
- `prompts/step3_prompt.txt` — 正文信息提取的 prompt（含严格 JSON Schema 与 8 条「特别提醒」）。
- `prompts/figure_code_generator_prompt.txt` — 图片分析代码生成 prompt（规定输出类型、允许/禁止库、读取方式、输出 JSON 格式）。
- `__init__.py` — 包标记。
- `tests/` — 当前仅有空的 `__init__.py`，无实质测试。

### 关键类 / 函数

**InformationExtractor**
- `extract(paper: dict) -> dict` — 主入口：拼正文 → `chat_with_text` → `_parse_json` → `_validate_and_fix`。
- `_parse_json(raw: str) -> dict`（静态）— 三级降级解析：直接 `json.loads` → 去 markdown 围栏再解析 → 正则抓最外层 `{...}`；全失败返回 `{"raw_response":…, "error":…}`。
- `_validate_and_fix(result, paper_text) -> dict` — 五类修复：model_architecture 字符串→dict、游离架构字段归位并补 `NOT_FOUND`、learning_rate/epoch 去前缀、缺字段正则兜底搜索、important_parameters 数组→对象、figures 无用字段清理。
- `_search_*`（`_search_batch_size`/`_search_learning_rate`/`_search_optimizer`/`_search_loss_function`/`_search_framework`/`_search_momentum`/`_search_weight_decay`/`_search_scheduler`）— 从原文正则兜底搜缺失参数。
- `_clean_numeric_value` / `_eval_expression` — 数值清洗、安全表达式求值（仅数字与 `/`）。

**FigureAnalyzer**
- `analyze(image_path: str) -> dict` — OCR → 生成代码 → 执行 → 解析 JSON；`None` 输出或非 dict 一律兜底为 `{"type":"other", …}`。

**FigureCodeGenerator**
- `generate_code(image_path, ocr_text) -> str` — 替换 `{image_path}`/`{ocr_text}` 占位符，调 LLM 生成代码，`_clean_code` 去围栏，`_save_code` 落盘。
- `_clean_code(code) -> str`（静态）— 移除 ` ```python ` / ` ``` ` 包裹。

**TableAnalyzer**
- `analyze(tables, paper_context) -> dict` — 线程池（≤5）并行逐表分析，输出 `tables_analyzed`/`table_results`/`aggregated_findings`。
- `_analyze_single_table` / `_format_table` / `_parse_json` / `_fallback_extract` — 单表分析、表格格式化、JSON 提取、规则兜底。
- `_aggregate` / `_llm_aggregate` — 汇总实验/对比/消融结论、追踪 best_result（区分指标方向）。
- `_metric_direction(metric) -> int` — 判断指标方向（越大越好/越小越好/未知）。

**OCRAnalyzer**
- `reader`（property）— EasyOCR reader 懒加载，双重检查锁保证只初始化一次。
- `analyze(image_path) -> list` — 读取图片（`np.fromfile` 支持中文路径）→ `readtext` → 过滤纯数字/过短 → 去重 → 前 30 个关键词；异常返回 `[]`。

**CodeExecutor**
- `run(code, image_path) -> Optional[str]` — 前置黑名单子串检查 → 写临时文件（沙箱 preamble + 代码）→ `subprocess.run`（300s 超时、捕获 stdout）→ 清理临时文件；返回 stdout 或 `None`。
- `SANDBOX_PREAMBLE`（类常量）— 替换 `builtins.__import__` 拦截重量级库导入、把 `os.system`/`os.popen` 抛异常、清空 `subprocess` 模块。

### 数据流与相邻 Step 衔接

`step2/structure_detect_paper.json` →（正文 LLM 提取 + 表格 LLM 分析 + 图片 OCR/代码执行分析）→ `step3/key_information.json`。

**key_information.json 顶层字段与结构**

```
title / research_problem / motivation / main_contribution     # 元信息
methodology:
  method_name / core_idea / pipeline / model
  model_architecture: { input, output, feature_extraction, classification,
                        pretrained_model, special_modules,
                        alexnet, googlenet, resnet, vgg, densenet,
                        mobilenet, efficientnet, transformer, lstm }
dataset: { dataset_name, dataset_source, dataset_size, number_of_classes,
           data_type, train_test_split, preprocessing }
implementation_details: { programming_framework, optimizer, learning_rate,
                          batch_size, epoch, loss_function, hardware,
                          training_strategy, momentum, weight_decay }   # momentum/weight_decay 由兜底搜索补入
experiment: { evaluation_metrics, baseline_methods, comparison_results,
              ablation_study, important_results }
reproduction_requirements: { required_modules[], important_parameters{}, possible_difficulties[] }
figures[]                  # 正文 LLM 理解的图片列表（type: model_architecture/data_example/workflow/experiment_result/comparison_chart/other）
paper_figures[]            # figures 的副本（同名键，内容一致）
table_analysis: { tables_analyzed, table_results[], aggregated_findings{} }
image_analysis[]           # FigureAnalyzer 逐图输出（type: architecture/experiment_result/dataset/other + id/page）
```

**Step4/5 依赖哪些字段（尤其 NOT_FOUND 约定）**

- `NOT_FOUND` 是全文契约：Step3 的 prompt 要求「论文没写就填 NOT_FOUND」，Step4/5 用 `x.get(field, "NOT_FOUND") != "NOT_FOUND"` 判断字段是否可复现；缺失/空值统一落入 `NOT_FOUND`，由 Step5 标记为「工程决策」或「论文未提供」。
- Step5 `base_analyzer.py` 维护了一份关键字段路径清单（`dataset.dataset_name`、`dataset.dataset_size`、`dataset.preprocessing`、`dataset.train_test_split`、`dataset.number_of_classes`、`implementation_details.optimizer/learning_rate/batch_size/epoch/loss_function/hardware/training_strategy`、`experiment.evaluation_metrics/important_results`、`reproduction_requirements.important_parameters/possible_difficulties` 等），据此逐字段扫描 NOT_FOUND 并决定是否用 LLM 补全。
- Step4（`step4_1_functional.py`、`step4_2_dataflow.py`、`step4_3_acceptance.py`、`step4_4_gap.py`）大量读取 `methodology.model_architecture`（尤其是 `alexnet/googlenet/resnet/vgg/...` 与 `pretrained_model`、`input/output`）与 `implementation_details`、`dataset` 字段，`step4_4_gap.py` 专门扫描 `NOT_FOUND` 生成「知识缺口」。
- 因此 Step3 输出的字段名、`NOT_FOUND` 占位、以及 `model_architecture` 必须是 dict（不能是字符串）对下游是硬约束——这正是 `_validate_and_fix` 存在的原因。
