

## 纸上得来终觉浅:基于` CodeArts` 代码智能体的科学论文可复现性验证系统

> **LLM 配置说明**: 本项目使用 **DeepSeek-V3-0324** (`deepseek-chat`) 进行论文代码复现的代码生成。
> API Key 可通过前端界面设置，或通过环境变量 `DEEPSEEK_API_KEY` 配置。
> 获取 API Key: [https://platform.deepseek.com](https://platform.deepseek.com)

**|——————|———————|——————|————————————|**

更新时间        2026-7-17           袁伟栋           `README`文档的创建
                      2026-8-10           袁伟栋            更新
                      2026-8-20           袁伟栋            更新到 Step8 最新进度
                      2026-8-22           袁伟栋            更新文件结构至最新（新增控制器/临时文件工具，移除陈旧 stub）



**目前`README.md`文档主要是用于保留我们的思维记录，后期我们需要用AI工具进行规范化修改**



## 终极目标：实现论文复现



### 项目的主要思路

```
Paper
 ↓
Understanding
 ↓
Configuration
 ↓
Software Design
 ↓
Implementation

```



### 主要步骤

`ps`: 因为项目没有做完，每个`step`的任务可能会发生变化，需要根据情况具体分析

```
Step 1  | pdf to json
↓
Step 2  | clean json
↓
Step 3  | extract key information (because key words may cause something unexpected)
↓
Step 4  | requirements analysis (论文事实 → 系统需求定义)
↓
Step 5  | engineering design (需求 → 工程设计 YAML)
↓
Step 6  | project integration (Step4需求 + Step5设计 → project_brief.yaml)
↓
Step 7  | code generation (project_brief → 逐文件生成可执行代码)
↓
Step 8  | experiment execution (RUN/SKIP 调度 + 实际执行实验)
↓ (the steps below are unexplored)
Step 9  | code to json
↓
Step 10 | analyse code
↓
Step 11 | categorize code
↓
Step 12 | report the result
```



### 目前的实验进度

```
pdf
↓
raw_json                       |this is step1
↓
cleaned_json                   |this is step2.1
↓
repaired_json                  |this is step2.2
↓
structure_detected_json        |this is step2.3
↓
information_extracted_json     |this is step3
↓
requirements_summary.yaml      |this is step4 (需求分析)
↓
project_analysis.yaml          |this is step5 (工程设计)
↓
project_brief.yaml             |this is step6 (项目整合)
↓
generated_project/             |this is step7 (代码生成)
↓
execution_report.yaml          |this is step8 (实验执行)
↓
......
the next step is on the way -->>>
```



### 实验代码的文件结构

```
paper2code/

│
├── code/
│
│   ├── __init__.py
│   ├── config.py                         | deepseek配置、输入输出路径配置
│   ├── llm_utils.py                      | DeepSeek模型调用（文本模型）
│   ├── logger.py                         | 日志配置
│   ├── requirements.txt                  | 项目依赖
│   ├── compare_paper.py                  | 工具：论文文本 vs 提取信息对比
│   ├── deep_dive.py                      | 工具：调试 Step3 LLM 输出质量
│   ├── temp_file_manager.py              | 临时文件管理核心模块（保留/丢弃）
│   ├── temp_file_api.py                  | 临时文件管理 HTTP 接口
│   ├── pipeline_controller.py            | 流程控制器核心模块（逐 step 推进）
│   ├── pipeline_api.py                   | 流程控制器 HTTP 接口
│   │
│   ├── step1/
│   │   ├── __init__.py
│   │   ├── step1_pdf_to_json.py          | Step1入口：PDF转换JSON
│   │   ├── pdf_parser.py                 | PDF解析
│   │   ├── text_utils.py                 | 文本处理工具
│   │   ├── json_writer.py                | JSON写入
│   │   ├── image_extractor.py            | PDF图片提取
│   │   └── table_extractor.py            | 表格提取
│   │
│   ├── step2/
│   │   ├── __init__.py
│   │   ├── step2_1_clean_json.py         | Step2.1 JSON清洗
│   │   ├── cleaner.py                    | JSON清洗逻辑
│   │   ├── step2_2_repair_json.py        | Step2.2 JSON修复
│   │   ├── repairer.py                   | JSON修复逻辑
│   │   ├── step2_3_detect_structure.py   | Step2.3结构检测
│   │   └── structure_detector.py         | 修改type属性，方便后续分析
│   │
│   ├── step3/
│   │   ├── __init__.py
│   │   ├── step3_extract_information.py  | Step3入口：论文信息提取
│   │   ├── information_extractor.py     | 提取论文核心信息
│   │   │
│   │   ├── figure_analyzer.py           | 图片分析
│   │   ├── table_analyzer.py             | 表格分析
│   │   ├── ocr_analyzer.py              | OCR分析
│   │   │
│   │   ├── figure_code_generator.py     | 图片分析代码生成
│   │   ├── code_executor.py             | 执行AI生成代码
│   │   ├── prompts/
│   │   │   ├── step3_prompt.txt
│   │   │   └── figure_code_generator_prompt.txt
│   │   │
│   │   └── tests/
│   │       └── __init__.py
│   │
│   ├── step4/
│   │   ├── __init__.py
│   │   ├── base_requirement_analyzer.py  | 基类（三层防编造、JSON读取、YAML保存）
│   │   │
│   │   ├── step4_main.py                | Step4总入口
│   │   ├── step4_1_functional.py        | 4.1 功能需求分析
│   │   ├── step4_2_dataflow.py          | 4.2 数据流与接口分析
│   │   ├── step4_3_acceptance.py        | 4.3 验收标准分析
│   │   └── step4_4_gap.py               | 4.4 歧义分析
│   │
│   ├── step5/
│   │   ├── __init__.py
│   │   ├── STEP5_IMPLEMENTATION.md
│   │   ├── base_analyzer.py             | 基类（三层防编造、JSON读取、YAML保存）
│   │   │
│   │   ├── step5_main.py                | Step5总入口
│   │   ├── step5_1_goal.py              | 5.1 复现目标分析
│   │   ├── step5_2_tech.py              | 5.2 技术栈分析
│   │   ├── step5_3_data.py              | 5.3 数据集分析
│   │   ├── step5_4_model.py             | 5.4 模型设计
│   │   ├── step5_5_exp.py               | 5.5 实验流程设计
│   │   ├── step5_6_struct.py            | 5.6 项目文件架构
│   │   ├── step5_7_train.py             | 5.7 训练参数配置
│   │   └── step5_8_spec.py              | 5.8 代码规格 (LLM)
│   │
│   ├── step6/
│   │   ├── __init__.py
│   │   ├── step6_main.py                | Step6总入口
│   │   ├── step6_consistency.py         | 6.1 一致性检查（Step4↔Step5）
│   │   └── step6_summarizer.py          | 6.2 项目整合（生成project_brief.yaml）
│   │
│   ├── step7/
│   │   ├── __init__.py
│   │   ├── step7_main.py                | Step7总入口
│   │   ├── project_initializer.py       | 7.1 项目骨架初始化
│   │   ├── code_context_builder.py      | 7.2 代码上下文构建
│   │   ├── code_generator.py            | 7.3 代码生成 (LLM)
│   │   ├── code_validator.py            | 7.4 代码检查 (规则驱动)
│   │   └── code_repairer.py             | 7.5 代码修复 (LLM)
│   │
│   └── step8/
│       ├── __init__.py
│       ├── step8_main.py                | Step8总入口
│       ├── experiment_analyzer.py       | 8.1 实验列表解析
│       ├── resource_checker.py          | 8.2 资源检查（真实环境检测）
│       ├── experiment_scheduler.py      | 8.3 调度决策 (RUN/SKIP)
│       ├── experiment_runner.py         | 8.4 实验执行 (子进程)
│       └── result_collector.py          | 8.5 结果汇总
│
├── data/
│
│   ├── input/
│   │   ├── 1利用深度学习进行基于图像的植物病害检测.pdf
│   │   ├── 论文1.pdf
│   │   ├── 论文3.pdf
│   │   └── 论文5.pdf
│   │
│   ├── output/
│   │   └── plant/                              | 示例论文（植物病害检测）
│   │       ├── step1/
│   │       │   ├── raw_paper.json              | Step1输出
│   │       │   ├── images/                     | 提取图片 image_1..14.png
│   │       │   └── tables/                     | 提取表格 table_1.csv / table_2.csv / extracted_tables.json
│   │       │
│   │       ├── step2/
│   │       │   ├── cleaned_paper.json          | Step2.1输出
│   │       │   ├── repaired_paper.json         | Step2.2输出
│   │       │   └── structure_detect_paper.json | Step2.3输出
│   │       │
│   │       ├── step3/
│   │       │   ├── key_information.json        | Step3输出
│   │       │   └── generated/                  | AI生成的图片分析代码
│   │       │
│   │       ├── step4/
│   │       │   ├── functional_requirements.yaml | 4.1 功能需求
│   │       │   ├── data_flow.yaml               | 4.2 数据流与接口
│   │       │   ├── acceptance_criteria.yaml     | 4.3 验收标准
│   │       │   ├── gap_analysis.yaml            | 4.4 歧义分析
│   │       │   └── requirements_summary.yaml    | 聚合摘要
│   │       │
│   │       ├── step5/
│   │       │   ├── reproduction_goal.yaml       | 5.1 复现目标
│   │       │   ├── technology.yaml              | 5.2 技术栈
│   │       │   ├── dataset.yaml                 | 5.3 数据集
│   │       │   ├── model_design.yaml            | 5.4 模型设计
│   │       │   ├── experiment_design.yaml       | 5.5 实验流程
│   │       │   ├── project_structure.yaml       | 5.6 项目结构
│   │       │   ├── training_config.yaml         | 5.7 训练参数
│   │       │   ├── optimized_project_spec.yaml  | 5.8 代码规格 (LLM)
│   │       │   └── project_analysis.yaml        | 聚合摘要
│   │       │
│   │       ├── step6/
│   │       │   └── project_brief.yaml           | 项目简报（需求+设计整合）
│   │       │
│   │       ├── step7/
│   │       │   ├── generated_project/           | 生成的可执行代码项目
│   │       │   ├── generation_report.yaml       | 生成报告
│   │       │   └── generation_log.json          | 生成日志
│   │       │
│   │       └── step8/
│   │           ├── execution_plan.yaml          | 执行计划 (RUN/SKIP)
│   │           └── execution_report.yaml        | 执行报告
│
├── logs/
│   └── app.log
│
├── _build_report_ppt.py                     | 工具：生成 Step4~Step8 汇报 PPT
├── .gitignore
│
└── README.md
```

------

## Project Overview

**Paper2Code** is an AI-driven scientific paper reproducibility verification system built on DeepSeek's Large Language Models. Its goal is to take an academic PDF paper as input, progressively parse, clean, extract structured information, and ultimately produce a runnable code project that can be executed to verify reproducibility.

The current implementation covers Steps 1 through 8. Steps 9-12 (code analysis, categorization, comparison, and final reporting) are planned but not yet built.

------

## Python Version

Based on the `__pycache__` directories, the project was run with **Python 3.13** (`*.cpython-313.pyc` files). No Python version constraint is explicitly declared in any dependency file.

------

## Dependency Management

There is exactly one dependency file:

- **`d:\paper2code-master\paper2code-master\code\requirements.txt`**

No `pyproject.toml`, `setup.py`, `setup.cfg`, `Pipfile`, or `environment.yml` were found.

------

## Third-Party Python Packages

| Package                             | Version   | Used In                                                      | Purpose                                                      |
| ----------------------------------- | --------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| `requests`                          | >= 2.28.0 | `code/llm_utils.py`                                          | HTTP client for calling the DeepSeek Chat API                |
| `PyMuPDF` (imported as `fitz`)      | >= 1.21.0 | `code/step1/pdf_parser.py`, `image_extractor.py`, `table_extractor.py` | PDF parsing: extracting text blocks, images, and grid-based tables from academic PDFs |
| `Pillow`                            | >= 9.0.0  | `code/step3/` (via generated figure code)                    | Image loading in the figure analysis pipeline                |
| `opencv-python` (imported as `cv2`) | >= 4.7.0  | `code/step3/ocr_analyzer.py`, generated figure code          | Image decoding (supports Chinese-character paths via `np.fromfile` + `cv2.imdecode`), and image processing in AI-generated analysis code |
| `numpy`                             | >= 1.24.0 | `code/step3/ocr_analyzer.py`, generated figure code          | Array operations for image data                              |
| `easyocr`                           | >= 1.7.0  | `code/step3/ocr_analyzer.py`                                 | OCR (Optical Character Recognition) to extract text words from PDF figures/illustrations |
| `pyyaml`                            | >= 6.0    | `code/step4/`, `code/step5/`, `code/step6/`, `code/step7/`, `code/step8/` | Reading/writing YAML configuration and report files throughout the Step4-Step8 pipeline |

No other third-party packages are used. Notably, there are **no deep learning framework dependencies** (no `torch`, `torchvision`, `tensorflow`, etc.) in the current codebase -- PyTorch/TensorFlow appear only as strings in Step5's recommended external dependencies for the code it plans to generate.

------

## Non-Python Technologies

- **No JavaScript/TypeScript, Go, Rust, shell scripts, Docker files, Makefiles, or CI/CD configurations** were found.
- The project is **pure Python**, intended to run on **Windows** (per the figure code generator prompt which specifies "Windows + Python").

------

## External Services / APIs

### DeepSeek API (the only external dependency)

| Details            | Values                                                       |
| ------------------ | ------------------------------------------------------------ |
| **Base URL**       | `https://api.deepseek.com/v1/chat/completions`               |
| **Text Model**     | `deepseek-chat`                                              |
| **API Key**        | Currently hardcoded in `code/config.py` (line 22: `sk-230fb0562f1341a2ae556a2b67516f8a`); the file also includes logic to read from `DEEPSEEK_API_KEY` environment variable |
| **Authentication** | Bearer token in HTTP header                                  |
| **Timeout**        | 300 seconds per request                                      |
| **Retry Logic**    | Up to 3 retries with exponential backoff (base delay 2s) on 5xx / timeout / connection / empty-or-malformed content; 4xx errors are NOT retried |

The API is called through a single wrapper function in `code/llm_utils.py`:

- `chat_with_text(prompt)` -- for structured information extraction from paper text, table analysis, and generating figure-analysis code

------

## Overall Architecture -- Step-by-Step Pipeline

```
PDF (data/input/)
    |
    v
Step 1  (code/step1/)        PDF --> raw JSON
    |                           PyMuPDF parses pages into paragraphs (text + bbox),
    |                           extracts embedded images as PNGs, and detects
    |                           grid-based tables. Saves raw_paper.json.
    v
Step 2.1 (code/step2/)       Clean JSON
    |                           Removes noise: dates, page numbers, duplicate
    |                           paragraphs, merges single-letter word breaks.
    |                           Saves cleaned_paper.json.
    v
Step 2.2 (code/step2/)       Repair JSON
    |                           Fixes PDF ligatures (fi, fl, ffi, ffl),
    |                           internal hyphens, and merges broken text blocks.
    |                           Saves repaired_paper.json.
    v
Step 2.3 (code/step2/)       Structure Detection
    |                           Classifies each paragraph as: title, author,
    |                           keywords, abstract, heading, or paragraph.
    |                           Saves structure_detect_paper.json.
    v
Step 3  (code/step3/)        Key Information Extraction
    |                           Three parallel pipelines:
    |                             a) Text: LLM (deepseek-chat) extracts structured
    |                                JSON per a rigid schema (methodology, dataset,
    |                                implementation_details, experiment, etc.)
    |                             b) Tables: LLM analyzes extracted tables
    |                                (experiment results, ablation studies, etc.)
    |                             c) Figures: OCR (EasyOCR) --> FigureCodeGenerator
    |                                (LLM generates Python analysis code) -->
    |                                CodeExecutor (sandboxed subprocess execution)
    |                                --> JSON figure classification
    |                           Saves key_information.json.
    v
Step 4  (code/step4/)        Requirements Analysis
    |                           4 sub-steps, all reading from key_information.json:
    |                             4.1: functional_requirements.yaml (功能需求)
    |                             4.2: data_flow.yaml (数据流与接口)
    |                             4.3: acceptance_criteria.yaml (验收标准)
    |                             4.4: gap_analysis.yaml (歧义分析)
    |                           Also aggregates into requirements_summary.yaml.
    v
Step 5  (code/step5/)        Engineering Design
    |                           8 sub-steps, all reading from key_information.json:
    |                             5.1: reproduction_goal.yaml (paper type, task type)
    |                             5.2: technology.yaml (framework, model backbone)
    |                             5.3: dataset.yaml (dataset specs, preprocessing)
    |                             5.4: model_design.yaml (model class names, types)
    |                             5.5: experiment_design.yaml (pipeline, metrics)
    |                             5.6: project_structure.yaml (code dir tree)
    |                             5.7: training_config.yaml (hyperparameters)
    |                             5.8: optimized_project_spec.yaml (LLM-refined
    |                                  per-file code requirements)
    |                           Also aggregates into project_analysis.yaml.
    v
Step 6  (code/step6/)        Project Integration
    |                           Rule-driven alignment of Step4 (requirements)
    |                           and Step5 (design): consistency checks (conflicts
    |                           / gaps / assumptions), FR-to-code traceability,
    |                           and an ordered code-generation plan.
    |                           Saves project_brief.yaml.
    v
Step 7  (code/step7/)        Code Generation
    |                           Initializes project skeleton, then generates
    |                           each file in dependency order via LLM. Each file
    |                           is validated (syntax / classes / methods / imports
    |                           / config refs / function interfaces) and repaired
    |                           (max 3 attempts) until it passes. Produces
    |                           generated_project/ + generation_report.yaml.
    v
Step 8  (code/step8/)        Experiment Execution
    |                           Parses experiment list from generated_project's
    |                           config.yaml, detects real environment resources
    |                           (GPU / memory / dataset), decides RUN vs SKIP per
    |                           experiment, executes RUN experiments in
    |                           subprocesses, and collects results.
    |                           Saves execution_plan.yaml + execution_report.yaml.
    v
Step 9+ (not yet implemented) Code analysis, categorization, comparison, and
                              final reproducibility reporting.
```

### Step4/Step5's "Three-Layer Anti-Fabrication" Mechanism

Each Step4 (requirements) and Step5 (design) analyzer uses a defense-in-depth strategy:

1. **Layer 1 (Rule extraction)**: Deterministic code extracts known facts from `key_information.json`. No LLM calls.
2. **Layer 2 (LLM completion)**: Only fields still marked `NOT_FOUND` are sent to the LLM with strict instructions to not invent data.
3. **Layer 3 (Post-validation)**: LLM outputs are cross-checked against the original paper text. Unverifiable concepts are flagged with warnings.

Step6 and Step7 are additionally rule-driven: Step6 aligns requirements and design via deterministic consistency checks (no LLM), and Step7 validates generated code with deterministic AST/YAML/interface checks before any repair.

------

## Code Organization



```
code/
  config.py              -- Global config (API keys, all file paths)
  llm_utils.py           -- DeepSeek API wrapper (text)
  logger.py              -- Lazy-initialized logging (file + console)
  compare_paper.py       -- Utility: compare paper text vs extracted info
  deep_dive.py           -- Utility: debug Step3 LLM output quality
  temp_file_manager.py   -- Temp file management core module (list/keep/discard)
  temp_file_api.py       -- Zero-dependency HTTP wrapper for temp file management
  pipeline_controller.py -- Step-by-step pipeline controller (run next step)
  pipeline_api.py        -- Zero-dependency HTTP wrapper for the pipeline controller
  requirements.txt       -- Package dependencies
  step1/                 -- PDF parsing
    step1_pdf_to_json.py -- Entry point (argparse CLI)
    pdf_parser.py        -- PyMuPDF paragraph extraction
    image_extractor.py   -- Embedded image extraction
    table_extractor.py   -- Grid table + structured text detection
    text_utils.py        -- Text cleaning (whitespace, newlines)
    json_writer.py       -- JSON serialization
  step2/                 -- JSON cleaning and repair
    step2_1_clean_json.py
    cleaner.py
    step2_2_repair_json.py
    repairer.py
    step2_3_detect_structure.py
    structure_detector.py
  step3/                 -- LLM-based information extraction
    step3_extract_information.py  -- Entry point
    information_extractor.py      -- Core LLM extraction + validation/repair
    figure_analyzer.py            -- Figure analysis pipeline
    figure_code_generator.py      -- LLM code generation for figures
    ocr_analyzer.py               -- EasyOCR wrapper
    code_executor.py              -- Sandboxed Python subprocess executor
    table_analyzer.py             -- LLM table content analysis
    prompts/                      -- LLM prompt templates
    tests/                        -- Test scripts
  step4/                 -- Requirements analysis (论文事实 → 需求定义)
    step4_main.py       -- Entry point
    base_requirement_analyzer.py -- Base class (3-layer protection, YAML I/O)
    step4_1_functional.py -- 4.1 Functional requirements
    step4_2_dataflow.py   -- 4.2 Data flow & interfaces
    step4_3_acceptance.py -- 4.3 Acceptance criteria
    step4_4_gap.py        -- 4.4 Ambiguity/gap analysis
  step5/                 -- Engineering design (需求 → 工程设计 YAML)
    step5_main.py       -- Entry point
    base_analyzer.py     -- Base class (3-layer protection, YAML I/O)
    step5_1_goal.py      -- 5.1 Reproduction goals
    step5_2_tech.py      -- 5.2 Technology stack
    step5_3_data.py      -- 5.3 Dataset analysis
    step5_4_model.py     -- 5.4 Model design
    step5_5_exp.py       -- 5.5 Experiment design
    step5_6_struct.py    -- 5.6 Project structure
    step5_7_train.py     -- 5.7 Training config
    step5_8_spec.py      -- 5.8 LLM-refined code specs
    STEP5_IMPLEMENTATION.md -- Detailed documentation
  step6/                 -- Project integration (Step4 + Step5 → project_brief)
    step6_main.py       -- Entry point
    step6_consistency.py -- 6.1 Consistency checker (rule-driven, no LLM)
    step6_summarizer.py  -- 6.2 Integrator (rule-driven, no LLM)
  step7/                 -- Code generation (project_brief → generated code)
    step7_main.py       -- Entry point
    project_initializer.py -- 7.1 Project skeleton init (no LLM)
    code_context_builder.py -- 7.2 Context builder
    code_generator.py    -- 7.3 LLM file generator
    code_validator.py    -- 7.4 Rule-driven validator (AST/YAML/interface)
    code_repairer.py     -- 7.5 LLM repairer (max 3 attempts)
  step8/                 -- Experiment execution (RUN/SKIP decision + execute)
    step8_main.py       -- Entry point
    experiment_analyzer.py -- 8.1 Experiment list parser
    resource_checker.py  -- 8.2 Real-environment resource check
    experiment_scheduler.py -- 8.3 RUN/SKIP decision
    experiment_runner.py -- 8.4 Subprocess runner
    result_collector.py  -- 8.5 Result aggregation
```

------

## 临时文件管理工具（Temp File Management）

流水线每一步都会在 `data/output/{paper}/` 下生成大量中间与最终产物。本工具把「一次运行的完整论文输出」作为管理单元，供前端在运行结束后询问用户「是否保存这次生成的文件」：

- **核心模块** `code/temp_file_manager.py` —— 纯 Python、零第三方依赖，可被前端通过子进程直接调用。
- **HTTP 接口** `code/temp_file_api.py` —— 基于标准库 `http.server`（无需安装 FastAPI/Flask），把核心模块包装成 REST 接口供 Web 前端调用，已开启 CORS。

### 核心模块用法（CLI / 子进程）

```bash
cd code
python -m temp_file_manager list                     # 列出所有论文输出及占用
python -m temp_file_manager summary                  # 汇总所有临时输出（总量/文件数）
python -m temp_file_manager detail  --paper plant    # 单次运行完整文件明细
python -m temp_file_manager keep    --paper plant    # 保留原位（不删除）
python -m temp_file_manager discard --paper plant    # 丢弃（删除该次运行的完整输出）
```

### HTTP 接口

启动：

```bash
cd code
python -m temp_file_api --port 8000                  # 本机
python -m temp_file_api --host 0.0.0.0 --port 8000   # 允许局域网/远程前端访问
```

| 方法   | 路径                     | 说明                                     |
| ------ | ------------------------ | ---------------------------------------- |
| GET    | `/health`                | 健康检查                                 |
| GET    | `/summary`               | 汇总所有临时输出（总量/文件数/每篇占用） |
| GET    | `/papers`                | 列出所有论文输出（摘要）                 |
| GET    | `/papers/{paper}`        | 单次运行完整文件明细                     |
| POST   | `/papers/{paper}/keep`   | 保留原位（不删除）                       |
| DELETE | `/papers/{paper}`        | 丢弃（删除 `data/output/{paper}/` 目录） |

保留/丢弃语义：**keep = 原位保留（文件不动）**；**discard = 删除该次运行的完整输出目录**。
前端「是否保存本次生成的文件」即对应：保存 → `POST /papers/{paper}/keep`，不保存 → `DELETE /papers/{paper}`。

安全：`keep`/`discard` 仅接受 `data/output/` 下的一级子目录名，已做路径穿越防护（拒绝 `..`、`/`、`\` 等）。

------

## 流程控制器（Pipeline Controller）

前端逐步骤驱动流水线的控制器。前端每发一次「下一步」指令，控制器就启动下一个未完成的 step；遇异常时把「异常情况」返回给前端，详细 traceback 由前端自行读取 `logs/app.log`。

- **核心模块** `code/pipeline_controller.py` —— 纯 Python、零第三方依赖，可被前端通过子进程直接调用。
- **HTTP 接口** `code/pipeline_api.py` —— 基于标准库 `http.server`，把控制器包装成 REST 接口供前端调用。

### 设计要点

- 一次只跑一个 step（前端每发一次「下一步」→ 推进一个 step）。
- 「下一个 step」由输出文件是否存在推导（输出已存在即视为已完成），天然支持断点续跑。
- 每个 step 用子进程 `python -m <module> ...` 运行；异常退出时把 stderr 尾巴写进 `logs/app.log`，并向前端返回简短错误摘要。
- 记录上次执行过的 step，`retry` 可强制重跑它（用于 step 失败后手动重试）。
- 线程锁保证同一时刻只运行一个 step（并发「下一步」返回 busy）。

### 核心模块用法（CLI / 子进程）

```bash
cd code
python -m pipeline_controller status                # 查看进度
python -m pipeline_controller next                  # 运行下一个未完成的 step
python -m pipeline_controller retry                 # 重跑上次执行过的 step
python -m pipeline_controller run step3             # 运行指定 step
python -m pipeline_controller run step3 --force     # 忽略已完成，强制重跑
```

### HTTP 接口

启动：

```bash
cd code
python -m pipeline_api --port 8001                  # 本机
python -m pipeline_api --paper plant --port 8001    # 指定论文
```

| 方法 | 路径          | 说明                                   |
| ---- | ------------- | -------------------------------------- |
| GET  | `/health`     | 健康检查                               |
| GET  | `/status`     | 当前进度（下一步/已完成/剩余）         |
| GET  | `/steps`      | 步骤清单（含每步完成状态，供前端渲染） |
| POST | `/next`       | 运行下一个未完成的 step                |
| POST | `/retry`      | 重跑上次执行过的 step                  |
| POST | `/run/{step}` | 运行指定 step（`?force=1` 强制重跑）   |

返回结果的 `status` 取值：`success` / `error` / `skipped` / `busy` / `complete`。异常时返回的 `error` 是简短摘要，前端读取 `logs/app.log` 获取完整 traceback。

与临时文件管理配合：`discard` 删除某次输出 → 反复 `next` 重新推进整个流水线。

### 重试（retry）

针对 DeepSeek 偶发「无法返回」做了两层重试：

1. **LLM 层（自动）**：`code/llm_utils.py` 对「返回空内容 / 非 JSON / 结构缺失」也纳入重试（原来只重试 5xx/超时/连接失败），最多重试 `MAX_RETRIES` 次。
2. **控制器层（手动）**：step 失败后，前端可调 `POST /retry`（或 CLI `retry`）强制重跑上次执行过的 step；`GET /status` 会带上 `last_result` 供前端判断是否需要重试。

------

## Key Configuration Files

- **`d:\paper2code-master\paper2code-master\code\config.py`** -- The single source of truth for all paths, API keys, and the text model name.
- **Prompt files**: `code/step3/prompts/step3_prompt.txt` and `code/step3/prompts/figure_code_generator_prompt.txt` -- Detailed Chinese + English LLM prompts defining the exact JSON schema for information extraction and the rules for AI-generated figure analysis code.

------

## Security Notes

- The **DeepSeek API key is hardcoded** in `config.py` (line 22). The `.gitignore` does list `.env` files to prevent accidental commits of secrets, but the key is currently embedded in source code.
- Step3's `CodeExecutor` implements a **sandboxed execution** model for AI-generated code: forbidden imports (`tensorflow`, `torch`, `pytesseract`), disabled `os.system`/`os.popen`/`subprocess`, and 300-second timeout. Execution runs in an isolated `subprocess` with temp files cleaned up afterwards.

------

## Input/Output Data Flow

- **Input**: PDF files in `data/input/` (currently: a Chinese paper on plant disease detection using deep learning)
- **Output**: Organized per-paper under `data/output/{paper_name}/` with JSON, YAML, PNG images, and CSV table files
- **The sample paper processed is `plant`**: `data/output/plant/` contains the full pipeline output including 14 extracted figures, 2 tables, Step4-Step8 YAML/JSON reports, and the generated executable code project under `step7/generated_project/`

Agent 跑完了，补充一些我之前的梳理没覆盖到的细节：

------

## 完整技术栈

### 第三方 Python 包

| 包                      | 版本要求 | 用途                                                     |
| ----------------------- | -------- | -------------------------------------------------------- |
| **PyMuPDF (fitz)**      | ≥ 1.21.0 | PDF 解析：提取文本块、图片、表格                         |
| **requests**            | ≥ 2.28.0 | 调用 DeepSeek Chat API                                   |
| **Pillow**              | ≥ 9.0.0  | 图片加载（图片分析流程）                                 |
| **opencv-python (cv2)** | ≥ 4.7.0  | 图片解码（支持中文路径，`np.fromfile` + `cv2.imdecode`） |
| **numpy**               | ≥ 1.24.0 | 数组运算                                                 |
| **easyocr**             | ≥ 1.7.0  | OCR 识别论文图表中的文字                                 |
| **pyyaml**              | ≥ 6.0    | Step4-Step8 读写 YAML 配置与报告文件                      |

### 无深度学习框架依赖

项目本身**不依赖 PyTorch / TensorFlow** — 这些只作为 Step5 生成代码时的推荐依赖出现。

### 外部服务

| 服务         | 模型                      | 用途                             |
| ------------ | ------------------------- | -------------------------------- |
| DeepSeek API | `deepseek-chat`（文本）   | 信息提取、表格/图片分析、工程设计与代码生成 |

### 纯 Python 项目

- 无 JavaScript/TypeScript/Go/Rust
- 无 Docker / Makefile / CI 配置
- Python 3.13（根据 `__pycache__` 文件名推断）

------

## Pipeline 全貌



```
PDF 输入
  ↓
Step1: PyMuPDF 解析 → raw_paper.json（段落 + 图片 + 表格）
  ↓
Step2.1: 文本清洗（去日期/页码/重复段落）→ cleaned_paper.json
  ↓
Step2.2: 文本修复（连字/断行合并）→ repaired_paper.json
  ↓
Step2.3: 结构检测（标题/作者/摘要/正文分类）→ structure_detect_paper.json
  ↓
Step3: LLM 关键信息提取（三路并行）
        ├─ 文本 → DeepSeek 提取结构化 JSON
        ├─ 表格 → LLM 分析实验结果
        └─ 图表 → EasyOCR + FigureCodeGenerator（LLM 生成分析代码 → 沙箱执行）
        → key_information.json
  ↓
Step4: 需求分析（4 个子步骤）
        4.1 功能需求 → 4.4 歧义分析
        → step4/*.yaml
  ↓
Step5: 工程设计（8 个子步骤）
        5.1 复现目标 → 5.8 LLM 优化后的代码规格
        → step5/*.yaml
  ↓
Step6: 项目整合（规则驱动，无 LLM）
        Step4 需求 + Step5 设计 → 一致性检查 + 追溯矩阵 + 生成计划
        → step6/project_brief.yaml
  ↓
Step7: 代码生成（LLM 生成 + 规则检查 + 修复）
        逐文件生成 → 检查(语法/类/方法/import/接口) → 修复(最多3次)
        → step7/generated_project/
  ↓
Step8: 实验执行（资源检查 + RUN/SKIP 调度）
        解析实验 → 检测真实环境 → 决策 → 子进程执行 RUN 实验
        → step8/execution_report.yaml
  ↓
Step9-12: 代码分析 + 迭代优化 + 报告（尚未实现）
```

### Step4/Step5 的三层防幻觉机制

1. **规则提取**：先用确定性代码从 `key_information.json` 中提取已知事实，不调 LLM
2. **LLM 补全**：仅对标记为 `NOT_FOUND` 的字段调用 LLM，且要求不得编造
3. **后验证**：LLM 输出与原始论文交叉校验，不可验证的概念加警告标记

Step6 与 Step7 的检查均为规则驱动（不调 LLM）：Step6 用确定性一致性检查对齐需求与设计，Step7 用 AST/YAML/接口检查验证生成代码。

### 安全措施

Step3 的 `CodeExecutor` 对 AI 生成的代码做了沙箱隔离：禁止 `os.system`/`subprocess`、禁止 import PyTorch/TensorFlow 等重框架、300 秒超时、临时文件自动清理。
