# Step 4：需求分析

## 思路

- 目标：把 Step3 产出的论文事实（`key_information.json`）转化为「系统应该做什么」的需求定义，供 Step5 做工程设计、Step6 做迭代修复判定。核心是把「论文说了什么」转成四类结构化需求：功能需求、数据流与接口、验收标准、歧义与缺失。

- 输入 / 输出：
  - 输入：`data/output/{paper}/step3/key_information.json`（Step3 的关键信息提取结果，唯一输入；另会回溯 Step2.3 的 `structure_detect_paper.json` 用于 LLM 补全时的原文上下文）。
  - 输出：`data/output/{paper}/step4/` 下的五份文件：
    - `functional_requirements.yaml`（4.1 功能需求）
    - `data_flow.yaml`（4.2 数据流与接口）
    - `acceptance_criteria.yaml`（4.3 验收标准）
    - `gap_analysis.yaml`（4.4 歧义与缺失）
    - `requirements_summary.yaml`（汇总，仅聚合计数，供 controller 判定 Step4 完成）

- 核心流程（四个子分析器、三层防编造机制）：
  - 四个子分析器（`FunctionalAnalyzer` / `DataFlowAnalyzer` / `AcceptanceAnalyzer` / `GapAnalyzer`）彼此独立：都只读 `key_information.json`，各自产出独立 yaml，由 `step4_main.run_paper` 用 `ThreadPoolExecutor`（最多 4 线程）并行调度。每个分析器是独立实例，ID 计数器与 `key_info` 均为实例级，互不冲突。
  - 三层防编造机制（继承自 `step5.BaseAnalyzer` 的 `analyze()`）：
    1. **Layer 1 规则提取**：各子类实现 `extract_by_rules()`，纯规则从 `key_info` 抽取，未知字段填 `"NOT_FOUND"`。
    2. **Layer 2 LLM 补全**：`_find_unknown()` 递归找出 `NOT_FOUND`/空字段，`_call_llm()` 以「只从论文信息提取、没提到填 NOT_FOUND、绝不编造」的 prompt 请求补全，`_merge()` 只允许填补 NOT_FOUND（规则结果优先）。
    3. **Layer 3 后验证**：`_validate()` 做关键字段存在性审计 + 词汇验证（检查产出字符串中的概念词是否在原文中有依据）。
  - `--skip-llm` 可跳过 Layer 2（仅规则提取）。

- 关键设计取舍与假设：
  - 复用 Step5 的 `BaseAnalyzer` 而非独立实现，避免三套防编造逻辑重复；代价是 Step4 与 Step5 强耦合（Step4 输出目录由子类覆盖为 `step4/`）。
  - 需求必须有论文出处（`source` 字段用点号路径追溯到 `key_information.json`），便于审计与反编造。
  - 图像 vs 非图像是贯穿四份产物的关键分支：`_is_image_data()` 通过数据集字段的信号词（image/rgb/grayscale/imagenet/cifar/mnist/leaf/pixel/resize 等）判定，非图像数据（合成向量/RL 状态/无数据）走通用模板，避免泄漏 PIL/ImageNet/3 通道/灰度 repeat/GoogLeNet aux 等图像专属概念。
  - 训练 vs 非训练论文分支：`has_training` 由 optimizer/learning_rate/epoch/loss_function 是否 NOT_FOUND 判定，决定是否生成训练阶段与迁移学习需求。
  - 迁移学习被显式建模为与「从头训练」并列的独立模式（`training_mode` 枚举），并禁止用单一 `pretrained=True` 布尔混过去。
  - 编号：FR/IF/AC/GAP 各自带前缀 + 两位序号（`FR-01` 等），计数器在每个分析器 `extract_by_rules()` 开头 `_reset_counter()` 重置。

## 具体实现

### 文件清单

- `__init__.py` — 模块级说明文档（四个子模块对应的输出文件）。
- `base_requirement_analyzer.py` — Step4 基类 `BaseRequirementAnalyzer`，继承 `step5.BaseAnalyzer`，覆盖输出目录为 `step4/`，定义 ID 前缀（FR/IF/AC/GAP）与计数器、`_source`/`_get_kv` 辅助。
- `step4_1_functional.py` — 4.1 功能需求分析，产出 `functional_requirements.yaml`。
- `step4_2_dataflow.py` — 4.2 数据流与接口分析，产出 `data_flow.yaml`。
- `step4_3_acceptance.py` — 4.3 验收标准分析，产出 `acceptance_criteria.yaml`。
- `step4_4_gap.py` — 4.4 歧义与缺失分析，产出 `gap_analysis.yaml`。
- `step4_main.py` — 总入口：并行调度 4.1~4.4，聚合生成 `requirements_summary.yaml`。

### 关键类 / 函数

- `BaseRequirementAnalyzer.__init__()` — 调用父类初始化（加载 `key_info`/`structure`）后，把 `output_dir` 覆盖为 `step4/` 并创建目录，初始化 `_counters`（FR/IF/AC/GAP 各从 0 开始）。
- `BaseRequirementAnalyzer._next_id(prefix)` / `_reset_counter(prefix)` — 生成带序号 ID / 重置计数器。
- `BaseRequirementAnalyzer._source(*paths)` — 生成来源追溯字符串（点号连接）。
- `BaseRequirementAnalyzer._get_kv(*paths, default)` — 按路径从 `key_info` 取值并转字符串。
- `BaseAnalyzer.analyze(skip_llm)`（继承自 Step5） — 三层防编造主流程：`extract_by_rules()` → `_find_unknown`/`_call_llm`/`_merge` → `_validate`，最后加 `_meta`。
- `BaseAnalyzer._find_unknown(data)` — 递归找 `NOT_FOUND`/空字段，返回点号/下标路径列表。
- `BaseAnalyzer._call_llm(unknown_fields, current)` — 构造「只提取、不编造」prompt 请求 LLM，返回 JSON。
- `BaseAnalyzer._merge(base, update)` — 合并 LLM 结果，只填补 NOT_FOUND（规则结果优先）。
- `BaseAnalyzer._validate(result)` — 关键字段存在性审计 + 词汇验证。
- `BaseAnalyzer._is_image_data(ds)` — 图像数据判定（信号词匹配）。
- `BaseAnalyzer._pick(*values)` / `_get_nested` / `_extract_num_classes` — 通用取值工具。
- `FunctionalAnalyzer.extract_by_rules()` — 串联数据/模型/训练/评估四类需求的规则提取，汇总 `total` 与 `categories`。
- `FunctionalAnalyzer._extract_data_requirements/_extract_model_requirements/_extract_training_requirements/_extract_evaluation_requirements` — 分别生成对应需求条目。
- `DataFlowAnalyzer.extract_by_rules()` — 构建 pipeline 阶段 + 接口契约 + 数据格式。
- `DataFlowAnalyzer._build_stages/_build_generic_stages` — 图像/非图像两条 pipeline 阶段模板。
- `DataFlowAnalyzer._build_interfaces/_build_generic_interfaces` — 按相邻阶段对生成接口契约（图像/非图像两套模板）。
- `DataFlowAnalyzer._infer_data_formats` — 推断数据格式（输入图像/张量/输出 shape）。
- `AcceptanceAnalyzer.extract_by_rules()` — 生成精度目标、功能测试、边界情况、可复现性约束。
- `AcceptanceAnalyzer._extract_accuracy_targets/_derive_functional_tests/_derive_edge_cases/_extract_reproducibility` — 各子块的规则提取。
- `GapAnalyzer.extract_by_rules()` — 扫描 `CRITICAL_PATHS` 关键路径 + 数值占位符 + 模糊措辞，汇总 gap 与已解决假设。
- `GapAnalyzer._check_vague_descriptions/_extract_resolved_assumptions` — 模糊措辞扫描 / 已解决假设收集。
- `step4_main.run_paper(paper_name, skip_llm, force)` — 校验前置文件、并行跑四个子步、聚合 summary。
- `step4_main._run_one(...)` — 单子步执行，带子产物级断点续跑（输出已存在且非 force 时复用）。
- `step4_main._build_summary(...)` — 聚合四份结果生成 `requirements_summary.yaml` 的计数摘要。

### 数据流与相邻 Step 衔接

#### 各 YAML 的顶层结构

`functional_requirements.yaml`：
```yaml
paper_name: str
requirements:            # 列表
  - id: "FR-01"
    name: str
    description: str
    input: str
    output: str/对象
    source: str          # 点号路径追溯
    verification: str
    # 可选: design_decision / implementation_notes
summary: {total: int, categories: {分类名: 计数}}
```

`data_flow.yaml`：
```yaml
paper_name: str
pipeline:                # 列表，按阶段顺序
  - stage: int
    name: str
    description: str
    entry_data: str
    exit_data: str
    related_fr: [str]    # 引用 4.1 的 FR 编号
    # 可选: constraints / detail / training_loop
interfaces:              # 列表，相邻阶段间的接口契约
  - id: "IF-01"
    from_stage: str
    to_stage: str
    data: str
    format: str
    shape: str
    contract: str
    validation: str
data_formats: {input_image|input_data, tensor, output}
```

`acceptance_criteria.yaml`：
```yaml
paper_name: str
accuracy_targets: {primary: {metric, target, source, tolerance}, secondary?: {}}
functional_tests:        # 列表
  - id: "AC-01"
    name: str
    description: str
    steps: [str]
    expected: str
    related_fr: [str]
edge_cases:              # 列表: {id, case, handling, severity}
reproducibility: {random_seed, hardware, environment, known_difficulties, suggested_validation_runs}
```

`gap_analysis.yaml`：
```yaml
paper_name: str
gaps:                    # 列表
  - id: "GAP-01"
    path: str            # 点号路径（key_information 中的缺失字段）
    question: str
    severity: str        # high/medium/low
    current_value: str
    affect_modules: [str]
    suggested: str
    alternatives: [str]
summary: {total: int, by_severity: {high, medium, low}}
resolved_assumptions: [str]
```

`requirements_summary.yaml`（汇总，仅计数）：
```yaml
paper_name: str
generated_at: str
functional_requirements: {total, categories}
data_flow: {pipeline_stages, interfaces}
acceptance_criteria: {accuracy_targets, functional_tests, edge_cases, suggested_validation_runs}
gap_analysis: {total_gaps, by_severity, resolved_assumptions}
```

#### Step5/6 依赖哪些字段

- **FR 编号**：`functional_requirements.yaml` 的 `requirements[].id`（`FR-01`…）是贯穿下游的引用锚点。`data_flow.yaml` 的 `pipeline[].related_fr` 与 `acceptance_criteria.yaml` 的 `functional_tests[].related_fr` 都靠它把阶段/测试回溯到功能需求；Step5 生成代码时据此确定模块边界与函数接口。
- **数据流 stages/interfaces**：`data_flow.yaml` 的 `pipeline`（阶段顺序、`entry_data`/`exit_data`）与 `interfaces`（`IF-01`… 的 `format`/`shape`/`contract`/`validation`）是 Step5 确定模块间数据契约、张量 shape、断言与训练循环（`training_loop`）的直接依据。
- **验收测试**：`acceptance_criteria.yaml` 的 `accuracy_targets.primary.target/tolerance`（精度目标）、`functional_tests`（测试步骤与 `related_fr`）、`edge_cases`、`reproducibility.suggested_validation_runs` 是 Step6 判定「复现成功 / 需迭代修复」的停止条件。
- **歧义清单**：`gap_analysis.yaml` 的 `gaps[].path/severity/affect_modules/suggested` 指引 Step5/6 在缺参数时如何兜底（默认超参、人工决策点），`resolved_assumptions` 记录已做的推断。
