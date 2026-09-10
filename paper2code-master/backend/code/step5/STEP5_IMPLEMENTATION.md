# Step5 实现总结：论文复现项目分析与代码工程设计准备

## 一、定位

Step5 是 **Paper → Code** 流程中的翻译层：

```
Step3: 论文讲了什么 (key_information.json)
        │
Step5: 怎么用代码复现 (工程设计 YAML)
        │
Step6: 生成可执行代码 (.py 文件)
```

核心任务：把 Step3 的论文自然语言信息，转化为 Step6 代码 Agent 可理解的结构化工程规格。

---

## 二、整体架构

### 设计原则

| 原则 | 说明 |
|------|------|
| **不修改 Step1-3** | Step5 是独立模块，只读取 `data/output/{论文}/` 下的 JSON |
| **三层防编造** | Layer1 规则提取 → Layer2 LLM 补全 → Layer3 后验证 |
| **NOT_FOUND 约定** | 论文未提及的信息统一标记 `NOT_FOUND`，绝不编造 |
| **回溯原始文本** | 同时读取 `key_information.json` 和 `structure_detect_paper.json`，交叉验证 |

### 三层防编造机制

```
Layer 1: 规则提取 (确定性，不调 LLM)
    ↓  从 key_information.json 中用代码直接提取已知事实
    ↓  例如: dataset.number_of_classes → "38"
    ↓  100% 可靠，不经过 LLM
    ↓
Layer 2: 约束型 LLM 补全
    ↓  只对规则无法覆盖的 NOT_FOUND 字段使用 LLM
    ↓  Prompt 严格限定: "只能从论文原文提取，不确定就填 NOT_FOUND"
    ↓
Layer 3: 后验证
    ↓  将 LLM 输出的概念词与原始论文文本比对
    ↓  标记无法验证的内容
```

---

## 三、子步骤说明

### 5.1 复现目标分析 → `reproduction_goal.yaml`

**文件**: `step5_1_goal.py`

| 字段 | 来源 | 方式 |
|------|------|------|
| `paper_type` | 综合判断 | 规则：检查 has_training + has_new_model + 关键词 |
| `task` | 论文标题/摘要 | 规则：词边界匹配（LSTM→时序, image→分类, code→代码分析） |
| `input/output` | methodology.model_architecture | 规则 |
| `is_trainable` | implementation_details | 规则：optimizer/lr/epoch/loss 是否有值 |
| `reproduction_targets` | 论文类型 + 训练状态 | 规则模板 |

**论文类型判定逻辑**:
- `is_code_analysis` → `empirical_comparison`
- `has_new_model and has_training` → `model_proposing`
- 其他 → `empirical_comparison`

**任务类型优先级**: 时序预测 > 代码分析 > 目标检测 > 分割 > 图像分类 > NLP > other

---

### 5.2 技术栈分析 → `technology.yaml`

**文件**: `step5_2_tech.py`

| 字段 | 来源 | 方式 |
|------|------|------|
| `framework` | 复现代码固定 PyTorch | 规则：recommended 固定 PyTorch；论文原框架记录在 programming_framework_mentioned |
| `model_backbone` | methodology.model | 规则：解析逗号分隔 |
| `optimizer/loss` | implementation_details | 规则 |
| `pretrained_weights` | model_architecture.pretrained_model | 规则 |

**框架字段**：`framework.recommended` 固定为 `PyTorch`（代码生成器为 PyTorch 专用，config.yaml 固定写 `framework: pytorch`），
`framework.candidates` 固定为 `["PyTorch"]`；论文原文使用的框架（如 Caffe）作为事实记录在
`programming_framework_mentioned`，供 Step6 生成「框架偏离」假设（论文用 X，代码改用 PyTorch）。

---

### 5.3 数据集分析 → `dataset.yaml`

**文件**: `step5_3_data.py`

全规则提取，从 `key_information.json` 的 `dataset` 字段直接映射。

---

### 5.4 模型结构解析 → `model_design.yaml`

**文件**: `step5_4_model.py`

| 字段 | 来源 | 方式 |
|------|------|------|
| `models[].name` | methodology.model | 规则：解析逗号分隔 |
| `models[].type` | 名称关键词 | 规则：CNN/RNN/VLM/MLP 分类 |
| `models[].code_class_name` | 模型名 | 规则：去空格、去特殊字符 |

**模型类型推断**: VLM关键词 > RNN关键词 > CNN关键词 > MLP > other

---

### 5.5 实验流程设计 → `experiment_design.yaml`

**文件**: `step5_5_exp.py`

| 字段 | 来源 |
|------|------|
| `pipeline` | experiment.technical_pipeline → methodology.pipeline → 默认模板 |
| `primary_metric` | experiment.evaluation_metrics 中匹配 F1/Accuracy/R²/RMSE 等 |
| `baselines/ablations` | experiment 字段 |

---

### 5.6 项目文件架构 → `project_structure.yaml`

**文件**: `step5_6_struct.py`

根据论文类型使用不同模板：

| 论文类型 | 模板特点 |
|---------|---------|
| `model_proposing` | evaluate.py |
| `empirical_comparison` | evaluate.py + compare.py |

模型文件名从 5.4 的 `model_design.yaml` 动态生成。

---

### 5.7 参数配置 → `training_config.yaml`

**文件**: `step5_7_train.py`

从 `implementation_details` 和 `reproduction_requirements.important_parameters` 提取：

- batch_size, epochs
- optimizer (name, lr, momentum, weight_decay)
- scheduler (StepLR/Cosine 自动推断)
- loss_function, early_stopping

---

### 5.8 代码规格 → `optimized_project_spec.yaml`

**文件**: `step5_8_spec.py`

**最关键的步骤**：汇总 5.1~5.7，调用 LLM 细化为代码级规格。

输出包含：
- `project`: 项目名、描述、论文类型
- `architecture.modules`: 每个模块的文件、类、函数
- `dependencies.files`: 所有代码文件列表
- `dependencies.external`: 外部依赖
- `execution_flow`: 执行流程（步骤序列）
- `code_requirements`: 每个 .py 文件的详细规格

**code_requirements 格式示例**:
```yaml
data/dataset.py:
  class: PlantDataset
  parent_class: torch.utils.data.Dataset
  methods: [__init__, __len__, __getitem__]
  functions: [load_data, preprocess]
  pseudocode: "class PlantDataset: __init__加载路径; __getitem__返回图像和标签"

models/alexnet.py:
  class: AlexNet
  parent_class: torch.nn.Module
  init_params: {num_classes: 38}
  forward_input: torch.Tensor (B, 3, 256, 256)
  forward_output: torch.Tensor (B, 38)

training/train.py:
  functions: [train_epoch, validate, main]
  uses: [models.*, data.dataset, utils.metrics]
```

---

### 聚合输出 → `project_analysis.yaml`

将 5.1~5.7 的 7 个 YAML 汇总为一个文件，给人阅读。

---

## 四、代码结构

```
code/step5/
├── __init__.py
├── base_analyzer.py          # 基类：三层防编造、JSON读取、YAML保存
├── step5_1_goal.py           # 5.1 复现目标
├── step5_2_tech.py           # 5.2 技术栈
├── step5_3_data.py           # 5.3 数据集
├── step5_4_model.py          # 5.4 模型设计
├── step5_5_exp.py            # 5.5 实验流程
├── step5_6_struct.py         # 5.6 项目结构
├── step5_7_train.py          # 5.7 训练参数
├── step5_8_spec.py           # 5.8 代码规格 (LLM)
├── step5_main.py             # 总入口
└── STEP5_IMPLEMENTATION.md   # 本文档
```

独立于 Step1-3，只读取 `data/output/{论文}/key_information.json` 和 `structure_detect_paper.json`。

---

## 五、运行方式

```bash
cd code

# 单篇论文
python -m step5.step5_main --paper AgriPath

# 全部论文（自动发现 data/output/ 下有 key_information.json 的目录）
python -m step5.step5_main --all

# 跳过 Step5.8 LLM（仅规则提取）
python -m step5.step5_main --all --skip-llm
```

---

## 六、输出文件

每篇论文的 `data/output/{论文名}/step5/` 目录下：

| 文件 | 步骤 | 受众 | 核心内容 |
|------|------|------|---------|
| `reproduction_goal.yaml` | 5.1 | 人+Agent | 论文类型、任务类型、复现目标 |
| `technology.yaml` | 5.2 | 人+Agent | 框架推荐、模型列表、优化器/损失函数 |
| `dataset.yaml` | 5.3 | 人+Agent | 数据集名称、大小、类别数、预处理 |
| `model_design.yaml` | 5.4 | Agent | 每个模型的 class 名、类型、输入/输出 |
| `experiment_design.yaml` | 5.5 | Agent | 实验 pipeline、主要指标、对比方法 |
| `project_structure.yaml` | 5.6 | Agent | 代码目录树、每个目录的文件列表 |
| `training_config.yaml` | 5.7 | Agent | batch_size、lr、epoch、optimizer 参数 |
| `project_analysis.yaml` | 聚合 | **人** | 5.1~5.7 全部汇总，方便阅读 |
| `optimized_project_spec.yaml` | 5.8 | **Agent** | 每个 .py 的 class/function/依赖/伪代码 |

---

## 七、当前处理状态

已处理 6 篇论文，全部有完整的 Step1→Step5 输出：

| 论文 | 类型 | 任务 | 图片 | Step5 |
|------|------|------|------|-------|
| plant (植物病害检测) | model_proposing | image_classification | 14 | ✅ |
| AgriPath | model_proposing | image_classification | 7 | ✅ |
| GPP-ML | model_proposing | time_series_prediction | 42 | ✅ |
| Human-AI-Code | empirical_comparison | code_analysis | 2 | ✅ |
| MtCro | model_proposing | genomic_prediction | 11 | ✅ |
| TimeCAP | model_proposing | time_series_prediction | 10 | ✅ |
