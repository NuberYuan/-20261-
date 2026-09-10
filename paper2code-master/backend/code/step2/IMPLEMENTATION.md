# Step 2：JSON 清洗 / 修复 / 结构检测

## 思路

### 三个子步各自目标与关系

Step2 是 paper2code 流水线中“PDF 已解析为结构化 JSON 之后、送入 LLM 做关键信息提取之前”的纯规则预处理阶段，全程无 LLM 调用，负责把 Step1 直接抓取、噪声较多的段落文本，整理成 Step3 可稳定消费的形态。三个子步是严格串行的：上游输出 = 下游输入。

- **step2_1（清洗 cleaner）**：去掉 PDF 提取的噪声段落（页码、日期、纯数字、单字符符号），归一化空白，修复段落内部的单字母断裂，并对重复段落去重。
- **step2_2（修复 repairer）**：修复 PDF 常见的连字（ligature，如 `ﬁ`→`fi`）、断词连字符，并把被切成多个 block 的单词重新合并（`M` + `odern` → `Modern`，`threat-` + `ened` → `threatened`）。
- **step2_3（结构检测 structure_detector）**：给每个段落打上 `type`（title / author / abstract / keywords / heading / paragraph）和 `confidence` 标签，供下游按角色取用。

三者职责递进：先“删错”，再“修错”，最后“贴标签”。清洗和修复都只改 `paragraphs` 列表，`images` / `tables` 原样透传。

### 输入 / 输出

| 子步 | 输入文件 | 输出文件 | 主要动作 |
|------|----------|----------|----------|
| step2_1 | `step1/raw_paper.json` | `step2/cleaned_paper.json` | 去噪、归一化、去重、段内单字母合并 |
| step2_2 | `step2/cleaned_paper.json` | `step2/repaired_paper.json` | 连字替换、连字符修复、跨 block 合并 |
| step2_3 | `step2/repaired_paper.json` | `step2/structure_detect_paper.json` | 每段标注 `type` + `confidence` |

输入 JSON 的顶层结构由 Step1 的 `PDFParser.parse()` 决定：

```json
{
  "paragraphs": [
    {"id": 1, "block_id": 3, "page": 1, "bbox": [x0,y0,x1,y1], "type": "unknown", "text": "..."}
  ],
  "images": [...],
  "tables": [...]
}
```

三个子步都不改变 `paragraphs` 每个元素除 `text`（以及 step2_3 新增的 `type`/`confidence`）之外的字段，`id`、`block_id`、`page`、`bbox` 全程保留。

### 核心流程

1. `step2_1` 读入 `raw_paper.json`，对每个 paragraph 依次：`_clean_text`（换行转空格、压缩空白）→ `_merge_single_letter`（段内 `M odern`→`Modern`）→ 空文本丢弃 → 日期/页码/单字符噪声过滤 → 按前 80 字符去重 → 写回。
2. `step2_2` 读入 `cleaned_paper.json`，对每个 block：`_fix_ligature`（连字）→ `_fix_internal_hyphen`（连字符）→ 若当前 block 是单字母、或当前以 `-` 结尾且下一 block 以小写开头，则跨 block 合并。
3. `step2_3` 读入 `repaired_paper.json`，对每个 paragraph 按优先级分类：位置 0 → title；机构关键词/逗号启发式 → author；`|` 或 `Keywords` 开头 → keywords；首页前 5 段且 >500 字 → abstract；全大写/数字编号/标题词 → heading；其余 → paragraph。

### 关键设计取舍与假设

- **纯规则、零 LLM**：为保证低成本与确定性，全部用正则 + 启发式，不用模型。代价是对非标准排版（如双栏、无摘要、前置封面页、超长标题页）的鲁棒性有限。
- **“删错优于修错”的分工**：清洗阶段用保守的白名单式噪声判定（整段匹配），修复阶段才做有损的字符级替换，尽量把有损操作限制在明确的断裂场景。
- **段落而非行的粒度**：Step1 已把文本聚合为 block（段落）级，Step2 不再处理行级信息，`bbox`/`page` 仅作为结构检测的辅助特征。
- **位置优先假设**：结构检测强依赖“第一个段落 = 标题”“摘要在前 5 段且首页”等位置假设，隐式假设输入是规范的英文学术论文（有标题、作者、摘要的线性排版）。
- **下游真正消费的是 `text`**：Step3 的 `InformationExtractor.extract()` 只读取 `paragraphs[].text` 并拼接成全文，当前并**没有**消费 step2_3 打出的 `type` 字段（`type` 仅用于 step2_3 内部的分布日志统计）。这意味着结构检测目前是“计算了但下游未使用”的能力储备，标题/摘要等角色仍由 LLM 自行理解。

## 具体实现

### 文件清单

- `cleaner.py` — `JSONCleaner` 类，噪声清洗、文本归一化、去重、段内单字母合并。
- `repairer.py` — `JSONRepairer` 类，连字替换、连字符修复、跨 block 断词合并。
- `structure_detector.py` — `StructureDetector` 类，段落角色分类与置信度标注。
- `step2_1_clean_json.py` — step2_1 入口脚本，`raw_paper.json` → `cleaned_paper.json`。
- `step2_2_repair_json.py` — step2_2 入口脚本，`cleaned_paper.json` → `repaired_paper.json`。
- `step2_3_detect_structure.py` — step2_3 入口脚本，`repaired_paper.json` → `structure_detect_paper.json`。
- `__init__.py` — 包声明（`# step2 Package`）。

### 关键类 / 函数

#### `JSONCleaner`（cleaner.py）

- `clean(paper: dict) -> dict` — 主入口。遍历 `paper["paragraphs"]`，逐个清洗/过滤/去重，写回 `paper["paragraphs"]`。输入输出均为顶层 `paper` dict；`images`/`tables` 不动。
- `_clean_text(text) -> str` — 换行转空格、`\s+` 压缩为单空格、`strip()`。
- `_merge_single_letter(text) -> str` — 正则 `\b([A-Z]) ([a-z]+)` 把大写单字母与紧随的小写词合并（`M odern`→`Modern`）。**注意**：会误伤独立的 `A`/`I`（如 `I think`→`Ithink`）及 `X axis`→`Xaxis`，见 bug 清单。
- `_is_date_noise(text) -> bool` — 命中“月份+日期+年份”“compiled/received/accepted/published/submitted+年份”等即判为噪音，整段删除。
- `_is_page_number_noise(text) -> bool` — 整段为纯数字、`Page N`、`- N -`、`N / M` 时判为页码噪音。
- `_is_noise(text) -> bool` — 空文本、单字符非字母判为噪音。
- 去重逻辑在 `clean()` 内联：取 `text[:80]` 作为 key，重复段落丢弃。

#### `JSONRepairer`（repairer.py）

- `repair(paper) -> dict` — 主入口。兼容 `paper["paragraphs"]`、`paper["blocks"]` 或纯 list 三种形态，分别调用 `_repair_blocks`。
- `_repair_blocks(blocks: list) -> list` — 遍历 block，做基础修复后，判断跨 block 断裂（单字母断裂 / `-` 结尾的单词断裂）并合并。
- `_fix_ligature(text) -> str` — 替换 `ﬃ/ﬄ/ﬁ/ﬂ/ﬀ` 为 `ffi/ffl/fi/fl/ff`。
- `_fix_internal_hyphen(text) -> str` — 正则 `-[\s ]*([a-z]{2,})` 去除连字符。**注意**：会误伤 `cross-entropy`→`crossentropy`、`state-of-the-art`→`stateoftheart` 等复合词，见 bug 清单。
- `_get_text(block) -> str` — 从 dict 中取 `text` 并 strip，非 dict 返回空串。
- `_is_single_letter(text) -> bool` — 整段是否恰为一个英文字母。
- `_is_word_break(text1, text2) -> bool` — `text1` 以 `-` 结尾、长度≥3、`text2` 以小写字母开头。
- `_merge_blocks(block1, block2, remove_hyphen, join_direct) -> dict` — 合并两个 block，`remove_hyphen` 去掉 `-` 直接拼接，`join_direct` 无空格拼接，否则空格拼接；返回 `block1.copy()`（仅保留 block1 的元数据）。

#### `StructureDetector`（structure_detector.py）

- `detect(paper) -> dict` — 主入口。若无 `paper["paragraphs"]` 直接原样返回；否则逐段调用 `_classify`，写入 `type`/`confidence`，并 `pop("section_hint", None)`（该字段实际从未由上游产生，属死代码）。
- `_classify(text, index, page, bbox) -> tuple` — 分类主逻辑，按“title → author → keywords → abstract → heading → paragraph”优先级返回 `(type, confidence)`。
- `_is_affiliation(text) -> bool` — 机构关键词命中 ≥2 个，或“长度<200 且含逗号且不含日期且非技术内容”判定为作者机构。
- `_is_not_author(text) -> bool` — 排除子弹点/`Figure`/技术关键词/英文虚词开头或包含的短文本。
- `_is_keywords(text) -> bool` — 含 `|` 或以 `keywords` 开头。
- `_is_abstract(text, index, page) -> bool` — `page == 1 and index <= 5 and len(text) > 500`。
- `_is_heading(text, length) -> bool` — 长度≤80 且（全大写 / 数字编号开头 / 标题词等值或前缀匹配）。
- `_contains_date(text) -> bool` — 正则匹配 `\d{1,2},\s\d{4}` 或 `\b\d{4}\b`。

### 数据流与相邻 Step 衔接

```
Step1 (step1/raw_paper.json)
   │  paragraphs[{id, block_id, page, bbox, type, text}], images, tables
   ▼
step2_1 (step2/cleaned_paper.json)
   │  paragraphs 数量减少（去噪/去重），text 被归一化；
   │  images/tables 不变；字段结构不变
   ▼
step2_2 (step2/repaired_paper.json)
   │  连字被替换、连字符被处理、断裂 block 被合并（数量可能进一步减少）；
   │  images/tables 不变；字段结构不变
   ▼
step2_3 (step2/structure_detect_paper.json)
   │  每段新增 type + confidence，移除 section_hint；
   │  images/tables 不变
   ▼
Step3 (step3/key_information.json)
```

**schema 变化要点**：

- 三个子步均以 `paper` dict 为整体单元流转，`images`、`tables` 无任何改动，只有 `paragraphs` 被改写。
- 段落元素的 `id` 不会随删除/合并重排（保留 Step1 原始 id），因此去重后 id 可能不连续；Step3 不依赖 id 连续性。
- step2_2 合并两个 block 时，新段落继承 `block1` 的全部元数据（`id`/`bbox`/`page`），`block2` 的元数据被丢弃，合并后的 `bbox` 不覆盖 `block2` 区域。

**Step3 依赖的字段**（来自 `step3/step3_extract_information.py` 与 `step3/information_extractor.py`）：

- `paper["paragraphs"]`：`InformationExtractor.extract()` 逐段读 `p["text"]` 拼接为全文送入 LLM（`information_extractor.py:41-49`）。
- `paper["paragraphs"][0]["text"]`：作为论文标题/上下文，传给 `TableAnalyzer`（`step3_extract_information.py:92`）。
- `paper["tables"]`：表格分析（含 `table["page"]`）。
- `paper["images"]`：图片分析，要求每项有 `img["path"]`，并读取 `id`、`page`。
- **`type` / `confidence` 字段当前不被 Step3 消费**——结构检测的结果只是被记录，标题/摘要/方法等角色仍交给 LLM 从拼接后的全文里自行提取。若后续要让 Step3 按角色分段，只需读取 `paragraphs[].type` 即可，schema 已具备该能力。

> 注意：Step3 在 `step3_extract_information.py:92` 用 `paper["paragraphs"][0]` 当标题，因此 step2_1/step2_2 若把第一段（可能是期刊页眉、DOI 等未清理掉的噪声）保留下来，会直接污染标题上下文。
