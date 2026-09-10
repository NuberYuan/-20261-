# Step 1：PDF → 原始 JSON

## 思路
- 目标：把一篇学术论文 PDF 解析成结构化的原始 JSON（`raw_paper.json`），为后续 Step2 的清洗/修复/结构检测提供输入。
- 输入 / 输出：
  - 输入：论文 PDF 文件（默认取 `data/input/` 下第一个 `.pdf`，见 `config.py::INPUT_PDF`）。
  - 输出：`data/output/<paper>/step1/raw_paper.json`，顶层含 `paragraphs` / `images` / `tables` 三个数组。
  - 副产品：`step1/images/*.png`（逐张提取的图片）、`step1/tables/extracted_tables.json` 及每个表格的 `.csv`。
- 核心流程（分阶段描述）：
  1. **加载 PDF**：`PDFParser.parse()` 用 PyMuPDF（`fitz`）打开 PDF。
  2. **逐页提取文本块**：对每页调用 `page.get_text("blocks")`，把每个非空 block 归一化为一个 paragraph（id 全局递增、block_id 页内编号、page 页码、bbox 坐标、type 暂为 `unknown`、text 经 `clean_text` 清洗）。
  3. **提取图片**：`ImageExtractor` 遍历每页 `page.get_images(full=True)`，用 `fitz.Pixmap` 把每个 xref 存成 PNG，记录 `{id, page, path}`。
  4. **提取表格**：`TableExtractor` 两阶段——先 `find_tables()` 拿带网格线的真表格，再用正则检测子弹点列表 / 编号列表等"类表格"文本；结果写入结构化 `{header, rows, method, bbox, ...}`。
  5. **写盘**：`JsonWriter.save()` 把整个 dict 以 UTF-8 落盘为 `raw_paper.json`。
- 关键设计取舍与假设：
  - 段落粒度取 PyMuPDF 的 block 级而非 span/word 级，牺牲精度换简单稳定，断词修复留给 Step2.2 处理。
  - 图片路径与表格副产物"就近"放在输出 JSON 同级的 `images/`、`tables/` 目录，避免多篇论文互相污染（`config.py` 里则是全局默认目录）。
  - 表格提取采用"网格表 + 结构化文本"两阶段，尽量覆盖无边框表格；结构化文本设了最小行数阈值（子弹点 ≥3、编号列表 ≥5）来抑制误报。
  - 假设 PDF 为文字型（非纯扫描件），且图片能被 PyMuPDF 直接读成 PNG；不含 OCR 兜底。

## 具体实现
### 文件清单
- `__init__.py` — Step1 包标记文件（无实际逻辑）。
- `pdf_parser.py` — `PDFParser` 主类：打开 PDF、逐页抽文本块，并调度图片/表格提取器。
- `image_extractor.py` — `ImageExtractor`：用 PyMuPDF 提取 PDF 内嵌图片并保存为 PNG。
- `table_extractor.py` — `TableExtractor`：两阶段提取网格表格与结构化文本，落盘 JSON + CSV。
- `text_utils.py` — `clean_text()`：换行/空格/空行归一化的文本清洗工具。
- `json_writer.py` — `JsonWriter`：把结果 dict 写入 UTF-8 JSON 文件（自动建目录）。
- `step1_pdf_to_json.py` — CLI 入口：argparse 解析参数，串联解析与写盘。

### 关键类 / 函数
- `PDFParser.__init__(pdf_path, images_dir=None, tables_dir=None)` — 保存 PDF 路径与图片/表格输出目录，目录为空时回退到 `config.py` 的全局目录。
- `PDFParser.parse() -> dict` — 主流程，返回 `{"paragraphs": [...], "images": [...], "tables": [...]}`；逐页抽取文本块，并调用图片/表格提取器。
- `ImageExtractor.extract(doc) -> list` — 输入 `fitz.Document`，遍历每页图片，逐张存 PNG，返回 `[{id, page, path}]`。
- `TableExtractor.extract(doc) -> list` — 遍历每页，先抽网格表再抽结构化文本，统一编号 `table_N`，有结果时调用 `_save()`。
- `TableExtractor._extract_grid_tables(page, page_no)` — 用 `page.find_tables()` 抽网格表，首行作 header、其余作 rows，过滤全空行，并附带 `_find_caption`。
- `TableExtractor._find_caption(page, table_bbox)` — 在表格上方 50pt 范围内寻找以 `Table`/`Tab.` 开头的标题文本。
- `TableExtractor._extract_structured_text(page, page_no)` — 按 y 排序文本块，依次检测子弹点组、编号列表。
- `TableExtractor._find_bullet_group(blocks)` / `_find_numbered_list(blocks)` — 用正则识别连续子弹点/编号行，尽量解析成 `key: value` 或 `No./Content` 两列。
- `TableExtractor._save(tables)` — 写 `extracted_tables.json`（去掉 bbox），并为每个表格写一个 CSV。
- `clean_text(text) -> str` — 统一 `\r`→`\n`、压缩连续空格、合并 3+ 连续空行，最后 strip。
- `JsonWriter.save(paper, output_path)` — 建目录后以 `ensure_ascii=False`、`indent=4` 写 UTF-8 JSON。
- `step1_pdf_to_json.py::main()` — argparse 入口，`--pdf` / `--out` / `--images-dir` / `--tables-dir`，图片表格默认放到输出 JSON 同级。

### 数据流与相邻 Step 衔接
- 产出的 JSON 结构要点（顶层字段）：
  - `paragraphs[]`：`id`（全局递增）、`block_id`（页内 block 编号）、`page`（1 起）、`bbox`（`[x0,y0,x1,y1]`）、`type`（Step1 恒为 `"unknown"`）、`text`（清洗后文本）。
  - `images[]`：`id`（`image_N`）、`page`、`path`（图片绝对/相对路径）。
  - `tables[]`：`id`（`table_N`）、`page`、`method`（`grid | bullet_list | numbered_list`）、`caption`、`header`、`rows`、`bbox`、`row_count`、`col_count`。
- Step2 依赖哪些字段：
  - **Step2.1 cleaner** 只消费 `paragraphs`（`paper.get("paragraphs", [])`），读取每项 `text` 做清洗、去噪、去重；不读 `images`/`tables`。
  - **Step2.2 repairer** 消费 `paragraphs[].text` 做连字/断词/单字母合并（也兼容旧的 `blocks` 字段）。
  - **Step2.3 structure_detector** 消费 `paragraphs[].text`、`.page`、`.bbox`，覆盖写 `.type`、新增 `.confidence`、删除 `.section_hint`。
  - `images` 与 `tables` 在 Step2 中只被透传保留，尚未被下游消费。
