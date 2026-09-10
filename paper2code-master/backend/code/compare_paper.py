"""Compare paper text against key_information.json to find missing data.

通用化改造：不再硬编码某一篇论文的专有事实（作物种类、具体模型层、准确率数值等），
而是直接从 key_information.json 的字段结构出发，做两类回查：

1. NOT_FOUND 缺口回查 —— 找出所有 LLM 标记为 NOT_FOUND 的字段，
   并用「字段名」作为关键词回到论文原文搜索，判断是否真的缺、能否补上。
2. 已提取事实回查 —— 把 key_information.json 里已提取到的短事实（数值、模型名等），
   回原文核对是否真的出现在论文里，识别潜在的推断/幻觉值。

用法:
    python compare_paper.py            # 使用 config.DEFAULT_PAPER
    python compare_paper.py <paper>    # 指定论文名（对应 data/output/<paper>/）
"""
import json
import re
import sys

from config import DEFAULT_PAPER, get_paper_paths

# ---- 论文名：命令行参数 > 默认值 ----
PAPER = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_PAPER
paths = get_paper_paths(PAPER)

with open(paths["structure_detect"], "r", encoding="utf-8") as f:
    paper = json.load(f)

with open(paths["key_information"], "r", encoding="utf-8") as f:
    ki = json.load(f)

paragraphs = paper.get("paragraphs", [])
full = " ".join(p.get("text", "") for p in paragraphs)

if not paragraphs:
    print("[WARN] structure_detect_paper.json 无 'paragraphs'，原文回查将无结果。")

# ============================================================
# 工具
# ============================================================

def iter_leaves(obj, path=""):
    """递归展开为 (path, value) 叶节点列表。"""
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from iter_leaves(v, f"{path}.{k}" if path else k)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from iter_leaves(v, f"{path}[{i}]")
    else:
        yield path, obj


def is_not_found(value) -> bool:
    s = str(value).strip()
    return s == "NOT_FOUND" or "NOT_FOUND" in s


def leaf_key(path: str) -> str:
    """路径最后一段字段名（去掉列表下标）。"""
    return re.sub(r'\[\d+\]$', '', path.split(".")[-1])


def key_phrases(path: str) -> list:
    """由字段名生成回查关键词。

    number_of_classes -> ['number of classes', 'number', 'classes']
    """
    words = [w for w in re.split(r'[^A-Za-z0-9]+', leaf_key(path)) if w]
    phrases = []
    if words:
        phrases.append(" ".join(words))
    phrases.extend(w for w in words if len(w) > 2)
    return phrases


def search_terms(value) -> list:
    """从值里提取可回查原文的候选词：数值 + 模型/框架/数据集名等专名。"""
    s = str(value)
    terms = []
    # 尺寸 256x256、百分比 99.35%、纯数值 0.005 / 100 / 38
    for m in re.finditer(r'\d+\s*x\s*\d+', s):
        terms.append(re.sub(r'\s+', '', m.group(0)))
    for m in re.finditer(r'\d+(?:\.\d+)?\s*%', s):
        terms.append(re.sub(r'\s+', '', m.group(0)))
    for m in re.finditer(r'\d+(?:\.\d+)?', s):
        n = m.group(0)
        if len(n) >= 2 or '.' in n:  # 过滤 0-9 单数字噪声
            terms.append(n)
    # 专名/缩写：AlexNet、GoogLeNet、ImageNet、SGD、Caffe 等
    # 仅对「无空白的单 token 值」提取，避免从整句描述里误提句首大写词。
    if " " not in s and "\n" not in s:
        for m in re.finditer(r'[A-Z][A-Za-z0-9]{2,}', s):
            terms.append(m.group(0))
    return list(dict.fromkeys(terms))  # 去重保序


# 通用字段名，回查无意义，跳过搜索（但仍计入 NOT_FOUND 清单）
GENERIC_KEYS = {
    "note", "description", "detail", "default", "value", "type", "name",
    "method", "source", "status", "path", "comment", "summary", "reason",
    "basis", "fallback", "alternatives", "choice", "config_key", "condition",
    "data", "format", "shape", "contract", "validation",
}

# ============================================================
# 1. NOT_FOUND 缺口回查
# ============================================================
print("=" * 60)
print(f"  PAPER: {PAPER}")
print("=" * 60)

not_found = [(p, v) for p, v in iter_leaves(ki) if is_not_found(v)]

print(f"\n  NOT_FOUND fields: {len(not_found)}\n")
for path, value in not_found:
    print(f"    {path}")

print("\n" + "=" * 60)
print("  GAP CHECK: can these NOT_FOUND fields be found in the paper text?")
print("=" * 60)

hittable = 0
hit_count = 0
for path, value in not_found:
    if leaf_key(path).lower() in GENERIC_KEYS:
        continue
    hittable += 1
    hits = []
    for phrase in key_phrases(path):
        for p in paragraphs:
            text = p.get("text", "")
            if phrase.lower() in text.lower():
                idx = text.lower().find(phrase.lower())
                start = max(0, idx - 40)
                end = min(len(text), idx + len(phrase) + 80)
                snippet = text[start:end].replace("\n", " ")
                hits.append((phrase, p.get("page", "?"), snippet))
                break  # 每个 phrase 每个 paragraph 只取一次
    if hits:
        hit_count += 1
        print(f"\n  [{path}]  <-- 可能可在原文找到:")
        for phrase, page, snippet in hits:
            print(f"    [p{page}] ...{snippet}...")

print(f"\n  (searchable NOT_FOUND fields: {hittable}, with hits: {hit_count})")

# ============================================================
# 2. 已提取事实回查（grounding）
# ============================================================
print("\n" + "=" * 60)
print("  FACT GROUNDING: KI facts present in the paper text?")
print("=" * 60)

grounded = 0
ungrounded = []

for path, value in iter_leaves(ki):
    if is_not_found(value):
        continue
    s = str(value).strip()
    if not s or len(s) > 80:  # 只回查「短事实」，跳过长篇描述
        continue
    terms = search_terms(s)
    if not terms:
        continue
    missing = [t for t in terms if t.lower() not in full.lower()]
    if missing:
        ungrounded.append((path, s, missing))
    else:
        grounded += 1

print(f"\n  grounded (all facts found in paper): {grounded}")
print(f"  not-in-paper (possible inference/hallucination): {len(ungrounded)}\n")

for path, value, missing in ungrounded:
    print(f"    [{path}] {value!r}  ->  missing: {', '.join(missing)}")
