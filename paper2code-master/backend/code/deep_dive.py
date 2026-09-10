"""Deep dive into what data Step3's LLM misses."""
import json, re, sys

# 论文名：默认 plant，可通过命令行参数覆盖
PAPER = sys.argv[1] if len(sys.argv) > 1 else "plant"

with open(f"data/output/{PAPER}/step2/structure_detect_paper.json", "r", encoding="utf-8") as f:
    paper = json.load(f)

with open(f"data/output/{PAPER}/step3/key_information.json", "r", encoding="utf-8") as f:
    ki = json.load(f)

# Build the paper text exactly as information_extractor does
paragraphs = paper.get("paragraphs", [])
text_parts = []
for p in paragraphs:
    text = p.get("text", "")
    if text.strip():
        text_parts.append(text)

paper_text = "\n".join(text_parts)
max_text_len = 44000  # 与 information_extractor.py 保持一致

# 复现 information_extractor 的「前 75% + 后 25%」截断策略
did_truncate = len(paper_text) > max_text_len
head_len = int(max_text_len * 0.75)
tail_len = max_text_len - head_len
sent_text = (
    paper_text[:head_len]
    + "\n\n... [中间内容已截断] ...\n\n"
    + paper_text[-tail_len:]
) if did_truncate else paper_text

print(f"Total paper text: {len(paper_text)} chars")
print(f"Text sent to LLM: {len(sent_text)} chars")
print(f"Truncated: {max(0, len(paper_text) - max_text_len)} chars")

# ============================================================
# 1. What's in the middle of the paper (the dropped part)?
# ============================================================
if did_truncate:
    dropped = paper_text[head_len : len(paper_text) - tail_len]
    print(f"\n{'='*60}")
    print("  DROPPED (MIDDLE) TEXT (first ~3000 chars)")
    print(f"{'='*60}")
    print(dropped[:3000])
    if len(dropped) > 3000:
        print(f"\n... ({len(dropped) - 3000} more chars)")

# ============================================================
# 2. What is the LLM getting in the last 2000 chars of the prompt?
# ============================================================
print(f"\n{'='*60}")
print("  LAST 2000 CHARS SENT TO LLM")
print(f"{'='*60}")
print(sent_text[-2000:])

# ============================================================
# 3. Check important_parameters quality
# ============================================================
print(f"\n{'='*60}")
print("  IMPORTANT_PARAMETERS QUALITY CHECK")
print(f"{'='*60}")
imp = ki.get("reproduction_requirements", {}).get("important_parameters", {})
for k, v in imp.items():
    is_dup = sum(1 for k2 in imp if k != k2 and v == imp[k2]) > 0
    dup_mark = " (DUPLICATE)" if is_dup else ""
    print(f"  {k}: {v}{dup_mark}")

# ============================================================
# 4. Count NOT_FOUND entries
# ============================================================
print(f"\n{'='*60}")
print("  NOT_FOUND FIELDS IN KI")
print(f"{'='*60}")

def find_not_found(obj, path=""):
    results = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            new_path = f"{path}.{k}" if path else k
            if v == "NOT_FOUND" or (isinstance(v, str) and v.startswith("NOT_FOUND")):
                results.append(new_path)
            elif isinstance(v, (dict, list)):
                results.extend(find_not_found(v, new_path))
    elif isinstance(obj, list):
        for i, item in enumerate(obj):
            results.extend(find_not_found(item, f"{path}[{i}]"))
    return results

not_found = find_not_found(ki)
if not_found:
    for nf in not_found:
        print(f"  {nf}")
else:
    print("  (none!)")
print(f"\n  Total: {len(not_found)} NOT_FOUND entries")

# ============================================================
# 5. Check image_analysis quality
# ============================================================
img_analysis = ki.get("image_analysis", [])
failed = [i for i in img_analysis if "无法确定" in i.get("description", "") or "failed" in i.get("description", "")]
print(f"\n{'='*60}")
print("  IMAGE ANALYSIS QUALITY")
print(f"{'='*60}")
print(f"  Total images analyzed: {len(img_analysis)}")
print(f"  Failed analyses: {len(failed)}")
print(f"  Success rate: {(len(img_analysis)-len(failed))/max(1,len(img_analysis))*100:.0f}%")
if failed:
    print(f"  Failed image IDs: {[i['id'] for i in failed]}")
