"""
MiMo-V2.6-Pro 图片识别性能测试

测试内容:
  1. 单页文字识别（OCR 能力）
  2. 图表描述（Figure 理解）
  3. 表格提取（Table 理解）
  4. 多页综合理解（长文档）

用法:
    python test_mimo_vision.py <pdf路径>
    python test_mimo_vision.py <pdf路径> --page 0 --task ocr
    python test_mimo_vision.py <pdf路径> --task all
"""

import argparse
import json
import sys
import time
import io
from pathlib import Path

# 修复 Windows 终端编码问题
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import requests

from config import API_URL, MODEL_NAME, API_KEY
from pdf_to_base64 import pdf_page_to_base64, get_pdf_info, pdf_all_pages_to_base64


# ──────────────────────────────────────────────
# 测试任务定义
# ──────────────────────────────────────────────

TASKS = {
    "ocr": {
        "name": "文字识别 (OCR)",
        "system": "你是一个精确的文字识别助手。请完整提取图片中的所有文字内容，保持原始排版。",
        "prompt": "请完整识别并提取这张图片中的所有文字内容。保持原文的段落结构和排版格式。",
    },
    "figure": {
        "name": "图表理解 (Figure)",
        "system": "你是一个学术论文图表分析专家。请详细分析图片中的图表内容。",
        "prompt": "请分析这张图片中的图表：\n1. 图表类型（折线图/柱状图/散点图/流程图/架构图等）\n2. 坐标轴含义\n3. 关键数据趋势\n4. 图表说明的结论",
    },
    "table": {
        "name": "表格提取 (Table)",
        "system": "你是一个精确的表格提取助手。请将图片中的表格转为结构化的 Markdown 格式。",
        "prompt": "请将这张图片中的表格完整提取为 Markdown 表格格式。保留所有行列、表头和数据。",
    },
    "summary": {
        "name": "页面摘要 (Summary)",
        "system": "你是一个学术论文阅读助手。请对论文页面进行精准摘要。",
        "prompt": "请用中文总结这页论文的核心内容，包括：\n1. 主要论点或方法\n2. 关键公式或算法\n3. 重要实验结果\n4. 与其他方法的对比（如有）",
    },
    "structure": {
        "name": "结构化提取 (Structure)",
        "system": "你是一个论文结构化信息提取专家。请按 JSON 格式提取关键信息。",
        "prompt": """请从这张论文页面中提取以下信息，以 JSON 格式输出：
{
  "title": "论文标题（如果可见）",
  "section": "当前章节名",
  "authors": "作者列表（如果可见）",
  "key_formulas": ["关键公式列表"],
  "key_terms": ["重要术语"],
  "figures_tables": ["图表编号列表"],
  "main_claims": ["主要论点"]
}
如果某项信息不存在，对应字段填 null。""",
    },
}


# ──────────────────────────────────────────────
# API 调用
# ──────────────────────────────────────────────

def call_mimo_vision(base64_str: str, system: str, prompt: str, max_tokens: int = 4096) -> dict:
    """
    调用 MiMo-V2.6-Pro Vision API。

    Returns:
        {"success": True, "response": "...", "latency_ms": 1234, "tokens": {...}}
        或 {"success": False, "error": "..."}
    """
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}",
    }

    body = {
        "model": MODEL_NAME,
        "max_completion_tokens": max_tokens,
        "messages": [
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_str}"
                        }
                    },
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
            }
        ]
    }

    start = time.time()
    try:
        resp = requests.post(API_URL, headers=headers, json=body, timeout=120)
        latency = int((time.time() - start) * 1000)

        if resp.status_code != 200:
            return {"success": False, "error": f"HTTP {resp.status_code}: {resp.text[:500]}"}

        data = resp.json()
        content = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {})

        return {
            "success": True,
            "response": content,
            "latency_ms": latency,
            "tokens": {
                "prompt": usage.get("prompt_tokens", 0),
                "completion": usage.get("completion_tokens", 0),
                "total": usage.get("total_tokens", 0),
            }
        }

    except requests.exceptions.Timeout:
        return {"success": False, "error": "请求超时 (120s)"}
    except Exception as e:
        return {"success": False, "error": str(e)}


# ──────────────────────────────────────────────
# 测试执行
# ──────────────────────────────────────────────

def run_single_test(pdf_path: str, page: int, task_key: str) -> dict:
    """执行单个测试任务"""
    task = TASKS[task_key]
    print(f"\n{'='*60}")
    print(f"测试: {task['name']}")
    print(f"PDF: {pdf_path}  第{page+1}页")
    print(f"{'='*60}")

    # 1. 转换图片
    print("转换 PDF → base64...")
    b64 = pdf_page_to_base64(pdf_path, page=page)
    size_kb = len(b64) * 3 // 4 // 1024
    print(f"图片大小: {size_kb} KB (base64: {len(b64)//1024} KB)")

    # 2. 调用 API
    print(f"调用 {MODEL_NAME}...")
    result = call_mimo_vision(b64, task["system"], task["prompt"])

    # 3. 输出结果
    if result["success"]:
        print(f"延迟: {result['latency_ms']}ms")
        print(f"Token: prompt={result['tokens']['prompt']}, completion={result['tokens']['completion']}")
        print(f"\n--- 回复 ---\n{result['response']}\n--- END ---")
    else:
        print(f"失败: {result['error']}")

    return {
        "task": task_key,
        "page": page,
        "image_size_kb": size_kb,
        **result,
    }


def run_all_tasks(pdf_path: str, page: int) -> list[dict]:
    """在指定页面上运行全部测试任务"""
    results = []
    for key in TASKS:
        r = run_single_test(pdf_path, page, key)
        results.append(r)
        time.sleep(1)  # 避免限流
    return results


def run_multi_page_summary(pdf_path: str, pages: list[int]) -> list[dict]:
    """多页连续摘要测试"""
    print(f"\n{'='*60}")
    print(f"多页连续摘要: 第 {[p+1 for p in pages]} 页")
    print(f"{'='*60}")

    results = []
    for page in pages:
        r = run_single_test(pdf_path, page, "summary")
        results.append(r)
        time.sleep(1)

    return results


# ──────────────────────────────────────────────
# 报告生成
# ──────────────────────────────────────────────

def generate_report(results: list[dict], output_path: str):
    """生成测试报告"""
    report_lines = [
        "# MiMo-V2.6-Pro 图片识别性能测试报告",
        "",
        f"- 模型: {MODEL_NAME}",
        f"- API: {API_URL}",
        f"- 测试时间: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"- 测试数量: {len(results)} 项",
        "",
        "## 测试结果汇总",
        "",
        "| 测试项 | 页码 | 图片大小 | 延迟 | Token | 状态 |",
        "|--------|------|----------|------|-------|------|",
    ]

    for r in results:
        status = "✅" if r.get("success") else "❌"
        task_name = TASKS.get(r["task"], {}).get("name", r["task"])
        size = f"{r.get('image_size_kb', 0)} KB"
        latency = f"{r.get('latency_ms', '-')}ms"
        tokens = r.get("tokens", {})
        token_str = f"{tokens.get('total', '-')}" if tokens else "-"
        report_lines.append(
            f"| {task_name} | {r.get('page', 0)+1} | {size} | {latency} | {token_str} | {status} |"
        )

    report_lines.extend(["", "## 详细结果", ""])

    for r in results:
        task_name = TASKS.get(r["task"], {}).get("name", r["task"])
        report_lines.append(f"### {task_name} (第{r.get('page', 0)+1}页)")
        report_lines.append("")
        if r.get("success"):
            report_lines.append(f"- 延迟: {r['latency_ms']}ms")
            tokens = r.get("tokens", {})
            report_lines.append(f"- Token: prompt={tokens.get('prompt',0)}, completion={tokens.get('completion',0)}")
            report_lines.append(f"- 图片大小: {r.get('image_size_kb',0)} KB")
            report_lines.append("")
            report_lines.append("```")
            report_lines.append(r.get("response", "")[:3000])
            report_lines.append("```")
        else:
            report_lines.append(f"- 错误: {r.get('error', '未知')}")
        report_lines.append("")

    report_text = "\n".join(report_lines)
    Path(output_path).write_text(report_text, encoding="utf-8")
    print(f"\n报告已保存: {output_path}")


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="MiMo-V2.6-Pro 图片识别测试")
    parser.add_argument("pdf", help="PDF 文件路径")
    parser.add_argument("--page", type=int, default=0, help="测试页码 (从0开始)")
    parser.add_argument("--task", type=str, default="all",
                        choices=list(TASKS.keys()) + ["all"],
                        help="测试任务 (默认 all)")
    parser.add_argument("--report", type=str, default="test_report.md",
                        help="报告输出路径")
    parser.add_argument("--multi", type=int, nargs="+", default=None,
                        help="多页摘要测试，指定页码列表 (如 --multi 0 1 2)")
    args = parser.parse_args()

    if not Path(args.pdf).exists():
        print(f"文件不存在: {args.pdf}")
        sys.exit(1)

    # 显示 PDF 信息
    info = get_pdf_info(args.pdf)
    print(f"PDF: {info['path']}")
    print(f"页数: {info['pages']}, 标题: {info['title']}")

    results = []

    if args.multi:
        results = run_multi_page_summary(args.pdf, args.multi)
    elif args.task == "all":
        results = run_all_tasks(args.pdf, args.page)
    else:
        results = [run_single_test(args.pdf, args.page, args.task)]

    if results:
        generate_report(results, args.report)


if __name__ == "__main__":
    main()