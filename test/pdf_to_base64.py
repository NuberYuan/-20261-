"""
PDF → Base64 图片转换工具

将 PDF 每页渲染为 JPEG 图片，再编码为 base64 字符串。
支持自动压缩以满足 50MB 上限。

用法:
    from pdf_to_base64 import pdf_page_to_base64, pdf_all_pages_to_base64

    # 单页
    b64 = pdf_page_to_base64("paper.pdf", page=0)

    # 全部页
    pages = pdf_all_pages_to_base64("paper.pdf")
"""

import base64
import io
import sys
from pathlib import Path

try:
    import fitz  # PyMuPDF
except ImportError:
    print("需要安装 PyMuPDF: pip install PyMuPDF")
    sys.exit(1)

try:
    from PIL import Image
except ImportError:
    print("需要安装 Pillow: pip install Pillow")
    sys.exit(1)

from config import MAX_BASE64_SIZE, DEFAULT_DPI, DEFAULT_QUALITY, DEFAULT_MAX_WIDTH


def _render_page(pdf_path: str, page: int, dpi: int) -> Image.Image:
    """将 PDF 指定页渲染为 PIL Image"""
    doc = fitz.open(pdf_path)
    if page < 0 or page >= len(doc):
        raise ValueError(f"页码 {page} 超出范围 (共 {len(doc)} 页)")
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    pix = doc[page].get_pixmap(matrix=mat)
    img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    doc.close()
    return img


def _resize_if_needed(img: Image.Image, max_width: int) -> Image.Image:
    """宽度超限时等比缩放"""
    if img.width <= max_width:
        return img
    ratio = max_width / img.width
    new_size = (max_width, int(img.height * ratio))
    return img.resize(new_size, Image.LANCZOS)


def _img_to_base64(img: Image.Image, quality: int) -> str:
    """PIL Image → JPEG → base64 字符串"""
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality, optimize=True)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _estimate_size(img: Image.Image, quality: int) -> int:
    """估算编码后的 base64 长度（字节）"""
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality, optimize=True)
    # base64 编码后约 4/3 倍
    return int(len(buf.getvalue()) * 4 / 3)


def pdf_page_to_base64(
    pdf_path: str,
    page: int = 0,
    dpi: int = DEFAULT_DPI,
    quality: int = DEFAULT_QUALITY,
    max_width: int = DEFAULT_MAX_WIDTH,
) -> str:
    """
    将 PDF 单页转为 base64 JPEG 字符串。

    如果超过 50MB 上限，自动降低 DPI / 质量 / 宽度。

    Args:
        pdf_path: PDF 文件路径
        page: 页码（从 0 开始）
        dpi: 渲染分辨率
        quality: JPEG 压缩质量 (1-100)
        max_width: 最大宽度（像素）

    Returns:
        base64 编码的 JPEG 字符串
    """
    img = _render_page(pdf_path, page, dpi)
    img = _resize_if_needed(img, max_width)

    # 自动压缩策略：逐步降质量 → 降宽度 → 降 DPI
    current_quality = quality
    current_max_width = max_width
    current_dpi = dpi

    for attempt in range(10):
        size = _estimate_size(img, current_quality)
        if size <= MAX_BASE64_SIZE:
            return _img_to_base64(img, current_quality)

        print(f"  [压缩] 第{attempt+1}次: 估算 {size/1024/1024:.1f}MB > 50MB，降低参数...")

        if current_quality > 40:
            current_quality -= 10
        elif current_max_width > 800:
            current_max_width = int(current_max_width * 0.75)
            img = _resize_if_needed(img, current_max_width)
        elif current_dpi > 100:
            current_dpi -= 30
            img = _render_page(pdf_path, page, current_dpi)
            img = _resize_if_needed(img, current_max_width)
        else:
            break

    # 最终兜底
    return _img_to_base64(img, max(current_quality, 20))


def pdf_all_pages_to_base64(
    pdf_path: str,
    dpi: int = DEFAULT_DPI,
    quality: int = DEFAULT_QUALITY,
    max_width: int = DEFAULT_MAX_WIDTH,
) -> list[dict]:
    """
    将 PDF 全部页面转为 base64。

    Returns:
        [{"page": 0, "base64": "...", "size_kb": 123}, ...]
    """
    doc = fitz.open(pdf_path)
    total = len(doc)
    doc.close()

    results = []
    for i in range(total):
        print(f"转换第 {i+1}/{total} 页...")
        b64 = pdf_page_to_base64(pdf_path, page=i, dpi=dpi, quality=quality, max_width=max_width)
        results.append({
            "page": i,
            "base64": b64,
            "size_kb": len(b64) * 3 // 4 // 1024,
        })
    return results


def get_pdf_info(pdf_path: str) -> dict:
    """获取 PDF 基本信息"""
    doc = fitz.open(pdf_path)
    info = {
        "path": str(pdf_path),
        "pages": len(doc),
        "title": doc.metadata.get("title", ""),
        "author": doc.metadata.get("author", ""),
    }
    doc.close()
    return info


# ── CLI ──
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="PDF → Base64 转换")
    parser.add_argument("pdf", help="PDF 文件路径")
    parser.add_argument("--page", type=int, default=0, help="页码 (从0开始，默认0)")
    parser.add_argument("--all", action="store_true", help="转换全部页面")
    parser.add_argument("--dpi", type=int, default=DEFAULT_DPI)
    parser.add_argument("--quality", type=int, default=DEFAULT_QUALITY)
    parser.add_argument("--out", type=str, default=None, help="输出文件路径")
    args = parser.parse_args()

    info = get_pdf_info(args.pdf)
    print(f"PDF: {info['path']}")
    print(f"页数: {info['pages']}, 标题: {info['title']}")

    if args.all:
        pages = pdf_all_pages_to_base64(args.pdf, dpi=args.dpi, quality=args.quality)
        for p in pages:
            print(f"  第{p['page']+1}页: {p['size_kb']} KB")
        if args.out:
            import json
            Path(args.out).write_text(json.dumps(pages, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"已保存到 {args.out}")
    else:
        b64 = pdf_page_to_base64(args.pdf, page=args.page, dpi=args.dpi, quality=args.quality)
        size_kb = len(b64) * 3 // 4 // 1024
        print(f"第{args.page+1}页: {size_kb} KB")
        if args.out:
            Path(args.out).write_text(b64, encoding="ascii")
            print(f"已保存到 {args.out}")
        else:
            print(f"(前100字符: {b64[:100]}...)")