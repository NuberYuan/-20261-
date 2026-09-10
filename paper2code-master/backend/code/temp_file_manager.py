"""
临时文件管理核心模块

管理 data/output/{paper}/ 下的生成物（一次运行 = 一篇论文的完整输出，含 step1~step8）。
- list / summary / detail：列出各次运行生成的完整输出、文件数与占用空间
- keep：保留原位（不移动、不删除，仅确认保留）
- discard：丢弃（删除该次运行的完整输出目录，释放磁盘空间）

纯 Python、零第三方依赖，可被前端通过子进程直接调用，
也可由 temp_file_api.py 包装成 HTTP 接口供 Web 前端调用。

Usage (CLI):
    cd code
    python -m temp_file_manager list
    python -m temp_file_manager summary
    python -m temp_file_manager detail  --paper plant
    python -m temp_file_manager keep    --paper plant
    python -m temp_file_manager discard --paper plant
"""

import argparse
import json
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import DATA_OUTPUT_DIR
from logger import logger


def format_size(num_bytes: int) -> str:
    """把字节数格式化为人类可读的字符串。"""
    n = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{int(n)} B" if unit == "B" else f"{n:.2f} {unit}"
        n /= 1024
    return f"{num_bytes} B"


def _dir_size(path: Path):
    """递归统计目录大小与文件数。返回 (bytes, file_count)。"""
    total = 0
    count = 0
    if not path.exists():
        return 0, 0
    for p in path.rglob("*"):
        if p.is_file():
            try:
                total += p.stat().st_size
            except OSError:
                pass
            count += 1
    return total, count


class TempFileManager:
    """临时文件管理：以「一次运行的完整论文输出」为管理单元。"""

    def __init__(self, output_root: Path = None):
        self.output_root = Path(output_root) if output_root else DATA_OUTPUT_DIR

    # =========================
    # 列出
    # =========================

    def list_papers(self) -> list:
        """列出 data/output/ 下所有论文输出（一次运行 = 一个 paper 目录）。"""
        if not self.output_root.exists():
            return []
        papers = []
        for d in sorted(self.output_root.iterdir()):
            if not d.is_dir() or d.name.startswith("."):
                continue
            size, count = _dir_size(d)
            papers.append(self._base_entry(d, size, count))
        return papers

    def paper_detail(self, paper: str) -> dict:
        """列出单次运行（单篇论文）下所有文件的明细（相对路径 + 大小）。"""
        d = self._resolve_paper(paper)
        if d is None:
            return {"paper": paper, "error": "not_found"}
        files = []
        for p in sorted(d.rglob("*")):
            if p.is_file():
                files.append({
                    "path": str(p.relative_to(d)),
                    "size_bytes": p.stat().st_size,
                    "size_human": format_size(p.stat().st_size),
                })
        size, count = _dir_size(d)
        entry = self._base_entry(d, size, count)
        entry["files"] = files
        return entry

    def summary(self) -> dict:
        """汇总所有临时输出：总量、文件数、每篇占用。"""
        papers = self.list_papers()
        total_bytes = sum(p["size_bytes"] for p in papers)
        total_files = sum(p["file_count"] for p in papers)
        return {
            "paper_count": len(papers),
            "total_bytes": total_bytes,
            "total_human": format_size(total_bytes),
            "total_files": total_files,
            "papers": papers,
        }

    # =========================
    # 保存 / 丢弃
    # =========================

    def keep(self, paper: str) -> dict:
        """保留原位：不移动、不删除，仅返回当前占用，供前端确认。"""
        d = self._resolve_paper(paper)
        if d is None:
            return {"paper": paper, "kept": False, "error": "not_found"}
        size, count = _dir_size(d)
        return {
            "paper": paper,
            "kept": True,
            "size_bytes": size,
            "size_human": format_size(size),
            "file_count": count,
        }

    def discard(self, paper: str) -> dict:
        """丢弃：删除该次运行的完整输出目录，返回释放的空间。"""
        d = self._resolve_paper(paper)
        if d is None:
            return {"paper": paper, "discarded": False, "error": "not_found"}
        size, count = _dir_size(d)
        shutil.rmtree(d)
        logger.info(f"  已丢弃临时输出 {d.name}: 释放 {format_size(size)} ({count} 个文件)")
        return {
            "paper": paper,
            "discarded": True,
            "freed_bytes": size,
            "freed_human": format_size(size),
            "file_count": count,
        }

    # =========================
    # 内部
    # =========================

    def _base_entry(self, d: Path, size: int, count: int) -> dict:
        """构造一条 paper 摘要记录。"""
        try:
            mtime = datetime.fromtimestamp(d.stat().st_mtime).isoformat(timespec="seconds")
        except OSError:
            mtime = None
        return {
            "paper": d.name,
            "path": str(d),
            "size_bytes": size,
            "size_human": format_size(size),
            "file_count": count,
            "steps": self._present_steps(d),
            "last_modified": mtime,
        }

    def _resolve_paper(self, paper: str):
        """安全解析论文目录；防止路径穿越（只允许 output_root 下的一级子目录）。"""
        if not paper or paper in (".", "..") or "/" in paper or "\\" in paper:
            return None
        d = (self.output_root / paper).resolve()
        try:
            d.relative_to(self.output_root.resolve())
        except ValueError:
            return None
        return d if d.is_dir() else None

    def _present_steps(self, d: Path) -> list:
        """列出该次运行已完成到哪些 step（存在 stepN 目录）。"""
        return sorted(
            s.name for s in d.iterdir()
            if s.is_dir() and s.name.startswith("step")
        )


def main():
    parser = argparse.ArgumentParser(
        description="临时文件管理：列出/保留/丢弃每次运行的完整论文输出"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("list", help="列出所有论文输出及占用")
    sub.add_parser("summary", help="汇总所有临时输出")

    p_detail = sub.add_parser("detail", help="查看单次运行的完整文件明细")
    p_detail.add_argument("--paper", required=True)

    p_keep = sub.add_parser("keep", help="保留原位（不删除）")
    p_keep.add_argument("--paper", required=True)

    p_discard = sub.add_parser("discard", help="删除该次运行的完整输出")
    p_discard.add_argument("--paper", required=True)

    args = parser.parse_args()
    mgr = TempFileManager()

    if args.command == "list":
        data = mgr.list_papers()
    elif args.command == "summary":
        data = mgr.summary()
    elif args.command == "detail":
        data = mgr.paper_detail(args.paper)
    elif args.command == "keep":
        data = mgr.keep(args.paper)
    elif args.command == "discard":
        data = mgr.discard(args.paper)
    else:
        parser.print_help()
        return

    print(json.dumps(data, ensure_ascii=False, indent=2))
    if isinstance(data, dict) and data.get("error"):
        sys.exit(1)


if __name__ == "__main__":
    main()
