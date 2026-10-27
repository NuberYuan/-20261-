"""
流程控制器核心模块

接收前端的「下一步」指令，逐步骤启动流水线；遇异常时把「异常情况」返回给前端，
详细的 traceback / 日志由前端自行读取 logs/app.log。

设计:
    - 一次只跑一个 step（前端每发一次「下一步」→ 跑下一个未完成的 step）
    - 「下一个 step」由输出文件是否存在推导（输出已存在即视为已完成，天然支持断点续跑）
    - 每个 step 用子进程 `python -m <module> ...` 运行，异常退出时把 stderr 尾巴写进
      logs/app.log，并向前端返回简短的错误摘要
    - 线程锁保证同一时刻只运行一个 step（并发「下一步」返回 busy）

纯 Python、零第三方依赖，可被前端通过子进程直接调用，
也可由 pipeline_api.py 包装成 HTTP 接口。

Usage (CLI):
    cd code
    python -m pipeline_controller status
    python -m pipeline_controller next
    python -m pipeline_controller retry
    python -m pipeline_controller run step3
    python -m pipeline_controller run step3 --force
    python -m pipeline_controller --paper plant next
    python -m pipeline_controller --pdf data/input/论文1.pdf next   # 指定 PDF，论文名取自文件名
"""

import argparse
import json
import subprocess
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import DEFAULT_PAPER, INPUT_PDF, LOG_DIR, get_paper_output_dir, get_paper_paths
from logger import logger

CODE_DIR = Path(__file__).resolve().parent
LOG_FILE = LOG_DIR / "app.log"

# 步骤定义: (key, 人类可读标签, 输出文件相对 paper 目录的路径)
STEP_DEFINITIONS = [
    ("step1",   "Step1  PDF → raw JSON",      "step1/raw_paper.json"),
    ("step2",   "Step2  JSON 清洗+修复+结构检测", "step2/structure_detect_paper.json"),
    ("step3",   "Step3  关键信息提取 (LLM)",   "step3/key_information.json"),
    ("step4",   "Step4  需求分析",             "step4/requirements_summary.yaml"),
    ("step5",   "Step5  工程设计 (LLM)",       "step5/project_analysis.yaml"),
    ("step6",   "Step6  项目整合",             "step6/project_brief.yaml"),
    ("step7",   "Step7  代码生成 (LLM)",       "step7/generation_report.yaml"),
    ("step8",   "Step8  实验执行",             "step8/execution_report.yaml"),
]

_STEP_KEYS = {key for key, _, _ in STEP_DEFINITIONS}
_LABELS = {key: label for key, label, _ in STEP_DEFINITIONS}
_OUTPUT_REL = {key: rel for key, _, rel in STEP_DEFINITIONS}

# 各 step 子进程超时（秒）。LLM 步可能远超 300s，超时会整步重跑。
DEFAULT_STEP_TIMEOUT = 300
STEP_TIMEOUTS = {
    "step3": 1800,
    "step4": 1200,
    "step5": 1200,
    "step7": 3600,
    "step8": 3600,
}


def step_timeout_seconds(step: str) -> int:
    return STEP_TIMEOUTS.get(step, DEFAULT_STEP_TIMEOUT)


def run_remaining_steps(run_next, sleep=time.sleep):
    """从当前进度跑完剩余 step。

    - complete / error：停止（失败不自动重试）
    - busy：等待 1s 再试
    - success / skipped：继续下一步
    """
    results = []
    while True:
        result = run_next()
        results.append(result)
        status = result.get("status")
        if status in ("complete", "error"):
            return results
        if status == "busy":
            sleep(1)
            continue


def public_pipeline_status(status: dict) -> str:
    """把控制器内部状态映射成前端使用的 status 字段。"""
    completed = status.get("completed") or []
    completed_count = len(completed)
    total = status.get("total") or 0
    last = status.get("last_result") or {}
    if status.get("running"):
        return "processing"
    if total > 0 and completed_count >= total:
        return "completed"
    if last.get("status") == "error":
        return "failed"
    if completed_count == 0:
        return "uploaded"
    return "processing"


class PipelineController:
    """逐步骤流程控制器：前端每发一次「下一步」就推进一个 step。"""

    def __init__(self, paper: str = "plant", pdf: str = None):
        self.paper = paper
        self.paper_dir = get_paper_output_dir(paper)
        # PDF 解析顺序：显式传入 > 该论文目录下的 source.pdf > config 的自动检测
        if pdf:
            self.pdf = str(Path(pdf).resolve())
        else:
            source_pdf = self.paper_dir / "source.pdf"
            self.pdf = str(source_pdf.resolve()) if source_pdf.exists() else INPUT_PDF
        self._lock = threading.Lock()
        self._last_step = None       # 上次实际执行过的 step
        self._last_result = None     # 上次执行的结果
        self._load_last_result()

    # =========================
    # 状态
    # =========================

    def output_path(self, step: str) -> Path:
        """某 step 的产出文件绝对路径。"""
        return self.paper_dir / _OUTPUT_REL[step]

    def is_done(self, step: str) -> bool:
        """产出文件存在即视为已完成。"""
        return self.output_path(step).exists()

    def completed_steps(self) -> list:
        return [k for k, _, _ in STEP_DEFINITIONS if self.is_done(k)]

    def remaining_steps(self) -> list:
        return [k for k, _, _ in STEP_DEFINITIONS if not self.is_done(k)]

    def next_step(self):
        """返回下一个未完成的 step key；全部完成则返回 None。"""
        for k, _, _ in STEP_DEFINITIONS:
            if not self.is_done(k):
                return k
        return None

    def _status_file(self) -> Path:
        return self.paper_dir / "pipeline_status.json"

    def _load_last_result(self):
        path = self._status_file()
        if not path.exists():
            return
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            self._last_result = data.get("last_result")
            self._last_step = data.get("last_step") or (
                (self._last_result or {}).get("step")
            )
        except (json.JSONDecodeError, OSError):
            pass

    def _save_last_result(self):
        try:
            self.paper_dir.mkdir(parents=True, exist_ok=True)
            self._status_file().write_text(
                json.dumps(
                    {
                        "last_step": self._last_step,
                        "last_result": self._last_result,
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
        except OSError:
            pass

    def status(self) -> dict:
        """当前进度（供前端渲染）。"""
        nxt = self.next_step()
        return {
            "paper": self.paper,
            "paper_dir": str(self.paper_dir),
            "total": len(STEP_DEFINITIONS),
            "completed": self.completed_steps(),
            "remaining": self.remaining_steps(),
            "next_step": nxt,
            "next_label": _LABELS.get(nxt),
            "running": self._lock.locked(),
            "last_result": self._last_result,
            "log_file": str(LOG_FILE),
        }

    # =========================
    # 运行
    # =========================

    def run_next(self) -> dict:
        """运行下一个未完成的 step。"""
        key = self.next_step()
        if key is None:
            return {
                "paper": self.paper,
                "step": None,
                "status": "complete",
                "message": "所有步骤已完成",
                "log_file": str(LOG_FILE),
            }
        return self.run_step(key)

    def run_step(self, step: str, force: bool = False) -> dict:
        """运行指定 step。force=True 时忽略「已完成」跳过。"""
        if step not in _STEP_KEYS:
            return {
                "paper": self.paper,
                "step": step,
                "status": "error",
                "error": f"未知步骤: {step}",
                "log_file": str(LOG_FILE),
            }
        if not force and self.is_done(step):
            return {
                "paper": self.paper,
                "step": step,
                "label": _LABELS[step],
                "status": "skipped",
                "message": "输出已存在，跳过（force=true 可强制重跑）",
                "output": str(self.output_path(step)),
                "log_file": str(LOG_FILE),
            }
        if not self._lock.acquire(blocking=False):
            return {
                "paper": self.paper,
                "step": step,
                "label": _LABELS[step],
                "status": "busy",
                "error": "已有 step 正在运行，请稍后再试",
                "log_file": str(LOG_FILE),
            }
        try:
            self._last_step = step
            self._last_result = self._run(step, force=force)
            self._save_last_result()
            return self._last_result
        finally:
            self._lock.release()

    def retry(self) -> dict:
        """重跑上次实际执行过的 step（force=True）。"""
        if self._last_step is None:
            return {
                "paper": self.paper,
                "step": None,
                "status": "error",
                "error": "没有可重试的步骤（尚未执行过任何 step）",
                "log_file": str(LOG_FILE),
            }
        return self.run_step(self._last_step, force=True)

    def run_all(self) -> dict:
        """一次性跑完整条流水线：从第一个未完成 step 推进到全部完成。

        遇到某个 step 失败即停止（结果里会带 failed_step），
        用户可继续用 `retry` 或再次 `full` 从断点续跑。
        """
        results = run_remaining_steps(self.run_next)
        ran = [r for r in results if r.get("status") not in ("complete",)]
        failed = next((r.get("step") for r in ran if r.get("status") == "error"), None)
        return {
            "paper": self.paper,
            "status": "success" if failed is None else "error",
            "ran": len(ran),
            "failed_step": failed,
            "results": results,
            "log_file": str(LOG_FILE),
        }

    def _run(self, step: str, force: bool = False) -> dict:
        """实际执行单个 step 的子进程。"""
        label = _LABELS[step]
        cmd = self._command(step, force=force)
        logger.info(f"[控制器] 启动 {step} ({label}) ...")
        start = time.time()
        timeout_sec = step_timeout_seconds(step)
        try:
            proc = subprocess.run(
                [sys.executable, "-m"] + cmd,
                cwd=str(CODE_DIR),
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout_sec,
            )
        except subprocess.TimeoutExpired:
            logger.error(f"[控制器] {step} ({label}) 超时 ({timeout_sec}s)")
            return self._error(step, f"{label} 执行超时 ({timeout_sec}s)", time.time() - start)
        except Exception as e:
            # 子进程本身无法启动等罕见异常，traceback 落 app.log
            logger.exception(f"[控制器] 运行 {step} ({label}) 时发生异常")
            return self._error(step, f"启动失败: {e}", time.time() - start)

        dur = time.time() - start
        out_path = self.output_path(step)

        if proc.returncode == 0 and out_path.exists():
            logger.info(f"[控制器] {step} ({label}) 完成，用时 {dur:.1f}s")
            return {
                "paper": self.paper,
                "step": step,
                "label": label,
                "status": "success",
                "output": str(out_path),
                "duration_seconds": round(dur, 2),
                "next_step": self.next_step(),
                "log_file": str(LOG_FILE),
            }

        # 失败：把 stderr/stdout 尾巴写进 app.log，向前端只返回简短摘要
        tail = (proc.stderr or proc.stdout or "").strip()
        if len(tail) > 2000:
            tail = tail[-2000:]
        logger.error(
            f"[控制器] {step} ({label}) 失败 (exit={proc.returncode})\n{tail}"
        )
        return self._error(step, f"{label} 退出码 {proc.returncode}", dur)

    def _error(self, step: str, message: str, dur: float) -> dict:
        return {
            "paper": self.paper,
            "step": step,
            "label": _LABELS[step],
            "status": "error",
            "error": message,
            "duration_seconds": round(dur, 2),
            "log_file": str(LOG_FILE),
        }

    # =========================
    # 命令构造
    # =========================

    def _command(self, step: str, force: bool = False) -> list:
        p = get_paper_paths(self.paper)
        cmd = {
            "step1":   ["step1.step1_pdf_to_json", "--pdf", self.pdf,
                        "--out", p["raw_json"],
                        "--images-dir", p["images_dir"],
                        "--tables-dir", p["tables_dir"]],
            "step2":   ["step2.main", "--in", p["raw_json"],
                        "--out", p["structure_detect"]],
            "step3":   ["step3.step3_extract_information", "--in", p["structure_detect"],
                        "--out", p["key_information"]],
            "step4":   ["step4.step4_main", "--paper", self.paper],
            "step5":   ["step5.step5_main", "--paper", self.paper],
            "step6":   ["step6.step6_main", "--paper", self.paper],
            "step7":   ["step7.step7_main", "--paper", self.paper],
            "step8":   ["step8.step8_main", "--paper", self.paper],
        }[step]
        # step4/step5 支持子产物级断点续跑：force 时忽略已存在的子 yaml
        if force and step in ("step4", "step5"):
            cmd = cmd + ["--force"]
        return cmd


def main():
    parser = argparse.ArgumentParser(
        description="流程控制器：接收「下一步」指令逐步骤推进流水线"
    )
    parser.add_argument("--paper", type=str, default=None,
                        help="论文名称（输出目录名，默认取自 PDF 文件名）")
    parser.add_argument("--pdf", type=str, default=None,
                        help="输入 PDF 路径（默认自动检测 data/input/ 下第一个 PDF）")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("status", help="查看当前进度")
    sub.add_parser("next", help="运行下一个未完成的 step")
    sub.add_parser("retry", help="重跑上次执行过的 step")
    sub.add_parser("full", help="一键跑完整个流水线（断点续跑）")

    p_run = sub.add_parser("run", help="运行指定 step")
    p_run.add_argument("step", type=str)
    p_run.add_argument("--force", action="store_true", help="忽略已完成，强制重跑")

    args = parser.parse_args()

    # 未显式给 --paper 时：指定了 PDF 就用文件名作为论文名，否则用默认
    if args.paper is None:
        args.paper = Path(args.pdf).stem if args.pdf else DEFAULT_PAPER

    ctrl = PipelineController(paper=args.paper, pdf=args.pdf)

    if args.command == "status":
        data = ctrl.status()
    elif args.command == "next":
        data = ctrl.run_next()
    elif args.command == "retry":
        data = ctrl.retry()
    elif args.command == "run":
        data = ctrl.run_step(args.step, force=args.force)
    elif args.command == "full":
        data = ctrl.run_all()
    else:
        parser.print_help()
        return

    print(json.dumps(data, ensure_ascii=False, indent=2))
    if data.get("status") in ("error", "busy"):
        sys.exit(1)


if __name__ == "__main__":
    main()
