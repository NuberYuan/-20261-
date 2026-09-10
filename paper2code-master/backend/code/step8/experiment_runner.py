"""
Step8.4: 实验执行器 (Experiment Runner)

只执行 decision=RUN 的实验，在子进程里运行入口，捕获 stdout/stderr，记录
exit_code / 开始时间 / 结束时间 / 运行时长，收集实验新生成的结果文件（快照差）。
不做结果分析：指标提取/对比不在此步，留给后续步骤。

Step8 不猜怎么运行：入口与命令来自 execution_spec.yaml（Step7 产出），并由
ExperimentScheduler 解析后写入 execution_plan.yaml 的每个 RUN 实验（execution 段）。
Runner 优先读计划里的 execution，缺失时回退 spec 现场解析（build_command）。

入口类型（entrypoint.type）：
    python   默认，运行脚本（sys.executable 或 command.template）
    shell    预留：交由 shell 执行（POSIX sh -c / Windows cmd /c）
    command  预留：直接按空白切分为 argv 执行（可执行文件 + 参数）
"""

import json
import os
import shlex
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import yaml

from logger import logger

# 支持的入口类型。shell / command 为预留：Step7 暂不产出，但 Runner 已能处理。
_ENTRYPOINT_TYPES = ("python", "shell", "command")


def build_command(spec: dict, run_index: int, config_path: Path, project: Path) -> tuple:
    """从 execution_spec 解析 (entrypoint, command)，纯函数，供 Scheduler 预填与 Runner 复用。

    entrypoint: {"type": "python", "script": "main.py"} 或
                {"type": "shell"/"command", "command": "..."}
    command:    argv 列表（非 shell 字符串；type=shell 时为解释器 argv）
    """
    spec = spec or {}
    entrypoint_cfg = spec.get("entrypoint") or {}
    command_cfg = spec.get("command") or {}
    entry_type = entrypoint_cfg.get("type") or "python"

    if entry_type not in _ENTRYPOINT_TYPES:
        raise ValueError(
            f"execution_spec.entrypoint.type 非法: {entry_type}（支持 {_ENTRYPOINT_TYPES}）"
        )

    if entry_type == "python":
        entry_script = _resolve_script(entrypoint_cfg.get("script"), project)
        entrypoint = {"type": "python", "script": entry_script}
        command = _python_command(command_cfg, entry_script, run_index, config_path)
    else:
        # shell / command：预留入口，命令为模板字符串（渲染 run_index/config 占位符）。
        entrypoint = {"type": entry_type, "command": command_cfg.get("template")}
        command = _shell_command(entry_type, command_cfg.get("template"), run_index, config_path)
    return entrypoint, command


def _resolve_script(script, project: Path) -> str:
    """解析入口脚本：spec 指定优先，缺失/不存在时回退 main.py/run.py。"""
    if script and (project / str(script)).exists():
        return str(script)
    if script:
        logger.warning(f"  execution_spec 指定的入口 {script} 不存在，回退搜索 main.py/run.py")
    for name in ("main.py", "run.py"):
        if (project / name).exists():
            return name
    raise FileNotFoundError(f"generated_project 下未找到入口 main.py: {project}")


def _python_command(command_cfg: dict, entry_script: str, run_index: int, config_path: Path) -> list:
    """python 入口：按 command.template 构建 argv（list 形式，非 shell 字符串）。"""
    template = command_cfg.get("template")
    mapping = {
        "script": entry_script,
        "run_index": str(run_index),
        "config": str(config_path),
    }
    if template:
        tokens = shlex.split(template)
        # 首 token 若是 python 解释器名，换成当前解释器，保证与管线同环境
        if tokens and tokens[0] in ("python", "python3"):
            tokens[0] = sys.executable
        try:
            return [tok.format(**mapping) for tok in tokens]
        except (KeyError, IndexError, ValueError):
            logger.warning("  command.template 含未知占位符，回退默认命令")
    # 默认命令（无 template 或解析失败时）
    return [
        sys.executable, entry_script,
        "--run", str(run_index),
        "--config", str(config_path),
    ]


def _shell_command(entry_type: str, template, run_index: int, config_path: Path) -> list:
    """shell / command 入口（预留）：渲染占位符后构建 argv。

    - command: 先按空白切分模板、再替换占位符（避免 Windows 路径反斜杠被 shlex 转义吞掉）。
    - shell:   整体渲染为单条字符串，交由 shell 执行（POSIX 用 sh -c，Windows 用 cmd /c）。
    """
    if not template:
        raise ValueError(f"execution_spec 缺少 command.template（type={entry_type} 需要）")
    mapping = {"run_index": str(run_index), "config": str(config_path)}
    if entry_type == "command":
        tokens = shlex.split(template)
        try:
            return [tok.format(**mapping) for tok in tokens]
        except (KeyError, IndexError, ValueError):
            return tokens
    try:
        rendered = template.format(**mapping)
    except (KeyError, IndexError, ValueError):
        rendered = template
    if os.name == "nt":
        return ["cmd", "/c", rendered]
    return ["sh", "-c", rendered]


class ExperimentRunner:
    """在子进程里运行单个实验，保存原始输出、执行元数据与解析出的指标。"""

    def __init__(self, generated_project: Path, experiments_dir: Path, timeout: int = 3600,
                 spec: dict = None):
        self.project = Path(generated_project)
        self.experiments_dir = Path(experiments_dir)
        self.timeout = timeout
        self.spec = spec or {}

    def run(self, exp: dict, config_path: Path) -> dict:
        """运行一个实验，返回结果 dict（execution + logs 结构）。"""
        out_dir = self.experiments_dir / f"experiment_{exp['run_index'] + 1:03d}"
        out_dir.mkdir(parents=True, exist_ok=True)

        # 优先从执行计划（execution_plan.yaml）读已解析的入口+命令；缺失则回退 spec 现场解析
        exec_cfg = exp.get("execution") or {}
        command = exec_cfg.get("command")
        cmd_error = exec_cfg.get("error")
        if not command and not cmd_error:
            try:
                _, command = build_command(
                    self.spec, exp["run_index"], config_path, self.project
                )
            except Exception as e:  # noqa: BLE001 - 入口解析失败记为 cmd_error，不中断 Step8
                cmd_error = str(e)

        started_at = datetime.now().isoformat(timespec="seconds")
        start_ts = time.time()

        result = {
            # plan 里的实验 dict 用 experiment_id 键（Scheduler 由 analyze 的 id 转换而来），
            # 但 Runner 可能被直接传入 analyze 的原始 dict（含 id）。两者都兼容。
            "experiment_id": exp.get("id") or exp.get("experiment_id"),
            "run_index": exp["run_index"],
            "command": {
                "argv": list(command) if command else [],
                "display": " ".join(command) if command else None,
            },
            "execution": {
                "status": "RUNNING",
                "exit_code": None,
                "started_at": started_at,
                "finished_at": None,
                "duration_seconds": None,
                "working_directory": str(self.project),
            },
            "logs": {
                "stdout": str(out_dir / "stdout.log"),
                "stderr": str(out_dir / "stderr.log"),
            },
        }

        stdout = stderr = ""
        result_files = []
        if cmd_error:
            result["execution"]["status"] = "FAILED"
            result["execution"]["error"] = cmd_error
        else:
            before = self._snapshot_files(self.project)
            try:
                proc = subprocess.run(
                    command,
                    cwd=str(self.project),
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=self.timeout,
                )
                stdout = proc.stdout or ""
                stderr = proc.stderr or ""

                # 兼容：若 exit_code=2 且 stderr 含 argparse 错误，去掉 --run 参数重试
                if proc.returncode == 2 and "unrecognized arguments" in (stderr or ""):
                    fallback_cmd = self._strip_run_arg(command)
                    if fallback_cmd and fallback_cmd != command:
                        logger.warning(f"  命令含不支持的参数，回退重试: {' '.join(fallback_cmd)}")
                        proc2 = subprocess.run(
                            fallback_cmd,
                            cwd=str(self.project),
                            capture_output=True,
                            text=True,
                            encoding="utf-8",
                            errors="replace",
                            timeout=self.timeout,
                        )
                        stdout = proc2.stdout or ""
                        stderr = proc2.stderr or ""
                        proc = proc2
                        result["command"]["argv"] = list(fallback_cmd)
                        result["command"]["display"] = " ".join(fallback_cmd)
                        result["command"]["fallback"] = True

                result["execution"]["exit_code"] = proc.returncode
                result["execution"]["status"] = "SUCCESS" if proc.returncode == 0 else "FAILED"
            except subprocess.TimeoutExpired:
                result["execution"]["exit_code"] = None
                result["execution"]["status"] = "FAILED"
                result["execution"]["error"] = f"超时（>{self.timeout}s）"
            except Exception as e:  # noqa: BLE001 - 记录任意失败，不中断整个 Step8
                result["execution"]["exit_code"] = None
                result["execution"]["status"] = "FAILED"
                result["execution"]["error"] = str(e)
            # 运行前后快照差 = 实验新生成/修改的结果文件（只收集路径，不解析内容）
            result_files = self._diff_result_files(before, self._snapshot_files(self.project))

        result["execution"]["finished_at"] = datetime.now().isoformat(timespec="seconds")
        result["execution"]["duration_seconds"] = round(time.time() - start_ts, 2)

        # 写原始日志
        (out_dir / "stdout.log").write_text(stdout, encoding="utf-8")
        (out_dir / "stderr.log").write_text(stderr, encoding="utf-8")

        result["result_files"] = result_files
        result["output_dir"] = str(out_dir)

        # 写状态与结果
        (out_dir / "status.yaml").write_text(
            yaml.safe_dump(
                {
                    "experiment_id": result["experiment_id"],
                    "execution": result["execution"],
                    "logs": result["logs"],
                },
                allow_unicode=True, sort_keys=False,
            ),
            encoding="utf-8",
        )
        (out_dir / "result.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return result

    @staticmethod
    def _strip_run_arg(command: list) -> list:
        """去掉命令中的 --run <value> 参数，用于 argparse 不支持 --run 时回退。"""
        if not command:
            return command
        result = []
        skip_next = False
        for tok in command:
            if skip_next:
                skip_next = False
                continue
            if tok == "--run":
                skip_next = True
                continue
            result.append(tok)
        return result

    @staticmethod
    def _snapshot_files(root: Path) -> dict:
        """快照目录下所有文件的 (size, mtime_ns)，返回 {相对路径: (size, mtime_ns)}。

        用于运行前后对比，识别实验新生成/修改的结果文件。忽略 __pycache__ / .pyc 噪声。
        """
        snap = {}
        for dirpath, dirs, files in os.walk(root):
            dirs[:] = [d for d in dirs if d != "__pycache__"]
            for f in files:
                if f.endswith(".pyc"):
                    continue
                p = os.path.join(dirpath, f)
                try:
                    st = os.stat(p)
                    # 归一化为正斜杠相对路径，报告跨平台可读
                    snap[os.path.relpath(p, root).replace(os.sep, "/")] = (st.st_size, st.st_mtime_ns)
                except OSError:
                    pass
        return snap

    @staticmethod
    def _diff_result_files(before: dict, after: dict) -> list:
        """返回运行后新增或变更的文件相对路径列表（按路径排序）。"""
        result_files = []
        for rel, state in sorted(after.items()):
            if before.get(rel) != state:
                result_files.append(rel)
        return result_files
