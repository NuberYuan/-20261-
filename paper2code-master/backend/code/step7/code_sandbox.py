"""
代码沙箱执行器: 在隔离环境中执行生成的代码，捕获运行时错误。

借鉴 TraceCoder (ICSE 2026) 的沙箱机制:
- 子进程隔离: subprocess.Popen 在独立 Python 进程中执行
- 临时目录隔离: 防止文件系统副作用
- 危险操作禁用: reliability_guard 限制 os.kill, shutil.rmtree 等
- 超时控制: 超时后 kill 进程
- 结构化结果: 捕获 stdout/stderr/traceback
"""

import ast
import json
import os
import signal
import subprocess
import sys
import tempfile
import textwrap
from dataclasses import dataclass, field
from typing import Dict, List, Optional


def _kill_process_tree(proc):
    """杀掉进程及其所有子进程（Windows下用taskkill，POSIX用killpg）。"""
    try:
        if sys.platform == "win32":
            # Windows: taskkill /T 杀进程树, /F 强制
            subprocess.run(
                ["taskkill", "/T", "/F", "/PID", str(proc.pid)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=5,
            )
        else:
            # POSIX: 杀进程组
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except Exception:
        # 兜底：至少杀掉主进程
        try:
            proc.kill()
        except Exception:
            pass


@dataclass
class SandboxResult:
    """沙箱执行结果。"""
    success: bool = False
    exit_code: int = -1
    stdout: str = ""
    stderr: str = ""
    error_type: str = ""       # ImportError / RuntimeError / TypeError / ...
    error_message: str = ""    # 错误摘要（单行）
    traceback: str = ""        # 完整 traceback
    tests_passed: int = 0
    tests_total: int = 0
    details: List[dict] = field(default_factory=list)  # 每条测试详情


# ---------------------------------------------------------------------------
# reliability_guard: 禁用危险操作（借鉴 TraceCoder）
# ---------------------------------------------------------------------------

_RELIABILITY_GUARD = textwrap.dedent("""\
import builtins, os as _os, sys as _sys
builtins.exit = None
builtins.quit = None
# 限制 CPU/内存线程数，防止系统卡顿
_os.environ['OMP_NUM_THREADS'] = '1'
_os.environ['MKL_NUM_THREADS'] = '1'
_os.environ['OPENBLAS_NUM_THREADS'] = '1'
# 禁止 GPU 使用（沙箱不需要 GPU，防止显存占用导致系统卡顿）
_os.environ['CUDA_VISIBLE_DEVICES'] = ''
_os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
# 只禁用最危险的操作（进程杀死），其他全部保留
# torch/numpy/transformers 等库需要 subprocess.Popen、文件 I/O 等
for _a in ['kill','killpg','fork','forkpty','setuid','chroot']:
    if hasattr(_os, _a): setattr(_os, _a, None)
if isinstance(__builtins__, dict):
    __builtins__['help'] = None
else:
    builtins.help = None
for _n in ['ipdb','joblib','resource','psutil']:
    _sys.modules[_n] = None
# 限制 torch 线程数（如果 torch 被导入的话）
try:
    import torch
    torch.set_num_threads(1)
    # 禁止 torch 使用 GPU
    torch.cuda.is_available = lambda: False
except ImportError:
    pass
""")


class CodeSandbox:
    """在隔离环境中执行生成的代码，捕获运行时错误。"""

    def __init__(self, timeout: int = 30):
        self.timeout = timeout

    # ------------------------------------------------------------------
    # 公开接口
    # ------------------------------------------------------------------

    def run_snippet(self, code: str, timeout: Optional[int] = None) -> SandboxResult:
        """在子进程中执行代码片段，返回执行结果。"""
        return self._execute_in_subprocess(code, timeout or self.timeout)

    def run_file_with_smoke(
        self,
        file: str,
        code: str,
        all_files: Dict[str, str],
        config: Optional[dict] = None,
        timeout: Optional[int] = None,
        project_root: Optional[str] = None,
        known_packages: Optional[set] = None,
    ) -> SandboxResult:
        """执行代码 + 自动生成的冒烟测试。

        流程: 写入临时目录 → 注入冒烟测试 → 子进程执行 → 解析结果。

        Args:
            project_root: 项目根目录路径，用于解析项目内跨文件 import。
                          如果提供，会将所有生成的文件写入沙箱临时目录。
            known_packages: 已知可安装的第三方包名集合（来自 requirements.txt），
                            跳过这些包的 import 检查。
        """
        from step7.smoke_test_generator import SmokeTestGenerator

        gen = SmokeTestGenerator()
        smoke_code = gen.generate(file, code, all_files, config,
                                  known_packages=known_packages)
        if not smoke_code.strip():
            # 没有可测试的内容，直接返回成功
            return SandboxResult(success=True, exit_code=0)

        full_code = code + "\n\n" + smoke_code
        return self._execute_in_subprocess(
            full_code, timeout or self.timeout,
            work_dir=file, project_root=project_root, all_files=all_files,
        )

    # ------------------------------------------------------------------
    # 内部实现
    # ------------------------------------------------------------------

    def _execute_in_subprocess(
        self, code: str, timeout: int, work_dir: Optional[str] = None,
        project_root: Optional[str] = None,
        all_files: Optional[Dict[str, str]] = None,
    ) -> SandboxResult:
        """在子进程中执行代码，返回结构化结果。"""
        result = SandboxResult()

        # 用户代码写入一个文件，wrapper 通过 exec(compile(open(...).read())) 执行它
        user_file = None
        wrapper_file = None
        sandbox_dir = None
        outer_dir = None
        try:
            # 1. 创建沙箱工作目录（如果需要项目文件，就在这里构建完整项目结构）
            if project_root and all_files:
                # 目录结构:
                #   outer_dir/           ← 项目根（config/, data/ 等在这里）
                #     project/           ← cwd 设在这里，import 从这里开始
                #       __sandbox_entry__.py
                #       __sandbox_wrapper__.py
                #       (所有项目文件也在这里，保证 import 正常)
                # 这样代码用 "../config/config.yaml" 时能正确解析到 outer_dir/config/
                outer_dir = tempfile.mkdtemp(prefix="sandbox_proj_")
                sandbox_dir = os.path.join(outer_dir, "project")
                os.makedirs(sandbox_dir, exist_ok=True)

                # 将所有生成的文件写入 outer_dir（项目根）
                for fpath, fcode in all_files.items():
                    full_path = os.path.join(outer_dir, fpath)
                    os.makedirs(os.path.dirname(full_path), exist_ok=True)
                    with open(full_path, "w", encoding="utf-8") as f:
                        f.write(fcode)

                # 同时复制一份到 sandbox_dir（project/ 子目录），保证 import 正常
                for fpath, fcode in all_files.items():
                    full_path = os.path.join(sandbox_dir, fpath)
                    os.makedirs(os.path.dirname(full_path), exist_ok=True)
                    with open(full_path, "w", encoding="utf-8") as f:
                        f.write(fcode)

                # 用户代码（含 smoke test）写入 project/ 子目录
                user_path = os.path.join(sandbox_dir, "__sandbox_entry__.py")
                with open(user_path, "w", encoding="utf-8") as f:
                    f.write(code)
            else:
                # 简单模式：只写用户代码到临时文件
                fd_u, user_path = tempfile.mkstemp(suffix=".py", prefix="sandbox_user_")
                os.close(fd_u)
                with open(user_path, "w", encoding="utf-8") as f:
                    f.write(code)

            user_file = user_path

            # 2. 写 wrapper 到临时文件（与用户代码同目录，方便 import）
            # cwd 设为 sandbox_dir（project/ 子目录），sys.path 也指向这里
            wrapper = self._build_wrapper(user_path, sandbox_dir=sandbox_dir)
            if sandbox_dir:
                wrapper_path = os.path.join(sandbox_dir, "__sandbox_wrapper__.py")
            else:
                fd_w, wrapper_path = tempfile.mkstemp(suffix=".py", prefix="sandbox_wrapper_")
                os.close(fd_w)
            wrapper_file = wrapper_path
            with open(wrapper_path, "w", encoding="utf-8") as f:
                f.write(wrapper)

            # 3. 在子进程中执行 wrapper
            popen_kwargs = dict(
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
            if sys.platform != "win32":
                popen_kwargs["start_new_session"] = True  # 新进程组，方便killpg
            proc = subprocess.Popen(
                [sys.executable, wrapper_path],
                **popen_kwargs,
            )

            stdout, stderr = "", ""
            try:
                stdout, stderr = proc.communicate(timeout=timeout)
            except subprocess.TimeoutExpired:
                _kill_process_tree(proc)
                try:
                    stdout, stderr = proc.communicate(timeout=2)
                except Exception:
                    pass
                result.success = False
                result.exit_code = -9
                result.error_type = "TimeoutError"
                result.error_message = f"Execution timed out after {timeout}s"
                result.stderr = stderr or ""
                result.stdout = stdout or ""
                return result

            result.exit_code = proc.returncode
            result.stdout = stdout or ""
            result.stderr = stderr or ""

            # 解析结构化结果（如果存在 JSON markers）
            parsed = self._parse_structured_output(result.stdout)
            if parsed:
                result.tests_passed = parsed.get("tests_passed", 0)
                result.tests_total = parsed.get("tests_total", 0)
                result.details = parsed.get("details", [])
                result.success = parsed.get("success", proc.returncode == 0)
                if not result.success and parsed.get("error_type"):
                    result.error_type = parsed["error_type"]
                    result.error_message = parsed.get("error_message", "")
                    result.traceback = parsed.get("traceback", "")
                return result

            # 无结构化输出：按 exit code + stderr 判断
            if proc.returncode == 0:
                result.success = True
            else:
                result.success = False
                result.error_type, result.error_message, result.traceback = (
                    self._classify_error(stderr or stdout)
                )

            return result

        except Exception as e:
            result.success = False
            result.error_type = type(e).__name__
            result.error_message = str(e)
            return result
        finally:
            # 清理临时文件和目录
            import shutil
            # 优先清理 outer_dir（包含 sandbox_dir 和项目文件）
            if outer_dir and os.path.exists(outer_dir):
                try:
                    shutil.rmtree(outer_dir, ignore_errors=True)
                except Exception:
                    pass
            elif sandbox_dir and os.path.exists(sandbox_dir):
                try:
                    shutil.rmtree(sandbox_dir, ignore_errors=True)
                except Exception:
                    pass
            else:
                for f in (user_file, wrapper_file):
                    if f and os.path.exists(f):
                        try:
                            os.remove(f)
                        except OSError:
                            pass

    def _build_wrapper(self, user_code_path: str,
                       sandbox_dir: Optional[str] = None) -> str:
        """构建沙箱包装代码: reliability_guard + 读取并执行用户代码 + 结果捕获。

        Args:
            user_code_path: 用户代码临时文件的绝对路径。
            sandbox_dir: 沙箱项目目录（包含所有生成的文件），用于 sys.path。
        """
        path_setup = ""
        if sandbox_dir:
            # 将项目根目录加入 sys.path，使项目内 import 可用
            path_setup = textwrap.dedent(f"""\
                if {sandbox_dir!r} not in sys.path:
                    sys.path.insert(0, {sandbox_dir!r})
            """)

        return textwrap.dedent("""\
            import json, sys, traceback, os, tempfile

            # --- reliability_guard (from TraceCoder) ---
            # 只禁用最危险的操作（进程控制），不禁用文件 I/O
            {guard}

            # --- sys.path setup for project-local imports ---
            {path_setup}

            # --- JSON result markers ---
            _MARKER_START = "---SANDBOX_RESULTS_START---"
            _MARKER_END = "---SANDBOX_RESULTS_END---"

            def _emit(data):
                sys.stdout.flush()
                print("\\n" + _MARKER_START)
                json.dump(data, sys.stdout, ensure_ascii=False)
                print("\\n" + _MARKER_END)
                sys.stdout.flush()

            # --- Read and execute user code ---
            _user_code_path = {user_code_path!r}
            try:
                _orig_cwd = os.getcwd()
                _work_dir = {sandbox_dir!r} or tempfile.mkdtemp(prefix="sandbox_wd_")
                os.chdir(_work_dir)

                with open(_user_code_path, "r", encoding="utf-8") as _f:
                    _user_code = _f.read()

                exec(compile(_user_code, "<sandbox>", "exec"))

                os.chdir(_orig_cwd)
                _emit({{"success": True}})
            except Exception as _exc:
                try:
                    os.chdir(_orig_cwd)
                except Exception:
                    pass
                _tb = traceback.format_exc()
                _emit({{
                    "success": False,
                    "error_type": type(_exc).__name__,
                    "error_message": str(_exc)[:500],
                    "traceback": _tb,
                }})
        """).format(
            guard=_RELIABILITY_GUARD,
            path_setup=path_setup,
            user_code_path=user_code_path,
            sandbox_dir=sandbox_dir or "",
        )

    @staticmethod
    def _parse_structured_output(stdout: str) -> Optional[dict]:
        """解析 JSON markers 包裹的结构化输出。

        支持两种 marker 格式:
        - ---SMOKE_TEST_RESULTS--- (冒烟测试生成器产出，含 tests_passed/tests_total/details)
        - ---SANDBOX_RESULTS_START/END--- (wrapper 产出，含 success/error_type)
        优先返回冒烟测试结果（信息更丰富）。
        """
        # 1. 尝试冒烟测试 marker（信息更丰富）
        smoke_marker = "---SMOKE_TEST_RESULTS---"
        smoke_idx = stdout.find(smoke_marker)
        if smoke_idx != -1:
            json_start = smoke_idx + len(smoke_marker)
            smoke_idx2 = stdout.find(smoke_marker, json_start)
            if smoke_idx2 != -1:
                raw = stdout[json_start:smoke_idx2].strip()
                if raw:
                    try:
                        return json.loads(raw)
                    except json.JSONDecodeError:
                        pass

        # 2. 尝试 wrapper marker
        start = "---SANDBOX_RESULTS_START---"
        end = "---SANDBOX_RESULTS_END---"
        idx_start = stdout.find(start)
        if idx_start == -1:
            return None
        json_start = idx_start + len(start)
        idx_end = stdout.find(end, json_start)
        if idx_end == -1:
            return None
        raw = stdout[json_start:idx_end].strip()
        if not raw:
            return None
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return None

    @staticmethod
    def _classify_error(stderr: str) -> tuple:
        """从 stderr 提取错误类型和摘要。"""
        if not stderr.strip():
            return ("UnknownError", "Unknown error (no stderr output)", "")

        lines = stderr.strip().splitlines()
        traceback_text = stderr.strip()

        # 查找最后一行 "XxxError: message"
        error_type = "RuntimeError"
        error_message = lines[-1] if lines else "Unknown error"

        for line in reversed(lines):
            line = line.strip()
            if "Error:" in line or "Exception:" in line:
                parts = line.split(":", 1)
                error_type = parts[0].strip().split(".")[-1]  # 取最后的类名
                error_message = parts[1].strip() if len(parts) > 1 else line
                break
            # Traceback 格式: "raise XxxError(...)"
            if line.startswith("raise "):
                error_type = "RuntimeError"
                error_message = line
                break

        return (error_type, error_message[:500], traceback_text)
