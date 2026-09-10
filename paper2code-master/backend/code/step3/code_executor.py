"""
code_executor.py

安全执行 AI 生成的图片分析代码。

安全措施:
  1. 禁止导入黑名单中的库 (pytesseract, tensorflow, torch 等重量级依赖)
  2. 禁用 os.system / os.popen，并清空 subprocess 模块
     （注：不涉及 eval/exec/compile/open；__import__ 仅做黑名单过滤）
  3. 超时保护 (300s)
  4. 临时文件执行后自动删除

Author: Paper2Code
"""

from typing import List, Dict, Optional, Any, Tuple, Union

import subprocess
import sys
import tempfile
import os

from logger import logger


class CodeExecutor:

    # 禁止导入的重量级依赖
    FORBIDDEN_IMPORTS = [
        "pytesseract",
        "tensorflow",
        "torch",
    ]

    # 在沙箱中禁用的危险内置函数
    SANDBOX_PREAMBLE = """
# === 安全沙箱 ===
import builtins
_original_import = builtins.__import__
def _safe_import(name, *args, **kwargs):
    forbidden = {forbidden_list}
    for f in forbidden:
        if name == f or name.startswith(f + '.'):
            raise ImportError(f"Blocked import: {name}")
    return _original_import(name, *args, **kwargs)
builtins.__import__ = _safe_import

# 禁用危险函数
import sys
sys.modules['os'].system = lambda *a,**k: (_ for _ in ()).throw(RuntimeError("os.system blocked"))
sys.modules['os'].popen = lambda *a,**k: (_ for _ in ()).throw(RuntimeError("os.popen blocked"))
sys.modules['subprocess'] = type(sys)('subprocess')
sys.modules['subprocess'].__dict__.clear()
""".replace("{forbidden_list}", repr(FORBIDDEN_IMPORTS))

    def run(self, code: str, image_path: str) -> Optional[str]:
        """
        在隔离的子进程中执行 Python 代码。

        Args:
            code: 要执行的 Python 代码
            image_path: 图片路径（作为 sys.argv[1] 传入）

        Returns:
            执行结果的 stdout，失败返回 None
        """
        logger.info("Executing generated code")

        # 检查禁止依赖
        for item in self.FORBIDDEN_IMPORTS:
            if item in code:
                logger.error(f"禁止依赖: {item}")
                return None

        # 写入临时文件
        with tempfile.NamedTemporaryFile(
            suffix=".py",
            delete=False,
            mode="w",
            encoding="utf-8-sig",
        ) as f:
            f.write("# -*- coding: utf-8 -*-\n")
            f.write(self.SANDBOX_PREAMBLE + "\n")
            f.write(code)
            temp_file = f.name

        try:
            result = subprocess.run(
                [sys.executable, temp_file, image_path],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=300,
                env={
                    **os.environ,
                    "PYTHONIOENCODING": "utf-8",
                },
            )

            if result.returncode != 0:
                logger.error(f"Generated code error:\n{result.stderr[:500]}")
                return None

            output = result.stdout.strip()

            if not output:
                logger.error("Generated code has no output")
                return None

            return output

        except subprocess.TimeoutExpired:
            logger.error("Generated code execution timed out (300s)")
            return None

        except Exception as e:
            logger.error(f"Execution error: {e}")
            return None

        finally:
            try:
                os.remove(temp_file)
            except OSError:
                pass
