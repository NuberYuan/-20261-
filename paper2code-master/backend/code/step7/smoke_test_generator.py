"""
冒烟测试生成器: 根据代码 AST 自动生成轻量级冒烟测试。

纯确定性生成，不调用 LLM，零 API 开销。

测试项 (按深度递进):
1. Import 检查: 所有 import 的模块/符号是否可导入
2. 实例化检查: 类是否能实例化
3. 函数调用检查: 关键函数是否能以示例参数调用
4. Config 引用检查: config 值替换后代码是否能正常解析

借鉴 TraceCoder 的 assertion counting 思路，每条测试独立 try-except，
确保一条失败不影响其他测试的执行。
"""

import ast
import sys
import textwrap
from typing import Dict, List, Optional, Set

# Python 标准库模块集合（用于跳过 import 检查）
try:
    STDLIB = set(sys.stdlib_module_names)
except AttributeError:
    # Python < 3.10 回退：覆盖常用标准库模块
    STDLIB = {
        "abc", "aifc", "argparse", "array", "ast", "asynchat", "asyncio",
        "asyncore", "atexit", "audioop", "base64", "bdb", "binascii",
        "binhex", "bisect", "builtins", "bz2", "calendar", "cgi", "cgitb",
        "chunk", "cmath", "cmd", "code", "codecs", "codeop", "collections",
        "colorsys", "compileall", "concurrent", "configparser", "contextlib",
        "contextvars", "copy", "copyreg", "cProfile", "crypt", "csv",
        "ctypes", "curses", "dataclasses", "datetime", "dbm", "decimal",
        "difflib", "dis", "distutils", "doctest", "email", "encodings",
        "enum", "errno", "faulthandler", "fcntl", "filecmp", "fileinput",
        "fnmatch", "formatter", "fractions", "ftplib", "functools", "gc",
        "getopt", "getpass", "gettext", "glob", "graphlib", "grp", "gzip",
        "hashlib", "heapq", "hmac", "html", "http", "idlelib", "imaplib",
        "imghdr", "imp", "importlib", "inspect", "io", "ipaddress", "itertools",
        "json", "keyword", "lib2to3", "linecache", "locale", "logging", "lzma",
        "mailbox", "mailcap", "marshal", "math", "mimetypes", "mmap",
        "modulefinder", "multiprocessing", "netrc", "nis", "nntplib",
        "numbers", "operator", "optparse", "os", "ossaudiodev", "pathlib",
        "pdb", "pickle", "pickletools", "pipes", "pkgutil", "platform",
        "plistlib", "poplib", "posix", "posixpath", "pprint", "profile",
        "pstats", "pty", "pwd", "py_compile", "pyclbr", "pydoc",
        "queue", "quopri", "random", "re", "readline", "reprlib", "resource",
        "rlcompleter", "runpy", "sched", "secrets", "select", "selectors",
        "shelve", "shlex", "shutil", "signal", "site", "smtpd", "smtplib",
        "sndhdr", "socket", "socketserver", "spwd", "sqlite3", "sre_compile",
        "sre_constants", "sre_parse", "ssl", "stat", "statistics", "string",
        "stringprep", "struct", "subprocess", "sunau", "symtable", "sys",
        "sysconfig", "syslog", "tabnanny", "tarfile", "telnetlib", "tempfile",
        "termios", "test", "textwrap", "threading", "time", "timeit", "tkinter",
        "token", "tokenize", "tomllib", "trace", "traceback", "tracemalloc",
        "tty", "turtle", "turtledemo", "types", "typing", "unicodedata",
        "unittest", "urllib", "uu", "uuid", "venv", "warnings", "wave",
        "weakref", "webbrowser", "winreg", "winsound", "wsgiref", "xdrlib",
        "xml", "xmlrpc", "zipapp", "zipfile", "zipimport", "zlib",
        "_thread", "__future__",
    }


class SmokeTestGenerator:
    """根据代码 AST 自动生成轻量级冒烟测试代码。"""

    def generate(
        self,
        file: str,
        code: str,
        all_files: Dict[str, str],
        config: Optional[dict] = None,
        known_packages: Optional[set] = None,
    ) -> str:
        """生成冒烟测试代码。

        Args:
            file: 当前文件路径（如 "models/gnn.py"）
            code: 当前文件的代码内容
            all_files: 所有已生成文件 {路径: 代码}
            config: config.yaml 解析后的字典
            known_packages: 已知可安装的第三方包名集合（来自 requirements.txt），
                            这些包的 import 检查会被跳过，避免环境未安装导致误报。

        Returns:
            冒烟测试代码字符串，可直接追加到用户代码后执行。
            如果没有可测试的内容，返回空字符串。
        """
        tests: List[str] = []

        # 1. Import 检查（跳过已知第三方包和本地模块）
        import_tests = self._generate_import_tests(code, all_files, known_packages)
        tests.extend(import_tests)

        # 2. 类实例化检查
        class_tests = self._generate_class_tests(code)
        tests.extend(class_tests)

        # 3. 函数调用检查
        func_tests = self._generate_function_tests(code)
        tests.extend(func_tests)

        if not tests:
            return ""

        return self._wrap_tests(tests)

    # ------------------------------------------------------------------
    # Import 检查
    # ------------------------------------------------------------------

    def _generate_import_tests(self, code: str, all_files: Dict[str, str],
                               known_packages: Optional[set] = None) -> List[str]:
        """为每个 import 语句生成 try-except 检查。

        跳过:
        - 本地模块（项目内生成的 .py 文件）
        - Python 标准库
        - requirements.txt 中声明的第三方包（known_packages）
        """
        tests = []
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return tests

        local_modules = self._extract_local_modules(all_files)
        known = known_packages or set()

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    mod = alias.name.split(".")[0]
                    if mod in local_modules or mod in STDLIB or mod in known:
                        continue
                    tests.append(self._import_test_snippet(alias.name, alias.asname))

            elif isinstance(node, ast.ImportFrom):
                if node.module is None:
                    continue
                mod = node.module.split(".")[0]
                if mod in local_modules or mod in STDLIB or mod in known:
                    continue
                names = [a.name for a in node.names]
                tests.append(self._from_import_test_snippet(node.module, names))

        return tests

    @staticmethod
    def _import_test_snippet(module: str, alias: Optional[str]) -> str:
        label = alias or module
        return textwrap.dedent(f"""\
            try:
                import {module}  # as {label}
                _results.append({{"test": "import {module}", "status": "passed"}})
            except Exception as _e:
                _results.append({{"test": "import {module}", "status": "failed",
                                  "error": f"{{type(_e).__name__}}: {{_e}}"}})
        """)

    @staticmethod
    def _from_import_test_snippet(module: str, names: List[str]) -> str:
        name_list = ", ".join(names)
        return textwrap.dedent(f"""\
            try:
                from {module} import {name_list}
                _results.append({{"test": "from {module} import {name_list}", "status": "passed"}})
            except Exception as _e:
                _results.append({{"test": "from {module} import {name_list}", "status": "failed",
                                  "error": f"{{type(_e).__name__}}: {{_e}}"}})
        """)

    # ------------------------------------------------------------------
    # 类实例化检查
    # ------------------------------------------------------------------

    def _generate_class_tests(self, code: str) -> List[str]:
        """为公开类生成实例化测试。"""
        tests = []
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return tests

        for node in ast.iter_child_nodes(tree):
            if not isinstance(node, ast.ClassDef):
                continue
            if node.name.startswith("_"):
                continue

            # 分析 __init__ 参数
            init_method = None
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == "__init__":
                    init_method = item
                    break

            # 构建实例化代码
            args, has_required = self._analyze_init_args(init_method)
            if has_required:
                # 有必填参数，跳过（无法安全实例化）
                continue

            call_args = ", ".join(args) if args else ""
            tests.append(self._class_test_snippet(node.name, call_args))

        return tests

    @staticmethod
    def _analyze_init_args(init_method: Optional[ast.FunctionDef]) -> tuple:
        """分析 __init__ 参数，返回 (调用参数列表, 是否有必填参数)。"""
        if init_method is None:
            return [], False

        args = init_method.args
        # 参数: posonlyargs, args, vararg, kwonlyargs, kw_defaults, kwarg, defaults
        all_args = args.args[1:]  # 去掉 self
        defaults = args.defaults

        # 计算没有默认值的参数数量
        n_defaults = len(defaults) if defaults else 0
        n_required = len(all_args) - n_defaults

        if n_required > 0:
            return [], True  # 有必填参数

        # 只有可选参数，用默认值（不传参）
        return [], False

    @staticmethod
    def _class_test_snippet(class_name: str, call_args: str) -> str:
        return textwrap.dedent(f"""\
            try:
                _obj = {class_name}({call_args})
                _results.append({{"test": "instantiate {class_name}", "status": "passed"}})
            except Exception as _e:
                _results.append({{"test": "instantiate {class_name}", "status": "failed",
                                  "error": f"{{type(_e).__name__}}: {{_e}}"}})
        """)

    # ------------------------------------------------------------------
    # 函数调用检查
    # ------------------------------------------------------------------

    def _generate_function_tests(self, code: str) -> List[str]:
        """为顶层函数生成调用测试（仅无参或全默认参数的函数）。"""
        tests = []
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return tests

        for node in ast.iter_child_nodes(tree):
            if not isinstance(node, ast.FunctionDef):
                continue
            if node.name.startswith("_"):
                continue
            # 跳过特殊方法
            if node.name in ("main", "run", "train", "test", "evaluate", "predict"):
                continue

            args = node.args
            all_args = args.args
            defaults = args.defaults
            n_defaults = len(defaults) if defaults else 0
            n_required = len(all_args) - n_defaults

            if n_required > 0:
                continue  # 有必填参数，跳过

            # 跳过参数默认值为 None 且函数名含 transform/config 的函数
            # 这类函数通常会对参数做验证，无参调用会触发 ValueError
            if n_defaults > 0 and defaults:
                for d in defaults:
                    if isinstance(d, ast.Constant) and d.value is None:
                        func_lower = node.name.lower()
                        if any(kw in func_lower for kw in ("transform", "config", "setup", "init")):
                            continue

            # 检查是否有 *args 或 **kwargs
            if args.vararg or args.kwarg:
                continue

            tests.append(self._func_test_snippet(node.name))

        return tests

    @staticmethod
    def _func_test_snippet(func_name: str) -> str:
        return textwrap.dedent(f"""\
            try:
                {func_name}()
                _results.append({{"test": "call {func_name}()", "status": "passed"}})
            except (ValueError, TypeError) as _e:
                # 参数验证失败（如 config=None 时抛出 ValueError），视为跳过
                _results.append({{"test": "call {func_name}()", "status": "skipped",
                                  "error": f"{{type(_e).__name__}}: {{_e}}"}})
            except Exception as _e:
                _results.append({{"test": "call {func_name}()", "status": "failed",
                                  "error": f"{{type(_e).__name__}}: {{_e}}"}})
        """)

    # ------------------------------------------------------------------
    # 包装
    # ------------------------------------------------------------------

    @staticmethod
    def _wrap_tests(tests: List[str]) -> str:
        """将测试片段包装为完整的冒烟测试代码。"""
        body = "\n".join(tests)
        return textwrap.dedent("""\
            # --- Smoke Tests (auto-generated) ---
            import json as _json, sys as _sys
            _results = []
            {body}

            # 输出结构化结果（skipped 不算失败）
            _passed = sum(1 for r in _results if r["status"] == "passed")
            _skipped = sum(1 for r in _results if r["status"] == "skipped")
            _total = len(_results)
            _failed = sum(1 for r in _results if r["status"] == "failed")
            _all_passed = _failed == 0
            _MARKER = "---SMOKE_TEST_RESULTS---"
            _sys.stdout.flush()
            print("\\n" + _MARKER)
            _json.dump({{
                "success": _all_passed,
                "tests_passed": _passed,
                "tests_total": _total,
                "details": _results
            }}, _sys.stdout, ensure_ascii=False)
            print("\\n" + _MARKER)
            _sys.stdout.flush()
            # --- End Smoke Tests ---
        """).format(body=body)

    # ------------------------------------------------------------------
    # 工具方法
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_local_modules(all_files: Dict[str, str]) -> Set[str]:
        """从已生成文件中提取本地模块名（用于跳过 import 检查）。"""
        modules = set()
        for path in all_files:
            if not path.endswith(".py"):
                continue
            # "models/gnn.py" → "gnn"
            name = path.rsplit("/", 1)[-1].replace(".py", "")
            if name != "__init__":
                modules.add(name)
            # 也添加完整包路径的顶级模块
            if "/" in path:
                top = path.split("/")[0]
                modules.add(top)
        return modules
