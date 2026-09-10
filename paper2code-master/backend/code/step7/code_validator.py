"""
Step7.4: 代码检查器

优先用 Python 本身的工具做规则检查（不调用 LLM）:

  逐文件检查（validate）:
  1. 语法检查      —— .py 用 ast.parse，.yaml 用 yaml.safe_load
  2. 类检查        —— spec.classes 里声明的类是否真的定义了
  3. 方法检查      —— spec.key_methods 里声明的方法是否真的定义了
  4. import 检查   —— 引用的本地模块是否存在，且 `from X import Y` 的 Y 是否在 X 中定义
  5. 配置引用检查  —— `config[...]` 读取的 key 是否在 config.yaml 中真实存在
  6. 函数接口检查  —— 函数调用（同文件 + 跨文件）的参数个数/关键字参数名/必填参数是否与定义一致
  7. 结论写死检查  —— .py/.md 中是否写死了论文结论性措辞

  项目级检查（validate_project，所有文件生成后调用）:
  8. 文件检查        —— 计划中的文件是否都生成且非空
  9. 第三方依赖检查  —— import 的第三方包是否在 requirements.txt 声明
  10. 需求追踪检查   —— FR → 实现符号（按 FR 分组）
  11. Step6 决策一致性检查 —— config.yaml 是否与 Step6 decisions 一致
  12. 函数接口检查  —— 汇总复查所有函数调用（同文件 + 跨文件）的接口是否一致
"""

import ast
import re
import sys
import yaml

# Python 标准库模块集合（用于第三方依赖判定）
try:
    STDLIB = set(sys.stdlib_module_names)
except AttributeError:  # Python < 3.10 兜底
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


class CodeValidator:
    """规则驱动的代码检查器。"""

    # 项目内的 Python 包目录不再硬编码：由 __init__ 从计划文件的 .py 顶层目录推导
    # （config 目录是 YAML 配置，不含 .py，自然不会被当作 Python 包）。

    # 第三方依赖: import 顶层名 → requirements.txt 中的包名
    IMPORT_TO_REQ = {
        "cv2": "opencv-python",
        "PIL": "pillow",
        "yaml": "pyyaml",
        "sklearn": "scikit-learn",
        "skimage": "scikit-image",
        "tensorboard": "tensorboard",
        "tqdm": "tqdm",
        "seaborn": "seaborn",
        "pandas": "pandas",
        "imblearn": "imbalanced-learn",
        "torchvision": "torchvision",
        "scipy": "scipy",
    }

    # 结论性措辞（论文结论，代码应动态计算而非写死）
    BANNED_CONCLUSIONS = [
        "consistently outperforms",
        "always yields",
        "always outperforms",
        "always better",
    ]

    def __init__(self, all_files: list):
        self.all_files = set(all_files)
        self.py_files = {f for f in all_files if f.endswith(".py")}

        # 项目内的 Python 包目录：从计划文件的 .py 顶层目录推导，使 src/algorithm
        # 等非 ML 包名不被误判为第三方依赖。
        self.project_pkgs = {
            f.split("/")[0] for f in self.py_files if "/" in f
        }

        # 跨文件检查所需的状态（由 step7_main 按生成顺序逐步注入）
        self.config = None          # config.yaml 内容（dict），用于配置引用检查
        self.config_keys = set()    # config.yaml 顶层 key
        self.symbols = {}           # file -> {"classes": set, "functions": set}

    # ============================================================
    # 状态注入
    # ============================================================

    def set_config(self, config_dict: dict):
        """注入 config.yaml 内容，启用配置引用检查。"""
        self.config = config_dict or {}
        self.config_keys = set(self.config.keys())

    def register(self, file: str, code: str):
        """登记已生成文件的类/函数符号，供跨文件 import 检查。"""
        if file.endswith(".py"):
            self.symbols[file] = {
                "classes": self._defined_classes(code),
                "functions": self._defined_functions(code),
                "signatures": self._function_signatures(code),
            }

    # ============================================================
    # 主入口
    # ============================================================

    def validate(self, file: str, code: str, spec: dict) -> dict:
        """检查单个文件，返回 {status, errors, checks}。"""
        errors = []
        checks = {}

        checks["syntax"] = self._check_syntax(file, code, errors)
        checks["classes"] = self._check_classes(code, spec.get("classes", []), errors)
        checks["methods"] = self._check_methods(code, spec.get("key_methods", []), errors)
        checks["imports"] = self._check_imports(code, errors)
        checks["config_refs"] = self._check_config_refs(file, code, errors)
        checks["function_interface"] = self._check_function_interface(code, errors)
        checks["conclusions"] = self._check_conclusions(file, code, errors)

        return {
            "status": "passed" if not errors else "failed",
            "errors": errors,
            "checks": checks,
        }

    # ============================================================
    # 1. 语法检查
    # ============================================================

    def _check_syntax(self, file: str, code: str, errors: list) -> bool:
        if not code.strip():
            errors.append(f"{file} 内容为空")
            return False

        if file.endswith(".py"):
            try:
                ast.parse(code)
                return True
            except SyntaxError as e:
                errors.append(f"语法错误: {e}")
                return False

        if file.endswith((".yaml", ".yml")):
            try:
                yaml.safe_load(code)
                return True
            except yaml.YAMLError as e:
                errors.append(f"YAML 解析错误: {e}")
                return False

        # .txt / .md 等不做语法检查
        return True

    # ============================================================
    # 2. 类检查 / 3. 方法检查
    # ============================================================

    def _check_classes(self, code: str, expected: list, errors: list) -> bool:
        if not expected:
            return True
        defined = self._defined_classes(code)
        ok = True
        for cls in expected:
            if cls not in defined:
                errors.append(f"缺少声明的类: {cls}")
                ok = False
        return ok

    def _check_methods(self, code: str, expected: list, errors: list) -> bool:
        if not expected:
            return True
        defined = self._defined_functions(code)
        ok = True
        for m in expected:
            if m not in defined:
                errors.append(f"缺少声明的方法: {m}")
                ok = False
        return ok

    # ============================================================
    # 4. import 检查（模块存在性 + 跨文件符号存在性）
    # ============================================================

    def _check_imports(self, code: str, errors: list) -> bool:
        """检查本地模块 import 是否存在；已登记的依赖再检查符号是否定义。"""
        if not code.strip():
            return True
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return True  # 语法错误已在 _check_syntax 报告

        missing = []

        def _is_package(top: str) -> bool:
            return any(f.startswith(top + "/") for f in self.py_files)

        for node in ast.walk(tree):
            # from data.dataset import X / from utils import metrics
            if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                top = node.module.split(".")[0]
                if top not in self.project_pkgs:
                    continue
                fpath = node.module.replace(".", "/") + ".py"
                # 模块本身存在，或模块是包（from pkg import submodule 合法）
                if fpath not in self.py_files and not _is_package(node.module):
                    missing.append(f"from {node.module} import ...")
                    continue
                # 符号存在性检查（仅当该文件已登记）
                syms = self.symbols.get(fpath)
                if syms is not None:
                    for alias in node.names:
                        if alias.name == "*":
                            continue
                        if (alias.name not in syms["classes"]
                                and alias.name not in syms["functions"]):
                            missing.append(
                                f"from {node.module} import {alias.name}"
                                f"（{fpath} 中未定义 {alias.name}）"
                            )
            # import data.dataset
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    top = alias.name.split(".")[0]
                    if top not in self.project_pkgs:
                        continue
                    fpath = alias.name.replace(".", "/") + ".py"
                    if fpath not in self.py_files and not _is_package(alias.name):
                        missing.append(f"import {alias.name}")

        for m in missing:
            errors.append(f"引用了不存在的模块/符号: {m}")
        return not missing

    # ============================================================
    # 5. 配置引用检查（代码读的 config key vs config.yaml 实际 key）
    # ============================================================

    def _check_config_refs(self, file: str, code: str, errors: list) -> bool:
        if self.config is None:
            return True  # config.yaml 尚未生成，跳过
        if not file.endswith(".py"):
            return True

        chains = self._config_access_chains(code)
        ok = True
        for chain in chains:
            # 先按原链查找
            node = self.config
            missing = None
            for k in chain:
                if isinstance(node, dict) and k in node:
                    node = node[k]
                else:
                    missing = k
                    break
            if missing is not None:
                # 兼容 LLM 常见的 config['data.batch_size'] 写法（点号分隔）：
                # 尝试把含点号的 key 拆成嵌套路径再查找一次
                expanded = []
                for k in chain:
                    expanded.extend(k.split(".") if "." in k else [k])
                node2 = self.config
                missing2 = None
                for k in expanded:
                    if isinstance(node2, dict) and k in node2:
                        node2 = node2[k]
                    else:
                        missing2 = k
                        break
                if missing2 is not None:
                    path = ".".join(chain)
                    # 收集所有有效嵌套路径，帮助 LLM 找到正确位置
                    valid_paths = self._collect_config_paths(self.config, max_depth=3)
                    # 过滤出与 missing key 相关的路径
                    hint_paths = [p for p in valid_paths if missing.split(".")[0] in p]
                    hint = ""
                    if hint_paths:
                        hint = f"\n  可能相关的路径: {sorted(hint_paths[:15])}"
                    errors.append(
                        f"配置引用错误: config['{path}'] 中 '{missing}' 不在 config.yaml"
                        f"（顶层键: {sorted(self.config_keys)}）{hint}"
                    )
                    ok = False
        return ok

    @staticmethod
    def _collect_config_paths(d: dict, prefix: str = "", max_depth: int = 3) -> list:
        """递归收集 config dict 的所有有效嵌套路径（点号分隔）。"""
        if max_depth <= 0 or not isinstance(d, dict):
            return []
        paths = []
        for k, v in d.items():
            full = f"{prefix}.{k}" if prefix else k
            paths.append(full)
            if isinstance(v, dict):
                paths.extend(
                    CodeValidator._collect_config_paths(v, full, max_depth - 1)
                )
        return paths

    # ============================================================
    # 辅助：提取代码中定义的类/函数
    # ============================================================

    @staticmethod
    def _defined_classes(code: str) -> set:
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return set()
        return {
            node.name for node in ast.walk(tree)
            if isinstance(node, ast.ClassDef)
        }

    @staticmethod
    def _defined_functions(code: str) -> set:
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return set()
        return {
            node.name for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }

    @staticmethod
    def _sig_from_node(node) -> dict:
        """从单个 FunctionDef/AsyncFunctionDef 节点提取签名 dict。"""
        a = node.args
        pos_params = [p.arg for p in a.posonlyargs] + [p.arg for p in a.args]
        kwonly_params = [p.arg for p in a.kwonlyargs]
        n_pos = len(pos_params)
        n_defaults = len(a.defaults)  # 只覆盖 a.args 尾部；posonly 默认值罕见，忽略
        n_required = max(0, n_pos - n_defaults)
        varargs = a.vararg is not None
        has_kwarg = a.kwarg is not None

        required_pos = set(pos_params[:n_required])
        required_kwonly = {p for p, d in zip(kwonly_params, a.kw_defaults) if d is None}

        all_params = set(pos_params) | set(kwonly_params)
        if a.vararg is not None:
            all_params.add(a.vararg.arg)
        if a.kwarg is not None:
            all_params.add(a.kwarg.arg)

        return {
            "min": n_required,
            "max": None if varargs else n_pos,
            "varargs": varargs,
            "kwarg": has_kwarg,
            "pos_params": pos_params,
            "kwonly_params": kwonly_params,
            "required_pos": required_pos,
            "required_kwonly": required_kwonly,
            "all_params": all_params,
        }

    @staticmethod
    def _function_signatures(code: str) -> dict:
        """提取文件中所有函数（含方法/嵌套函数）的签名。

        返回 {name: {min, max, varargs, kwarg, pos_params, kwonly_params,
        required_pos, required_kwonly, all_params}}。
        """
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return {}
        return {
            node.name: CodeValidator._sig_from_node(node)
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }

    @staticmethod
    def _top_level_function_signatures(code: str) -> dict:
        """提取模块顶层函数签名（排除类方法、嵌套函数，避免 self 参数误报）。"""
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return {}
        return {
            node.name: CodeValidator._sig_from_node(node)
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }

    # ============================================================
    # 6. 函数接口检查（参数个数 / 关键字参数名 / 必填参数）
    # ============================================================

    def _check_function_interface(self, code: str, errors: list) -> bool:
        """检查 `from X import Y` 引入的函数，其调用接口是否与定义一致。"""
        errs = self._interface_errors(code)
        errors.extend(errs)
        return not errs

    def _interface_errors(self, code: str) -> list:
        """返回当前文件中所有函数调用接口不匹配的错误描述。

        覆盖两类调用:
          1. 跨文件: `from X import Y` 引入的函数
          2. 同文件: 本文件内定义的顶层函数
        """
        if not code.strip():
            return []
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return []

        # 收集当前文件的 `from <项目模块> import Y` → (源文件, 原名)
        imports = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                top = node.module.split(".")[0]
                if top not in self.project_pkgs:
                    continue
                fpath = node.module.replace(".", "/") + ".py"
                for alias in node.names:
                    if alias.name == "*":
                        continue
                    imports[alias.asname or alias.name] = (fpath, alias.name)

        # 同文件顶层函数签名（类方法/嵌套函数不参与，避免 self 误报）
        local_sigs = self._top_level_function_signatures(code)

        errors = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
                continue  # 跳过 obj.method(...) 等属性调用
            name = node.func.id

            # 1) 跨文件调用
            if name in imports:
                fpath, orig = imports[name]
                syms = self.symbols.get(fpath)
                sig = (syms or {}).get("signatures", {}).get(orig)
                if not sig:
                    continue  # 依赖文件尚未登记，留待项目级检查
                err = self._check_single_call(name, orig, fpath, node, sig)
                if err:
                    errors.append(err)
                continue

            # 2) 同文件调用
            if name in local_sigs:
                err = self._check_single_call(name, name, "本文件", node, local_sigs[name])
                if err:
                    errors.append(err)
        return errors

    @staticmethod
    def _check_single_call(name, orig, fpath, node, sig):
        """校验单次函数调用，返回错误描述字符串；无错误返回 None。"""
        pos_params = sig["pos_params"]
        all_params = sig["all_params"]
        n_pos = len(node.args)

        # 1) 位置参数过多
        if sig["max"] is not None and n_pos > sig["max"]:
            return (
                f"函数接口不匹配: {name}(...) 传了 {n_pos} 个位置参数，"
                f"但 {fpath} 中 {orig} 最多接受 {sig['max']} 个位置参数"
            )

        # 2) 关键字参数名是否合法 + 收集按关键字提供的参数
        covered_by_pos = set(pos_params[:n_pos])
        provided_by_kw = set()
        for kw in node.keywords:
            if kw.arg is None:
                continue  # f(**dict) 无法静态推断，跳过
            if not sig["kwarg"] and kw.arg not in all_params:
                return (
                    f"函数接口不匹配: {name}(...) 传了未知关键字参数 '{kw.arg}'，"
                    f"但 {fpath} 中 {orig} 的可接受参数为 {sorted(all_params)}"
                )
            provided_by_kw.add(kw.arg)

        # 3) 同一参数同时按位置和关键字传入
        dup = covered_by_pos & provided_by_kw
        if dup:
            return (
                f"函数接口不匹配: {name}(...) 参数 '{sorted(dup)[0]}' 同时按位置和关键字传入"
            )

        # 4) 必填参数缺失
        provided = covered_by_pos | provided_by_kw
        missing = (
            [p for p in sorted(sig["required_pos"]) if p not in provided]
            + [p for p in sorted(sig["required_kwonly"]) if p not in provided_by_kw]
        )
        if missing:
            required = sorted(sig["required_pos"] | sig["required_kwonly"])
            return (
                f"函数接口不匹配: {name}(...) 缺少必填参数 {missing}，"
                f"但 {fpath} 中 {orig} 要求 {required}"
            )
        return None

    # ============================================================
    # 7. 结论写死检查（论文结论应动态计算，而非硬编码）
    # ============================================================

    def _check_conclusions(self, file: str, code: str, errors: list) -> bool:
        """检查 .py / .md 中是否写死了论文结论性措辞。"""
        if not (file.endswith(".py") or file.endswith(".md")):
            return True
        ok = True
        for i, line in enumerate(code.splitlines(), 1):
            low = line.lower()
            for phrase in self.BANNED_CONCLUSIONS:
                if phrase in low:
                    errors.append(
                        f"写死了论文结论: 第 {i} 行含 '{phrase}'，"
                        f"应改为基于实际评估结果动态比较（如输出布尔值/差值），不要断言为固定事实"
                    )
                    ok = False
        return ok

    # ============================================================
    # 项目级检查（所有文件生成后调用）
    # ============================================================

    def validate_project(self, plan_files: list, generated: dict, brief: dict) -> dict:
        """跨文件/项目级检查，返回 {检查名: {status, ...}}。

        - files        文件检查：计划中的文件是否都生成且非空
        - third_party  第三方依赖检查：import 的第三方包是否在 requirements.txt 声明
        - traceability 需求追踪检查：FR → 实现符号（按 FR 分组）
        - decisions    Step6 决策一致性检查：config.yaml 是否与 Step6 decisions 一致
        """
        return {
            "files": self._check_files(plan_files, generated),
            "third_party": self._check_third_party(generated),
            "function_interface": self._check_function_interface_project(generated),
            "traceability": self._check_traceability(brief, generated),
            "decisions": self._check_decisions(brief),
            "unresolved": self._check_unresolved(brief),
        }

    def _check_function_interface_project(self, generated: dict) -> dict:
        """跨文件函数接口一致性检查：所有文件生成后，用完整符号表复查。"""
        errors = []
        for f, code in generated.items():
            if f.endswith(".py"):
                errors.extend(self._interface_errors(code))
        errors = list(dict.fromkeys(errors))  # 去重，保持顺序
        return {"status": "passed" if not errors else "failed", "errors": errors}

    def _check_files(self, plan_files: list, generated: dict) -> dict:
        errors = []
        for f in plan_files:
            if not (generated.get(f) or "").strip():
                errors.append(f"缺少文件或文件为空: {f}")
        extra = set(generated) - set(plan_files)
        for f in sorted(extra):
            errors.append(f"生成了计划之外的文件: {f}")
        return {"status": "passed" if not errors else "failed", "errors": errors}

    def _check_third_party(self, generated: dict) -> dict:
        imported = set()
        for f, code in generated.items():
            if f.endswith(".py"):
                imported |= self._top_level_imports(code)

        # 本地包目录从实际生成的 .py 文件推导（含 src/algorithm 等非 ML 包）
        project_pkgs = {
            f.split("/")[0] for f in generated if f.endswith(".py") and "/" in f
        }
        # 额外识别：生成的 .py 文件的模块名（不含目录前缀和 .py 后缀）
        # 用于处理 sys.path 动态添加目录后的本地导入（如 main.py 中 sys.path.insert(0, "src")）
        local_module_names = set()
        for f in generated:
            if f.endswith(".py"):
                # 取文件名（不含 .py）作为模块名
                module_name = f.rsplit("/", 1)[-1].replace(".py", "")
                if module_name != "__init__":
                    local_module_names.add(module_name)
                # 也取不含目录的路径作为可能的包名（如 src/foo.py -> foo）
                if "/" in f:
                    parts = f.replace(".py", "").split("/")
                    if len(parts) >= 2:
                        local_module_names.add(parts[-1])

        third_party = {
            name for name in imported
            if name not in STDLIB and name not in project_pkgs and name not in local_module_names
        }

        reqs = self._parse_requirements(generated.get("requirements.txt", ""))

        errors = []
        for name in sorted(third_party):
            req_name = self.IMPORT_TO_REQ.get(name, name.lower())
            if req_name not in reqs:
                errors.append(
                    f"第三方依赖未声明: 代码 import '{name}' 但 requirements.txt 缺少 '{req_name}'"
                )
        return {"status": "passed" if not errors else "failed", "errors": errors}

    def _check_traceability(self, brief: dict, generated: dict) -> dict:
        """FR → 实现符号，按 requirement_id 分组（一个 FR 一条记录）。"""
        symbols = {}
        for f, code in generated.items():
            if f.endswith(".py"):
                symbols[f] = {
                    "classes": self._defined_classes(code),
                    "functions": self._defined_functions(code),
                }

        records = []
        all_ok = True
        for entry in brief.get("traceability", []):
            impls = []
            for impl in entry.get("planned_implementation", []):
                file, _, symbol = impl.partition("::")
                syms = symbols.get(file, {"classes": set(), "functions": set()})
                ok = symbol in syms["classes"] or symbol in syms["functions"]
                if not ok:
                    all_ok = False
                impls.append({
                    "file": file,
                    "symbol": symbol,
                    "status": "passed" if ok else "failed",
                })
            records.append({
                "requirement_id": entry.get("requirement_id"),
                "requirement_name": entry.get("requirement_name"),
                "status": "passed"
                          if all(i["status"] == "passed" for i in impls) else "failed",
                "implementations": impls,
            })
        return {"status": "passed" if all_ok else "failed", "records": records}

    def _check_decisions(self, brief: dict) -> dict:
        """校验 config.yaml 与 Step6 decisions 是否一致。"""
        decisions = brief.get("decisions", {})
        if not decisions or self.config is None:
            return {"status": "passed", "errors": []}
        errors = []

        expected_val = decisions.get("val_split_ratio")
        if expected_val is not None:
            actual = (self.config.get("split_strategy") or {}).get("val_split_ratio")
            if actual != expected_val:
                errors.append(
                    f"决策不一致: val_split_ratio 应为 {expected_val}(80/20)，"
                    f"但 config.yaml 为 {actual}"
                )

        expected_fw = decisions.get("framework")
        if expected_fw:
            actual = str((self.config.get("project") or {}).get("framework", "")).lower()
            if expected_fw.lower() not in actual:
                errors.append(
                    f"决策不一致: framework 应为 {expected_fw}，但 config.yaml 为 {actual}"
                )

        expected_modes = decisions.get("training_modes")
        if expected_modes:
            tm = self.config.get("training_modes") or {}
            if isinstance(tm, dict):
                actual = set(tm.keys())
            elif isinstance(tm, list):
                actual = set(tm)
            else:
                actual = set()
            missing = [m for m in expected_modes if m not in actual]
            if missing:
                errors.append(f"决策不一致: training_modes 缺少 {missing}")

        expected_bm = decisions.get("best_metric")
        if expected_bm:
            actual = (self.config.get("checkpoint") or {}).get("best_metric")
            if actual != expected_bm:
                errors.append(
                    f"决策不一致: checkpoint.best_metric 应为 {expected_bm}，但为 {actual}"
                )

        # 数据依赖字段一致性：config.yaml 顶层 data_source/acquisition 应与 Step6 决策一致
        expected_ds = decisions.get("data_source")
        if expected_ds:
            actual = self.config.get("data_source")
            if actual != expected_ds:
                errors.append(
                    f"决策不一致: 顶层 data_source 应为 {expected_ds}，但 config.yaml 为 {actual}"
                )
        expected_acq = decisions.get("acquisition")
        if expected_acq:
            actual = self.config.get("acquisition")
            if actual != expected_acq:
                errors.append(
                    f"决策不一致: 顶层 acquisition 应为 {expected_acq}，但 config.yaml 为 {actual}"
                )

        return {"status": "passed" if not errors else "failed", "errors": errors}

    def _check_unresolved(self, brief: dict) -> dict:
        """校验未定义来源的 dataset version 是否在 config.yaml 中标记为 unresolved。

        返回 status 语义（区别于其他检查的 passed/failed）：
        - "none"     无未定义来源的 dataset version
        - "detected" 存在未定义来源的版本，且已正确标记为 unresolved（这是"识别到了"，不是"失败"）
        - "failed"   存在未定义来源的版本，但 config.yaml 未正确标记
        """
        unresolved = brief.get("unresolved_dataset_versions", [])
        items = [u.get("name") for u in unresolved if u.get("name")]
        if not items:
            return {"status": "none", "items": [], "errors": []}
        if self.config is None:
            return {"status": "detected", "items": items, "errors": []}
        errors = []
        data = self.config.get("data", {}) if isinstance(self.config, dict) else {}
        for u in unresolved:
            name = u.get("name")
            if not name:
                continue
            marker = data.get(name)
            if not isinstance(marker, dict) or marker.get("status") != "unresolved":
                errors.append(
                    f"未定义来源的 dataset version '{name}' 未在 config.yaml 中标记为 "
                    f"{{status: unresolved}}"
                )
        status = "failed" if errors else "detected"
        return {"status": status, "items": items, "errors": errors}

    # ============================================================
    # 辅助：第三方依赖解析
    # ============================================================

    @staticmethod
    def _top_level_imports(code: str) -> set:
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return set()
        names = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    names.add(alias.name.split(".")[0])
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                names.add(node.module.split(".")[0])
        return names

    @staticmethod
    def _parse_requirements(text: str) -> set:
        reqs = set()
        for line in text.splitlines():
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("-"):
                continue
            name = re.split(r"[<>=!~\[\s]", line, 1)[0].strip().lower()
            if name:
                reqs.add(name)
        return reqs

    # ============================================================
    # 辅助：提取 config[...] / config.get(...) 的静态 key 链
    # ============================================================

    @staticmethod
    def _const_str(node) -> str:
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value
        return None

    def _config_access_chains(self, code: str) -> list:
        """返回代码中所有 `config[...]` 访问的静态字符串 key 链（去重）。"""
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return []

        chains = set()
        for node in ast.walk(tree):
            chain = self._chain_from_config_access(node)
            if chain:
                chains.add(tuple(chain))
        return [list(c) for c in chains]

    def _chain_from_config_access(self, node):
        """对某个 AST 节点，若它是 config 访问表达式，返回其静态 key 链；否则 None。"""
        if isinstance(node, ast.Subscript):
            key = self._const_str(node.slice)
            if isinstance(node.value, ast.Name) and node.value.id == "config":
                return [key] if key else None
            base = self._chain_from_config_access(node.value)
            if base is None:
                return None
            # 动态 key（变量）无法静态校验：直接跳过整条访问，避免把
            # config["data"][version]["status"] 误判成 config["data"]["status"] 的假阳性
            return None if key is None else base + [key]

        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) \
                and node.func.attr == "get":
            key = self._const_str(node.args[0]) if node.args else None
            if isinstance(node.func.value, ast.Name) and node.func.value.id == "config":
                return [key] if key else None
            base = self._chain_from_config_access(node.func.value)
            if base is None:
                return None
            # 动态 key（变量）无法静态校验：直接跳过整条访问，避免把
            # config["data"][version]["status"] 误判成 config["data"]["status"] 的假阳性
            return None if key is None else base + [key]

        return None
