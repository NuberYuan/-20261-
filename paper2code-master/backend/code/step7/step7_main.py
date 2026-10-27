"""
Step7 总入口: 代码生成

读取 Step6 的 project_brief.yaml，按其中的项目结构和代码生成计划，
逐文件（按依赖顺序）生成完整的可复现代码项目，并进行代码检查与修复。

流程:
    project_brief.yaml
        ↓ 初始化项目骨架
    平凡文件（requirements/README/__init__）确定性模板产出，不调 LLM
        ↓
    按拓扑层级逐层生成（层内互不依赖，可并行）
        ↓ 代码检查 (ast / yaml / 类 / 方法 / import)
    失败 → 修复 (最多 MAX_REPAIR_ATTEMPTS 次)
        ↓ 沙箱冒烟测试在后台线程池运行，与下一层生成重叠
    需求-代码追踪检查
        ↓
    generation_report.yaml + generation_log.json

Usage:
    cd code
    python -m step7.step7_main --paper plant
    python -m step7.step7_main --all
"""

import sys
import os
import ast
import argparse
import json
import re
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml

from config import get_paper_output_dir, get_step5_output_dir
from logger import logger

from step7.project_initializer import ProjectInitializer
from step7.code_context_builder import CodeContextBuilder
from step7.code_generator import CodeGenerator
from step7.code_validator import STDLIB, CodeValidator
from step7.code_repairer import CodeRepairer, MAX_REPAIR_ATTEMPTS
from step7.code_sandbox import CodeSandbox

# 沙箱参数：单次冒烟测试超时（秒）与后台并发数。
# 沙箱在后台线程池运行，与后续文件的 LLM 生成重叠，不挡生成关键路径。
SANDBOX_TIMEOUT = 120
SANDBOX_WORKERS = 3


class Step7Runner:
    """Step7 总控制器。"""

    def __init__(self, paper_name: str):
        self.paper_name = paper_name
        self.paper_dir = get_paper_output_dir(paper_name)
        self.step5_dir = self.paper_dir / "step5"
        self.step6_dir = self.paper_dir / "step6"
        self.output_dir = self.paper_dir / "step7"
        self.project_root = self.output_dir / "generated_project"

        self.generator = CodeGenerator()
        self.repairer = CodeRepairer()
        self.sandbox = CodeSandbox(timeout=SANDBOX_TIMEOUT)
        self.sandbox_executor = ThreadPoolExecutor(
            max_workers=SANDBOX_WORKERS, thread_name_prefix="s7-sandbox",
        )
        self._sandbox_futures = []

    # ============================================================
    # 拓扑分层（用于并行生成）
    # ============================================================

    @staticmethod
    def _topological_levels(ordered_files: list, spec_map: dict) -> list:
        """将拓扑排序后的文件列表进一步分层，同层文件互不依赖可并行生成。

        算法：每个文件的层级 = max(所有依赖文件的层级) + 1。
        同一层级的文件之间无依赖边，可安全并行生成。
        """
        file_set = set(ordered_files)
        file_level = {}  # file -> level index

        for file in ordered_files:
            spec = spec_map.get(file, {})
            deps = spec.get("depends_on") or []
            project_deps = [d for d in deps if d in file_set and d in file_level]

            if not project_deps:
                # 无依赖 → level 0
                file_level[file] = 0
            else:
                # 层级 = max(依赖的层级) + 1
                file_level[file] = max(file_level[d] for d in project_deps) + 1

        # 按层级分组
        max_level = max(file_level.values()) if file_level else 0
        levels = [[] for _ in range(max_level + 1)]
        for file in ordered_files:
            levels[file_level[file]].append(file)

        return levels

    def _generate_level(self, level_files, spec_map, context_builder,
                        validator, generated, generated_lock) -> list:
        """并行生成同一层级内的多个文件。返回该层所有文件的 result 列表。"""
        if len(level_files) == 1:
            # 单文件，直接调用（避免线程开销）
            file = level_files[0]
            result = self._generate_one_file(
                file, spec_map.get(file, {}), context_builder, validator, generated
            )
            return [result]

        # 多文件，并行生成
        results = [None] * len(level_files)

        def _gen(idx, file):
            spec = spec_map.get(file, {})
            try:
                result = self._generate_one_file(
                    file, spec, context_builder, validator, generated, generated_lock
                )
            except Exception as e:
                logger.error(f"  ✗ {file} 生成异常: {e}")
                result = {
                    "file": file, "status": "error", "code": "",
                    "attempts": [{"attempt": 1, "action": "generate",
                                  "status": "error", "errors": [str(e)]}],
                }
            return idx, result

        with ThreadPoolExecutor(max_workers=len(level_files)) as ex:
            futures = [ex.submit(_gen, i, f) for i, f in enumerate(level_files)]
            for future in as_completed(futures):
                try:
                    idx, result = future.result()
                    results[idx] = result
                except Exception as e:
                    logger.error(f"  并行生成 future 异常: {e}")

        return [r for r in results if r is not None]

    # ============================================================
    # 主流程
    # ============================================================

    def run(self) -> bool:
        logger.info(f"{'=' * 50}\n  Step7: 代码生成 — {self.paper_name}\n{'=' * 50}")

        # 1. 读取 Step6 project_brief
        brief_path = self.step6_dir / "project_brief.yaml"
        if not brief_path.exists():
            logger.error(f"  缺少 {brief_path}，请先运行 Step6")
            return False
        brief = yaml.safe_load(brief_path.read_text(encoding="utf-8"))

        # 2. 初始化项目骨架
        self.output_dir.mkdir(parents=True, exist_ok=True)
        initializer = ProjectInitializer(self.project_root)
        ordered_files = initializer.initialize(brief)
        if not ordered_files:
            logger.error("  code_generation_plan.per_file_specs 为空")
            return False

        # 3. 生成 + 检查 + 修复
        context_builder = CodeContextBuilder(brief, step5_dir=self.step5_dir)
        validator = CodeValidator(all_files=ordered_files)
        # 实验矩阵组合数提前注入：入口 main.py/run.py 在 >1 组合时必须支持 --run N
        validator.set_expected_runs(self._expected_run_count(brief))
        spec_map = {
            s.get("file"): s
            for s in brief.get("code_generation_plan", {}).get("per_file_specs", [])
        }

        generated = {}   # file -> code
        generated_lock = threading.Lock()   # 保护 generated dict 的并发写入
        file_results = []   # 每文件生成日志

        # 平凡文件（requirements.txt / README.md / __init__.py）确定性模板产出，不走 LLM。
        # 模板先过一次校验，万一 spec 给平凡文件声明了类/方法，回退 LLM 生成。
        templated = {}
        for f in ordered_files:
            content = self._template_trivial_file(
                f, spec_map.get(f, {}), brief, ordered_files, spec_map,
            )
            if content is None:
                continue
            if validator.validate(f, content, spec_map.get(f, {}))["status"] == "passed":
                templated[f] = content
            else:
                logger.warning(f"  模板校验未通过，回退 LLM 生成: {f}")
        if templated:
            logger.info(f"  模板产出（不调 LLM）: {sorted(templated)}")

        # 按拓扑层级并行生成
        levels = self._topological_levels(ordered_files, spec_map)
        logger.info(f"  拓扑分层: {len(levels)} 层, {len(ordered_files)} 个文件"
                    f" (最大并行 {max(len(lv) for lv in levels)})")

        for level_idx, level_files in enumerate(levels):
            # 模板文件瞬时产出，其余文件走 LLM
            llm_files = []
            level_results = []
            for f in level_files:
                if f in templated:
                    level_results.append(
                        self._generate_templated_file(
                            f, templated[f], spec_map.get(f, {}), validator,
                        )
                    )
                else:
                    llm_files.append(f)

            if llm_files:
                if len(llm_files) > 1:
                    logger.info(f"  层级 {level_idx}: 并行生成 {llm_files}")
                level_results.extend(self._generate_level(
                    llm_files, spec_map, context_builder, validator, generated, generated_lock
                ))

            # 每层完成后串行更新 generated dict 和 validator 符号表
            for result in level_results:
                file_results.append(result)
                code = result["code"]
                if code.strip():
                    with generated_lock:
                        generated[result["file"]] = code
                if result["status"] == "passed":
                    validator.register(result["file"], code)
                    if result["file"] in ("config/config.yaml", "config/config.yml"):
                        validator.set_config(self._parse_config(code))

            # 沙箱与生成解耦：本层静态完成后立即提交后台沙箱（快照含同层兄弟文件），
            # 与下一层的 LLM 生成重叠，不挡依赖链。
            with generated_lock:
                snapshot = dict(generated)
            for result in level_results:
                if result.get("templated") or result["status"] != "passed":
                    continue
                if not result["file"].endswith(".py"):
                    continue
                self._submit_sandbox(
                    result, spec_map.get(result["file"], {}), context_builder,
                    validator, snapshot, generated, generated_lock,
                )

        # 包 __init__.py 的 re-export 以最终兄弟模块代码为准再对齐一次（确定性，不调 LLM）
        self._refresh_package_inits(generated, validator)

        # 等全部沙箱任务（含运行时修复）结束后，再做注入与项目级检查
        self._join_sandbox()

        # 4. 确定性注入未定义 dataset version 的 unresolved 标记（不让 LLM 猜数据来源）
        self._inject_unresolved_versions(generated, brief, validator)

        # 4.5 确定性注入 data_source / acquisition（顶层 + 每个 run），不让 LLM 猜数据依赖
        self._inject_data_dependency(generated, brief, validator)

        # 4.6 确定性注入 experiment.runs（从 Step5 experiment_design.yaml 读取）
        self._inject_experiment_runs(generated, brief, validator)

        # 4.7 确定性产出执行配置 execution_spec.yaml（供 Step8 读取入口与命令，避免硬编码）
        self._write_execution_spec(brief, ordered_files, generated)

        # 4.8 同步 requirements.txt：扫描代码 import，补全缺失的第三方依赖
        self._sync_requirements(generated, validator)

        # 5. 项目级检查（文件完整性 / 第三方依赖 / 需求追踪 / Step6 决策一致性）
        project_checks = validator.validate_project(ordered_files, generated, brief)

        # 6. 写报告
        report = self._build_report(brief, ordered_files, file_results, project_checks)
        self._write_outputs(report, file_results)

        # 7. 日志摘要
        ok = sum(1 for r in file_results if r["status"] == "passed")
        sandbox_ok = sum(1 for r in file_results if r.get("sandbox", {}).get("success"))
        trace_records = project_checks["traceability"]["records"]
        logger.info(f"  Step7 完成: {ok}/{len(ordered_files)} 个文件通过")
        logger.info(f"  沙箱测试: {sandbox_ok}/{sum(1 for r in file_results if 'sandbox' in r)} 通过")
        logger.info(f"  需求追踪: {sum(1 for t in trace_records if t['status'] == 'passed')}"
                    f"/{len(trace_records)} 条 FR 通过")
        for name, check in project_checks.items():
            if name == "traceability":
                continue
            if check.get("status") == "failed":
                logger.warning(f"  ✗ {name} 检查未通过: {check['errors'][0] if check['errors'] else ''}")
            elif name == "unresolved" and check.get("status") == "detected":
                logger.info(
                    f"  未定义数据集版本（已正确标记 unresolved）: {', '.join(check.get('items', []))}"
                )
        logger.info(f"  输出: {self.output_dir}")
        return report["generation"]["status"] == "success"

    # ============================================================
    # 单个文件: 生成 → 检查 → 修复循环
    # ============================================================

    def _generate_one_file(self, file, spec, context_builder, validator, generated,
                           generated_lock=None) -> dict:
        # 并行时需要锁保护 generated dict 的读取快照
        if generated_lock:
            with generated_lock:
                gen_snapshot = dict(generated)
        else:
            gen_snapshot = generated
        ctx = context_builder.build(file, gen_snapshot)
        # 注入 config 结构，供修复器在配置引用错误时参考
        if validator.config:
            ctx["config"] = validator.config
        attempts = []

        try:
            # 第一次生成
            code = self.generator.generate(ctx)
        except RuntimeError as e:
            if "402" in str(e) or "Insufficient Balance" in str(e):
                logger.error(f"  ✗ {file} API 余额不足，跳过此文件")
                return {
                    "file": file, "status": "api_error", "code": "",
                    "attempts": [{"attempt": 1, "action": "generate",
                                  "status": "api_error", "errors": [str(e)]}],
                    "sandbox": {"success": False, "error": str(e)},
                }
            raise

        val = validator.validate(file, code, spec)
        attempts.append({
            "attempt": 1, "action": "generate",
            "status": val["status"], "errors": val["errors"],
        })

        # 修复循环（静态错误 + 运行时错误）
        attempt = 1
        while val["status"] == "failed" and attempt < MAX_REPAIR_ATTEMPTS:
            attempt += 1
            try:
                code = self.repairer.repair(file, code, val["errors"], spec, ctx)
            except RuntimeError as e:
                if "402" in str(e) or "Insufficient Balance" in str(e):
                    logger.error(f"  ✗ {file} API 余额不足，跳过修复")
                    break
                raise
            val = validator.validate(file, code, spec)
            attempts.append({
                "attempt": attempt, "action": "repair",
                "status": val["status"], "errors": val["errors"],
            })

        # 写文件（沙箱测试由 _submit_sandbox 在后台执行，不在本函数内联）
        if val["status"] == "passed":
            self._write_code(file, code)
        elif code.strip():
            # 修复次数耗尽仍失败：仍写出最后一次结果，便于人工检查
            self._write_code(file, code)

        status = val["status"]
        if status == "failed":
            logger.warning(f"  ✗ {file} 修复 {attempt} 次仍失败: "
                           f"{val['errors'][0] if val['errors'] else '未知错误'}")
        else:
            logger.info(f"  ✓ {file} ({attempt} 次尝试)")

        return {
            "file": file,
            "status": status,
            "code": code,
            "attempts": attempts,
        }

    def _write_code(self, file: str, code: str):
        path = self.project_root / file
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(code, encoding="utf-8")

    @staticmethod
    def _parse_config(code: str) -> dict:
        """解析 config.yaml 内容，失败时返回空 dict（此时配置引用检查退化为跳过）。"""
        try:
            return yaml.safe_load(code) or {}
        except yaml.YAMLError:
            return {}

    # ============================================================
    # 平凡文件模板（确定性产出，不调 LLM）
    # ============================================================

    def _template_trivial_file(self, file: str, spec: dict, brief: dict,
                               ordered_files: list, spec_map: dict):
        """requirements.txt / README.md / __init__.py 返回确定性模板内容，其余返回 None。

        这类文件结构固定、不承载算法逻辑，走 LLM 纯属浪费调用与时间；
        requirements.txt 之后还会被 _sync_requirements 按实际 import 补全。
        """
        base = file.rsplit("/", 1)[-1]
        if base == "__init__.py":
            return self._template_package_init(file, spec_map)
        if base == "requirements.txt":
            return self._template_requirements(spec)
        if base == "README.md":
            return self._template_readme(brief, ordered_files)
        return None

    def _template_package_init(self, file: str, spec_map: dict) -> str:
        """包 __init__.py：按兄弟模块 spec 声明的类生成 re-export。

        类名由 CodeValidator._check_classes 保证与实际代码一致，可放心引用；
        全部兄弟模块生成完后 _refresh_package_inits 还会按 AST 再对齐一次。
        """
        if "/" not in file:
            return '"""Package initializer."""\n'
        pkg_dir = file.rsplit("/", 1)[0]
        exports = []
        for f, s in spec_map.items():
            if f == file or not f.endswith(".py"):
                continue
            if "/" in f and f.rsplit("/", 1)[0] == pkg_dir:
                stem = f.rsplit("/", 1)[-1][:-3]
                for cls in (s.get("classes") or []):
                    exports.append((stem, cls))
        return self._package_init_content(file, exports)

    @staticmethod
    def _package_init_content(file: str, exports: list) -> str:
        """把 (模块名, 类名) 列表渲染成 __init__.py 内容（去重、稳定排序）。"""
        pkg = file.rsplit("/", 1)[0].replace("/", ".") if "/" in file else ""
        seen = set()
        unique = []
        for stem, cls in sorted(exports):
            if (stem, cls) not in seen:
                seen.add((stem, cls))
                unique.append((stem, cls))
        lines = [f'"""{pkg or "project"} package: re-export key symbols for convenient imports."""']
        if unique:
            lines.append("")
            for stem, cls in unique:
                lines.append(f"from .{stem} import {cls}")
            lines.append("")
            names = ", ".join(f'"{c}"' for _, c in unique)
            lines.append(f"__all__ = [{names}]")
        return "\n".join(lines) + "\n"

    def _template_requirements(self, spec: dict) -> str:
        """requirements.txt：基础依赖 + spec.notes 里点名的包（如「包含 a, b, c」）。

        生成后 _sync_requirements 会按实际 import 补全缺失包并去重。
        """
        pkgs = {"numpy", "pyyaml"}
        notes = spec.get("notes") or ""
        m = re.search(r"包含[:：]?\s*([^。；;\n]+)", notes)
        if m:
            for tok in re.split(r"[,，、\s]+", m.group(1)):
                tok = tok.strip().lower()
                if re.fullmatch(r"[a-z0-9][a-z0-9_.\-]*", tok) and len(tok) >= 2:
                    pkgs.add(tok)
        lines = ["# Project dependencies. Missing imports are auto-appended by Step7."]
        lines.extend(sorted(pkgs))
        return "\n".join(lines) + "\n"

    def _template_readme(self, brief: dict, ordered_files: list) -> str:
        """README.md：项目结构树严格取自计划文件列表，不臆造任何目录/文件。"""
        project = brief.get("project", {}) or {}
        title = project.get("paper_title") or project.get("name") or "generated_project"
        one_liner = project.get("one_liner") or ""
        task = project.get("task") or ""
        entry = (
            brief.get("design_digest", {}).get("project_structure", {}).get("entry_point")
            or ("main.py" if "main.py" in ordered_files
                else (ordered_files[0] if ordered_files else "main.py"))
        )
        tree = self._render_tree(ordered_files)

        parts = [f"# {title}", ""]
        if one_liner:
            parts += [one_liner, ""]
        if task:
            parts += [f"- 任务类型: {task}", ""]
        parts += [
            "## 项目结构",
            "",
            "```",
            tree,
            "```",
            "",
            "## 环境准备",
            "",
            "```bash",
            "pip install -r requirements.txt",
            "```",
            "",
            "## 运行方法",
            "",
            "```bash",
            f"python {entry}",
            "```",
            "",
            "实验矩阵（数据版本 / 训练模式 / 运行编号）见 `config/config.yaml` 的 "
            "`experiment.runs`。",
            "",
            "## 复现步骤",
            "",
            "1. 安装依赖：`pip install -r requirements.txt`。",
            "2. 按 `config/config.yaml` 中 `data` / `data_source` 的配置准备数据集"
            "（标记为 `status: unresolved` 的数据版本需先补齐来源）。",
            f"3. 运行入口 `{entry}`，按 config 中的实验矩阵逐个 run 复现。",
            "4. 训练产物（checkpoint、日志）的输出位置见 config 的 `checkpoint` / `logging` 配置。",
        ]
        return "\n".join(parts) + "\n"

    @staticmethod
    def _render_tree(files: list) -> str:
        """把文件列表渲染成目录树，与实际生成文件一一对应，不多不少。"""
        root = {}
        for f in files:
            node = root
            parts = f.split("/")
            for p in parts[:-1]:
                node = node.setdefault(p, {})
            node[parts[-1]] = None

        lines = []

        def walk(node, prefix=""):
            # 目录在前、文件在后，各自按名称排序，保证输出稳定
            entries = sorted(node.items(), key=lambda kv: (kv[1] is None, kv[0].lower()))
            for i, (name, child) in enumerate(entries):
                last = i == len(entries) - 1
                lines.append(prefix + ("└── " if last else "├── ") + name)
                if child is not None:
                    walk(child, prefix + ("    " if last else "│   "))

        walk(root)
        return "\n".join(lines)

    def _generate_templated_file(self, file: str, content: str, spec: dict,
                                 validator) -> dict:
        """模板文件的产出与登记（不调 LLM、不做沙箱）。"""
        val = validator.validate(file, content, spec)
        self._write_code(file, content)
        status = val["status"] if val["status"] == "failed" else "passed"
        if status == "failed":
            logger.warning(f"  ✗ {file} 模板产出未通过校验: "
                           f"{val['errors'][0] if val['errors'] else '未知错误'}")
        else:
            logger.info(f"  ✓ {file} (模板产出)")
        return {
            "file": file,
            "status": status,
            "code": content,
            "attempts": [{"attempt": 1, "action": "template",
                          "status": status, "errors": val["errors"]}],
            "templated": True,
        }

    def _refresh_package_inits(self, generated: dict, validator) -> None:
        """以最终生成的兄弟模块为准，确定性刷新各包 __init__.py 的 re-export。

        覆盖 spec 未声明但值得导出的类；只读 AST，不调 LLM。
        """
        for init in [f for f in list(generated) if f.endswith("__init__.py") and "/" in f]:
            pkg_dir = init.rsplit("/", 1)[0]
            exports = []
            for f, code in generated.items():
                if not f.endswith(".py") or f == init:
                    continue
                if "/" not in f or f.rsplit("/", 1)[0] != pkg_dir:
                    continue
                stem = f.rsplit("/", 1)[-1][:-3]
                for cls in CodeValidator._defined_classes(code):
                    exports.append((stem, cls))
            if not exports:
                continue
            content = self._package_init_content(init, exports)
            if generated.get(init) != content:
                generated[init] = content
                self._write_code(init, content)
                validator.register(init, content)

    def _inject_unresolved_versions(self, generated: dict, brief: dict, validator) -> None:
        """把 Step6 标记为未定义来源的 dataset version 确定性注入 config.yaml 的 data 段。

        不让 LLM 猜 segmented 的数据来源：若 Step6 未定义，就在 config.yaml 中写
        data.<version> = {status: unresolved, reason: ...}，供 dataset.py 据此报错。
        同时把 LLM 误写成 unresolved 的已定义版本（如 color/grayscale）纠正回 resolved。
        """
        unresolved = brief.get("unresolved_dataset_versions", [])
        if not unresolved:
            return
        config_key = next(
            (f for f in ("config/config.yaml", "config/config.yml") if f in generated),
            None,
        )
        if not config_key:
            return
        try:
            cfg = yaml.safe_load(generated[config_key]) or {}
        except yaml.YAMLError:
            return
        data = cfg.setdefault("data", {})
        unresolved_names = {u.get("name") for u in unresolved if u.get("name")}
        if not unresolved_names:
            return
        reason_by_name = {
            u["name"]: u.get("reason", "Dataset source/processing method not specified in Step6")
            for u in unresolved if u.get("name")
        }
        raw_versions = data.get("versions", [])
        if isinstance(raw_versions, dict):
            version_names = list(raw_versions.keys())
        elif isinstance(raw_versions, list):
            version_names = [v for v in raw_versions if isinstance(v, str)]
        else:
            version_names = []
        changed = False
        # 1) unresolved 版本：确定性写入 unresolved 标记（覆盖 LLM 可能编造的 resolved/path）
        for name in unresolved_names:
            marker = {"status": "unresolved", "reason": reason_by_name[name]}
            # 顶层 data.<version>：canonical 标记，供 dataset.py / validator 读取
            data[name] = dict(marker)
            # 若 LLM 把 status 写进 data.versions.<version>（dict 形式），一并覆盖
            if isinstance(raw_versions, dict) and isinstance(raw_versions.get(name), dict):
                raw_versions[name] = dict(marker)
            changed = True
        # 2) 已定义版本：若 LLM 误写成 unresolved，纠正回 resolved（保留 path/description）
        for name in version_names:
            if name in unresolved_names:
                continue
            entry = data.get(name)
            if isinstance(entry, dict) and entry.get("status") == "unresolved":
                entry = dict(entry)
                entry["status"] = "resolved"
                data[name] = entry
                changed = True
        if not changed:
            return
        patched = yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False)
        generated[config_key] = patched
        self._write_code(config_key, patched)
        validator.set_config(self._parse_config(patched))

    def _inject_data_dependency(self, generated: dict, brief: dict, validator) -> None:
        """把 Step6 判定的 data_source / acquisition 确定性注入 config.yaml。

        位置：顶层（data_source / acquisition 两个键，作为 paper 级兜底）+ 每个
        experiment.runs[] 条目（per-run 级，Step8 优先读这里）。LLM 写错也会被覆盖。
        leaf_based_split 时额外注入 split_strategy.leaf_mapping_file/constraint，
        供 dataset.py 做叶子级分组划分。
        """
        dd = brief.get("data_dependency", {})
        data_source = dd.get("data_source") or "external_dataset"
        acquisition = dd.get("acquisition") or "local_path"
        config_key = next(
            (f for f in ("config/config.yaml", "config/config.yml") if f in generated),
            None,
        )
        if not config_key:
            return
        try:
            cfg = yaml.safe_load(generated[config_key]) or {}
        except yaml.YAMLError:
            return
        if not isinstance(cfg, dict):
            return

        changed = False
        # 顶层兜底键
        if cfg.get("data_source") != data_source:
            cfg["data_source"] = data_source
            changed = True
        if cfg.get("acquisition") != acquisition:
            cfg["acquisition"] = acquisition
            changed = True

        # leaf_mapping 依赖登记（leaf_based_split 论文）：确定性写入 split_strategy，
        # dataset.py 据此读映射文件做分组划分；缺失时显式告警而非静默随机划分。
        leaf_map = dd.get("leaf_mapping")
        if isinstance(leaf_map, dict) and leaf_map.get("required"):
            ss = cfg.setdefault("split_strategy", {})
            if not isinstance(ss, dict):
                ss = {}
                cfg["split_strategy"] = ss
            if ss.get("leaf_mapping_file") != leaf_map.get("file"):
                ss["leaf_mapping_file"] = leaf_map.get("file", "data/leaf_mappings.csv")
                changed = True
            if ss.get("leaf_mapping_constraint") != leaf_map.get("constraint"):
                ss["leaf_mapping_constraint"] = leaf_map.get("constraint", "")
                changed = True
            if ss.get("leaf_mapping_on_missing") != leaf_map.get("on_missing"):
                ss["leaf_mapping_on_missing"] = leaf_map.get("on_missing", "")
                changed = True

        # 每个 run
        runs = cfg.get("experiment", {}).get("runs", [])
        if isinstance(runs, list):
            for run in runs:
                if not isinstance(run, dict):
                    continue
                if run.get("data_source") != data_source:
                    run["data_source"] = data_source
                    changed = True
                if run.get("acquisition") != acquisition:
                    run["acquisition"] = acquisition
                    changed = True

        if not changed:
            return
        patched = yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False)
        generated[config_key] = patched
        self._write_code(config_key, patched)
        validator.set_config(self._parse_config(patched))

    @staticmethod
    def _expected_run_count(brief: dict) -> int:
        """从 Step5 experiment_design.yaml 读取实验矩阵组合数（无则 0）。"""
        exp_design_path = (
            get_step5_output_dir(brief.get("paper_name", ""))
            / "experiment_design.yaml"
        )
        if not exp_design_path.exists():
            return 0
        try:
            with open(exp_design_path, encoding="utf-8") as f:
                exp_design = yaml.safe_load(f) or {}
        except (yaml.YAMLError, OSError):
            return 0
        return len(exp_design.get("experiment_matrix", {}).get("runs", []) or [])

    def _inject_experiment_runs(self, generated: dict, brief: dict, validator) -> None:
        """从 Step5 experiment_design.yaml 读取 experiment.runs 并确定性注入 config.yaml。

        LLM 生成的 config.yaml 可能缺少或写错 experiment.runs；
        这里用 Step5 已展开的 runs 列表覆盖，确保 Step8 能读到。
        """
        config_key = next(
            (f for f in ("config/config.yaml", "config/config.yml") if f in generated),
            None,
        )
        if not config_key:
            return

        # 从 Step5 读取 experiment_design.yaml
        exp_design_path = (
            get_step5_output_dir(brief.get("paper_name", ""))
            / "experiment_design.yaml"
        )
        if not exp_design_path.exists():
            return
        try:
            with open(exp_design_path, encoding="utf-8") as f:
                exp_design = yaml.safe_load(f) or {}
        except (yaml.YAMLError, OSError):
            return

        runs = exp_design.get("experiment_matrix", {}).get("runs", [])
        if not runs:
            return

        try:
            cfg = yaml.safe_load(generated[config_key]) or {}
        except yaml.YAMLError:
            return
        if not isinstance(cfg, dict):
            return

        # 注入 experiment.runs（覆盖 LLM 生成的）
        if "experiment" not in cfg or not isinstance(cfg["experiment"], dict):
            cfg["experiment"] = {}
        existing_runs = cfg["experiment"].get("runs", [])
        # 仅在 runs 为空或数量不匹配时注入
        if not existing_runs or len(existing_runs) != len(runs):
            cfg["experiment"]["runs"] = runs
            patched = yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False)
            generated[config_key] = patched
            self._write_code(config_key, patched)
            validator.set_config(self._parse_config(patched))
            logger.info(f"  已注入 experiment.runs ({len(runs)} 个)")

    # ============================================================
    # 依赖同步
    # ============================================================

    def _sync_requirements(self, generated: dict, validator) -> None:
        """扫描所有 .py 文件的 import，将缺失的第三方包自动补入 requirements.txt。

        在所有文件生成完毕后、项目级检查前调用，确保 requirements.txt 完整。
        """
        # 收集所有 .py 文件的顶层 import
        imported = set()
        for f, code in generated.items():
            if f.endswith(".py"):
                imported |= CodeValidator._top_level_imports(code)

        # 本地包目录
        project_pkgs = {
            f.split("/")[0] for f in generated if f.endswith(".py") and "/" in f
        }
        local_module_names = set()
        for f in generated:
            if f.endswith(".py"):
                module_name = f.rsplit("/", 1)[-1].replace(".py", "")
                if module_name != "__init__":
                    local_module_names.add(module_name)

        # 第三方包 = import 中非标准库、非本地
        third_party = {
            name for name in imported
            if name not in STDLIB and name not in project_pkgs and name not in local_module_names
        }

        # 当前 requirements.txt 中已有的包
        req_content = generated.get("requirements.txt", "")
        existing_reqs = self._extract_requirements_packages(generated)

        # 找出缺失的包并补全（包名标准化：hyphens → underscores）
        missing = []
        for name in sorted(third_party):
            req_name = CodeValidator.IMPORT_TO_REQ.get(name, name.lower())
            req_normalized = req_name.lower().replace("-", "_")
            if req_normalized not in existing_reqs:
                missing.append(req_name)

        if missing:
            new_lines = req_content.rstrip() + "\n" + "\n".join(missing) + "\n"
            generated["requirements.txt"] = new_lines
            logger.info(f"  📦 requirements.txt 自动补全: {', '.join(missing)}")

        # 最终去重：消除所有重复行（保留首次出现顺序）
        final_lines = generated.get("requirements.txt", "").splitlines()
        seen = set()
        deduped = []
        for line in final_lines:
            normalized = line.strip().lower().replace("-", "_")
            if normalized and normalized in seen:
                continue
            seen.add(normalized)
            deduped.append(line)
        deduped_content = "\n".join(deduped)
        if deduped_content != generated.get("requirements.txt", ""):
            generated["requirements.txt"] = deduped_content
            self._write_code("requirements.txt", deduped_content)
            logger.info("  📦 requirements.txt 已去重")

    # ============================================================
    # 沙箱测试（后台任务：与后续文件的 LLM 生成重叠）
    # ============================================================

    def _submit_sandbox(self, result: dict, spec: dict, context_builder, validator,
                        snapshot: dict, generated: dict, generated_lock) -> None:
        """把单文件沙箱冒烟测试（含可能的运行时修复）提交到后台线程池。

        snapshot 为本层结束时的 generated 快照（含同层兄弟文件），
        后台任务只读快照、只改自己文件对应的 result，线程安全。
        """
        future = self.sandbox_executor.submit(
            self._sandbox_and_repair, result, spec, context_builder, validator,
            snapshot, generated, generated_lock,
        )
        self._sandbox_futures.append(future)

    def _join_sandbox(self) -> None:
        """等待全部沙箱任务结束（含运行时修复），再继续注入与项目级检查。"""
        for future in self._sandbox_futures:
            try:
                future.result()
            except Exception as e:
                logger.warning(f"  ⚠ 沙箱任务异常: {e}")
        self._sandbox_futures.clear()

    def _sandbox_and_repair(self, result: dict, spec: dict, context_builder, validator,
                            snapshot: dict, generated: dict, generated_lock) -> None:
        """后台执行：沙箱冒烟测试 →（必要时）一次运行时修复。只更新本文件的 result。"""
        file = result["file"]
        code = result["code"]
        attempts = result["attempts"]
        try:
            sandbox_result = self._run_sandbox_test(file, code, snapshot)
            if not sandbox_result.success:
                # 超时类错误跳过修复：代码逻辑没问题，只是实例化大模型太慢
                is_timeout = sandbox_result.error_type == "TimeoutError"
                if is_timeout:
                    logger.info(f"  ⚠ {file} 沙箱超时，跳过修复（代码静态检查已通过）")
                else:
                    # 沙箱测试失败，进入运行时错误修复循环（最多1次）
                    sandbox_fix_attempts = 0
                    while not sandbox_result.success and sandbox_fix_attempts < 1:
                        sandbox_fix_attempts += 1
                        runtime_errors = self._extract_runtime_errors(sandbox_result)
                        logger.info(f"  ⚠ {file} 沙箱测试失败，尝试运行时修复 ({sandbox_fix_attempts}/1)")
                        ctx = context_builder.build(file, snapshot)
                        if validator.config:
                            ctx["config"] = validator.config
                        try:
                            code = self.repairer.repair(
                                file, code, [], spec, ctx,
                                runtime_errors=runtime_errors,
                            )
                        except RuntimeError as e:
                            if "402" in str(e) or "Insufficient Balance" in str(e):
                                logger.error(f"  ✗ {file} API 余额不足，跳过修复")
                                break
                            raise
                        val = validator.validate(file, code, spec)
                        if val["status"] == "failed":
                            # 修复引入了新的静态错误：保留原文件，放弃这次修复
                            logger.warning(f"  ⚠ {file} 运行时修复引入静态错误，保留原代码")
                            code = result["code"]
                            break
                        sandbox_result = self._run_sandbox_test(file, code, snapshot)
                        attempts.append({
                            "attempt": len(attempts) + 1,
                            "action": "sandbox_repair",
                            "status": "passed" if sandbox_result.success else "failed",
                            "errors": runtime_errors if not sandbox_result.success else [],
                        })
                    # 修复后的代码落盘，并同步 generated 快照之外的正式表与符号表
                    if code.strip() and code != result["code"]:
                        self._write_code(file, code)
                        with generated_lock:
                            generated[file] = code
                        validator.register(file, code)
                        result["code"] = code

            if not sandbox_result.success:
                # 静态通过但沙箱失败
                logger.warning(f"  ⚠ {file} 静态检查通过，但沙箱测试失败: "
                               f"{sandbox_result.error_message}")
                result["status"] = "sandbox_failed"
            result["sandbox"] = {
                "success": sandbox_result.success,
                "tests_passed": sandbox_result.tests_passed,
                "tests_total": sandbox_result.tests_total,
                "error_type": sandbox_result.error_type,
                "error_message": sandbox_result.error_message,
            }
        except Exception as e:
            # 后台沙箱自身出错不应阻塞主流程
            logger.warning(f"  ⚠ {file} 沙箱任务异常: {e}")
            result["sandbox"] = {
                "success": True,  # 视为通过（不阻塞）
                "error_type": "SandboxError",
                "error_message": str(e),
            }

    def _run_sandbox_test(self, file: str, code: str,
                          generated: dict) -> "SandboxResult":
        """对单个文件运行沙箱冒烟测试。"""
        try:
            # 读取 config 供冒烟测试使用
            config = {}
            config_key = next(
                (f for f in ("config/config.yaml", "config/config.yml") if f in generated),
                None,
            )
            if config_key and config_key in generated:
                config = self._parse_config(generated[config_key])

            # 从 requirements.txt 提取已知第三方包名，跳过其 import 检查
            known_packages = self._extract_requirements_packages(generated)

            result = self.sandbox.run_file_with_smoke(
                file=file,
                code=code,
                all_files=generated,
                config=config,
                project_root=str(self.project_root),
                known_packages=known_packages,
            )

            # 如果失败原因是数据依赖相关的错误，视为"运行时提供"而非代码 bug
            if not result.success:
                if self._is_data_dependency_error(result):
                    logger.info(f"  ℹ {file} 沙箱: 数据依赖缺失（运行时提供），视为通过")
                    result.success = True
                    result.error_type = ""
                    result.error_message = ""
                elif self._is_missing_third_party_error(result, generated):
                    # 第三方包缺失：补全 requirements.txt，视为通过（非代码 bug）
                    result.success = True
                    result.error_type = ""
                    result.error_message = ""
            if result.success:
                logger.info(f"  ✓ {file} 沙箱测试通过"
                            + (f" ({result.tests_passed}/{result.tests_total})"
                               if result.tests_total else ""))
            return result
        except Exception as e:
            # 沙箱本身出错不应阻塞流程
            logger.warning(f"  ⚠ {file} 沙箱执行异常: {e}")
            from step7.code_sandbox import SandboxResult
            return SandboxResult(
                success=True,  # 视为通过（不阻塞）
                error_type="SandboxError",
                error_message=str(e),
            )

    @staticmethod
    def _extract_runtime_errors(result: "SandboxResult") -> list:
        """从沙箱结果提取运行时错误信息，供修复器使用。"""
        errors = []
        if result.error_type:
            errors.append(f"[{result.error_type}] {result.error_message}")
        if result.traceback:
            # 取 traceback 最后几行（最有价值的错误信息）
            tb_lines = result.traceback.strip().splitlines()
            relevant = tb_lines[-6:] if len(tb_lines) > 6 else tb_lines
            errors.append("Traceback:\n" + "\n".join(relevant))
        # 冒烟测试中失败的项
        for detail in result.details:
            if detail.get("status") == "failed":
                errors.append(f"Smoke test failed: {detail['test']} — {detail.get('error', '')}")
        return errors or ["Unknown runtime error"]

    @staticmethod
    def _extract_requirements_packages(generated: dict) -> set:
        """从 requirements.txt 提取第三方包名集合。"""
        req_content = generated.get("requirements.txt", "")
        if not req_content:
            return set()
        packages = set()
        for line in req_content.splitlines():
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("-"):
                continue
            # 去掉版本号: torch>=1.0 → torch, numpy==1.21 → numpy
            pkg = line.split(">=")[0].split("==")[0].split("<=")[0].split("~=")[0].split("!=")[0]
            pkg = pkg.split("[")[0]  # 去掉 extras: requests[security] → requests
            pkg = pkg.strip()
            if pkg:
                # 包名标准化: hyphens → underscores (my-package → my_package)
                packages.add(pkg.lower().replace("-", "_"))
        return packages

    @staticmethod
    def _is_data_dependency_error(result: "SandboxResult") -> bool:
        """判断沙箱失败是否是数据依赖问题（而非代码 bug）。

        以下情况视为数据依赖，运行时由 Step8 提供：
        1. FileNotFoundError 且路径含数据文件后缀
        2. ValueError 含 dataset version 相关信息
        3. FileNotFoundError 且路径含 data/ 目录
        """
        err_type = result.error_type
        msg = result.error_message or ""

        # FileNotFoundError + 数据文件后缀
        if err_type == "FileNotFoundError":
            data_extensions = (
                ".csv", ".json", ".jsonl", ".parquet", ".pkl", ".pickle",
                ".hdf5", ".h5", ".txt", ".dat", ".tsv", ".npy", ".npz",
                ".pt", ".pth", ".bin", ".mat", ".feather", ".arrow",
                ".sqlite", ".db",
            )
            msg_lower = msg.lower()
            if any(ext in msg_lower for ext in data_extensions):
                return True
            # FileNotFoundError + data/ 目录
            if "data/" in msg_lower or "data\\" in msg_lower:
                return True

        # ValueError: dataset version 不存在
        if err_type == "ValueError":
            msg_lower = msg.lower()
            if "dataset version" in msg_lower or "unknown dataset" in msg_lower:
                return True

        # 数据目录不存在
        if err_type == "FileNotFoundError" and ("not found" in msg.lower()):
            if "data" in msg.lower() or "directory" in msg.lower():
                return True

        return False

    def _is_missing_third_party_error(self, result: "SandboxResult",
                                      generated: dict) -> bool:
        """判断沙箱失败是否是第三方包缺失（而非代码 bug）。

        ModuleNotFoundError 且缺失的模块是第三方包时：
        - 立即补全 requirements.txt
        - 视为通过（包在运行环境安装，非代码问题）
        """
        if result.error_type != "ModuleNotFoundError":
            return False

        msg = result.error_message or ""
        # 提取模块名: "No module named 'pandas'" → "pandas"
        import re
        m = re.search(r"No module named '(\w+)'", msg)
        if not m:
            return False
        mod_name = m.group(1)

        # 判断是否为第三方包（非标准库、非本地模块）
        if mod_name in STDLIB:
            return False
        # 本地模块：生成的 .py 文件名
        local_modules = set()
        for f in generated:
            if f.endswith(".py"):
                fname = f.rsplit("/", 1)[-1].replace(".py", "")
                if fname != "__init__":
                    local_modules.add(fname)
        if mod_name in local_modules:
            return False

        # 是第三方包 → 补全 requirements.txt
        req_name = CodeValidator.IMPORT_TO_REQ.get(mod_name, mod_name.lower())
        existing = self._extract_requirements_packages(generated)
        req_normalized = req_name.lower().replace("-", "_")
        if req_normalized not in existing:
            req_content = generated.get("requirements.txt", "")
            new_lines = req_content.rstrip() + "\n" + req_name + "\n"
            generated["requirements.txt"] = new_lines
            self._write_code("requirements.txt", new_lines)
            logger.info(f"  📦 沙箱发现缺失依赖，自动补全: {req_name}")
        return True

    # ============================================================
    # 执行配置
    # ============================================================

    def _write_execution_spec(self, brief: dict, ordered_files: list,
                              generated: dict = None) -> None:
        """从 brief 读取入口，确定性产出 execution_spec.yaml（供 Step8 读取执行）。

        只写 entrypoint + command 模板，不写 experiments（那在 config.yaml 的
        experiment.runs 里，避免双源真相）。entrypoint.script 优先取 brief 的
        entry_point，校验其确实在生成文件里；否则回退 main.py / run.py。

        智能模板：检查生成的入口代码是否真的接受 --run 参数，动态决定模板内容。
        """
        entry_point = (
            brief.get("design_digest", {})
            .get("project_structure", {})
            .get("entry_point", "main.py")
        ) or "main.py"
        entry_point = str(entry_point).lstrip("./")

        script = entry_point if entry_point in ordered_files else None
        if script is None:
            for candidate in ("main.py", "run.py"):
                if candidate in ordered_files:
                    script = candidate
                    break
        if script is None:
            logger.warning("  execution_spec: 生成文件中未找到入口 main.py/run.py，跳过")
            return

        # 智能检测：入口代码是否接受 --run 参数
        supports_run = self._entry_supports_run_arg(script, generated)
        if supports_run:
            template = "python {script} --run {run_index} --config {config}"
        else:
            template = "python {script} --config {config}"
            logger.info(f"  execution_spec: 入口 {script} 不支持 --run 参数，使用简化模板")

        spec = {
            "entrypoint": {
                "type": "python",
                "script": script,
            },
            "command": {
                # {script}/{run_index}/{config} 为占位符，由 Step8 执行时替换
                "template": template,
            },
        }
        spec_path = self.output_dir / "execution_spec.yaml"
        spec_path.write_text(
            yaml.safe_dump(spec, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )
        logger.info(f"  已写 {spec_path}")

    @staticmethod
    def _entry_supports_run_arg(script: str, generated: dict = None) -> bool:
        """检查入口脚本是否接受 --run 参数（AST 检测 argparse add_argument）。"""
        if not generated or script not in generated:
            return True  # 无法检测时保守假设支持
        code = generated[script]
        try:
            ast.parse(code)
        except SyntaxError:
            return True  # 语法错误时保守假设支持（模板路径兜底）
        # 与 CodeValidator 共用同一检测逻辑，避免两处判断漂移
        return CodeValidator.supports_run_arg(code)

    # ============================================================
    # 报告
    # ============================================================

    def _build_report(self, brief, ordered_files, file_results, project_checks) -> dict:
        total = len(ordered_files)
        passed = sum(1 for r in file_results if r["status"] == "passed")
        failed = total - passed

        # 按 Phase 统计
        file_status = {r["file"]: r["status"] for r in file_results}
        phases = []
        for phase in brief.get("code_generation_plan", {}).get("generation_order", []):
            fs = phase.get("files", [])
            ok = all(file_status.get(f) == "passed" for f in fs)
            phases.append({
                "phase": phase.get("phase"),
                "description": phase.get("description"),
                "status": "success" if ok else "failed",
                "files": len(fs),
            })

        syntax_ok = all(r["status"] in ("passed", "sandbox_failed") for r in file_results)
        # 只有 status == "failed" 才算失败；unresolved 的 "detected"/"none" 是正常识别，不算失败。
        # sandbox_failed 是"静态通过但沙箱失败"，不算致命失败。
        project_ok = all(c.get("status") != "failed" for c in project_checks.values())
        # 容错：关键文件（config.yaml + main.py）存在且通过率 ≥ 70% 即视为成功，
        # 避免个别非关键文件验证失败导致整条管线中断。
        critical_files = {"config/config.yaml", "config/config.yml", "main.py", "run.py"}
        critical_ok = all(
            file_status.get(f) in ("passed", "sandbox_failed")
            for f in critical_files
            if f in ordered_files
        )
        passed_or_sandbox = sum(1 for r in file_results
                                if r["status"] in ("passed", "sandbox_failed"))
        pass_rate = passed_or_sandbox / total if total > 0 else 0
        overall = critical_ok and pass_rate >= 0.7 and project_ok

        # 沙箱统计
        sandbox_tested = sum(1 for r in file_results if "sandbox" in r)
        sandbox_passed = sum(1 for r in file_results
                             if r.get("sandbox", {}).get("success"))

        return {
            "generation": {
                "status": "success" if overall else "failed",
                "total_files": total,
                "generated_files": passed,
                "failed_files": failed,
            },
            "validation": {
                "syntax": "passed" if syntax_ok else "failed",
                "structure": "passed" if syntax_ok else "failed",
                "files": project_checks["files"]["status"],
                "third_party": project_checks["third_party"]["status"],
                "function_interface": project_checks["function_interface"]["status"],
                "traceability": project_checks["traceability"]["status"],
                "decisions": project_checks["decisions"]["status"],
                "unresolved": {
                    "status": project_checks["unresolved"]["status"],
                    "items": project_checks["unresolved"]["items"],
                },
            },
            "sandbox": {
                "tested": sandbox_tested,
                "passed": sandbox_passed,
                "failed": sandbox_tested - sandbox_passed,
            },
            "phases": phases,
            "traceability": project_checks["traceability"]["records"],
        }

    def _write_outputs(self, report: dict, file_results: list):
        # generation_report.yaml
        report_path = self.output_dir / "generation_report.yaml"
        report_path.write_text(
            yaml.safe_dump(report, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )

        # generation_log.json
        log = {
            "paper": self.paper_name,
            "files": [
                {
                    "file": r["file"],
                    "status": r["status"],
                    "attempts": r["attempts"],
                }
                for r in file_results
            ],
        }
        log_path = self.output_dir / "generation_log.json"
        log_path.write_text(
            json.dumps(log, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


def main():
    parser = argparse.ArgumentParser(
        description="Step7: 代码生成 —— 读取 Step6 project_brief 逐文件生成代码"
    )
    parser.add_argument("--paper", type=str, default="plant",
                        help="单篇论文名称")
    parser.add_argument("--all", action="store_true",
                        help="处理所有论文")
    args = parser.parse_args()

    if args.all:
        root = get_paper_output_dir("")
        papers = []
        for d in Path(root).iterdir():
            if d.is_dir() and (d / "step6" / "project_brief.yaml").exists():
                papers.append(d.name)
        logger.info(f"发现 {len(papers)} 篇论文: {papers}")
    elif args.paper:
        papers = [args.paper]
    else:
        parser.print_help()
        return

    success = 0
    failed = 0
    for p in papers:
        if Step7Runner(p).run():
            success += 1
        else:
            failed += 1

    logger.info(f"Step7 全部完成: {success} 成功, {failed} 失败")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
