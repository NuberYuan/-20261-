"""
Step7.3: 代码生成器

调用 DeepSeek 文本模型，一次生成一个代码文件。
"""

import re
import yaml
from llm_utils import chat_with_text
from logger import logger


class CodeGenerator:
    """调用 LLM 生成单个代码文件。"""

    def generate(self, context: dict) -> str:
        prompt = self._build_prompt(context)
        raw = chat_with_text(prompt)
        return self._extract_code(raw)

    # ============================================================
    # Prompt 构建
    # ============================================================

    def _build_prompt(self, ctx: dict) -> str:
        spec = ctx.get("spec", {})
        project = ctx.get("project", {})
        tech = ctx.get("tech_stack", {})
        training = ctx.get("training_config", {})
        file = ctx.get("file", "")

        parts = [
            "你是一名资深 PyTorch 工程师。请为下面的项目生成一个完整的代码文件。",
            "",
            "【项目背景】",
            f"- 论文: {project.get('paper_title', '')}",
            f"- 任务: {project.get('task', '')}",
            f"- 框架: {tech.get('framework', '')}",
            f"- 硬件: {tech.get('hardware', '')}",
            "",
            "【当前要生成的文件】",
            f"文件: {file}",
            f"所属 Phase: {ctx.get('phase', '')}",
            f"用途: {spec.get('purpose', '')}",
        ]

        classes = spec.get("classes") or []
        methods = spec.get("key_methods") or []
        if classes:
            parts.append(f"需要定义的类: {', '.join(classes)}")
        if methods:
            parts.append(f"需要实现的方法: {', '.join(methods)}")

        deps = spec.get("depends_on") or []
        if deps:
            parts.append(f"依赖的其他文件: {', '.join(deps)}")

        # 需求
        reqs = ctx.get("requirements", [])
        if reqs:
            parts.append("")
            parts.append("【对应需求】")
            for r in reqs:
                if r.get("what"):
                    parts.append(f"- {r.get('id')}: {r.get('what')}")
                if r.get("acceptance"):
                    parts.append(f"  验收: {r.get('acceptance')}")

        # Step5 详细设计
        detail = ctx.get("step5_detail", {})
        if detail:
            parts.append("")
            parts.append("【详细设计（Step5）】")
            parts.append(self._format_detail(detail))

        # 全局训练配置
        if training:
            parts.append("")
            parts.append("【训练配置】")
            parts.append(f"- 优化器: {training.get('optimizer', '')}")
            parts.append(f"- 学习率: {training.get('learning_rate', '')}")
            parts.append(f"- batch_size: {training.get('batch_size', '')}")
            parts.append(f"- epochs: {training.get('epochs', '')}")
            parts.append(f"- 损失函数: {training.get('loss', '')}")
            parts.append(f"- 调度器: {training.get('scheduler', '')}")

        # 已生成的相关代码
        related = ctx.get("related_code", {})
        if related:
            parts.append("")
            parts.append("【已生成的相关代码（引用时保持一致）】")
            for dep_file, dep_code in related.items():
                parts.append(f"===== {dep_file} =====")
                parts.append(dep_code)
                parts.append("")

        # config 路径树（生成时就告诉 LLM 哪些路径真实存在）
        config_paths = ctx.get("config_paths", [])
        if config_paths:
            parts.append("")
            parts.append("【config.yaml 可用路径（只能用这些路径，不要臆造其他路径）】")
            for p in config_paths:
                parts.append(f"  - config['{p}']")

        # 注意事项
        notes = spec.get("notes")
        if notes:
            parts.append("")
            parts.append(f"【注意事项】{notes}")

        # 给代码 Agent 的全局注意事项（Step6 汇总，含 train/test 语义等）
        agent_notes = ctx.get("agent_notes", [])
        if agent_notes:
            parts.append("")
            parts.append("【全局约束（必须遵守）】")
            for note in agent_notes:
                parts.append(f"- {note}")

        # 输出要求
        parts.extend([
            "",
            "【输出要求】",
        ])
        parts.extend(self._output_requirements(file, has_config=ctx.get("has_config", True)))

        return "\n".join(parts)

    def _output_requirements(self, file: str, has_config: bool = True) -> list:
        """按文件类型给出针对性的输出格式要求。"""
        reqs = ["1. 只输出文件内容本身，不要任何解释文字、不要 markdown 代码块包裹、不要前后缀。"]

        if file.endswith(".yaml") or file.endswith(".yml"):
            reqs += [
                "2. 这是 YAML 配置文件：输出纯 YAML 键值对，能被 yaml.safe_load 直接解析。",
                "3. 绝对不要输出 Python 代码 —— 不要 import、不要函数定义、不要变量赋值语句。",
                "4. 严格遵循需求与详细设计中的参数取值，不要擅自改动。",
            ]
        elif file.endswith(".py"):
            reqs += [
                "2. 这是 Python 源文件：输出纯 Python 代码，能通过 ast.parse 语法检查。",
                "3. 引用的类/方法名必须与「已生成的相关代码」中的定义完全一致，不要自行改名。",
                "4. 严格遵循需求与详细设计，不要擅自添加需求之外的功能。",
                "5. 函数参数必须有合理默认值（config=None, threshold=0.5, batch_size=32 等），"
                "确保外部可以无参调用进行基本测试，不会因缺少参数而崩溃。",
                "6. 数值参数禁止传 None 给要求 float/int 的位置（如 weight_decay, momentum 必须是 float）。",
                "7. torch.nn.Embedding 的 num_embeddings 必须大于所有可能的输入 index 值；"
                "tensor 拼接/stack 前确保维度一致，必要时做 pad 或 clamp。",
                "8. 代码必须完整输出，严禁截断 —— 每个 def/class/if/for/try 都必须有完整的代码体，"
                "每个字符串字面量都必须有闭合引号。",
            ]
            if has_config:
                reqs += [
                    "9. 超参数不要硬编码，统一从 config/config.yaml 读取（用 yaml.safe_load）。",
                    "10. 读取 config 时只能用「已生成的相关代码」里 config.yaml 中真实存在的键，"
                    "严禁臆造不存在的顶层键（如 data/system/experiment 等必须与 config.yaml 完全对齐）。",
                    "11. config 路径必须逐级验证存在性，不要猜测嵌套路径。"
                    "如果不确定某个 key 是否存在，用 config.get('key', default) 安全访问。",
                ]
        elif file.endswith(".txt"):
            reqs += [
                "2. 这是纯文本文件：每行一项，例如依赖列表每行一个包名。",
                "3. 不要输出任何 Python 代码。",
            ]
        elif file.endswith(".md"):
            reqs += [
                "2. 这是 Markdown 文档：用 Markdown 语法组织内容（标题、列表、代码块）。",
            ]

        return reqs

    def _format_detail(self, detail: dict) -> str:
        """把 Step5 详细设计格式化为 prompt 文本。"""
        lines = []
        # 标量字段直接一行展示
        for k in ("architecture_summary", "name", "type", "input", "output",
                  "num_classes", "pretrained", "preprocessing", "split_strategy",
                  "dataset_versions", "train_test_splits", "paper_type", "task"):
            v = detail.get(k)
            if v:
                lines.append(f"- {k}: {v}")

        # 模型架构特殊组件
        arch = detail.get("architecture")
        if arch and isinstance(arch, dict):
            if arch.get("special_components"):
                lines.append(f"- 特殊组件: {arch.get('special_components')}")
            if arch.get("implementation_notes"):
                lines.append(f"- 实现注意: {arch.get('implementation_notes')}")

        # 结构化字段用 YAML 展开，便于模型读取嵌套参数
        for k in ("training", "training_modes", "checkpoint", "inference",
                  "version_selector"):
            v = detail.get(k)
            if v:
                lines.append(f"- {k} (YAML):")
                lines.append(yaml.safe_dump(v, allow_unicode=True, sort_keys=False).strip())

        return "\n".join(lines) if lines else "（无）"

    # ============================================================
    # 提取代码
    # ============================================================

    @staticmethod
    def _extract_code(raw: str) -> str:
        """从 LLM 输出中提取代码（处理 markdown 代码块包裹）。"""
        raw = raw.strip()

        # 提取 ```...``` 代码块。
        # 语言标签任意（python/py/yaml/yml/json/markdown/text/...），
        # 不能只认 python/py，否则 YAML/MD/TXT 文件会带着 fence 返回导致校验失败。
        blocks = re.findall(r"```[a-zA-Z0-9_+.-]*\s*\n(.*?)```", raw, re.DOTALL)
        if blocks:
            if len(blocks) == 1:
                return blocks[0].strip()
            # 多个代码块时，优先选看起来像完整文件的（有 import/class/def 语句）
            def _score(block: str) -> tuple:
                has_import = bool(re.search(r"^import |^from ", block, re.MULTILINE))
                has_class = bool(re.search(r"^class ", block, re.MULTILINE))
                has_def = bool(re.search(r"^def ", block, re.MULTILINE))
                return (int(has_import) + int(has_class) + int(has_def), len(block))
            return max(blocks, key=_score).strip()

        # 没有代码块包裹，直接返回
        return raw
