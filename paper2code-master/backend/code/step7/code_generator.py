"""
Step7.3: 代码生成器

调用 LLM 文本模型，一次生成一个代码文件。
"""

import yaml
from llm_manager import llm_call, extract_code_block
from logger import logger

# 核心算法文件走 heavy 档，辅助文件（utils/evaluation/文档等）走 light 档。
# light 档可在 settings.json 的 task_models.light 配更快/更便宜的模型。
CORE_HEAVY_PREFIXES = ("models/", "training/", "data/")
CORE_HEAVY_FILES = ("main.py", "run.py", "config/config.yaml", "config/config.yml")


def task_type_for_file(file: str) -> str:
    """按文件复杂度路由 LLM 档位：核心算法 heavy，辅助文件 light。"""
    if file.endswith("__init__.py"):
        return "light"
    if file.startswith(CORE_HEAVY_PREFIXES) or file in CORE_HEAVY_FILES:
        return "heavy"
    return "light"


class CodeGenerator:
    """调用 LLM 生成单个代码文件。"""

    def generate(self, context: dict) -> str:
        prompt = self._build_prompt(context)
        try:
            raw = llm_call(prompt, task_type=task_type_for_file(context.get("file", "")))
        except (RuntimeError, TimeoutError) as e:
            logger.error(f"代码生成 LLM 调用失败: {e}")
            return f"# LLM 调用失败: {e}"
        return extract_code_block(raw)

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

        # 依赖文件的接口摘要（签名级，不含实现）
        related = ctx.get("related_code", {})
        if related:
            parts.append("")
            parts.append("【依赖文件接口摘要（引用时保持一致）】")
            parts.append("（仅含类/方法/函数签名与文档首行，不含实现；引用的名称与签名必须与之完全一致）")
            for dep_file, dep_digest in related.items():
                parts.append(f"===== {dep_file} =====")
                parts.append(dep_digest)
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
                "3. 引用的类/方法名必须与「依赖文件接口摘要」中的定义完全一致，不要自行改名。",
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
                    "10. 读取 config 时只能用「config.yaml 可用路径」列表中真实存在的键，"
                    "严禁臆造不存在的顶层键（如 data/system/experiment 等必须与 config.yaml 完全对齐）。",
                    "11. config 路径必须逐级验证存在性，不要猜测嵌套路径。"
                    "如果不确定某个 key 是否存在，用 config.get('key', default) 安全访问。",
                ]
            # 入口文件：必须支持 --run N 逐组合执行（Step8 实验矩阵依赖此参数）
            base = file.replace("\\", "/").rsplit("/", 1)[-1]
            if base in ("main.py", "run.py"):
                reqs += [
                    f"{len(reqs) + 1}. 这是项目入口文件，必须支持命令行参数 "
                    "--run N（argparse add_argument(\"--run\", type=int, ...)）与 "
                    "--config：N 为 config['experiment']['runs'] 列表的下标（从 0 开始），"
                    "指定 --run N 时只执行第 N 个实验组合（model/training_mode/"
                    "dataset_version/train_ratio 等取自该条 run），不指定时可执行默认/全部组合。"
                    "缺少 --run 参数将导致实验矩阵无法逐个执行，属严重缺陷。",
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

