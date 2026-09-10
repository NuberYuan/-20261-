"""
Step7.5: 代码修复器

只有 code_validator 判定 FAILED 时才调用。
输入: 原代码 + 错误信息 + Step6 要求 + 相关依赖代码
输出: 修复后的代码（仍需再次通过 code_validator 检查）

修复次数上限 MAX_REPAIR_ATTEMPTS = 3 由 step7_main 控制，
本模块只负责"给定错误，生成一次修复结果"。
"""

import re
from llm_utils import chat_with_text
from logger import logger

# 单个文件最多修复次数（在 step7_main 中消费）
MAX_REPAIR_ATTEMPTS = 3


class CodeRepairer:
    """调用 DeepSeek 修复一个未通过检查的代码文件。"""

    def repair(self, file: str, code: str, errors: list,
               spec: dict, context: dict,
               runtime_errors: list = None) -> str:
        """
        给定原代码与错误，返回修复后的代码。

        Args:
            file: 文件相对路径
            code: 当前（有问题的）代码
            errors: code_validator 报告的错误列表
            spec: Step6 per_file_specs 中该文件的规格
            context: CodeContextBuilder.build() 返回的完整上下文
            runtime_errors: 沙箱测试发现的运行时错误列表（可选）
        """
        prompt = self._build_prompt(file, code, errors, spec, context,
                                    runtime_errors=runtime_errors)
        raw = chat_with_text(prompt)
        return self._extract_code(raw)

    # ============================================================
    # Prompt 构建
    # ============================================================

    def _build_prompt(self, file: str, code: str, errors: list,
                      spec: dict, context: dict,
                      runtime_errors: list = None) -> str:
        parts = [
            "你是一名资深 PyTorch 工程师。下面这个代码文件没有通过自动检查，请修复它。",
            "",
            f"【文件】{file}",
            f"【用途】{spec.get('purpose', '')}",
            "",
            "【检查发现的错误】",
        ]
        for e in errors:
            parts.append(f"- {e}")

        # 运行时错误（来自沙箱测试）
        runtime_text = ""
        if runtime_errors:
            parts.append("")
            parts.append("【运行时错误（沙箱测试捕获）】")
            for re_err in runtime_errors:
                parts.append(f"- {re_err}")

            # 针对常见运行时错误提供修复指导
            runtime_text = " ".join(runtime_errors)
            if "stack expects each tensor to be equal size" in runtime_text:
                parts.append("")
                parts.append("【修复指导：tensor 尺寸不一致】")
                parts.append("DataLoader 默认 collate_fn 无法处理不同尺寸的 tensor。修复方法：")
                parts.append("1. 在 Dataset.__getitem__ 中确保返回的 tensor 形状一致（padding 到固定长度）")
                parts.append("2. 或在 DataLoader 中使用自定义 collate_fn 进行 padding：")
                parts.append("   def collate_fn(batch):")
                parts.append("       # 按最大长度 padding")
                parts.append("       max_len = max(t.shape[0] for t in batch)")
                parts.append("       padded = torch.zeros(len(batch), max_len)")
                parts.append("       for i, t in enumerate(batch):")
                parts.append("           padded[i, :t.shape[0]] = t")
                parts.append("       return padded")

        # weight_decay / momentum 等数值参数为 None
        if "not supported between instances of 'NoneType' and 'float'" in runtime_text:
            parts.append("\n【修复指导：NoneType 比较错误】")
            parts.append("某个数值参数传了 None 给要求 float/int 的位置。")
            parts.append("常见位置：optimizer 的 weight_decay, momentum; scheduler 的 eta_min 等。")
            parts.append("修复：确保所有数值参数在传入前有合理的 float 默认值（如 weight_decay=0.0）。")

        # embedding index 越界
        if "index out of range" in runtime_text and "embedding" in runtime_text.lower():
            parts.append("\n【修复指导：Embedding 越界】")
            parts.append("Embedding 的 num_embeddings 小于输入中的最大 index。")
            parts.append("修复方式（任选一种）：")
            parts.append("1. 增大 num_embeddings 使其大于所有可能的输入值")
            parts.append("2. 在输入前做 clamp: input_ids = torch.clamp(input_ids, 0, num_embeddings-1)")
            parts.append("3. 设置 padding_idx 并确保 padding 值不超过 num_embeddings")

        # 语法错误：未闭合字符串 / 截断
        if "unterminated string literal" in runtime_text or "unexpected EOF" in runtime_text:
            parts.append("\n【修复指导：代码截断/未闭合字符串】")
            parts.append("代码被截断导致字符串未闭合或函数体不完整。")
            parts.append("修复：确保输出完整代码 —— 每个字符串都有闭合引号，每个函数/class 都有完整代码体。"
                         "如果代码较长，可以适当精简注释和 docstring，但不能截断代码逻辑。")

        # 如果有配置引用错误，提供 config 结构供 LLM 参考
        has_config_err = any("配置引用" in e for e in errors)
        if has_config_err and context.get("config"):
            cfg = context["config"]
            from step7.code_validator import CodeValidator
            paths = CodeValidator._collect_config_paths(cfg, max_depth=3)
            if paths:
                parts.append("\n【config.yaml 完整路径结构】")
                parts.append("以下是 config.yaml 中所有可用的嵌套路径（点号分隔），请使用正确的路径：")
                for p in sorted(paths):
                    parts.append(f"  - config['{p}']")

        if spec.get("classes"):
            parts.append(f"\n【必须定义的类】{', '.join(spec['classes'])}")
        if spec.get("key_methods"):
            parts.append(f"\n【必须实现的方法】{', '.join(spec['key_methods'])}")

        # 相关依赖代码（用于对齐接口/类名）
        related = context.get("related_code", {})
        if related:
            parts.append("\n【已生成的相关代码（引用必须与之一致）】")
            for dep_file, dep_code in related.items():
                parts.append(f"===== {dep_file} =====")
                parts.append(dep_code)
                parts.append("")

        if file.endswith((".yaml", ".yml")):
            lang = "yaml"
        elif file.endswith(".md"):
            lang = "markdown"
        elif file.endswith(".txt"):
            lang = "text"
        else:
            lang = "python"
        parts.extend([
            "【当前代码（需要修复）】",
            f"```{lang}",
            code,
            "```",
            "",
            "【修复要求】",
        ])
        parts.extend(self._fix_requirements(file))

        return "\n".join(parts)

    def _fix_requirements(self, file: str) -> list:
        reqs = ["1. 只修复检查报告指出的错误（包括静态错误和运行时错误），不要改动其他无关内容。",
                "2. 只输出修复后的完整内容，不要任何解释文字、不要 markdown 代码块包裹、不要前后缀。"]

        if file.endswith(".yaml") or file.endswith(".yml"):
            reqs += [
                "3. 这是 YAML 配置文件：输出纯 YAML 键值对，能被 yaml.safe_load 解析。",
                "4. 绝对不要输出 Python 代码 —— 不要 import、不要函数定义、不要变量赋值。",
            ]
        elif file.endswith(".py"):
            reqs += [
                "3. 引用的类/方法名必须与「已生成的相关代码」完全一致。",
                "4. 修复后必须能通过 ast.parse 语法检查，且包含所有【必须定义的类】和【必须实现的方法】。",
            ]

        return reqs

    # ============================================================
    # 提取代码
    # ============================================================

    @staticmethod
    def _extract_code(raw: str) -> str:
        raw = raw.strip()
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
        return raw
