"""
information_extractor.py

使用 LLM 从论文文本中提取关键信息。
包含结构校验和自动修复。

Author: Paper2Code
"""

from typing import List, Dict, Optional, Any, Tuple, Union

import json
import re

from llm_utils import chat_with_text
from logger import logger


class InformationExtractor:

    # model_architecture 必须包含的子字段
    REQUIRED_ARCH_FIELDS = [
        "input", "output", "feature_extraction", "classification",
        "pretrained_model", "special_modules",
        "alexnet", "googlenet", "resnet", "vgg", "densenet",
        "mobilenet", "efficientnet", "transformer", "lstm",
    ]

    # 不应该含前缀的数值字段（值只能是纯数字或简单名称）
    CLEAN_VALUE_FIELDS = [
        "learning_rate",
        "epoch",
    ]

    def __init__(self, prompt: str):
        self.prompt = prompt

    def extract(self, paper: dict) -> dict:
        """从 paper JSON 中提取关键信息"""
        # 只分析文本部分
        paragraphs = paper.get("paragraphs", [])
        text_parts = []

        for p in paragraphs:
            text = p.get("text", "")
            if text.strip():
                text_parts.append(text)

        paper_text = "\n".join(text_parts)

        # DeepSeek-chat 支持 64K tokens ≈ 48K chars
        # 预留 ~4K 给 prompt 模板自身
        max_text_len = 44000
        did_truncate = len(paper_text) > max_text_len

        if did_truncate:
            # 智能截取：保留开头 + 末尾（超参数通常在末尾）
            head_len = int(max_text_len * 0.75)  # 前 75%
            tail_len = max_text_len - head_len     # 后 25%
            truncated_text = (
                paper_text[:head_len]
                + "\n\n... [中间内容已截断] ...\n\n"
                + paper_text[-tail_len:]
            )
            logger.warning(
                f"Paper text truncated: {len(paper_text)} → {max_text_len} "
                f"(head={head_len}, tail={tail_len})"
            )
            final_prompt = self.prompt.replace("{paper_text}", truncated_text)
        else:
            final_prompt = self.prompt.replace("{paper_text}", paper_text)

        logger.info(
            f"Prompt: {len(final_prompt)} chars "
            f"(paper: {len(paper_text)} chars, "
            f"truncated: {did_truncate})"
        )

        raw_result = chat_with_text(final_prompt)
        result = self._parse_json(raw_result)

        # 结构校验和修复
        result = self._validate_and_fix(result, paper_text)

        return result

    # =========================
    # JSON 解析
    # =========================

    @staticmethod
    def _parse_json(raw: str) -> dict:
        """
        从 LLM 返回的文本中提取 JSON。

        支持以下格式:
          - 纯 JSON: {"key": "value"}
          - Markdown 代码块: ```json ... ```
          - 带前缀/后缀文本: some text {"key": "value"} more text
        """
        raw = raw.strip()

        # 尝试 1: 直接解析
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass

        # 尝试 2: 去掉 markdown 代码块
        cleaned = InformationExtractor._strip_markdown_fences(raw)
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass

        # 尝试 3: 从文本中提取 JSON 对象 (最外层 { ... })
        match = re.search(r'\{.*\}', raw, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass

        # 全部失败 → 返回原始响应
        logger.warning("Failed to parse LLM response as JSON")
        return {
            "raw_response": raw[:2000],
            "error": "JSON parse failed after all attempts",
        }

    @staticmethod
    def _strip_markdown_fences(text: str) -> str:
        """移除 ```json ... ``` 或 ``` ... ``` 包裹"""
        text = re.sub(r'^```(?:json)?\s*\n?', '', text)
        text = re.sub(r'\n?```\s*$', '', text)
        return text.strip()

    # =========================
    # 结构校验与修复
    # =========================

    # architecture 可能散落在 methodology 层的字段名
    ARCH_FIELD_NAMES = {
        "input", "output", "network_layers", "feature_extraction",
        "classification", "pretrained_model", "special_modules",
    }

    def _validate_and_fix(self, result: dict, paper_text: str) -> dict:
        """
        校验 LLM 输出的结构，修复常见问题:

        1) model_architecture 被拍平为字符串 → 转为 dict
        2) model_architecture 子字段散落在 methodology 顶层 → 移回 dict
        3) learning_rate / epoch 含描述前缀 → 提取纯数字
        4) batch_size 被标为 NOT_FOUND → 从原文搜索兜底
        5) important_parameters 是数组 → 转为对象
        """
        if "error" in result:
            return result  # JSON 解析都没过，不修了

        meth = result.get("methodology", {})
        impl = result.get("implementation_details", {})
        repro = result.get("reproduction_requirements", {})

        # ---- 修复 model_architecture ----
        arch = meth.get("model_architecture", {})
        if isinstance(arch, str):
            logger.warning("model_architecture 是字符串，正在修复...")
            arch = {
                field: "NOT_FOUND" for field in self.REQUIRED_ARCH_FIELDS
            }
            arch["_raw_description"] = meth["model_architecture"]
            meth["model_architecture"] = arch

        if isinstance(arch, dict):
            # 收集 methodology 层游离的架构字段，移入 model_architecture
            for field in self.ARCH_FIELD_NAMES:
                if field in meth and field not in arch:
                    logger.warning(
                        f"将 methodology.{field} 移入 model_architecture"
                    )
                    arch[field] = meth.pop(field)

            # 补全缺失的子字段
            for field in self.REQUIRED_ARCH_FIELDS:
                if field not in arch:
                    arch[field] = "NOT_FOUND"

        # ---- 修复 learning_rate / epoch 前缀 ----
        if isinstance(impl, dict):
            for field in self.CLEAN_VALUE_FIELDS:
                val = impl.get(field, "")
                if isinstance(val, str) and val not in ("NOT_FOUND", ""):
                    cleaned = self._clean_numeric_value(val)
                    if cleaned != val:
                        logger.warning(
                            f"{field}: '{val}' → '{cleaned}'"
                        )
                        impl[field] = cleaned

        # ---- 兜底搜索：LLM 常漏掉的字段 ----
        if isinstance(impl, dict):
            fallbacks = {
                "batch_size": self._search_batch_size,
                "learning_rate": self._search_learning_rate,
                "optimizer": self._search_optimizer,
                "loss_function": self._search_loss_function,
                "programming_framework": self._search_framework,
                "momentum": self._search_momentum,
                "weight_decay": self._search_weight_decay,
            }
            for field, searcher in fallbacks.items():
                val = str(impl.get(field, "")).lower()
                if val in ("not_found", "not found", "not mentioned", ""):
                    found = searcher(paper_text)
                    if found:
                        logger.warning(f"{field}: 从原文搜索兜底 → '{found}'")
                        impl[field] = found

        # ---- 修复 important_parameters: 数组 → 对象 + 补全 ----
        if isinstance(repro, dict):
            imp = repro.get("important_parameters")
            if isinstance(imp, list):
                logger.warning("important_parameters 是数组，转为对象...")
                params_dict = {}
                for item in imp:
                    if isinstance(item, str) and ":" in item:
                        k, v = item.split(":", 1)
                        params_dict[k.strip()] = v.strip()
                    else:
                        params_dict[str(item)] = str(item)
                imp = params_dict
                repro["important_parameters"] = imp

            # 把兜底搜索到的值也写入 important_parameters（只填缺失的）
            if isinstance(imp, dict):
                params_to_propagate = [
                    "learning_rate", "batch_size", "momentum",
                    "weight_decay", "dropout_ratio",
                ]
                for key in params_to_propagate:
                    val = impl.get(key, "")
                    if val and val not in ("NOT_FOUND", "not found", "not mentioned", ""):
                        if key not in imp or imp[key] in ("NOT_FOUND", ""):
                            imp[key] = val

                # ---- 兜底搜索 scheduler ----
                sched = self._search_scheduler(paper_text)
                if sched:
                    for k, v in sched.items():
                        if k not in imp or imp.get(k) in ("NOT_FOUND", ""):
                            imp[k] = v

                # ---- 清洗 scheduler step_size: "30/3 epochs" → "10" ----
                for sk in ("scheduler_step_size", "step_size"):
                    val = imp.get(sk, "")
                    if isinstance(val, str) and val not in ("NOT_FOUND", ""):
                        evaluated = self._eval_expression(val)
                        if evaluated is not None:
                            evaluated_str = str(int(evaluated)) if evaluated == int(evaluated) else str(evaluated)
                            if evaluated_str != val:
                                logger.warning(
                                    f"scheduler step_size: '{val}' → '{evaluated_str}'"
                                )
                                imp[sk] = evaluated_str
                            # 同时提取纯净 epoch 数（去掉解释文本）
                            elif "epoch" in val.lower() or "/" in val:
                                # 值本身已经正确但含解释文本 → 提取数字
                                num = re.search(r'(\d+)', val)
                                if num and num.group(1) != val:
                                    imp[sk] = num.group(1)
                                    logger.warning(
                                        f"scheduler step_size cleaned: '{val}' → '{imp[sk]}'"
                                    )

        # ---- 清理 figures: 非曲线图的 x_axis/y_axis/trend/conclusion 无意义 ----
        for fig_list_key in ("figures", "paper_figures"):
            fig_list = result.get(fig_list_key, [])
            for fig in fig_list:
                fig_type = fig.get("type", "")
                # 只有 experiment_result / comparison_chart 才有 x/y/trend
                if fig_type not in ("experiment_result", "comparison_chart"):
                    for field in ("x_axis", "y_axis", "trend", "conclusion"):
                        if fig.get(field) == "NOT_FOUND":
                            del fig[field]
                # 只有 data_example 才有 data_characteristics/categories/visual_features
                if fig_type != "data_example":
                    for field in ("data_characteristics", "categories", "visual_features"):
                        if fig.get(field) == "NOT_FOUND":
                            del fig[field]

        return result

    # =========================
    # 数值清洗
    # =========================

    @staticmethod
    def _clean_numeric_value(val: str) -> str:
        """从含前缀的字符串中提取纯数值。如 'Base learning rate: 0.005' → '0.005'"""
        match = re.search(r'(\d+\.?\d*)\s*$', val)
        if match:
            return match.group(1)
        return val

    @staticmethod
    def _eval_expression(raw: str) -> Optional[Union[float, int]]:
        """
        评估字符串中的数学表达式。
        如 "30/3 epochs" → 10,  "8 x 8" → None,  "30" → 30
        """
        # 只提取数字和运算符部分
        m = re.search(r'([\d]+(?:\s*/\s*\d+)?)', raw)
        if not m:
            return None
        expr = m.group(1).replace(" ", "")
        try:
            # 安全评估：只允许数字和/号
            if re.fullmatch(r'\d+(?:/\d+)?', expr):
                if "/" in expr:
                    a, b = expr.split("/")
                    return int(a) / int(b)
                return int(expr)
        except (ValueError, ZeroDivisionError):
            pass
        return None

    # =========================
    # 兜底搜索——从原文中正则匹配
    # =========================

    @staticmethod
    def _search_patterns(text: str, patterns: List[str]) -> Optional[str]:
        """对 text 依次尝试每个正则，返回第一个捕获组"""
        for p in patterns:
            match = re.search(p, text)
            if match:
                return match.group(1).strip()
        return None

    @classmethod
    def _search_batch_size(cls, text: str) -> Optional[str]:
        return cls._search_patterns(text, [
            r'[Bb]atch\s*[Ss]ize\s*[:\-=\s]+(\d+[^.\n]*)',
            r'batch\s*size\s*[:\-]\s*(\d+[^.\n]*)',
        ])

    @classmethod
    def _search_learning_rate(cls, text: str) -> Optional[str]:
        return cls._search_patterns(text, [
            r'[Bb]ase\s+learning\s+rate\s*[:\-=\s]+(\d+\.?\d*)',
            r'learning\s+rate\s*[:\-=\s]+(\d+\.?\d*)',
        ])

    @classmethod
    def _search_optimizer(cls, text: str) -> Optional[str]:
        return cls._search_patterns(text, [
            r'[Ss]olver\s+type\s*[:\-=\s]+(.+?)(?:[\n\.\)]|$)',
            r'(Stochastic\s+Gradient\s+Descent)',
            r'[Oo]ptimizer\s*[:\-=\s]+(\w[\w\s]{0,20}?)(?:[\n\.\)]|$)',
        ])

    @classmethod
    def _search_loss_function(cls, text: str) -> Optional[str]:
        result = cls._search_patterns(text, [
            r'[Ll]oss\s+function\s*[:\-=\s]+(.+?)(?:[\n\.\)]|$)',
            r'(softmax\s+cross.entropy)',
            r'(CrossEntropyLoss)',
            r'(softmax\s+loss)',
        ])
        if result:
            return result
        # 论文没明确说 loss function，但从 softmax 分类层推断
        if re.search(r'softmax\s+layer', text, re.I):
            return "softmax cross-entropy (inferred from softmax layer)"
        return None

    @classmethod
    def _search_framework(cls, text: str) -> Optional[str]:
        return cls._search_patterns(text, [
            r'(?:our\s+own\s+)?fork\s+of\s+(Caffe|PyTorch|TensorFlow)\d*',
            r'framework\s*(?:is|:|was)?\s*(Caffe|PyTorch|TensorFlow)\b',
            r'using\s+(Caffe|PyTorch|TensorFlow)\b',
            r'(?:implemented|developed|coded|conducted)\s+(?:in|using)\s+(?:our\s+own\s+)?(?:fork\s+of\s+)?(Caffe|PyTorch|TensorFlow)',
        ])

    @classmethod
    def _search_momentum(cls, text: str) -> Optional[str]:
        return cls._search_patterns(text, [
            r'[Mm]omentum\s*[:\-=\s]+(\d+\.?\d*)',
        ])

    @classmethod
    def _search_weight_decay(cls, text: str) -> Optional[str]:
        return cls._search_patterns(text, [
            r'[Ww]eight\s+decay\s*[:\-=\s]+(\d+\.?\d*)',
        ])

    @classmethod
    def _search_scheduler(cls, text: str) -> Optional[dict]:
        """从原文搜索学习率调度器信息。返回 None 或包含 scheduler/step_size/gamma 的 dict"""
        # 先检测调度器类型（覆盖 "step" / "StepLR" / "step_size" / "step-size" / "learning rate policy/schedule/decay"）
        if not re.search(
            r'[Ss]tep(?:LR|\b|[_ ]?size|-)|[Ll]earning\s+rate\s+(?:policy|schedule|decay)',
            text,
        ):
            return None

        result = {}

        # 调度器名称: "Step" policy → StepLR
        if re.search(r'[Ss]tep\b.*policy|policy.*[Ss]tep', text):
            result["scheduler"] = "StepLR"

        # ---- gamma ----
        # 优先显式给出衰减系数: "gamma = 0.1" / "decays by 0.1" / "multiplied by 0.1"
        gamma = None
        m = re.search(r'gamma\s*[=: ]+\s*(\d+\.?\d*)', text, re.IGNORECASE)
        if m:
            gamma = m.group(1)
        else:
            m = re.search(
                r'(?:decays?|multiplied|reduced)\s+by\s+(\d+\.?\d*)',
                text, re.IGNORECASE,
            )
            if m:
                gamma = m.group(1)

        # ---- step_size ----
        # "step_size = 30" / "step size 30" / "every 30 epochs"
        step = None
        m = re.search(r'step[_ ]?size\s*[=: ]+\s*(\d+)', text, re.IGNORECASE)
        if m:
            step = m.group(1)
        else:
            m = re.search(
                r'(?:every|per)\s*(\d+(?:\s*/\s*\d+)?)[^\n\.]*?epochs?',
                text, re.IGNORECASE,
            )
            if m:
                step = str(InformationExtractor._eval_expression(m.group(1)))

        # ---- 兜底: 旧式 "decreases by a factor of N every M"（gamma 从 factor 反推）----
        if gamma is None or step is None:
            m = re.search(
                r'(?:decreases?\s+by\s+(?:a\s+)?factor\s+of\s+)?(\d+)'
                r'\s*(?:every|per|/)\s*(\d+(?:\s*/\s*\d+)?[^.\n\)]*)',
                text,
            )
            if m:
                factor = m.group(1)
                epochs_raw = m.group(2).strip().rstrip(")")
                if step is None:
                    # 评估 "30/3" → 10，提取纯数字
                    step = str(InformationExtractor._eval_expression(epochs_raw))
                if gamma is None:
                    # gamma = 1/factor
                    try:
                        gamma = str(1.0 / int(factor))
                    except (ValueError, ZeroDivisionError):
                        gamma = f"1/{factor}"

        if step is not None:
            result["scheduler_step_size"] = step
        if gamma is not None:
            result["scheduler_gamma"] = gamma

        return result if result else None
