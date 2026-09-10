"""
Step5 基类 — 三层防编造: 规则提取 → LLM补全 → 后验证

独立于 step1-3，只读取 data/output/ 下的 JSON。
"""

import json
import yaml
from datetime import datetime
from pathlib import Path

# 项目根目录 (code/ 的父目录)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
OUTPUT_ROOT = PROJECT_ROOT / "data" / "output"

from logger import logger
from llm_utils import chat_with_text


class BaseAnalyzer:
    """Step5 所有子步骤的基类"""

    # ------------------------------------------------------------------
    # 关键字段白名单 — 用于 Layer3 验证时检查重要字段是否被遗漏
    # ------------------------------------------------------------------
    CRITICAL_FIELDS = [
        "title",
        "research_problem",
        "motivation",
        "main_contribution",
        "methodology.method_name",
        "methodology.core_idea",
        "methodology.model_architecture.input",
        "methodology.model_architecture.output",
        "methodology.model_architecture.pretrained_model",
        "methodology.model_architecture.special_modules",
        "dataset.dataset_name",
        "dataset.dataset_size",
        "dataset.number_of_classes",
        "dataset.train_test_split",
        "dataset.preprocessing",
        "implementation_details.optimizer",
        "implementation_details.learning_rate",
        "implementation_details.batch_size",
        "implementation_details.epoch",
        "implementation_details.loss_function",
        "implementation_details.hardware",
        "implementation_details.training_strategy",
        "experiment.evaluation_metrics",
        "experiment.important_results",
        "reproduction_requirements.important_parameters",
        "reproduction_requirements.possible_difficulties",
    ]

    def __init__(self, paper_name: str):
        self.paper_name = paper_name
        self.paper_dir = OUTPUT_ROOT / paper_name
        self.output_dir = self.paper_dir / "step5"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 加载 key_information.json
        ki_path = self.paper_dir / "step3" / "key_information.json"
        if ki_path.exists():
            with open(ki_path, "r", encoding="utf-8") as f:
                self.key_info = json.load(f)
        else:
            logger.warning(f"key_information.json not found for {paper_name}")
            self.key_info = {}

        # 回溯 Step2.3 原始文本
        struct_path = self.paper_dir / "step2" / "structure_detect_paper.json"
        if struct_path.exists():
            with open(struct_path, "r", encoding="utf-8") as f:
                self.structure = json.load(f)
        else:
            self.structure = {}

    # ============================================================
    # 数据获取
    # ============================================================

    def get_paper_text(self) -> str:
        """提取论文纯文本（用于 LLM prompt 上下文）"""
        paragraphs = self.structure.get("paragraphs", [])
        parts = []
        for p in paragraphs[:200]:
            text = (p.get("text") or "").strip()
            if text and len(text) > 20:
                parts.append(text)
        return "\n\n".join(parts)

    def get_ki_summary(self) -> str:
        """key_information.json 紧凑摘要"""
        return json.dumps(self.key_info, ensure_ascii=False, indent=2)

    # ============================================================
    # 主流程
    # ============================================================

    def analyze(self, skip_llm: bool = False) -> dict:
        """三层防护主流程"""
        logger.info(f"[{self.__class__.__name__}] {self.paper_name}")

        # Layer 1: 规则提取
        result = self.extract_by_rules()

        # Layer 2: LLM 补全未知字段
        if skip_llm:
            logger.info("  跳过 LLM 补全 (--skip-llm)")
        else:
            unknown = self._find_unknown(result)
            if unknown:
                logger.info(f"  LLM补全 {len(unknown)} 个字段: {unknown}")
                llm_out = self._call_llm(unknown, result)
                result = self._merge(result, llm_out)

        # Layer 3: 后验证
        self._validate(result)

        result["_meta"] = {
            "paper_name": self.paper_name,
            "generated_at": datetime.now().isoformat(),
        }
        return result

    def extract_by_rules(self) -> dict:
        """Layer 1: 纯规则提取（子类实现）"""
        raise NotImplementedError

    # ============================================================
    # Layer 2: LLM
    # ============================================================

    def _call_llm(self, unknown_fields: list, current: dict) -> dict:
        prompt = f"""你是一个论文信息提取器。严格规则:
1. 只能从下方论文信息中提取
2. 没提到的填 "NOT_FOUND"
3. 绝对不编造

=== 论文信息 ===
{self.get_ki_summary()[:6000]}

=== 需补全字段（扁平路径列表，含 [i] 下标）===
{json.dumps(unknown_fields, ensure_ascii=False)}

=== 当前结果（供上下文参考）===
{json.dumps(current, ensure_ascii=False, indent=2)[:3000]}

输出要求：只输出一个 JSON 对象。对象的每个 key 必须是上面【需补全字段】里的路径字符串
（原样保留，含 [i] 下标），value 是该路径的补全值。不要嵌套，不要重排或改写路径，
不要输出路径之外的其他键。"""
        try:
            resp = chat_with_text(prompt).strip()
            if resp.startswith("```"):
                lines = resp.split("\n")
                lines = [l for l in lines if not l.startswith("```")]
                resp = "\n".join(lines)
            return json.loads(resp)
        except Exception:
            return {}

    # ============================================================
    # 辅助方法
    # ============================================================

    def _find_unknown(self, data, prefix="") -> list:
        """递归找 NOT_FOUND / 空值字段"""
        unknown = []
        if isinstance(data, dict):
            for k, v in data.items():
                path = f"{prefix}.{k}" if prefix else k
                if v in ("NOT_FOUND", "", None):
                    unknown.append(path)
                elif isinstance(v, (dict, list)):
                    unknown.extend(self._find_unknown(v, path))
        elif isinstance(data, list):
            for i, v in enumerate(data):
                if isinstance(v, (dict, list)):
                    unknown.extend(self._find_unknown(v, f"{prefix}[{i}]"))
        return unknown

    def _merge(self, base: dict, update: dict) -> dict:
        """合并 LLM 补全结果回 base（规则提取优先，LLM 只补 NOT_FOUND/空值）。

        支持两种 update 形态：
          1) 扁平路径 → 值（如 "requirements[0].output"），与 _find_unknown 产出的
             路径对齐，用 _set_nested 写回；
          2) 嵌套 dict/list（与 base 同构），递归合并（含 list 逐元素）。
        两种都只覆盖 NOT_FOUND/空值叶子，绝不覆盖规则已提取的真实值。
        """
        if not update or not isinstance(update, dict):
            return base
        flat = any(
            isinstance(k, str) and ("." in k or "[" in k)
            for k in update
        )
        if flat:
            for path, v in update.items():
                if isinstance(path, str):
                    self._set_nested(base, path, v)
            return base
        self._merge_nested(base, update)
        return base

    def _merge_nested(self, base, update):
        """递归合并嵌套 dict/list；list 按 index 逐元素合并，不覆盖真实值。"""
        for k, v in update.items():
            if k not in base:
                base[k] = v
            elif base[k] in ("NOT_FOUND", "", None):
                base[k] = v
            elif isinstance(base[k], dict) and isinstance(v, dict):
                self._merge_nested(base[k], v)
            elif isinstance(base[k], list) and isinstance(v, list):
                for i, item in enumerate(v):
                    if i >= len(base[k]):
                        base[k].append(item)
                    elif base[k][i] in ("NOT_FOUND", "", None):
                        base[k][i] = item
                    elif isinstance(base[k][i], (dict, list)) and isinstance(item, (dict, list)):
                        self._merge_nested(base[k][i], item)

    def _set_nested(self, obj, path: str, value):
        """按扁平路径写回（"a.b" / "a[0].b" / "a.b[1]"）。

        只覆盖 NOT_FOUND/空值/缺失的叶子；越界下标或类型不匹配则静默跳过，
        避免把 LLM 误报的路径写进错误位置。
        """
        import re
        tokens = re.findall(r'[^.\[\]]+|\[\d+\]', path)
        if not tokens:
            return
        cur = obj
        for i, tok in enumerate(tokens):
            last = (i == len(tokens) - 1)
            if tok.startswith("[") and tok.endswith("]"):
                idx = int(tok[1:-1])
                if not isinstance(cur, list) or idx < 0 or idx >= len(cur):
                    return
                if last:
                    if cur[idx] in ("NOT_FOUND", "", None):
                        cur[idx] = value
                    return
                if not isinstance(cur[idx], (dict, list)):
                    return
                cur = cur[idx]
            else:
                if last:
                    if not isinstance(cur, dict):
                        return
                    if tok not in cur or cur[tok] in ("NOT_FOUND", "", None):
                        cur[tok] = value
                    return
                if not isinstance(cur, dict) or tok not in cur:
                    return
                cur = cur[tok]

    def _validate(self, result: dict):
        """
        Layer 3: 验证

        1) 关键字段存在性审计：列出 key_info 中有关键信息字段（仅审计，不做硬性缺失判定）
        2) 词汇验证：检查 LLM 输出的概念词在原文中是否有依据（防编造）
        """
        src = self.get_ki_summary().lower()

        # 关键字段存在性审计：列出 key_info 中有关键信息字段，供排查遗漏参考。
        # 注意：key_info 的字段路径与各子类 result 结构并非一一对应（如
        # dataset.number_of_classes → result.num_classes），故只做审计日志，
        # 字段级缺失映射由各子类负责，避免产生误报。
        present = []
        for field_path in self.CRITICAL_FIELDS:
            value = self._get_nested(self.key_info, field_path)
            if value and value not in ("NOT_FOUND", "", None, "not mentioned"):
                present.append(field_path)
        if present:
            logger.debug(
                f"  [{self.paper_name}] key_info 中有关键信息的字段: {', '.join(present)}"
            )

        # 词汇验证
        def check(obj, depth=0):
            if depth > 10:
                return
            if isinstance(obj, dict):
                for v in obj.values():
                    check(v, depth + 1)
            elif isinstance(obj, list):
                for v in obj:
                    check(v, depth + 1)
            elif isinstance(obj, str) and obj not in ("NOT_FOUND", "", None):
                # 对于超过 3 个单词的字符串，检查关键概念是否在原文中出现
                words = [w for w in obj.lower().split() if len(w) > 3]
                if len(words) >= 3:
                    found = sum(1 for w in words[:5] if w in src)
                    if found == 0:
                        logger.warning(
                            f"  [{self.paper_name}] 可能未验证: {obj[:80]}"
                        )

        check(result)

    # ============================================================
    # YAML 输出
    # ============================================================

    def save_yaml(self, data: dict, filename: str) -> str:
        data = {k: v for k, v in data.items() if not k.startswith("_")}
        path = self.output_dir / filename
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(data, f, allow_unicode=True, default_flow_style=False,
                      sort_keys=False, indent=2)
        logger.info(f"  -> {path}")
        return str(path)

    def load_yaml(self, filename: str) -> dict:
        path = self.output_dir / filename
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        return {}

    # ============================================================
    # 通用工具
    # ============================================================

    @staticmethod
    def _get_nested(data: dict, path: str, default="NOT_FOUND"):
        """按点号路径从嵌套 dict 中取值"""
        keys = path.split(".")
        for k in keys:
            if isinstance(data, dict):
                data = data.get(k, default)
            else:
                return default
        return data

    @staticmethod
    def _safe_str(value, default="NOT_FOUND") -> str:
        """安全转为字符串"""
        if value is None or value == "":
            return default
        return str(value)

    @staticmethod
    def _pick(*values):
        """返回第一个非 NOT_FOUND / 空 的值"""
        for v in values:
            if v and v not in ("NOT_FOUND", "", None, "not mentioned"):
                return v
        return "NOT_FOUND"

    @staticmethod
    def _extract_num_classes(ds: dict) -> str:
        """从 number_of_classes 字符串提取纯数字: "38 classes" → "38"."""
        import re
        raw = str(ds.get("number_of_classes", ""))
        m = re.search(r'(\d+)', raw)
        return m.group(1) if m else "NOT_FOUND"

    # 图像数据信号词：命中任一即认为数据集是图像类（存在 color/grayscale/segmented
    # 等图像版本概念）。非图像数据（合成向量/RL 状态/无数据）不应套用图像模板。
    _IMAGE_DATA_SIGNALS = (
        "image", "图像", "图片", "rgb", "grayscale", "灰度", "segmented",
        "imagenet", "cifar", "mnist", "coco", "leaf", "叶片", "plantvillage",
        "植物", "pixel", "像素", "resize", "分辨率",
    )

    @staticmethod
    def _is_image_data(ds: dict) -> bool:
        """判断数据集是否为图像数据（用于决定是否套用图像预处理/迁移学习模板）。"""
        if not isinstance(ds, dict):
            return False
        text = " ".join(str(ds.get(k, "")) for k in
                        ("data_type", "dataset_name", "preprocessing", "dataset_source")).lower()
        return any(sig in text for sig in BaseAnalyzer._IMAGE_DATA_SIGNALS)
