"""
Step4 需求分析基类 —— 继承 Step5 的 BaseAnalyzer，复用三层防编造机制

独立于 step1-3，只读取 data/output/ 下的 JSON。
输出到 data/output/{paper}/step4/，与 Step5 的 step5/ 目录分离。
"""

from typing import List, Dict, Optional, Any, Tuple, Union

import json
from pathlib import Path

# 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
OUTPUT_ROOT = PROJECT_ROOT / "data" / "output"

from step5.base_analyzer import BaseAnalyzer
from logger import logger


class BaseRequirementAnalyzer(BaseAnalyzer):
    """
    Step4 需求分析基类。
    继承 Step5 的 BaseAnalyzer，复用:
      - key_info / structure 加载
      - 三层防编造 (rule → LLM → validate)
      - YAML 读写
      - NOT_FOUND 约定
    """

    # ---- 需求 ID 前缀 ----
    FR_PREFIX = "FR"       # Functional Requirement
    IF_PREFIX = "IF"       # InterFace
    AC_PREFIX = "AC"       # Acceptance Criterion
    GAP_PREFIX = "GAP"     # Gap

    def __init__(self, paper_name: str):
        super().__init__(paper_name)
        # 覆盖输出目录: Step4 → step4/, Step5 → step5/
        self.output_dir = self.paper_dir / "step4"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 步骤计数器 (用于生成 ID)
        self._counters = {self.FR_PREFIX: 0, self.IF_PREFIX: 0,
                          self.AC_PREFIX: 0, self.GAP_PREFIX: 0}

    # ============================================================
    # ID 生成
    # ============================================================

    def _next_id(self, prefix: str) -> str:
        """生成带序号的 ID，如 FR-01, IF-02"""
        self._counters[prefix] += 1
        return f"{prefix}-{self._counters[prefix]:02d}"

    def _reset_counter(self, prefix: Optional[str] = None):
        """重置计数器"""
        if prefix:
            self._counters[prefix] = 0
        else:
            for k in self._counters:
                self._counters[k] = 0

    # ============================================================
    # 通用辅助
    # ============================================================

    def _source(self, *paths: str) -> str:
        """
        生成来源追溯字符串。
        如: _source("methodology", "model_architecture", "alexnet")
          → "methodology.model_architecture.alexnet"
        """
        return ".".join(str(p) for p in paths if p)

    def _get_kv(self, *paths: str, default="NOT_FOUND") -> str:
        """
        从 key_info 中按路径取值，返回字符串。
        如: _get_kv("implementation_details", "optimizer") → "SGD"
        """
        data = self.key_info
        for p in paths:
            if isinstance(data, dict):
                data = data.get(p, default)
            else:
                return default
        return str(data) if data else default
