"""
figure_analyzer.py

图片分析：直接使用视觉 LLM（MiMo-V2.6-Pro）分析图片，base64 编码传输。

替代旧流程：OCR → FigureCodeGenerator → CodeExecutor
新流程：base64(img) → llm_vision_call → JSON 结果

Author: Paper2Code
"""

import json
from pathlib import Path

from llm_manager import llm_vision_call, parse_llm_json
from logger import logger

# 视觉分析的 system prompt
VISION_SYSTEM = (
    "You are an expert at analyzing scientific paper figures, charts, tables, and diagrams. "
    "Always respond with valid JSON only, no extra text."
)

# 视觉分析的 user prompt 模板
VISION_PROMPT = """\
请分析这张论文图片/图表，提取与代码复现相关的关键信息。

要求返回严格的 JSON 格式，包含以下字段：
{{
  "type": "figure|table|chart|diagram|algorithm|other",
  "description": "图片的整体描述（1-2句话）",
  "important_components": ["关键组件1", "关键组件2", ...],
  "code_relevant_information": "与代码复现直接相关的信息（模型结构、参数、数据流、公式等）"
}}

注意事项：
- type 根据图片内容判断：流程图/架构图→diagram，数据表格→table，统计图→chart，算法伪代码→algorithm，实验结果图→figure
- important_components 列出图中可识别的关键元素（模块名、层名、数据维度、标注文字等）
- code_relevant_information 重点提取：模型结构、层参数、输入输出维度、损失函数、优化器、超参数、公式等
- 如果图片包含 LaTeX 公式，用文字描述公式含义
- 只返回 JSON，不要其他内容
"""


class FigureAnalyzer:

    def __init__(self):
        pass

    def analyze(self, image_path: str) -> dict:
        """
        使用视觉 LLM 分析单张图片。

        Args:
            image_path: 图片文件路径

        Returns:
            分析结果 dict:
            {"type": "...", "description": "...", "important_components": [...],
             "code_relevant_information": "..."}
        """
        logger.info(f"[Vision] 分析图片: {image_path}")

        # 检查文件存在
        if not Path(image_path).exists():
            logger.error(f"图片文件不存在: {image_path}")
            return self._error_result("image file not found")

        # 判断 MIME 类型
        mime_type = self._detect_mime(image_path)

        try:
            raw = llm_vision_call(
                VISION_PROMPT,
                image_path=image_path,
                mime_type=mime_type,
                system=VISION_SYSTEM,
                task_type="heavy",
            )
        except (RuntimeError, ValueError, TimeoutError) as e:
            logger.error(f"[Vision] 图片分析失败: {e}")
            return self._error_result(str(e))

        # 解析 JSON 结果
        result = parse_llm_json(raw)

        # 校验结构
        if not isinstance(result, dict) or "error" in result:
            logger.warning(f"[Vision] JSON 解析失败，使用兜底: {raw[:200]}")
            return {
                "type": "other",
                "description": raw[:500] if raw else "empty response",
                "important_components": [],
                "code_relevant_information": raw[:2000] if raw else "",
            }

        # 确保必要字段存在
        result.setdefault("type", "other")
        result.setdefault("description", "")
        result.setdefault("important_components", [])
        result.setdefault("code_relevant_information", "")

        logger.info(
            f"[Vision] 分析完成: type={result['type']}, "
            f"components={len(result['important_components'])}"
        )
        return result

    @staticmethod
    def _detect_mime(image_path: str) -> str:
        """根据文件扩展名判断 MIME 类型"""
        ext = Path(image_path).suffix.lower()
        return {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".gif": "image/gif",
            ".bmp": "image/bmp",
            ".webp": "image/webp",
        }.get(ext, "image/png")

    @staticmethod
    def _error_result(error: str) -> dict:
        return {
            "type": "other",
            "description": f"analysis failed: {error}",
            "important_components": [],
            "code_relevant_information": "",
        }
