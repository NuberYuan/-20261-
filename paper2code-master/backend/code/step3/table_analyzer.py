"""
table_analyzer.py

使用 LLM 分析从 PDF 提取的表格数据，提取实验数据、对比结果等结构化信息。

流程:
  1. 读取 Step1 输出的表格 (header + rows)
  2. 格式化为 LLM 可理解的文本表示
  3. 调用 LLM 解析表格内容 → 结构化 JSON
  4. 提取: 实验结果、消融研究、参数对比、数据集统计

Author: Paper2Code
"""

from typing import List, Optional

from concurrent.futures import ThreadPoolExecutor

from llm_manager import llm_call, parse_llm_json
from logger import logger


_ANALYSIS_PROMPT = """You are a research paper reproduction expert. Analyze the following table extracted from an academic paper.

Your task:
1. Identify what this table represents:
   - experiment_results: main experiment results (accuracy, precision, etc.)
   - ablation_study: ablation experiments testing individual components
   - comparison: comparing different methods/models
   - dataset_statistics: dataset composition, class distribution
   - hyperparameter_table: hyperparameter settings
   - model_architecture: architecture details by layer
   - other: anything else

2. Extract ALL numerical data precisely. Do not round or approximate.

3. For experiment results, extract:
   - metric names (e.g., accuracy, F1-score, precision, recall)
   - values for each method/configuration
   - the best result and which method achieved it

4. For comparison tables, extract:
   - methods being compared
   - metrics and their values
   - which method performs best

Return ONLY valid JSON in this exact format (no markdown fences, no extra text):
{
  "table_id": "<table id from input>",
  "table_type": "experiment_results | ablation_study | comparison | dataset_statistics | hyperparameter_table | model_architecture | other",
  "caption_summary": "<one-line summary of what this table shows>",
  "extracted_data": {
    "metrics": ["metric1", "metric2", ...],
    "methods": ["method1", "method2", ...],
    "best_result": {"method": "...", "metric": "...", "value": ...},
    "key_values": [
      {"label": "...", "value": "..."},
      ...
    ]
  },
  "table_summary": "<2-3 sentence paragraph summarizing the table's findings>"
}

If you cannot determine the table type or content, set table_type to "other" and provide your best effort.
Do NOT include markdown code fences (```json) in your output. Return raw JSON only."""


class TableAnalyzer:
    """使用 LLM 分析论文表格，提取结构化信息"""

    def __init__(self):
        pass

    # =========================
    # 主入口
    # =========================

    def analyze(self, tables: List[dict], paper_context: str = "") -> dict:
        """
        分析所有表格。

        Args:
            tables: Step1 输出的表格列表 [{"id": ..., "header": [...], "rows": [[...], ...]}, ...]
            paper_context: 论文摘要或标题，用于提供上下文

        Returns:
            {
              "tables_analyzed": 3,
              "table_results": [
                { ... per-table analysis ... },
              ],
              "aggregated_findings": {
                "experiment_results": [...],
                "comparisons": [...],
                "ablation_studies": [...],
                "best_reported_result": "...",
              }
            }
        """
        if not tables:
            logger.info("没有表格需要分析")
            return {
                "tables_analyzed": 0,
                "table_results": [],
                "aggregated_findings": {},
            }

        logger.info(f"开始分析 {len(tables)} 个表格")

        # 表格之间互不依赖，且每张表的 LLM 调用是纯网络 I/O，
        # 用线程池并行分析，ex.map 会按输入顺序返回结果。
        def _analyze_one(table: dict) -> Optional[dict]:
            logger.info(f"  分析 {table.get('id', '?')} (method={table.get('method', '?')})")
            return self._analyze_single_table(table, paper_context)

        max_workers = min(len(tables), 5)
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            table_results = [r for r in ex.map(_analyze_one, tables) if r]

        # ---- 汇总 ----
        aggregated = self._aggregate(table_results, paper_context)

        result = {
            "tables_analyzed": len(table_results),
            "table_results": table_results,
            "aggregated_findings": aggregated,
        }

        logger.info(f"表格分析完成: {len(table_results)}/{len(tables)} 个成功")
        return result

    # =========================
    # 单个表格分析
    # =========================

    def _analyze_single_table(self, table: dict, context: str) -> Optional[dict]:
        """调用 LLM 分析单个表格"""
        table_text = self._format_table(table)

        # 限制输入长度（LLM 64K context，单个表格不会超）
        if len(table_text) > 12000:
            table_text = table_text[:12000] + "\n... [表格内容已截断]"

        prompt = f"{_ANALYSIS_PROMPT}\n\n--- Paper Context ---\n{context[:500]}\n\n--- Table to Analyze ---\n{table_text}"

        try:
            response = llm_call(prompt, task_type="heavy")
            parsed = parse_llm_json(response)

            if parsed is None or "error" in parsed:
                logger.warning(f"    表格 {table.get('id')} LLM 返回无法解析，使用规则提取")
                parsed = self._fallback_extract(table)

            # 注入 table id
            parsed["table_id"] = table.get("id", "unknown")
            return parsed

        except Exception as e:
            logger.warning(f"    表格 {table.get('id')} 分析失败: {e}")
            fallback = self._fallback_extract(table)
            fallback["table_id"] = table.get("id", "unknown")
            return fallback

    # =========================
    # 表格格式化
    # =========================

    @staticmethod
    def _format_table(table: dict) -> str:
        """将表格格式化为 LLM 可理解的文本"""
        lines = []

        tid = table.get("id", "?")
        method = table.get("method", "?")
        caption = table.get("caption", "")
        page = table.get("page", "?")

        lines.append(f"Table ID: {tid}")
        lines.append(f"Source Page: {page}")
        lines.append(f"Detection Method: {method}")
        if caption:
            lines.append(f"Caption: {caption}")

        header = table.get("header", [])
        rows = table.get("rows", [])

        if header:
            lines.append("")
            lines.append(" | ".join(str(h) for h in header))
            lines.append("-" * 60)

        for row in rows:
            lines.append(" | ".join(str(c) for c in row))

        return "\n".join(lines)

    # =========================
    # JSON 解析
    # =========================
    # 规则兜底
    # =========================

    @staticmethod
    def _fallback_extract(table: dict) -> dict:
        """
        当 LLM 解析失败时，用简单规则提取数值。
        至少保留原始数据，避免信息丢失。
        """
        header = table.get("header", [])
        rows = table.get("rows", [])

        # 尝试找数值列
        numeric_values = []
        for row in rows:
            for i, cell in enumerate(row):
                col_name = header[i] if i < len(header) else f"col_{i}"
                cell_str = str(cell).strip()
                # 检测可能的数值
                try:
                    # 去掉 % 和逗号
                    cleaned = cell_str.replace("%", "").replace(",", "").replace("±", " ")
                    parts = cleaned.split()
                    for p in parts:
                        v = float(p)
                        numeric_values.append({
                            "column": col_name,
                            "row_label": str(row[0]) if row else "",
                            "value": v,
                            "raw": cell_str,
                        })
                except ValueError:
                    pass

        return {
            "table_id": table.get("id", "unknown"),
            "table_type": "other",
            "caption_summary": table.get("caption", ""),
            "extracted_data": {
                "metrics": [],
                "methods": [],
                "best_result": {},
                "key_values": numeric_values[:20] if numeric_values else [],
            },
            "table_summary": f"表格包含 {len(rows)} 行 x {len(header)} 列数据。LLM 解析失败，已通过规则提取 {len(numeric_values)} 个数值。",
        }

    # =========================
    # 汇总
    # =========================

    @staticmethod
    def _metric_direction(metric: str) -> int:
        """判断指标方向：1=越大越好，-1=越小越好，0=无法判断。

        accuracy/F1/precision/recall/AUC 等越大越好；error/loss/RMSE/MAE/困惑度等越小越好。
        无法判断时返回 0，调用方应跳过该指标，避免把"越高越好"套用到 loss 上。
        """
        m = (metric or "").lower()
        if not m:
            return 0
        # 先匹配"越小越好"：注意 "mape" 同时包含 "map"，必须先于 "map" 判断
        lower_kw = [
            "error", "loss", "rmse", "mae", "mse", "mape", "perplexity",
            "cross-entropy", "crossentropy", "log loss", "logloss", "log_loss",
            "wer", "distance", "brier", "regret",
        ]
        higher_kw = [
            "accuracy", "acc", "f1", "f-score", "fscore", "f-measure",
            "precision", "recall", "auc", "auroc", "auprc", "map", "iou",
            "dice", "sensitivity", "specificity", "r2", "r²", "correlation",
            "pearson", "spearman", "bleu", "rouge", "meteor", "cider", "kappa",
            "mcc", "gmean",
        ]
        if any(k in m for k in lower_kw):
            return -1
        if any(k in m for k in higher_kw):
            return 1
        return 0

    def _aggregate(self, table_results: List[dict], context: str) -> dict:
        """
        汇总所有表格分析结果。

        如果多张表格需要交叉分析，调用 LLM 进行汇总。
        """
        if not table_results:
            return {}

        experiment_results = []
        comparisons = []
        ablation_studies = []
        best_higher = None   # (value, text) — 越大越好的指标（accuracy/F1/...）
        best_lower = None    # (value, text) — 越小越好的指标（error/loss/...）

        for t in table_results:
            ttype = t.get("table_type", "other")
            summary = t.get("table_summary", "")

            if ttype == "experiment_results":
                experiment_results.append(summary)
            elif ttype == "comparison":
                comparisons.append(summary)
            elif ttype == "ablation_study":
                ablation_studies.append(summary)

            # 追踪最佳结果（区分指标方向，避免把 loss 误判成"越大越好"）
            best = t.get("extracted_data", {}).get("best_result", {})
            if best and best.get("value"):
                try:
                    v = float(best["value"])
                    metric = str(best.get("metric", ""))
                    direction = self._metric_direction(metric)
                    text = f"{best.get('method', '?')}: {metric} = {best['value']}"
                    if direction == 1 and (best_higher is None or v > best_higher[0]):
                        best_higher = (v, text)
                    elif direction == -1 and (best_lower is None or v < best_lower[0]):
                        best_lower = (v, text)
                    # direction == 0：方向无法判断，跳过，不参与最优比较
                except (ValueError, TypeError):
                    pass

        best_reported = "; ".join(
            x[1] for x in (best_higher, best_lower) if x is not None
        )

        # 如果表格较多，用 LLM 做汇总
        aggregated_summary = ""
        if len(table_results) >= 2:
            aggregated_summary = self._llm_aggregate(table_results, context)

        return {
            "experiment_results": experiment_results,
            "comparisons": comparisons,
            "ablation_studies": ablation_studies,
            "best_reported_result": best_reported,
            "aggregated_summary": aggregated_summary,
        }

    def _llm_aggregate(self, table_results: List[dict], context: str) -> str:
        """使用 LLM 汇总多张表格的发现"""
        summaries = []
        for t in table_results:
            summaries.append(
                f"[{t.get('table_id')}] ({t.get('table_type')}) {t.get('table_summary', '')}"
            )

        prompt = f"""Summarize the following table analysis results from a research paper into 3-5 key findings.

Paper context: {context[:300]}

Table analyses:
{chr(10).join(summaries[:8])}

Return a concise paragraph summarizing the main experimental findings across all tables."""

        try:
            response = llm_call(prompt, task_type="heavy")
            return response.strip()[:800]
        except Exception as e:
            logger.warning(f"汇总 LLM 调用失败: {e}")
            return ""
