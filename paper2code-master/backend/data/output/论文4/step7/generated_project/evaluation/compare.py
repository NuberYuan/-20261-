#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""evaluation/compare.py

规则式图程序 (rule-based graph program) 与命令式基线 (imperative baseline)
的评估与报告生成模块。

本模块只提供评估/报告相关函数（evaluate_model / generate_report 及绘图辅助），
入口统一在 main.py。所有超参数从 config/config.yaml 读取（yaml.safe_load），
matplotlib 仅作为可选的可视化辅助依赖：缺失时指标计算与文本报告仍然全部完成。
"""

from __future__ import annotations

import copy
import json
import math
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence

try:
    import yaml
except ImportError:  # pragma: no cover - yaml 为可选依赖
    yaml = None

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover - matplotlib 为可选依赖
    plt = None


# ---------------------------------------------------------------------------
# 配置读取
# ---------------------------------------------------------------------------

_CONFIG_RELATIVE_PATH = os.path.join("config", "config.yaml")

_DEFAULT_CONFIG: Dict[str, Any] = {
    "project": {
        "name": "rule_based_graph_programs",
        "paper": "Rule-Based Graph Programs Matching the Time Complexity of Imperative Algorithms",
        "framework": "Python 标准库",
        "task": "other",
        "implementation_scope": "rule_based_graph_program_vs_imperative_baseline",
    },
    "data_source": "synthetic",
    "acquisition": "n/a",
    "dataset_versions": ["default"],
    "version_selector": {
        "config_key": "dataset_version",
        "values": ["default"],
        "default": "default",
        "note": "非图像数据仅一个 default 版本",
    },
    "experiment": {
        "name": "graph_program_time_complexity",
        "description": "对比规则式图程序与命令式算法在合成图上的时间复杂度",
        "runs": [],
    },
    "graph": {
        "generator": "erdos_renyi",
        "directed": False,
        "weighted": False,
        "self_loops": False,
        "multi_edges": False,
        "ensure_connected": True,
        "weight_min": 1,
        "weight_max": 10,
        "weight_integer": True,
    },
    "rules": {
        "match_mode": "sequential",
        "max_rule_depth": 3,
        "max_rounds": 10000,
        "apply_per_round": 1,
        "confluence_check": True,
        "termination_check": "fixed_point",
        "store_trace": False,
    },
    "algorithm": {
        "reachability": {"source_vertex": 0, "explore_neighbors": "adjacency_list"},
        "shortest_path": {
            "source_vertex": 0,
            "edge_weight_key": "weight",
            "unreachable_distance": 1.0e18,
        },
        "connected_components": {
            "label_key": "component_id",
            "union_mode": "rule_based",
        },
    },
    "benchmark": {
        "num_trials": 10,
        "warmup_runs": 3,
        "timing_unit": "seconds",
        "timer": "time.perf_counter",
        "timeout_seconds": 600,
        "verify_correctness": True,
        "comparison_baseline": "imperative",
        "aggregate_stat": "median",
        "report_quantiles": [0.25, 0.5, 0.75],
    },
    "logging": {
        "level": "INFO",
        "output_dir": "results",
        "log_file": "logs/run.log",
        "save_raw_timings": True,
        "export_format": "json",
        "timestamp_in_filename": True,
    },
    "system": {
        "device": "cpu",
        "num_workers": 0,
        "seed": 42,
    },
}


def _cfg_get(config: Optional[Dict[str, Any]], dotted_key: str, default: Any = None) -> Any:
    """按 ``a.b.c`` 形式逐级安全读取配置，任何一级缺失都返回 default。"""
    if not isinstance(config, dict):
        return default
    node: Any = config
    for part in dotted_key.split("."):
        if not isinstance(node, dict):
            return default
        node = node.get(part, None)
        if node is None:
            return default
    return node


def _load_config(config: Optional[Any] = None) -> Dict[str, Any]:
    """加载配置：传入 dict 直接返回；传入路径或 None 时从 config/config.yaml 读取。"""
    if isinstance(config, dict):
        return config
    cfg_path = config if isinstance(config, str) and config else _CONFIG_RELATIVE_PATH
    loaded: Dict[str, Any] = {}
    if yaml is not None:
        try:
            with open(cfg_path, "r", encoding="utf-8") as fh:
                loaded = yaml.safe_load(fh) or {}
        except (OSError, ValueError):
            loaded = {}
    if not isinstance(loaded, dict) or not loaded:
        loaded = copy.deepcopy(_DEFAULT_CONFIG)
    return loaded


# ---------------------------------------------------------------------------
# 通用工具
# ---------------------------------------------------------------------------

def _to_float_list(values: Optional[Sequence[Any]]) -> List[float]:
    """把任意序列转换为浮点数列表，忽略 None / 非数值项。"""
    if values is None:
        return []
    if isinstance(values, (int, float)) and not isinstance(values, bool):
        return [float(values)]
    result: List[float] = []
    try:
        iterator = list(values)
    except TypeError:
        return result
    for item in iterator:
        if item is None:
            continue
        if isinstance(item, bool):
            result.append(1.0 if item else 0.0)
            continue
        if isinstance(item, (int, float)):
            result.append(float(item))
            continue
        try:
            result.append(float(item))
        except (TypeError, ValueError):
            continue
    return result


def _quantile(sorted_values: Sequence[float], q: float) -> float:
    """对已排序序列做线性插值分位数计算（纯标准库实现）。"""
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    q_clamped = min(max(float(q), 0.0), 1.0)
    pos = q_clamped * (len(sorted_values) - 1)
    low = int(math.floor(pos))
    high = int(math.ceil(pos))
    if low == high:
        return float(sorted_values[low])
    frac = pos - low
    return float(sorted_values[low] * (1.0 - frac) + sorted_values[high] * frac)


def _safe_mean(values: Sequence[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _safe_std(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = _safe_mean(values)
    var = sum((v - mean) ** 2 for v in values) / float(len(values) - 1)
    return float(math.sqrt(var))


def _safe_div(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _sanitize_name(name: Any, default: str = "run") -> str:
    text = str(name).strip() if name is not None else ""
    if not text:
        text = default
    cleaned = []
    for ch in text:
        if ch.isalnum() or ch in ("-", "_", "."):
            cleaned.append(ch)
        else:
            cleaned.append("_")
    return "".join(cleaned) or default


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (int, float, str, bool)) or obj is None:
        return obj
    return str(obj)


# ---------------------------------------------------------------------------
# 指标计算
# ---------------------------------------------------------------------------

def _build_confusion_matrix(
    targets: Sequence[Any],
    predictions: Sequence[Any],
    labels: Sequence[Any],
) -> List[List[int]]:
    index = {label: i for i, label in enumerate(labels)}
    size = len(labels)
    matrix = [[0 for _ in range(size)] for _ in range(size)]
    for truth, pred in zip(targets, predictions):
        if truth not in index or pred not in index:
            continue
        matrix[index[truth]][index[pred]] += 1
    return matrix


def _compute_classification_metrics(
    targets: Sequence[Any],
    predictions: Sequence[Any],
    labels: Sequence[Any],
) -> Dict[str, Any]:
    matrix = _build_confusion_matrix(targets, predictions, labels)
    num_labels = len(labels)
    total = sum(sum(row) for row in matrix)
    correct = sum(matrix[i][i] for i in range(num_labels)) if num_labels else 0
    accuracy = _safe_div(float(correct), float(total)) if total else 0.0

    per_class: List[Dict[str, Any]] = []
    precisions: List[float] = []
    recalls: List[float] = []
    f1s: List[float] = []
    weighted_p = 0.0
    weighted_r = 0.0
    weighted_f = 0.0

    for i, label in enumerate(labels):
        tp = float(matrix[i][i]) if num_labels else 0.0
        fp = float(sum(matrix[r][i] for r in range(num_labels))) - tp
        fn = float(sum(matrix[i][c] for c in range(num_labels))) - tp
        support = tp + fn
        precision = _safe_div(tp, tp + fp)
        recall = _safe_div(tp, tp + fn)
        f1 = _safe_div(2.0 * precision * recall, precision + recall)
        per_class.append(
            {
                "label": label,
                "support": int(support),
                "tp": int(tp),
                "fp": int(fp),
                "fn": int(fn),
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
        weighted_p += precision * support
        weighted_r += recall * support
        weighted_f += f1 * support

    macro_precision = _safe_mean(precisions) if precisions else 0.0
    macro_recall = _safe_mean(recalls) if recalls else 0.0
    macro_f1 = _safe_mean(f1s) if f1s else 0.0
    weight_total = float(sum(item["support"] for item in per_class))
    weighted_precision = _safe_div(weighted_p, weight_total)
    weighted_recall = _safe_div(weighted_r, weight_total)
    weighted_f1 = _safe_div(weighted_f, weight_total)

    metrics: Dict[str, Any] = {
        "accuracy": float(accuracy),
        "precision": float(macro_precision),
        "recall": float(macro_recall),
        "f1": float(macro_f1),
        "mean_f1": float(macro_f1),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "weighted_precision": float(weighted_precision),
        "weighted_recall": float(weighted_recall),
        "weighted_f1": float(weighted_f1),
        "num_samples": int(total),
        "num_correct": int(correct),
        "labels": list(labels),
        "per_class": per_class,
        "confusion_matrix": matrix,
    }

    if num_labels == 2:
        positive = per_class[1]
        metrics["binary"] = {
            "positive_label": labels[1],
            "precision": positive["precision"],
            "recall": positive["recall"],
            "f1": positive["f1"],
            "accuracy": float(accuracy),
        }
    return metrics


def _compute_timing_metrics(
    timings: Optional[Sequence[Any]],
    config: Optional[Dict[str, Any]],
    prefix: str = "",
) -> Dict[str, Any]:
    values = _to_float_list(timings)
    values_sorted = sorted(values)
    unit = str(_cfg_get(config, "benchmark.timing_unit", "seconds"))
    aggregate_name = str(_cfg_get(config, "benchmark.aggregate_stat", "median"))
    quantile_keys = _cfg_get(config, "benchmark.report_quantiles", [0.25, 0.5, 0.75]) or []

    quantiles: Dict[str, float] = {}
    for q in quantile_keys:
        try:
            q_val = float(q)
        except (TypeError, ValueError):
            continue
        quantiles["q{:.2f}".format(q_val)] = _quantile(values_sorted, q_val)

    if values_sorted:
        aggregate = _quantile(values_sorted, 0.5) if "median" in aggregate_name else _safe_mean(values_sorted)
    else:
        aggregate = 0.0

    stats: Dict[str, Any] = {
        "unit": unit,
        "count": len(values),
        "mean": _safe_mean(values_sorted),
        "median": _quantile(values_sorted, 0.5),
        "std": _safe_std(values_sorted),
        "min": float(values_sorted[0]) if values_sorted else 0.0,
        "max": float(values_sorted[-1]) if values_sorted else 0.0,
        "aggregate_stat": aggregate_name,
        "aggregate": float(aggregate),
        "quantiles": quantiles,
    }
    if prefix:
        stats["raw_timings"] = list(values)
    return stats


def evaluate_model(
    predictions: Optional[Sequence[Any]] = None,
    targets: Optional[Sequence[Any]] = None,
    timings: Optional[Sequence[Any]] = None,
    config: Optional[Any] = None,
    baseline_timings: Optional[Sequence[Any]] = None,
    labels: Optional[Sequence[Any]] = None,
    run_name: str = "run",
) -> Dict[str, Any]:
    """计算评估指标（分类指标 + 计时统计 + 与基线的动态比较）。

    Args:
        predictions: 模型/规则程序的预测标签序列（可为 None，返回零值指标）。
        targets: 真实标签序列。
        timings: 规则式图程序的单次运行耗时序列。
        config: 配置字典或 config.yaml 路径；为 None 时读取 config/config.yaml。
        baseline_timings: 命令式基线的耗时序列（可选，用于动态计算比值）。
        labels: 显式标签集合（可选）。
        run_name: 当前运行名称（写入指标元信息）。

    Returns:
        Dict[str, Any]: 包含 accuracy / precision / recall / mean_f1 /
        confusion_matrix / timing 等字段的指标字典。
    """
    cfg = _load_config(config)

    pred_list = list(predictions) if predictions is not None else []
    targ_list = list(targets) if targets is not None else []
    if len(pred_list) != len(targ_list):
        keep = min(len(pred_list), len(targ_list))
        pred_list = pred_list[:keep]
        targ_list = targ_list[:keep]

    if labels is not None and len(list(labels)) > 0:
        label_list = list(labels)
    else:
        seen: List[Any] = []
        seen_set = set()
        for item in list(targ_list) + list(pred_list):
            try:
                if item not in seen_set:
                    seen_set.add(item)
                    seen.append(item)
            except TypeError:
                if item not in seen:
                    seen.append(item)
        try:
            label_list = sorted(seen)
        except TypeError:
            label_list = list(seen)

    metrics: Dict[str, Any] = _compute_classification_metrics(targ_list, pred_list, label_list)

    rule_timing = _compute_timing_metrics(timings, cfg, prefix="rule")
    metrics["timing"] = rule_timing
    metrics["timing_unit"] = rule_timing.get("unit", "seconds")

    save_raw = bool(_cfg_get(config, "logging.save_raw_timings", True)) if isinstance(config, dict) else True
    save_raw = bool(_cfg_get(cfg, "logging.save_raw_timings", True))
    if not save_raw:
        metrics["timing"].pop("raw_timings", None)

    comparison_baseline = str(_cfg_get(cfg, "benchmark.comparison_baseline", "imperative"))
    metrics["comparison_baseline"] = comparison_baseline

    if baseline_timings is not None:
        base_timing = _compute_timing_metrics(baseline_timings, cfg, prefix="baseline")
        if not save_raw:
            base_timing.pop("raw_timings", None)
        metrics["baseline_timing"] = base_timing
        base_mean = base_timing.get("mean", 0.0)
        base_median = base_timing.get("median", 0.0)
        rule_mean = rule_timing.get("mean", 0.0)
        rule_median = rule_timing.get("median", 0.0)
        mean_ratio = _safe_div(base_mean, rule_mean) if rule_mean else 0.0
        median_ratio = _safe_div(base_median, rule_median) if rule_median else 0.0
        metrics["speedup"] = {
            "baseline_name": comparison_baseline,
            "mean_ratio_baseline_over_rule": float(mean_ratio),
            "median_ratio_baseline_over_rule": float(median_ratio),
            "rule_mean": float(rule_mean),
            "baseline_mean": float(base_mean),
            "rule_median": float(rule_median),
            "baseline_median": float(base_median),
            "rule_faster_on_mean": bool(rule_mean < base_mean) if (rule_mean or base_mean) else False,
            "rule_faster_on_median": bool(rule_median < base_median) if (rule_median or base_median) else False,
        }

    metrics["verify_correctness_enabled"] = bool(_cfg_get(cfg, "benchmark.verify_correctness", True))
    metrics["num_trials_configured"] = int(_cfg_get(cfg, "benchmark.num_trials", 0) or 0)
    metrics["warmup_runs_configured"] = int(_cfg_get(cfg, "benchmark.warmup_runs", 0) or 0)
    metrics["run_name"] = str(run_name)
    metrics["experiment_name"] = str(_cfg_get(cfg, "experiment.name", "graph_program_time_complexity"))
    metrics["project_name"] = str(_cfg_get(cfg, "project.name", "rule_based_graph_programs"))
    metrics["seed"] = _cfg_get(cfg, "system.seed", 42)
    return metrics


# ---------------------------------------------------------------------------
# 绘图辅助（matplotlib 可选）
# ---------------------------------------------------------------------------

def _generate_metrics_bar_chart(
    metrics: Dict[str, Any],
    output_dir: str = "results",
    run_name: str = "report",
    config: Optional[Any] = None,
) -> Optional[str]:
    """绘制 mean_f1 / precision / recall / accuracy 指标柱状图。

    必须传入完整的 metrics dict（函数内部一次性读取四个指标）。
    matplotlib 缺失时直接返回 None。
    """
    if plt is None:
        return None
    if not isinstance(metrics, dict):
        return None

    keys = ("mean_f1", "precision", "recall", "accuracy")
    values: List[float] = []
    labels: List[str] = []
    for key in keys:
        raw = metrics.get(key)
        if raw is None:
            continue
        try:
            values.append(float(raw))
        except (TypeError, ValueError):
            continue
        labels.append(key)
    if not values:
        return None

    try:
        os.makedirs(output_dir, exist_ok=True)
        safe_run = _sanitize_name(run_name, "run")
        fig, ax = plt.subplots(figsize=(7.0, 4.5))
        bars = ax.bar(range(len(values)), values, color="#4C72B0", edgecolor="#2F4B7C")
        ax.set_xticks(range(len(values)))
        ax.set_xticklabels(labels, rotation=20, ha="right")
        ax.set_ylim(0.0, 1.05)
        ax.set_ylabel("score")
        ax.set_title("Evaluation metrics - {}".format(safe_run))
        ax.grid(axis="y", linestyle=":", alpha=0.5)
        for bar, value in zip(bars, values):
            ax.annotate(
                "{:.4f}".format(value),
                xy=(bar.get_x() + bar.get_width() / 2.0, bar.get_height()),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
            )
        fig.tight_layout()
        path = os.path.join(output_dir, "{}_metrics_bar.png".format(safe_run))
        fig.savefig(path, dpi=150)
        plt.close(fig)
        return path
    except Exception:
        try:
            plt.close("all")
        except Exception:
            pass
        return None


def _generate_confusion_matrix_plot(
    confusion_matrix: Sequence[Sequence[Any]],
    output_dir: str = "results",
    run_name: str = "report",
    labels: Optional[Sequence[Any]] = None,
    config: Optional[Any] = None,
) -> Optional[str]:
    """绘制混淆矩阵热力图。

    Args:
        confusion_matrix: metrics['confusion_matrix'] 二维数组。
        output_dir: 图片输出目录。
        run_name: 当前运行名称。
        labels: 类别标签（可选）。
        config: 配置（可选）。

    Returns:
        图片路径；matplotlib 缺失或输入非法时返回 None。
    """
    if plt is None:
        return None
    if not confusion_matrix:
        return None
    try:
        matrix = [[int(cell) for cell in row] for row in confusion_matrix]
    except (TypeError, ValueError):
        return None
    if not matrix or not matrix[0]:
        return None

    width = len(matrix[0])
    if any(len(row) != width for row in matrix):
        return None

    label_list = list(labels) if labels is not None else ["class_{}".format(i) for i in range(width)]
    if len(label_list) != width:
        label_list = ["class_{}".format(i) for i in range(width)]

    try:
        os.makedirs(output_dir, exist_ok=True)
        safe_run = _sanitize_name(run_name, "run")
        fig, ax = plt.subplots(figsize=(6.0, 5.0))
        im = ax.imshow(matrix, cmap="Blues", aspect="auto")
        ax.set_xticks(range(width))
        ax.set_yticks(range(width))
        ax.set_xticklabels([str(x) for x in label_list], rotation=30, ha="right")
        ax.set_yticklabels([str(x) for x in label_list])
        ax.set_xlabel("predicted label")
        ax.set_ylabel("true label")
        ax.set_title("Confusion matrix - {}".format(safe_run))
        for i in range(len(matrix)):
            for j in range(width):
                ax.text(j, i, str(matrix[i][j]), ha="center", va="center", fontsize=8, color="black")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        path = os.path.join(output_dir, "{}_confusion_matrix.png".format(safe_run))
        fig.savefig(path, dpi=150)
        plt.close(fig)
        return path
    except Exception:
        try:
            plt.close("all")
        except Exception:
            pass
        return None


# ---------------------------------------------------------------------------
# 报告生成
# ---------------------------------------------------------------------------

def _format_float(value: Any, digits: int = 6) -> str:
    try:
        return "{:.{prec}f}".format(float(value), prec=digits)
    except (TypeError, ValueError):
        return "n/a"


def _build_text_report(
    metrics: Dict[str, Any],
    run_name: str,
    plot_paths: Dict[str, Optional[str]],
    cfg: Dict[str, Any],
) -> str:
    lines: List[str] = []
    separator = "=" * 72

    lines.append(separator)
    lines.append("Evaluation report")
    lines.append(separator)
    lines.append("project.name          : {}".format(_cfg_get(cfg, "project.name", metrics.get("project_name", "n/a"))))
    lines.append("project.paper         : {}".format(_cfg_get(cfg, "project.paper", "n/a")))
    lines.append("project.framework     : {}".format(_cfg_get(cfg, "project.framework", "n/a")))
    lines.append("experiment.name       : {}".format(_cfg_get(cfg, "experiment.name", metrics.get("experiment_name", "n/a"))))
    lines.append("experiment.description: {}".format(_cfg_get(cfg, "experiment.description", "n/a")))
    lines.append("run_name              : {}".format(run_name))
    lines.append("generated_at          : {}".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    lines.append("data_source           : {}".format(_cfg_get(cfg, "data_source", "n/a")))
    lines.append("system.seed           : {}".format(_cfg_get(cfg, "system.seed", "n/a")))
    lines.append("system.device         : {}".format(_cfg_get(cfg, "system.device", "n/a")))
    lines.append("")

    lines.append("-" * 72)
    lines.append("Classification metrics")
    lines.append("-" * 72)
    num_samples = int(metrics.get("num_samples", 0) or 0)
    lines.append("num_samples           : {}".format(num_samples))
    if num_samples == 0:
        lines.append("note                  : no samples available, metrics are undefined/zero-filled")
    for key, label in (
        ("accuracy", "accuracy"),
        ("precision", "precision (macro)"),
        ("recall", "recall (macro)"),
        ("mean_f1", "mean_f1 (macro F1)"),
        ("weighted_precision", "weighted_precision"),
        ("weighted_recall", "weighted_recall"),
        ("weighted_f1", "weighted_f1"),
    ):
        lines.append("{:<22}: {}".format(label, _format_float(metrics.get(key, 0.0))))

    binary = metrics.get("binary")
    if isinstance(binary, dict):
        lines.append("")
        lines.append("binary metrics (positive_label={})".format(binary.get("positive_label", "n/a")))
        lines.append("  precision           : {}".format(_format_float(binary.get("precision", 0.0))))
        lines.append("  recall              : {}".format(_format_float(binary.get("recall", 0.0))))
        lines.append("  f1                  : {}".format(_format_float(binary.get("f1", 0.0))))

    per_class = metrics.get("per_class")
    if isinstance(per_class, list) and per_class:
        lines.append("")
        lines.append("-" * 72)
        lines.append("Per-class metrics")
        lines.append("-" * 72)
        header = "{:<16}{:>10}{:>12}{:>12}{:>12}{:>10}".format(
            "label", "support", "precision", "recall", "f1", "accuracy"
        )
        lines.append(header)
        overall_accuracy = float(metrics.get("accuracy", 0.0) or 0.0)
        for item in per_class:
            lines.append(
                "{:<16}{:>10}{:>12}{:>12}{:>12}{:>10}".format(
                    str(item.get("label", "n/a")),
                    str(item.get("support", 0)),
                    _format_float(item.get("precision", 0.0), 4),
                    _format_float(item.get("recall", 0.0), 4),
                    _format_float(item.get("f1", 0.0), 4),
                    _format_float(overall_accuracy, 4),
                )
            )

    confusion_matrix = metrics.get("confusion_matrix")
    if isinstance(confusion_matrix, list) and confusion_matrix:
        lines.append("")
        lines.append("-" * 72)
        lines.append("Confusion matrix (rows=true, cols=predicted)")
        lines.append("-" * 72)
        for row in confusion_matrix:
            lines.append("  " + " ".join("{:>6}".format(str(cell)) for cell in row))

    lines.append("")
    lines.append("-" * 72)
    lines.append("Timing statistics")
    lines.append("-" * 72)
    timing = metrics.get("timing")
    if isinstance(timing, dict) and timing:
        lines.append("timing_unit           : {}".format(timing.get("unit", "n/a")))
        lines.append("count                 : {}".format(timing.get("count", 0)))
        lines.append("mean                  : {}".format(_format_float(timing.get("mean", 0.0))))
        lines.append("median                : {}".format(_format_float(timing.get("median", 0.0))))
        lines.append("std                   : {}".format(_format_float(timing.get("std", 0.0))))
        lines.append("min                   : {}".format(_format_float(timing.get("min", 0.0))))
        lines.append("max                   : {}".format(_format_float(timing.get("max", 0.0))))
        lines.append("aggregate_stat        : {}".format(timing.get("aggregate_stat", "n/a")))
        lines.append("aggregate             : {}".format(_format_float(timing.get("aggregate", 0.0))))
        quantiles = timing.get("quantiles")
        if isinstance(quantiles, dict) and quantiles:
            for q_name in sorted(quantiles.keys()):
                lines.append("quantile {:<12}: {}".format(q_name, _format_float(quantiles[q_name])))
    else:
        lines.append("note                  : no timing samples recorded for this run")

    baseline_timing = metrics.get("baseline_timing")
    if isinstance(baseline_timing, dict) and baseline_timing:
        lines.append("")
        lines.append("baseline ({}) timing".format(metrics.get("comparison_baseline", "imperative")))
        lines.append("  mean                : {}".format(_format_float(baseline_timing.get("mean", 0.0))))
        lines.append("  median              : {}".format(_format_float(baseline_timing.get("median", 0.0))))
        lines.append("  min                 : {}".format(_format_float(baseline_timing.get("min", 0.0))))
        lines.append("  max                 : {}".format(_format_float(baseline_timing.get("max", 0.0))))

    speedup = metrics.get("speedup")
    if isinstance(speedup, dict):
        lines.append("")
        lines.append("Dynamic comparison (computed from the recorded timings of this run)")
        lines.append("  baseline_name                    : {}".format(speedup.get("baseline_name", "n/a")))
        lines.append("  mean_ratio_baseline_over_rule    : {}".format(_format_float(speedup.get("mean_ratio_baseline_over_rule", 0.0))))
        lines.append("  median_ratio_baseline_over_rule  : {}".format(_format_float(speedup.get("median_ratio_baseline_over_rule", 0.0))))
        lines.append("  rule_faster_on_mean              : {}".format(bool(speedup.get("rule_faster_on_mean", False))))
        lines.append("  rule_faster_on_median            : {}".format(bool(speedup.get("rule_faster_on_median", False))))

    lines.append("")
    lines.append("-" * 72)
    lines.append("Run configuration (read from config/config.yaml)")
    lines.append("-" * 72)
    lines.append("benchmark.num_trials       : {}".format(_cfg_get(cfg, "benchmark.num_trials", "n/a")))
    lines.append("benchmark.warmup_runs      : {}".format(_cfg_get(cfg, "benchmark.warmup_runs", "n/a")))
    lines.append("benchmark.aggregate_stat   : {}".format(_cfg_get(cfg, "benchmark.aggregate_stat", "n/a")))
    lines.append("benchmark.timer            : {}".format(_cfg_get(cfg, "benchmark.timer", "n/a")))
    lines.append("benchmark.verify_correctness: {}".format(_cfg_get(cfg, "benchmark.verify_correctness", "n/a")))
    lines.append("rules.match_mode           : {}".format(_cfg_get(cfg, "rules.match_mode", "n/a")))
    lines.append("rules.max_rule_depth       : {}".format(_cfg_get(cfg, "rules.max_rule_depth", "n/a")))
    lines.append("rules.max_rounds           : {}".format(_cfg_get(cfg, "rules.max_rounds", "n/a")))
    lines.append("rules.termination_check    : {}".format(_cfg_get(cfg, "rules.termination_check", "n/a")))
    lines.append("graph.generator            : {}".format(_cfg_get(cfg, "graph.generator", "n/a")))

    lines.append("")
    lines.append("-" * 72)
    lines.append("Artifacts")
    lines.append("-" * 72)
    for key, path in plot_paths.items():
        if path:
            lines.append("{:<24}: {}".format(key, path))
        else:
            reason = "matplotlib unavailable" if plt is None else "not generated"
            lines.append("{:<24}: skipped ({})".format(key, reason))

    lines.append("")
    lines.append("-" * 72)
    lines.append("Computed observations (derived from this run's measurements)")
    lines.append("-" * 72)
    if isinstance(per_class, list) and per_class:
        scored = [item for item in per_class if isinstance(item, dict)]
        if scored:
            best = max(scored, key=lambda item: float(item.get("f1", 0.0) or 0.0))
            worst = min(scored, key=lambda item: float(item.get("f1", 0.0) or 0.0))
            lines.append(
                "class with highest f1: {} (f1={})".format(
                    best.get("label", "n/a"), _format_float(best.get("f1", 0.0), 4)
                )
            )
            lines.append(
                "class with lowest  f1: {} (f1={})".format(
                    worst.get("label", "n/a"), _format_float(worst.get("f1", 0.0), 4)
                )
            )
    if isinstance(speedup, dict):
        ratio = float(speedup.get("mean_ratio_baseline_over_rule", 0.0) or 0.0)
        if ratio > 0.0:
            lines.append("baseline_mean / rule_mean = {} (computed from raw timings)".format(_format_float(ratio, 4)))
    lines.append("accuracy - mean_f1 = {}".format(
        _format_float(float(metrics.get("accuracy", 0.0) or 0.0) - float(metrics.get("mean_f1", 0.0) or 0.0), 6)
    ))
    lines.append("")
    lines.append(separator)
    return "\n".join(lines) + "\n"


def generate_report(
    metrics: Optional[Dict[str, Any]] = None,
    run_name: str = "run",
    output_dir: Optional[str] = None,
    config: Optional[Any] = None,
) -> str:
    """生成单个 run 的评估报告（文本 + 可选 JSON + 可选图表）。

    Args:
        metrics: evaluate_model 返回的指标字典。
        run_name: 当前运行名称。
        output_dir: 报告输出目录；为 None 时使用 config['logging.output_dir']。
        config: 配置字典或 config.yaml 路径。

    Returns:
        str: 文本报告文件路径。
    """
    cfg = _load_config(config)
    metrics_dict: Dict[str, Any] = metrics if isinstance(metrics, dict) else {}

    target_dir = output_dir if output_dir else str(_cfg_get(cfg, "logging.output_dir", "results"))
    if not isinstance(target_dir, str) or not target_dir.strip():
        target_dir = "results"
    os.makedirs(target_dir, exist_ok=True)

    safe_run = _sanitize_name(run_name, "run")
    timestamp_flag = bool(_cfg_get(cfg, "logging.timestamp_in_filename", True))
    suffix = datetime.now().strftime("%Y%m%d_%H%M%S") if timestamp_flag else "report"

    plot_paths: Dict[str, Optional[str]] = {}
    plot_paths["metrics_bar_chart"] = _generate_metrics_bar_chart(
        metrics_dict, target_dir, safe_run, cfg
    )
    confusion_matrix = metrics_dict.get("confusion_matrix")
    if isinstance(confusion_matrix, list) and confusion_matrix:
        plot_paths["confusion_matrix_plot"] = _generate_confusion_matrix_plot(
            confusion_matrix,
            target_dir,
            safe_run,
            metrics_dict.get("labels"),
            cfg,
        )
    else:
        plot_paths["confusion_matrix_plot"] = None

    report_text = _build_text_report(metrics_dict, safe_run, plot_paths, cfg)
    report_path = os.path.join(target_dir, "{}_{}_report.txt".format(safe_run, suffix))
    with open(report_path, "w", encoding="utf-8") as fh:
        fh.write(report_text)

    export_format = str(_cfg_get(cfg, "logging.export_format", "json")).lower()
    if export_format == "json":
        payload = {
            "run_name": safe_run,
            "generated_at": datetime.now().isoformat(),
            "metrics": _jsonable(metrics_dict),
            "artifacts": {k: v for k, v in plot_paths.items()},
            "config_snapshot": _jsonable(cfg),
        }
        json_path = os.path.join(target_dir, "{}_{}_metrics.json".format(safe_run, suffix))
        try:
            with open(json_path, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, ensure_ascii=False, indent=2, default=str)
        except (OSError, TypeError, ValueError):
            pass

    return report_path