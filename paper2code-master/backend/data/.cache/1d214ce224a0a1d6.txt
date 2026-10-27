# -*- coding: utf-8 -*-
"""evaluation/evaluate.py —— 模型评估与报告生成。

实现内容
--------
1. ``evaluate_model``   : 在测试结果上计算评估指标，重点输出论文要求的
                          runtime measurements（均值 / 中位数 / 分位数等），
                          并与命令式基线的实测运行时做动态比较。
2. ``generate_report``  : 为单个 run 生成文本报告 + (可选) JSON 导出 +
                          工程辅助图表，返回该 run 的报告文件路径。
3. ``_generate_metrics_bar_chart``       : 绘制 mean_f1/precision/recall/accuracy 条形图。
4. ``_generate_confusion_matrix_plot``   : 绘制混淆矩阵热力图。

约束
----
* 仅依赖 Python 标准库；matplotlib 为可选依赖，缺失时跳过绘图，不影响指标计算与文本报告。
* 所有超参数从 ``config/config.yaml`` 读取（yaml.safe_load），逐级安全取值。
* 报告中的任何比较性措辞均由实测数值动态生成，不写死论文结论。
"""

from __future__ import annotations

import json
import logging
import math
import os
import statistics
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

try:  # matplotlib 为可选依赖
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: F401
except ImportError:  # pragma: no cover
    matplotlib = None
    plt = None

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = os.path.join("config", "config.yaml")

# 仅当 config/config.yaml 不可读取时使用的最小兜底值（与 config.yaml 保持一致）
_FALLBACK_CONFIG: Dict[str, Any] = {
    "project": {
        "name": "rule_based_graph_programs",
        "paper": "Rule-Based Graph Programs Matching the Time Complexity of Imperative Algorithms",
        "framework": "Python 标准库",
        "task": "other",
        "implementation_scope": "rule_based_graph_program_vs_imperative_baseline",
    },
    "experiment": {
        "name": "graph_program_time_complexity",
        "description": "对比规则式图程序与命令式算法在合成图上的时间复杂度",
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
    "system": {"device": "cpu", "num_workers": 0, "seed": 42},
}

# ---------------------------------------------------------------------------
# 配置读取
# ---------------------------------------------------------------------------


def load_config(config_path: Optional[str] = DEFAULT_CONFIG_PATH) -> Dict[str, Any]:
    """读取 ``config/config.yaml``；文件或 yaml 不可用时返回兜底配置字典。"""
    cfg: Dict[str, Any] = {}
    path = config_path if config_path else DEFAULT_CONFIG_PATH
    if yaml is not None and path and os.path.isfile(path):
        try:
            with open(path, "r", encoding="utf-8") as handle:
                loaded = yaml.safe_load(handle)
            if isinstance(loaded, dict):
                cfg = loaded
        except Exception as exc:  # noqa: BLE001
            logger.warning("读取 %s 失败(%s)，使用兜底配置。", path, exc)
    if not cfg:
        logger.warning("config.yaml 不可读取，使用内置兜底配置。")
        cfg = json.loads(json.dumps(_FALLBACK_CONFIG))
    return cfg


def _cfg_get(config: Optional[Dict[str, Any]], *keys: str, default: Any = None) -> Any:
    """逐级安全取值：任意一级缺失即返回 default。"""
    if not isinstance(config, dict):
        return default
    node: Any = config
    for key in keys:
        if not isinstance(node, dict):
            return default
        if key not in node:
            return default
        node = node[key]
        if node is None:
            return default
    return node


def _resolve_config(config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if isinstance(config, dict) and config:
        return config
    return load_config()


# ---------------------------------------------------------------------------
# 基础数值工具
# ---------------------------------------------------------------------------


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(result) or math.isinf(result):
        return default
    return result


def _to_float_list(values: Any) -> List[float]:
    """把标量 / 嵌套序列 / 带 ``tolist()`` 的对象展开成 float 列表。"""
    out: List[float] = []
    if values is None:
        return out
    if hasattr(values, "tolist") and not isinstance(values, (list, tuple, dict)):
        try:
            values = values.tolist()
        except Exception:  # noqa: BLE001
            values = [values]
    if isinstance(values, dict):
        for item in values.values():
            out.extend(_to_float_list(item))
        return out
    if isinstance(values, (list, tuple)):
        for item in values:
            out.extend(_to_float_list(item))
        return out
    try:
        out.append(float(values))
    except (TypeError, ValueError):
        pass
    return out


def _to_int_list(values: Any) -> List[int]:
    out: List[int] = []
    if values is None:
        return out
    if hasattr(values, "tolist") and not isinstance(values, (list, tuple, dict)):
        try:
            values = values.tolist()
        except Exception:  # noqa: BLE001
            values = [values]
    if isinstance(values, dict):
        for item in values.values():
            out.extend(_to_int_list(item))
        return out
    if isinstance(values, (list, tuple)):
        for item in values:
            out.extend(_to_int_list(item))
        return out
    try:
        out.append(int(round(float(values))))
    except (TypeError, ValueError):
        pass
    return out


def _percentile(sorted_values: Sequence[float], q: float) -> float:
    """线性插值分位数（q ∈ [0, 1]）。"""
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    q = min(max(float(q), 0.0), 1.0)
    position = (len(sorted_values) - 1) * q
    low = int(math.floor(position))
    high = int(math.ceil(position))
    if low == high:
        return float(sorted_values[low])
    weight = position - low
    return float(sorted_values[low] * (1.0 - weight) + sorted_values[high] * weight)


def _looks_like_size_map(obj: Any) -> bool:
    """判断 timings 是否是 {图规模: 运行时列表} 形式的映射。"""
    if not isinstance(obj, dict) or not obj:
        return False
    for key, value in obj.items():
        try:
            float(key)
        except (TypeError, ValueError):
            return False
        if not isinstance(value, (list, tuple)) and not hasattr(value, "tolist"):
            return False
    return True


# ---------------------------------------------------------------------------
# 运行时统计
# ---------------------------------------------------------------------------


def _aggregate_timings(
    timings: Any,
    quantiles: Optional[Sequence[float]] = None,
    timing_unit: str = "seconds",
    save_raw: bool = True,
) -> Dict[str, Any]:
    """对一组运行时采样做聚合统计（count/mean/median/std/min/max/quantiles）。"""
    values = sorted(_to_float_list(timings))
    if quantiles is None:
        quantiles = [0.25, 0.5, 0.75]
    quantile_map: Dict[str, float] = {}
    for q in quantiles:
        try:
            qf = float(q)
        except (TypeError, ValueError):
            continue
        quantile_map["q{:.2f}".format(qf)] = _percentile(values, qf)

    if not values:
        result: Dict[str, Any] = {
            "count": 0,
            "mean": 0.0,
            "median": 0.0,
            "std": 0.0,
            "min": 0.0,
            "max": 0.0,
            "quantiles": quantile_map,
            "unit": timing_unit,
            "raw": [] if save_raw else None,
        }
        return result

    mean_val = float(statistics.fmean(values)) if hasattr(statistics, "fmean") else float(sum(values) / len(values))
    median_val = float(statistics.median(values))
    std_val = float(statistics.pstdev(values)) if len(values) > 1 else 0.0

    result = {
        "count": len(values),
        "mean": mean_val,
        "median": median_val,
        "std": std_val,
        "min": float(values[0]),
        "max": float(values[-1]),
        "quantiles": quantile_map,
        "unit": timing_unit,
        "raw": list(values) if save_raw else None,
    }
    if timing_unit == "seconds":
        result["mean_ms"] = mean_val * 1000.0
        result["median_ms"] = median_val * 1000.0
    return result


# ---------------------------------------------------------------------------
# 分类指标
# ---------------------------------------------------------------------------


def _compute_confusion_matrix(
    predictions: Sequence[int],
    labels: Sequence[int],
    num_classes: Optional[int] = None,
) -> Tuple[List[List[int]], List[int]]:
    """返回 (confusion_matrix, class_labels)；confusion_matrix[i][j] = 真实 i 预测 j。"""
    pred = _to_int_list(predictions)
    true = _to_int_list(labels)
    size = min(len(pred), len(true))
    pred, true = pred[:size], true[:size]

    if num_classes is None:
        max_label = max(pred + true) if (pred or true) else -1
        num_classes = max(max_label + 1, 0)
    num_classes = max(int(num_classes), 0)

    matrix = [[0 for _ in range(num_classes)] for _ in range(num_classes)]
    for p, t in zip(pred, true):
        if 0 <= t < num_classes and 0 <= p < num_classes:
            matrix[t][p] += 1
    class_labels = list(range(num_classes))
    return matrix, class_labels


def _classification_metrics(
    predictions: Sequence[int],
    labels: Sequence[int],
    num_classes: Optional[int] = None,
) -> Dict[str, Any]:
    """计算 accuracy / macro-precision / macro-recall / mean_f1 与混淆矩阵。"""
    pred = _to_int_list(predictions)
    true = _to_int_list(labels)
    size = min(len(pred), len(true))
    pred, true = pred[:size], true[:size]

    matrix, class_labels = _compute_confusion_matrix(pred, true, num_classes)
    total = sum(sum(row) for row in matrix)
    correct = sum(matrix[i][i] for i in range(len(matrix))) if matrix else 0
    accuracy = (correct / total) if total > 0 else 0.0

    per_class: Dict[str, Dict[str, float]] = {}
    precisions: List[float] = []
    recalls: List[float] = []
    f1s: List[float] = []
    for idx in range(len(matrix)):
        tp = float(matrix[idx][idx])
        fp = float(sum(matrix[r][idx] for r in range(len(matrix)) if r != idx))
        fn = float(sum(matrix[idx][c] for c in range(len(matrix[idx])) if c != idx))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        per_class[str(class_labels[idx])] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": int(tp + fn),
        }
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

    mean_precision = float(sum(precisions) / len(precisions)) if precisions else 0.0
    mean_recall = float(sum(recalls) / len(recalls)) if recalls else 0.0
    mean_f1 = float(sum(f1s) / len(f1s)) if f1s else 0.0

    return {
        "accuracy": accuracy,
        "precision": mean_precision,
        "recall": mean_recall,
        "mean_f1": mean_f1,
        "f1_macro": mean_f1,
        "per_class": per_class,
        "confusion_matrix": matrix,
        "class_labels": class_labels,
        "num_samples": total,
    }


# ---------------------------------------------------------------------------
# 复杂度估计（由实测数据动态拟合，不写死结论）
# ---------------------------------------------------------------------------


def _estimate_complexity(timings_by_size: Optional[Dict[Any, Any]]) -> Optional[Dict[str, Any]]:
    """对 {规模: 运行时列表} 做 log-log 最小二乘拟合，返回斜率与经验复杂度标签。"""
    if not isinstance(timings_by_size, dict) or len(timings_by_size) < 2:
        return None

    points: List[Tuple[float, float]] = []
    for size, vals in timings_by_size.items():
        try:
            size_f = float(size)
        except (TypeError, ValueError):
            continue
        if size_f <= 0:
            continue
        samples = sorted(_to_float_list(vals))
        if not samples:
            continue
        median_val = _percentile(samples, 0.5)
        if median_val > 0:
            points.append((size_f, median_val))
    if len(points) < 2:
        return None

    points.sort(key=lambda item: item[0])
    xs = [math.log(p[0]) for p in points]
    ys = [math.log(p[1]) for p in points]
    n = len(xs)
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    sxx = sum((x - mean_x) ** 2 for x in xs)
    if sxx <= 0:
        return None
    slope = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) / sxx
    intercept = mean_y - slope * mean_x
    ss_tot = sum((y - mean_y) ** 2 for y in ys)
    ss_res = sum((y - (slope * x + intercept)) ** 2 for x, y in zip(xs, ys))
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 1.0

    if 0.7 <= slope <= 1.3:
        label = "approximately_linear"
    elif 1.3 < slope <= 2.3:
        label = "between_linear_and_quadratic"
    elif slope > 2.3:
        label = "super_quadratic"
    else:
        label = "sub_linear"

    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "r2": float(r2),
        "label": label,
        "sizes": [int(p[0]) if float(p[0]).is_integer() else p[0] for p in points],
        "median_times": [p[1] for p in points],
    }


# ---------------------------------------------------------------------------
# 对外接口 1：evaluate_model
# ---------------------------------------------------------------------------


def evaluate_model(
    predictions: Optional[Any] = None,
    labels: Optional[Any] = None,
    timings: Optional[Any] = None,
    baseline_timings: Optional[Any] = None,
    timings_by_size: Optional[Dict[Any, Any]] = None,
    reference_outputs: Optional[Any] = None,
    actual_outputs: Optional[Any] = None,
    config: Optional[Dict[str, Any]] = None,
    run_name: str = "default",
    num_classes: Optional[int] = None,
) -> Dict[str, Any]:
    """在测试结果上计算评估指标，重点产出 runtime measurements。

    参数
    ----
    predictions / labels : 可选，分类任务的预测与真值（序列或带 ``tolist()`` 的对象）。
    timings : 规则式图程序的运行时采样（秒）。若传入 ``{规模: [采样]}`` 形式的映射，
              会自动作为 ``timings_by_size`` 用于经验复杂度拟合。
    baseline_timings : 命令式基线的运行时采样（用于动态比较）。
    timings_by_size : 可选，按图规模聚合的运行时采样。
    reference_outputs / actual_outputs : 可选，用于正确性校验的输出序列。
    config : ``config/config.yaml`` 解析出的字典；为 None 时自动加载。
    run_name : 运行名称，用于报告文件命名。
    num_classes : 类别数；为 None 时由标签推断。

    返回
    ----
    包含 accuracy/precision/recall/mean_f1/confusion_matrix、
    runtime_measurements、comparison、empirical_complexity 的字典。
    """
    cfg = _resolve_config(config)

    timing_unit = str(_cfg_get(cfg, "benchmark", "timing_unit", default="seconds"))
    quantiles_raw = _cfg_get(cfg, "benchmark", "report_quantiles", default=[0.25, 0.5, 0.75])
    if not isinstance(quantiles_raw, (list, tuple)):
        quantiles_raw = [0.25, 0.5, 0.75]
    save_raw = bool(_cfg_get(cfg, "logging", "save_raw_timings", default=True))
    baseline_name = str(_cfg_get(cfg, "benchmark", "comparison_baseline", default="imperative"))
    verify_correctness = bool(_cfg_get(cfg, "benchmark", "verify_correctness", default=True))

    # ---- 时间采样归一化 -------------------------------------------------
    if isinstance(timings, dict) and _looks_like_size_map(timings):
        if timings_by_size is None:
            timings_by_size = timings
        flat: List[float] = []
        for vals in timings.values():
            flat.extend(_to_float_list(vals))
        timings = flat

    timings_list = _to_float_list(timings)
    baseline_list = _to_float_list(baseline_timings)

    runtime_stats = _aggregate_timings(timings_list, quantiles_raw, timing_unit, save_raw)
    baseline_stats = (
        _aggregate_timings(baseline_list, quantiles_raw, timing_unit, save_raw) if baseline_list else None
    )

    runtime_measurements: Dict[str, Any] = {
        "rule_based": runtime_stats,
        "imperative_baseline": baseline_stats,
        "timing_unit": timing_unit,
        "aggregate_stat": str(_cfg_get(cfg, "benchmark", "aggregate_stat", default="median")),
        "num_trials_config": int(_cfg_get(cfg, "benchmark", "num_trials", default=10) or 10),
        "warmup_runs_config": int(_cfg_get(cfg, "benchmark", "warmup_runs", default=0) or 0),
    }

    # ---- 分类指标 -------------------------------------------------------
    has_classification = predictions is not None and labels is not None
    if has_classification:
        cls_metrics = _classification_metrics(predictions, labels, num_classes)
    else:
        cls_metrics = {
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "mean_f1": 0.0,
            "f1_macro": 0.0,
            "per_class": {},
            "confusion_matrix": [],
            "class_labels": [],
            "num_samples": 0,
        }

    # ---- 与基线的动态比较（数值来自实测） -------------------------------
    comparison: Dict[str, Any] = {"baseline": baseline_name}
    if baseline_stats and runtime_stats["count"] > 0 and baseline_stats["count"] > 0:
        rule_median = _safe_float(runtime_stats["median"])
        base_median = _safe_float(baseline_stats["median"])
        rule_mean = _safe_float(runtime_stats["mean"])
        base_mean = _safe_float(baseline_stats["mean"])
        ratio_median = (rule_median / base_median) if base_median > 0 else None
        ratio_mean = (rule_mean / base_mean) if base_mean > 0 else None
        speedup_median = (base_median / rule_median) if rule_median > 0 else None
        comparison.update(
            {
                "rule_based_median": rule_median,
                "baseline_median": base_median,
                "rule_based_mean": rule_mean,
                "baseline_mean": base_mean,
                "runtime_ratio_median": ratio_median,
                "runtime_ratio_mean": ratio_mean,
                "speedup_median": speedup_median,
            }
        )
        if ratio_median is not None:
            comparison["observation"] = (
                "实测 median runtime: rule_based={:.6g}{} vs {}={:.6g}{}，比值={:.4f}。".format(
                    rule_median, timing_unit, baseline_name, base_median, timing_unit, ratio_median
                )
            )
        else:
            comparison["observation"] = "实测数据不足，无法计算运行时比值。"
    else:
        comparison["observation"] = "缺少基线运行时采样，仅报告 rule_based 的 runtime measurements。"

    # ---- 正确性校验 -----------------------------------------------------
    correctness: Optional[Dict[str, Any]] = None
    if verify_correctness and reference_outputs is not None and actual_outputs is not None:
        ref = _to_float_list(reference_outputs)
        act = _to_float_list(actual_outputs)
        size = min(len(ref), len(act))
        matched = sum(1 for i in range(size) if abs(ref[i] - act[i]) <= 1e-9)
        correctness = {
            "checked": size,
            "matched": matched,
            "match_ratio": (matched / size) if size > 0 else 0.0,
            "passed": bool(size > 0 and matched == size),
        }

    complexity = _estimate_complexity(timings_by_size)

    metrics: Dict[str, Any] = {
        "run_name": run_name,
        "timestamp": time.strftime("%Y%m%d_%H%M%S"),
        "timing_unit": timing_unit,
        # 分类指标（绘图与报告统一使用这些键）
        "accuracy": cls_metrics["accuracy"],
        "precision": cls_metrics["precision"],
        "recall": cls_metrics["recall"],
        "mean_f1": cls_metrics["mean_f1"],
        "f1_macro": cls_metrics["f1_macro"],
        "per_class": cls_metrics["per_class"],
        "confusion_matrix": cls_metrics["confusion_matrix"],
        "class_labels": cls_metrics["class_labels"],
        "num_samples": cls_metrics["num_samples"],
        "classification_available": bool(has_classification and cls_metrics["num_samples"] > 0),
        # 运行时指标
        "runtime": runtime_stats,
        "baseline_runtime": baseline_stats,
        "runtime_measurements": runtime_measurements,
        "comparison": comparison,
        "empirical_complexity": complexity,
        "correctness": correctness,
        "config_snapshot": {
            "project.name": _cfg_get(cfg, "project", "name", default=""),
            "project.paper": _cfg_get(cfg, "project", "paper", default=""),
            "experiment.name": _cfg_get(cfg, "experiment", "name", default=""),
            "benchmark.aggregate_stat": _cfg_get(cfg, "benchmark", "aggregate_stat", default="median"),
            "benchmark.num_trials": _cfg_get(cfg, "benchmark", "num_trials", default=None),
            "system.seed": _cfg_get(cfg, "system", "seed", default=None),
        },
    }
    logger.info("evaluate_model 完成: run_name=%s, timing_samples=%d", run_name, runtime_stats["count"])
    return metrics


# ---------------------------------------------------------------------------
# 绘图（matplotlib 可选；缺失时直接返回 None）
# ---------------------------------------------------------------------------


def _generate_metrics_bar_chart(
    metrics: Dict[str, Any],
    output_dir: Optional[str] = None,
    run_name: str = "default",
) -> Optional[str]:
    """绘制 mean_f1 / precision / recall / accuracy 条形图；无 matplotlib 时返回 None。"""
    if plt is None:
        logger.info("matplotlib 不可用，跳过指标条形图。")
        return None
    if not isinstance(metrics, dict):
        logger.warning("_generate_metrics_bar_chart 收到非法 metrics，跳过绘图。")
        return None

    target_dir = output_dir if output_dir else "results"
    try:
        os.makedirs(target_dir, exist_ok=True)
    except OSError as exc:
        logger.warning("无法创建目录 %s (%s)，跳过绘图。", target_dir, exc)
        return None

    metric_keys = ["mean_f1", "precision", "recall", "accuracy"]
    values = [_safe_float(metrics.get(key, 0.0), 0.0) for key in metric_keys]

    try:
        figure, axis = plt.subplots(figsize=(7.0, 4.5))
        bars = axis.bar(metric_keys, values, color=["#4C72B0", "#55A868", "#C44E52", "#8172B2"])
        axis.set_ylim(0.0, 1.05)
        axis.set_ylabel("value")
        axis.set_title("Metrics — {}".format(run_name))
        for bar, value in zip(bars, values):
            axis.annotate(
                "{:.4f}".format(value),
                xy=(bar.get_x() + bar.get_width() / 2.0, bar.get_height()),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9,
            )
        figure.tight_layout()
        path = os.path.join(target_dir, "metrics_bar_{}.png".format(run_name))
        figure.savefig(path, dpi=150)
        plt.close(figure)
        return path
    except Exception as exc:  # noqa: BLE001
        logger.warning("绘制指标条形图失败: %s", exc)
        return None


def _generate_confusion_matrix_plot(
    matrix: Optional[Sequence[Sequence[int]]],
    output_dir: Optional[str] = None,
    run_name: str = "default",
) -> Optional[str]:
    """绘制混淆矩阵热力图；无 matplotlib 或矩阵为空时返回 None。"""
    if plt is None:
        logger.info("matplotlib 不可用，跳过混淆矩阵图。")
        return None
    if not matrix or not isinstance(matrix, (list, tuple)):
        return None
    try:
        rows = [[int(round(_safe_float(v, 0.0))) for v in row] for row in matrix]
    except Exception:  # noqa: BLE001
        logger.warning("混淆矩阵数据非法，跳过绘图。")
        return None
    if not rows or not rows[0]:
        return None

    target_dir = output_dir if output_dir else "results"
    try:
        os.makedirs(target_dir, exist_ok=True)
    except OSError as exc:
        logger.warning("无法创建目录 %s (%s)，跳过绘图。", target_dir, exc)
        return None

    try:
        figure, axis = plt.subplots(figsize=(6.0, 5.0))
        image = axis.imshow(rows, interpolation="nearest", cmap="Blues")
        figure.colorbar(image, ax=axis)
        num_rows = len(rows)
        num_cols = len(rows[0])
        axis.set_xticks(range(num_cols))
        axis.set_yticks(range(num_rows))
        axis.set_xlabel("predicted")
        axis.set_ylabel("true")
        axis.set_title("Confusion Matrix — {}".format(run_name))
        threshold_value = max(max(row) for row in rows) / 2.0 if rows else 0.0
        for i in range(num_rows):
            for j in range(num_cols):
                value = rows[i][j]
                axis.text(
                    j,
                    i,
                    str(value),
                    ha="center",
                    va="center",
                    color="white" if value > threshold_value else "black",
                    fontsize=9,
                )
        figure.tight_layout()
        path = os.path.join(target_dir, "confusion_matrix_{}.png".format(run_name))
        figure.savefig(path, dpi=150)
        plt.close(figure)
        return path
    except Exception as exc:  # noqa: BLE001
        logger.warning("绘制混淆矩阵图失败: %s", exc)
        return None


# ---------------------------------------------------------------------------
# 对外接口 2：generate_report
# ---------------------------------------------------------------------------


def _format_runtime_block(stats: Optional[Dict[str, Any]], title: str) -> List[str]:
    lines: List[str] = ["[{}]".format(title)]
    if not stats:
        lines.append("  (无采样数据)")
        return lines
    unit = stats.get("unit", "seconds")
    lines.append("  count   = {}".format(stats.get("count", 0)))
    lines.append("  mean    = {:.6g} {}".format(_safe_float(stats.get("mean", 0.0)), unit))
    lines.append("  median  = {:.6g} {}".format(_safe_float(stats.get("median", 0.0)), unit))
    lines.append("  std     = {:.6g} {}".format(_safe_float(stats.get("std", 0.0)), unit))
    lines.append("  min     = {:.6g} {}".format(_safe_float(stats.get("min", 0.0)), unit))
    lines.append("  max     = {:.6g} {}".format(_safe_float(stats.get("max", 0.0)), unit))
    quantiles = stats.get("quantiles") or {}
    if isinstance(quantiles, dict):
        for key in sorted(quantiles.keys()):
            lines.append(
                "  {} = {:.6g} {}".format(key, _safe_float(quantiles.get(key, 0.0)), unit)
            )
    return lines


def generate_report(
    metrics: Optional[Dict[str, Any]] = None,
    run_name: str = "default",
    output_dir: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
) -> str:
    """为单个 run 生成评估报告，返回该 run 的报告文件路径。

    产出内容：
      1. 文本报告（无条件生成）：runtime measurements、分类指标、动态比较、经验复杂度。
      2. JSON 导出（当 ``logging.export_format == 'json'``）。
      3. 工程辅助图表（matplotlib 可用时）：指标条形图 + 混淆矩阵图。
    """
    cfg = _resolve_config(config)
    if not isinstance(metrics, dict):
        metrics = {}

    default_out = str(_cfg_get(cfg, "logging", "output_dir", default="results"))
    target_dir = output_dir if output_dir else default_out
    try:
        os.makedirs(target_dir, exist_ok=True)
    except OSError as exc:
        logger.warning("无法创建输出目录 %s (%s)，回退到当前目录。", target_dir, exc)
        target_dir = "."

    use_timestamp = bool(_cfg_get(cfg, "logging", "timestamp_in_filename", default=True))
    suffix = "_{}".format(metrics.get("timestamp", time.strftime("%Y%m%d_%H%M%S"))) if use_timestamp else ""
    safe_name = str(run_name).replace(os.sep, "_").replace(" ", "_")

    # ---------- 文本报告 ----------
    lines: List[str] = []
    lines.append("=" * 78)
    lines.append("Evaluation Report — run: {}".format(safe_name))
    lines.append("=" * 78)
    lines.append("project   : {}".format(_cfg_get(cfg, "project", "name", default="")))
    lines.append("paper     : {}".format(_cfg_get(cfg, "project", "paper", default="")))
    lines.append("experiment: {}".format(_cfg_get(cfg, "experiment", "name", default="")))
    lines.append("generated : {}".format(time.strftime("%Y-%m-%d %H:%M:%S")))
    lines.append("")

    lines.append("-" * 78)
    lines.append("1. Runtime measurements (FR-03 / FR-04)")
    lines.append("-" * 78)
    runtime_measurements = metrics.get("runtime_measurements") or {}
    unit = runtime_measurements.get("timing_unit", _cfg_get(cfg, "benchmark", "timing_unit", default="seconds"))
    lines.append("timing unit          : {}".format(unit))
    lines.append("aggregate statistic  : {}".format(runtime_measurements.get("aggregate_stat", "median")))
    lines.append("num_trials (config)  : {}".format(runtime_measurements.get("num_trials_config", "n/a")))
    lines.append("warmup_runs (config) : {}".format(runtime_measurements.get("warmup_runs_config", "n/a")))
    lines.append("")
    lines.extend(_format_runtime_block(metrics.get("runtime"), "rule_based graph program"))
    lines.append("")
    lines.extend(_format_runtime_block(metrics.get("baseline_runtime"), "imperative baseline"))
    lines.append("")

    lines.append("-" * 78)
    lines.append("2. Comparison with baseline (computed from measured values)")
    lines.append("-" * 78)
    comparison = metrics.get("comparison") or {}
    if comparison.get("runtime_ratio_median") is not None:
        lines.append("baseline name             : {}".format(comparison.get("baseline", "imperative")))
        lines.append("rule_based median         : {:.6g} {}".format(_safe_float(comparison.get("rule_based_median")), unit))
        lines.append("baseline median           : {:.6g} {}".format(_safe_float(comparison.get("baseline_median")), unit))
        lines.append("runtime ratio (median)    : {:.6f}".format(_safe_float(comparison.get("runtime_ratio_median"))))
        if comparison.get("runtime_ratio_mean") is not None:
            lines.append("runtime ratio (mean)      : {:.6f}".format(_safe_float(comparison.get("runtime_ratio_mean"))))
        if comparison.get("speedup_median") is not None:
            lines.append("speedup (median, base/rb) : {:.6f}".format(_safe_float(comparison.get("speedup_median"))))
    lines.append("observation: {}".format(comparison.get("observation", "n/a")))
    lines.append("")

    lines.append("-" * 78)
    lines.append("3. Classification metrics")
    lines.append("-" * 78)
    if metrics.get("classification_available"):
        lines.append("num_samples : {}".format(metrics.get("num_samples", 0)))
        lines.append("accuracy    : {:.6f}".format(_safe_float(metrics.get("accuracy"))))
        lines.append("precision   : {:.6f} (macro)".format(_safe_float(metrics.get("precision"))))
        lines.append("recall      : {:.6f} (macro)".format(_safe_float(metrics.get("recall"))))
        lines.append("mean_f1     : {:.6f}".format(_safe_float(metrics.get("mean_f1"))))
        per_class = metrics.get("per_class") or {}
        for class_key in sorted(per_class.keys(), key=lambda item: (len(item), item)):
            entry = per_class.get(class_key) or {}
            lines.append(
                "  class {:<12s} precision={:.4f} recall={:.4f} f1={:.4f} support={}".format(
                    str(class_key),
                    _safe_float(entry.get("precision")),
                    _safe_float(entry.get("recall")),
                    _safe_float(entry.get("f1")),
                    int(entry.get("support", 0) or 0),
                )
            )
    else:
        lines.append("本次评估未提供 predictions/labels，分类指标不适用。")
    lines.append("")

    complexity = metrics.get("empirical_complexity")
    lines.append("-" * 78)
    lines.append("4. Empirical time-complexity fit (log-log least squares)")
    lines.append("-" * 78)
    if complexity:
        lines.append("slope (log t vs log n) : {:.6f}".format(_safe_float(complexity.get("slope"))))
        lines.append("r^2                    : {:.6f}".format(_safe_float(complexity.get("r2"))))
        lines.append("fitted label           : {}".format(complexity.get("label", "n/a")))
        lines.append("sizes                  : {}".format(complexity.get("sizes", [])))
        lines.append("median times           : {}".format(
            ["{:.6g}".format(_safe_float(v)) for v in complexity.get("median_times", [])]
        ))
    else:
        lines.append("未提供按规模聚合的运行时采样，跳过复杂度拟合。")
    lines.append("")

    correctness = metrics.get("correctness")
    lines.append("-" * 78)
    lines.append("5. Correctness verification")
    lines.append("-" * 78)
    if correctness:
        lines.append(
            "checked={} matched={} match_ratio={:.6f} passed={}".format(
                correctness.get("checked", 0),
                correctness.get("matched", 0),
                _safe_float(correctness.get("match_ratio")),
                correctness.get("passed", False),
            )
        )
    else:
        lines.append("未启用或未提供参考输出，跳过正确性校验。")
    lines.append("")

    report_path = os.path.join(target_dir, "report_{}{}.txt".format(safe_name, suffix))
    report_text = "\n".join(lines) + "\n"
    try:
        with open(report_path, "w", encoding="utf-8") as handle:
            handle.write(report_text)
    except OSError as exc:
        logger.error("写入报告文件失败: %s", exc)
        report_path = os.path.join(".", "report_{}{}.txt".format(safe_name, suffix))
        with open(report_path, "w", encoding="utf-8") as handle:
            handle.write(report_text)

    # ---------- JSON 导出 ----------
    export_format = str(_cfg_get(cfg, "logging", "export_format", default="json")).lower()
    if export_format == "json":
        json_path = os.path.join(target_dir, "metrics_{}{}.json".format(safe_name, suffix))
        serializable = dict(metrics)
        try:
            with open(json_path, "w", encoding="utf-8") as handle:
                json.dump(serializable, handle, ensure_ascii=False, indent=2, default=str)
        except (OSError, TypeError) as exc:
            logger.warning("导出 JSON 失败: %s", exc)

    # ---------- 图表（可选） ----------
    bar_chart_path = _generate_metrics_bar_chart(metrics, target_dir, safe_name)
    if bar_chart_path:
        logger.info("指标条形图已生成: %s", bar_chart_path)
    matrix = metrics.get("confusion_matrix") if isinstance(metrics, dict) else None
    matrix_plot_path = _generate_confusion_matrix_plot(matrix, target_dir, safe_name)
    if matrix_plot_path:
        logger.info("混淆矩阵图已生成: %s", matrix_plot_path)

    logger.info("报告已生成: %s", report_path)
    return report_path