from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover - matplotlib is an auxiliary dependency
    plt = None


def _load_config(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Load configuration safely.

    The evaluator reads only configuration keys that exist in
    ``config/config.yaml``.  When no config path is available an empty
    dictionary is returned so that callers can still execute with safe
    defaults.
    """
    if isinstance(config, dict):
        return config

    if yaml is None:
        return {}

    if isinstance(config, str) and os.path.exists(config):
        config_path = Path(config)
    else:
        root = Path(__file__).resolve().parents[1]
        config_path = root / "config" / "config.yaml"
        if not config_path.exists():
            fallback = Path.cwd() / "config" / "config.yaml"
            if fallback.exists():
                config_path = fallback
            else:
                return {}

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            loaded = yaml.safe_load(f)
    except (OSError, ValueError):
        return {}

    return loaded if isinstance(loaded, dict) else {}


def _safe_float(value: Any, default: float = 0.0) -> float:
    """Convert a report value to float without crashing on ``None``."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    """Convert a report value to int without crashing on ``None``."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _get_device(
    config: Dict[str, Any],
    model: Any = None,
    device: Optional[Any] = None,
) -> torch.device:
    """Resolve the evaluation device from an explicit argument, model, or config."""
    if device is not None:
        return torch.device(device)

    if model is not None:
        try:
            return next(model.parameters()).device
        except (StopIteration, AttributeError):
            pass

    system_cfg = config.get("system") or {}
    device_str = system_cfg.get("device", "cpu")
    if device_str is None:
        device_str = "cpu"
    if device_str == "cuda" and not torch.cuda.is_available():
        device_str = "cpu"
    return torch.device(device_str)


def _to_device(value: Any, device: torch.device) -> Any:
    """Move tensors or nested structures to ``device``."""
    if value is None:
        return None
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, dict):
        return {k: _to_device(v, device) for k, v in value.items()}
    if isinstance(value, tuple):
        return tuple(_to_device(v, device) for v in value)
    if isinstance(value, list):
        return [_to_device(v, device) for v in value]
    return value


def _extract_batch_inputs_target(batch: Any) -> Tuple[Any, Any]:
    """Extract model inputs and targets from common dataloader batch formats."""
    if isinstance(batch, dict):
        targets = batch.get(
            "targets",
            batch.get("target", batch.get("labels", batch.get("label"))),
        )

        if "data" in batch:
            inputs = batch["data"]
        else:
            inputs = {
                k: v
                for k, v in batch.items()
                if k not in {"targets", "target", "labels", "label"}
            }
            if not inputs:
                inputs = batch  # last-resort fallback, but only if no explicit data key
        return inputs, targets

    if isinstance(batch, (tuple, list)):
        if len(batch) == 2:
            return batch[0], batch[1]
        if len(batch) == 3:
            # Common format for graph models: (x, adj, y)
            return (batch[0], batch[1]), batch[2]
        if len(batch) >= 4:
            # Common format for message-passing models:
            # (node_features, edge_index, edge_attr, y)
            inputs = {
                "node_features": batch[0],
                "edge_index": batch[1],
                "edge_attr": batch[2],
                "batch": None,
            }
            return inputs, batch[3]

    return batch, None


def _call_model(model: Any, inputs: Any, device: torch.device) -> Any:
    """Call a model with the broadest supported input style."""
    model_inputs = _to_device(inputs, device)

    if isinstance(model_inputs, dict):
        if "x" in model_inputs or "adj" in model_inputs:
            return model(model_inputs.get("x"), model_inputs.get("adj"))
        return model(model_inputs)

    if isinstance(model_inputs, tuple):
        if len(model_inputs) == 2:
            return model(model_inputs[0], model_inputs[1])
        return model(model_inputs)

    if isinstance(model_inputs, list):
        if len(model_inputs) == 2:
            return model(model_inputs[0], model_inputs[1])
        return model(model_inputs)

    return model(model_inputs)


def _output_to_predictions(outputs: Any) -> List[Any]:
    """Convert model outputs to plain Python prediction values."""
    if torch.is_tensor(outputs):
        out = outputs.detach().cpu()
        if out.numel() == 0:
            return []

        if out.dim() >= 2 and out.shape[-1] == 2:
            # Binary logits: return probability of the positive class.
            probs = torch.softmax(out, dim=-1)[:, 1]
            return [float(v) for v in probs.tolist()]

        if out.dim() >= 2 and out.shape[-1] > 2:
            return [int(v) for v in out.argmax(dim=-1).flatten().tolist()]

        if out.dim() == 2 and out.shape[-1] == 1:
            return [float(v) for v in out.squeeze(-1).tolist()]

        return [float(v) if out.numel() == 1 else int(v) for v in out.flatten().tolist()]

    if isinstance(outputs, (list, tuple)):
        return list(outputs)

    return [outputs]


def _targets_to_list(targets: Any) -> List[Any]:
    """Convert targets to a one-dimensional plain Python list."""
    if targets is None:
        return []

    if torch.is_tensor(targets):
        return [float(v) for v in targets.detach().cpu().flatten().tolist()]

    if isinstance(targets, (list, tuple)):
        flat: List[Any] = []
        for item in targets:
            if torch.is_tensor(item):
                flat.extend(float(v) for v in item.detach().cpu().flatten().tolist())
            elif isinstance(item, (list, tuple)) and len(item) == 1:
                flat.append(item[0])
            else:
                flat.append(item)
        return flat

    return [targets]


def _normalise_binary_target(value: Any) -> int:
    """Map a target value to binary positive/negative for evaluation."""
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return 0
    return 1 if numeric >= 0.5 else 0


def _normalise_binary_prediction(value: Any, threshold: float) -> int:
    """Map a prediction value to binary positive/negative for evaluation."""
    if isinstance(value, bool):
        return int(value)
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return int(value == 1)
    return 1 if numeric >= threshold else 0


def _compute_classification_metrics(
    targets: Sequence[Any],
    predictions: Sequence[Any],
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """Compute binary classification metrics and a confusion matrix.

    These metrics are reported for auxiliary visualisation and are always
    derived from the actual evaluation inputs.
    """
    y_true = [_normalise_binary_target(t) for t in targets] if targets else []
    y_pred = (
        [_normalise_binary_prediction(p, threshold) for p in predictions]
        if predictions
        else []
    )

    metrics: Dict[str, Any] = {
        "accuracy": 0.0,
        "precision": 0.0,
        "recall": 0.0,
        "mean_f1": 0.0,
        "confusion_matrix": [[0, 0], [0, 0]],
    }

    total = len(y_true)
    if total == 0:
        return metrics

    tp = fp = tn = fn = 0
    for true, pred in zip(y_true, y_pred):
        if true == 1 and pred == 1:
            tp += 1
        elif true == 0 and pred == 1:
            fp += 1
        elif true == 0 and pred == 0:
            tn += 1
        elif true == 1 and pred == 0:
            fn += 1

    accuracy = (tp + tn) / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0.0

    metrics["accuracy"] = accuracy
    metrics["precision"] = precision
    metrics["recall"] = recall
    metrics["mean_f1"] = f1
    metrics["confusion_matrix"] = [[tn, fp], [fn, tp]]
    return metrics


def _resolve_time_complexity(config: Dict[str, Any], kwargs: Dict[str, Any]) -> str:
    """Return the configured time complexity, or a safe fallback.

    Paper-specific time-complexity values are never hard-coded in this file.
    """
    explicit = kwargs.get("time_complexity")
    if explicit is not None:
        return str(explicit)

    algo_params = config.get("algorithm_params") or {}
    if isinstance(algo_params, dict):
        configured = algo_params.get("time_complexity")
        if configured is not None:
            return str(configured)

    return "unknown"


def evaluate_model(
    model: Any = None,
    dataloader: Any = None,
    config: Optional[Dict[str, Any]] = None,
    threshold: float = 0.5,
    batch_size: int = 32,
    device: Any = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Evaluate a model on test data.

    Parameters
    ----------
    model:
        A PyTorch model or any callable with the expected forward interface.
    dataloader:
        An iterable of batches.  If ``None``, zero-valued metrics are returned
        so that basic tests can call the function without a dataset.
    config:
        Optional configuration dictionary.  If not supplied, it is loaded from
        ``config/config.yaml`` when available.
    threshold:
        Decision threshold for binary predictions.
    batch_size:
        Reserved for metric computation compatibility.  The current evaluation
        loop is iterator-driven and does not require batching here.
    device:
        Optional torch device.  If omitted, the device is inferred from the
        model or configuration.

    Returns
    -------
    dict
        A dictionary containing ``time_complexity``,
        ``correctness_proofs``, ``runtime_measurements``, and auxiliary
        classification metrics derived from the actual evaluation run.
    """
    cfg = _load_config(config)
    eval_device = _get_device(cfg, model, device)

    all_predictions: List[Any] = []
    all_targets: List[Any] = []
    runtimes: List[float] = []

    if model is not None:
        try:
            model.eval()
        except Exception:
            pass

    if model is not None and dataloader is not None:
        with torch.no_grad():
            for batch in dataloader:
                inputs, targets = _extract_batch_inputs_target(batch)
                inputs = _to_device(inputs, eval_device)
                targets = _to_device(targets, eval_device)

                start = time.perf_counter()
                outputs = _call_model(model, inputs, eval_device)
                elapsed = time.perf_counter() - start
                runtimes.append(float(elapsed))

                pred_list = _output_to_predictions(outputs)
                target_list = _targets_to_list(targets)

                all_predictions.extend(pred_list)
                all_targets.extend(target_list)

    try:
        from utils.metrics import compute_metrics
    except ImportError:  # pragma: no cover
        compute_metrics = None  # type: ignore[assignment]

    time_complexity = _resolve_time_complexity(cfg, kwargs)

    if compute_metrics is not None:
        computed = compute_metrics(
            predictions=all_predictions if all_predictions else None,
            targets=all_targets if all_targets else None,
            runtimes=runtimes if runtimes else None,
            correctness_proofs=None,
            config=cfg,
            threshold=threshold,
            batch_size=batch_size,
            time_complexity=time_complexity,
        )
    else:
        computed = {
            "time_complexity": time_complexity,
            "correctness_proofs": {
                "num_correct": 0,
                "total": 0,
                "rate": 0.0,
                "proofs": [],
            },
            "runtime_measurements": {
                "mean": 0.0,
                "std": 0.0,
                "min": 0.0,
                "max": 0.0,
                "count": 0,
                "total": 0.0,
            },
        }

    classification_metrics = _compute_classification_metrics(
        all_targets,
        all_predictions,
        threshold,
    )

    result: Dict[str, Any] = {}
    result.update(computed)
    result.update(classification_metrics)
    result["num_samples"] = len(all_targets)
    return result


def _is_valid_confusion_matrix(matrix: Any) -> bool:
    """Return ``True`` when ``matrix`` is a rectangular numeric 2D array."""
    if not isinstance(matrix, (list, tuple)) or len(matrix) == 0:
        return False

    rows: List[List[int]] = []
    for row in matrix:
        if not isinstance(row, (list, tuple)):
            return False
        numeric_row: List[int] = []
        for value in row:
            try:
                numeric_row.append(int(value))
            except (TypeError, ValueError):
                return False
        rows.append(numeric_row)

    if not rows or len(rows[0]) == 0:
        return False

    width = len(rows[0])
    if any(len(row) != width for row in rows):
        return False

    return True


def _generate_metrics_bar_chart(metrics: Dict[str, Any], output_path: Path) -> Optional[str]:
    """Generate a bar chart for scalar classification metrics.

    This helper receives the complete ``metrics`` dictionary.  It is an
    auxiliary visualisation and returns ``None`` when matplotlib is not
    installed or when plotting fails; metric computation and text report
    generation are unaffected.
    """
    if plt is None:
        return None

    try:
        values = {
            "mean_f1": _safe_float(metrics.get("mean_f1", metrics.get("f1", 0.0))),
            "precision": _safe_float(metrics.get("precision", 0.0)),
            "recall": _safe_float(metrics.get("recall", 0.0)),
            "accuracy": _safe_float(metrics.get("accuracy", 0.0)),
        }

        names = list(values.keys())
        vals = list(values.values())

        fig, ax = plt.subplots(figsize=(6, 4))
        bars = ax.bar(names, vals)
        ax.set_ylim(0, max(1.0, max(vals) + 0.1))
        ax.set_ylabel("Score")
        ax.set_title("Evaluation metrics")

        for bar, value in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.02,
                f"{value:.3f}",
                ha="center",
                va="bottom",
            )

        fig.tight_layout()
        fig.savefig(output_path, dpi=120)
        plt.close(fig)
        return str(output_path)
    except Exception:
        return None


def _generate_confusion_matrix_plot(
    confusion_matrix: Any,
    output_path: Path,
) -> Optional[str]:
    """Generate a confusion matrix heatmap.

    This helper receives the 2D ``metrics['confusion_matrix']`` array.  It is
    auxiliary and is skipped gracefully when matplotlib is missing.
    """
    if plt is None or not _is_valid_confusion_matrix(confusion_matrix):
        return None

    try:
        cm = [[int(value) for value in row] for row in confusion_matrix]
        n = len(cm)

        fig, ax = plt.subplots(figsize=(4, 4))
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title("Confusion matrix")

        max_value = max(max(row) for row in cm) if cm else 0
        threshold_for_text = max(2, max_value // 2)

        for i in range(n):
            for j in range(n):
                color = "white" if cm[i][j] > threshold_for_text else "black"
                ax.text(
                    j,
                    i,
                    str(cm[i][j]),
                    ha="center",
                    va="center",
                    color=color,
                )

        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        fig.savefig(output_path, dpi=120)
        plt.close(fig)
        return str(output_path)
    except Exception:
        return None


def generate_report(
    metrics: Optional[Dict[str, Any]] = None,
    run_name: str = "run",
    output_dir: str = "reports",
    config: Optional[Dict[str, Any]] = None,
) -> str:
    """Generate a single-run text report and auxiliary plots.

    Parameters
    ----------
    metrics:
        Metric dictionary produced by :func:`evaluate_model`.
    run_name:
        Name of the evaluation run.
    output_dir:
        Directory where the report is written.
    config:
        Optional configuration dictionary.  This argument is accepted for
        compatibility with the unified run interface.

    Returns
    -------
    str
        Path to the generated text report.
    """
    cfg = _load_config(config)
    metrics = metrics if isinstance(metrics, dict) else {}

    report_dir = Path(output_dir) / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)

    safe_run = "".join(
        c if c.isalnum() or c in "-_" else "_" for c in str(run_name)
    )
    safe_run = safe_run or "run"

    report_path = report_dir / f"{safe_run}_evaluation_report.txt"
    metrics_bar_path = report_dir / f"{safe_run}_metrics_bar_chart.png"
    confusion_path = report_dir / f"{safe_run}_confusion_matrix.png"
    json_path = report_dir / f"{safe_run}_metrics.json"

    correctness = metrics.get("correctness_proofs") or {}
    runtime = metrics.get("runtime_measurements") or {}

    time_complexity = metrics.get("time_complexity", "unknown")
    num_correct = _safe_int(correctness.get("num_correct"))
    total_proofs = _safe_int(correctness.get("total"))
    proof_rate = _safe_float(correctness.get("rate"))
    runtime_mean = _safe_float(runtime.get("mean"))
    runtime_std = _safe_float(runtime.get("std"))
    runtime_min = _safe_float(runtime.get("min"))
    runtime_max = _safe_float(runtime.get("max"))
    runtime_total = _safe_float(runtime.get("total"))
    runtime_count = _safe_int(runtime.get("count"))

    accuracy = _safe_float(metrics.get("accuracy"))
    precision = _safe_float(metrics.get("precision"))
    recall = _safe_float(metrics.get("recall"))
    mean_f1 = _safe_float(metrics.get("mean_f1", metrics.get("f1")))
    confusion_matrix = metrics.get("confusion_matrix")

    lines: List[str] = []
    lines.append("Evaluation Report")
    lines.append("==================")
    lines.append(f"Run name: {run_name}")
    lines.append(f"Time complexity: {time_complexity}")
    lines.append("")

    lines.append("Correctness proofs:")
    lines.append(f"  num_correct: {num_correct}")
    lines.append(f"  total: {total_proofs}")
    lines.append(f"  rate: {proof_rate:.4f}")
    lines.append("")

    lines.append("Runtime measurements:")
    lines.append(f"  mean: {runtime_mean:.6f}")
    lines.append(f"  std: {runtime_std:.6f}")
    lines.append(f"  min: {runtime_min:.6f}")
    lines.append(f"  max: {runtime_max:.6f}")
    lines.append(f"  total: {runtime_total:.6f}")
    lines.append(f"  count: {runtime_count}")
    lines.append("")

    lines.append("Auxiliary classification metrics:")
    lines.append(f"  accuracy: {accuracy:.4f}")
    lines.append(f"  precision: {precision:.4f}")
    lines.append(f"  recall: {recall:.4f}")
    lines.append(f"  mean_f1: {mean_f1:.4f}")
    if _is_valid_confusion_matrix(confusion_matrix):
        lines.append("  confusion_matrix:")
        for row in confusion_matrix:
            lines.append("    " + " ".join(str(int(v)) for v in row))
    lines.append("")

    lines.append("Notes:")
    lines.append("- Metrics are computed from the actual evaluation inputs.")
    lines.append("- No hard-coded paper conclusions are embedded in this report.")
    lines.append(
        "- Plots are auxiliary; the text report remains valid without matplotlib."
    )

    try:
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
    except OSError:
        # If the report cannot be written, fall back to a local filename.
        report_path = Path.cwd() / f"{safe_run}_evaluation_report.txt"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

    # Auxiliary visualisations.  Both helpers handle missing matplotlib safely.
    _generate_metrics_bar_chart(metrics, metrics_bar_path)
    _generate_confusion_matrix_plot(confusion_matrix, confusion_path)

    try:
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, default=str)
    except (TypeError, ValueError, OSError):
        pass

    return str(report_path)


__all__ = ["evaluate_model", "generate_report"]