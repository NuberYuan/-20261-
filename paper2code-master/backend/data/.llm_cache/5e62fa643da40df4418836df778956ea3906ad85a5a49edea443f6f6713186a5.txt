"""Evaluation and report generation for the time-series prediction task.

Implements model evaluation on test sets and generates paper-style reports
with metrics: Success rate, Test accuracy, Median training time.

The module is designed to work both with and without matplotlib.  Metric
computation and text report generation are unconditional; plots are skipped
gracefully when matplotlib is unavailable.
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import yaml

from models.lstm import Lstm
from training.train import _build_loss_function, _prepare_batch, _run_epoch
from utils.metrics import compute_metrics

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover - depends on environment
    plt = None

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------
def _load_config(config: Optional[Any]) -> Dict[str, Any]:
    """Load config from a path, dict, or return empty dict if None."""
    if config is None:
        return {}
    if isinstance(config, dict):
        return config
    if isinstance(config, str):
        with open(config, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    raise TypeError("config must be None, a dict, or a path string.")


def _safe_get(config: Dict[str, Any], *keys: str, default: Any = None) -> Any:
    """Safely traverse nested dict keys."""
    current: Any = config
    for key in keys:
        if not isinstance(current, dict):
            return default
        if key not in current:
            return default
        current = current[key]
    return current


def _resolve_device(config: Dict[str, Any]) -> torch.device:
    """Resolve the device from config or fall back to available hardware."""
    device_str = _safe_get(config, "system", "device", default=None)
    if device_str is not None and str(device_str).strip() != "":
        return torch.device(str(device_str))
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def evaluate_model(
    model: Optional[Lstm] = None,
    test_loader: Optional[Any] = None,
    config: Optional[Any] = None,
    device: Optional[torch.device] = None,
    training_times: Optional[List[float]] = None,
    checkpoint_path: Optional[str] = None,
    threshold: float = 0.5,
    batch_size: int = 32,
) -> Dict[str, float]:
    """Evaluate a trained model on the test set.

    Parameters
    ----------
    model : Lstm, optional
        Trained model to evaluate.  If None and ``checkpoint_path`` is
        provided, the model is loaded from the checkpoint.
    test_loader : DataLoader, optional
        Test data loader.  If None, it is built from config.
    config : dict or str, optional
        Config dict or path to config.yaml.
    device : torch.device, optional
        Device to run evaluation on.  Overrides config.
    training_times : list of float, optional
        Training times in seconds for each run.  Used to compute median
        training time.
    checkpoint_path : str, optional
        Path to a saved model checkpoint.  Used when ``model`` is None.
    threshold : float
        Threshold for binary classification metrics (kept for API
        compatibility; not used for symbol-prediction accuracy).
    batch_size : int
        Batch size used during evaluation (kept for API compatibility).

    Returns
    -------
    Dict[str, float]
        Dictionary with keys ``success_rate``, ``test_accuracy``, and
        ``median_training_time``.
    """
    cfg = _load_config(config)

    if device is None:
        device = _resolve_device(cfg)

    # Load model from checkpoint if not provided.
    if model is None:
        if checkpoint_path is None:
            checkpoint_path = _safe_get(
                cfg, "checkpoint", "resume_from", default=None
            )
        if checkpoint_path is None or not os.path.exists(checkpoint_path):
            raise ValueError(
                "evaluate_model requires either a model instance or a valid "
                "checkpoint_path."
            )
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model = Lstm(config=cfg)
        model.load_state_dict(checkpoint["model_state_dict"])
        logger.info("Loaded model from checkpoint %s", checkpoint_path)

    model = model.to(device)
    model.eval()

    # Build test loader if not provided.
    if test_loader is None:
        from data.dataset import get_dataloaders

        dataset_version = str(
            _safe_get(
                cfg,
                "experiment",
                "runs",
                0,
                "dataset_version",
                default="default",
            )
        )
        loaders = get_dataloaders(
            config=cfg,
            version=dataset_version,
            batch_size=int(
                _safe_get(cfg, "training", "batch_size", "default", default=20)
            ),
            num_workers=int(_safe_get(cfg, "system", "num_workers", default=0)),
            seed=int(_safe_get(cfg, "system", "seed", default=42)),
        )
        test_loader = loaders.get("test")
        if test_loader is None:
            raise ValueError("No test loader available from config.")

    # Collect predictions and targets.
    all_predictions: List[int] = []
    all_targets: List[int] = []
    loss_fn = _build_loss_function(cfg)

    with torch.no_grad():
        for batch in test_loader:
            x, target_ids, _ = _prepare_batch(batch, device, model)
            logits, _ = model(x)
            predictions = logits.argmax(dim=-1)

            # Flatten and filter padding positions.
            mask = target_ids != 0  # PAD_IDX is 0 in data.dataset
            pred_flat = predictions[mask].cpu().numpy().tolist()
            tgt_flat = target_ids[mask].cpu().numpy().tolist()
            all_predictions.extend(pred_flat)
            all_targets.extend(tgt_flat)

    if len(all_predictions) == 0:
        logger.warning("No valid predictions collected; returning zero metrics.")
        return {
            "success_rate": 0.0,
            "test_accuracy": 0.0,
            "median_training_time": float(
                np.median(training_times) if training_times else 0.0
            ),
        }

    # Compute metrics using utils.metrics.compute_metrics.
    metrics = compute_metrics(
        predictions=np.asarray(all_predictions),
        targets=np.asarray(all_targets),
        training_times=training_times,
        config=cfg,
        threshold=threshold,
        batch_size=batch_size,
    )

    logger.info(
        "Evaluation metrics - success_rate: %.4f, test_accuracy: %.4f, "
        "median_training_time: %.4f",
        metrics["success_rate"],
        metrics["test_accuracy"],
        metrics["median_training_time"],
    )

    return metrics


# ---------------------------------------------------------------------------
# Plotting helpers (matplotlib optional)
# ---------------------------------------------------------------------------
def _generate_metrics_bar_chart(
    metrics: Dict[str, Any],
    output_path: str,
    run_name: str = "run",
) -> Optional[str]:
    """Generate a bar chart of key metrics.

    Parameters
    ----------
    metrics : dict
        Full metrics dictionary.  Must contain at least one of the keys
        ``mean_f1``, ``precision``, ``recall``, ``accuracy``.
    output_path : str
        Path where the plot should be saved.
    run_name : str
        Name of the run for the plot title.

    Returns
    -------
    Optional[str]
        Path to the saved plot, or None if matplotlib is unavailable or no
        plottable metrics are present.
    """
    if plt is None:
        logger.info("matplotlib not installed; skipping metrics bar chart.")
        return None

    plottable_keys = ["mean_f1", "precision", "recall", "accuracy"]
    available = {k: metrics.get(k) for k in plottable_keys if metrics.get(k) is not None}
    if not available:
        logger.info("No plottable metrics found; skipping bar chart.")
        return None

    labels = list(available.keys())
    values = [float(available[k]) for k in labels]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, values, color=["#4C72B0", "#55A868", "#C44E52", "#8172B2"])
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Score")
    ax.set_title(f"Metrics for {run_name}")
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + 0.02,
            f"{val:.4f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info("Saved metrics bar chart to %s", output_path)
    return output_path


def _generate_confusion_matrix_plot(
    confusion_matrix: Optional[np.ndarray],
    output_path: str,
    run_name: str = "run",
) -> Optional[str]:
    """Generate a confusion matrix heatmap.

    Parameters
    ----------
    confusion_matrix : np.ndarray, optional
        Square confusion matrix.  If None, the plot is skipped.
    output_path : str
        Path where the plot should be saved.
    run_name : str
        Name of the run for the plot title.

    Returns
    -------
    Optional[str]
        Path to the saved plot, or None if matplotlib is unavailable or the
        matrix is None.
    """
    if plt is None:
        logger.info("matplotlib not installed; skipping confusion matrix plot.")
        return None
    if confusion_matrix is None:
        logger.info("No confusion matrix provided; skipping plot.")
        return None

    cm = np.asarray(confusion_matrix)
    if cm.ndim != 2 or cm.shape[0] != cm.shape[1]:
        logger.warning("Confusion matrix must be square 2D; skipping plot.")
        return None

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.set_title(f"Confusion Matrix for {run_name}")
    fig.colorbar(im, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info("Saved confusion matrix plot to %s", output_path)
    return output_path


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------
def generate_report(
    metrics: Dict[str, Any],
    run_name: str = "time_series_prediction_default",
    output_dir: str = "reports",
    config: Optional[Any] = None,
) -> str:
    """Generate a text report and optional plots for a single run.

    Parameters
    ----------
    metrics : dict
        Metrics dictionary returned by :func:`evaluate_model`.  Must contain
        at least ``success_rate``, ``test_accuracy``, and
        ``median_training_time``.  May optionally contain ``mean_f1``,
        ``precision``, ``recall``, ``accuracy``, and ``confusion_matrix``
        for additional plots.
    run_name : str
        Name of the run, used in report filenames and titles.
    output_dir : str
        Directory where the report and plots are saved.
    config : dict or str, optional
        Config dict or path to config.yaml.  Used for metadata in the report.

    Returns
    -------
    str
        Path to the generated text report file.
    """
    cfg = _load_config(config)

    os.makedirs(output_dir, exist_ok=True)

    # Extract core metrics with safe defaults.
    success_rate = float(metrics.get("success_rate", 0.0))
    test_accuracy = float(metrics.get("test_accuracy", 0.0))
    median_training_time = float(metrics.get("median_training_time", 0.0))

    # Build report text.
    report_lines: List[str] = []
    report_lines.append("=" * 60)
    report_lines.append("TIME SERIES PREDICTION - EVALUATION REPORT")
    report_lines.append("=" * 60)
    report_lines.append(f"Run name: {run_name}")
    report_lines.append(f"Generated at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("")

    # Paper metadata.
    paper_name = _safe_get(cfg, "project", "paper", default="LEARNING SIMPLE ALGORITHMS FROM EXAMPLES")
    framework = _safe_get(cfg, "project", "framework", default="PyTorch")
    report_lines.append(f"Paper: {paper_name}")
    report_lines.append(f"Framework: {framework}")
    report_lines.append("")

    # Core metrics.
    report_lines.append("-" * 60)
    report_lines.append("CORE METRICS")
    report_lines.append("-" * 60)
    report_lines.append(f"Success rate:          {success_rate:.4f}")
    report_lines.append(f"Test accuracy:         {test_accuracy:.4f}")
    report_lines.append(f"Median training time:  {median_training_time:.4f} s")
    report_lines.append("")

    # Optional additional metrics.
    extra_keys = ["mean_f1", "precision", "recall", "accuracy"]
    extra_present = [k for k in extra_keys if metrics.get(k) is not None]
    if extra_present:
        report_lines.append("-" * 60)
        report_lines.append("ADDITIONAL METRICS")
        report_lines.append("-" * 60)
        for key in extra_present:
            report_lines.append(f"{key}: {float(metrics[key]):.4f}")
        report_lines.append("")

    # Note about comparison to paper values.
    report_lines.append("-" * 60)
    report_lines.append("NOTES")
    report_lines.append("-" * 60)
    report_lines.append(
        "Compare the above values with the paper's reported results. "
        "The paper reports success on all six tasks at test length 100 "
        "when using all three Q-learning enhancements.  At test length "
        "1000, generalization holds for all tasks except multiplication."
    )
    report_lines.append("")
    report_lines.append("=" * 60)

    report_text = "\n".join(report_lines)

    # Write text report.
    report_path = os.path.join(output_dir, f"{run_name}_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)
    logger.info("Saved evaluation report to %s", report_path)

    # Generate optional plots (matplotlib may be unavailable).
    if plt is not None:
        # Metrics bar chart (only if additional metrics are present).
        if any(metrics.get(k) is not None for k in ["mean_f1", "precision", "recall", "accuracy"]):
            bar_chart_path = os.path.join(output_dir, f"{run_name}_metrics_bar.png")
            _generate_metrics_bar_chart(metrics, bar_chart_path, run_name=run_name)

        # Confusion matrix plot (only if confusion_matrix is present).
        cm = metrics.get("confusion_matrix")
        if cm is not None:
            cm_path = os.path.join(output_dir, f"{run_name}_confusion_matrix.png")
            _generate_confusion_matrix_plot(cm, cm_path, run_name=run_name)
    else:
        logger.info("matplotlib not installed; skipping all plots.")

    return report_path