import os
import sys
import time
import logging
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import yaml

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover - matplotlib is optional
    plt = None

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from training.train import BCEWithDiceLoss
from utils.metrics import compute_metrics

logger = logging.getLogger(__name__)


def _load_config(config: Optional[Any] = None) -> Dict[str, Any]:
    """Load configuration from a path, a dict, or the default config.yaml."""
    if config is None:
        config_path = os.path.join(PROJECT_ROOT, "config", "config.yaml")
        if os.path.isfile(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        return {}

    if isinstance(config, dict):
        return config

    if isinstance(config, str):
        if os.path.isfile(config):
            with open(config, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        return {}

    config_path = os.path.join(PROJECT_ROOT, "config", "config.yaml")
    if os.path.isfile(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


def _resolve_device(config: Optional[Any] = None, device: Optional[str] = None) -> str:
    """Resolve the target device from config or explicit argument."""
    if device is not None:
        return device

    cfg = _load_config(config)
    requested_device = cfg.get("system", {}).get("device", "cuda:0")
    if (
        requested_device
        and "cuda" in str(requested_device).lower()
        and not torch.cuda.is_available()
    ):
        print("Warning: CUDA device requested but not available; falling back to CPU.")
        return "cpu"
    return requested_device or "cpu"


def evaluate_model(
    model: Optional[nn.Module] = None,
    dataloader: Optional[torch.utils.data.DataLoader] = None,
    criterion: Optional[nn.Module] = None,
    device: Optional[str] = None,
    config: Optional[Any] = None,
    threshold: float = 0.5,
    batch_size: int = 32,
) -> Dict[str, Any]:
    """Evaluate a model on a dataloader and return segmentation metrics.

    The returned metrics include loss, pixel accuracy, Dice, IoU,
    precision/recall/F1 for the foreground class, macro-F1 over both classes,
    and a 2x2 confusion matrix arranged as [[TN, FP], [FN, TP]].
    """

    cfg = _load_config(config)
    target_device = _resolve_device(cfg, device)

    if model is None or dataloader is None:
        logger.warning("evaluate_model: model and dataloader are required; returning zero metrics.")
        return {
            "loss": 0.0,
            "accuracy": 0.0,
            "val_accuracy": 0.0,
            "dice_coefficient": 0.0,
            "iou": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "mean_f1": 0.0,
            "confusion_matrix": [[0, 0], [0, 0]],
        }

    model.to(target_device)
    model.eval()

    if criterion is None:
        criterion = BCEWithDiceLoss()

    total_loss = 0.0
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    total_pixels = 0
    dice_scores = []
    iou_scores = []

    with torch.no_grad():
        for images, masks in dataloader:
            images = images.to(target_device)
            masks = masks.to(target_device)

            outputs = model(images)
            loss = criterion(outputs, masks)
            total_loss += loss.item()

            probs = outputs.float()
            masks = masks.float()

            batch_metrics = compute_metrics(
                preds=probs,
                targets=masks,
                threshold=threshold,
                batch_size=batch_size,
                config=cfg,
            )
            dice_scores.append(float(batch_metrics.get("dice_coefficient", 0.0)))
            iou_scores.append(float(batch_metrics.get("iou", 0.0)))

            if probs.dim() == 4 and probs.size(1) == 1:
                probs = probs.squeeze(1)
            if masks.dim() == 4 and masks.size(1) == 1:
                masks = masks.squeeze(1)

            preds = (probs > threshold).float()
            preds_flat = preds.reshape(-1).long()
            masks_flat = masks.reshape(-1).long()

            tp += int(((preds_flat == 1) & (masks_flat == 1)).sum().item())
            tn += int(((preds_flat == 0) & (masks_flat == 0)).sum().item())
            fp += int(((preds_flat == 1) & (masks_flat == 0)).sum().item())
            fn += int(((preds_flat == 0) & (masks_flat == 1)).sum().item())
            total_pixels += masks_flat.numel()

    num_batches = max(len(dataloader), 1)
    avg_loss = total_loss / num_batches
    accuracy = (tp + tn) / float(total_pixels) if total_pixels > 0 else 0.0

    precision = tp / float(tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / float(tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    precision_neg = tn / float(tn + fn) if (tn + fn) > 0 else 0.0
    recall_neg = tn / float(tn + fp) if (tn + fp) > 0 else 0.0
    f1_neg = (
        2.0 * precision_neg * recall_neg / (precision_neg + recall_neg)
        if (precision_neg + recall_neg) > 0
        else 0.0
    )
    mean_f1 = (f1 + f1_neg) / 2.0

    dice = 2.0 * tp / float(2.0 * tp + fp + fn) if (2.0 * tp + fp + fn) > 0 else 0.0
    iou = tp / float(tp + fp + fn) if (tp + fp + fn) > 0 else 0.0

    avg_dice = float(np.mean(dice_scores)) if dice_scores else float(dice)
    avg_iou = float(np.mean(iou_scores)) if iou_scores else float(iou)

    confusion_matrix = [[int(tn), int(fp)], [int(fn), int(tp)]]

    return {
        "loss": float(avg_loss),
        "accuracy": float(accuracy),
        "val_accuracy": float(accuracy),
        "dice_coefficient": float(avg_dice),
        "iou": float(avg_iou),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "mean_f1": float(mean_f1),
        "confusion_matrix": confusion_matrix,
    }


def _generate_metrics_bar_chart(metrics: Dict[str, Any], run_name: str, output_dir: str) -> Optional[str]:
    """Create a bar chart for mean_f1, precision, recall, and accuracy.

    This function always receives the complete metrics dictionary so all four
    values can be plotted together.
    """
    if plt is None:
        return None
    if not isinstance(metrics, dict):
        return None

    metric_names = ["mean_f1", "precision", "recall", "accuracy"]
    values = []
    for name in metric_names:
        value = metrics.get(name, 0.0)
        if isinstance(value, (int, float)):
            values.append(float(value))
        else:
            values.append(0.0)

    os.makedirs(output_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(metric_names, values, color=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title(f"{run_name} metrics")
    fig.tight_layout()

    path = os.path.join(output_dir, f"{run_name}_metrics_bar.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _generate_confusion_matrix_plot(confusion_matrix: Any, run_name: str, output_dir: str) -> Optional[str]:
    """Plot a 2D confusion matrix and save it as an image."""
    if plt is None or confusion_matrix is None:
        return None

    try:
        cm = np.asarray(confusion_matrix, dtype=float)
    except (TypeError, ValueError) as exc:
        logger.warning("Could not convert confusion_matrix to array: %s", exc)
        return None

    if cm.ndim != 2:
        logger.warning("Confusion matrix must be 2D; got %dD.", cm.ndim)
        return None

    os.makedirs(output_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, interpolation="nearest", cmap="viridis")
    fig.colorbar(im, ax=ax)

    ax.set_title(f"{run_name} confusion matrix")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")

    fig.tight_layout()
    path = os.path.join(output_dir, f"{run_name}_confusion_matrix.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def generate_report(
    metrics: Optional[Dict[str, Any]] = None,
    run_name: str = "run",
    output_dir: str = "reports",
    config: Optional[Any] = None,
) -> str:
    """Generate a text report and optional plots for a single run.

    Returns the path to the generated text report. The report is always
    generated even when matplotlib is unavailable; plots are skipped in that
    case.
    """
    cfg = _load_config(config)
    if metrics is None:
        metrics = {}

    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, f"{run_name}_report.txt")

    lines = [
        f"Evaluation report for run: {run_name}",
        f"Generated at: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
    ]

    for key in sorted(metrics.keys()):
        value = metrics.get(key)
        if key == "confusion_matrix":
            lines.append(f"{key}:")
            cm = value
            if isinstance(cm, np.ndarray):
                lines.extend([f"  {list(row)}" for row in cm.tolist()])
            elif isinstance(cm, (list, tuple)):
                lines.extend([f"  {list(row)}" for row in cm])
            else:
                lines.append(f"  {cm}")
        else:
            if isinstance(value, float):
                lines.append(f"{key}: {value:.6f}")
            else:
                lines.append(f"{key}: {value}")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    _generate_metrics_bar_chart(metrics, run_name, output_dir)
    _generate_confusion_matrix_plot(metrics.get("confusion_matrix"), run_name, output_dir)

    return report_path