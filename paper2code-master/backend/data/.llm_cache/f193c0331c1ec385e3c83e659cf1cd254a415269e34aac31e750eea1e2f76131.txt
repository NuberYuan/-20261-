import os
import yaml
import numpy as np
import torch

from utils.metrics import compute_metrics

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


def _resolve_config(config):
    """Return a config dict from a dict, YAML path, or empty dict."""
    if isinstance(config, dict):
        return config
    if isinstance(config, str):
        if os.path.isfile(config):
            with open(config, "r", encoding="utf-8") as f:
                return yaml.safe_load(f)
        return {}
    return {}


def _squeeze_channel(tensor):
    """Squeeze a 1-channel dimension if present, returning (N, H, W) tensor."""
    if tensor.dim() == 4 and tensor.size(1) == 1:
        return tensor.squeeze(1)
    return tensor


def _safe_metric_str(metrics, key, default="N/A"):
    """Return a safely formatted metric string for text reports."""
    value = metrics.get(key)
    if value is None:
        return default
    try:
        return f"{float(value):.6f}"
    except (TypeError, ValueError):
        return str(value)


def evaluate_model(
    model=None,
    dataloader=None,
    device="cpu",
    config=None,
    threshold=0.5,
    smooth=1e-6,
):
    """Evaluate a segmentation model on a dataloader.

    Metrics computed:
      - Dice coefficient
      - Intersection over Union (IoU)
      - Pixel accuracy
      - Precision
      - Recall
      - Mean/binary F1
      - Confusion matrix (2x2, [[TN, FP], [FN, TP]])

    Args:
        model: PyTorch nn.Module producing sigmoid probability outputs.
        dataloader: DataLoader yielding (images, masks).
        device: Target device for evaluation.
        config: Optional config dict (unused except for future compatibility).
        threshold: Binarization threshold for probabilities.
        smooth: Smoothing term for Dice/IoU to avoid division by zero.

    Returns:
        dict with evaluation metrics.
    """
    _ = config

    empty_metrics = {
        "dice_coefficient": 0.0,
        "iou": 0.0,
        "accuracy": 0.0,
        "precision": 0.0,
        "recall": 0.0,
        "mean_f1": 0.0,
        "confusion_matrix": [[0, 0], [0, 0]],
    }

    if model is None or dataloader is None:
        return empty_metrics

    model.to(device)
    model.eval()

    total_dice = 0.0
    total_iou = 0.0
    total_samples = 0

    tp_total = 0
    fp_total = 0
    fn_total = 0
    tn_total = 0

    with torch.no_grad():
        for images, masks in dataloader:
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)
            if isinstance(outputs, (list, tuple)):
                # In deep supervision mode, evaluate the highest-resolution head.
                outputs = outputs[-1]

            batch_size = images.size(0)

            batch_metrics = compute_metrics(
                preds=outputs,
                targets=masks,
                threshold=threshold,
                smooth=smooth,
            )
            total_dice += float(batch_metrics["dice_coefficient"]) * batch_size
            total_iou += float(batch_metrics["iou"]) * batch_size
            total_samples += batch_size

            preds_bin = _squeeze_channel((outputs > threshold).float())
            masks_bin = _squeeze_channel(masks.float())

            tp = torch.logical_and(preds_bin == 1, masks_bin == 1).sum().item()
            fp = torch.logical_and(preds_bin == 1, masks_bin == 0).sum().item()
            fn = torch.logical_and(preds_bin == 0, masks_bin == 1).sum().item()
            tn = torch.logical_and(preds_bin == 0, masks_bin == 0).sum().item()

            tp_total += int(tp)
            fp_total += int(fp)
            fn_total += int(fn)
            tn_total += int(tn)

    dice = total_dice / total_samples if total_samples > 0 else 0.0
    iou = total_iou / total_samples if total_samples > 0 else 0.0

    total_pixels = tp_total + fp_total + fn_total + tn_total
    accuracy = (tp_total + tn_total) / total_pixels if total_pixels > 0 else 0.0
    precision = tp_total / (tp_total + fp_total) if (tp_total + fp_total) > 0 else 0.0
    recall = tp_total / (tp_total + fn_total) if (tp_total + fn_total) > 0 else 0.0
    if precision + recall > 0:
        f1 = 2.0 * precision * recall / (precision + recall)
    else:
        f1 = 0.0

    confusion_matrix = [[tn_total, fp_total], [fn_total, tp_total]]

    model.train()

    return {
        "dice_coefficient": float(dice),
        "iou": float(iou),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "mean_f1": float(f1),
        "confusion_matrix": confusion_matrix,
    }


def _generate_metrics_bar_chart(metrics, output_dir, run_name=None):
    """Generate a bar chart for mean_f1, precision, recall, and accuracy.

    Args:
        metrics: Complete metrics dict. The function extracts only the four
            specified keys. Missing or non-numeric values are skipped.
        output_dir: Directory in which to save the chart.
        run_name: Optional run name used in the plot title.

    Returns:
        Path to the saved PNG, or None if matplotlib is unavailable or no
        plottable values are present.
    """
    if plt is None:
        return None
    if not isinstance(metrics, dict):
        return None

    keys = ["mean_f1", "precision", "recall", "accuracy"]
    labels = []
    values = []

    for key in keys:
        value = metrics.get(key)
        if value is None:
            continue
        try:
            values.append(float(value))
            labels.append(key.replace("_", " ").title())
        except (TypeError, ValueError):
            continue

    if not labels:
        return None

    try:
        os.makedirs(output_dir, exist_ok=True)
        plt.figure(figsize=(6, 4))
        plt.bar(labels, values)
        plt.ylim(0.0, 1.0)
        plt.ylabel("Score")
        plt.title(f"{run_name} metrics" if run_name else "Metrics")
        plt.tight_layout()
        path = os.path.join(output_dir, "metrics_bar_chart.png")
        plt.savefig(path, dpi=150)
        plt.close()
        return path
    except Exception:
        plt.close()
        return None


def _generate_confusion_matrix_plot(confusion_matrix, output_dir, run_name=None):
    """Generate a 2x2 confusion matrix plot.

    Args:
        confusion_matrix: 2D array in [[TN, FP], [FN, TP]] order.
        output_dir: Directory in which to save the plot.
        run_name: Optional run name used in the plot title.

    Returns:
        Path to the saved PNG, or None if matplotlib is unavailable or the
        input is invalid.
    """
    if plt is None:
        return None
    if confusion_matrix is None:
        return None

    try:
        cm = np.asarray(confusion_matrix, dtype=np.int64)
        if cm.ndim != 2 or cm.shape[0] < 2 or cm.shape[1] < 2 or cm.size != 4:
            return None

        os.makedirs(output_dir, exist_ok=True)
        plt.figure(figsize=(5, 5))
        plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)

        classes = ["Negative", "Positive"]
        tick_marks = np.arange(2)
        plt.xticks(tick_marks, classes)
        plt.yticks(tick_marks, classes)
        plt.xlabel("Predicted label")
        plt.ylabel("True label")
        plt.title(f"{run_name} confusion matrix" if run_name else "Confusion matrix")

        thresh = cm.max() / 2.0 if cm.max() > 0 else 0.5
        for i in range(2):
            for j in range(2):
                plt.text(
                    j,
                    i,
                    str(cm[i, j]),
                    ha="center",
                    va="center",
                    color="white" if cm[i, j] > thresh else "black",
                )

        plt.tight_layout()
        path = os.path.join(output_dir, "confusion_matrix.png")
        plt.savefig(path, dpi=150)
        plt.close()
        return path
    except Exception:
        plt.close()
        return None


def generate_report(metrics=None, run_name="run", output_dir=None, config=None):
    """Generate a text report and optional diagnostic plots for one run.

    The text report is always written, even if matplotlib is missing. Plots are
    generated only when matplotlib is available.

    Args:
        metrics: Metrics dict returned by evaluate_model.
        run_name: Run identifier used for file names and titles.
        output_dir: Directory in which to store report outputs. If None, fall
            back to config['logging']['save_dir']/reports or logs/reports.
        config: Optional config dict. Used only to resolve default output_dir.

    Returns:
        Path to the text report file.
    """
    metrics = metrics if isinstance(metrics, dict) else {}
    cfg = _resolve_config(config)

    if output_dir is None:
        log_dir = cfg.get("logging", {}).get("save_dir", "logs")
        output_dir = os.path.join(log_dir, "reports")

    os.makedirs(output_dir, exist_ok=True)

    report_path = os.path.join(output_dir, f"{run_name}_report.txt")
    lines = [
        f"Evaluation Report for {run_name}",
        "=" * 40,
        "Metrics:",
        f"  Dice coefficient : {_safe_metric_str(metrics, 'dice_coefficient')}",
        f"  IoU              : {_safe_metric_str(metrics, 'iou')}",
        f"  Accuracy         : {_safe_metric_str(metrics, 'accuracy')}",
        f"  Precision        : {_safe_metric_str(metrics, 'precision')}",
        f"  Recall           : {_safe_metric_str(metrics, 'recall')}",
        f"  Mean F1          : {_safe_metric_str(metrics, 'mean_f1')}",
        "",
    ]

    if plt is not None:
        bar_path = _generate_metrics_bar_chart(metrics, output_dir, run_name=run_name)
        if bar_path:
            lines.append(f"Metrics bar chart saved to: {bar_path}")

        cm = metrics.get("confusion_matrix")
        if cm is not None:
            cm_path = _generate_confusion_matrix_plot(cm, output_dir, run_name=run_name)
            if cm_path:
                lines.append(f"Confusion matrix plot saved to: {cm_path}")
    else:
        lines.append("Matplotlib not available; skipping chart generation.")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return report_path