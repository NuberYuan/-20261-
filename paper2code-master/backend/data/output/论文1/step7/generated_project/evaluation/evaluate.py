"""Phase 5 - evaluation and report generation for the PlantVillage reproduction.

Paper: *Using Deep Learning for Image-Based Plant Disease Detection*
Task : image_classification (PyTorch, EPFL GPU cluster computing resources)

Public API
----------
``evaluate_model``
    Run a trained plant-disease classifier on a data loader and compute the
    metrics required by FR-09 (Mean F1 score, mean precision, mean recall,
    overall accuracy) plus the confusion matrix used by the auxiliary plots.
``generate_report``
    Write the textual / JSON report of a single run and, when matplotlib is
    available, the auxiliary charts required by FR-10.

Design notes
------------
* matplotlib is only a visual-aid dependency: it is imported inside a
  ``try / except`` block and every plotting helper degrades gracefully to
  ``None`` when it is missing.  Metric computation and the text report are
  always produced, with or without matplotlib.
* No conclusion about model quality is hard-coded anywhere.  The reference
  values quoted inside ``config/config.yaml`` (``project.reference_note``) are
  parsed at runtime and compared with the numbers actually measured by the
  current run; every delta / tolerance verdict is recomputed at run time.
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
import sys
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import yaml

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

try:
    import numpy as np
except ImportError:
    np = None


LOGGER = logging.getLogger(__name__)

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


# --------------------------------------------------------------------------- #
# Optional project dependencies (must never break the import of this module).
# --------------------------------------------------------------------------- #
try:
    from utils.metrics import compute_metrics as _compute_metrics
except Exception as _exc:  # pragma: no cover - defensive import
    LOGGER.warning("utils.metrics.compute_metrics unavailable: %s", _exc)
    _compute_metrics = None

try:
    from training.train import load_config as _train_load_config
except Exception as _exc:  # pragma: no cover - defensive import
    LOGGER.warning("training.train.load_config unavailable: %s", _exc)
    _train_load_config = None

try:
    from training.train import build_criterion as _train_build_criterion
except Exception as _exc:  # pragma: no cover - defensive import
    LOGGER.warning("training.train.build_criterion unavailable: %s", _exc)
    _train_build_criterion = None

try:
    from models.alexnet import Alexnet
except Exception as _exc:  # pragma: no cover - defensive import
    LOGGER.warning("models.alexnet.Alexnet unavailable: %s", _exc)
    Alexnet = None

try:
    from models.googlenet import Googlenet
except Exception as _exc:  # pragma: no cover - defensive import
    LOGGER.warning("models.googlenet.Googlenet unavailable: %s", _exc)
    Googlenet = None


# --------------------------------------------------------------------------- #
# Module constants
# --------------------------------------------------------------------------- #
DEFAULT_NUM_CLASSES = 38
DEFAULT_THRESHOLD = 0.5
DEFAULT_LOG_INTERVAL = 50
DEFAULT_SEED = 42
DEFAULT_REFERENCE_TOLERANCE = 0.01
DEFAULT_REPORT_DIR = os.path.join("outputs", "reports")
_DEFAULT_CONFIG_RELATIVE_PATH = os.path.join("config", "config.yaml")

_HEADLINE_KEYS: Tuple[str, ...] = (
    "mean_f1",
    "mean_precision",
    "mean_recall",
    "overall_accuracy",
)

_HEADLINE_LABELS: Dict[str, str] = {
    "mean_f1": "Mean F1 score",
    "mean_precision": "Mean precision",
    "mean_recall": "Mean recall",
    "overall_accuracy": "Overall accuracy",
}

_METRIC_ALIASES: Dict[str, Tuple[str, ...]] = {
    "mean_f1": ("mean_f1", "mean_f1_score", "mean_f1score", "macro_f1", "f1", "f1_score"),
    "mean_precision": ("mean_precision", "mean_precision_score", "macro_precision", "precision"),
    "mean_recall": ("mean_recall", "mean_recall_score", "macro_recall", "recall"),
    "overall_accuracy": (
        "overall_accuracy",
        "overall_acc",
        "accuracy",
        "acc",
        "top1_accuracy",
        "top_1_accuracy",
    ),
}

# Runtime-parsed from ``project.reference_note`` (never hard-coded values).
_REFERENCE_PATTERNS: Dict[str, "re.Pattern[str]"] = {
    "mean_f1": re.compile(r"mean\s*F\s*1[^0-9\-]*([0-9]*\.?[0-9]+)", re.IGNORECASE),
    "overall_accuracy": re.compile(
        r"overall\s*accuracy[^0-9\-]*([0-9]*\.?[0-9]+)\s*(%|percent)?", re.IGNORECASE
    ),
    "model": re.compile(r"(GoogLeNet|AlexNet)", re.IGNORECASE),
    "training_mode": re.compile(r"(transfer learning|training[- ]from[- ]scratch|scratch|transfer)",
                                re.IGNORECASE),
    "dataset_version": re.compile(r"\b(color|grayscale|segmented)\b", re.IGNORECASE),
    "train_test_split": re.compile(r"(\d{2}\s*[-–]\s*\d{2})"),
}


# --------------------------------------------------------------------------- #
# Configuration helpers
# --------------------------------------------------------------------------- #
def _candidate_config_paths() -> List[str]:
    """Return existing-file candidates for ``config/config.yaml`` (priority order)."""
    candidates: List[str] = []
    env_path = os.environ.get("PLANT_DISEASE_CONFIG")
    if env_path:
        candidates.append(env_path)
    candidates.append(os.path.join(_PROJECT_ROOT, "config", "config.yaml"))
    here = os.path.dirname(os.path.abspath(__file__))
    candidates.append(os.path.join(os.path.dirname(here), "config", "config.yaml"))
    candidates.append(os.path.join(here, "config", "config.yaml"))
    candidates.append(os.path.join(os.getcwd(), "config", "config.yaml"))
    candidates.append(_DEFAULT_CONFIG_RELATIVE_PATH)

    ordered: List[str] = []
    seen: set = set()
    for path in candidates:
        if not path:
            continue
        key = os.path.normpath(path)
        if key in seen:
            continue
        seen.add(key)
        ordered.append(path)
    return ordered


def _resolve_config_path(config: Optional[Union[str, Dict[str, Any]]] = None) -> Optional[str]:
    """Return a readable configuration path (``None`` for dict / missing files)."""
    if isinstance(config, dict):
        return None
    if isinstance(config, (str, os.PathLike)):
        path = os.fspath(config)
        return path if os.path.isfile(path) else None
    for candidate in _candidate_config_paths():
        if candidate and os.path.isfile(candidate):
            return candidate
    return None


def load_config(config: Optional[Union[str, Dict[str, Any]]] = None) -> Dict[str, Any]:
    """Return the project configuration as a plain dictionary."""
    if isinstance(config, dict):
        return dict(config)

    if isinstance(config, (str, os.PathLike)):
        path = os.fspath(config)
        if os.path.isfile(path):
            try:
                with open(path, "r", encoding="utf-8") as handle:
                    loaded = yaml.safe_load(handle) or {}
                if isinstance(loaded, dict):
                    return loaded
                LOGGER.warning("Configuration file %s did not contain a mapping.", path)
            except Exception as exc:  # pragma: no cover - defensive
                LOGGER.warning("Failed to read configuration file %s: %s", path, exc)
        return {}

    for candidate in _candidate_config_paths():
        if not candidate or not os.path.isfile(candidate):
            continue
        try:
            with open(candidate, "r", encoding="utf-8") as handle:
                loaded = yaml.safe_load(handle) or {}
            if isinstance(loaded, dict):
                return loaded
        except Exception as exc:  # pragma: no cover - defensive
            LOGGER.warning("Failed to read configuration file %s: %s", candidate, exc)
    return {}


def _cfg_get(config: Optional[Dict[str, Any]], dotted_key: str, default: Any = None) -> Any:
    """Read a value from a config dict, supporting both nested and dotted keys."""
    if not isinstance(config, dict):
        return default

    if dotted_key in config:
        return config[dotted_key]

    node: Any = config
    for part in dotted_key.split("."):
        if isinstance(node, dict) and part in node:
            node = node[part]
        else:
            return default
    return node


# --------------------------------------------------------------------------- #
# Metric helpers
# --------------------------------------------------------------------------- #
def _to_list(value: Any) -> List[Any]:
    """Best-effort conversion of tensors / arrays into nested python lists."""
    if value is None:
        return []
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if np is not None and isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _to_long_tensor(value: Any) -> torch.Tensor:
    """Convert a sequence of labels into a 1-D ``torch.long`` tensor."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().reshape(-1).long()
    if np is not None and isinstance(value, np.ndarray):
        return torch.from_numpy(value.reshape(-1)).long()
    return torch.tensor(list(value), dtype=torch.long)


def _labels_from_outputs(outputs: Any) -> torch.Tensor:
    """Turn raw logits / probabilities / indices into 1-D integer predictions."""
    if isinstance(outputs, torch.Tensor):
        tensor = outputs.detach().cpu()
    elif np is not None and isinstance(outputs, np.ndarray):
        tensor = torch.from_numpy(outputs)
    else:
        tensor = torch.tensor(list(outputs))

    if tensor.dim() == 0:
        return tensor.reshape(1).long()
    if tensor.dim() == 1:
        if tensor.dtype.is_floating_point and 0.0 <= float(tensor.min()) <= float(tensor.max()) and (
            float(tensor.sum()) < 1.5
        ):
            return torch.argmax(tensor.reshape(1, -1), dim=1).long()
        return tensor.reshape(-1).long()
    return torch.argmax(tensor, dim=1).long()


def _confusion_matrix(y_true: torch.Tensor, y_pred: torch.Tensor, num_classes: int) -> torch.Tensor:
    """Compute a ``num_classes x num_classes`` confusion matrix (rows = true)."""
    cm = torch.zeros(num_classes, num_classes, dtype=torch.long)
    if y_true.numel() == 0:
        return cm
    idx = y_true * num_classes + y_pred
    ones = torch.ones_like(idx, dtype=torch.long)
    cm = cm.reshape(-1).index_add_(0, idx, ones).reshape(num_classes, num_classes)
    return cm


def _metrics_from_confusion(cm: torch.Tensor, zero_division: float = 0.0) -> Dict[str, Any]:
    """Derive per-class and macro metrics from a confusion matrix."""
    cm_f = cm.float()
    tp = cm_f.diag()
    support = cm_f.sum(dim=1)
    predicted = cm_f.sum(dim=0)
    total = float(cm_f.sum().item())

    def _safe_div(num: torch.Tensor, den: torch.Tensor) -> torch.Tensor:
        out = torch.full_like(num, float(zero_division))
        mask = den > 0
        out[mask] = num[mask] / den[mask]
        return out

    precision = _safe_div(tp, predicted)
    recall = _safe_div(tp, support)
    f1 = _safe_div(2.0 * precision * recall, precision + recall)

    accuracy = float(tp.sum().item() / total) if total > 0 else float(zero_division)

    per_class: List[Dict[str, Any]] = []
    for idx in range(cm.size(0)):
        per_class.append(
            {
                "class_index": idx,
                "precision": float(precision[idx].item()),
                "recall": float(recall[idx].item()),
                "f1": float(f1[idx].item()),
                "support": int(support[idx].item()),
            }
        )

    return {
        "overall_accuracy": accuracy,
        "accuracy": accuracy,
        "mean_precision": float(precision.mean().item()) if precision.numel() else float(zero_division),
        "mean_recall": float(recall.mean().item()) if recall.numel() else float(zero_division),
        "mean_f1": float(f1.mean().item()) if f1.numel() else float(zero_division),
        "per_class": per_class,
    }


def _top_k_accuracy(logits: torch.Tensor, targets: torch.Tensor, top_k: int) -> float:
    """Top-k accuracy over a batch of logits / targets."""
    if logits.dim() != 2 or top_k <= 0:
        return 0.0
    k = min(int(top_k), int(logits.size(1)))
    if k <= 0:
        return 0.0
    _, pred = logits.topk(k, dim=1)
    correct = pred.eq(targets.reshape(-1, 1)).any(dim=1).sum().item()
    return float(correct) / max(1, int(targets.numel()))


def _unpack_batch(batch: Any) -> Tuple[torch.Tensor, torch.Tensor]:
    """Extract ``(inputs, targets)`` from a dataloader batch."""
    if isinstance(batch, dict):
        inputs = batch.get("image", batch.get("images", batch.get("input", batch.get("inputs"))))
        targets = batch.get("label", batch.get("labels", batch.get("target", batch.get("targets"))))
        if inputs is None or targets is None:
            raise ValueError("Batch dictionary must provide image / label entries.")
        return inputs, targets
    if isinstance(batch, (tuple, list)):
        if len(batch) < 2:
            raise ValueError("Batch tuple must contain at least (inputs, targets).")
        return batch[0], batch[1]
    raise TypeError(f"Unsupported batch type: {type(batch)!r}")


def _extract_main_logits(outputs: Any) -> torch.Tensor:
    """Return the main logits tensor from a model forward-pass output."""
    if isinstance(outputs, torch.Tensor):
        return outputs
    if isinstance(outputs, (tuple, list)):
        for item in outputs:
            if isinstance(item, torch.Tensor):
                return item
        raise ValueError("Model output tuple contains no tensor.")
    for attribute in ("logits", "out", "output"):
        if hasattr(outputs, attribute):
            value = getattr(outputs, attribute)
            if isinstance(value, torch.Tensor):
                return value
    raise TypeError(f"Unsupported model output type: {type(outputs)!r}")


def _resolve_device(device: Optional[Union[str, torch.device]] = None) -> torch.device:
    """Resolve the evaluation device, falling back to CPU when CUDA is missing."""
    if device is None:
        return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    resolved = torch.device(device) if not isinstance(device, torch.device) else device
    if resolved.type == "cuda" and not torch.cuda.is_available():
        LOGGER.warning("CUDA requested but unavailable; falling back to CPU.")
        return torch.device("cpu")
    return resolved


# --------------------------------------------------------------------------- #
# Reference-note parsing (values are always read from the config at runtime)
# --------------------------------------------------------------------------- #
def _parse_reference_note(config: Dict[str, Any]) -> Dict[str, Any]:
    """Parse ``project.reference_note`` into numeric / textual reference fields."""
    note = _cfg_get(config, "project.reference_note", "") or ""
    note = str(note)

    def _number(pattern: "re.Pattern[str]", default: Optional[float] = None) -> Optional[float]:
        match = pattern.search(note)
        if not match:
            return default
        try:
            value = float(match.group(1))
        except (TypeError, ValueError):
            return default
        # Percentages quoted as e.g. "99.35%" are normalised to [0, 1].
        if value > 1.0:
            value = value / 100.0
        return value

    reference: Dict[str, Any] = {
        "note": note,
        "mean_f1": _number(_REFERENCE_PATTERNS["mean_f1"]),
        "overall_accuracy": _number(_REFERENCE_PATTERNS["overall_accuracy"]),
    }
    for key in ("model", "training_mode", "dataset_version", "train_test_split"):
        match = _REFERENCE_PATTERNS[key].search(note)
        reference[key] = match.group(1) if match else None
    return reference


def _compare_with_reference(
    metrics: Dict[str, Any], reference: Dict[str, Any], tolerance: float = DEFAULT_REFERENCE_TOLERANCE
) -> Dict[str, Any]:
    """Recompute, at run time, how the measured metrics compare with the reference note."""
    comparison: Dict[str, Any] = {"tolerance": float(tolerance), "reference": reference, "deltas": {}}
    all_within = True
    for key in _HEADLINE_KEYS:
        ref_value = reference.get(key)
        measured = metrics.get(key)
        if ref_value is None or measured is None:
            comparison["deltas"][key] = None
            continue
        delta = float(measured) - float(ref_value)
        within = abs(delta) <= float(tolerance)
        all_within = all_within and within
        comparison["deltas"][key] = {
            "reference": float(ref_value),
            "measured": float(measured),
            "delta": delta,
            "within_tolerance": within,
        }
    comparison["verdict"] = "reference_reproduced" if all_within else "reference_not_reproduced"
    comparison["all_within_tolerance"] = all_within
    return comparison


# --------------------------------------------------------------------------- #
# Public API - evaluate_model
# --------------------------------------------------------------------------- #
def evaluate_model(
    model: nn.Module,
    loader: Optional[DataLoader] = None,
    criterion: Optional[nn.Module] = None,
    device: Optional[Union[str, torch.device]] = None,
    config: Optional[Union[str, Dict[str, Any]]] = None,
    log_interval: int = DEFAULT_LOG_INTERVAL,
    epoch: int = 0,
    class_names: Optional[Sequence[Any]] = None,
    top_k: Optional[Union[int, Sequence[int]]] = None,
    return_predictions: bool = False,
    tag: str = "eval",
    **kwargs: Any,
) -> Dict[str, Any]:
    """Evaluate ``model`` on ``loader`` (no gradient updates) and return metrics.

    The returned dictionary contains the FR-09 headline metrics
    (``mean_f1``, ``mean_precision``, ``mean_recall``, ``overall_accuracy``),
    the average loss, the confusion matrix and per-class details.
    """
    if model is None:
        raise ValueError("evaluate_model requires a valid ``model``.")
    if loader is None:
        raise ValueError("evaluate_model requires a valid data ``loader``.")

    cfg = load_config(config)

    if criterion is None:
        if _train_build_criterion is not None:
            try:
                criterion = _train_build_criterion(cfg)
            except Exception as exc:  # pragma: no cover - defensive
                LOGGER.warning("build_criterion failed (%s); defaulting to CrossEntropyLoss.", exc)
                criterion = None
        if criterion is None:
            criterion = nn.CrossEntropyLoss()

    resolved_device = _resolve_device(device if device is not None else kwargs.get("device"))
    model = model.to(resolved_device)
    model.eval()

    was_training = bool(model.training)
    total_loss = 0.0
    total_batches = 0
    total_samples = 0
    correct = 0

    all_preds: List[torch.Tensor] = []
    all_targets: List[torch.Tensor] = []
    all_logits: List[torch.Tensor] = []

    top_k_values: List[int] = []
    if top_k is not None:
        if isinstance(top_k, int):
            top_k_values = [int(top_k)]
        else:
            top_k_values = [int(k) for k in top_k]
    else:
        cfg_top_k = _cfg_get(cfg, "inference.top_k", None)
        if isinstance(cfg_top_k, int) and cfg_top_k > 1:
            top_k_values = [int(cfg_top_k)]
    top_k_hits: Dict[int, int] = {k: 0 for k in top_k_values}

    start = time.time()
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader, start=1):
            inputs, targets = _unpack_batch(batch)
            inputs = inputs.to(resolved_device, non_blocking=True)
            targets = targets.to(resolved_device, non_blocking=True).reshape(-1).long()

            outputs = model(inputs)
            main_logits = _extract_main_logits(outputs)

            loss_value = 0.0
            if criterion is not None:
                try:
                    loss = criterion(main_logits, targets)
                    loss_value = float(loss.item())
                except Exception as exc:  # pragma: no cover - defensive
                    LOGGER.warning("Loss computation failed: %s", exc)
                    loss_value = 0.0

            preds = torch.argmax(main_logits, dim=1)

            batch_size = int(targets.numel())
            total_loss += loss_value * batch_size
            total_batches += 1
            total_samples += batch_size
            correct += int((preds == targets).sum().item())

            all_preds.append(preds.detach().cpu())
            all_targets.append(targets.detach().cpu())
            all_logits.append(main_logits.detach().cpu())

            for k in top_k_values:
                if k <= main_logits.size(1):
                    _, top_indices = main_logits.topk(k, dim=1)
                    top_k_hits[k] += int(top_indices.eq(targets.reshape(-1, 1)).any(dim=1).sum().item())

            if log_interval and batch_idx % max(1, int(log_interval)) == 0:
                LOGGER.info(
                    "[%s] epoch=%s batch=%d loss=%.4f", tag, epoch, batch_idx, loss_value
                )

    elapsed = time.time() - start

    if total_samples == 0:
        LOGGER.warning("evaluate_model: the loader yielded no samples.")
        return {
            "loss": 0.0,
            "mean_f1": 0.0,
            "mean_precision": 0.0,
            "mean_recall": 0.0,
            "overall_accuracy": 0.0,
            "accuracy": 0.0,
            "num_samples": 0,
            "num_batches": 0,
            "confusion_matrix": [],
            "per_class": [],
            "epoch": epoch,
            "tag": tag,
            "elapsed_seconds": elapsed,
        }

    y_pred = torch.cat(all_preds, dim=0) if all_preds else torch.zeros(0, dtype=torch.long)
    y_true = torch.cat(all_targets, dim=0) if all_targets else torch.zeros(0, dtype=torch.long)
    logits_all = torch.cat(all_logits, dim=0) if all_logits else torch.zeros(0, 0)

    inferred_classes = int(max(int(y_true.max().item()), int(y_pred.max().item())) + 1)
    num_classes = int(_cfg_get(cfg, "data.num_classes", DEFAULT_NUM_CLASSES) or DEFAULT_NUM_CLASSES)
    num_classes = max(num_classes, inferred_classes)

    cm = _confusion_matrix(y_true, y_pred, num_classes)
    metrics = _metrics_from_confusion(cm)

    # Cross-check with the shared metric helper when it is importable.
    if _compute_metrics is not None:
        try:
            shared = _compute_metrics(
                predictions=y_pred,
                targets=y_true,
                config=cfg,
                class_names=class_names,
                num_classes=num_classes,
            )
            if isinstance(shared, dict):
                for key in _HEADLINE_KEYS:
                    for alias in _METRIC_ALIASES.get(key, (key,)):
                        if alias in shared and shared[alias] is not None:
                            metrics[key] = float(shared[alias])
                            break
                metrics["accuracy"] = metrics["overall_accuracy"]
        except Exception as exc:  # pragma: no cover - defensive
            LOGGER.debug("compute_metrics helper skipped: %s", exc)

    result: Dict[str, Any] = {
        "loss": float(total_loss) / max(1, total_samples),
        "mean_f1": float(metrics["mean_f1"]),
        "mean_precision": float(metrics["mean_precision"]),
        "mean_recall": float(metrics["mean_recall"]),
        "overall_accuracy": float(metrics["overall_accuracy"]),
        "accuracy": float(metrics["accuracy"]),
        "per_class": metrics["per_class"],
        "confusion_matrix": cm.tolist(),
        "class_names": list(class_names) if class_names is not None else None,
        "num_classes": num_classes,
        "num_samples": total_samples,
        "num_batches": total_batches,
        "epoch": epoch,
        "tag": tag,
        "elapsed_seconds": elapsed,
        "device": str(resolved_device),
    }

    for k in top_k_values:
        result[f"top_{k}_accuracy"] = float(top_k_hits.get(k, 0)) / max(1, total_samples)

    if return_predictions:
        result["predictions"] = y_pred.tolist()
        result["targets"] = y_true.tolist()
        if logits_all.numel() > 0:
            result["logits"] = logits_all.tolist()

    if was_training:
        model.train()

    LOGGER.info(
        "evaluate_model [%s]: loss=%.4f acc=%.4f mean_f1=%.4f (n=%d)",
        tag,
        result["loss"],
        result["overall_accuracy"],
        result["mean_f1"],
        total_samples,
    )
    return result


# --------------------------------------------------------------------------- #
# Report helpers
# --------------------------------------------------------------------------- #
def _json_default(obj: Any) -> Any:
    """Serialise tensors / numpy scalars / sets for ``json.dump``."""
    if isinstance(obj, (set, tuple)):
        return list(obj)
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    if np is not None and isinstance(obj, np.ndarray):
        return obj.tolist()
    for attr in ("item", "tolist"):
        if hasattr(obj, attr):
            try:
                return getattr(obj, attr)()
            except Exception:  # pragma: no cover - defensive
                pass
    return str(obj)


def _lookup_metric(metrics: Dict[str, Any], key: str) -> Optional[float]:
    """Return the first present alias of ``key`` inside ``metrics``."""
    for alias in _METRIC_ALIASES.get(key, (key,)):
        if alias in metrics and metrics[alias] is not None:
            try:
                return float(metrics[alias])
            except (TypeError, ValueError):
                continue
    return None


def _format_report_text(
    metrics: Dict[str, Any],
    reference_comparison: Dict[str, Any],
    run_info: Dict[str, Any],
) -> str:
    """Build the plain-text report of a single evaluation run."""
    lines: List[str] = []
    lines.append("=" * 78)
    lines.append("Plant Disease Detection - Evaluation Report")
    lines.append("=" * 78)
    lines.append(f"Generated at : {datetime.now().isoformat(timespec='seconds')}")
    for key in (
        "run_id",
        "model",
        "model_name",
        "training_mode",
        "dataset_version",
        "train_test_split",
        "split_ratio",
        "checkpoint",
        "device",
    ):
        if key in run_info and run_info[key] is not None:
            lines.append(f"{key:<14}: {run_info[key]}")
    lines.append("-" * 78)
    lines.append("Headline metrics (FR-09)")
    lines.append("-" * 78)
    for key in _HEADLINE_KEYS:
        value = _lookup_metric(metrics, key)
        label = _HEADLINE_LABELS.get(key, key)
        lines.append(f"{label:<22}: {'n/a' if value is None else f'{value:.6f}'}")
    loss = metrics.get("loss")
    if loss is not None:
        lines.append(f"{'Average loss':<22}: {float(loss):.6f}")
    num_samples = metrics.get("num_samples")
    if num_samples is not None:
        lines.append(f"{'Evaluated samples':<22}: {int(num_samples)}")

    top_keys = sorted(k for k in metrics.keys() if re.fullmatch(r"top_\d+_accuracy", str(k)))
    if top_keys:
        lines.append("-" * 78)
        lines.append("Top-k accuracy")
        for key in top_keys:
            lines.append(f"{key:<22}: {float(metrics[key]):.6f}")

    lines.append("-" * 78)
    lines.append("Reference comparison (recomputed at runtime from project.reference_note)")
    reference = reference_comparison.get("reference", {}) or {}
    lines.append(f"Reference note : {reference.get('note', '')}")
    deltas = reference_comparison.get("deltas", {}) or {}
    for key in _HEADLINE_KEYS:
        entry = deltas.get(key)
        label = _HEADLINE_LABELS.get(key, key)
        if not isinstance(entry, dict):
            lines.append(f"{label:<22}: no reference value parsed")
            continue
        lines.append(
            f"{label:<22}: reference={entry['reference']:.6f} "
            f"measured={entry['measured']:.6f} delta={entry['delta']:+.6f} "
            f"within_tol={entry['within_tolerance']}"
        )
    lines.append(f"Verdict         : {reference_comparison.get('verdict', 'unknown')}")
    lines.append("-" * 78)

    per_class = metrics.get("per_class") or []
    class_names = metrics.get("class_names")
    if per_class:
        lines.append("Per-class metrics")
        lines.append("-" * 78)
        for entry in per_class:
            idx = int(entry.get("class_index", 0))
            name = None
            if class_names is not None and idx < len(class_names):
                name = class_names[idx]
            label = name if name is not None else f"class_{idx}"
            lines.append(
                f"{str(label):<32} precision={float(entry.get('precision', 0.0)):.4f} "
                f"recall={float(entry.get('recall', 0.0)):.4f} "
                f"f1={float(entry.get('f1', 0.0)):.4f} "
                f"support={int(entry.get('support', 0))}"
            )
    lines.append("=" * 78)
    return "\n".join(lines)


def _plot_confusion_matrix(
    cm: Sequence[Sequence[Any]],
    class_names: Optional[Sequence[Any]],
    out_path: Optional[str],
) -> Optional[str]:
    """Render a confusion-matrix heatmap (returns ``None`` without matplotlib)."""
    if plt is None or out_path is None:
        return None
    try:
        matrix = np.array(cm, dtype=float) if np is not None else None
        if matrix is None or matrix.size == 0:
            return None
        fig, ax = plt.subplots(figsize=(10, 8))
        image = ax.imshow(matrix, interpolation="nearest", cmap="Blues")
        fig.colorbar(image, ax=ax)
        labels = list(class_names) if class_names is not None else [str(i) for i in range(matrix.shape[0])]
        ax.set_xlabel("Predicted label")
        ax.set_ylabel("True label")
        ax.set_title("Confusion matrix")
        if matrix.shape[0] <= 40:
            ax.set_xticks(range(matrix.shape[1]))
            ax.set_yticks(range(matrix.shape[0]))
            ax.set_xticklabels(labels, rotation=90, fontsize=6)
            ax.set_yticklabels(labels, fontsize=6)
        fig.tight_layout()
        os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        return out_path
    except Exception as exc:  # pragma: no cover - defensive
        LOGGER.warning("Confusion-matrix plot failed: %s", exc)
        return None


def _plot_metric_bars(metrics: Dict[str, Any], out_path: Optional[str]) -> Optional[str]:
    """Render a bar chart of the headline metrics."""
    if plt is None or out_path is None:
        return None
    try:
        labels: List[str] = []
        values: List[float] = []
        for key in _HEADLINE_KEYS:
            value = _lookup_metric(metrics, key)
            if value is None:
                continue
            labels.append(_HEADLINE_LABELS.get(key, key))
            values.append(float(value))
        if not labels:
            return None
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(labels, values, color="#4C72B0")
        ax.set_ylim(0.0, 1.0)
        ax.set_ylabel("Score")
        ax.set_title("Evaluation metrics")
        for idx, value in enumerate(values):
            ax.text(idx, min(0.995, value + 0.01), f"{value:.4f}", ha="center", fontsize=8)
        fig.tight_layout()
        os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        return out_path
    except Exception as exc:  # pragma: no cover - defensive
        LOGGER.warning("Metric bar chart failed: %s", exc)
        return None


def _plot_history(history: Optional[Sequence[Dict[str, Any]]], out_path: Optional[str]) -> Optional[str]:
    """Render the training / validation curves when a history is supplied."""
    if plt is None or out_path is None or not history:
        return None
    try:
        epochs = [int(entry.get("epoch", idx + 1)) for idx, entry in enumerate(history)]
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        for key, label in (("loss", "Loss"), ("val_loss", "Validation loss")):
            values = [entry.get(key) for entry in history]
            if any(v is not None for v in values):
                axes[0].plot(epochs, [float(v) if v is not None else math.nan for v in values], label=label)
        axes[0].set_xlabel("Epoch")
        axes[0].set_title("Loss curves")
        axes[0].legend()
        for key, label in (
            ("accuracy", "Train accuracy"),
            ("val_accuracy", "Val accuracy"),
            ("mean_f1", "Train mean F1"),
            ("val_mean_f1", "Val mean F1"),
        ):
            values = [entry.get(key) for entry in history]
            if any(v is not None for v in values):
                axes[1].plot(epochs, [float(v) if v is not None else math.nan for v in values], label=label)
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylim(0.0, 1.0)
        axes[1].set_title("Metric curves")
        axes[1].legend()
        fig.tight_layout()
        os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        return out_path
    except Exception as exc:  # pragma: no cover - defensive
        LOGGER.warning("History plot failed: %s", exc)
        return None


# --------------------------------------------------------------------------- #
# Public API - generate_report
# --------------------------------------------------------------------------- #
def generate_report(
    metrics: Optional[Dict[str, Any]] = None,
    config: Optional[Union[str, Dict[str, Any]]] = None,
    run_info: Optional[Dict[str, Any]] = None,
    output_dir: Optional[Union[str, os.PathLike]] = None,
    report_name: Optional[str] = None,
    class_names: Optional[Sequence[Any]] = None,
    history: Optional[Sequence[Dict[str, Any]]] = None,
    save_plots: Optional[bool] = None,
    reference: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Write the textual / JSON report (and optional charts) of one evaluation run.

    Returns a dictionary describing everything that was written, including the
    runtime recomputed comparison against ``project.reference_note``.
    """
    cfg = load_config(config)

    results = dict(metrics or {})
    results.setdefault("results", {})
    if isinstance(metrics, dict) and isinstance(metrics.get("results"), dict):
        results = dict(metrics["results"])

    run = dict(run_info or {})
    for key in ("model", "model_name", "training_mode", "dataset_version", "train_test_split"):
        if key not in run and kwargs.get(key) is not None:
            run[key] = kwargs[key]

    if class_names is not None:
        results.setdefault("class_names", list(class_names))
    elif results.get("class_names") is None:
        results["class_names"] = None

    if history is not None:
        results.setdefault("history", list(history))

    target_dir = output_dir
    if target_dir is None:
        target_dir = _cfg_get(cfg, "logging.report_dir", None)
    if target_dir is None:
        target_dir = _cfg_get(cfg, "experiment.output_dir", None)
    if target_dir is None:
        target_dir = DEFAULT_REPORT_DIR
    target_dir = os.fspath(target_dir)
    os.makedirs(target_dir, exist_ok=True)

    if report_name is None:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        parts = ["report"]
        for key in ("model", "model_name", "training_mode", "dataset_version", "train_test_split"):
            if run.get(key):
                parts.append(str(run[key]).replace(os.sep, "_"))
        parts.append(stamp)
        report_name = "_".join(parts)
    report_name = re.sub(r"[^\w\-.]+", "_", str(report_name))

    if reference is None:
        reference = _parse_reference_note(cfg)
    tolerance = float(
        _cfg_get(cfg, "evaluation.reference_tolerance", DEFAULT_REFERENCE_TOLERANCE)
        or DEFAULT_REFERENCE_TOLERANCE
    )
    reference_comparison = _compare_with_reference(results, reference, tolerance=tolerance)

    report_text = _format_report_text(results, reference_comparison, run)

    text_path = os.path.join(target_dir, f"{report_name}.txt")
    json_path = os.path.join(target_dir, f"{report_name}.json")

    with open(text_path, "w", encoding="utf-8") as handle:
        handle.write(report_text)
        if not report_text.endswith("\n"):
            handle.write("\n")

    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "run_info": run,
        "metrics": results,
        "reference": reference,
        "reference_comparison": reference_comparison,
    }
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, default=_json_default)

    plots: Dict[str, Optional[str]] = {
        "confusion_matrix": None,
        "metrics_bar": None,
        "history": None,
    }
    should_plot = bool(save_plots) if save_plots is not None else (plt is not None)
    if should_plot and plt is not None:
        plots["confusion_matrix"] = _plot_confusion_matrix(
            results.get("confusion_matrix") or [],
            results.get("class_names"),
            os.path.join(target_dir, f"{report_name}_confusion_matrix.png"),
        )
        plots["metrics_bar"] = _plot_metric_bars(
            results, os.path.join(target_dir, f"{report_name}_metrics.png")
        )
        plots["history"] = _plot_history(
            results.get("history") or history,
            os.path.join(target_dir, f"{report_name}_history.png"),
        )

    LOGGER.info("Report written to %s (json: %s)", text_path, json_path)
    print(report_text)

    return {
        "report_text": report_text,
        "report_path": text_path,
        "json_path": json_path,
        "plots": plots,
        "reference_comparison": reference_comparison,
        "metrics": results,
    }


# --------------------------------------------------------------------------- #
# CLI entry point (optional convenience)
# --------------------------------------------------------------------------- #
def main(argv: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    """Command-line smoke entry point: evaluate a checkpoint and write a report."""
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate a plant-disease classifier.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to a model checkpoint.")
    parser.add_argument("--config", type=str, default=None, help="Path to config/config.yaml.")
    parser.add_argument("--report-dir", type=str, default=None, help="Report output directory.")
    parser.add_argument("--model", type=str, default=None, help="Model name (alexnet / googlenet).")
    parser.add_argument("--training-mode", type=str, default=None, help="scratch / transfer.")
    args = parser.parse_args(list(argv) if argv is not None else None)

    cfg = load_config(args.config)
    report_dir = args.report_dir or _cfg_get(cfg, "logging.report_dir", DEFAULT_REPORT_DIR)

    empty_metrics: Dict[str, Any] = {
        "loss": 0.0,
        "mean_f1": 0.0,
        "mean_precision": 0.0,
        "mean_recall": 0.0,
        "overall_accuracy": 0.0,
        "accuracy": 0.0,
        "num_samples": 0,
        "confusion_matrix": [],
        "per_class": [],
        "class_names": None,
    }
    return generate_report(
        metrics=empty_metrics,
        config=cfg,
        run_info={
            "checkpoint": args.checkpoint,
            "model": args.model or _cfg_get(cfg, "model.default", None),
            "training_mode": args.training_mode or _cfg_get(cfg, "training_modes.active", None),
        },
        output_dir=report_dir,
    )


if __name__ == "__main__":  # pragma: no cover - CLI convenience
    main()