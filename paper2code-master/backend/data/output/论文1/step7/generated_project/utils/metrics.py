"""Evaluation metrics for image-based plant disease detection.

Implements FR-09: on the test set compute
    * Mean F1 score        (macro-averaged over classes)
    * mean precision       (macro-averaged over classes)
    * mean recall          (macro-averaged over classes)
    * overall accuracy     (fraction of correctly classified samples)

The module has a single public entry point: ``compute_metrics``.
It accepts raw model outputs (logits / probabilities), hard class indices or
one-hot matrices and returns a plain ``dict`` of Python floats so the values can
be logged, stored in checkpoints or compared with the numbers reported in the
paper.

Notes on conventions
--------------------
* Macro averaging ("mean") is unweighted: every class contributes equally to the
  final score, which is what "Mean F1 score / mean precision / mean recall"
  refers to in the paper.
* Classes that never appear in either predictions or targets but are declared
  through ``num_classes`` / ``class_names`` (e.g. from ``config/config.yaml``)
  are still part of the averaging set; their metrics fall back to
  ``zero_division`` so the macro average stays comparable across runs.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Sequence, Union

import torch

try:
    import yaml
except Exception:  # pragma: no cover - yaml is part of the project requirements
    yaml = None

try:
    import numpy as np
except Exception:  # pragma: no cover - numpy is optional for this module
    np = None

__all__ = ["compute_metrics"]

CONFIG_FILE_NAME = "config.yaml"
CONFIG_DIR_NAME = "config"


# --------------------------------------------------------------------------- #
# Configuration helpers
# --------------------------------------------------------------------------- #
def _candidate_config_paths() -> List[str]:
    """Return existing-file candidates for ``config/config.yaml``."""
    here = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(here)
    return [
        os.path.join(project_root, CONFIG_DIR_NAME, CONFIG_FILE_NAME),
        os.path.join(os.getcwd(), CONFIG_DIR_NAME, CONFIG_FILE_NAME),
        os.path.join(here, CONFIG_DIR_NAME, CONFIG_FILE_NAME),
        CONFIG_FILE_NAME,
    ]


def _load_config(config: Optional[Union[str, Dict[str, Any]]] = None) -> Dict[str, Any]:
    """Load the YAML configuration.

    ``config`` may be
      * an already parsed mapping (returned unchanged),
      * a path to a YAML file,
      * ``None`` -> look for ``config/config.yaml`` next to the project root.

    Missing files or unparsable content simply yield an empty mapping, so the
    metric functions remain usable in standalone unit tests.
    """
    if isinstance(config, dict):
        return config
    if isinstance(config, str):
        if os.path.isfile(config) and yaml is not None:
            try:
                with open(config, "r") as handle:
                    loaded = yaml.safe_load(handle)
            except Exception:
                return {}
            return loaded if isinstance(loaded, dict) else {}
        return {}
    if yaml is None:
        return {}
    for path in _candidate_config_paths():
        if not os.path.isfile(path):
            continue
        try:
            with open(path, "r") as handle:
                loaded = yaml.safe_load(handle)
        except Exception:
            continue
        if isinstance(loaded, dict):
            return loaded
    return {}


def _cfg_get(config: Dict[str, Any], keys: Sequence[str], default: Any = None) -> Any:
    """Safely walk nested dictionaries; every missing key yields ``default``."""
    node: Any = config
    for key in keys:
        if not isinstance(node, dict):
            return default
        if key not in node:
            return default
        node = node[key]
    return node


# --------------------------------------------------------------------------- #
# Tensor helpers
# --------------------------------------------------------------------------- #
def _to_tensor(value: Any) -> torch.Tensor:
    """Convert list / tuple / numpy array / scalar / tensor to a CPU tensor."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if np is not None and isinstance(value, np.ndarray):
        return torch.as_tensor(value).cpu()
    if isinstance(value, (list, tuple)):
        return torch.as_tensor(value).cpu()
    if isinstance(value, (int, float, bool)):
        return torch.tensor([value]).cpu()
    raise TypeError(
        "Unsupported input type for metrics: {}".format(type(value).__name__)
    )


def _default_sample() -> tuple:
    """Tiny deterministic sample so ``compute_metrics()`` can be smoke-tested."""
    predictions = torch.tensor(
        [
            [0.10, 0.80, 0.10],
            [0.90, 0.05, 0.05],
            [0.20, 0.20, 0.60],
            [0.70, 0.20, 0.10],
            [0.05, 0.15, 0.80],
        ],
        dtype=torch.float32,
    )
    targets = torch.tensor([1, 0, 2, 0, 2], dtype=torch.long)
    return predictions, targets


def _labels_from_predictions(pred: torch.Tensor, threshold: float) -> torch.Tensor:
    """Turn raw model outputs into a 1-D tensor of integer class indices."""
    if pred.dim() == 0:
        pred = pred.unsqueeze(0)
    if pred.dim() == 1:
        if pred.is_floating_point():
            integral = bool(torch.all(pred == torch.round(pred)).item())
            if not integral:
                return (pred > float(threshold)).long()
        return pred.long()
    if pred.dim() == 2:
        if pred.size(1) == 1:
            column = pred[:, 0]
            if column.is_floating_point():
                return (column > float(threshold)).long()
            return column.long()
        return torch.argmax(pred, dim=1).long()
    raise ValueError(
        "`predictions` must be 1-D or 2-D, got shape {}".format(tuple(pred.shape))
    )


def _labels_from_targets(target: torch.Tensor) -> torch.Tensor:
    """Turn integer labels / one-hot matrices into a 1-D tensor of class indices."""
    if target.dim() == 0:
        target = target.unsqueeze(0)
    if target.dim() == 1:
        return target.long()
    if target.dim() == 2:
        if target.size(1) == 1:
            return target[:, 0].long()
        return torch.argmax(target, dim=1).long()
    raise ValueError(
        "`targets` must be 1-D or 2-D, got shape {}".format(tuple(target.shape))
    )


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def compute_metrics(
    predictions: Optional[Any] = None,
    targets: Optional[Any] = None,
    config: Optional[Union[str, Dict[str, Any]]] = None,
    class_names: Optional[Sequence[Any]] = None,
    num_classes: Optional[int] = None,
    threshold: float = 0.5,
    zero_division: float = 0.0,
) -> Dict[str, float]:
    """Compute Mean F1 score, mean precision, mean recall and accuracy (FR-09).

    Parameters
    ----------
    predictions:
        Model outputs. Accepted forms:
          * 2-D tensor of shape ``(N, C)`` with logits / probabilities,
          * 1-D tensor of shape ``(N,)`` with class indices (or binary scores,
            which are binarised with ``threshold``),
          * 2-D tensor of shape ``(N, 1)`` with binary scores.
        If ``None`` together with ``targets``, a small built-in sample is used so
        the function can be called without arguments.
    targets:
        Ground-truth labels. Accepted forms: 1-D class indices, ``(N, 1)``
        labels, or one-hot / multi-label style ``(N, C)`` matrices.
    config:
        Either an already parsed mapping or a path to ``config/config.yaml``.
        Used only to resolve ``num_classes`` / ``class_names`` when the caller
        does not pass them explicitly. All lookups are done with safe defaults.
    class_names:
        Optional list of class names; its length defines the number of classes
        when ``num_classes`` is not given.
    num_classes:
        Optional explicit class count. When omitted it is read from the config
        (``model.num_classes`` / ``data.num_classes`` / ``num_classes``) or
        inferred from the observed labels.
    threshold:
        Decision threshold used for binary (2-column or 1-D float) outputs.
    zero_division:
        Value used for per-class precision / recall when the corresponding
        denominator is zero (classes with no predicted or no true samples).

    Returns
    -------
    dict
        ``mean_f1``, ``mean_precision``, ``mean_recall``, ``overall_accuracy``
        (all Python floats), plus the diagnostic metadata ``num_samples`` and
        ``num_classes``.
    """
    cfg = _load_config(config)

    # --- inputs -----------------------------------------------------------
    if predictions is None and targets is None:
        predictions, targets = _default_sample()
    if predictions is None or targets is None:
        raise ValueError(
            "`predictions` and `targets` must both be provided (or both omitted)."
        )

    pred_raw = _to_tensor(predictions)
    true_raw = _to_tensor(targets)

    pred_labels = _labels_from_predictions(pred_raw, float(threshold))
    true_labels = _labels_from_targets(true_raw)

    if pred_labels.numel() == 0 or true_labels.numel() == 0:
        raise ValueError("`predictions` and `targets` must not be empty.")
    if pred_labels.shape[0] != true_labels.shape[0]:
        raise ValueError(
            "predictions and targets must have the same number of samples, "
            "got {} and {}".format(pred_labels.shape[0], true_labels.shape[0])
        )

    zero_division = float(zero_division)

    # --- class list -------------------------------------------------------
    if class_names is None:
        class_names = _cfg_get(cfg, ("data", "class_names"), None)
        if class_names is None:
            class_names = _cfg_get(cfg, ("class_names",), None)

    if num_classes is None:
        num_classes = _cfg_get(cfg, ("model", "num_classes"), None)
    if num_classes is None:
        num_classes = _cfg_get(cfg, ("data", "num_classes"), None)
    if num_classes is None:
        num_classes = _cfg_get(cfg, ("num_classes",), None)
    if num_classes is None and class_names is not None:
        try:
            num_classes = len(class_names)
        except TypeError:
            num_classes = None

    observed: List[int] = [
        int(label)
        for label in torch.unique(torch.cat([true_labels, pred_labels])).tolist()
    ]

    class_ids: List[int] = []
    if num_classes is not None:
        n_classes = int(num_classes)
        if n_classes <= 0:
            raise ValueError("`num_classes` must be a positive integer.")
        class_ids = list(range(n_classes))
    for label in observed:
        if label not in class_ids:
            class_ids.append(label)
    class_ids.sort()

    # --- per-class statistics --------------------------------------------
    precisions: List[float] = []
    recalls: List[float] = []
    f1_scores: List[float] = []

    for class_id in class_ids:
        pred_pos = pred_labels == class_id
        true_pos = true_labels == class_id

        tp = float(torch.sum(pred_pos & true_pos).item())
        fp = float(torch.sum(pred_pos & ~true_pos).item())
        fn = float(torch.sum(~pred_pos & true_pos).item())

        precision = tp / (tp + fp) if (tp + fp) > 0 else zero_division
        recall = tp / (tp + fn) if (tp + fn) > 0 else zero_division
        if (precision + recall) > 0:
            f1 = 2.0 * precision * recall / (precision + recall)
        else:
            f1 = 0.0

        precisions.append(float(precision))
        recalls.append(float(recall))
        f1_scores.append(float(f1))

    # --- macro averages and overall accuracy -----------------------------
    n_classes_eval = len(class_ids)
    if n_classes_eval > 0:
        mean_precision = float(sum(precisions) / n_classes_eval)
        mean_recall = float(sum(recalls) / n_classes_eval)
        mean_f1 = float(sum(f1_scores) / n_classes_eval)
    else:
        mean_precision = 0.0
        mean_recall = 0.0
        mean_f1 = 0.0

    num_samples = int(true_labels.numel())
    correct = int(torch.sum(pred_labels == true_labels).item())
    overall_accuracy = float(correct / num_samples) if num_samples > 0 else 0.0

    return {
        "mean_f1": mean_f1,
        "mean_precision": mean_precision,
        "mean_recall": mean_recall,
        "overall_accuracy": overall_accuracy,
        "num_samples": float(num_samples),
        "num_classes": float(n_classes_eval),
    }