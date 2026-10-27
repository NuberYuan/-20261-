"""Evaluation metrics for UNet++ segmentation (FR-06).

Implements the metric computation required on the test set:
    * Dice coefficient
    * Intersection over Union (IoU)

Public entry point: :func:`compute_metrics`.
"""

from __future__ import annotations

import os
import warnings
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch
import yaml

DEFAULT_THRESHOLD = 0.5
DEFAULT_SMOOTH = 1e-7

_CONFIG_SEARCH_PATHS = (
    os.path.join("config", "config.yaml"),
    os.path.join("configs", "config.yaml"),
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "config",
        "config.yaml",
    ),
)


def _load_config(config: Optional[Union[Dict[str, Any], str]] = None) -> Dict[str, Any]:
    """Load ``config/config.yaml`` safely and return it as a dictionary."""
    if config is None:
        for path in _CONFIG_SEARCH_PATHS:
            if os.path.isfile(path):
                try:
                    with open(path, "r", encoding="utf-8") as handle:
                        loaded = yaml.safe_load(handle)
                    return loaded if isinstance(loaded, dict) else {}
                except (OSError, yaml.YAMLError):
                    continue
        return {}
    if isinstance(config, dict):
        return config
    if isinstance(config, str):
        if os.path.isfile(config):
            try:
                with open(config, "r", encoding="utf-8") as handle:
                    loaded = yaml.safe_load(handle)
                return loaded if isinstance(loaded, dict) else {}
            except (OSError, yaml.YAMLError):
                return {}
        return {}
    return {}


def _cfg_get(config: Dict[str, Any], path: Tuple[str, ...], default: Any) -> Any:
    """Safely walk ``path`` inside ``config`` and return ``default`` if missing."""
    node = config
    for key in path:
        if isinstance(node, dict) and key in node:
            node = node[key]
        else:
            return default
    return default if node is None else node


def _to_float_tensor(data: Any) -> torch.Tensor:
    """Convert numpy arrays / lists / tensors into a detached float32 tensor."""
    if isinstance(data, np.ndarray):
        tensor = torch.from_numpy(np.ascontiguousarray(data))
    elif torch.is_tensor(data):
        tensor = data
    else:
        tensor = torch.as_tensor(data)
    tensor = tensor.detach()
    if not torch.is_floating_point(tensor):
        tensor = tensor.float()
    else:
        tensor = tensor.float()
    return tensor.contiguous()


def _binarize_prediction(tensor: torch.Tensor, threshold: float) -> torch.Tensor:
    """Turn a probability / logit map into a binary mask."""
    if tensor.numel() == 0:
        return (tensor > 0.5).float()
    min_value = float(tensor.min().item())
    max_value = float(tensor.max().item())
    if min_value < 0.0 or max_value > 1.0:
        tensor = torch.sigmoid(tensor)
    return (tensor >= threshold).float()


def _dice_iou_binary(
    pred_bin: torch.Tensor,
    target_bin: torch.Tensor,
    smooth: float,
) -> Tuple[float, float]:
    """Compute Dice coefficient and IoU for a pair of aligned binary masks."""
    if pred_bin.numel() == 0 or target_bin.numel() == 0:
        return 0.0, 0.0

    pred = pred_bin.float()
    target = target_bin.float()

    intersection = torch.sum(pred * target)
    pred_sum = torch.sum(pred)
    target_sum = torch.sum(target)
    union = pred_sum + target_sum - intersection

    dice = (2.0 * intersection + smooth) / (pred_sum + target_sum + smooth)
    iou = (intersection + smooth) / (union + smooth)
    return float(dice.item()), float(iou.item())


def _align_shapes(pred: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Squeeze redundant singleton channel axes so both tensors match."""
    if pred.shape == target.shape:
        return pred, target

    if (
        pred.dim() == target.dim() + 1
        and pred.dim() in (4, 5)
        and pred.shape[1] == 1
        and pred.shape[0] == target.shape[0]
        and pred.shape[2:] == target.shape[1:]
    ):
        return pred.squeeze(1), target

    if (
        target.dim() == pred.dim() + 1
        and target.dim() in (4, 5)
        and target.shape[1] == 1
        and target.shape[0] == pred.shape[0]
        and target.shape[2:] == pred.shape[1:]
    ):
        return pred, target.squeeze(1)

    if (
        pred.dim() == target.dim() + 1
        and pred.dim() in (4, 5)
        and pred.shape[1] > 1
        and pred.shape[0] == target.shape[0]
        and pred.shape[2:] == target.shape[1:]
    ):
        # per-class prediction maps (N, C, ...) with integer label targets (N, ...)
        return pred, target

    if pred.shape != target.shape:
        raise ValueError(
            "predictions and targets must have compatible shapes, got "
            f"{tuple(pred.shape)} and {tuple(target.shape)}"
        )
    return pred, target


def compute_metrics(
    predictions: Optional[Any] = None,
    targets: Optional[Any] = None,
    threshold: float = DEFAULT_THRESHOLD,
    config: Optional[Union[Dict[str, Any], str]] = None,
    smooth: float = DEFAULT_SMOOTH,
) -> Dict[str, float]:
    """Compute the Dice coefficient and the Intersection over Union (IoU).

    Parameters
    ----------
    predictions:
        Model outputs. Accepted forms:
        * binary masks / probability maps of shape ``(N, H, W)`` or ``(N, 1, H, W)``
        * multi-class score maps of shape ``(N, C, H, W)`` with ``C > 1``
        numpy arrays, lists and torch tensors are all supported.
    targets:
        Ground-truth masks with a shape compatible with ``predictions``
        (integer label maps are supported for the multi-class case).
    threshold:
        Binarisation threshold applied to probability / logit maps (default 0.5).
    config:
        Optional mapping or path to ``config/config.yaml``. The keys
        ``threshold`` and ``smooth`` are read with safe defaults when present.
    smooth:
        Smoothing constant used to keep the divisions numerically stable.

    Returns
    -------
    dict
        ``{"dice": float, "iou": float}``.

    Notes
    -----
    The Dice coefficient is ``(2 * |P ∩ G|) / (|P| + |G|)`` and IoU is
    ``|P ∩ G| / |P ∪ G|``, both averaged over the whole evaluation set
    (micro average for binary masks, macro average over classes for the
    multi-class case).
    """
    cfg = _load_config(config)
    threshold = float(_cfg_get(cfg, ("threshold",), float(threshold)))
    smooth = float(_cfg_get(cfg, ("smooth",), float(smooth)))

    if predictions is None or targets is None:
        warnings.warn(
            "compute_metrics() was called without predictions/targets; "
            "returning zero metrics as a placeholder.",
            RuntimeWarning,
        )
        return {"dice": 0.0, "iou": 0.0}

    pred = _to_float_tensor(predictions)
    target = _to_float_tensor(targets)
    pred, target = _align_shapes(pred, target)

    # ---- multi-class case: batched channel-first tensors with C > 1 ----
    if pred.dim() in (4, 5) and pred.shape[1] > 1:
        num_classes = int(pred.shape[1])
        pred_idx = pred.argmax(dim=1)

        if target.dim() == pred.dim() and target.shape[1] == num_classes:
            target_idx = target.argmax(dim=1)
        elif (
            target.dim() == pred.dim() - 1
            and target.shape[0] == pred.shape[0]
            and target.shape[1:] == pred.shape[2:]
        ):
            target_idx = target
        else:
            raise ValueError(
                "multi-class targets must be either one-hot/probability maps "
                f"of shape {tuple(pred.shape)} or label maps of shape "
                f"{tuple(pred.shape[0:1] + pred.shape[2:])}, got {tuple(target.shape)}"
            )

        dice_values = []
        iou_values = []
        for class_idx in range(num_classes):
            pred_bin = (pred_idx == class_idx).float()
            target_bin = (target_idx == class_idx).float()
            dice_value, iou_value = _dice_iou_binary(pred_bin, target_bin, smooth)
            dice_values.append(dice_value)
            iou_values.append(iou_value)

        return {
            "dice": float(sum(dice_values) / max(len(dice_values), 1)),
            "iou": float(sum(iou_values) / max(len(iou_values), 1)),
        }

    # ---- binary case: drop singleton channel axes ----
    if pred.dim() in (4, 5) and pred.shape[1] == 1:
        pred = pred.squeeze(1)
    if target.dim() in (4, 5) and target.shape[1] == 1:
        target = target.squeeze(1)

    if pred.shape != target.shape:
        raise ValueError(
            "predictions and targets must have the same shape in the binary case, "
            f"got {tuple(pred.shape)} and {tuple(target.shape)}"
        )

    pred_bin = _binarize_prediction(pred, threshold)
    target_bin = (target > 0.5).float()

    dice, iou = _dice_iou_binary(pred_bin, target_bin, smooth)
    return {"dice": float(dice), "iou": float(iou)}


if __name__ == "__main__":
    torch.manual_seed(0)

    # basic smoke test with random binary masks
    _pred = (torch.rand(4, 1, 64, 64) > 0.5).float()
    _target = (torch.rand(4, 1, 64, 64) > 0.5).float()
    print(compute_metrics(_pred, _target))
    print(compute_metrics())
    print(compute_metrics(torch.rand(2, 3, 32, 32), torch.randint(0, 3, (2, 32, 32))))