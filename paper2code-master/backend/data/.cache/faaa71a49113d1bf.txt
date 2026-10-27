"""Evaluation metrics for the Temporal Fusion Transformer (TFT).

Implements the normalized quantile loss (q-Risk) reported in
"Temporal Fusion Transformers for Interpretable Multihorizon Time Series
Forecasting" (Lim et al., 2021).

The quantile loss is defined as
    QL(y, y_hat, q) = q * (y - y_hat)_+ + (1 - q) * (y_hat - y)_+
and the normalized quantile loss (q-Risk) of a quantile level q is
    q-Risk(q) = sum_t QL(y_t, y_hat_t, q) / sum_t |y_t|

where the sums run over every forecast horizon and every sample that is
valid (finite) in both the target and the forecast.

All tunable parameters (quantile levels, numerical epsilon, ...) are read
from ``config/config.yaml`` whenever they are present; otherwise safe
defaults are used so the module can be exercised without a config file.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch

try:  # PyYAML is optional at import time to keep the module lightweight.
    import yaml
except Exception:  # pragma: no cover - yaml missing or broken install
    yaml = None


DEFAULT_CONFIG_PATH = os.path.join("config", "config.yaml")
DEFAULT_QUANTILES = (0.5, 0.9)
DEFAULT_EPS = 1e-8

ConfigLike = Union[None, str, Dict[str, Any]]
TensorLike = Any


# ---------------------------------------------------------------------------
# Configuration helpers
# ---------------------------------------------------------------------------
def load_config(config: ConfigLike = None) -> Dict[str, Any]:
    """Load a configuration mapping.

    Parameters
    ----------
    config:
        ``None`` -> read ``config/config.yaml`` (if it exists).
        ``str``  -> path to a YAML file.
        ``dict`` -> used as-is.

    Returns
    -------
    dict
        The configuration mapping (possibly empty on failure).
    """
    if isinstance(config, dict):
        return config

    cfg_path: Optional[str] = None
    if isinstance(config, str) and config.strip() != "":
        cfg_path = config
    else:
        cfg_path = os.environ.get("TFT_CONFIG_PATH", DEFAULT_CONFIG_PATH)

    if cfg_path is None or not isinstance(cfg_path, str):
        return {}

    if not os.path.isfile(cfg_path):
        # Fall back to <project_root>/config/config.yaml
        here = os.path.dirname(os.path.abspath(__file__))
        root = os.path.dirname(here)
        alternative = os.path.join(root, "config", "config.yaml")
        if os.path.isfile(alternative):
            cfg_path = alternative
        else:
            return {}

    if yaml is None:
        return {}

    try:
        with open(cfg_path, "r", encoding="utf-8") as handle:
            loaded = yaml.safe_load(handle)
        return loaded if isinstance(loaded, dict) else {}
    except Exception:
        return {}


def _get_nested(config: Any, keys: Sequence[str], default: Any = None) -> Any:
    """Safely read ``config[k1][k2]...`` returning ``default`` if absent."""
    node = config
    for key in keys:
        if not isinstance(node, dict) or key not in node:
            return default
        node = node[key]
    return node


def _resolve_quantiles(config: Any, quantiles: Any = None) -> List[float]:
    """Return the list of quantile levels to evaluate.

    Priority: explicit argument > known config locations > default (0.5, 0.9).
    """
    resolved: Optional[List[float]] = None

    if quantiles is not None:
        if isinstance(quantiles, (int, float)) and not isinstance(quantiles, bool):
            resolved = [float(quantiles)]
        else:
            resolved = [float(q) for q in quantiles]

    if resolved is None:
        candidates = (
            ("quantiles",),
            ("model", "quantiles"),
            ("training", "quantiles"),
            ("metrics", "quantiles"),
            ("data", "quantiles"),
        )
        for path in candidates:
            value = _get_nested(config, path, None)
            if isinstance(value, (list, tuple)) and len(value) > 0:
                try:
                    resolved = [float(q) for q in value]
                except (TypeError, ValueError):
                    resolved = None
                if resolved is not None:
                    break
            elif isinstance(value, (int, float)) and not isinstance(value, bool):
                resolved = [float(value)]
                break

    if resolved is None:
        resolved = [float(q) for q in DEFAULT_QUANTILES]

    if len(resolved) == 0:
        resolved = [float(q) for q in DEFAULT_QUANTILES]

    for q in resolved:
        if not (0.0 < q < 1.0):
            raise ValueError(f"Quantile level must lie in (0, 1); got {q!r}.")
    return resolved


def _resolve_eps(config: Any, eps: Optional[float] = None) -> float:
    """Return the numerical stabilizer as a plain float (never ``None``)."""
    if eps is not None:
        try:
            return float(eps)
        except (TypeError, ValueError):
            return float(DEFAULT_EPS)
    value = _get_nested(config, ("metrics", "eps"), None)
    if value is None:
        value = _get_nested(config, ("training", "eps"), None)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(DEFAULT_EPS)


# ---------------------------------------------------------------------------
# Tensor helpers
# ---------------------------------------------------------------------------
def _to_float_tensor(data: TensorLike) -> torch.Tensor:
    """Convert tensors / arrays / nested sequences into a float32 tensor."""
    if isinstance(data, torch.Tensor):
        tensor = data.detach()
    elif isinstance(data, np.ndarray):
        tensor = torch.from_numpy(np.ascontiguousarray(data))
    elif isinstance(data, (tuple, list)):
        if len(data) > 0 and isinstance(data[0], torch.Tensor):
            tensor = data[0].detach()
        else:
            tensor = torch.as_tensor(np.asarray(data, dtype=np.float64))
    else:
        tensor = torch.as_tensor(data)
    tensor = tensor.to(torch.float32).contiguous()
    return tensor


def _extract_forecast(data: TensorLike) -> TensorLike:
    """Accept raw forecasts or a container dict with a forecast payload."""
    if isinstance(data, dict):
        for key in ("predictions", "prediction", "y_hat", "y_pred",
                    "forecast", "quantiles", "outputs", "output"):
            if key in data:
                return data[key]
        raise ValueError(
            "Could not find a forecast tensor inside the provided mapping; "
            "expected one of the keys "
            "['predictions', 'prediction', 'y_hat', 'y_pred', 'forecast', 'quantiles']."
        )
    return data


def _extract_target(data: TensorLike) -> TensorLike:
    """Accept raw targets or a container dict with a target payload."""
    if isinstance(data, dict):
        for key in ("targets", "target", "y_true", "y", "labels", "values"):
            if key in data:
                return data[key]
        raise ValueError(
            "Could not find a target tensor inside the provided mapping; "
            "expected one of the keys "
            "['targets', 'target', 'y_true', 'y', 'labels', 'values']."
        )
    return data


def _select_quantile_prediction(
    predictions: torch.Tensor,
    quantile: float,
    quantile_grid: Optional[Sequence[float]] = None,
) -> torch.Tensor:
    """Return the forecast slice that corresponds to ``quantile``.

    * 2-D input (batch, horizon)            -> returned unchanged.
    * 3-D input (batch, horizon, n_quant)   -> slice along the last axis.
    """
    if predictions.dim() <= 2:
        return predictions

    n_quantiles = int(predictions.shape[-1])
    if n_quantiles == 1:
        return predictions[..., 0]

    grid: Optional[np.ndarray] = None
    if quantile_grid is not None and len(quantile_grid) == n_quantiles:
        try:
            grid = np.asarray([float(q) for q in quantile_grid], dtype=np.float64)
        except (TypeError, ValueError):
            grid = None
    if grid is None:
        # Fallback grid: evenly spaced levels centred away from 0 and 1.
        grid = np.linspace(0.5 / n_quantiles, 1.0 - 0.5 / n_quantiles, n_quantiles)

    index = int(np.argmin(np.abs(grid - float(quantile))))
    return predictions[..., index]


def _align_prediction_target(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    quantile_grid: Sequence[float],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Swap arguments if needed and validate the resulting shapes.

    The first positional argument is interpreted as the forecast and the
    second one as the target, but the pair is swapped automatically when the
    shape signature unambiguously indicates the opposite order.
    """
    n_quantiles = len(quantile_grid)

    if predictions.shape == targets.shape:
        return predictions, targets

    if predictions.dim() == targets.dim() + 1 and tuple(predictions.shape[:-1]) == tuple(targets.shape):
        return predictions, targets

    if targets.dim() == predictions.dim() + 1 and tuple(targets.shape[:-1]) == tuple(predictions.shape):
        return targets, predictions

    if (
        predictions.dim() == targets.dim()
        and predictions.dim() > 0
        and int(targets.shape[-1]) == n_quantiles
        and int(predictions.shape[-1]) != n_quantiles
    ):
        return targets, predictions

    raise ValueError(
        "Incompatible forecast/target shapes: "
        f"predictions={tuple(predictions.shape)}, targets={tuple(targets.shape)}. "
        "Expected targets shaped (batch, horizon) and predictions shaped "
        "(batch, horizon) or (batch, horizon, n_quantiles)."
    )


def _dummy_forecast(
    quantile_grid: Sequence[float],
    n_samples: int = 8,
    horizon: int = 4,
    seed: int = 42,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Small deterministic forecast/target pair used for smoke tests."""
    generator = torch.Generator()
    generator.manual_seed(int(seed))

    base = torch.randn(n_samples, horizon, generator=generator)
    noise = 0.1 * torch.randn(n_samples, horizon, generator=generator)
    targets = base + noise

    slices: List[torch.Tensor] = []
    for index, _q in enumerate(quantile_grid):
        offset = (index - (len(quantile_grid) - 1) / 2.0) * 0.05
        slices.append(base + offset)
    predictions = torch.stack(slices, dim=-1)

    return predictions, targets


# ---------------------------------------------------------------------------
# Metric primitives
# ---------------------------------------------------------------------------
def quantile_loss(
    y_true: TensorLike,
    y_pred: TensorLike,
    q: float = 0.5,
    mask: Optional[TensorLike] = None,
    reduction: str = "sum",
) -> torch.Tensor:
    """Pinball (quantile) loss  QL(y, y_hat, q).

    Parameters
    ----------
    y_true, y_pred:
        Broadcastable tensors / arrays of identical logical shape.
    q:
        Quantile level in (0, 1).
    mask:
        Optional boolean / float validity mask.
    reduction:
        ``'sum'`` (default), ``'mean'`` or ``'none'``.
    """
    quantile = float(q)
    if not (0.0 < quantile < 1.0):
        raise ValueError(f"q must lie in (0, 1); got {q!r}.")

    true_t = _to_float_tensor(y_true)
    pred_t = _to_float_tensor(y_pred)
    true_t, pred_t = torch.broadcast_tensors(true_t, pred_t)

    valid = torch.isfinite(true_t) & torch.isfinite(pred_t)
    if mask is not None:
        mask_t = _to_float_tensor(mask).to(torch.bool)
        valid_b, mask_b = torch.broadcast_tensors(valid, mask_t)
        valid = valid_b & mask_b

    diff = true_t - pred_t
    elementwise = (
        quantile * torch.clamp(diff, min=0.0)
        + (1.0 - quantile) * torch.clamp(-diff, min=0.0)
    )
    elementwise = torch.where(valid, elementwise, torch.zeros_like(elementwise))

    if reduction == "none":
        return elementwise
    if reduction == "mean":
        count = torch.clamp(valid.to(elementwise.dtype).sum(), min=1.0)
        return elementwise.sum() / count
    return elementwise.sum()


def normalized_quantile_loss(
    y_true: TensorLike,
    y_pred: TensorLike,
    q: float = 0.5,
    mask: Optional[TensorLike] = None,
    eps: float = DEFAULT_EPS,
) -> float:
    """Normalized quantile loss (q-Risk).

    q-Risk(q) = sum_t QL(y_t, y_hat_t, q) / sum_t |y_t|
    """
    quantile = float(q)
    if not (0.0 < quantile < 1.0):
        raise ValueError(f"q must lie in (0, 1); got {q!r}.")

    true_t = _to_float_tensor(y_true)
    pred_t = _to_float_tensor(y_pred)
    true_t, pred_t = torch.broadcast_tensors(true_t, pred_t)

    valid = torch.isfinite(true_t) & torch.isfinite(pred_t)
    if mask is not None:
        mask_t = _to_float_tensor(mask).to(torch.bool)
        valid_b, mask_b = torch.broadcast_tensors(valid, mask_t)
        valid = valid_b & mask_b

    if not bool(valid.any()):
        return float("nan")

    valid_true = true_t[valid]
    valid_pred = pred_t[valid]

    diff = valid_true - valid_pred
    total_loss = (
        quantile * torch.clamp(diff, min=0.0)
        + (1.0 - quantile) * torch.clamp(-diff, min=0.0)
    ).sum()
    normalizer = valid_true.abs().sum()

    normalizer_value = float(normalizer.item())
    if normalizer_value <= float(eps):
        total_loss_value = float(total_loss.item())
        return 0.0 if total_loss_value <= float(eps) else float("inf")

    return float(total_loss.item() / normalizer_value)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def compute_metrics(
    predictions: TensorLike = None,
    targets: TensorLike = None,
    config: ConfigLike = None,
    quantiles: Any = None,
    y_true: TensorLike = None,
    y_pred: TensorLike = None,
    mask: Optional[TensorLike] = None,
    eps: Optional[float] = None,
) -> Dict[str, float]:
    """Compute the evaluation metrics required by FR-08.

    Specifically the normalized quantile loss (q-Risk) for the P50 and P90
    quantile forecasts on a test set.

    Parameters
    ----------
    predictions:
        Forecast tensor of shape ``(batch, horizon)`` (single quantile) or
        ``(batch, horizon, n_quantiles)``. Also accepts a dict containing one
        of the keys ``'predictions'``, ``'y_hat'``, ``'forecast'``, ... .
    targets:
        Ground-truth tensor of shape ``(batch, horizon)`` (broadcastable with
        the forecasts). Also accepts a dict containing ``'targets'`` / ``'y_true'``.
    config:
        ``None`` (read ``config/config.yaml``), a dict, or a path to a YAML file.
    quantiles:
        Iterable of quantile levels. Defaults to the quantile levels declared in
        the config, or ``(0.5, 0.9)``.
    y_true, y_pred:
        Keyword aliases for ``targets`` / ``predictions`` (sklearn-style calls).
    mask:
        Optional validity mask broadcastable to the target shape.
    eps:
        Numerical stabilizer for the normalization; defaults to the config value
        or ``1e-8``.

    Returns
    -------
    dict
        ``{'qRisk_P50': float, 'qRisk_P90': float}`` (plus one entry per extra
        requested quantile level).
    """
    if y_true is not None:
        targets = y_true
    if y_pred is not None:
        predictions = y_pred

    cfg = load_config(config)
    quantile_grid = _resolve_quantiles(cfg, quantiles)
    epsilon = _resolve_eps(cfg, eps)

    if predictions is None or targets is None:
        predictions, targets = _dummy_forecast(quantile_grid)

    pred_raw = _extract_forecast(predictions)
    true_raw = _extract_target(targets)

    pred_t = _to_float_tensor(pred_raw)
    true_t = _to_float_tensor(true_raw)

    pred_t, true_t = _align_prediction_target(pred_t, true_t, quantile_grid)

    results: Dict[str, float] = {}
    for quantile in quantile_grid:
        pred_q = _select_quantile_prediction(pred_t, float(quantile), quantile_grid)
        value = normalized_quantile_loss(
            true_t,
            pred_q,
            q=float(quantile),
            mask=mask,
            eps=epsilon,
        )
        label = f"qRisk_P{int(round(float(quantile) * 100))}"
        results[label] = float(value)

    return results


if __name__ == "__main__":  # minimal smoke test (no arguments required)
    smoke = compute_metrics()
    for key in sorted(smoke.keys()):
        print(f"{key}: {smoke[key]:.6f}")