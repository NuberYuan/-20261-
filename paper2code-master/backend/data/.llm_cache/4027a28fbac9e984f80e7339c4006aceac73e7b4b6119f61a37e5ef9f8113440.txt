"""Evaluation metrics for TFT time-series forecasting.

The implementation follows the TFT paper's evaluation protocol.  For a
quantile q, the normalized quantile loss is::

    sum_{samples, horizons} QL(y, yhat_q, q) / sum_{samples, horizons} |y|

and the corresponding q-risk is twice that value.  P50 and P90 risk are the
q-risks at q = 0.5 and q = 0.9 respectively.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

import numpy as np
import torch
import yaml

__all__ = ["compute_metrics"]

DEFAULT_QUANTILES = (0.1, 0.5, 0.9)


def _load_config(config: Union[str, Path, Dict, None]) -> Dict:
    """Load a config from a path or dict, returning {} when unavailable."""
    if config is None:
        return {}

    if isinstance(config, dict):
        return config

    if isinstance(config, (str, Path)):
        config_path = Path(config)
        if not config_path.exists():
            # Fall back to the conventional project config location.
            config_path = Path("config") / "config.yaml"
        if config_path.exists():
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    loaded = yaml.safe_load(f)
                    return loaded if isinstance(loaded, dict) else {}
            except Exception:
                return {}

    return {}


def _to_tensor(x) -> torch.Tensor:
    """Convert a numpy array, list, or tensor to a detached float tensor."""
    if torch.is_tensor(x):
        return x.detach().float()
    return torch.as_tensor(np.asarray(x), dtype=torch.float32)


def _prepare_tensors(y_true, y_pred) -> tuple[torch.Tensor, torch.Tensor]:
    """Normalize shapes to ``(N, H)`` and ``(N, H, Q)``.

    ``y_true`` should contain actual values, and ``y_pred`` should contain
    quantile predictions with the quantile dimension last.
    """
    y_true = _to_tensor(y_true)

    # If predictions are provided as a list/tuple of quantile arrays, stack
    # them along a new last dimension.
    if isinstance(y_pred, (list, tuple)):
        parts = []
        for p in y_pred:
            p = _to_tensor(p)
            # Remove trailing singleton dimensions that do not represent
            # quantiles (e.g. shape (N, H, 1) -> (N, H)).
            while p.dim() > 2 and p.size(-1) == 1:
                p = p.squeeze(-1)
            parts.append(p)
        y_pred = torch.stack(parts, dim=-1)
    else:
        y_pred = _to_tensor(y_pred)

    # Remove trailing singleton dimensions from ground truth only.
    while y_true.dim() > 2 and y_true.size(-1) == 1:
        y_true = y_true.squeeze(-1)

    # If y_true accidentally has the same trailing dimension as the quantile
    # dimension, it is probably duplicated ground truth; average it away.
    if y_true.dim() == y_pred.dim():
        if y_true.size(-1) == y_pred.size(-1):
            if y_true.size(-1) > 1:
                y_true = y_true.mean(dim=-1)
            else:
                y_true = y_true.squeeze(-1)

    # Ensure y_pred is 3D: (batch, horizon, num_quantiles).
    if y_pred.dim() == 1:
        y_pred = y_pred.unsqueeze(0).unsqueeze(-1)
    elif y_pred.dim() == 2:
        y_pred = y_pred.unsqueeze(-1)

    # Make y_true at least 2D before flattening.
    if y_true.dim() == 1:
        y_true = y_true.unsqueeze(-1)

    y_true_flat = y_true.reshape(-1)
    y_pred_flat = y_pred.reshape(-1, y_pred.size(-1))
    return y_true_flat, y_pred_flat


def _get_quantiles(config, quantiles: Optional[Sequence[float]]) -> List[float]:
    """Resolve quantiles from arguments, config, or defaults."""
    if quantiles is not None:
        return [float(q) for q in quantiles]

    cfg = _load_config(config)
    if isinstance(cfg, dict):
        # Safe direct access, falling back to common config sections.
        if "quantiles" in cfg:
            return [float(q) for q in cfg["quantiles"]]
        for section in ("model", "evaluation", "metrics"):
            section_cfg = cfg.get(section, {})
            if isinstance(section_cfg, dict) and "quantiles" in section_cfg:
                return [float(q) for q in section_cfg["quantiles"]]

    return list(DEFAULT_QUANTILES)


def _resolve_quantiles(
    config, quantiles: Optional[Sequence[float]], num_quantiles: int
) -> List[float]:
    """Return quantile values aligned with the prediction tensor."""
    resolved = _get_quantiles(config, quantiles)

    if num_quantiles == 1:
        # A single predicted quantile is conventionally the median.
        return [0.5]

    if len(resolved) == num_quantiles:
        return resolved

    if len(resolved) >= num_quantiles:
        return resolved[:num_quantiles]

    if num_quantiles == 3:
        return list(DEFAULT_QUANTILES)

    raise ValueError(
        f"Expected {num_quantiles} quantile values, got {len(resolved)}"
    )


def _quantile_loss(y_true: torch.Tensor, y_pred: torch.Tensor, q: float) -> torch.Tensor:
    """Elementwise quantile loss."""
    diff = y_true - y_pred
    # q * (y - yhat)_+ + (1 - q) * (yhat - y)_+
    return torch.where(diff >= 0, q * diff, (q - 1.0) * diff)


def _nearest_quantile_index(quantiles: Sequence[float], target: float) -> int:
    """Return the index of the quantile closest to ``target``."""
    arr = np.asarray([float(q) for q in quantiles], dtype=np.float32)
    return int(np.argmin(np.abs(arr - target)))


def _format_quantile(q: float) -> str:
    """Format a quantile key consistently (e.g. 0.1 -> '0.1')."""
    return f"{q:g}"


def compute_metrics(
    y_true,
    y_pred,
    config=None,
    threshold: float = 0.5,
    batch_size: int = 32,
    quantiles: Optional[Sequence[float]] = None,
) -> Dict[str, Union[float, Dict[str, float]]]:
    """Compute TFT evaluation metrics.

    Parameters
    ----------
    y_true:
        Ground-truth values.  Expected shape ``(batch, horizon)`` or
        ``(batch, horizon, 1)``.  Higher-dimensional trailing singleton
        dimensions are squeezed automatically.
    y_pred:
        Quantile predictions.  Expected shape
        ``(batch, horizon, num_quantiles)``, or a list/tuple of
        ``(batch, horizon)`` arrays, one per quantile.  A single
        ``(batch, horizon)`` tensor is interpreted as the median forecast.
    config:
        Optional config path or dict from which quantiles can be read.
    threshold:
        Ignored; retained for API compatibility with other metric functions.
    batch_size:
        Ignored; retained for API compatibility with other metric functions.
    quantiles:
        Optional explicit quantile values.  If omitted, defaults are
        ``[0.1, 0.5, 0.9]`` or values read from ``config``.

    Returns
    -------
    dict with keys:
        - ``q_risk``: average q-risk across all evaluated quantiles
        - ``p50_risk``: q-risk at the 50th percentile
        - ``p90_risk``: q-risk at the 90th percentile
        - ``normalized_quantile_losses``: dict mapping quantile to
          normalized quantile loss (no factor of two)
    """
    del threshold, batch_size  # retained for API compatibility

    y_true, y_pred = _prepare_tensors(y_true, y_pred)
    num_quantiles = y_pred.size(-1)
    quantile_values = _resolve_quantiles(config, quantiles, num_quantiles)

    total_abs = torch.sum(torch.abs(y_true))
    if total_abs.item() < 1e-12:
        # Avoid division by zero; the only reasonable metric is zero.
        zero_nql = {_format_quantile(q): 0.0 for q in quantile_values}
        return {
            "q_risk": 0.0,
            "p50_risk": 0.0,
            "p90_risk": 0.0,
            "normalized_quantile_losses": zero_nql,
        }

    risk_values: List[float] = []
    normalized_losses: Dict[str, float] = {}

    for i, q in enumerate(quantile_values):
        q = float(q)
        per_element_loss = _quantile_loss(y_true, y_pred[:, i], q)
        sum_ql = torch.sum(per_element_loss)

        nql = float((sum_ql / total_abs).item())
        risk = 2.0 * nql

        key = _format_quantile(q)
        risk_values.append(risk)
        normalized_losses[key] = nql

    q_risk = float(np.mean(risk_values)) if risk_values else 0.0

    p50_idx = _nearest_quantile_index(quantile_values, 0.5)
    p90_idx = _nearest_quantile_index(quantile_values, 0.9)

    return {
        "q_risk": q_risk,
        "p50_risk": risk_values[p50_idx],
        "p90_risk": risk_values[p90_idx],
        "normalized_quantile_losses": normalized_losses,
    }