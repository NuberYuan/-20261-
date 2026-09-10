"""Evaluation metrics for rule-based graph programs.

This module provides compute_metrics for reporting:
- time complexity
- correctness proofs
- runtime measurements
"""

from __future__ import annotations

import os
import statistics
from typing import Any, Dict, List, Optional, Sequence

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None


def _load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Load configuration from YAML file if available.

    Args:
        config_path: Path to config file. Defaults to config/config.yaml.

    Returns:
        Dictionary containing configuration values, or an empty dict when
        the file is missing or YAML is unavailable.
    """
    if config_path is None:
        config_path = os.path.join("config", "config.yaml")
    # Validate path existence before attempting to load.
    if not os.path.exists(config_path):
        return {}
    if yaml is None:
        return {}
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            loaded = yaml.safe_load(f)
    except (OSError, ValueError):
        return {}
    if not isinstance(loaded, dict):
        return {}
    return loaded


def compute_metrics(
    predictions: Optional[Sequence[Any]] = None,
    targets: Optional[Sequence[Any]] = None,
    runtimes: Optional[Sequence[float]] = None,
    correctness_proofs: Optional[Sequence[bool]] = None,
    config: Optional[Dict[str, Any]] = None,
    threshold: float = 0.5,
    batch_size: int = 32,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Compute evaluation metrics.

    The function reports three metric groups required by FR-05:
    time complexity, correctness proofs, and runtime measurements.
    All values are computed from the supplied data or configuration;
    no hard-coded conclusions are embedded.

    Args:
        predictions: Model predictions for the test set.
        targets: Ground-truth labels for the test set.
        runtimes: Runtime measurements (e.g., seconds per batch or sample).
        correctness_proofs: Optional precomputed correctness proof flags.
        config: Optional configuration dictionary. If None, loaded from
            config/config.yaml when available.
        threshold: Decision threshold for binary classification.
        batch_size: Batch size used in evaluation (reserved for future use;
            not required by current metric computations).
        **kwargs: Additional optional overrides such as time_complexity.

    Returns:
        A dictionary with keys:
        - time_complexity: str or numeric
        - correctness_proofs: dict with num_correct, total, rate, proofs
        - runtime_measurements: dict with mean, std, min, max, count, total
    """
    # Load configuration only when needed and always safely.
    if config is None:
        config = _load_config()
    elif not isinstance(config, dict):
        config = {}

    # Metric-related nested configuration is optional.
    metrics_cfg = config.get("metrics", {})
    if not isinstance(metrics_cfg, dict):
        metrics_cfg = {}

    # Time complexity: prefer config, then keyword override, then unknown.
    time_complexity = metrics_cfg.get("time_complexity")
    if time_complexity is None:
        time_complexity = config.get("time_complexity")
    if time_complexity is None:
        time_complexity = kwargs.get("time_complexity", "unknown")

    # Correctness proofs.
    proof_flags: List[bool] = []
    if correctness_proofs is not None:
        proof_flags = [bool(p) for p in correctness_proofs]
        num_correct = sum(1 for p in proof_flags if p)
        total_proofs = len(proof_flags)
    elif predictions is not None and targets is not None:
        pred_list = list(predictions)
        target_list = list(targets)
        if len(pred_list) != len(target_list):
            raise ValueError("predictions and targets must have the same length")
        num_correct = 0
        total_proofs = len(pred_list)
        proof_flags = []
        for pred, target in zip(pred_list, target_list):
            # Binary probabilistic prediction vs integer target.
            if (
                isinstance(pred, float)
                and isinstance(target, (int, float))
                and int(target) in (0, 1)
            ):
                pred_binary = 1 if pred >= threshold else 0
                is_correct = pred_binary == int(target)
            else:
                is_correct = pred == target
            if is_correct:
                num_correct += 1
            proof_flags.append(is_correct)
    else:
        num_correct = 0
        total_proofs = 0

    correctness_rate = (num_correct / total_proofs) if total_proofs > 0 else 0.0
    correctness_result = {
        "num_correct": num_correct,
        "total": total_proofs,
        "rate": correctness_rate,
        "proofs": proof_flags,
    }

    # Runtime measurements.
    runtime_result: Dict[str, Any] = {}
    if runtimes:
        cleaned = [float(r) for r in runtimes if r is not None]
        if cleaned:
            runtime_result = {
                "mean": statistics.mean(cleaned),
                "std": statistics.stdev(cleaned) if len(cleaned) > 1 else 0.0,
                "min": min(cleaned),
                "max": max(cleaned),
                "count": len(cleaned),
                "total": sum(cleaned),
            }
    if not runtime_result:
        runtime_result = {
            "mean": 0.0,
            "std": 0.0,
            "min": 0.0,
            "max": 0.0,
            "count": 0,
            "total": 0.0,
        }

    return {
        "time_complexity": time_complexity,
        "correctness_proofs": correctness_result,
        "runtime_measurements": runtime_result,
    }


__all__ = ["compute_metrics"]