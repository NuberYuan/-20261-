import numpy as np
from typing import Dict, Any, List, Optional, Union
import time


def compute_metrics(
    predictions: Union[List[int], np.ndarray],
    targets: Union[List[int], np.ndarray],
    training_times: Optional[List[float]] = None,
    config: Optional[Dict[str, Any]] = None,
    threshold: float = 0.5,
    batch_size: int = 32,
) -> Dict[str, float]:
    """
    Compute evaluation metrics for time series prediction task.

    Args:
        predictions: Predicted symbols or values
        targets: Ground truth symbols or values
        training_times: List of training times in seconds for each run
        config: Configuration dictionary (optional)
        threshold: Threshold for binary classification metrics
        batch_size: Batch size used during evaluation

    Returns:
        Dictionary containing:
            - success_rate: Fraction of sequences predicted exactly correctly
            - test_accuracy: Element-wise accuracy across all predictions
            - median_training_time: Median training time across runs (if provided)
    """
    predictions = np.asarray(predictions)
    targets = np.asarray(targets)

    if predictions.shape != targets.shape:
        raise ValueError(
            f"Shape mismatch: predictions {predictions.shape} vs targets {targets.shape}"
        )

    # Element-wise accuracy
    if predictions.size == 0:
        test_accuracy = 0.0
        success_rate = 0.0
    else:
        correct = np.sum(predictions == targets)
        test_accuracy = float(correct) / float(predictions.size)

        # Success rate: fraction of sequences where all elements match
        if predictions.ndim == 1:
            # Single sequence
            success_rate = 1.0 if np.array_equal(predictions, targets) else 0.0
        elif predictions.ndim == 2:
            # Multiple sequences: each row is a sequence
            row_matches = np.all(predictions == targets, axis=1)
            success_rate = float(np.mean(row_matches))
        else:
            # Flatten higher dimensions for success rate computation
            flat_pred = predictions.reshape(predictions.shape[0], -1)
            flat_target = targets.reshape(targets.shape[0], -1)
            row_matches = np.all(flat_pred == flat_target, axis=1)
            success_rate = float(np.mean(row_matches))

    # Median training time
    median_training_time = 0.0
    if training_times is not None and len(training_times) > 0:
        median_training_time = float(np.median(training_times))
    elif config is not None:
        # Try to extract from config if available
        cfg_times = config.get("training_times", None)
        if cfg_times is not None and len(cfg_times) > 0:
            median_training_time = float(np.median(cfg_times))

    metrics = {
        "success_rate": float(success_rate),
        "test_accuracy": float(test_accuracy),
        "median_training_time": float(median_training_time),
    }

    return metrics