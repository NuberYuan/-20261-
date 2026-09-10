"""Project entry point for time-series prediction.

Loads configuration, executes experiment runs, and generates evaluation
reports.  Supports optional command-line arguments:

    --run <int>       Run only the experiment at the given index.
    --config <path>   Load configuration from the given YAML path.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from data.dataset import get_dataloaders
from evaluation.evaluate import evaluate_model, generate_report
from training.train import initialize_model, train_model

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------
def _load_config(config_path: str = "config/config.yaml") -> Dict[str, Any]:
    """Load configuration from a YAML file."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


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


def _resolve_device(config: Dict[str, Any]) -> Any:
    """Resolve the device from config or fall back to available hardware."""
    device_str = _safe_get(config, "system", "device", default=None)
    if device_str is not None and str(device_str).strip() != "":
        return str(device_str)
    import torch

    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


# ---------------------------------------------------------------------------
# Single-run execution
# ---------------------------------------------------------------------------
def _run_single_experiment(
    run: Dict[str, Any],
    config: Dict[str, Any],
    output_dir: Path,
    run_index: int,
) -> Dict[str, Any]:
    """Execute a single experiment run.

    Parameters
    ----------
    run : dict
        Run configuration from ``experiment.runs``.
    config : dict
        Full configuration dictionary.
    output_dir : Path
        Directory for reports and checkpoints.
    run_index : int
        Index of this run in the runs list.

    Returns
    -------
    Dict[str, Any]
        Result dictionary with keys ``run_name``, ``status``, and metrics.
    """
    run_name = str(run.get("name", f"run_{run_index}"))
    dataset_version = str(run.get("dataset_version", "default"))
    reproduction_mode = str(
        run.get("reproduction_mode", config.get("reproduction_mode", "engineering"))
    )

    # Check dataset version status.
    data_status = _safe_get(
        config, "data", dataset_version, "status", default="resolved"
    )
    if str(data_status).lower() == "unresolved":
        logger.warning(
            "Skipping run '%s': dataset version '%s' has status 'unresolved'.",
            run_name,
            dataset_version,
        )
        return {
            "run_name": run_name,
            "status": "skipped",
            "reason": f"dataset version '{dataset_version}' is unresolved",
        }

    # Build dataloaders with explicit version.
    batch_size = int(
        _safe_get(config, "training", "batch_size", "default", default=20)
    )
    num_workers = int(_safe_get(config, "system", "num_workers", default=0))
    seed = int(_safe_get(config, "system", "seed", default=42))

    loaders = get_dataloaders(
        config=config,
        version=dataset_version,
        batch_size=batch_size,
        num_workers=num_workers,
        seed=seed,
    )

    train_loader = loaders.get("train")
    val_loader = loaders.get("val")
    test_loader = loaders.get("test")

    if train_loader is None:
        logger.error("Run '%s': no train loader available.", run_name)
        return {"run_name": run_name, "status": "failed", "reason": "no train loader"}

    # Initialize model.
    model = initialize_model(config=config)

    # Checkpoint directory for this run.
    checkpoint_dir = Path("checkpoints") / run_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Train model.
    start_time = time.time()
    try:
        history = train_model(
            config=config,
            model=model,
            train_loader=train_loader,
            val_loader=val_loader if reproduction_mode != "strict" else None,
            test_loader=test_loader,
            epochs=int(_safe_get(config, "training", "epochs", default=30)),
            batch_size=batch_size,
            device=_resolve_device(config),
            checkpoint_dir=str(checkpoint_dir),
            save_best=bool(_safe_get(config, "checkpoint", "save_best", default=True)),
            save_final=bool(_safe_get(config, "checkpoint", "save_final", default=True)),
            best_metric=str(
                _safe_get(config, "checkpoint", "best_metric", default="val_accuracy")
            ),
            best_mode=str(_safe_get(config, "checkpoint", "best_mode", default="max")),
            save_optimizer_state=bool(
                _safe_get(
                    config, "checkpoint", "save_optimizer_state", default=True
                )
            ),
            resume_from=_safe_get(config, "checkpoint", "resume_from", default=None),
        )
    except Exception as exc:  # noqa: BLE001 - log and continue
        logger.exception("Run '%s' failed during training: %s", run_name, exc)
        return {"run_name": run_name, "status": "failed", "reason": str(exc)}

    training_time = time.time() - start_time

    # Evaluate model.
    best_checkpoint = checkpoint_dir / "best_model.pt"
    checkpoint_path = str(best_checkpoint) if best_checkpoint.exists() else None

    try:
        metrics = evaluate_model(
            model=model,
            test_loader=test_loader,
            config=config,
            device=_resolve_device(config),
            training_times=[training_time],
            checkpoint_path=checkpoint_path,
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("Run '%s' failed during evaluation: %s", run_name, exc)
        return {"run_name": run_name, "status": "failed", "reason": str(exc)}

    # Generate per-run report.
    try:
        report_path = generate_report(
            metrics=metrics,
            run_name=run_name,
            output_dir=str(output_dir),
            config=config,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Run '%s': report generation failed: %s", run_name, exc)
        report_path = None

    result: Dict[str, Any] = {
        "run_name": run_name,
        "status": "success",
        "metrics": metrics,
        "training_time": training_time,
        "report_path": report_path,
    }
    return result


# ---------------------------------------------------------------------------
# Summary report
# ---------------------------------------------------------------------------
def _write_summary_report(
    output_dir: Path,
    results: List[Dict[str, Any]],
    config: Dict[str, Any],
) -> str:
    """Write a summary report for all runs.

    Parameters
    ----------
    output_dir : Path
        Directory for the summary report.
    results : list of dict
        Results from all executed runs.
    config : dict
        Full configuration dictionary.

    Returns
    -------
    str
        Path to the summary report file.
    """
    lines: List[str] = []
    lines.append("=" * 70)
    lines.append("TIME SERIES PREDICTION - SUMMARY REPORT")
    lines.append("=" * 70)
    lines.append(f"Generated at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")

    paper_name = _safe_get(
        config, "project", "paper", default="LEARNING SIMPLE ALGORITHMS FROM EXAMPLES"
    )
    framework = _safe_get(config, "project", "framework", default="PyTorch")
    lines.append(f"Paper: {paper_name}")
    lines.append(f"Framework: {framework}")
    lines.append("")

    lines.append("-" * 70)
    lines.append("RUN RESULTS")
    lines.append("-" * 70)

    success_count = 0
    skipped_count = 0
    failed_count = 0

    for result in results:
        run_name = result.get("run_name", "unknown")
        status = result.get("status", "unknown")
        lines.append(f"\nRun: {run_name}")
        lines.append(f"  Status: {status}")

        if status == "success":
            success_count += 1
            metrics = result.get("metrics", {})
            lines.append(
                f"  Success rate:          {metrics.get('success_rate', 0.0):.4f}"
            )
            lines.append(
                f"  Test accuracy:         {metrics.get('test_accuracy', 0.0):.4f}"
            )
            lines.append(
                "  Median training time:  "
                f"{metrics.get('median_training_time', 0.0):.4f} s"
            )
            if result.get("report_path"):
                lines.append(f"  Report: {result['report_path']}")
        elif status == "skipped":
            skipped_count += 1
            lines.append(f"  Reason: {result.get('reason', 'unknown')}")
        else:
            failed_count += 1
            lines.append(f"  Reason: {result.get('reason', 'unknown')}")

    lines.append("")
    lines.append("-" * 70)
    lines.append("SUMMARY")
    lines.append("-" * 70)
    lines.append(f"Total runs:    {len(results)}")
    lines.append(f"Successful:    {success_count}")
    lines.append(f"Skipped:       {skipped_count}")
    lines.append(f"Failed:        {failed_count}")
    lines.append("")
    lines.append("=" * 70)

    summary_path = output_dir / "summary_report.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    logger.info("Saved summary report to %s", summary_path)
    return str(summary_path)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def main(
    config: Optional[Any] = None,
    run_index: Optional[int] = None,
    output_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Execute experiment runs and generate reports.

    Parameters
    ----------
    config : dict or str, optional
        Config dict or path to config.yaml.  If None, the default path
        ``config/config.yaml`` is used.
    run_index : int, optional
        If provided, only the run at this index in ``experiment.runs`` is
        executed.  If None, all runs are executed.
    output_dir : str, optional
        Directory for reports.  Defaults to ``reports``.

    Returns
    -------
    Dict[str, Any]
        Dictionary with keys ``results`` (list of per-run results) and
        ``summary_report`` (path to summary report).
    """
    # Load config.
    if config is None:
        config = _load_config("config/config.yaml")
    elif isinstance(config, str):
        config = _load_config(config)
    elif not isinstance(config, dict):
        raise TypeError("config must be None, a dict, or a path string.")

    # Initialize report directory before the loop.
    if output_dir is None:
        output_dir = "reports"
    report_dir = Path(output_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    # Get runs.
    runs = _safe_get(config, "experiment", "runs", default=[])
    if not isinstance(runs, list) or len(runs) == 0:
        logger.warning("No experiment runs found in config.")
        results: List[Dict[str, Any]] = []
        summary_path = _write_summary_report(report_dir, results, config)
        return {"results": results, "summary_report": summary_path}

    # Select runs to execute.
    if run_index is not None:
        if run_index < 0 or run_index >= len(runs):
            raise IndexError(
                f"run_index {run_index} out of range for {len(runs)} runs."
            )
        selected_runs = [(run_index, runs[run_index])]
    else:
        selected_runs = list(enumerate(runs))

    # Execute runs.
    results = []
    for idx, run in selected_runs:
        logger.info(
            "Executing run %d/%d: %s",
            idx + 1,
            len(runs),
            run.get("name", f"run_{idx}"),
        )
        result = _run_single_experiment(run, config, report_dir, idx)
        results.append(result)

    # Write summary report.
    summary_path = _write_summary_report(report_dir, results, config)

    return {"results": results, "summary_report": summary_path}


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Time-series prediction experiment runner."
    )
    parser.add_argument(
        "--run",
        type=int,
        default=None,
        help="Run only the experiment at the given index (0-based).",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path to config.yaml (default: config/config.yaml).",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    args = _parse_args()
    try:
        result = main(config=args.config, run_index=args.run)
        logger.info("Summary report: %s", result["summary_report"])
        sys.exit(0)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Fatal error: %s", exc)
        sys.exit(1)