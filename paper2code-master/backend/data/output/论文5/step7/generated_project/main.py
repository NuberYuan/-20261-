import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from training.train import train_model, build_dataloader
from evaluation.evaluate import evaluate_model, generate_report

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = "config/config.yaml"
DEFAULT_OUTPUT_DIR = "reports"
DEFAULT_LOOKBACK = 168
DEFAULT_HORIZON = 24


def _load_config(config_path: str) -> Dict[str, Any]:
    """Load YAML config with UTF-8 encoding.

    Returns an empty dict if the config cannot be read, rather than raising,
    so callers can safely inspect it.
    """
    path = Path(config_path)
    if not path.exists():
        logger.warning("Config file '%s' not found; using empty config.", path)
        return {}

    try:
        with open(path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
    except Exception:
        logger.exception("Failed to load config '%s'.", path)
        return {}

    return cfg if isinstance(cfg, dict) else {}


def _get_runs(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Return the list of experiment run dicts from config."""
    runs = config.get("experiment", {}).get("runs", [])
    if isinstance(runs, list):
        return [run for run in runs if isinstance(run, dict)]
    return []


def _get_dataset_names(config: Dict[str, Any]) -> List[str]:
    """Return dataset names from training.batch_size keys.

    The sample config explicitly enumerates Electricity, Traffic, Retail, and
    Volatility under ``training.batch_size``.  This avoids hardcoding dataset
    names while still honoring the known config structure.
    """
    batch_size_map = config.get("training", {}).get("batch_size", {})
    if isinstance(batch_size_map, dict):
        names = [
            key
            for key, value in batch_size_map.items()
            if isinstance(key, str)
            and isinstance(value, (int, float))
            and key != "batch_size_note"
        ]
        if names:
            return names

    # Fallback for safe no-config / minimal-config execution.
    return ["Electricity"]


def _is_unresolved(config: Dict[str, Any], version: str) -> bool:
    """Return True when the requested dataset version is marked unresolved."""
    data_cfg = config.get("data", {})
    if not isinstance(data_cfg, dict):
        return True

    version_cfg = data_cfg.get(str(version), {})
    if not isinstance(version_cfg, dict):
        return True

    status = str(version_cfg.get("status", "")).strip().lower()
    return status == "unresolved"


def main(
    config: str = DEFAULT_CONFIG_PATH,
    run: Optional[int] = None,
    threshold: float = 0.5,
    batch_size: Optional[int] = None,
) -> int:
    """Run the project pipeline for configured experiments.

    Parameters
    ----------
    config:
        Path to the YAML configuration file.
    run:
        Optional zero-based index into ``experiment.runs``.  If omitted, all
        runs are executed.
    threshold:
        Validation-accuracy threshold used by the training module's model
        selection heuristic.  It is not the paper's primary metric.
    batch_size:
        Optional override for DataLoader batch size.  When omitted, the value
        from ``training.batch_size`` is used.

    Returns
    -------
    int
        0 on clean completion, 1 when an invalid run index is requested.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    cfg = _load_config(config)

    # FR/engineering requirement: initialize report directory before the loop,
    # so summary_report.txt can always be generated even if every run fails or
    # is skipped.
    output_dir = Path(DEFAULT_OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary_report.txt"

    runs = _get_runs(cfg)
    if not runs:
        lines = ["No experiment.runs found in config."]
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        logger.warning("No experiment.runs found in config; wrote empty summary.")
        return 0

    if run is not None:
        if run < 0 or run >= len(runs):
            lines = [
                f"Requested run index {run} out of range; "
                f"available indices are 0..{len(runs) - 1}."
            ]
            with open(summary_path, "w", encoding="utf-8") as f:
                f.write("\n".join(lines) + "\n")
            logger.error("Invalid --run index: %s", run)
            return 1
        selected_runs = [runs[run]]
        logger.info("Running only experiment.runs[%d].", run)
    else:
        selected_runs = runs
        logger.info("Running all %d experiment.runs entries.", len(runs))

    dataset_names = _get_dataset_names(cfg)
    summary_lines: List[str] = []
    skipped_runs: List[str] = []
    successful_metrics: List[Dict[str, Any]] = []

    for run_cfg in selected_runs:
        run_name = str(run_cfg.get("name", "unnamed_run"))
        dataset_version = str(run_cfg.get("dataset_version", "default"))

        # Segmented skipping: if the configured dataset version is explicitly
        # unresolved, skip this run instead of failing.
        if _is_unresolved(cfg, dataset_version):
            logger.info(
                "Skipping run '%s': dataset_version '%s' status=unresolved",
                run_name,
                dataset_version,
            )
            skipped_runs.append(run_name)
            summary_lines.append(
                f"SKIPPED {run_name}: dataset_version '{dataset_version}' "
                "is unresolved"
            )
            continue

        for dataset_name in dataset_names:
            experiment_name = f"{run_name}_{dataset_name}"
            logger.info(
                "=== Starting run '%s', dataset '%s', version '%s' ===",
                run_name,
                dataset_name,
                dataset_version,
            )

            try:
                # Build train/test loaders with the selected run's dataset
                # version explicitly passed through.
                train_loader = build_dataloader(
                    config=cfg,
                    dataset_name=dataset_name,
                    split="train",
                    version=dataset_version,
                    lookback=DEFAULT_LOOKBACK,
                    horizon=DEFAULT_HORIZON,
                    batch_size=batch_size,
                    shuffle=True,
                )
                test_loader = build_dataloader(
                    config=cfg,
                    dataset_name=dataset_name,
                    split="test",
                    version=dataset_version,
                    lookback=DEFAULT_LOOKBACK,
                    horizon=DEFAULT_HORIZON,
                    batch_size=batch_size,
                    shuffle=False,
                )

                # train_model respects config['reproduction_mode']:
                # strict mode skips validation; engineering mode may split the
                # training set into train/val internally.
                train_result = train_model(
                    config=cfg,
                    model=None,
                    train_loader=train_loader,
                    val_loader=None,
                    test_loader=test_loader,
                    dataset_name=dataset_name,
                    device=None,
                    lookback=DEFAULT_LOOKBACK,
                    horizon=DEFAULT_HORIZON,
                    threshold=threshold,
                )

                model = train_result.get("model")
                if model is None:
                    raise RuntimeError("train_model did not return a trained model")

                metrics = evaluate_model(
                    model=model,
                    test_loader=test_loader,
                    config=cfg,
                    device=None,
                    lookback=DEFAULT_LOOKBACK,
                    horizon=DEFAULT_HORIZON,
                    threshold=threshold,
                    batch_size=32,
                )

                report_path = generate_report(
                    metrics,
                    experiment_name,
                    output_dir,
                    cfg,
                )

                q_risk = metrics.get("q_risk", float("nan"))
                p50_risk = metrics.get("p50_risk", float("nan"))
                p90_risk = metrics.get("p90_risk", float("nan"))
                normalized_quantile_losses = metrics.get(
                    "normalized_quantile_losses", {}
                )

                successful_metrics.append(
                    {
                        "run": run_name,
                        "dataset": dataset_name,
                        "q_risk": q_risk,
                        "p50_risk": p50_risk,
                        "p90_risk": p90_risk,
                        "normalized_quantile_losses": normalized_quantile_losses,
                        "report_path": report_path,
                    }
                )

                summary_lines.append(
                    f"SUCCESS {experiment_name}: q-risk={q_risk:.6f}, "
                    f"P50 risk={p50_risk:.6f}, P90 risk={p90_risk:.6f}, "
                    f"normalized_quantile_losses={normalized_quantile_losses}"
                )
                logger.info(
                    "Finished %s | q-risk=%.6f | P50 risk=%.6f | P90 risk=%.6f",
                    experiment_name,
                    q_risk,
                    p50_risk,
                    p90_risk,
                )

            except Exception as exc:
                logger.exception(
                    "Failed run '%s', dataset '%s': %s",
                    run_name,
                    dataset_name,
                    exc,
                )
                summary_lines.append(
                    f"FAILED {experiment_name}: {exc}"
                )

    if skipped_runs:
        summary_lines.insert(0, "Skipped runs: " + ", ".join(skipped_runs))

    if not summary_lines:
        summary_lines = ["No runs executed."]

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines) + "\n")

    logger.info("Summary report written to %s", summary_path)
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Temporal Fusion Transformer time-series forecasting entrypoint."
    )
    parser.add_argument(
        "--run",
        type=int,
        default=None,
        help="Optional zero-based index into experiment.runs. If omitted, all runs are executed.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=DEFAULT_CONFIG_PATH,
        help="Path to config YAML file.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    sys.exit(main(config=args.config, run=args.run))