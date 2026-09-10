import argparse
import logging
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from data.dataset import get_dataloaders, load_config
from training.train import initialize_model, train_model, _resolve_device
from evaluation.evaluate import evaluate_model, generate_report


logger = logging.getLogger(__name__)


def main(config_path=None, run_index=None):
    """Project entrypoint: load config, run experiments, evaluate and report."""
    if config_path is None:
        config_path = os.path.join("config", "config.yaml")

    config = load_config(config_path)

    log_level = config.get("logging", {}).get("level", "INFO")
    logging.basicConfig(level=getattr(logging, str(log_level).upper(), logging.INFO))

    seed = int(config.get("system", {}).get("seed", 42))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = _resolve_device(config)

    output_dir = Path(config.get("logging", {}).get("save_dir", "logs")) / "reports"
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_dir = Path(config.get("logging", {}).get("save_dir", "logs")) / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    batch_size = int(config.get("training", {}).get("batch_size", 16))
    num_workers = int(config.get("system", {}).get("num_workers", 4))
    reproduction_mode = config.get("reproduction_mode", "engineering")

    runs = config.get("experiment", {}).get("runs", [])
    if not runs:
        default_version = config.get("version_selector", {}).get("default", "color")
        runs = [
            {
                "run_name": "unetplusplus_scratch_color",
                "architecture": "UNet++",
                "dataset_version": default_version,
                "training_mode": "scratch",
            }
        ]

    if run_index is not None:
        if 0 <= run_index < len(runs):
            runs = [runs[run_index]]
        else:
            logger.warning(
                "--run %s is out of range for %d configured run(s). No runs will be executed.",
                run_index,
                len(runs),
            )
            runs = []

    summary_lines = []
    summary_lines.append("UNet++ Project Run Summary")
    summary_lines.append("=" * 40)

    for run in runs:
        run_name = run.get("run_name", "run")
        version = run.get(
            "dataset_version",
            config.get("version_selector", {}).get("default", "color"),
        )

        data_cfg = config.get("data", {})
        version_cfg = data_cfg.get(version, {}) or {}
        if version_cfg.get("status") == "unresolved":
            msg = f"{run_name}: SKIPPED (dataset version '{version}' unresolved)"
            logger.info(msg)
            summary_lines.append(msg)
            continue

        data_root = data_cfg.get("root_dir")
        if data_root is None or not os.path.isdir(str(data_root)):
            msg = f"{run_name}: SKIPPED (data.root_dir unavailable)"
            logger.warning(msg)
            summary_lines.append(msg)
            continue

        try:
            train_loader, val_loader, test_loader = get_dataloaders(
                config=config,
                version=version,
                batch_size=batch_size,
                num_workers=num_workers,
                root_dir=data_root,
            )
        except Exception as exc:
            msg = f"{run_name}: SKIPPED (data loader error: {exc})"
            logger.warning(msg)
            summary_lines.append(msg)
            continue

        if len(train_loader.dataset) == 0:
            msg = f"{run_name}: SKIPPED (no training samples)"
            logger.warning(msg)
            summary_lines.append(msg)
            continue

        model = initialize_model(config=config, device=device)

        val_loader_for_train = None if reproduction_mode == "strict" else val_loader

        train_result = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader_for_train,
            config=config,
            device=device,
            checkpoint_dir=str(checkpoint_dir),
            run_name=run_name,
        )

        best_path = train_result.get("best_model_path")
        final_path = train_result.get("final_model_path")

        checkpoint_path = None
        if best_path and os.path.isfile(best_path):
            checkpoint_path = best_path
        elif final_path and os.path.isfile(final_path):
            checkpoint_path = final_path

        if checkpoint_path is not None:
            checkpoint = torch.load(checkpoint_path, map_location=device)
            state_dict = checkpoint.get("model_state_dict", checkpoint)
            model.load_state_dict(state_dict)
            logger.info("Loaded checkpoint %s for evaluation.", checkpoint_path)
        else:
            logger.info("No checkpoint found for %s; evaluating current model weights.", run_name)

        if test_loader is None or len(test_loader) == 0:
            msg = f"{run_name}: SKIPPED (no test samples)"
            logger.warning(msg)
            summary_lines.append(msg)
            continue

        metrics = evaluate_model(
            model=model,
            dataloader=test_loader,
            device=device,
            config=config,
            threshold=0.5,
        )

        report_path = generate_report(
            metrics=metrics,
            run_name=run_name,
            output_dir=str(output_dir),
            config=config,
        )
        logger.info("Report saved to %s", report_path)

        dice = metrics.get("dice_coefficient", 0.0)
        iou = metrics.get("iou", 0.0)
        accuracy = metrics.get("accuracy", 0.0)
        msg = (
            f"{run_name}: Dice={dice:.6f}, IoU={iou:.6f}, "
            f"Accuracy={accuracy:.6f}"
        )
        summary_lines.append(msg)
        logger.info(msg)

    summary_lines.append("")
    summary_lines.append("Skipped runs are listed above if any.")

    summary_path = output_dir / "summary_report.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines))

    logger.info("Summary report saved to %s", summary_path)
    return str(summary_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="UNet++ segmentation project entrypoint")
    parser.add_argument(
        "--run",
        type=int,
        default=None,
        help="Run only the experiment at this index in experiment.runs",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=os.path.join("config", "config.yaml"),
        help="Path to config.yaml",
    )
    args = parser.parse_args()

    main(config_path=args.config, run_index=args.run)