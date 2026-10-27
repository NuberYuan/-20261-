"""项目入口，串联整个流程。"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from typing import Any, Dict, List, Optional, Sequence

_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

_DEFAULT_CONFIG_PATH = os.path.join(_PROJECT_ROOT, "config", "config.yaml")

LOGGER = logging.getLogger(__name__)

_IMPORT_ERRORS: Dict[str, BaseException] = {}

try:
    from data.dataset import PlantVillageDataset
except Exception as exc:  # pragma: no cover - dependency may be unavailable
    PlantVillageDataset = None
    _IMPORT_ERRORS["data.dataset"] = exc

try:
    from training.train import (
        build_criterion,
        build_dataloaders,
        build_optimizer,
        build_scheduler,
        initialize_model,
        load_config,
        resolve_device,
        set_seed,
        train_model,
    )
except Exception as exc:  # pragma: no cover
    build_criterion = None
    build_dataloaders = None
    build_optimizer = None
    build_scheduler = None
    initialize_model = None
    load_config = None
    resolve_device = None
    set_seed = None
    train_model = None
    _IMPORT_ERRORS["training.train"] = exc

try:
    from evaluation.evaluate import evaluate_model, generate_report
except Exception as exc:  # pragma: no cover
    evaluate_model = None
    generate_report = None
    _IMPORT_ERRORS["evaluation.evaluate"] = exc


def _cfg_get(config: Optional[Dict[str, Any]], dotted_key: str, default: Any = None) -> Any:
    """Read a value from a config dict, supporting both nested and dotted keys."""
    if not isinstance(config, dict) or not dotted_key:
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


def _to_int(value: Any, default: int, low: int = 0) -> int:
    """Coerce ``value`` to an int >= ``low``."""
    try:
        result = int(value)
    except (TypeError, ValueError):
        return default
    return result if result >= low else default


def _to_float(value: Any, default: float, low: Optional[float] = None, high: Optional[float] = None) -> float:
    """Coerce ``value`` to a float inside the optional ``[low, high]`` range."""
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    if result != result:  # NaN
        return default
    if low is not None and result < low:
        return default
    if high is not None and result > high:
        return default
    return result


def _to_bool(value: Any, default: bool) -> bool:
    """Coerce YAML-ish values to ``bool``."""
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        text = value.strip().lower()
        if text in ("1", "true", "yes", "on", "y"):
            return True
        if text in ("0", "false", "no", "off", "n"):
            return False
    return default


def _require(*names: str) -> None:
    """Raise a clear error when a required dependency object is unavailable."""
    missing = [name for name in names if globals().get(name) is None]
    if missing:
        details = "; ".join(f"{mod}: {err}" for mod, err in _IMPORT_ERRORS.items())
        raise RuntimeError(
            "Missing required dependency objects: "
            + ", ".join(missing)
            + (f" ({details})" if details else "")
        )


def _class_names(dataset: Any) -> List[str]:
    """Best-effort extraction of the class names from a dataset / split dict."""
    if dataset is None:
        return []
    classes = getattr(dataset, "classes", None)
    if callable(classes):
        try:
            classes = classes()
        except Exception:
            classes = None
    if classes is not None:
        try:
            return [str(name) for name in classes]
        except Exception:
            pass
    if isinstance(dataset, dict):
        for value in dataset.values():
            names = _class_names(value)
            if names:
                return names
    return []


def _num_samples(loader: Any) -> Optional[int]:
    """Number of samples behind a dataloader (``None`` when unavailable)."""
    if loader is None:
        return None
    inner = getattr(loader, "dataset", None)
    if inner is not None:
        try:
            return len(inner)
        except Exception:
            pass
    try:
        return len(loader)
    except Exception:
        return None


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plant disease detection reproduction - project entry point."
    )
    parser.add_argument("--config", default=_DEFAULT_CONFIG_PATH, help="Path to config/config.yaml")
    parser.add_argument("--model", default=None, choices=["googlenet", "alexnet"],
                        help="Model architecture (default: model.default)")
    parser.add_argument("--training-mode", default=None, choices=["scratch", "transfer"],
                        help="Training mode (default: training_modes.active)")
    parser.add_argument("--dataset-version", default=None,
                        choices=["color", "grayscale", "segmented"],
                        help="PlantVillage image version (default: data.default_version)")
    parser.add_argument("--train-ratio", type=float, default=None,
                        help="Train/test split ratio (default: split_strategy.default_train_ratio)")
    parser.add_argument("--epochs", type=int, default=None, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Batch size override (default: per-model training.batch_size)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--device", default=None, help="Torch device override (e.g. cuda / cpu)")
    parser.add_argument("--output-dir", default=None, help="Output directory for reports")
    parser.add_argument("--checkpoint-dir", default=None, help="Directory used to save checkpoints")
    parser.add_argument("--report-name", default=None, help="Base name of the generated report")
    parser.add_argument("--skip-training", action="store_true",
                        help="Skip training and only run the final evaluation / report stage")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    """Project entry point: load config, build data / model / optimiser, train, evaluate, report."""
    args = _build_arg_parser().parse_args(argv)

    _require("load_config", "set_seed", "resolve_device", "initialize_model",
             "build_dataloaders", "build_optimizer", "build_scheduler", "build_criterion",
             "train_model", "evaluate_model", "generate_report", "PlantVillageDataset")

    config_arg: Optional[str] = args.config if (args.config and os.path.isfile(args.config)) else None
    cfg: Dict[str, Any] = load_config(config_arg)

    seed = args.seed if args.seed is not None else _to_int(_cfg_get(cfg, "system.seed", 42), 42)
    deterministic = _to_bool(_cfg_get(cfg, "system.deterministic", True), True)
    set_seed(seed, deterministic)

    device = resolve_device(args.device, cfg)

    model_name = args.model or _cfg_get(cfg, "model.default", "googlenet")
    training_mode = args.training_mode or _cfg_get(cfg, "training_modes.active", "transfer")
    dataset_version = args.dataset_version or _cfg_get(cfg, "data.default_version", "color")

    train_ratio = _to_float(
        args.train_ratio if args.train_ratio is not None
        else _cfg_get(cfg, "split_strategy.default_train_ratio", 0.8),
        0.8, low=0.05, high=0.95,
    )
    num_classes = _to_int(
        _cfg_get(cfg, "model.num_classes", _cfg_get(cfg, "data.num_classes", 38)), 38, low=1
    )
    epochs = args.epochs if args.epochs is not None else _to_int(_cfg_get(cfg, "training.epochs", 30), 30, low=1)
    log_interval = _to_int(
        _cfg_get(cfg, "training.log_interval", _cfg_get(cfg, "logging.log_interval", 50)), 50, low=1
    )

    output_dir = args.output_dir or _cfg_get(cfg, "experiment.output_dir",
                                             _cfg_get(cfg, "system.output_root", "./outputs"))
    checkpoint_dir = args.checkpoint_dir or _cfg_get(cfg, "checkpoint.dir", "./checkpoints")

    reproduction_mode = _cfg_get(cfg, "reproduction_mode", "engineering")
    val_ratio = _cfg_get(cfg, "split_strategy.val_split_ratio", 0.2) if _to_bool(
        _cfg_get(cfg, "reproduction_modes.engineering.use_validation", True), True
    ) else None

    # --- data -------------------------------------------------------------
    dataset = PlantVillageDataset(
        config=cfg,
        version=dataset_version,
        seed=seed,
        val_ratio=val_ratio,
        reproduction_mode=reproduction_mode,
    )

    # --- dataloaders ------------------------------------------------------
    train_loader, val_loader, test_loader, dataset = build_dataloaders(
        config=cfg,
        dataset=dataset,
        model_name=model_name,
        batch_size=args.batch_size,
        train_ratio=train_ratio,
        seed=seed,
        reproduction_mode=reproduction_mode,
    )

    # --- model / optimisation --------------------------------------------
    model = initialize_model(
        config=cfg,
        model_name=model_name,
        training_mode=training_mode,
        num_classes=num_classes,
        device=device,
    )
    optimizer = build_optimizer(model, config=cfg)
    scheduler = build_scheduler(optimizer, config=cfg)
    criterion = build_criterion(config=cfg)

    class_names = _class_names(dataset)

    # --- training ---------------------------------------------------------
    results: Dict[str, Any] = {}
    if not args.skip_training:
        results = train_model(
            config=cfg,
            model=model,
            model_name=model_name,
            training_mode=training_mode,
            num_classes=num_classes,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            dataset=dataset,
            epochs=epochs,
            device=device,
            optimizer=optimizer,
            scheduler=scheduler,
            criterion=criterion,
            checkpoint_dir=checkpoint_dir,
            save_best=_to_bool(_cfg_get(cfg, "checkpoint.save_best", True), True),
            save_final=_to_bool(_cfg_get(cfg, "checkpoint.save_final", True), True),
            log_interval=log_interval,
            seed=seed,
        )

    # --- final test evaluation -------------------------------------------
    metrics: Dict[str, Any] = {}
    if test_loader is not None:
        metrics = evaluate_model(
            model,
            test_loader,
            criterion=criterion,
            device=device,
            config=cfg,
            log_interval=log_interval,
            epoch=epochs,
            class_names=class_names,
            tag="test",
        )

    history = results.get("history") if isinstance(results, dict) else None

    split_label = f"{int(round(train_ratio * 100))}-{int(round((1.0 - train_ratio) * 100))}"
    run_info: Dict[str, Any] = {
        "model_name": model_name,
        "training_mode": training_mode,
        "dataset_version": dataset_version,
        "train_test_split": split_label,
        "train_ratio": train_ratio,
        "num_classes": num_classes,
        "epochs": epochs,
        "seed": seed,
        "device": str(device),
        "reproduction_mode": reproduction_mode,
        "train_size": _num_samples(train_loader),
        "val_size": _num_samples(val_loader),
        "test_size": _num_samples(test_loader),
    }

    report: Dict[str, Any] = {}
    if metrics:
        report = generate_report(
            metrics=metrics,
            config=cfg,
            run_info=run_info,
            output_dir=output_dir,
            report_name=args.report_name,
            class_names=class_names,
            history=history,
            save_plots=_to_bool(_cfg_get(cfg, "logging.save_confusion_matrix", True), True),
        )

    summary: Dict[str, Any] = {
        "config": cfg,
        "device": str(device),
        "model_name": model_name,
        "training_mode": training_mode,
        "dataset_version": dataset_version,
        "train_ratio": train_ratio,
        "num_classes": num_classes,
        "epochs": epochs,
        "seed": seed,
        "run_info": run_info,
        "metrics": metrics,
        "training": results,
        "report": report,
    }

    LOGGER.info(
        "Finished run: model=%s mode=%s version=%s split=%s accuracy=%s",
        model_name,
        training_mode,
        dataset_version,
        split_label,
        metrics.get("accuracy", metrics.get("overall_accuracy", "n/a")) if metrics else "n/a",
    )
    return summary


if __name__ == "__main__":
    main()