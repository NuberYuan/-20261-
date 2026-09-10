import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, random_split

from data.dataset import ElectricityTrafficRetailVolatilityDataset
from models.lstm import Lstm
from models.transformer import Transformer

logger = logging.getLogger(__name__)

DEFAULT_LOOKBACK = 168
DEFAULT_HORIZON = 24
DEFAULT_QUANTILES = (0.1, 0.5, 0.9)


# ---------------------------------------------------------------------------
# Config utilities
# ---------------------------------------------------------------------------
def _load_config(config: Optional[Union[str, Path, Dict[str, Any]]] = None) -> Dict[str, Any]:
    """Load configuration from a dict, path or project config.yaml.

    All YAML file reads use UTF-8 encoding as required by the project.
    """
    if config is None:
        cfg_path = Path(__file__).resolve().parents[1] / "config" / "config.yaml"
        if not cfg_path.exists():
            logger.warning("config/config.yaml not found; using empty config.")
            return {}
        with open(cfg_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    if isinstance(config, dict):
        return config
    if isinstance(config, (str, Path)):
        path = Path(config)
        if not path.exists():
            logger.warning("Config path %s not found; using empty config.", path)
            return {}
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


def _parse_int(value: Any, default: int) -> int:
    try:
        return int(float(str(value)))
    except (TypeError, ValueError):
        return int(default)


def _parse_float(value: Any, default: float) -> float:
    if value is None or value == "NOT_FOUND":
        return float(default)
    try:
        return float(str(value))
    except (TypeError, ValueError):
        return float(default)


def _is_not_found(value: Any) -> bool:
    return value is None or value == "NOT_FOUND" or value == ""


def _get_dataset_version(config: Dict[str, Any]) -> str:
    runs = config.get("experiment", {}).get("runs", [])
    if isinstance(runs, list) and len(runs) > 0 and isinstance(runs[0], dict):
        return str(runs[0].get("dataset_version", "default"))
    return "default"


def _get_batch_size(config: Dict[str, Any], dataset_name: str) -> int:
    batch_size_map = config.get("training", {}).get("batch_size", {})
    if isinstance(batch_size_map, dict):
        value = batch_size_map.get(dataset_name)
        if value is not None:
            return _parse_int(value, 32)
    return 32


def _get_dropout(config: Dict[str, Any], dataset_name: str, fallback: float = 0.1) -> float:
    for section in ("model", "training"):
        dr = config.get(section, {}).get("dropout_ratio")
        if isinstance(dr, dict):
            value = dr.get(dataset_name)
            if value is not None and not _is_not_found(value):
                return _parse_float(value, fallback)
        elif dr is not None and not _is_not_found(dr):
            return _parse_float(dr, fallback)
    return float(fallback)


# ---------------------------------------------------------------------------
# Dataset / DataLoader helpers
# ---------------------------------------------------------------------------
def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Collate a list of dataset dicts into a batched dict of tensors."""
    if len(batch) == 0:
        return {}
    out: Dict[str, torch.Tensor] = {}
    for key in batch[0].keys():
        if isinstance(batch[0][key], torch.Tensor):
            out[key] = torch.stack([item[key] for item in batch], dim=0)
        else:
            out[key] = torch.tensor([item[key] for item in batch])
    return out


def build_dataloader(
    config: Optional[Dict[str, Any]] = None,
    dataset_name: str = "Electricity",
    split: str = "train",
    version: Optional[str] = None,
    lookback: int = DEFAULT_LOOKBACK,
    horizon: int = DEFAULT_HORIZON,
    batch_size: Optional[int] = None,
    shuffle: Optional[bool] = None,
    num_workers: Optional[int] = None,
) -> DataLoader:
    cfg = _load_config(config)
    if version is None:
        version = _get_dataset_version(cfg)
    if batch_size is None:
        batch_size = _get_batch_size(cfg, dataset_name)
    if shuffle is None:
        shuffle = split == "train"
    if num_workers is None:
        num_workers = _parse_int(cfg.get("system", {}).get("num_workers", 0), 0)

    dataset = ElectricityTrafficRetailVolatilityDataset(
        version=version,
        config=cfg,
        dataset_name=dataset_name,
        split=split,
        lookback=lookback,
        horizon=horizon,
        local_path=None,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )


# ---------------------------------------------------------------------------
# Model initialization
# ---------------------------------------------------------------------------
def initialize_model(
    config: Optional[Union[str, Path, Dict[str, Any]]] = None,
    dataset_name: str = "Electricity",
    device: Optional[torch.device] = None,
    lookback: int = DEFAULT_LOOKBACK,
    horizon: int = DEFAULT_HORIZON,
) -> nn.Module:
    """Initialize a randomly initialized model from scratch.

    This function never loads pretrained weights and does not use torchvision
    or any transfer-learning logic.
    """
    cfg = _load_config(config)

    # Dataset feature metadata is available without constructing the full
    # dataset, which keeps initialization fast and avoids synthetic-data cost.
    meta = ElectricityTrafficRetailVolatilityDataset._dataset_meta()[dataset_name]
    num_real_features = len(meta["real_cols"])
    num_categorical_features = len(meta["cat_cols"])

    architectures = cfg.get("model", {}).get("architectures", [])
    use_transformer = any(
        isinstance(a, str) and ("transformer" in a.lower() or "tft" in a.lower())
        for a in architectures
    )

    dropout = _get_dropout(cfg, dataset_name, fallback=0.1)

    if use_transformer or not architectures:
        # Default to the TFT-style Transformer when no architecture is explicitly
        # configured.  Feature counts are derived from dataset metadata.
        model = Transformer(
            config=cfg,
            d_model=160,
            num_heads=4,
            encoder_len=lookback,
            decoder_len=horizon,
            num_static=num_categorical_features,
            num_past_features=num_real_features,
            num_future_features=num_real_features,
            quantiles=list(DEFAULT_QUANTILES),
            dropout=dropout,
        )
    else:
        # LSTM fallback.  It uses real-valued inputs only; categorical values are
        # intentionally ignored in this lightweight fallback model.
        model = Lstm(
            input_size=num_real_features,
            hidden_size=128,
            num_layers=2,
            output_size=1,
            quantiles=list(DEFAULT_QUANTILES),
            dropout=dropout,
            config=cfg,
        )

    # Apply kaiming normal initialization for scratch-training mode.
    training_mode = cfg.get("training_modes", {}).get("scratch", {})
    if training_mode.get("weight_init") == "kaiming_normal":
        def _init_weights(m: nn.Module) -> None:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        model.apply(_init_weights)

    if device is not None:
        model.to(device)
    return model


# ---------------------------------------------------------------------------
# Loss and metrics
# ---------------------------------------------------------------------------
def quantile_loss(
    preds: torch.Tensor,
    targets: torch.Tensor,
    quantiles: Tuple[float, ...] = DEFAULT_QUANTILES,
) -> torch.Tensor:
    """Compute Quantile loss averaged over batch, horizons and quantiles.

    Args:
        preds: [batch, horizon, num_quantiles] or [batch, horizon, num_quantiles, 1]
        targets: [batch, horizon]
        quantiles: tuple/list of quantile levels, e.g. (0.1, 0.5, 0.9).

    Returns:
        Scalar loss tensor.
    """
    if preds.dim() == 4:
        preds = preds.squeeze(-1)
    if targets.dim() == 2:
        targets = targets.unsqueeze(-1)  # [B, H, 1]

    q = torch.tensor(quantiles, device=preds.device, dtype=preds.dtype).view(1, 1, -1)
    errors = targets - preds  # [B, H, Q]
    # QL = max(q * error, (q - 1) * error)
    loss = torch.max(q * errors, (q - 1.0) * errors)
    return loss.mean()


def compute_val_accuracy(
    preds: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
    median_index: int = 1,
) -> float:
    """Compute a simple regression accuracy proxy for model selection.

    It uses the median quantile as point forecast and treats a sample as
    accurate when the relative absolute error is below ``threshold``.
    This is an engineering deviation from the paper, whose original metric is
    quantile risk; it is only used to choose checkpoints in engineering mode.
    """
    if preds.dim() == 4:
        preds = preds.squeeze(-1)
    if preds.dim() == 3:
        if preds.size(-1) > 1:
            preds = preds[:, :, median_index]
        else:
            preds = preds.squeeze(-1)

    denom = targets.abs() + 1e-6
    rel_error = (preds - targets).abs() / denom
    accurate = (rel_error < threshold).float()
    return float(accurate.mean().item())


def _move_batch_to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}


def _get_model_inputs(
    model: nn.Module,
    batch: Dict[str, torch.Tensor],
    lookback: int,
    horizon: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Extract predictions and targets from a batch for LSTM or Transformer."""
    x_real = batch["x_real"]
    x_cat = batch.get("x_cat")
    y = batch["y"]

    if isinstance(model, Transformer):
        if x_cat is None:
            raise ValueError("Transformer requires x_cat in the batch")
        static_features = x_cat[:, 0, :]  # [B, num_static]
        past_features = x_real
        last_step = x_real[:, -1:, :]
        future_known = last_step.repeat(1, horizon, 1)
        future_features = torch.cat([past_features, future_known], dim=1)
        preds = model(static_features, past_features, future_features)
    elif isinstance(model, Lstm):
        past_inputs = x_real
        last_step = x_real[:, -1:, :]
        future_inputs = last_step.repeat(1, horizon, 1)
        preds = model(past_inputs, future_inputs)
    else:
        raise TypeError(f"Unsupported model type: {type(model)}")

    if preds.dim() == 4:
        preds = preds.squeeze(-1)
    return preds, y


def validate_one_epoch(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    quantiles: Tuple[float, ...] = DEFAULT_QUANTILES,
    threshold: float = 0.5,
    lookback: int = DEFAULT_LOOKBACK,
    horizon: int = DEFAULT_HORIZON,
) -> Dict[str, float]:
    """Run one validation pass and return val_loss and val_accuracy."""
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    n_batches = max(1, len(val_loader))

    with torch.no_grad():
        for batch in val_loader:
            batch = _move_batch_to_device(batch, device)
            preds, targets = _get_model_inputs(model, batch, lookback, horizon)
            loss = quantile_loss(preds, targets, quantiles)
            acc = compute_val_accuracy(preds, targets, threshold=threshold)
            total_loss += float(loss.item())
            total_acc += acc

    return {
        "val_loss": total_loss / n_batches,
        "val_accuracy": total_acc / n_batches,
    }


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train_model(
    config: Optional[Union[str, Path, Dict[str, Any]]] = None,
    model: Optional[nn.Module] = None,
    train_loader: Optional[DataLoader] = None,
    val_loader: Optional[DataLoader] = None,
    test_loader: Optional[DataLoader] = None,
    dataset_name: str = "Electricity",
    device: Optional[torch.device] = None,
    lookback: int = DEFAULT_LOOKBACK,
    horizon: int = DEFAULT_HORIZON,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """Run the model training loop.

    Behavior depends on ``config['reproduction_mode']``:

    - ``engineering``: train -> val -> best checkpoint. If no ``val_loader`` is
      supplied, a validation set is split from the training set according to
      ``config['split_strategy']['val_split_ratio']``.
    - ``strict``: train -> final model only. Validation is skipped even if a
      ``val_loader`` is provided, matching the paper's train/test split.

    ``test_loader`` is used only for final evaluation and never enters the
    train or validation loop.
    """
    cfg = _load_config(config)

    # Seed for reproducibility.
    seed = _parse_int(cfg.get("system", {}).get("seed", 42), 42)
    torch.manual_seed(seed)
    np.random.seed(seed)

    if device is None:
        device_str = cfg.get("system", {}).get("device", "cuda")
        if device_str == "cuda" and torch.cuda.is_available():
            device = torch.device("cuda")
        elif device_str == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA requested but not available; falling back to CPU.")
            device = torch.device("cpu")
        else:
            device = torch.device(device_str)

    # Build model if not provided.
    if model is None:
        model = initialize_model(cfg, dataset_name=dataset_name, device=device, lookback=lookback, horizon=horizon)
    model.to(device)

    # Read training config.
    training_cfg = cfg.get("training", {})
    optimizer_cfg = training_cfg.get("optimizer", {}) or {}
    scheduler_cfg = training_cfg.get("scheduler", {}) or {}
    checkpoint_cfg = cfg.get("checkpoint", {}) or {}

    epochs = _parse_int(training_cfg.get("epochs", 30), 30)
    lr = _parse_float(optimizer_cfg.get("lr", 0.01), 0.01)
    weight_decay = _parse_float(optimizer_cfg.get("weight_decay"), 0.0)
    momentum = _parse_float(optimizer_cfg.get("momentum"), 0.0)

    optimizer_name = optimizer_cfg.get("name")
    if _is_not_found(optimizer_name):
        optimizer_name = "Adam"
    optimizer_name = str(optimizer_name).lower()

    if optimizer_name in ("adam", "adamw"):
        optimizer_class = torch.optim.AdamW if optimizer_name == "adamw" else torch.optim.Adam
        if optimizer_name == "adamw":
            optimizer = optimizer_class(model.parameters(), lr=lr, weight_decay=weight_decay)
        else:
            optimizer = optimizer_class(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name in ("sgd", "rmsprop"):
        if optimizer_name == "sgd":
            optimizer = torch.optim.SGD(
                model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay
            )
        else:
            optimizer = torch.optim.RMSprop(
                model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay
            )
    else:
        logger.info("Unknown optimizer %s; falling back to Adam.", optimizer_name)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Optional scheduler. Default config has no scheduler, so this branch is
    # normally skipped.
    scheduler = None
    scheduler_name = scheduler_cfg.get("name")
    if not _is_not_found(scheduler_name):
        scheduler_name = str(scheduler_name).lower()
        if scheduler_name == "steplr":
            step_size = _parse_int(scheduler_cfg.get("step_size"), 10)
            gamma = _parse_float(scheduler_cfg.get("gamma"), 0.1)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
        elif scheduler_name == "reducelronplateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=5)

    # Build train_loader if needed.
    if train_loader is None:
        version = _get_dataset_version(cfg)
        batch_size = _get_batch_size(cfg, dataset_name)
        num_workers = _parse_int(cfg.get("system", {}).get("num_workers", 0), 0)
        train_loader = build_dataloader(
            cfg,
            dataset_name=dataset_name,
            split="train",
            version=version,
            lookback=lookback,
            horizon=horizon,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
        )

    reproduction_mode = cfg.get("reproduction_mode", "engineering")
    if reproduction_mode == "strict":
        # Paper-style train/test split: no validation, no best checkpoint.
        val_loader = None
    else:
        reproduction_mode = "engineering"  # fallback to engineering mode
        if val_loader is None:
            # Engineering deviation: split training set into 80/20 train/val.
            val_ratio = _parse_float(
                cfg.get("split_strategy", {}).get("val_split_ratio", 0.2), 0.2
            )
            full_train_dataset = train_loader.dataset
            n = len(full_train_dataset)
            val_len = max(1, int(n * val_ratio))
            train_len = max(1, n - val_len)

            train_subset, val_subset = random_split(
                full_train_dataset,
                [train_len, val_len],
                generator=torch.Generator().manual_seed(seed),
            )
            batch_size = train_loader.batch_size or _get_batch_size(cfg, dataset_name)
            num_workers = _parse_int(cfg.get("system", {}).get("num_workers", 0), 0)

            train_loader = DataLoader(
                train_subset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                collate_fn=collate_fn,
            )
            val_loader = DataLoader(
                val_subset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                collate_fn=collate_fn,
            )

    # Checkpoint settings.
    save_best = bool(checkpoint_cfg.get("save_best", True))
    save_final = bool(checkpoint_cfg.get("save_final", True))
    best_metric_name = checkpoint_cfg.get("best_metric", "val_accuracy")
    best_mode = checkpoint_cfg.get("best_mode", "max")
    best_score = float("-inf") if best_mode == "max" else float("inf")
    best_checkpoint_path = Path(checkpoint_cfg.get("resume_from") or "checkpoints") / "best_model.pt"
    final_checkpoint_path = Path("checkpoints") / "final_model.pt"

    # Ensure checkpoint directory exists.
    best_checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    quantiles = getattr(model, "quantiles", DEFAULT_QUANTILES)
    history: Dict[str, List[float]] = {
        "train_loss": [],
        "val_loss": [],
        "val_accuracy": [],
    }

    logger.info("=== Starting training ===")
    logger.info("Dataset: %s | Epochs: %d | Device: %s | Mode: %s",
                dataset_name, epochs, device, reproduction_mode)

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_train_batches = max(1, len(train_loader))

        for batch in train_loader:
            batch = _move_batch_to_device(batch, device)
            preds, targets = _get_model_inputs(model, batch, lookback, horizon)

            loss = quantile_loss(preds, targets, quantiles)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item())

        avg_train_loss = epoch_loss / n_train_batches
        history["train_loss"].append(avg_train_loss)

        log_msg = f"Epoch {epoch}/{epochs} | train_loss={avg_train_loss:.6f}"

        if val_loader is not None:
            val_metrics = validate_one_epoch(
                model,
                val_loader,
                device,
                quantiles=quantiles,
                threshold=threshold,
                lookback=lookback,
                horizon=horizon,
            )
            history["val_loss"].append(val_metrics["val_loss"])
            history["val_accuracy"].append(val_metrics["val_accuracy"])
            log_msg += (
                f" | val_loss={val_metrics['val_loss']:.6f}"
                f" | val_accuracy={val_metrics['val_accuracy']:.6f}"
            )

            # Choose score according to checkpoint config.
            if best_metric_name == "val_accuracy":
                score = val_metrics["val_accuracy"]
            elif best_metric_name == "val_loss":
                score = val_metrics["val_loss"]
            else:
                score = val_metrics["val_accuracy"]

            is_better = False
            if best_mode == "max":
                is_better = score > best_score
            else:
                is_better = score < best_score

            if is_better and save_best:
                best_score = score
                save_model(
                    model,
                    best_checkpoint_path,
                    optimizer=optimizer if checkpoint_cfg.get("save_optimizer_state", True) else None,
                    epoch=epoch,
                    extra={"val_loss": val_metrics["val_loss"],
                           "val_accuracy": val_metrics["val_accuracy"]},
                )
                log_msg += " [best saved]"
        else:
            # strict mode: no validation.
            history["val_loss"].append(float("nan"))
            history["val_accuracy"].append(float("nan"))

        logger.info(log_msg)

        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                if val_loader is not None:
                    scheduler.step(history["val_loss"][-1])
            else:
                scheduler.step()

    # Save final model if requested.
    if save_final:
        save_model(
            model,
            final_checkpoint_path,
            optimizer=optimizer if checkpoint_cfg.get("save_optimizer_state", True) else None,
            epoch=epochs,
            extra={"train_loss": avg_train_loss},
        )
        logger.info("Final model saved to %s", final_checkpoint_path)

    # Final evaluation on test set only, never used for model selection.
    test_metrics = None
    if test_loader is not None:
        test_metrics = validate_one_epoch(
            model,
            test_loader,
            device,
            quantiles=quantiles,
            threshold=threshold,
            lookback=lookback,
            horizon=horizon,
        )
        logger.info(
            "Test evaluation | val_loss=%s | val_accuracy=%s",
            test_metrics.get("val_loss"),
            test_metrics.get("val_accuracy"),
        )

    return {
        "model": model,
        "history": history,
        "best_checkpoint_path": str(best_checkpoint_path) if save_best else None,
        "final_checkpoint_path": str(final_checkpoint_path) if save_final else None,
        "test_metrics": test_metrics,
    }


# ---------------------------------------------------------------------------
# Model saving
# ---------------------------------------------------------------------------
def save_model(
    model: nn.Module,
    path: Union[str, Path],
    optimizer: Optional[torch.optim.Optimizer] = None,
    epoch: Optional[int] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> str:
    """Save model weights and optionally optimizer state to a checkpoint file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint: Dict[str, Any] = {
        "model_state_dict": model.state_dict(),
        "epoch": epoch,
    }
    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()
    if extra is not None:
        checkpoint.update(extra)

    torch.save(checkpoint, path)
    return str(path)


# ---------------------------------------------------------------------------
# Optional CLI entrypoint for basic testing
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    cfg = _load_config()
    dataset_name = cfg.get("experiment", {}).get("runs", [{}])[0].get("name", "Electricity")
    if isinstance(dataset_name, str) and not dataset_name.endswith("_default"):
        dataset_name = "Electricity"
    # A best-effort smoke test: build model + one small dataloader and run one epoch.
    model = initialize_model(cfg, dataset_name=dataset_name)
    loader = build_dataloader(
        cfg, dataset_name=dataset_name, split="train", version=_get_dataset_version(cfg)
    )
    train_model(
        cfg,
        model=model,
        train_loader=loader,
        val_loader=None,
        test_loader=None,
        dataset_name=dataset_name,
    )