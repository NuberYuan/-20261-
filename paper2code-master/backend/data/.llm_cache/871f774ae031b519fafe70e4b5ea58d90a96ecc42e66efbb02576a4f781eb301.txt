"""Training pipeline for the time-series prediction task.

Implements model initialization, training loop, validation, and checkpoint
saving for the LSTM controller described in:
    LEARNING SIMPLE ALGORITHMS FROM EXAMPLES

The pipeline reads all hyperparameters from config/config.yaml.  Two
reproduction modes are supported:

* ``engineering``: train -> val -> best -> test.  A validation split is
  carved out of the training set (80/20) for model selection.
* ``strict``: train -> final -> test.  No validation split is used; only
  the final model is saved.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import yaml

from data.dataset import get_dataloaders, PAD_IDX, VOCAB
from models.lstm import Lstm

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config loading helpers
# ---------------------------------------------------------------------------
def _load_config(config: Optional[Any]) -> Dict[str, Any]:
    """Load config from a path, dict, or return empty dict if None."""
    if config is None:
        return {}
    if isinstance(config, dict):
        return config
    if isinstance(config, str):
        with open(config, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    raise TypeError("config must be None, a dict, or a path string.")


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


def _resolve_device(config: Dict[str, Any]) -> torch.device:
    """Resolve the device from config or fall back to available hardware."""
    device_str = _safe_get(config, "system", "device", default=None)
    if device_str is not None and str(device_str).strip() != "":
        return torch.device(str(device_str))
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _resolve_optimizer_name(config: Dict[str, Any]) -> str:
    """Resolve optimizer name from config with a safe default."""
    name = _safe_get(config, "training", "optimizer", "name", default=None)
    if name is None or str(name).strip() == "":
        # Paper uses SGD with lr=0.1 for the LSTM controller.
        return "sgd"
    return str(name).lower()


def _resolve_optimizer_kwargs(
    config: Dict[str, Any], model_params: Any
) -> Dict[str, Any]:
    """Build optimizer keyword arguments from config."""
    name = _resolve_optimizer_name(config)
    lr = float(_safe_get(config, "training", "optimizer", "lr", default=0.1))

    kwargs: Dict[str, Any] = {"lr": lr}

    if name in ("sgd", "momentum"):
        momentum = _safe_get(
            config, "training", "optimizer", "momentum", default=None
        )
        if momentum is not None:
            kwargs["momentum"] = float(momentum)

    weight_decay = _safe_get(
        config, "training", "optimizer", "weight_decay", default=None
    )
    if weight_decay is not None:
        kwargs["weight_decay"] = float(weight_decay)

    return kwargs


def _build_optimizer(
    config: Dict[str, Any], model: nn.Module
) -> torch.optim.Optimizer:
    """Build the optimizer specified in config."""
    name = _resolve_optimizer_name(config)
    kwargs = _resolve_optimizer_kwargs(config, model.parameters())

    if name == "adam":
        return torch.optim.Adam(model.parameters(), **kwargs)
    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), **kwargs)
    if name == "rmsprop":
        return torch.optim.RMSprop(model.parameters(), **kwargs)
    # Default to SGD, which is the optimizer used in the paper.
    return torch.optim.SGD(model.parameters(), **kwargs)


def _build_scheduler(
    config: Dict[str, Any], optimizer: torch.optim.Optimizer
) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
    """Build the learning-rate scheduler specified in config."""
    scheduler_name = _safe_get(
        config, "training", "scheduler", "name", default=None
    )
    if scheduler_name is None or str(scheduler_name).strip() == "":
        return None

    scheduler_name = str(scheduler_name).lower()

    if scheduler_name == "steplr":
        step_size = int(
            _safe_get(config, "training", "scheduler", "step_size", default=0)
        )
        gamma = float(
            _safe_get(config, "training", "scheduler", "gamma", default=1.0)
        )
        # PyTorch StepLR requires step_size >= 1.  The config value 0 is
        # interpreted as "no stepping" (identity scheduler).
        if step_size < 1:
            return None
        return torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=step_size, gamma=gamma
        )

    if scheduler_name == "exponentiallr":
        gamma = float(
            _safe_get(config, "training", "scheduler", "gamma", default=1.0)
        )
        return torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)

    if scheduler_name == "reducelronplateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.1, patience=5
        )

    return None


# ---------------------------------------------------------------------------
# Loss function
# ---------------------------------------------------------------------------
def _build_loss_function(config: Dict[str, Any]) -> nn.Module:
    """Build the loss function.

    The paper uses cross-entropy loss for symbol prediction.  We use
    ``nn.CrossEntropyLoss`` with ``ignore_index=PAD_IDX`` so that padded
    positions do not contribute to the loss.
    """
    loss_name = _safe_get(
        config, "training", "loss_function", default="cross_entropy"
    )
    loss_name = str(loss_name).lower()

    if "cross" in loss_name or "entropy" in loss_name:
        return nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    if "mse" in loss_name or "mean_squared" in loss_name:
        return nn.MSELoss()
    # Default to cross-entropy for symbol prediction.
    return nn.CrossEntropyLoss(ignore_index=PAD_IDX)


# ---------------------------------------------------------------------------
# Model initialization
# ---------------------------------------------------------------------------
def initialize_model(
    config: Optional[Any] = None,
    input_size: Optional[int] = None,
    hidden_size: Optional[int] = None,
    num_classes: Optional[int] = None,
    dropout_ratio: Optional[float] = None,
) -> Lstm:
    """Initialize the LSTM model from scratch.

    Parameters
    ----------
    config : dict or str, optional
        Config dict or path to config.yaml.
    input_size : int, optional
        Override for LSTM input size.
    hidden_size : int, optional
        Override for LSTM hidden size.
    num_classes : int, optional
        Override for vocabulary size.
    dropout_ratio : float, optional
        Override for dropout ratio.

    Returns
    -------
    Lstm
        Freshly initialized model with random weights.
    """
    cfg = _load_config(config)

    # Resolve num_classes from vocabulary if not explicitly provided.
    if num_classes is None:
        num_classes = len(VOCAB)

    # Resolve hidden_size from config or paper default.
    if hidden_size is None:
        hidden_size = int(
            _safe_get(cfg, "model", "hidden_size", default=200)
        )

    # Resolve dropout from config.
    if dropout_ratio is None:
        dropout_ratio = _safe_get(
            cfg, "model", "dropout_ratio", default=None
        )
        if dropout_ratio is not None:
            dropout_ratio = float(dropout_ratio)

    model = Lstm(
        config=cfg,
        input_size=input_size,
        hidden_size=hidden_size,
        num_classes=num_classes,
        dropout_ratio=dropout_ratio,
    )

    # Apply kaiming_normal initialization as specified in config.
    weight_init = _safe_get(
        cfg, "training_modes", "scratch", "weight_init", default="kaiming_normal"
    )
    if weight_init == "kaiming_normal":
        for name, param in model.named_parameters():
            if "weight" in name and param.dim() >= 2:
                nn.init.kaiming_normal_(param, mode="fan_in", nonlinearity="relu")
            elif "bias" in name:
                nn.init.zeros_(param)

    logger.info(
        "Initialized LSTM model: input_size=%d, hidden_size=%d, num_classes=%d",
        model.input_size,
        model.hidden_size,
        model.num_classes,
    )
    return model


# ---------------------------------------------------------------------------
# Training / validation helpers
# ---------------------------------------------------------------------------
def _prepare_batch(
    batch: Dict[str, Any], device: torch.device, model: Lstm
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert a batch of token indices into model inputs and targets.

    The LSTM expects a one-hot encoded input of shape
    ``(batch_size, seq_len, input_size)``.  We convert token indices to
    one-hot vectors using the model's input size.
    """
    input_ids = batch["input_ids"].to(device)
    target_ids = batch["target_ids"].to(device)

    batch_size, seq_len = input_ids.shape
    input_size = model.input_size

    # Clamp indices to valid range for one-hot encoding.
    clamped = input_ids.clamp(0, input_size - 1)
    x = torch.zeros(
        batch_size, seq_len, input_size, device=device, dtype=torch.float32
    )
    x.scatter_(2, clamped.unsqueeze(-1), 1.0)

    return x, target_ids, input_ids


def _compute_accuracy(
    logits: torch.Tensor, targets: torch.Tensor, pad_idx: int = PAD_IDX
) -> float:
    """Compute token-level accuracy, ignoring padding positions."""
    predictions = logits.argmax(dim=-1)
    mask = targets != pad_idx
    if mask.sum().item() == 0:
        return 0.0
    correct = (predictions == targets) & mask
    return correct.sum().item() / mask.sum().item()


def _run_epoch(
    model: Lstm,
    dataloader: Any,
    loss_fn: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    device: torch.device,
    training: bool = True,
) -> Tuple[float, float]:
    """Run one epoch of training or validation.

    Returns
    -------
    (avg_loss, avg_accuracy)
    """
    if training:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_accuracy = 0.0
    num_batches = 0

    with torch.set_grad_enabled(training):
        for batch in dataloader:
            x, target_ids, _ = _prepare_batch(batch, device, model)

            if training:
                optimizer.zero_grad()

            logits, _ = model(x)
            # Reshape for CrossEntropyLoss: (batch*seq, num_classes)
            logits_flat = logits.reshape(-1, model.num_classes)
            targets_flat = target_ids.reshape(-1)

            loss = loss_fn(logits_flat, targets_flat)

            if training:
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            total_accuracy += _compute_accuracy(logits, target_ids)
            num_batches += 1

    if num_batches == 0:
        return 0.0, 0.0

    return total_loss / num_batches, total_accuracy / num_batches


def _validate(
    model: Lstm,
    val_loader: Any,
    loss_fn: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """Run validation and return (loss, accuracy)."""
    return _run_epoch(
        model, val_loader, loss_fn, None, device, training=False
    )


# ---------------------------------------------------------------------------
# Checkpoint saving
# ---------------------------------------------------------------------------
def save_model(
    model: nn.Module,
    path: str,
    optimizer: Optional[torch.optim.Optimizer] = None,
    epoch: Optional[int] = None,
    save_optimizer_state: bool = True,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """Save model weights (and optionally optimizer state) to disk.

    Parameters
    ----------
    model : nn.Module
        Model to save.
    path : str
        Output file path.
    optimizer : torch.optim.Optimizer, optional
        Optimizer whose state should be saved.
    epoch : int, optional
        Current epoch number.
    save_optimizer_state : bool
        Whether to include optimizer state in the checkpoint.
    metadata : dict, optional
        Additional metadata to store in the checkpoint.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)

    checkpoint: Dict[str, Any] = {
        "model_state_dict": model.state_dict(),
        "epoch": epoch,
    }

    if save_optimizer_state and optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()

    if metadata is not None:
        checkpoint["metadata"] = metadata

    torch.save(checkpoint, path)
    logger.info("Saved checkpoint to %s", path)


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------
def train_model(
    config: Optional[Any] = None,
    model: Optional[Lstm] = None,
    train_loader: Optional[Any] = None,
    val_loader: Optional[Any] = None,
    test_loader: Optional[Any] = None,
    epochs: Optional[int] = None,
    batch_size: Optional[int] = None,
    device: Optional[torch.device] = None,
    checkpoint_dir: str = "checkpoints",
    save_best: Optional[bool] = None,
    save_final: Optional[bool] = None,
    best_metric: str = "val_accuracy",
    best_mode: str = "max",
    save_optimizer_state: Optional[bool] = None,
    resume_from: Optional[str] = None,
) -> Dict[str, Any]:
    """Train the LSTM model.

    Parameters
    ----------
    config : dict or str, optional
        Config dict or path to config.yaml.
    model : Lstm, optional
        Model to train.  If None, a new model is initialized.
    train_loader : DataLoader, optional
        Training data loader.  If None, it is built from config.
    val_loader : DataLoader, optional
        Validation data loader.  May be None in strict mode.
    test_loader : DataLoader, optional
        Test data loader.  Only used for final evaluation.
    epochs : int, optional
        Number of training epochs.  Overrides config.
    batch_size : int, optional
        Batch size.  Overrides config.
    device : torch.device, optional
        Device to train on.  Overrides config.
    checkpoint_dir : str
        Directory for saving checkpoints.
    save_best : bool, optional
        Whether to save the best model.  Overrides config.
    save_final : bool, optional
        Whether to save the final model.  Overrides config.
    best_metric : str
        Metric used for best-model selection.
    best_mode : str
        'max' or 'min' for best metric comparison.
    save_optimizer_state : bool, optional
        Whether to save optimizer state.  Overrides config.
    resume_from : str, optional
        Path to a checkpoint to resume from.

    Returns
    -------
    Dict[str, Any]
        Training history containing per-epoch metrics.
    """
    cfg = _load_config(config)

    # Resolve reproduction mode.
    reproduction_mode = str(
        _safe_get(cfg, "reproduction_mode", default="engineering")
    ).lower()

    # Resolve device.
    if device is None:
        device = _resolve_device(cfg)

    # Resolve epochs.
    if epochs is None:
        epochs = int(_safe_get(cfg, "training", "epochs", default=30))

    # Resolve batch size.
    if batch_size is None:
        batch_size = int(
            _safe_get(cfg, "training", "batch_size", "default", default=20)
        )

    # Resolve checkpoint settings.
    if save_best is None:
        save_best = bool(
            _safe_get(cfg, "checkpoint", "save_best", default=True)
        )
    if save_final is None:
        save_final = bool(
            _safe_get(cfg, "checkpoint", "save_final", default=True)
        )
    if save_optimizer_state is None:
        save_optimizer_state = bool(
            _safe_get(cfg, "checkpoint", "save_optimizer_state", default=True)
        )
    if resume_from is None:
        resume_from = _safe_get(cfg, "checkpoint", "resume_from", default=None)

    # Resolve best metric settings from config if not explicitly overridden.
    if best_metric == "val_accuracy":
        best_metric = str(
            _safe_get(cfg, "checkpoint", "best_metric", default="val_accuracy")
        )
    if best_mode == "max":
        best_mode = str(
            _safe_get(cfg, "checkpoint", "best_mode", default="max")
        )

    # Build dataloaders if not provided.
    if train_loader is None or val_loader is None or test_loader is None:
        # Read dataset version from experiment runs.
        dataset_version = str(
            _safe_get(
                cfg,
                "experiment",
                "runs",
                0,
                "dataset_version",
                default="default",
            )
        )
        loaders = get_dataloaders(
            config=cfg,
            version=dataset_version,
            batch_size=batch_size,
            num_workers=int(_safe_get(cfg, "system", "num_workers", default=0)),
            seed=int(_safe_get(cfg, "system", "seed", default=42)),
        )
        if train_loader is None:
            train_loader = loaders["train"]
        if val_loader is None:
            val_loader = loaders.get("val")
        if test_loader is None:
            test_loader = loaders.get("test")

    # In strict mode, force val_loader to None.
    if reproduction_mode == "strict":
        val_loader = None

    # Initialize model if not provided.
    if model is None:
        model = initialize_model(config=cfg)
    model = model.to(device)

    # Build loss function and optimizer.
    loss_fn = _build_loss_function(cfg)
    optimizer = _build_optimizer(cfg, model)
    scheduler = _build_scheduler(cfg, optimizer)

    # Resume from checkpoint if requested.
    start_epoch = 0
    if resume_from is not None and os.path.exists(resume_from):
        checkpoint = torch.load(resume_from, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        if save_optimizer_state and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "epoch" in checkpoint and checkpoint["epoch"] is not None:
            start_epoch = int(checkpoint["epoch"]) + 1
        logger.info("Resumed from checkpoint %s at epoch %d", resume_from, start_epoch)

    # Training history.
    history: Dict[str, Any] = {
        "train_loss": [],
        "train_accuracy": [],
        "val_loss": [],
        "val_accuracy": [],
        "best_epoch": None,
        "best_val_accuracy": None,
    }

    best_score: Optional[float] = None
    best_model_path = os.path.join(checkpoint_dir, "best_model.pt")
    final_model_path = os.path.join(checkpoint_dir, "final_model.pt")

    os.makedirs(checkpoint_dir, exist_ok=True)

    for epoch in range(start_epoch, epochs):
        # Training.
        train_loss, train_acc = _run_epoch(
            model, train_loader, loss_fn, optimizer, device, training=True
        )
        history["train_loss"].append(train_loss)
        history["train_accuracy"].append(train_acc)

        # Validation (guarded by val_loader is not None).
        val_loss: Optional[float] = None
        val_acc: Optional[float] = None
        if val_loader is not None:
            val_loss, val_acc = _validate(model, val_loader, loss_fn, device)
            history["val_loss"].append(val_loss)
            history["val_accuracy"].append(val_acc)

            # Best model selection.
            if save_best and val_acc is not None:
                if best_mode == "max":
                    is_better = best_score is None or val_acc > best_score
                else:
                    is_better = best_score is None or val_acc < best_score

                if is_better:
                    best_score = val_acc
                    history["best_epoch"] = epoch
                    history["best_val_accuracy"] = val_acc
                    save_model(
                        model,
                        best_model_path,
                        optimizer=optimizer,
                        epoch=epoch,
                        save_optimizer_state=save_optimizer_state,
                        metadata={
                            "epoch": epoch,
                            "val_accuracy": val_acc,
                            "val_loss": val_loss,
                            "train_loss": train_loss,
                            "train_accuracy": train_acc,
                        },
                    )

        # Scheduler step.
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                if val_loss is not None:
                    scheduler.step(val_loss)
            else:
                scheduler.step()

        log_msg = (
            f"Epoch {epoch + 1}/{epochs} - "
            f"train_loss: {train_loss:.4f}, train_acc: {train_acc:.4f}"
        )
        if val_loss is not None and val_acc is not None:
            log_msg += f", val_loss: {val_loss:.4f}, val_acc: {val_acc:.4f}"
        logger.info(log_msg)

    # Save final model.
    if save_final:
        save_model(
            model,
            final_model_path,
            optimizer=optimizer,
            epoch=epochs - 1,
            save_optimizer_state=save_optimizer_state,
            metadata={
                "epoch": epochs - 1,
                "train_loss": history["train_loss"][-1] if history["train_loss"] else None,
                "train_accuracy": history["train_accuracy"][-1] if history["train_accuracy"] else None,
            },
        )

    # Final test evaluation (test set is never used during training).
    if test_loader is not None:
        test_loss, test_acc = _run_epoch(
            model, test_loader, loss_fn, None, device, training=False
        )
        history["test_loss"] = test_loss
        history["test_accuracy"] = test_acc
        logger.info(
            "Final test evaluation - test_loss: %.4f, test_accuracy: %.4f",
            test_loss,
            test_acc,
        )

    return history


# ---------------------------------------------------------------------------
# Convenience entry point
# ---------------------------------------------------------------------------
def main(config_path: str = "config/config.yaml") -> Dict[str, Any]:
    """Load config and run training."""
    cfg = _load_config(config_path)
    return train_model(config=cfg)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    main()