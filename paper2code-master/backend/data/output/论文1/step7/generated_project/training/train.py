"""Training pipeline for the plant-disease detection reproduction (Step 5).

Implements the model-training flow required by FR-06 / FR-07 / FR-08:

* FR-06 - two independent training modes, ``scratch`` and ``transfer``, selected through the
  ``training_mode`` enum (never through a single ``pretrained`` boolean, because the two
  initialisation strategies are fundamentally different).
* FR-07 - Stochastic Gradient Descent (lr=0.005, momentum=0.9, weight_decay=0.0005) for 30
  epochs with softmax cross-entropy loss and a StepLR(step_size=10, gamma=0.1) schedule.
* FR-08 - saving of the best model weights (selected by validation accuracy, engineering
  deviation from the paper) and of the final model weights.

Paper protocol note (FR-03): the paper only uses train/test splits. When
``reproduction_mode == 'engineering'`` a validation set is carved out of the training set
(80/20, leaf-constraint aware) and used for best-checkpoint selection. In ``strict`` mode no
validation split exists, ``val_loader`` stays ``None`` and the validation branch is skipped.
"""

import argparse
import json
import logging
import os
import random
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, Dataset, TensorDataset

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from data.dataset import PlantVillageDataset
from models.alexnet import Alexnet
from models.googlenet import Googlenet

LOGGER = logging.getLogger(__name__)
if not LOGGER.handlers:
    _STREAM_HANDLER = logging.StreamHandler(stream=sys.stdout)
    _STREAM_HANDLER.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s"))
    LOGGER.addHandler(_STREAM_HANDLER)
LOGGER.setLevel(logging.INFO)

_DEFAULT_CONFIG_PATH = os.path.join(_PROJECT_ROOT, "config", "config.yaml")

DEFAULT_NUM_CLASSES = 38
DEFAULT_EPOCHS = 30
DEFAULT_LR = 0.005
DEFAULT_MOMENTUM = 0.9
DEFAULT_WEIGHT_DECAY = 0.0005
DEFAULT_NESTEROV = False
DEFAULT_STEP_SIZE = 10
DEFAULT_GAMMA = 0.1
DEFAULT_BATCH_SIZE_GOOGLENET = 24
DEFAULT_BATCH_SIZE_ALEXNET = 100
DEFAULT_DROPOUT_RATIO = 0.5
DEFAULT_IMAGE_SIZE = 256
DEFAULT_SEED = 42
DEFAULT_NUM_WORKERS = 4
DEFAULT_LOG_INTERVAL = 50

_TRAINING_MODES = ("scratch", "transfer")
_MODEL_NAMES = ("googlenet", "alexnet")
_RESET_LAYER_DEFAULTS = {
    "alexnet": ("classifier.6",),
    "googlenet": ("fc", "aux1.fc2", "aux2.fc2"),
}


# --------------------------------------------------------------------------- #
# Configuration helpers
# --------------------------------------------------------------------------- #
def load_config(config: Optional[Union[str, Dict[str, Any]]] = None) -> Dict[str, Any]:
    """Return the project configuration as a plain dictionary.

    ``config`` may be ``None`` (default ``config/config.yaml`` is read), a path to a YAML file
    or an already loaded dictionary.  Every ``open()`` of a YAML file uses ``encoding='utf-8'``.
    """
    if isinstance(config, dict):
        return config
    if config is None:
        config = _DEFAULT_CONFIG_PATH
    if isinstance(config, (str, os.PathLike)):
        path = os.fspath(config)
        if not os.path.isfile(path):
            LOGGER.warning("Config file %s not found; falling back to built-in defaults.", path)
            return {}
        try:
            with open(path, "r", encoding="utf-8") as handle:
                loaded = yaml.safe_load(handle)
            return loaded if isinstance(loaded, dict) else {}
        except Exception as exc:  # pragma: no cover - defensive
            LOGGER.warning("Failed to read config %s (%s); using defaults.", path, exc)
            return {}
    return {}


def _cfg_get(config: Optional[Dict[str, Any]], dotted_key: str, default: Any = None) -> Any:
    """Read a value from a config dict, supporting both nested and dotted keys."""
    if not isinstance(config, dict):
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


def _to_float(value: Any, default: float, low: Optional[float] = None, high: Optional[float] = None) -> float:
    """Coerce ``value`` to a finite float inside the optional ``[low, high]`` range."""
    fallback = float(default)
    try:
        result = float(value)
    except (TypeError, ValueError):
        return fallback
    if result != result:  # NaN
        return fallback
    if low is not None and result < low:
        return fallback
    if high is not None and result > high:
        return fallback
    return result


def _to_int(value: Any, default: int, low: int = 0) -> int:
    """Coerce ``value`` to an int >= ``low``."""
    fallback = int(default)
    try:
        result = int(value)
    except (TypeError, ValueError):
        return fallback
    return result if result >= low else fallback


def _to_bool(value: Any, default: bool) -> bool:
    """Coerce YAML-ish values to ``bool``."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in ("true", "yes", "1", "y"):
            return True
        if lowered in ("false", "no", "0", "n"):
            return False
    return bool(default)


# --------------------------------------------------------------------------- #
# Reproducibility / device helpers
# --------------------------------------------------------------------------- #
def set_seed(seed: int = DEFAULT_SEED, deterministic: bool = True) -> int:
    """Seed python / torch RNGs and configure cuDNN for deterministic behaviour."""
    seed = _to_int(seed, DEFAULT_SEED, low=0)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = bool(deterministic)
    torch.backends.cudnn.benchmark = not bool(deterministic)
    return seed


def resolve_device(device: Optional[Union[str, torch.device]] = None,
                   config: Optional[Union[str, Dict[str, Any]]] = None) -> torch.device:
    """Resolve the torch device, falling back to CPU when CUDA is unavailable."""
    cfg = load_config(config)
    if isinstance(device, torch.device):
        return device
    if device is None:
        device = _cfg_get(cfg, "inference.device", None) or _cfg_get(cfg, "system.device", "cuda")
    device_str = str(device).strip().lower()
    if device_str.startswith("cuda"):
        if torch.cuda.is_available():
            gpu_id = _cfg_get(cfg, "system.gpu_id", None)
            if gpu_id is not None:
                try:
                    gpu_index = int(gpu_id)
                except (TypeError, ValueError):
                    gpu_index = None
                if gpu_index is not None and 0 <= gpu_index < torch.cuda.device_count():
                    return torch.device("cuda", gpu_index)
            return torch.device("cuda")
        LOGGER.warning("CUDA requested but unavailable; falling back to CPU.")
        return torch.device("cpu")
    if device_str.startswith("cpu"):
        return torch.device("cpu")
    try:
        return torch.device(device_str)
    except (TypeError, ValueError):
        LOGGER.warning("Unrecognised device %r; falling back to CPU.", device)
        return torch.device("cpu")


# --------------------------------------------------------------------------- #
# Model construction helpers
# --------------------------------------------------------------------------- #
def _get_module_by_name(model: nn.Module, dotted_name: str) -> Optional[nn.Module]:
    """Return the submodule addressed by ``dotted_name`` (``None`` when absent)."""
    module: Any = model
    for part in dotted_name.split("."):
        if not part:
            continue
        if part.isdigit() and isinstance(module, (nn.Sequential, nn.ModuleList)):
            index = int(part)
            if index >= len(module):
                return None
            module = module[index]
        else:
            module = getattr(module, part, None)
        if module is None:
            return None
    return module if isinstance(module, nn.Module) else None


def _init_layer_kaiming(module: nn.Module) -> None:
    """Re-initialise ``module`` with kaiming_normal weights and zero bias."""
    if isinstance(module, (nn.Linear, nn.Conv2d)):
        nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
        if module.weight is not None:
            nn.init.ones_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


def _reset_classification_layers(model: nn.Module, model_name: str, num_classes: int) -> List[str]:
    """Reset (and re-initialise) the classification heads of ``model``."""
    reset: List[str] = []
    for name in _RESET_LAYER_DEFAULTS.get(model_name, ()):
        module = _get_module_by_name(model, name)
        if isinstance(module, nn.Linear):
            _init_layer_kaiming(module)
            reset.append(name)
    if not reset:
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear) and getattr(module, "out_features", None) == num_classes:
                _init_layer_kaiming(module)
                reset.append(name)
    return reset


def _load_image_net_weights(model: nn.Module, model_name: str) -> bool:
    """Copy ImageNet-pretrained torchvision weights into ``model`` (shape-matched tensors)."""
    try:
        from torchvision import models as tv_models
    except Exception as exc:  # pragma: no cover - torchvision missing
        LOGGER.warning("torchvision unavailable (%s); pretrained weights not loaded.", exc)
        return False

    donor = None
    try:
        if model_name == "alexnet":
            donor = tv_models.alexnet(pretrained=True)
        elif model_name == "googlenet":
            donor = tv_models.googlenet(pretrained=True, aux_logits=True, init_weights=False)
    except TypeError:
        try:
            if model_name == "alexnet":
                donor = tv_models.alexnet(weights="IMAGENET1K_V1")
            elif model_name == "googlenet":
                donor = tv_models.googlenet(weights="IMAGENET1K_V1", aux_logits=True, init_weights=False)
        except Exception as exc:
            LOGGER.warning("Could not download ImageNet weights for %s (%s).", model_name, exc)
            return False
    except Exception as exc:
        LOGGER.warning("Could not download ImageNet weights for %s (%s).", model_name, exc)
        return False

    if donor is None:
        return False

    donor_state = donor.state_dict()
    own_state = model.state_dict()
    matched = {
        key: value for key, value in donor_state.items()
        if key in own_state and own_state[key].shape == value.shape
    }
    if not matched:
        LOGGER.warning("No matching tensors between torchvision and project %s model.", model_name)
        return False
    own_state.update(matched)
    model.load_state_dict(own_state)
    LOGGER.info("Loaded %d/%d ImageNet tensors into '%s'.", len(matched), len(own_state), model_name)
    return True


def initialize_model(config: Optional[Union[str, Dict[str, Any]]] = None,
                     model_name: Optional[str] = None,
                     training_mode: Optional[str] = None,
                     num_classes: Optional[int] = None,
                     device: Optional[Union[str, torch.device]] = None,
                     **kwargs: Any) -> nn.Module:
    """Build and initialise a plant-disease classifier (FR-06).

    Parameters
    ----------
    config:
        Configuration mapping or path to ``config/config.yaml``.
    model_name:
        ``'googlenet'`` (default) or ``'alexnet'``.
    training_mode:
        ``'scratch'`` (all layers kaiming_normal) or ``'transfer'`` (ImageNet weights +
        re-initialised classification heads).
    num_classes:
        Number of output classes (defaults to ``model.num_classes`` = 38).
    device:
        Optional device the model is moved to.
    """
    cfg = load_config(config)

    resolved_name = str(model_name or _cfg_get(cfg, "model.default", "googlenet")).strip().lower()
    if resolved_name not in _MODEL_NAMES:
        raise ValueError(f"Unknown model {resolved_name!r}; expected one of {_MODEL_NAMES}.")

    resolved_mode = str(training_mode or _cfg_get(cfg, "training_modes.active", "transfer")).strip().lower()
    if resolved_mode not in _TRAINING_MODES:
        raise ValueError(f"Unknown training_mode {resolved_mode!r}; expected one of {_TRAINING_MODES}.")

    classes = _to_int(
        num_classes if num_classes is not None else _cfg_get(cfg, "model.num_classes", DEFAULT_NUM_CLASSES),
        DEFAULT_NUM_CLASSES,
        low=1,
    )
    dropout_ratio = _to_float(_cfg_get(cfg, "model.dropout_ratio", DEFAULT_DROPOUT_RATIO), DEFAULT_DROPOUT_RATIO, 0.0, 1.0)
    in_channels = _to_int(_cfg_get(cfg, "model.in_channels", 3), 3, low=1)

    if resolved_name == "alexnet":
        model = Alexnet(config=cfg, num_classes=classes, dropout_ratio=dropout_ratio, in_channels=in_channels)
    else:
        aux_logits = _to_bool(_cfg_get(cfg, "model.googlenet.aux_logits", True), True)
        init_weights = _to_bool(_cfg_get(cfg, "model.googlenet.init_weights", False), False)
        model = Googlenet(
            config=cfg,
            num_classes=classes,
            aux_logits=aux_logits,
            dropout_ratio=dropout_ratio,
            init_weights=init_weights,
        )

    pretrained = _to_bool(
        _cfg_get(cfg, f"training_modes.{resolved_mode}.pretrained", resolved_mode == "transfer"),
        resolved_mode == "transfer",
    )
    if pretrained:
        _load_image_net_weights(model, resolved_name)

    _reset_classification_layers(model, resolved_name, classes)

    if not pretrained:
        for module in model.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear, nn.BatchNorm1d, nn.BatchNorm2d)):
                _init_layer_kaiming(module)

    freeze_backbone = _to_bool(
        _cfg_get(cfg, f"training_modes.{resolved_mode}.freeze_backbone", False),
        False,
    )
    if freeze_backbone:
        head_names = tuple(_RESET_LAYER_DEFAULTS.get(resolved_name, ()))
        for name, param in model.named_parameters():
            param.requires_grad = not any(name.startswith(head) for head in head_names)
        LOGGER.info("Backbone frozen; only %s layers are trainable.", ", ".join(head_names) or "classifier")

    if device is not None:
        model = model.to(resolve_device(device, cfg))

    LOGGER.info("Initialised '%s' (%s mode, %d classes).", resolved_name, resolved_mode, classes)
    return model


# --------------------------------------------------------------------------- #
# Checkpoint helpers
# --------------------------------------------------------------------------- #
def _checkpoint_filename(cfg: Dict[str, Any],
                         model_name: str,
                         training_mode: str,
                         dataset_version: Any,
                         train_test_split: Any,
                         suffix: str = "best") -> str:
    """Build a checkpoint filename from ``checkpoint.filename_pattern``."""
    pattern = _cfg_get(
        cfg,
        "checkpoint.filename_pattern",
        "{model}_{training_mode}_{dataset_version}_{train_test_split}_best.pth",
    )
    try:
        name = str(pattern).format(
            model=model_name,
            training_mode=training_mode,
            dataset_version=dataset_version,
            train_test_split=train_test_split,
            suffix=suffix,
        )
    except Exception:
        name = f"{model_name}_{training_mode}_{dataset_version}_{train_test_split}_best.pth"

    if suffix and suffix != "best":
        root, ext = os.path.splitext(name)
        if root.endswith("_best"):
            root = root[: -len("_best")]
        name = f"{root}_{suffix}{ext or '.pth'}"
    return name


def save_model(model: nn.Module,
               path: Optional[Union[str, os.PathLike]] = None,
               config: Optional[Union[str, Dict[str, Any]]] = None,
               optimizer: Optional[torch.optim.Optimizer] = None,
               scheduler: Optional[Any] = None,
               epoch: Optional[int] = None,
               metrics: Optional[Dict[str, Any]] = None,
               model_name: Optional[str] = None,
               training_mode: Optional[str] = None,
               dataset_version: Optional[str] = None,
               train_test_split: Optional[str] = None,
               checkpoint_dir: Optional[Union[str, os.PathLike]] = None,
               save_optimizer_state: Optional[bool] = None,
               extra: Optional[Dict[str, Any]] = None,
               **kwargs: Any) -> str:
    """Persist ``model`` (and optional training state) to disk; return the saved path.

    ``path`` may be

    * ``None``               -> ``checkpoint.dir`` + ``checkpoint.filename_pattern``,
    * an existing directory  -> the pattern filename is created inside it,
    * a file path            -> used verbatim.
    """
    cfg = load_config(config)
    if model is None:
        raise ValueError("save_model requires a model instance.")

    filename = kwargs.get("filename")
    resolved_mode = str(training_mode or _cfg_get(cfg, "training_modes.active", "transfer")).strip().lower()
    resolved_name = str(model_name or _cfg_get(cfg, "model.default", "googlenet")).strip().lower()

    target: Optional[str] = None
    if path is not None:
        target = os.fspath(path)
    if target is None or os.path.isdir(target) or target.endswith(os.sep):
        directory = target if target else (checkpoint_dir or _cfg_get(cfg, "checkpoint.dir", os.path.join(_PROJECT_ROOT, "checkpoints")))
        directory = os.fspath(directory)
        if filename is None:
            filename = _checkpoint_filename(
                cfg,
                resolved_name,
                resolved_mode,
                dataset_version if dataset_version is not None else _cfg_get(cfg, "data.default_version", "color"),
                train_test_split if train_test_split is not None else _cfg_get(cfg, "split_strategy.default_train_ratio", 0.8),
                suffix=str(kwargs.get("suffix", "best")),
            )
        target = os.path.join(directory, str(filename))
    elif filename is not None:
        target = os.path.join(target, str(filename))

    if not str(target).lower().endswith(".pth"):
        target = f"{target}.pth"

    directory = os.path.dirname(os.path.abspath(target))
    if directory:
        os.makedirs(directory, exist_ok=True)

    if save_optimizer_state is None:
        save_optimizer_state = _to_bool(_cfg_get(cfg, "checkpoint.save_optimizer_state", True), True)

    state_dict = model.state_dict()
    payload: Dict[str, Any] = {
        "model_state_dict": state_dict,
        "state_dict": state_dict,
        "model_name": resolved_name,
        "training_mode": resolved_mode,
        "num_classes": getattr(model, "num_classes", None),
        "epoch": epoch,
        "metrics": metrics,
        "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "pytorch_version": torch.__version__,
    }
    if save_optimizer_state and optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    if scheduler is not None:
        try:
            payload["scheduler_state_dict"] = scheduler.state_dict()
        except Exception:  # pragma: no cover - defensive
            pass
    if isinstance(extra, dict):
        payload.update(extra)

    torch.save(payload, target)
    LOGGER.info("Checkpoint saved to %s (epoch=%s).", target, epoch)
    return target


# --------------------------------------------------------------------------- #
# Data helpers
# --------------------------------------------------------------------------- #
def _unpack_batch(batch: Any) -> Tuple[torch.Tensor, torch.Tensor]:
    """Extract ``(inputs, targets)`` from a dataloader batch."""
    if isinstance(batch, dict):
        inputs = None
        targets = None
        for key in ("image", "images", "input", "inputs", "x"):
            if key in batch:
                inputs = batch[key]
                break
        for key in ("label", "labels", "target", "targets", "y"):
            if key in batch:
                targets = batch[key]
                break
        if inputs is not None and targets is not None:
            return inputs, targets
        raise ValueError("Batch dictionary must contain image and label entries.")
    if isinstance(batch, (tuple, list)) and len(batch) >= 2:
        return batch[0], batch[1]
    raise ValueError("Unsupported batch structure; expected (inputs, targets).")


def _extract_outputs(outputs: Any) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    """Return ``(main_logits, aux_logits_list)`` from a forward-pass output."""
    if isinstance(outputs, (tuple, list)):
        main = outputs[0]
        aux = [item for item in outputs[1:] if item is not None]
        return main, aux
    return outputs, []


def _compute_loss(outputs: Any, targets: torch.Tensor, criterion: nn.Module) -> Tuple[torch.Tensor, torch.Tensor]:
    """Cross-entropy over the main logits plus 0.3-weighted auxiliary logits."""
    main, aux = _extract_outputs(outputs)
    loss = criterion(main, targets)
    for aux_logits in aux:
        loss = loss + 0.3 * criterion(aux_logits, targets)
    return loss, main


def _run_one_epoch(model: nn.Module,
                   loader: DataLoader,
                   criterion: nn.Module,
                   optimizer: Optional[torch.optim.Optimizer],
                   device: torch.device,
                   log_interval: int = DEFAULT_LOG_INTERVAL,
                   scaler: Optional[Any] = None,
                   grad_clip: Optional[float] = None,
                   epoch: int = 1,
                   tag: str = "train") -> Dict[str, Any]:
    """Run a single pass over ``loader`` (training when ``optimizer`` is given)."""
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    total_correct = 0
    num_samples = 0
    num_batches = 0
    start_time = time.time()

    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for batch_idx, batch in enumerate(loader):
            inputs, targets = _unpack_batch(batch)
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            if targets.dtype != torch.long:
                targets = targets.long()

            if training:
                optimizer.zero_grad(set_to_none=True)

            outputs = model(inputs)
            loss, logits = _compute_loss(outputs, targets, criterion)

            if training:
                if scaler is not None:
                    try:
                        scaler.scale(loss).backward()
                        if grad_clip is not None:
                            scaler.unscale_(optimizer)
                            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                        scaler.step(optimizer)
                        scaler.update()
                    except AttributeError:
                        loss.backward()
                        if grad_clip is not None:
                            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                        optimizer.step()
                else:
                    loss.backward()
                    if grad_clip is not None:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    optimizer.step()

            batch_size = int(targets.size(0))
            total_loss += float(loss.detach().item()) * batch_size
            predictions = logits.detach().argmax(dim=1)
            total_correct += int((predictions == targets).sum().item())
            num_samples += batch_size
            num_batches += 1

            if log_interval and num_batches % log_interval == 0:
                LOGGER.info(
                    "%s | epoch %d | batch %d/%d | loss %.4f | acc %.4f",
                    tag, epoch, num_batches, len(loader),
                    total_loss / max(num_samples, 1),
                    total_correct / max(num_samples, 1),
                )

    elapsed = time.time() - start_time
    metrics: Dict[str, Any] = {
        "loss": total_loss / max(num_samples, 1),
        "accuracy": total_correct / max(num_samples, 1),
        "correct": total_correct,
        "num_samples": num_samples,
        "num_batches": num_batches,
        "seconds": elapsed,
    }
    LOGGER.info(
        "%s | epoch %d | loss %.4f | accuracy %.4f (%d/%d) | %.1fs",
        tag, epoch, metrics["loss"], metrics["accuracy"],
        total_correct, num_samples, elapsed,
    )
    return metrics


def evaluate_model(model: nn.Module,
                   loader: DataLoader,
                   criterion: Optional[nn.Module] = None,
                   device: Optional[Union[str, torch.device]] = None,
                   config: Optional[Union[str, Dict[str, Any]]] = None,
                   log_interval: int = DEFAULT_LOG_INTERVAL,
                   epoch: int = 0,
                   **kwargs: Any) -> Dict[str, Any]:
    """Evaluate ``model`` on ``loader`` (no gradient updates) and return metrics."""
    cfg = load_config(config)
    resolved_device = resolve_device(device, cfg) if not isinstance(device, torch.device) else device
    if criterion is None:
        criterion = nn.CrossEntropyLoss()
    return _run_one_epoch(
        model, loader, criterion, None, resolved_device,
        log_interval=log_interval, epoch=epoch, tag="eval",
    )


# --------------------------------------------------------------------------- #
# Data-loader construction helpers
# --------------------------------------------------------------------------- #
def _resolve_batch_size(cfg: Dict[str, Any], model_name: str, batch_size: Any = None) -> int:
    """Per-model batch size (GoogLeNet 24, AlexNet 100) with sane fallbacks."""
    if batch_size is not None:
        return _to_int(batch_size, DEFAULT_BATCH_SIZE_GOOGLENET, low=1)
    value = _cfg_get(cfg, f"training.batch_size.{model_name}", None)
    if value is None:
        value = _cfg_get(cfg, "training.batch_size", None)
    if value is None:
        value = DEFAULT_BATCH_SIZE_GOOGLENET if model_name == "googlenet" else DEFAULT_BATCH_SIZE_ALEXNET
    return _to_int(value, DEFAULT_BATCH_SIZE_GOOGLENET, low=1)


def _as_dataset(value: Any, base: Optional[Dataset] = None, split: Optional[str] = None) -> Optional[Dataset]:
    """Normalise a split entry into a ``Dataset`` (indices -> ``base._subset``)."""
    if value is None:
        return None
    if isinstance(value, Dataset):
        return value
    if isinstance(value, (list, tuple)) and base is not None:
        subset = getattr(base, "_subset", None)
        if callable(subset):
            try:
                return subset(list(value), split=split)
            except TypeError:
                return subset(list(value))
    return None


def _make_loader(dataset: Optional[Dataset],
                 batch_size: int,
                 shuffle: bool,
                 num_workers: int,
                 pin_memory: bool,
                 drop_last: bool = False) -> Optional[DataLoader]:
    """Wrap ``dataset`` in a ``DataLoader`` (``None`` when the dataset is missing).

    When the dataset has zero samples the shuffle flag is forced to ``False`` so that
    ``RandomSampler`` is never instantiated with ``num_samples=0`` (which would raise
    ``ValueError: num_samples should be a positive integer value``).
    """
    if dataset is None:
        return None
    try:
        num_items = len(dataset)
    except (TypeError, NotImplementedError):
        num_items = None
    if num_items == 0:
        shuffle = False
        drop_last = False
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )


def build_dataloaders(config: Optional[Union[str, Dict[str, Any]]] = None,
                      dataset: Optional[PlantVillageDataset] = None,
                      model_name: Optional[str] = None,
                      train_loader: Optional[DataLoader] = None,
                      val_loader: Optional[DataLoader] = None,
                      test_loader: Optional[DataLoader] = None,
                      batch_size: Optional[int] = None,
                      **kwargs: Any) -> Tuple[Optional[DataLoader], Optional[DataLoader], Optional[DataLoader], Any]:
    """Create ``(train_loader, val_loader, test_loader, dataset)`` from the project config."""
    cfg = load_config(config)
    resolved_name = str(model_name or _cfg_get(cfg, "model.default", "googlenet")).strip().lower()
    resolved_batch = _resolve_batch_size(cfg, resolved_name, batch_size)
    num_workers = _to_int(
        kwargs.get("num_workers", _cfg_get(cfg, "training.num_workers", DEFAULT_NUM_WORKERS)),
        DEFAULT_NUM_WORKERS,
    )
    pin_memory = _to_bool(kwargs.get("pin_memory", _cfg_get(cfg, "training.pin_memory", True)), True)
    shuffle_train = _to_bool(kwargs.get("shuffle", _cfg_get(cfg, "training.shuffle_train", True)), True)
    drop_last = _to_bool(kwargs.get("drop_last", _cfg_get(cfg, "training.drop_last", False)), False)

    if train_loader is not None or val_loader is not None or test_loader is not None:
        return train_loader, val_loader, test_loader, dataset

    if dataset is None:
        dataset = PlantVillageDataset(config=cfg, seed=kwargs.get("seed", None),
                                      reproduction_mode=kwargs.get("reproduction_mode", None))

    train_ds: Optional[Dataset] = dataset
    val_ds: Optional[Dataset] = None
    test_ds: Optional[Dataset] = None

    split_fn = getattr(dataset, "split_dataset", None)
    if callable(split_fn):
        try:
            split = split_fn(
                train_ratio=kwargs.get("train_ratio", _cfg_get(cfg, "split_strategy.default_train_ratio", 0.8)),
                val_ratio=kwargs.get("val_ratio", None),
                seed=kwargs.get("seed", _cfg_get(cfg, "split_strategy.seed", DEFAULT_SEED)),
                stratify=_to_bool(kwargs.get("stratify", _cfg_get(cfg, "split_strategy.stratify_by_class", True)), True),
                leaf_constraint=_to_bool(kwargs.get("leaf_constraint", True), True),
                reproduction_mode=kwargs.get("reproduction_mode", _cfg_get(cfg, "reproduction_mode", "engineering")),
                shuffle=shuffle_train,
            )
        except Exception as exc:  # pragma: no cover - defensive
            LOGGER.warning("split_dataset failed (%s); using the full dataset for training.", exc)
            split = None
        if isinstance(split, dict):
            train_ds = _as_dataset(split.get("train"), dataset, "train") or dataset
            val_ds = _as_dataset(split.get("val", split.get("validation")), dataset, "val")
            test_ds = _as_dataset(split.get("test"), dataset, "test")

    train_loader = _make_loader(train_ds, resolved_batch, shuffle_train, num_workers, pin_memory, drop_last)
    val_loader = _make_loader(val_ds, resolved_batch, False, num_workers, pin_memory, False)
    test_loader = _make_loader(test_ds, resolved_batch, False, num_workers, pin_memory, False)
    return train_loader, val_loader, test_loader, dataset


# --------------------------------------------------------------------------- #
# Optimizer / scheduler / criterion helpers
# --------------------------------------------------------------------------- #
def build_optimizer(model: nn.Module,
                    config: Optional[Union[str, Dict[str, Any]]] = None,
                    **kwargs: Any) -> torch.optim.Optimizer:
    """SGD (lr=0.005, momentum=0.9, weight_decay=0.0005) over the trainable parameters (FR-07)."""
    cfg = load_config(config)
    lr = _to_float(kwargs.get("lr", _cfg_get(cfg, "training.optimizer.lr", DEFAULT_LR)), DEFAULT_LR, low=0.0)
    momentum = _to_float(kwargs.get("momentum", _cfg_get(cfg, "training.optimizer.momentum", DEFAULT_MOMENTUM)), DEFAULT_MOMENTUM, low=0.0)
    weight_decay = _to_float(kwargs.get("weight_decay", _cfg_get(cfg, "training.optimizer.weight_decay", DEFAULT_WEIGHT_DECAY)), DEFAULT_WEIGHT_DECAY, low=0.0)
    nesterov = _to_bool(kwargs.get("nesterov", _cfg_get(cfg, "training.optimizer.nesterov", DEFAULT_NESTEROV)), DEFAULT_NESTEROV)

    params = [param for param in model.parameters() if param.requires_grad]
    return torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay, nesterov=nesterov)


def build_scheduler(optimizer: torch.optim.Optimizer,
                    config: Optional[Union[str, Dict[str, Any]]] = None,
                    **kwargs: Any) -> Any:
    """StepLR(step_size=10, gamma=0.1) - multiply LR by 0.1 every 10 epochs (FR-07)."""
    cfg = load_config(config)
    step_size = _to_int(kwargs.get("step_size", _cfg_get(cfg, "training.scheduler.step_size", DEFAULT_STEP_SIZE)), DEFAULT_STEP_SIZE, low=1)
    gamma = _to_float(kwargs.get("gamma", _cfg_get(cfg, "training.scheduler.gamma", DEFAULT_GAMMA)), DEFAULT_GAMMA, low=0.0)
    return torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)


def build_criterion(config: Optional[Union[str, Dict[str, Any]]] = None,
                    **kwargs: Any) -> nn.Module:
    """Softmax cross-entropy on raw logits (FR-07)."""
    cfg = load_config(config)
    label_smoothing = _to_float(
        kwargs.get("label_smoothing", _cfg_get(cfg, "training.label_smoothing", 0.0)),
        0.0, low=0.0, high=1.0,
    )
    try:
        return nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    except TypeError:  # pragma: no cover - older torch
        return nn.CrossEntropyLoss()


# --------------------------------------------------------------------------- #
# Training loop
# --------------------------------------------------------------------------- #
def train_model(config: Optional[Union[str, Dict[str, Any]]] = None,
                model: Optional[nn.Module] = None,
                model_name: Optional[str] = None,
                training_mode: Optional[str] = None,
                num_classes: Optional[int] = None,
                train_loader: Optional[DataLoader] = None,
                val_loader: Optional[DataLoader] = None,
                test_loader: Optional[DataLoader] = None,
                train_dataset: Optional[Dataset] = None,
                val_dataset: Optional[Dataset] = None,
                test_dataset: Optional[Dataset] = None,
                dataset: Optional[PlantVillageDataset] = None,
                epochs: Optional[int] = None,
                device: Optional[Union[str, torch.device]] = None,
                optimizer: Optional[torch.optim.Optimizer] = None,
                scheduler: Optional[Any] = None,
                criterion: Optional[nn.Module] = None,
                checkpoint_dir: Optional[Union[str, os.PathLike]] = None,
                save_best: Optional[bool] = None,
                save_final: Optional[bool] = None,
                log_interval: Optional[int] = None,
                seed: Optional[int] = None,
                **kwargs: Any) -> Dict[str, Any]:
    """Run the full training procedure (FR-06 / FR-07 / FR-08).

    Parameters
    ----------
    config:
        Configuration mapping or path to ``config/config.yaml``.
    model:
        Optional pre-built model; otherwise :func:`initialize_model` is used.
    model_name / training_mode / num_classes:
        Forwarded to :func:`initialize_model` (``training_mode`` is the ``scratch`` /
        ``transfer`` enum required by FR-06).
    train_loader / val_loader / test_loader:
        Optional data loaders; when absent they are built from ``PlantVillageDataset`` using the
        leaf-aware split strategy (a validation split only exists in ``engineering`` mode).
    epochs / device / optimizer / scheduler / criterion:
        Optional training hyper-parameters; defaults come from the project configuration.
    save_best / save_final:
        Control the FR-08 checkpointing of the best (by ``val_accuracy``) and final weights.

    Returns
    -------
    dict
        ``model``, ``history`` (per-epoch metrics), ``best_epoch``, ``best_val_accuracy``,
        ``best_checkpoint_path``, ``final_checkpoint_path``, ``optimizer``, ``scheduler``,
        ``test_metrics``.
    """
    cfg = load_config(config)

    # --- reproducibility --------------------------------------------------- #
    resolved_seed = _to_int(
        seed if seed is not None else _cfg_get(cfg, "split_strategy.seed", _cfg_get(cfg, "system.seed", DEFAULT_SEED)),
        DEFAULT_SEED, low=0,
    )
    deterministic = _to_bool(_cfg_get(cfg, "system.deterministic", True), True)
    set_seed(resolved_seed, deterministic)

    # --- names / hyper-parameters ----------------------------------------- #
    resolved_name = str(model_name or _cfg_get(cfg, "model.default", "googlenet")).strip().lower()
    if resolved_name not in _MODEL_NAMES:
        raise ValueError(f"Unknown model {resolved_name!r}; expected one of {_MODEL_NAMES}.")

    resolved_mode = str(training_mode or _cfg_get(cfg, "training_modes.active", "transfer")).strip().lower()
    if resolved_mode not in _TRAINING_MODES:
        raise ValueError(f"Unknown training_mode {resolved_mode!r}; expected one of {_TRAINING_MODES}.")

    reproduction_mode = str(
        kwargs.get("reproduction_mode") or _cfg_get(cfg, "reproduction_mode", "engineering")
    ).strip().lower()

    resolved_epochs = _to_int(epochs if epochs is not None else _cfg_get(cfg, "training.epochs", DEFAULT_EPOCHS), DEFAULT_EPOCHS, low=1)
    resolved_log_interval = _to_int(
        log_interval if log_interval is not None else _cfg_get(cfg, "training.log_interval", DEFAULT_LOG_INTERVAL),
        DEFAULT_LOG_INTERVAL,
    )
    grad_clip_raw = kwargs.get("grad_clip", _cfg_get(cfg, "training.gradient_clip_norm", None))
    grad_clip: Optional[float] = None
    if grad_clip_raw is not None:
        try:
            grad_clip = float(grad_clip_raw)
        except (TypeError, ValueError):
            grad_clip = None

    resolved_device = device if isinstance(device, torch.device) else resolve_device(device, cfg)

    # --- data ------------------------------------------------------------- #
    if train_loader is None and train_dataset is not None:
        dataset = train_dataset
    if any(loader is None for loader in (train_loader, val_loader, test_loader)):
        if train_loader is None or val_loader is None or test_loader is None:
            built_train, built_val, built_test, dataset = build_dataloaders(
                cfg,
                dataset=dataset,
                model_name=resolved_name,
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader,
                batch_size=kwargs.get("batch_size"),
                num_workers=kwargs.get("num_workers", _cfg_get(cfg, "training.num_workers", DEFAULT_NUM_WORKERS)),
                pin_memory=kwargs.get("pin_memory", _cfg_get(cfg, "training.pin_memory", True)),
                shuffle=kwargs.get("shuffle", _cfg_get(cfg, "training.shuffle_train", True)),
                drop_last=kwargs.get("drop_last", _cfg_get(cfg, "training.drop_last", False)),
                seed=resolved_seed,
                reproduction_mode=reproduction_mode,
            )
            if train_loader is None:
                train_loader = built_train
            if val_loader is None:
                val_loader = built_val
            if test_loader is None:
                test_loader = built_test

    if val_dataset is not None and val_loader is None:
        val_loader = _make_loader(
            val_dataset, _resolve_batch_size(cfg, resolved_name, kwargs.get("batch_size")),
            False,
            _to_int(kwargs.get("num_workers", _cfg_get(cfg, "training.num_workers", DEFAULT_NUM_WORKERS)), DEFAULT_NUM_WORKERS),
            _to_bool(kwargs.get("pin_memory", _cfg_get(cfg, "training.pin_memory", True)), True),
        )
    if test_dataset is not None and test_loader is None:
        test_loader = _make_loader(
            test_dataset, _resolve_batch_size(cfg, resolved_name, kwargs.get("batch_size")),
            False,
            _to_int(kwargs.get("num_workers", _cfg_get(cfg, "training.num_workers", DEFAULT_NUM_WORKERS)), DEFAULT_NUM_WORKERS),
            _to_bool(kwargs.get("pin_memory", _cfg_get(cfg, "training.pin_memory", True)), True),
        )

    if train_loader is None:
        raise ValueError("train_model requires a training DataLoader or a dataset to build one from.")

    # Strict reproduction mode never uses a validation split.
    if reproduction_mode == "strict":
        val_loader = None

    # --- model / optimizer / scheduler / criterion ------------------------ #
    if model is None:
        model = initialize_model(
            cfg,
            model_name=resolved_name,
            training_mode=resolved_mode,
            num_classes=num_classes,
            device=resolved_device,
        )
    else:
        model = model.to(resolved_device)

    if criterion is None:
        criterion = build_criterion(cfg, **kwargs)
    if optimizer is None:
        optimizer = build_optimizer(model, cfg, **kwargs)
    if scheduler is None:
        scheduler = build_scheduler(optimizer, cfg, **kwargs)

    mixed_precision = _to_bool(_cfg_get(cfg, "training.mixed_precision", False), False) or _to_bool(
        kwargs.get("mixed_precision", False), False
    )
    scaler = None
    if mixed_precision and resolved_device.type == "cuda":
        try:
            scaler = torch.cuda.amp.GradScaler()
        except Exception:  # pragma: no cover - defensive
            scaler = None

    # --- checkpointing flags (FR-08) -------------------------------------- #
    use_best = _to_bool(
        save_best if save_best is not None else _cfg_get(cfg, "checkpoint.save_best", True),
        True,
    ) and reproduction_mode != "strict"
    use_final = _to_bool(
        save_final if save_final is not None else _cfg_get(cfg, "checkpoint.save_final", True),
        True,
    )
    ckpt_dir = checkpoint_dir if checkpoint_dir is not None else _cfg_get(cfg, "checkpoint.dir", os.path.join(_PROJECT_ROOT, "checkpoints"))
    dataset_version = kwargs.get("dataset_version", _cfg_get(cfg, "data.default_version", "color"))
    train_test_split = kwargs.get(
        "train_test_split",
        _cfg_get(cfg, "split_strategy.default_train_ratio", 0.8),
    )

    LOGGER.info(
        "Starting training: model=%s mode=%s version=%s split=%s epochs=%d device=%s",
        resolved_name, resolved_mode, dataset_version, train_test_split, resolved_epochs, resolved_device,
    )

    history: List[Dict[str, Any]] = []
    best_val_accuracy = float("-inf")
    best_epoch = -1
    best_checkpoint_path: Optional[str] = None
    final_checkpoint_path: Optional[str] = None

    for epoch in range(1, resolved_epochs + 1):
        train_metrics = _run_one_epoch(
            model, train_loader, criterion, optimizer, resolved_device,
            log_interval=resolved_log_interval, scaler=scaler, grad_clip=grad_clip,
            epoch=epoch, tag="train",
        )

        val_metrics: Optional[Dict[str, Any]] = None
        if val_loader is not None:
            val_metrics = _run_one_epoch(
                model, val_loader, criterion, None, resolved_device,
                log_interval=resolved_log_interval, grad_clip=None,
                epoch=epoch, tag="val",
            )

        try:
            current_lr = optimizer.param_groups[0]["lr"]
        except Exception:  # pragma: no cover - defensive
            current_lr = None

        record: Dict[str, Any] = {
            "epoch": epoch,
            "train_loss": train_metrics.get("loss"),
            "train_accuracy": train_metrics.get("accuracy"),
            "val_loss": val_metrics.get("loss") if val_metrics else None,
            "val_accuracy": val_metrics.get("accuracy") if val_metrics else None,
            "lr": current_lr,
        }
        history.append(record)

        if scheduler is not None:
            try:
                scheduler.step()
            except Exception:  # pragma: no cover - defensive
                pass

        if use_best and val_metrics is not None:
            val_acc = float(val_metrics.get("accuracy", float("-inf")))
            if val_acc > best_val_accuracy:
                best_val_accuracy = val_acc
                best_epoch = epoch
                best_checkpoint_path = save_model(
                    model,
                    path=ckpt_dir,
                    config=cfg,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch=epoch,
                    metrics=record,
                    model_name=resolved_name,
                    training_mode=resolved_mode,
                    dataset_version=dataset_version,
                    train_test_split=train_test_split,
                    checkpoint_dir=ckpt_dir,
                    suffix="best",
                    extra={"reproduction_mode": reproduction_mode, "seed": resolved_seed},
                )

    # --- final checkpoint + test evaluation ------------------------------- #
    if use_final:
        final_checkpoint_path = save_model(
            model,
            path=ckpt_dir,
            config=cfg,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=resolved_epochs,
            metrics=history[-1] if history else None,
            model_name=resolved_name,
            training_mode=resolved_mode,
            dataset_version=dataset_version,
            train_test_split=train_test_split,
            checkpoint_dir=ckpt_dir,
            suffix="final",
            extra={"reproduction_mode": reproduction_mode, "seed": resolved_seed},
        )

    test_metrics: Optional[Dict[str, Any]] = None
    if test_loader is not None:
        test_metrics = _run_one_epoch(
            model, test_loader, criterion, None, resolved_device,
            log_interval=resolved_log_interval, grad_clip=None,
            epoch=resolved_epochs, tag="test",
        )

    result: Dict[str, Any] = {
        "model": model,
        "model_name": resolved_name,
        "training_mode": resolved_mode,
        "reproduction_mode": reproduction_mode,
        "dataset_version": dataset_version,
        "train_test_split": train_test_split,
        "epochs": resolved_epochs,
        "history": history,
        "best_epoch": best_epoch,
        "best_val_accuracy": None if best_val_accuracy == float("-inf") else best_val_accuracy,
        "best_checkpoint_path": best_checkpoint_path,
        "final_checkpoint_path": final_checkpoint_path,
        "test_metrics": test_metrics,
        "optimizer": optimizer,
        "scheduler": scheduler,
        "criterion": criterion,
        "device": resolved_device,
    }
    LOGGER.info(
        "Training finished: model=%s mode=%s best_epoch=%s best_val_acc=%s test=%s",
        resolved_name, resolved_mode, best_epoch, result["best_val_accuracy"],
        None if test_metrics is None else test_metrics.get("accuracy"),
    )
    return result


# --------------------------------------------------------------------------- #
# CLI entry point
# --------------------------------------------------------------------------- #
def main(argv: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    """Command-line entry point: build the model and run :func:`train_model`."""
    parser = argparse.ArgumentParser(description="Train a plant-disease classifier.")
    parser.add_argument("--config", default=None, help="Path to config/config.yaml")
    parser.add_argument("--model", default=None, choices=list(_MODEL_NAMES), help="Architecture")
    parser.add_argument("--training-mode", default=None, choices=list(_TRAINING_MODES), help="Training mode")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--checkpoint-dir", default=None)
    args = parser.parse_args(list(argv) if argv is not None else None)

    return train_model(
        config=args.config,
        model_name=args.model,
        training_mode=args.training_mode,
        epochs=args.epochs,
        batch_size=args.batch_size,
        device=args.device,
        seed=args.seed,
        checkpoint_dir=args.checkpoint_dir,
    )


if __name__ == "__main__":  # pragma: no cover - CLI
    main()