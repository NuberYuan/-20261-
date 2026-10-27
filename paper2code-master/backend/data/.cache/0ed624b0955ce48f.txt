"""models/alexnet.py

AlexNet model definition for the reproduction of
"Using Deep Learning for Image-Based Plant Disease Detection".

Scope of this file (Phase 3, FR-04):
    * define the ``Alexnet`` class (architecture only, random initialisation);
    * read all shape-affecting hyper-parameters (num_classes / dropout_ratio /
      in_channels) from ``config/config.yaml`` with safe fallbacks;
    * NOT handle weight initialisation / pretrained loading / training_mode
      branching -- that is done centrally in ``training/train.py``
      (``initialize_model``), so that scratch and transfer modes share exactly
      the same architecture (FR-06).

Architecture (matches ``torchvision.models.AlexNet`` exactly so that ImageNet
pre-trained weights can be loaded 1:1):
    features  : Conv(11x11, s4, p2) -> ReLU -> MaxPool(3, s2)
                -> Conv(5x5, p2)    -> ReLU -> MaxPool(3, s2)
                -> Conv(3x3, p1) -> ReLU
                -> Conv(3x3, p1) -> ReLU
                -> Conv(3x3, p1) -> ReLU -> MaxPool(3, s2)
    avgpool   : AdaptiveAvgPool2d((6, 6))
    classifier: Dropout(0.5) -> Linear(9216, 4096) -> ReLU
                -> Dropout(0.5) -> Linear(4096, 4096) -> ReLU
                -> Linear(4096, num_classes)

The paper mentions a "normalisation" layer after conv1/conv2.  The torchvision
reference implementation used here does not contain such a layer, and adding one
(e.g. LocalResponseNorm) would break the 1:1 loading of ImageNet pre-trained
weights; therefore it is intentionally NOT added (see global constraint on not
adding layers absent from torchvision).  The final softmax is intentionally not
part of the model: the loss is ``torch.nn.CrossEntropyLoss`` applied to the raw
logits returned by ``forward``.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional, Union

import torch
import torch.nn as nn
import yaml

try:  # torchvision >= 0.13 exposes AlexNet in torchvision.models
    from torchvision.models import AlexNet as _TorchvisionAlexNet
except ImportError:  # pragma: no cover - extremely old torchvision fallback
    from torchvision.models.alexnet import AlexNet as _TorchvisionAlexNet


# --------------------------------------------------------------------------- #
# Fallback constants
# --------------------------------------------------------------------------- #
# num_classes fallback: paper / PlantVillage defines 38 classes
# (14 crop species x disease or healthy status); see config['data.num_classes'].
DEFAULT_NUM_CLASSES = 38
# dropout fallback: config['training.dropout_ratio'] = config['model.dropout_ratio'] = 0.5
DEFAULT_DROPOUT_RATIO = 0.5
# channel fallback: config['model.in_channels'] = config['data.channels'] = 3 (RGB)
DEFAULT_IN_CHANNELS = 3
# torchvision AlexNet fixed intermediate width (used only for sanity checks)
_ALEXNET_FC_IN_FEATURES = 256 * 6 * 6

_DEFAULT_CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "config",
    "config.yaml",
)


# --------------------------------------------------------------------------- #
# Config helpers
# --------------------------------------------------------------------------- #
def _read_yaml(path: str) -> Dict[str, Any]:
    """Read a YAML file with explicit UTF-8 encoding; return {} on any problem."""
    if not os.path.isfile(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
    except (OSError, yaml.YAMLError):
        return {}
    return data if isinstance(data, dict) else {}


def load_config(config: Optional[Union[str, Dict[str, Any]]] = None) -> Dict[str, Any]:
    """Return a configuration dictionary.

    Parameters
    ----------
    config:
        ``None`` -> load ``config/config.yaml`` next to the project root;
        ``str``  -> path to a YAML file;
        ``dict`` -> used as-is (already loaded configuration).

    An empty dict is returned when nothing can be resolved; every consumer of
    the configuration must therefore provide a sane default value.
    """
    if config is None:
        return _read_yaml(_DEFAULT_CONFIG_PATH)
    if isinstance(config, dict):
        return config
    if isinstance(config, str):
        return _read_yaml(config)
    raise TypeError("config must be None, a dict, or a path to a YAML file")


def _cfg_get(
    config: Dict[str, Any],
    *paths: str,
    default: Any = None,
) -> Any:
    """Resolve the first existing key among ``paths``.

    Tries nested lookup (``model`` -> ``num_classes``) first, then falls back to
    a flat dotted key (``'model.num_classes'``).  Never raises on missing keys.
    """
    for path in paths:
        if not path:
            continue
        node: Any = config
        found = True
        for part in path.split("."):
            if isinstance(node, dict) and part in node:
                node = node[part]
            else:
                found = False
                break
        if found:
            return node
        if isinstance(config, dict) and path in config:
            return config[path]
    return default


def _coerce_int(value: Any, default: int, name: str) -> int:
    """Coerce ``value`` to a positive int, falling back to ``default``.

    Guarantees that a ``None`` / invalid value coming from the configuration
    never reaches ``nn.Linear`` / ``nn.Conv2d`` shape arguments.
    """
    if value is None:
        result = int(default)
    elif isinstance(value, bool):
        raise ValueError(f"{name} must be an int, got bool")
    elif isinstance(value, (int,)):
        result = int(value)
    elif isinstance(value, float) and float(value).is_integer():
        result = int(value)
    elif isinstance(value, str):
        try:
            result = int(float(value.strip()))
        except ValueError:
            result = int(default)
    else:
        result = int(default)
    if result <= 0:
        raise ValueError(f"{name} must be a positive integer, got {result}")
    return result


def _coerce_float(value: Any, default: float, name: str, low: float = 0.0, high: float = 1.0) -> float:
    """Coerce ``value`` to a float inside ``[low, high)``, falling back to ``default``."""
    if value is None:
        result = float(default)
    elif isinstance(value, bool):
        raise ValueError(f"{name} must be a float, got bool")
    elif isinstance(value, (int, float)):
        result = float(value)
    elif isinstance(value, str):
        try:
            result = float(value.strip())
        except ValueError:
            result = float(default)
    else:
        result = float(default)
    if not (low <= result < high):
        raise ValueError(f"{name} must lie in [{low}, {high}), got {result}")
    return result


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
class Alexnet(_TorchvisionAlexNet):
    """AlexNet adapted for plant-disease classification.

    The class only defines the network structure (randomly initialised).  All
    weight initialisation, ImageNet pre-trained loading and ``training_mode``
    (``scratch`` / ``transfer``) handling is performed outside of this class,
    in ``training/train.py`` (``initialize_model``), keeping a single shared
    architecture for both modes (FR-06).

    Parameters
    ----------
    config:
        ``None`` -> ``config/config.yaml``; or an already-loaded dict; or a
        path to a YAML file.  All shape parameters are read from it.
    num_classes:
        Overrides ``config['model.num_classes']``.  Fallback: 38
        (PlantVillage: 14 crop species x 26 diseases or healthy status).
    dropout_ratio:
        Overrides ``config['model.alexnet.dropout_ratio']``
        (then ``config['model.dropout_ratio']``).  Fallback: 0.5.
    in_channels:
        Overrides ``config['model.in_channels']`` (then ``config['data.channels']``).
        Fallback: 3.  Only used to build the first convolution; when equal to 3
        the layer is byte-identical to the torchvision reference so ImageNet
        weights remain loadable 1:1.
    """

    def __init__(
        self,
        config: Optional[Union[str, Dict[str, Any]]] = None,
        num_classes: Optional[int] = None,
        dropout_ratio: Optional[float] = None,
        in_channels: Optional[int] = None,
    ) -> None:
        cfg = load_config(config)

        # ---- shape-affecting hyper-parameters (never None at the torch side) ----
        resolved_num_classes = _cfg_get(
            cfg,
            "model.num_classes",
            "data.num_classes",
            default=None,
        )
        self.num_classes = _coerce_int(
            num_classes if num_classes is not None else resolved_num_classes,
            DEFAULT_NUM_CLASSES,
            "num_classes",
        )

        resolved_dropout = _cfg_get(
            cfg,
            "model.alexnet.dropout_ratio",
            "model.dropout_ratio",
            "training.dropout_ratio",
            default=None,
        )
        self.dropout_ratio = _coerce_float(
            dropout_ratio if dropout_ratio is not None else resolved_dropout,
            DEFAULT_DROPOUT_RATIO,
            "dropout_ratio",
            low=0.0,
            high=1.0,
        )

        resolved_channels = _cfg_get(
            cfg,
            "model.in_channels",
            "data.channels",
            default=None,
        )
        self.in_channels = _coerce_int(
            in_channels if in_channels is not None else resolved_channels,
            DEFAULT_IN_CHANNELS,
            "in_channels",
        )

        # ---- architecture (identical to torchvision.models.AlexNet) ----
        super().__init__(num_classes=self.num_classes, dropout=self.dropout_ratio)

        # Keep the exact torchvision convolution hyper-parameters, only the
        # input channel count may differ (e.g. grayscale runs: 1 channel).
        if self.in_channels != 3:
            self.features[0] = nn.Conv2d(
                in_channels=self.in_channels,
                out_channels=64,
                kernel_size=11,
                stride=4,
                padding=2,
            )

        # Bookkeeping for external code (train.py / evaluation).
        self.input_size = tuple(
            _cfg_get(cfg, "model.input_size", "data.image_size_hw", default=[256, 256])
            or [256, 256]
        )
        self.fc_in_features = _ALEXNET_FC_IN_FEATURES

    # ------------------------------------------------------------------ #
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x:
            Batch of preprocessed images, shape ``[N, C, H, W]`` with
            ``C == self.in_channels`` (3 for the RGB "color" dataset version).

        Returns
        -------
        torch.Tensor
            Raw logits of shape ``[N, num_classes]``.  No softmax is applied
            here: the training loss is ``torch.nn.CrossEntropyLoss`` on logits
            (config['training.loss_function'] = 'cross_entropy').
        """
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    # ------------------------------------------------------------------ #
    def extra_repr(self) -> str:
        return (
            f"num_classes={self.num_classes}, "
            f"dropout_ratio={self.dropout_ratio}, "
            f"in_channels={self.in_channels}"
        )


__all__ = ["Alexnet", "load_config"]