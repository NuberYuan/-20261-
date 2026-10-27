"""GoogLeNet model definition for image-based plant disease detection.

Paper
-----
"Using Deep Learning for Image-Based Plant Disease Detection"
(Mohanty, Hughes and Salathé, 2016).

This module defines :class:`Googlenet`, the GoogLeNet (Inception v1) architecture
used by the reproduction.  GoogLeNet is a 22-layer deep network that is much
wider than AlexNet yet has considerably fewer parameters (~5 million versus
~60 million).  Its core building block is the *Inception* module, of which 9 are
stacked: every module runs 1x1, 3x3 and 5x5 convolutions plus a 3x3 max-pooling
branch in parallel and concatenates their outputs along the channel dimension;
1x1 convolutions placed before the 3x3/5x5 branches (and after the pooling
branch) reduce the channel dimensionality.

Implementation policy
---------------------
* The class derives from ``torchvision.models.GoogLeNet`` so that the module
  tree (and therefore every ``state_dict`` key) is identical to the official
  implementation.  This guarantees that ImageNet-pretrained weights can be
  loaded 1:1 for the *transfer* training mode and that the classification heads
  ``fc``, ``aux1.fc2`` and ``aux2.fc2`` can be reset individually
  (``training_modes.transfer.reset_layers`` / FR-06).
* Only the structure is defined here (weights are randomly initialised).
  Weight initialisation (kaiming_normal for the *scratch* mode) as well as
  pretrained loading and classifier resetting for the *transfer* mode are
  performed centrally in ``training/train.py`` (``initialize_model``); no
  ``load_state_dict`` call happens in ``__init__``.
* All shape-related hyper-parameters (``num_classes``, dropout, ``aux_logits``)
  are read from ``config/config.yaml``; hard-coded numbers are used only as
  safe fallbacks derived from the paper (38 PlantVillage classes, dropout 0.5,
  3 input channels).

Config keys consumed (dotted-flat or nested access is supported)
---------------------------------------------------------------
* ``model.num_classes``
* ``model.in_channels``
* ``model.googlenet.aux_logits``
* ``model.googlenet.dropout_ratio``
* ``model.googlenet.init_weights``
"""

from __future__ import annotations

import inspect
import logging
import os
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn

try:
    import yaml
except Exception:
    yaml = None

try:
    from torchvision.models.googlenet import GoogLeNet as _TvGoogLeNet
except Exception:
    from torchvision.models import GoogLeNet as _TvGoogLeNet

__all__ = ["Googlenet", "load_config"]

LOGGER = logging.getLogger(__name__)

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DEFAULT_CONFIG_RELATIVE_PATH = os.path.join("config", "config.yaml")

# ---------------------------------------------------------------------------
# Fallback defaults (only used when config/config.yaml is unavailable or a
# shape parameter is null).  A torch shape argument must never receive None:
# these values are derived from the paper description.
# ---------------------------------------------------------------------------
_DEFAULT_NUM_CLASSES = 38      # 14 crop species x disease/healthy -> 38 PlantVillage classes
_DEFAULT_DROPOUT_RATIO = 0.5   # config['model.googlenet.dropout_ratio']
_DEFAULT_AUX_LOGITS = True     # aux heads must exist so that aux1.fc2 / aux2.fc2 can be reset
_DEFAULT_INIT_WEIGHTS = False  # config['model.googlenet.init_weights']
_DEFAULT_IN_CHANNELS = 3       # RGB input (config['data.channels'])


def _candidate_config_paths(config_path: Optional[str] = None) -> list:
    """Return the list of candidate config file locations, in priority order."""
    paths = []
    if isinstance(config_path, str) and config_path:
        paths.append(config_path)
    paths.append(os.path.join(_PROJECT_ROOT, _DEFAULT_CONFIG_RELATIVE_PATH))
    paths.append(_DEFAULT_CONFIG_RELATIVE_PATH)
    return paths


def load_config(config: Any = None) -> Dict[str, Any]:
    """Return the project configuration as a plain dictionary.

    Parameters
    ----------
    config:
        ``None`` -> load ``config/config.yaml`` from the project root;
        ``dict`` -> returned unchanged;
        ``str``  -> path of a YAML file to load.

    The YAML file is always opened with ``encoding='utf-8'``.  If nothing can be
    loaded an empty dictionary is returned and the built-in paper-derived
    defaults are used by the model.
    """
    if isinstance(config, dict):
        return config
    if yaml is None:
        LOGGER.warning(
            "PyYAML is not available; cannot read %s, falling back to built-in defaults.",
            _DEFAULT_CONFIG_RELATIVE_PATH,
        )
        return {}
    tried = []
    for path in _candidate_config_paths(config if isinstance(config, str) else None):
        tried.append(path)
        if isinstance(path, str) and os.path.isfile(path):
            with open(path, "r", encoding="utf-8") as handle:
                loaded = yaml.safe_load(handle)
            return loaded if isinstance(loaded, dict) else {}
    LOGGER.warning(
        "config/config.yaml was not found (tried: %s); falling back to built-in defaults.",
        tried,
    )
    return {}


def _cfg_get(cfg: Any, dotted_key: str, default: Any = None) -> Any:
    """Safely read ``dotted_key`` from ``cfg``.

    Supports both nested dictionaries (``{'model': {'num_classes': 38}}``) and
    flat dictionaries with dotted keys (``{'model.num_classes': 38}``).  A
    missing key or an explicit ``None`` value returns ``default`` so that no
    ``None`` can leak into a torch shape argument.
    """
    if not isinstance(cfg, dict) or not dotted_key:
        return default
    if dotted_key in cfg:
        value = cfg[dotted_key]
        return default if value is None else value
    node = cfg
    for part in dotted_key.split("."):
        if isinstance(node, dict) and part in node:
            node = node[part]
        else:
            return default
    return default if node is None else node


class Googlenet(_TvGoogLeNet):
    """GoogLeNet (Inception v1) adapted to PlantVillage plant disease classes.

    The module hierarchy is exactly ``torchvision.models.GoogLeNet``'s one:
    stem convolutions, 9 Inception modules (``inception3a`` .. ``inception5b``),
    two auxiliary classifiers (``aux1``, ``aux2``) used only while training, an
    average pooling layer, a dropout layer and the main classifier ``fc``.

    Parameters
    ----------
    config:
        ``None``, a configuration ``dict`` or the path to a YAML file.
        Shape parameters are read from ``model.*`` keys.
    num_classes:
        Overrides ``model.num_classes``.  Defaults to the config value, then to
        38 (PlantVillage classes, from the paper).
    aux_logits:
        Overrides ``model.googlenet.aux_logits``.  Keeps the auxiliary heads so
        that their ``fc2`` layers can be reset in transfer mode.
    dropout_ratio:
        Overrides ``model.googlenet.dropout_ratio`` (main head dropout, and the
        auxiliary heads' dropout where the installed torchvision exposes it).
    init_weights:
        Overrides ``model.googlenet.init_weights``.  Kept ``False`` because
        weight initialisation is centralised in ``training/train.py``.

    Notes
    -----
    The architecture is shared by the *scratch* and *transfer* training modes;
    the two modes differ only in how the weights are initialised, which is done
    outside of this class (FR-06).
    """

    def __init__(
        self,
        config: Any = None,
        num_classes: Optional[int] = None,
        aux_logits: Optional[bool] = None,
        dropout_ratio: Optional[float] = None,
        init_weights: Optional[bool] = None,
    ) -> None:
        cfg = load_config(config)

        # ---- num_classes: config -> explicit argument -> paper default -------
        resolved_num_classes = num_classes if num_classes is not None else _cfg_get(
            cfg, "model.num_classes", _DEFAULT_NUM_CLASSES
        )
        try:
            resolved_num_classes = int(resolved_num_classes)
        except (TypeError, ValueError):
            resolved_num_classes = _DEFAULT_NUM_CLASSES
        if resolved_num_classes <= 0:
            resolved_num_classes = _DEFAULT_NUM_CLASSES

        # ---- aux_logits ----------------------------------------------------
        resolved_aux_logits = aux_logits if aux_logits is not None else _cfg_get(
            cfg, "model.googlenet.aux_logits", _cfg_get(cfg, "model.aux_logits", _DEFAULT_AUX_LOGITS)
        )
        resolved_aux_logits = bool(resolved_aux_logits)

        # ---- dropout -------------------------------------------------------
        resolved_dropout = dropout_ratio if dropout_ratio is not None else _cfg_get(
            cfg,
            "model.googlenet.dropout_ratio",
            _cfg_get(cfg, "model.dropout_ratio", _DEFAULT_DROPOUT_RATIO),
        )
        try:
            resolved_dropout = float(resolved_dropout)
        except (TypeError, ValueError):
            resolved_dropout = float(_DEFAULT_DROPOUT_RATIO)
        if resolved_dropout < 0.0 or resolved_dropout >= 1.0:
            LOGGER.warning(
                "dropout_ratio=%s out of [0, 1) range; falling back to %s.",
                resolved_dropout,
                _DEFAULT_DROPOUT_RATIO,
            )
            resolved_dropout = float(_DEFAULT_DROPOUT_RATIO)

        # ---- init_weights (structure only; no pretrained loading here) ------
        resolved_init_weights = init_weights if init_weights is not None else _cfg_get(
            cfg, "model.googlenet.init_weights", _DEFAULT_INIT_WEIGHTS
        )
        resolved_init_weights = bool(resolved_init_weights)

        # ---- input channels: torchvision GoogLeNet is a 3-channel RGB net ---
        resolved_in_channels = _cfg_get(cfg, "model.in_channels", _cfg_get(cfg, "data.channels", _DEFAULT_IN_CHANNELS))
        try:
            resolved_in_channels = int(resolved_in_channels)
        except (TypeError, ValueError):
            resolved_in_channels = int(_DEFAULT_IN_CHANNELS)
        if resolved_in_channels != _DEFAULT_IN_CHANNELS:
            raise ValueError(
                "Googlenet expects {}-channel RGB input (config['model.in_channels'] = {}).".format(
                    _DEFAULT_IN_CHANNELS, resolved_in_channels
                )
            )

        # ---- build the torchvision backbone --------------------------------
        kwargs = {
            "num_classes": resolved_num_classes,
            "aux_logits": resolved_aux_logits,
            "transform_input": False,
            "init_weights": resolved_init_weights,
        }
        try:
            sig_params = inspect.signature(_TvGoogLeNet.__init__).parameters
        except (TypeError, ValueError):
            sig_params = {}

        if "dropout" in sig_params:
            kwargs["dropout"] = resolved_dropout
        if "dropout_aux" in sig_params:
            kwargs["dropout_aux"] = resolved_dropout

        if sig_params:
            supported_kwargs = {k: v for k, v in kwargs.items() if k in sig_params}
            super().__init__(**supported_kwargs)
        else:
            try:
                super().__init__(**kwargs)
            except TypeError:
                try:
                    super().__init__(
                        num_classes=resolved_num_classes,
                        aux_logits=resolved_aux_logits,
                        transform_input=False,
                    )
                except TypeError:
                    super().__init__(num_classes=resolved_num_classes)

        # Old torchvision releases hard-code the dropout probability (0.2) and do
        # not expose it as a constructor argument; patch the existing Dropout
        # module so that config['model.googlenet.dropout_ratio'] is honoured.
        # Dropout has no parameters, hence the state_dict keys are unchanged.
        if sig_params and "dropout" not in sig_params:
            if isinstance(getattr(self, "dropout", None), nn.Dropout):
                self.dropout.p = resolved_dropout

        # ---- expose resolved hyper-parameters ------------------------------
        self.config = cfg
        self.num_classes = resolved_num_classes
        self.aux_logits_enabled = resolved_aux_logits
        self.dropout_ratio = resolved_dropout
        self.in_channels = resolved_in_channels

    def forward(
        self,
        x: torch.Tensor,
        return_aux: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]]:
        """Forward pass of GoogLeNet.

        Parameters
        ----------
        x:
            Mini-batch of preprocessed images of shape ``(N, 3, 256, 256)``
            (``data.image_size``), already resized and normalised by the data
            pipeline.  The internal average pooling is adaptive, so other
            spatial sizes are accepted as well.
        return_aux:
            If ``True`` and the auxiliary heads produced outputs (they are
            active while ``self.training`` is ``True`` and ``self.aux_logits``
            is ``True``), a tuple ``(logits, aux1, aux2)`` is returned so that
            the training loop can add the weighted auxiliary losses.  ``aux1``
            stems from ``inception4d`` and ``aux2`` from ``inception5b``.

        Returns
        -------
        torch.Tensor
            Main classifier logits of shape ``(N, num_classes)`` (default).
        tuple
            ``(logits, aux1, aux2)`` when ``return_aux=True`` and auxiliary
            outputs are available.
        """
        # super().forward handles transform_input and returns either the main
        # logits or GoogLeNetOutputs(logits, aux1, aux2) during training.
        output = super().forward(x)

        if isinstance(output, tuple):
            logits = output[0]
            aux1 = output[1] if len(output) > 1 else None
            aux2 = output[2] if len(output) > 2 else None
        else:
            logits = output
            aux1 = None
            aux2 = None

        if return_aux and (aux1 is not None or aux2 is not None):
            return (logits, aux1, aux2)
        return logits