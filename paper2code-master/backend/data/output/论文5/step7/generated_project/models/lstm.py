# -*- coding: utf-8 -*-
"""
models/lstm.py

LSTM encoder-decoder module ("locality enhancement" layer) used as the
recurrent baseline for the paper
"Temporal Fusion Transformers for Interpretable Multihorizon Time Series
Forecasting".

Architecture (as described in the paper):
  * Static covariates s_i in R^{m_s} are mapped by static covariate encoders
    into two context vectors c_c (cell state) and c_h (hidden state) that
    initialise the first LSTM layer of the encoder.
  * Past observed inputs xi~(t-k:t) are consumed by the LSTM encoder.
  * Known future inputs xi~(t+1:t+tau_max) are consumed by the LSTM decoder,
    which is initialised with the final encoder state.
  * A gated skip connection is applied over this LSTM layer.
  * Quantile forecasts y_hat(q, t, tau) for tau in {1, ..., tau_max} are
    produced by a linear transformation of the decoder output.

The module is randomly initialised (no pretraining / no transfer branch);
weight initialisation is handled by training/train.py.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import yaml

__all__ = ["Lstm"]


# --------------------------------------------------------------------------- #
#  config/config.yaml helpers
# --------------------------------------------------------------------------- #
def _candidate_config_paths(explicit: Optional[str] = None) -> List[str]:
    """Return candidate filesystem paths for ``config/config.yaml``."""
    paths: List[str] = []
    if explicit is not None:
        paths.append(str(explicit))
    here = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(here)
    paths.append(os.path.join(project_root, "config", "config.yaml"))
    paths.append(os.path.join(os.getcwd(), "config", "config.yaml"))
    paths.append(os.path.join(here, "config", "config.yaml"))
    paths.append("config/config.yaml")
    return paths


def load_config(config: Optional[Union[Dict[str, Any], str]] = None) -> Dict[str, Any]:
    """Load ``config/config.yaml`` with ``yaml.safe_load`` (encoding='utf-8')."""
    if isinstance(config, dict):
        return config
    if isinstance(config, (str, os.PathLike)):
        with open(str(config), "r", encoding="utf-8") as handle:
            return yaml.safe_load(handle) or {}
    for path in _candidate_config_paths():
        if path and os.path.isfile(path):
            with open(path, "r", encoding="utf-8") as handle:
                return yaml.safe_load(handle) or {}
    return {}


def _cfg_get(config: Dict[str, Any], key: str, default: Any = None) -> Any:
    """Safe lookup of ``config['a.b.c']``; ``None`` values fall back to ``default``."""
    if not isinstance(config, dict):
        return default
    if key in config:
        value = config[key]
        return default if value is None else value
    node: Any = config
    for part in key.split("."):
        if isinstance(node, dict) and part in node:
            node = node[part]
        else:
            return default
    return default if node is None else node


def _cfg_list(config: Dict[str, Any], key: str) -> List[Any]:
    value = _cfg_get(config, key, None)
    if isinstance(value, (list, tuple)):
        return list(value)
    return []


def _to_int(value: Any, default: int, minimum: int = 1) -> int:
    try:
        out = int(value)
    except (TypeError, ValueError):
        return max(minimum, int(default))
    return max(minimum, out)


def _to_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


# --------------------------------------------------------------------------- #
#  tensor shape helpers
# --------------------------------------------------------------------------- #
def _fit_features(x: torch.Tensor, feature_dim: int) -> torch.Tensor:
    """Pad / truncate the last dimension to ``feature_dim`` (no shape crash)."""
    current = int(x.size(-1))
    if current == feature_dim:
        return x
    if current > feature_dim:
        return x[..., :feature_dim]
    pad = x.new_zeros(x.shape[:-1] + (feature_dim - current,))
    return torch.cat([x, pad], dim=-1)


def _fit_time(x: torch.Tensor, seq_len: int) -> torch.Tensor:
    """Pad / truncate dimension 1 to ``seq_len``."""
    current = int(x.size(1))
    if current == seq_len:
        return x
    if current > seq_len:
        return x[:, :seq_len, :]
    pad = x.new_zeros(x.size(0), seq_len - current, x.size(2))
    return torch.cat([x, pad], dim=1)


def _to_3d(x: torch.Tensor) -> torch.Tensor:
    """Normalise a tensor to shape (batch, time, features)."""
    if x.dim() == 1:
        return x.unsqueeze(0).unsqueeze(-1)
    if x.dim() == 2:
        return x.unsqueeze(-1)
    if x.dim() == 3:
        return x
    # (batch, time, features, ...) -> flatten trailing dims
    return x.reshape(x.size(0), x.size(1), -1)


def _to_2d(x: torch.Tensor) -> torch.Tensor:
    """Normalise a static covariate tensor to shape (batch, features)."""
    if x.dim() == 1:
        return x.unsqueeze(0)
    if x.dim() == 2:
        return x
    if x.dim() == 3:
        return x[:, -1, :]
    return x.reshape(x.size(0), -1)


# --------------------------------------------------------------------------- #
#  static covariate encoder -> (c_c, c_h) context vectors
# --------------------------------------------------------------------------- #
class _StaticContextEncoder(nn.Module):
    """Static covariate encoder producing cell-state / hidden-state contexts."""

    def __init__(self, static_dim: int, hidden_size: int, num_layers: int) -> None:
        super().__init__()
        width = int(hidden_size) * int(num_layers)
        self.c_proj = nn.Linear(static_dim, width)
        self.c_gate = nn.Linear(static_dim, width)
        self.h_proj = nn.Linear(static_dim, width)
        self.h_gate = nn.Linear(static_dim, width)
        self.width = width

    def forward(self, static: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        c_c = torch.tanh(self.c_proj(static)) * torch.sigmoid(self.c_gate(static))
        c_h = torch.tanh(self.h_proj(static)) * torch.sigmoid(self.h_gate(static))
        return c_c, c_h


# --------------------------------------------------------------------------- #
#  main model
# --------------------------------------------------------------------------- #
class Lstm(nn.Module):
    """LSTM encoder-decoder with gated skip connection and quantile head.

    Parameters
    ----------
    config : dict | str | None
        Loaded configuration mapping or path to ``config/config.yaml``.
        When ``None`` the file is searched on disk; missing / null values fall
        back to documented defaults.
    """

    def __init__(
        self,
        config: Optional[Union[Dict[str, Any], str]] = None,
        input_window: Optional[int] = None,
        forecast_horizon: Optional[int] = None,
        past_input_dim: Optional[int] = None,
        future_input_dim: Optional[int] = None,
        static_input_dim: Optional[int] = None,
        hidden_size: Optional[int] = None,
        num_layers: Optional[int] = None,
        dropout_ratio: Optional[float] = None,
        quantiles: Optional[Sequence[float]] = None,
        target_dim: Optional[int] = None,
    ) -> None:
        super().__init__()

        cfg = load_config(config)
        self.config = cfg

        # ---------------- target dimensionality ---------------- #
        # config['data.target_dim'] == 1 (scalar targets y_{i,t}).
        self.target_dim = _to_int(
            target_dim if target_dim is not None else _cfg_get(cfg, "data.target_dim", 1),
            default=1,
            minimum=1,
        )

        # ---------------- windows ---------------- #
        # config['model.input_window'] / config['data.input_window'] are null;
        # fallback k = 168 (Electricity / Traffic lookback window in the paper).
        default_window = _to_int(
            _cfg_get(cfg, "model.input_window", _cfg_get(cfg, "data.input_window", 168)),
            default=168,
            minimum=1,
        )
        self.input_window = _to_int(
            input_window if input_window is not None else default_window,
            default=168,
            minimum=1,
        )
        # config['model.forecast_horizon'] / config['data.forecast_horizon'] are
        # null; fallback tau_max = 24 (Electricity / Traffic horizon in the paper).
        default_horizon = _to_int(
            _cfg_get(
                cfg,
                "model.forecast_horizon",
                _cfg_get(cfg, "data.forecast_horizon", 24),
            ),
            default=24,
            minimum=1,
        )
        self.forecast_horizon = _to_int(
            forecast_horizon if forecast_horizon is not None else default_horizon,
            default=24,
            minimum=1,
        )

        # ---------------- feature dimensionalities ---------------- #
        # Past window xi~(t-k:t) = observed inputs z_{i,t} + scalar target y_{i,t}.
        observed = _cfg_list(cfg, "data.observed_inputs")
        default_past_dim = max(1, len(observed) + self.target_dim)
        self.past_input_dim = _to_int(
            past_input_dim if past_input_dim is not None else default_past_dim,
            default=default_past_dim,
            minimum=1,
        )
        # Decoder input xi~(t+1:t+tau_max) = known inputs x_{i,t} + lagged target.
        known = _cfg_list(cfg, "data.known_inputs")
        default_future_dim = max(1, len(known) + self.target_dim)
        self.future_input_dim = _to_int(
            future_input_dim if future_input_dim is not None else default_future_dim,
            default=default_future_dim,
            minimum=1,
        )
        statics = _cfg_list(cfg, "data.static_inputs")
        default_static_dim = max(1, len(statics))
        self.static_input_dim = _to_int(
            static_input_dim if static_input_dim is not None else default_static_dim,
            default=default_static_dim,
            minimum=1,
        )

        # ---------------- recurrent width ---------------- #
        # config['model.hidden_size'] is null (paper leaves it to the dataset) ->
        # fallback 32, the LSTM width used in the paper's recurrent baselines.
        cfg_hidden = _cfg_get(cfg, "model.hidden_size", None)
        self.hidden_size = _to_int(
            hidden_size if hidden_size is not None else cfg_hidden,
            default=32,
            minimum=1,
        )
        # config['model.num_lstm_layers'] is null -> fallback 1
        # (the locality-enhancement block of the paper is a single LSTM layer).
        cfg_layers = _cfg_get(cfg, "model.num_lstm_layers", None)
        self.num_layers = _to_int(
            num_layers if num_layers is not None else cfg_layers,
            default=1,
            minimum=1,
        )

        # ---------------- dropout ---------------- #
        # config['model.dropout_ratio'] == 0.1
        cfg_dropout = _cfg_get(cfg, "model.dropout_ratio", _cfg_get(cfg, "training.dropout_ratio", 0.1))
        self.dropout_ratio = _to_float(
            dropout_ratio if dropout_ratio is not None else cfg_dropout,
            default=0.1,
        )
        self.dropout_ratio = min(max(self.dropout_ratio, 0.0), 1.0)

        # ---------------- quantile outputs ---------------- #
        # config['model.quantiles'] == [0.1, 0.5, 0.9]; num_classes is NOT_FOUND
        # for this regression task and is therefore never passed to nn.Linear.
        if quantiles is None:
            cfg_quantiles = _cfg_list(cfg, "model.quantiles")
            quantile_list = [float(q) for q in cfg_quantiles] if cfg_quantiles else [0.1, 0.5, 0.9]
        else:
            quantile_list = [float(q) for q in quantiles]
        self.quantiles: List[float] = quantile_list
        self.num_quantiles = max(1, len(quantile_list))

        # ---------------- layers ---------------- #
        self.static_encoder = _StaticContextEncoder(
            static_dim=self.static_input_dim,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
        )

        rnn_dropout = self.dropout_ratio if self.num_layers > 1 else 0.0
        self.encoder_lstm = nn.LSTM(
            input_size=self.past_input_dim,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=rnn_dropout,
        )
        self.decoder_lstm = nn.LSTM(
            input_size=self.future_input_dim,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=rnn_dropout,
        )

        # Gated skip connection over the LSTM layer (paper Fig. 3).
        self.skip_projection = nn.Linear(self.future_input_dim, self.hidden_size)
        self.skip_gate = nn.Linear(self.hidden_size, self.hidden_size)
        self.skip_dropout = nn.Dropout(self.dropout_ratio)
        self.layer_norm = nn.LayerNorm(self.hidden_size)

        # Quantile head: linear transformation of the decoder output.
        self.quantile_head = nn.Linear(self.hidden_size, self.num_quantiles)

    # ------------------------------------------------------------------ #
    #  input handling
    # ------------------------------------------------------------------ #
    def _resolve_inputs(
        self,
        static_inputs: Any,
        past_inputs: Any,
        future_inputs: Any,
        targets: Any,
        kwargs: Dict[str, Any],
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Unpack dict / tuple style batches and keyword aliases."""
        if isinstance(static_inputs, dict):
            batch = static_inputs
            static_inputs = batch.get(
                "static_inputs", batch.get("static", batch.get("s", batch.get("static_covariates", None)))
            )
            past_inputs = batch.get(
                "past_inputs", batch.get("past", batch.get("z", batch.get("observed_inputs", past_inputs)))
            )
            future_inputs = batch.get(
                "future_inputs", batch.get("future", batch.get("x", batch.get("known_inputs", future_inputs)))
            )
            targets = batch.get("targets", batch.get("target", batch.get("y", targets)))
        elif isinstance(static_inputs, (tuple, list)) and past_inputs is None:
            bundle = list(static_inputs)
            if len(bundle) >= 1:
                static_inputs = bundle[0]
            if len(bundle) >= 2:
                past_inputs = bundle[1]
            if len(bundle) >= 3:
                future_inputs = bundle[2]
            if len(bundle) >= 4:
                targets = bundle[3]

        if kwargs:
            if static_inputs is None:
                static_inputs = kwargs.get("static_inputs", kwargs.get("static", kwargs.get("s", None)))
            if past_inputs is None:
                past_inputs = kwargs.get("past_inputs", kwargs.get("past", kwargs.get("z", None)))
            if future_inputs is None:
                future_inputs = kwargs.get("future_inputs", kwargs.get("future", kwargs.get("x", None)))
            if targets is None:
                targets = kwargs.get("targets", kwargs.get("target", kwargs.get("y", None)))

        return static_inputs, past_inputs, future_inputs, targets

    @staticmethod
    def _infer_batch_size(*tensors: Optional[torch.Tensor]) -> int:
        for tensor in tensors:
            if tensor is not None and torch.is_tensor(tensor) and tensor.dim() >= 2:
                return int(tensor.size(0))
        return 1

    # ------------------------------------------------------------------ #
    #  forward
    # ------------------------------------------------------------------ #
    def forward(
        self,
        static_inputs: Optional[torch.Tensor] = None,
        past_inputs: Optional[torch.Tensor] = None,
        future_inputs: Optional[torch.Tensor] = None,
        targets: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        static_inputs : (batch, static_input_dim) static covariates s_i.
        past_inputs : (batch, k, past_input_dim) past observed inputs xi~(t-k:t).
        future_inputs : (batch, tau_max, future_input_dim) known future inputs.
        targets : optional (batch, tau_max) ground-truth used for teacher
            forcing of the lagged decoder target channel.

        Returns
        -------
        torch.Tensor
            Quantile forecasts of shape (batch, tau_max, num_quantiles).
        """
        static_inputs, past_inputs, future_inputs, targets = self._resolve_inputs(
            static_inputs, past_inputs, future_inputs, targets, kwargs
        )

        batch_size = self._infer_batch_size(past_inputs, future_inputs, targets, static_inputs)

        reference_device = None
        for tensor in (past_inputs, future_inputs, static_inputs, targets):
            if tensor is not None and torch.is_tensor(tensor):
                reference_device = tensor.device
                break
        if reference_device is None:
            reference_device = torch.device("cpu")

        # ---------------- static covariates ---------------- #
        if static_inputs is None or not torch.is_tensor(static_inputs):
            static = torch.zeros(batch_size, self.static_input_dim, device=reference_device)
        else:
            static = _to_2d(static_inputs)
            static = _fit_features(static, self.static_input_dim)

        # ---------------- past window ---------------- #
        if past_inputs is None or not torch.is_tensor(past_inputs):
            past = torch.zeros(
                batch_size, self.input_window, self.past_input_dim, device=reference_device
            )
        else:
            past = _to_3d(past_inputs)
            past = _fit_features(past, self.past_input_dim)

        # ---------------- decoder horizon ---------------- #
        if future_inputs is not None and torch.is_tensor(future_inputs):
            horizon_len = int(_to_3d(future_inputs).size(1))
        elif targets is not None and torch.is_tensor(targets):
            horizon_len = int(_to_3d(targets).size(1))
        else:
            horizon_len = self.forecast_horizon
        horizon_len = max(1, horizon_len)

        if future_inputs is not None and torch.is_tensor(future_inputs):
            future = _to_3d(future_inputs)
            future = _fit_features(future, self.future_input_dim)
            future = _fit_time(future, horizon_len)
        else:
            future = torch.zeros(
                batch_size, horizon_len, self.future_input_dim, device=reference_device
            )
            if targets is not None and torch.is_tensor(targets):
                # Teacher forcing: the decoder receives the lagged target.
                t3 = _to_3d(targets)
                t3 = _fit_time(t3, horizon_len)
                width = min(self.target_dim, int(t3.size(-1)), int(past.size(-1)), self.future_input_dim)
                if width > 0:
                    lag = torch.cat(
                        [past[:, -1:, past.size(-1) - width :], t3[:, :-1, :width]], dim=1
                    )
                    future[:, :, self.future_input_dim - width :] = lag

        # ---------------- static context vectors ---------------- #
        c_ctx, h_ctx = self.static_encoder(static)  # (batch, num_layers * hidden)
        c0 = c_ctx.view(batch_size, self.num_layers, self.hidden_size).transpose(0, 1).contiguous()
        h0 = h_ctx.view(batch_size, self.num_layers, self.hidden_size).transpose(0, 1).contiguous()

        # ---------------- LSTM encoder / decoder ---------------- #
        _, (h_n, c_n) = self.encoder_lstm(past, (h0, c0))
        decoder_output, _ = self.decoder_lstm(future, (h_n, c_n))  # (batch, tau, hidden)

        # ---------------- gated skip connection ---------------- #
        skip = self.skip_projection(future)  # (batch, tau, hidden)
        gate = torch.sigmoid(self.skip_gate(decoder_output))
        gated = self.skip_dropout(decoder_output * gate)
        lstm_out = self.layer_norm(skip + gated)

        # ---------------- quantile head ---------------- #
        quantile_output = self.quantile_head(lstm_out)  # (batch, tau, num_quantiles)
        return quantile_output

    # ------------------------------------------------------------------ #
    #  convenience
    # ------------------------------------------------------------------ #
    def extra_repr(self) -> str:  # pragma: no cover - debug helper
        return (
            "past_input_dim={past_input_dim}, future_input_dim={future_input_dim}, "
            "static_input_dim={static_input_dim}, hidden_size={hidden_size}, "
            "num_layers={num_layers}, dropout_ratio={dropout_ratio}, "
            "input_window={input_window}, forecast_horizon={forecast_horizon}, "
            "num_quantiles={num_quantiles}".format(
                past_input_dim=self.past_input_dim,
                future_input_dim=self.future_input_dim,
                static_input_dim=self.static_input_dim,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
                dropout_ratio=self.dropout_ratio,
                input_window=self.input_window,
                forecast_horizon=self.forecast_horizon,
                num_quantiles=self.num_quantiles,
            )
        )