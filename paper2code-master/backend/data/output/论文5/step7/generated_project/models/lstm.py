import math
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple, Union

import torch
import torch.nn as nn
import yaml

try:
    DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "config.yaml"
except NameError:
    DEFAULT_CONFIG_PATH = Path("config") / "config.yaml"


class Lstm(nn.Module):
    """Sequence-to-sequence LSTM encoder-decoder with gated skip connection.

    Past selected features are fed into the encoder and future selected features
    into the decoder. Static context vectors can initialize the first LSTM layer.
    A gated skip connection is applied over the LSTM outputs, followed by one
    linear head per quantile to produce horizon forecasts.

    Outputs are returned with shape:
        [batch_size, decoder_len, num_quantiles, output_size]
    """

    def __init__(
        self,
        input_size: int = 1,
        hidden_size: int = 128,
        num_layers: int = 2,
        output_size: int = 1,
        quantiles: Iterable[float] = (0.1, 0.5, 0.9),
        dropout: Optional[float] = None,
        config: Optional[Union[str, Path, Dict]] = None,
    ) -> None:
        super().__init__()

        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.num_layers = int(num_layers)
        self.output_size = int(output_size)
        self.quantiles = tuple(float(q) for q in quantiles)
        self.num_quantiles = len(self.quantiles)

        if config is None and DEFAULT_CONFIG_PATH.exists():
            config = DEFAULT_CONFIG_PATH
        cfg = self._load_config(config)

        # 默认维度来自 TFT 论文参考实现；config.yaml 未提供 LSTM 输入/隐藏维度，
        # 因此这些值通过构造参数注入，避免在 forward 中硬编码。
        resolved_dropout = dropout if dropout is not None else self._resolve_dropout(cfg, 0.1)
        self.dropout = float(resolved_dropout)

        # PyTorch LSTM 在 num_layers=1 时使用 dropout 会触发警告，这里显式关闭。
        lstm_dropout = self.dropout if self.num_layers > 1 else 0.0

        self.encoder = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=lstm_dropout,
        )
        self.decoder = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=lstm_dropout,
        )

        # Gated skip connection
        self.skip_projection = (
            nn.Identity()
            if self.input_size == self.hidden_size
            else nn.Linear(self.input_size, self.hidden_size)
        )
        self.gate = nn.Linear(self.hidden_size, self.hidden_size)
        self.layer_norm = nn.LayerNorm(self.hidden_size)
        self.dropout_layer = nn.Dropout(self.dropout)

        # Quantile heads: yhat(q, t, tau) = W_q * psi_tilde(t, tau) + b_q
        self.quantile_heads = nn.ModuleList(
            [nn.Linear(self.hidden_size, self.output_size) for _ in range(self.num_quantiles)]
        )

    @staticmethod
    def _load_config(config: Optional[Union[str, Path, Dict]]) -> Dict:
        """Load YAML config safely. All file reads use UTF-8."""
        if config is None:
            return {}
        if isinstance(config, dict):
            return config
        if isinstance(config, (str, Path)):
            path = Path(config)
            if path.exists():
                with open(path, "r", encoding="utf-8") as f:
                    return yaml.safe_load(f) or {}
        return {}

    @staticmethod
    def _resolve_dropout(cfg: Dict, default: float) -> float:
        """Read dropout_ratio from config when available, otherwise use default.

        config.yaml stores dataset-specific dropout ratios, so the first numeric
        value is used as a safe fallback. The value is never allowed to be None.
        """
        for section in ("model", "training"):
            value = cfg.get(section, {}).get("dropout_ratio")
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                return float(value)
            if isinstance(value, dict):
                for dataset in ("Electricity", "Traffic", "Retail", "Volatility"):
                    if dataset in value and isinstance(value[dataset], (int, float)):
                        return float(value[dataset])
        return float(default)

    def _expand_state(self, state: torch.Tensor, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Expand a 2D state [batch, hidden_size] to [num_layers, batch, hidden_size]."""
        if not torch.is_tensor(state):
            state = torch.as_tensor(state, dtype=dtype, device=device)
        else:
            state = state.to(dtype=dtype, device=device)

        if state.dim() == 2:
            state = state.unsqueeze(0).expand(self.num_layers, -1, -1).contiguous()
        elif state.dim() == 3:
            # Already in LSTM-compatible shape.
            pass
        else:
            raise ValueError(f"Initial hidden/cell state must be 2D or 3D, got {state.dim()}D")
        return state

    def _prepare_initial_state(
        self,
        static_context: Optional[Tuple[torch.Tensor, torch.Tensor]],
        device: torch.device,
        dtype: torch.dtype,
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """Create LSTM initial state from static context vectors cc and ch."""
        if static_context is None:
            return None
        if isinstance(static_context, (tuple, list)) and len(static_context) == 2:
            h, c = static_context
            h = self._expand_state(h, device, dtype)
            c = self._expand_state(c, device, dtype)
            return h, c
        return None

    def forward(
        self,
        past_inputs: torch.Tensor,
        future_inputs: torch.Tensor,
        static_context: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            past_inputs: [batch_size, encoder_len, input_size] or [batch_size, encoder_len]
            future_inputs: [batch_size, decoder_len, input_size] or [batch_size, decoder_len]
            static_context: Optional tuple (cc, ch), each [batch_size, hidden_size] or
                [num_layers, batch_size, hidden_size].

        Returns:
            Quantile forecasts with shape
            [batch_size, decoder_len, num_quantiles, output_size].
        """
        if past_inputs.dim() == 2:
            past_inputs = past_inputs.unsqueeze(-1)
        if future_inputs.dim() == 2:
            future_inputs = future_inputs.unsqueeze(-1)

        batch_size = past_inputs.size(0)
        device = past_inputs.device
        dtype = past_inputs.dtype

        encoder_init = self._prepare_initial_state(static_context, device, dtype)
        _, (h_n, c_n) = self.encoder(past_inputs, encoder_init)

        decoder_outputs, _ = self.decoder(future_inputs, (h_n, c_n))

        # Gated skip connection over the LSTM layer.
        residual = self.skip_projection(future_inputs)
        gate = torch.sigmoid(self.gate(decoder_outputs))
        lstm_out = self.layer_norm(residual + gate * decoder_outputs)
        lstm_out = self.dropout_layer(lstm_out)

        quantile_outputs = [head(lstm_out) for head in self.quantile_heads]
        return torch.stack(quantile_outputs, dim=2)