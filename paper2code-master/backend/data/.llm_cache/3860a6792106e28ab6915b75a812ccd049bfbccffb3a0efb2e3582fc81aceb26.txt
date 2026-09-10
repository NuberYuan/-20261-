import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml


def _load_config(config):
    """Load config from a dict, a path, or the project config file.

    All file reads use UTF-8 encoding.
    """
    if config is None:
        cfg_path = Path(__file__).resolve().parents[1] / "config" / "config.yaml"
        if cfg_path.exists():
            with cfg_path.open("r", encoding="utf-8") as f:
                return yaml.safe_load(f)
        return {}
    if isinstance(config, (str, Path)):
        with open(config, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    return config or {}


def _get_dropout(config, fallback=0.1):
    """Read dropout from config if available, otherwise fallback.

    In config/config.yaml, dropout_ratio is stored per dataset. Since this
    model file is dataset-agnostic, we use the first available numeric value.
    """
    cfg = config or {}
    dr = cfg.get("model", {}).get("dropout_ratio")
    if dr is None:
        dr = cfg.get("training", {}).get("dropout_ratio")
    if isinstance(dr, dict):
        numeric_vals = [v for v in dr.values() if isinstance(v, (int, float))]
        if numeric_vals:
            return float(numeric_vals[0])
        return float(fallback)
    if dr is not None:
        return float(dr)
    return float(fallback)


class GatedLinearUnit(nn.Module):
    """Gated Linear Unit used in the TFT gated residual network."""

    def __init__(self, input_size, hidden_size, dropout=0.1):
        super().__init__()
        self.fc = nn.Linear(input_size, hidden_size)
        self.gate = nn.Linear(input_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.fc(x)) * torch.sigmoid(self.gate(x))


class GatedResidualNetwork(nn.Module):
    """Gated Residual Network (GRN) from the TFT paper."""

    def __init__(self, input_size, hidden_size, output_size, dropout=0.1, context_size=None):
        super().__init__()
        self.input_linear = nn.Linear(input_size, hidden_size)
        self.context_linear = None
        if context_size is not None:
            self.context_linear = nn.Linear(context_size, hidden_size, bias=False)
        self.elu = nn.ELU()
        self.glu = GatedLinearUnit(hidden_size, hidden_size, dropout)
        self.output_linear = nn.Linear(hidden_size, output_size)

        if input_size == output_size:
            self.skip = nn.Identity()
        else:
            self.skip = nn.Linear(input_size, output_size)

        self.layer_norm = nn.LayerNorm(output_size)

    def forward(self, x, context=None):
        residual = self.skip(x)
        h = self.input_linear(x)
        if context is not None:
            if self.context_linear is not None:
                h = h + self.context_linear(context)
            else:
                # If no learned context projection is available, assume the
                # context already has the same last-dimension size as hidden.
                h = h + context
        h = self.elu(h)
        h = self.glu(h)
        output = self.output_linear(h)
        return self.layer_norm(residual + output)


class InterpretableMultiHeadAttention(nn.Module):
    """Modified interpretable multi-head attention from TFT.

    Attention is computed as A(Q,K)V with scaled dot-product
    A(Q,K)=Softmax(QK^T/sqrt(d_attn)). Value weights are shared across heads
    and the head outputs are averaged before a final output projection.
    """

    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        self.num_heads = num_heads
        self.d_model = d_model
        self.d_k = d_model // num_heads

        self.w_q = nn.ModuleList(
            [nn.Linear(d_model, self.d_k, bias=False) for _ in range(num_heads)]
        )
        self.w_k = nn.ModuleList(
            [nn.Linear(d_model, self.d_k, bias=False) for _ in range(num_heads)]
        )
        # Value weights are shared across all attention heads.
        self.w_v = nn.Linear(d_model, self.d_k, bias=False)
        self.wo = nn.Linear(self.d_k, d_model, bias=False)

        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        batch_size, query_len, _ = query.size()
        key_len = key.size(1)
        head_outputs = []

        for h in range(self.num_heads):
            q = self.w_q[h](query)          # [B, Q, d_k]
            k = self.w_k[h](key)            # [B, K, d_k]
            v = self.w_v(value)             # shared value projection

            scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
            if mask is not None:
                mask = mask.bool()
                if mask.dim() == 2:
                    # mask is [Q, K], broadcast across batch and heads.
                    scores = scores.masked_fill(~mask.unsqueeze(0), -1e9)
                elif mask.dim() == 3:
                    # mask is already [B, Q, K].
                    scores = scores.masked_fill(~mask, -1e9)

            attn = F.softmax(scores, dim=-1)
            attn = self.dropout(attn)
            head_outputs.append(torch.matmul(attn, v))

        # H_tilde = (1/H) sum_{h=1}^{H} Attention(Q W_Q^(h), K W_K^(h)) V W_V
        H_tilde = torch.stack(head_outputs, dim=0).mean(dim=0)  # [B, Q, d_k]
        return self.wo(H_tilde)                                 # [B, Q, d_model]


class StaticVariableSelection(nn.Module):
    """Variable selection network for static covariates."""

    def __init__(self, num_vars, hidden_size, output_size, dropout=0.1):
        super().__init__()
        self.num_vars = num_vars
        self.hidden_size = hidden_size

        self.var_grns = nn.ModuleList(
            [
                GatedResidualNetwork(
                    input_size=1,
                    hidden_size=hidden_size,
                    output_size=hidden_size,
                    dropout=dropout,
                )
                for _ in range(num_vars)
            ]
        )
        self.weight_grn = GatedResidualNetwork(
            input_size=num_vars * hidden_size,
            hidden_size=hidden_size,
            output_size=num_vars,
            dropout=dropout,
        )

    def forward(self, x):
        # x: [batch, num_static]
        batch_size = x.size(0)
        var_outputs = []
        for i, grn in enumerate(self.var_grns):
            xi = x[:, i : i + 1]      # [batch, 1]
            var_outputs.append(grn(xi))  # [batch, hidden]
        var_outputs = torch.stack(var_outputs, dim=1)  # [B, num_vars, hidden]

        flat = var_outputs.reshape(batch_size, self.num_vars * self.hidden_size)
        weights = self.weight_grn(flat)                # [B, num_vars]
        weights = F.softmax(weights, dim=-1).unsqueeze(-1)  # [B, num_vars, 1]

        output = torch.sum(var_outputs * weights, dim=1)  # [B, hidden]
        return output


class TemporalVariableSelection(nn.Module):
    """Variable selection network for time-varying inputs.

    Input shape is [batch, time_steps, num_vars].
    """

    def __init__(self, num_vars, hidden_size, output_size, dropout=0.1, context_size=None):
        super().__init__()
        self.num_vars = num_vars
        self.hidden_size = hidden_size

        self.var_grns = nn.ModuleList(
            [
                GatedResidualNetwork(
                    input_size=1,
                    hidden_size=hidden_size,
                    output_size=hidden_size,
                    dropout=dropout,
                )
                for _ in range(num_vars)
            ]
        )
        self.weight_grn = GatedResidualNetwork(
            input_size=num_vars * hidden_size,
            hidden_size=hidden_size,
            output_size=num_vars,
            dropout=dropout,
            context_size=context_size,
        )

    def forward(self, x, context=None):
        # x: [batch, time_steps, num_vars]
        batch_size, seq_len, _ = x.size()
        var_outputs = []
        for i, grn in enumerate(self.var_grns):
            xi = x[:, :, i : i + 1]          # [B, T, 1]
            var_outputs.append(grn(xi))      # [B, T, hidden]
        var_outputs = torch.stack(var_outputs, dim=2)  # [B, T, num_vars, hidden]

        flat = var_outputs.reshape(batch_size, seq_len, self.num_vars * self.hidden_size)

        if context is not None and context.dim() == 2:
            context = context.unsqueeze(1).expand(-1, seq_len, -1)

        weights = self.weight_grn(flat, context=context)  # [B, T, num_vars]
        weights = F.softmax(weights, dim=-1).unsqueeze(-1)  # [B, T, num_vars, 1]

        output = torch.sum(var_outputs * weights, dim=2)  # [B, T, hidden]
        return output


class Transformer(nn.Module):
    """Temporal Fusion Transformer (TFT) style Transformer.

    Inputs:
        static_features: [batch, num_static]
        past_features:   [batch, encoder_len, num_past_features]
        future_features: [batch, encoder_len + decoder_len, num_future_features]

    Output:
        Quantile forecasts of shape
        [batch, decoder_len, len(quantiles)].
    """

    def __init__(
        self,
        config=None,
        d_model=160,
        num_heads=4,
        encoder_len=30,
        decoder_len=5,
        num_static=8,
        num_past_features=16,
        num_future_features=8,
        quantiles=None,
        dropout=None,
    ):
        super().__init__()

        # Load config if provided or fallback to project yaml.
        cfg = _load_config(config)

        # `num_classes` is intentionally unused because this is a quantile
        # regression model, not a discrete classification model.
        if dropout is not None:
            self.dropout = float(dropout)
        else:
            self.dropout = _get_dropout(cfg, fallback=0.1)

        if quantiles is None:
            quantiles = [0.1, 0.5, 0.9]
        self.quantiles = list(quantiles)
        self.num_quantiles = len(self.quantiles)

        self.d_model = d_model
        self.num_heads = num_heads
        self.encoder_len = int(encoder_len)
        self.decoder_len = int(decoder_len)
        self.num_static = int(num_static)
        self.num_past_features = int(num_past_features)
        self.num_future_features = int(num_future_features)

        # Static variable selection network.
        self.static_vsn = StaticVariableSelection(
            num_vars=self.num_static,
            hidden_size=d_model,
            output_size=d_model,
            dropout=self.dropout,
        )

        # Temporal variable selection networks.
        self.past_vsn = TemporalVariableSelection(
            num_vars=self.num_past_features,
            hidden_size=d_model,
            output_size=d_model,
            dropout=self.dropout,
            context_size=d_model,
        )
        self.future_vsn = TemporalVariableSelection(
            num_vars=self.num_future_features,
            hidden_size=d_model,
            output_size=d_model,
            dropout=self.dropout,
            context_size=d_model,
        )

        # LSTM encoder/decoder.
        self.lstm_encoder = nn.LSTM(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=1,
            batch_first=True,
        )
        self.lstm_decoder = nn.LSTM(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=1,
            batch_first=True,
        )

        # Gated skip connections around each LSTM.
        self.encoder_gate = GatedLinearUnit(d_model, d_model, dropout=self.dropout)
        self.decoder_gate = GatedLinearUnit(d_model, d_model, dropout=self.dropout)

        # Interpretable temporal attention.
        self.temporal_attention = InterpretableMultiHeadAttention(
            d_model=d_model,
            num_heads=num_heads,
            dropout=self.dropout,
        )

        # Fusion GRN after attention.
        self.attention_grn = GatedResidualNetwork(
            input_size=d_model * 2,
            hidden_size=d_model,
            output_size=d_model,
            dropout=self.dropout,
            context_size=d_model,
        )

        # Quantile-specific output heads.
        self.quantile_heads = nn.ModuleList(
            [nn.Linear(d_model, 1) for _ in range(self.num_quantiles)]
        )

    def forward(self, static_features, past_features, future_features=None):
        batch_size = past_features.size(0)
        device = past_features.device

        if static_features is None:
            static_features = torch.zeros(
                batch_size, self.num_static, device=device
            )

        total_len = self.encoder_len + self.decoder_len
        if future_features is None:
            future_features = torch.zeros(
                batch_size, total_len, self.num_future_features, device=device
            )
        elif future_features.size(1) < total_len:
            pad_size = total_len - future_features.size(1)
            padding = torch.zeros(
                batch_size,
                pad_size,
                self.num_future_features,
                device=device,
                dtype=future_features.dtype,
            )
            future_features = torch.cat([future_features, padding], dim=1)
        else:
            future_features = future_features[:, :total_len, :]

        # Static variable selection.
        static_context = self.static_vsn(static_features)  # [B, d_model]

        # Temporal variable selection.
        past_selected = self.past_vsn(past_features, static_context)  # [B, T_enc, d_model]
        future_selected = self.future_vsn(future_features, static_context)  # [B, total_len, d_model]

        future_known_past = future_selected[:, : self.encoder_len, :]
        future_known_future = future_selected[:, self.encoder_len :, :]

        # Encoder input combines past observed and known future up to t.
        encoder_input = past_selected + future_known_past  # [B, T_enc, d_model]
        decoder_input = future_known_future                  # [B, T_dec, d_model]

        # LSTM sequence-to-sequence block.
        lstm_encoder_output, (h, c) = self.lstm_encoder(encoder_input)
        encoder_output = lstm_encoder_output + self.encoder_gate(encoder_input)

        lstm_decoder_output, _ = self.lstm_decoder(decoder_input, (h, c))
        decoder_output = lstm_decoder_output + self.decoder_gate(decoder_input)

        # Causal attention mask: decoder positions can attend to all encoder
        # positions and only previous/subsequent decoder positions up to themselves.
        mask = torch.zeros(
            self.decoder_len,
            total_len,
            dtype=torch.bool,
            device=device,
        )
        mask[:, : self.encoder_len] = True
        mask[:, self.encoder_len :] = torch.tril(
            torch.ones(self.decoder_len, self.decoder_len, dtype=torch.bool, device=device)
        )

        memory = torch.cat([encoder_output, decoder_output], dim=1)  # [B, total_len, d_model]
        attended = self.temporal_attention(
            query=decoder_output,
            key=memory,
            value=memory,
            mask=mask,
        )

        # Static enrichment for attention fusion.
        static_context_expanded = static_context.unsqueeze(1).expand(
            -1, self.decoder_len, -1
        )
        fused = self.attention_grn(
            torch.cat([decoder_output, attended], dim=-1),
            context=static_context_expanded,
        )

        # Generate one quantile per horizon step.
        quantile_outputs = []
        for head in self.quantile_heads:
            q = head(fused).squeeze(-1)  # [B, T_dec]
            quantile_outputs.append(q)

        return torch.stack(quantile_outputs, dim=-1)  # [B, T_dec, num_quantiles]