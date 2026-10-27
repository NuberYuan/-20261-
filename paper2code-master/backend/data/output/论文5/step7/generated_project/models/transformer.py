import math
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    """Multi-head attention with optional attention weights output."""

    def __init__(self, hidden_size: int, num_attention_heads: int, dropout_ratio: float = 0.1):
        super().__init__()
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                "hidden_size ({}) must be divisible by num_attention_heads ({})".format(
                    hidden_size, num_attention_heads
                )
            )
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.head_dim = hidden_size // num_attention_heads

        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)

        self.attn_dropout = nn.Dropout(dropout_ratio)
        self.resid_dropout = nn.Dropout(dropout_ratio)

    def _shape(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.size()
        x = x.view(batch_size, seq_len, self.num_attention_heads, self.head_dim)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = query.size(0)

        q = self._shape(self.q_proj(query))
        k = self._shape(self.k_proj(key))
        v = self._shape(self.v_proj(value))

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(1).unsqueeze(1)
            elif mask.dim() == 3:
                mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, float("-inf"))

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        context = torch.matmul(attn_weights, v)
        context = context.permute(0, 2, 1, 3).contiguous()
        context = context.view(batch_size, -1, self.hidden_size)

        output = self.resid_dropout(self.out_proj(context))
        return output, attn_weights


class PositionwiseFeedForward(nn.Module):
    """Position-wise feed-forward network used inside each Transformer layer."""

    def __init__(self, hidden_size: int, dropout_ratio: float = 0.1, expansion: int = 4):
        super().__init__()
        inner_size = hidden_size * expansion
        self.linear_1 = nn.Linear(hidden_size, inner_size)
        self.linear_2 = nn.Linear(inner_size, hidden_size)
        self.dropout = nn.Dropout(dropout_ratio)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_2(self.dropout(self.activation(self.linear_1(x))))


class TransformerEncoderLayer(nn.Module):
    """Single Transformer encoder layer (self-attention + feed-forward)."""

    def __init__(self, hidden_size: int, num_attention_heads: int, dropout_ratio: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(hidden_size, num_attention_heads, dropout_ratio)
        self.norm_1 = nn.LayerNorm(hidden_size)
        self.norm_2 = nn.LayerNorm(hidden_size)
        self.dropout_1 = nn.Dropout(dropout_ratio)
        self.dropout_2 = nn.Dropout(dropout_ratio)
        self.feed_forward = PositionwiseFeedForward(hidden_size, dropout_ratio)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        attn_out, attn_weights = self.self_attn(x, x, x, mask=mask)
        x = self.norm_1(x + self.dropout_1(attn_out))
        ff_out = self.feed_forward(x)
        x = self.norm_2(x + self.dropout_2(ff_out))
        return x, attn_weights


class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding."""

    def __init__(self, hidden_size: int, max_seq_len: int = 512, dropout_ratio: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout_ratio)

        position = torch.arange(0, max_seq_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, hidden_size, 2, dtype=torch.float32) * (-math.log(10000.0) / hidden_size)
        )
        pe = torch.zeros(max_seq_len, hidden_size, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :]
        return self.dropout(x)


class Transformer(nn.Module):
    """Transformer encoder used by the Temporal Fusion Transformer.

    The module maps a sequence of hidden states to contextualised hidden states
    and (optionally) returns multi-head attention weights for interpretability.
    """

    def __init__(
        self,
        hidden_size: Optional[int] = None,
        num_layers: Optional[int] = None,
        num_attention_heads: Optional[int] = None,
        dropout_ratio: Optional[float] = None,
        max_seq_len: int = 512,
        use_positional_encoding: bool = True,
    ):
        super().__init__()

        hidden_size = 128 if hidden_size is None else int(hidden_size)
        num_layers = 2 if num_layers is None else int(num_layers)
        num_attention_heads = 4 if num_attention_heads is None else int(num_attention_heads)
        dropout_ratio = 0.1 if dropout_ratio is None else float(dropout_ratio)

        if hidden_size % num_attention_heads != 0:
            num_attention_heads = max(1, math.gcd(hidden_size, num_attention_heads))

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_attention_heads = num_attention_heads
        self.dropout_ratio = dropout_ratio
        self.max_seq_len = max_seq_len
        self.use_positional_encoding = use_positional_encoding

        self.input_proj = nn.Linear(hidden_size, hidden_size)

        self.positional_encoding = (
            PositionalEncoding(hidden_size, max_seq_len, dropout_ratio)
            if use_positional_encoding
            else None
        )

        self.layers = nn.ModuleList(
            [
                TransformerEncoderLayer(hidden_size, num_attention_heads, dropout_ratio)
                for _ in range(num_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout_ratio)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_attention_weights: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            x: Tensor of shape [batch_size, seq_len, hidden_size].
            mask: Optional attention mask of shape [batch_size, seq_len],
                  [batch_size, seq_len, seq_len] or [batch_size, 1, seq_len, seq_len],
                  where 1 (True) marks valid positions and 0 (False) marks masked ones.
            return_attention_weights: whether to also return attention weights.

        Returns:
            output: Tensor of shape [batch_size, seq_len, hidden_size].
            attn_weights: Tensor of shape
                [num_layers, batch_size, num_attention_heads, seq_len, seq_len]
                (only when ``return_attention_weights`` is True).
        """
        if x.dim() != 3:
            raise ValueError("Transformer expects input of shape [batch, seq_len, hidden_size]")

        if x.size(-1) != self.hidden_size:
            raise ValueError(
                "Input feature dim ({}) does not match hidden_size ({})".format(
                    x.size(-1), self.hidden_size
                )
            )

        if self.positional_encoding is not None:
            if x.size(1) > self.max_seq_len:
                raise ValueError(
                    "Sequence length ({}) exceeds max_seq_len ({})".format(
                        x.size(1), self.max_seq_len
                    )
                )
            x = self.positional_encoding(x)

        x = self.dropout(self.input_proj(x))

        attention_weights = [] if return_attention_weights else None

        for layer in self.layers:
            x, attn_w = layer(x, mask=mask)
            if return_attention_weights:
                attention_weights.append(attn_w)

        output = self.final_norm(x)

        if return_attention_weights:
            attn_stack = torch.stack(attention_weights, dim=0)
            return output, attn_stack

        return output