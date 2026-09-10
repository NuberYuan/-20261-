"""GP2 connectedness programme model.

This module defines a randomly-initialised graph neural network that learns the
two-class decision boundary of a GP 2 connectedness programme.  The architecture
is intentionally small and does not load external weights; model initialisation
is deferred to ``training/train.py`` where the training loop handles it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, Optional

import torch
import torch.nn as nn
import yaml


class GraphConvLayer(nn.Module):
    """A simple message-passing layer for undirected graphs."""

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)
        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """Apply one graph convolution step.

        Args:
            x: Node features of shape ``[B, N, in_dim]``.
            adj: Adjacency matrix of shape ``[B, N, N]``.

        Returns:
            Updated node features of shape ``[B, N, out_dim]``.
        """
        adj = adj.to(dtype=x.dtype)
        degree = adj.sum(dim=-1, keepdim=True).clamp(min=1.0)
        norm_adj = adj / degree
        h = torch.bmm(norm_adj, x)
        h = self.linear(h)
        h = self.norm(h)
        return self.activation(h)


class GP2ConnectednessProgram(nn.Module):
    """Graph model for predicting whether an input graph is connected.

    The forward pass receives a node-feature matrix and an optional adjacency
    matrix.  It applies a small stack of graph convolutions, pools node
    features via mean and max readouts, and produces class logits.
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        threshold: float = 0.5,
        batch_size: int = 32,
    ) -> None:
        super().__init__()

        self.config = self._load_config(config)
        params: Dict[str, Any] = {}
        if isinstance(self.config, dict):
            params = self.config.get("algorithm_params") or {}

        # Paper-level defaults:
        # connectedness is a binary classification problem, so num_classes=2.
        # No node-feature dimension is specified in the paper; 16 is a
        # conservative default for synthetic graph features and can be
        # overridden through ``algorithm_params.input_dim``.
        self.num_classes = max(2, self._read_int(params, "num_classes", 2))
        self.input_dim = max(1, self._read_int(params, "input_dim", 16))
        self.hidden_dim = max(1, self._read_int(params, "hidden_dim", 64))
        self.num_layers = max(1, self._read_int(params, "num_layers", 2))
        self.dropout_ratio = min(
            1.0, max(0.0, self._read_float(params, "dropout_ratio", 0.1))
        )
        self.threshold = float(threshold)
        self.batch_size = int(batch_size)

        self.input_proj = nn.Linear(self.input_dim, self.hidden_dim)
        self.gcn_layers = nn.ModuleList(
            [GraphConvLayer(self.hidden_dim, self.hidden_dim)
             for _ in range(self.num_layers)]
        )
        self.dropout = nn.Dropout(self.dropout_ratio)

        # Mean and max readout are concatenated.
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout_ratio),
            nn.Linear(self.hidden_dim, self.num_classes),
        )

    @staticmethod
    def _load_config(config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Return the supplied config or load ``config/config.yaml``.

        The config path is located relative to this file: ``models/`` is one
        level below the project root.  This avoids hard-coding an absolute
        path while still respecting the required UTF-8 read policy.
        """
        if config is not None:
            return config

        config_path = Path(__file__).resolve().parents[1] / "config" / "config.yaml"
        if config_path.exists():
            with open(config_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        return {}

    @staticmethod
    def _read_int(params: Dict[str, Any], key: str, default: int) -> int:
        """Read an integer parameter, falling back to ``default`` on None/invalid."""
        value = params.get(key, default)
        if value is None:
            return default
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _read_float(params: Dict[str, Any], key: str, default: float) -> float:
        """Read a float parameter, falling back to ``default`` on None/invalid."""
        value = params.get(key, default)
        if value is None:
            return default
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    def _ensure_feature_dim(self, x: torch.Tensor) -> torch.Tensor:
        """Pad or truncate node features to match ``self.input_dim``."""
        if x.size(-1) == self.input_dim:
            return x
        if x.size(-1) < self.input_dim:
            pad_shape = list(x.shape[:-1]) + [self.input_dim - x.size(-1)]
            pad = torch.zeros(pad_shape, dtype=x.dtype, device=x.device)
            return torch.cat([x, pad], dim=-1)
        return x[..., : self.input_dim]

    @staticmethod
    def _ensure_adj_size(adj: torch.Tensor, n_nodes: int) -> torch.Tensor:
        """Return an adjacency matrix of shape ``[B, N, N]`` by slice or zero pad."""
        if adj.size(1) == n_nodes and adj.size(2) == n_nodes:
            return adj

        batch = adj.size(0)
        new_adj = torch.zeros(
            batch, n_nodes, n_nodes, dtype=adj.dtype, device=adj.device
        )
        h = min(adj.size(1), n_nodes)
        w = min(adj.size(2), n_nodes)
        new_adj[:, :h, :w] = adj[:, :h, :w]
        return new_adj

    def forward(
        self,
        x: torch.Tensor,
        adj: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Node features of shape ``[N, F]`` or ``[B, N, F]``.
            adj: Adjacency matrix of shape ``[N, N]`` or ``[B, N, N]``.
                If omitted, an identity graph is assumed, which is useful for
                isolated feature-only baselines.
            mask: Optional node mask of shape ``[N]``, ``[B, N]`` or
                ``[B, N, 1]``.  Masked nodes contribute less to pooling.

        Returns:
            Class logits of shape ``[B, num_classes]``.
        """
        if x.dim() == 2:
            x = x.unsqueeze(0)
        if x.dim() != 3:
            raise ValueError(
                "Expected x to have shape [N, F] or [B, N, F], "
                f"got {tuple(x.shape)}."
            )

        x = self._ensure_feature_dim(x)
        batch, n_nodes, _ = x.shape

        if adj is None:
            adj = torch.eye(
                n_nodes, device=x.device, dtype=x.dtype
            ).unsqueeze(0).expand(batch, n_nodes, n_nodes)
        if adj.dim() == 2:
            adj = adj.unsqueeze(0).expand(batch, n_nodes, n_nodes)
        if adj.dim() != 3:
            raise ValueError(
                "Expected adj to have shape [N, N] or [B, N, N], "
                f"got {tuple(adj.shape)}."
            )
        adj = self._ensure_adj_size(adj, n_nodes)

        if mask is not None:
            if mask.dim() == 1:
                mask = mask.unsqueeze(0)
            if mask.dim() == 2:
                mask = mask.unsqueeze(-1)
            if mask.dim() != 3 or mask.size(-1) != 1:
                raise ValueError(
                    "Expected mask to have shape [N], [B, N], or [B, N, 1], "
                    f"got {tuple(mask.shape)}."
                )
            mask = mask.to(dtype=x.dtype)

        h = self.input_proj(x)

        for gcn_layer in self.gcn_layers:
            residual = h
            h = gcn_layer(h, adj)
            h = self.dropout(h)
            h = h + residual

        h = self.dropout(h)

        if mask is not None:
            h = h * mask
            count = mask.sum(dim=1).clamp(min=1e-6)
            pooled_mean = h.sum(dim=1) / count
            masked_neg_inf = h.masked_fill(mask < 0.5, float("-inf"))
            pooled_max = torch.nan_to_num(
                masked_neg_inf.amax(dim=1),
                nan=0.0,
                posinf=1e9,
                neginf=-1e9,
            )
            pooled = torch.cat([pooled_mean, pooled_max], dim=-1)
        else:
            pooled = torch.cat([h.mean(dim=1), h.amax(dim=1)], dim=-1)

        return self.classifier(pooled)