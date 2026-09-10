import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml


def _load_config(config_path=None):
    """Load config.yaml from the project root using utf-8 encoding.

    The model file lives in models/, so the default path points to
    ../config/config.yaml relative to this file. If the file is not found
    there (for example when the working directory is the project root),
    fall back to ./config/config.yaml. If neither exists, return an empty
    dict so that parameterless calls remain usable with conservative
    defaults.
    """
    if config_path is None:
        config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'config.yaml')

    if not os.path.exists(config_path):
        alt_path = os.path.join(os.getcwd(), 'config', 'config.yaml')
        if os.path.exists(alt_path):
            config_path = alt_path
        else:
            return {}

    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


class _GraphConvLayer(nn.Module):
    """Simple dense graph convolution layer used by the GP2 acyclicity model.

    It applies the symmetric normalisation A_hat = D^{-1/2} (A + I) D^{-1/2}
    followed by a linear transformation and ReLU activation. The
    implementation is intentionally small and does not rely on any external
    graph library.
    """

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.bias = nn.Parameter(torch.zeros(out_dim))

    def forward(self, x, adj):
        # x: [B, N, in_dim], adj: [B, N, N]
        B, N, _ = adj.size()
        adj_self = adj + torch.eye(N, dtype=adj.dtype, device=adj.device).unsqueeze(0)
        degree = adj_self.sum(dim=-1, keepdim=True)  # [B, N, 1]
        degree_inv_sqrt = torch.pow(degree.clamp(min=1e-8), -0.5)
        norm_adj = degree_inv_sqrt * adj_self * degree_inv_sqrt.transpose(-1, -2)
        support = torch.bmm(norm_adj, x)
        out = self.linear(support) + self.bias.unsqueeze(0).unsqueeze(0)
        return F.relu(out)


class GP2AcyclicityProgram(nn.Module):
    """Model structure for the GP2 acyclicity program.

    The network receives preprocessed graph inputs (node features plus an
    adjacency matrix) and outputs logits for downstream evaluation. All
    shape-related hyperparameters are read from config['algorithm_params']
    when available; otherwise conservative defaults derived from the paper
    context are used. The model is randomly initialised by design and does
    not load any pretrained weights.
    """

    def __init__(self, config=None):
        super().__init__()

        if config is None:
            config = _load_config()

        # algorithm_params exists in the supplied config.yaml; use it as the
        # single source for optional model hyperparameters. If it is not a
        # dict (e.g. missing or null), fall back to an empty dict.
        params = config.get('algorithm_params', {})
        if not isinstance(params, dict):
            params = {}

        # Default values are chosen because the paper does not provide exact
        # architecture details. GP2 acyclicity detection is a binary decision
        # (acyclic vs. cyclic), hence num_classes=2. Synthetic graph inputs
        # are represented with 8-dimensional node features by default; a
        # hidden dimension of 64 and two graph-convolution layers keep the
        # parameter count in a reasonable range.
        raw_num_classes = params.get('num_classes', 2)
        self.num_classes = int(raw_num_classes) if raw_num_classes is not None else 2

        raw_input_dim = params.get('input_dim', 8)
        self.input_dim = int(raw_input_dim) if raw_input_dim is not None else 8

        raw_hidden_dim = params.get('hidden_dim', 64)
        self.hidden_dim = int(raw_hidden_dim) if raw_hidden_dim is not None else 64

        raw_num_layers = params.get('num_layers', 2)
        self.num_layers = int(raw_num_layers) if raw_num_layers is not None else 2

        raw_dropout = params.get('dropout_ratio', 0.2)
        self.dropout_ratio = float(raw_dropout) if raw_dropout is not None else 0.2

        raw_pretrained = params.get('pretrained', False)
        self.pretrained = bool(raw_pretrained) if raw_pretrained is not None else False

        # Build GNN layers. The first layer maps input_dim -> hidden_dim;
        # subsequent layers map hidden_dim -> hidden_dim.
        layers = []
        in_dim = self.input_dim
        for _ in range(self.num_layers):
            layers.append(_GraphConvLayer(in_dim, self.hidden_dim))
            in_dim = self.hidden_dim
        self.gnn_layers = nn.ModuleList(layers)

        # Classification head: summary graph representation -> logits.
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout_ratio),
            nn.Linear(self.hidden_dim, self.num_classes),
        )

    def forward(self, x=None, adj=None):
        """Forward pass for batched or single graph inputs.

        Args:
            x: Optional node features. Shape [N, F] or [B, N, F]. If None,
                a small random default graph is created for smoke tests.
            adj: Optional adjacency matrix. Shape [N, N] or [B, N, N]. If
                None, a lower-triangular adjacency matrix is created to
                simulate an acyclic graph.

        Returns:
            Logits tensor of shape [B, num_classes].
        """
        # Parameterless-call fallback: create a small random graph.
        if x is None:
            B, N = 1, 8
            x = torch.randn(B, N, self.input_dim)

        if adj is None:
            B, N = x.size(0), x.size(1)
            adj = torch.zeros(B, N, N, dtype=x.dtype, device=x.device)
            for b in range(B):
                # Lower-triangular adjacency means an acyclic directed graph,
                # which is a sensible default for this model.
                adj[b] = torch.tril(torch.ones(N, N, dtype=x.dtype, device=x.device), diagonal=-1)

        # Normalise input shapes.
        if x.dim() == 2:
            x = x.unsqueeze(0)
        elif x.dim() != 3:
            raise ValueError("x must be a 2D or 3D tensor, got shape {}".format(tuple(x.shape)))

        if adj.dim() == 2:
            adj = adj.unsqueeze(0)
        elif adj.dim() != 3:
            raise ValueError("adj must be a 2D or 3D tensor, got shape {}".format(tuple(adj.shape)))

        # Convert to float32 and move to the same device to avoid bmm issues.
        x = x.float()
        adj = adj.float()
        if adj.device != x.device:
            adj = adj.to(x.device)

        B, N = x.size(0), x.size(1)

        # Pad or truncate the adjacency matrix if its node count differs from
        # the node count in x. This keeps the forward pass robust for
        # preprocessed inputs with partial alignment.
        if adj.size(1) != N or adj.size(2) != N:
            new_adj = torch.zeros(B, N, N, dtype=adj.dtype, device=adj.device)
            min_n = min(N, adj.size(1), adj.size(2))
            new_adj[:, :min_n, :min_n] = adj[:, :min_n, :min_n]
            adj = new_adj

        # Align batch size between x and adj. If one batch is larger, the
        # smaller batch size is used to prevent shape mismatches.
        if adj.size(0) != B:
            if adj.size(0) == 1:
                adj = adj.repeat(B, 1, 1)
            else:
                B = min(B, adj.size(0))
                x = x[:B]
                adj = adj[:B]

        # Message passing through the graph convolution layers.
        for layer in self.gnn_layers:
            x = layer(x, adj)

        # Readout: mean-pool the node embeddings into a graph embedding.
        graph_repr = x.mean(dim=1)  # [B, hidden_dim]

        # Classification head.
        return self.classifier(graph_repr)