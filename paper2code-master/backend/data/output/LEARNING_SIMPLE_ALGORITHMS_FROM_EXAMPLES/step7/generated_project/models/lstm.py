"""LSTM controller model for the time-series prediction task.

This module implements the LSTM architecture described in:
    LEARNING SIMPLE ALGORITHMS FROM EXAMPLES

The paper explores 1-layer LSTM controllers with 200 or 400 hidden units.
The controller receives interface readings at each time step and outputs
symbols from a vocabulary via a softmax layer.

All hyperparameters are read from config/config.yaml.  When a value is
missing (None) in the config, a concrete default derived from the paper's
description is used instead of passing None to torch shape parameters.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional

import torch
import torch.nn as nn


class Lstm(nn.Module):
    """One-layer LSTM controller.

    Parameters
    ----------
    config : dict, optional
        Parsed configuration dictionary.  If None, safe defaults are used.
    input_size : int, optional
        Dimensionality of the controller input at each time step.
        If None, it is inferred from ``config`` or set to a paper-derived
        default.
    hidden_size : int, optional
        Number of LSTM hidden units.  The paper uses 200 or 400.
        If None, defaults to 200.
    num_classes : int, optional
        Vocabulary size for the symbol-prediction softmax output.
        If None, a paper-derived default is used (see notes below).
    dropout_ratio : float, optional
        Dropout probability applied to the LSTM output.  The paper does not
        explicitly report dropout for the controller; if None, 0.0 is used.
    """

    # Paper-derived defaults.  The paper's controller emits symbols from a
    # vocabulary.  For the time-series prediction task, the vocabulary is
    # composed of the digits 0-9 plus a small set of special symbols used by
    # the interface (e.g., start/end markers).  We therefore use 12 as a
    # conservative concrete default when config does not specify a value.
    DEFAULT_INPUT_SIZE = 10
    DEFAULT_HIDDEN_SIZE = 200
    DEFAULT_NUM_CLASSES = 12
    DEFAULT_DROPOUT = 0.0

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        input_size: Optional[int] = None,
        hidden_size: Optional[int] = None,
        num_classes: Optional[int] = None,
        dropout_ratio: Optional[float] = None,
    ) -> None:
        super().__init__()

        cfg = config if config is not None else {}

        # Resolve input size.
        if input_size is not None:
            self.input_size = int(input_size)
        else:
            feature_dim = cfg.get("data", {}).get("feature_dim")
            if feature_dim is not None:
                self.input_size = int(feature_dim)
            else:
                # Paper-derived default: interface readings are represented
                # as a fixed-size vector.  We use 10 as a safe concrete value.
                self.input_size = self.DEFAULT_INPUT_SIZE

        # Resolve hidden size.
        if hidden_size is not None:
            self.hidden_size = int(hidden_size)
        else:
            # The paper explores 200-unit and 400-unit LSTM controllers.
            # We default to the smaller configuration.
            self.hidden_size = self.DEFAULT_HIDDEN_SIZE

        # Resolve number of output classes (vocabulary size).
        if num_classes is not None:
            self.num_classes = int(num_classes)
        else:
            model_cfg = cfg.get("model", {})
            cfg_num_classes = model_cfg.get("num_classes")
            if cfg_num_classes is not None:
                self.num_classes = int(cfg_num_classes)
            else:
                # Paper-derived default: digits 0-9 plus special symbols.
                self.num_classes = self.DEFAULT_NUM_CLASSES

        # Resolve dropout ratio.
        if dropout_ratio is not None:
            self.dropout_ratio = float(dropout_ratio)
        else:
            model_cfg = cfg.get("model", {})
            cfg_dropout = model_cfg.get("dropout_ratio")
            if cfg_dropout is not None:
                self.dropout_ratio = float(cfg_dropout)
            else:
                self.dropout_ratio = self.DEFAULT_DROPOUT

        # One-layer LSTM as described in the paper.
        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=1,
            batch_first=True,
        )

        # Optional dropout applied to LSTM outputs before the classifier.
        self.dropout = nn.Dropout(p=self.dropout_ratio) if self.dropout_ratio > 0.0 else nn.Identity()

        # Symbol-prediction head.  The paper uses a softmax over the
        # vocabulary; CrossEntropyLoss internally applies log-softmax, so we
        # output raw logits here.
        self.fc = nn.Linear(self.hidden_size, self.num_classes)

        # Weight initialization is intentionally left to training/train.py's
        # initialize_model function.  We only perform a minimal safe
        # initialization here to keep the module usable when instantiated
        # directly for smoke tests.
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        """Initialize parameters with small random values.

        This mirrors common PyTorch LSTM initialization and is only a
        fallback.  The training pipeline may re-initialize weights according
        to the configured scheme.
        """
        for name, param in self.lstm.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.zeros_(param)

        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(
        self,
        x: torch.Tensor,
        hidden: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass through the LSTM controller.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape ``(batch_size, seq_len, input_size)``.
        hidden : tuple of torch.Tensor, optional
            Initial hidden state ``(h_0, c_0)``.  Each has shape
            ``(num_layers, batch_size, hidden_size)``.  If None, the LSTM
            uses zero initial states.

        Returns
        -------
        logits : torch.Tensor
            Symbol logits of shape ``(batch_size, seq_len, num_classes)``.
        (h_n, c_n) : tuple of torch.Tensor
            Final hidden and cell states, each of shape
            ``(num_layers, batch_size, hidden_size)``.
        """
        lstm_out, (h_n, c_n) = self.lstm(x, hidden)
        lstm_out = self.dropout(lstm_out)
        logits = self.fc(lstm_out)
        return logits, (h_n, c_n)

    def get_initial_state(
        self, batch_size: int, device: Optional[torch.device] = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return zero initial states for the LSTM.

        Parameters
        ----------
        batch_size : int
            Number of sequences in the batch.
        device : torch.device, optional
            Device on which to create the tensors.  Defaults to the device
            of the LSTM weights.

        Returns
        -------
        (h_0, c_0) : tuple of torch.Tensor
            Zero initial hidden and cell states.
        """
        if device is None:
            device = next(self.parameters()).device

        h_0 = torch.zeros(1, batch_size, self.hidden_size, device=device)
        c_0 = torch.zeros(1, batch_size, self.hidden_size, device=device)
        return h_0, c_0