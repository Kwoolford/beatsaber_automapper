"""Shared model building blocks.

Reusable components for the audio encoder, onset model, sequence model,
and lighting model: multi-head attention, positional encoding, feed-forward
networks, and layer normalization utilities.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class SinusoidalPositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for transformer inputs.

    Adds position-dependent sinusoidal signals to input embeddings,
    allowing the model to reason about sequence order.

    Args:
        d_model: Embedding dimension.
        max_len: Maximum sequence length.
        dropout: Dropout rate applied after adding positional encoding.
    """

    def __init__(self, d_model: int = 512, max_len: int = 10000, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input tensor.

        Args:
            x: Input tensor [batch, seq_len, d_model].

        Returns:
            Tensor with positional encoding added.
        """
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)
