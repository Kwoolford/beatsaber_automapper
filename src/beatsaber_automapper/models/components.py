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

    def _extend_pe(self, length: int) -> None:
        """Extend the positional encoding buffer to handle longer sequences."""
        d_model = self.pe.size(2)
        pe = torch.zeros(length, d_model, device=self.pe.device, dtype=self.pe.dtype)
        position = torch.arange(0, length, dtype=torch.float, device=self.pe.device).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float, device=self.pe.device)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(0)  # [1, length, d_model]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input tensor.

        Args:
            x: Input tensor [batch, seq_len, d_model].

        Returns:
            Tensor with positional encoding added.
        """
        if x.size(1) > self.pe.size(1):
            self._extend_pe(x.size(1))
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


def peak_picking(
    probs: torch.Tensor,
    threshold: float = 0.5,
    min_distance: int = 5,
) -> torch.Tensor:
    """Pick peaks from a 1-D probability tensor.

    Finds frames above threshold that are local maxima, then applies
    greedy suppression by min_distance (keeping the highest first).

    Args:
        probs: 1-D tensor of probabilities [T].
        threshold: Minimum probability to consider.
        min_distance: Minimum frames between peaks.

    Returns:
        Sorted 1-D tensor of frame indices where peaks occur.
    """
    if probs.ndim != 1:
        raise ValueError(f"Expected 1-D tensor, got {probs.ndim}-D")

    # Find candidates above threshold
    above = (probs >= threshold).nonzero(as_tuple=True)[0]
    if len(above) == 0:
        return torch.tensor([], dtype=torch.long, device=probs.device)

    # Filter to local maxima (higher than both neighbors)
    peaks = []
    for idx in above:
        i = idx.item()
        left = probs[i - 1].item() if i > 0 else -1.0
        right = probs[i + 1].item() if i < len(probs) - 1 else -1.0
        if probs[i].item() >= left and probs[i].item() >= right:
            peaks.append(i)

    if not peaks:
        return torch.tensor([], dtype=torch.long, device=probs.device)

    # Greedy suppression: sort by probability descending, keep if far enough
    peaks_t = torch.tensor(peaks, dtype=torch.long, device=probs.device)
    peak_probs = probs[peaks_t]
    order = peak_probs.argsort(descending=True)
    peaks_sorted = peaks_t[order]

    kept: list[int] = []
    for p in peaks_sorted.tolist():
        if all(abs(p - k) >= min_distance for k in kept):
            kept.append(p)

    result = torch.tensor(sorted(kept), dtype=torch.long, device=probs.device)
    return result
