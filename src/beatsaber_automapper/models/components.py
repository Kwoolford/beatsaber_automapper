"""Shared model building blocks.

Reusable components for the audio encoder, onset model, sequence model,
and lighting model: multi-head attention, positional encoding, feed-forward
networks, and layer normalization utilities.

Includes KV cache support for fast autoregressive inference.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

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


# ---------------------------------------------------------------------------
# KV Cache for fast autoregressive decoding
# ---------------------------------------------------------------------------


@dataclass
class KVCache:
    """Key-Value cache for a single attention layer.

    Stores concatenated K and V tensors from previous decoding steps.
    On each new step, new K/V are appended and the full cache is returned
    for attention computation.
    """

    k: torch.Tensor | None = None  # [B, num_heads, seq_len, head_dim]
    v: torch.Tensor | None = None

    def update(self, new_k: torch.Tensor, new_v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Append new K/V and return full cached K/V.

        Args:
            new_k: New key tensor [B, num_heads, 1, head_dim].
            new_v: New value tensor [B, num_heads, 1, head_dim].

        Returns:
            Tuple of (full_k, full_v) with all cached positions.
        """
        if self.k is None:
            self.k = new_k
            self.v = new_v
        else:
            self.k = torch.cat([self.k, new_k], dim=2)
            self.v = torch.cat([self.v, new_v], dim=2)
        return self.k, self.v

    @property
    def seq_len(self) -> int:
        """Current number of cached positions."""
        return 0 if self.k is None else self.k.shape[2]


@dataclass
class LayerCaches:
    """KV caches for all layers in a decoder: self-attention + cross-attention."""

    self_attn: KVCache = field(default_factory=KVCache)
    cross_attn: KVCache = field(default_factory=KVCache)


class CachedDecoderLayer(nn.Module):
    """Transformer decoder layer with KV cache support.

    Drop-in replacement for nn.TransformerDecoderLayer that supports
    incremental decoding by caching self-attention and cross-attention
    keys/values from previous steps.

    Args:
        d_model: Model dimension.
        nhead: Number of attention heads.
        dim_feedforward: FFN dimension.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        d_model: int = 512,
        nhead: int = 8,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: torch.Tensor | None = None,
        tgt_key_padding_mask: torch.Tensor | None = None,
        cache: LayerCaches | None = None,
    ) -> torch.Tensor:
        """Forward pass with optional KV cache.

        When cache is provided, only the new token position is processed
        and the cache is updated in-place.

        Args:
            tgt: Target sequence [B, S, D] (full sequence or single new token).
            memory: Encoder output [B, T, D].
            tgt_mask: Causal attention mask [S, S] (not needed when using cache).
            tgt_key_padding_mask: Padding mask for target [B, S].
            cache: Optional LayerCaches for incremental decoding.

        Returns:
            Output tensor, same shape as tgt.
        """
        # Pre-norm self-attention
        x = self.norm1(tgt)

        if cache is not None:
            # Incremental: only compute attention for the new position(s)
            # but attend to all cached positions
            q = x
            # For self-attention: cache K/V from all previous + current positions
            # We need to project K/V for the new token and concatenate with cache
            # nn.MultiheadAttention.forward handles this via the key/value inputs
            # We concatenate the cached K/V with current K/V internally

            # Get K/V for current position(s) by using x as both key and value
            # But we need the full sequence for K/V attention
            if cache.self_attn.k is not None:
                # Reconstruct full K/V: cached positions + new position
                # We store the pre-norm'd values as K/V
                full_kv = torch.cat([
                    cache.self_attn.k,  # [B, cached_len, D] stored as 3D
                    x,
                ], dim=1)
            else:
                full_kv = x

            # Update cache with new pre-norm token(s)
            if cache.self_attn.k is None:
                cache.self_attn.k = x.clone()
                cache.self_attn.v = x.clone()
            else:
                cache.self_attn.k = full_kv.clone()
                cache.self_attn.v = full_kv.clone()

            attn_out, _ = self.self_attn(q, full_kv, full_kv)
        else:
            attn_out, _ = self.self_attn(x, x, x, attn_mask=tgt_mask,
                                          key_padding_mask=tgt_key_padding_mask)

        tgt = tgt + self.dropout1(attn_out)

        # Pre-norm cross-attention
        x = self.norm2(tgt)

        if cache is not None and cache.cross_attn.k is not None:
            # Cross-attention K/V are static (from encoder) — reuse from cache
            cross_out, _ = self.cross_attn(x, cache.cross_attn.k, cache.cross_attn.v)
        else:
            cross_out, _ = self.cross_attn(x, memory, memory)
            if cache is not None:
                # Cache cross-attention K/V (static, computed once)
                cache.cross_attn.k = memory
                cache.cross_attn.v = memory

        tgt = tgt + self.dropout2(cross_out)

        # Pre-norm FFN
        x = self.norm3(tgt)
        tgt = tgt + self.ffn(x)

        return tgt


class CachedTransformerDecoder(nn.Module):
    """Stacked CachedDecoderLayers with KV cache support.

    Replaces nn.TransformerDecoder for models that need fast
    autoregressive inference.

    Args:
        d_model: Model dimension.
        nhead: Number of attention heads.
        num_layers: Number of decoder layers.
        dim_feedforward: FFN dimension.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 8,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList([
            CachedDecoderLayer(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])
        self.num_layers = num_layers

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: torch.Tensor | None = None,
        tgt_key_padding_mask: torch.Tensor | None = None,
        layer_caches: list[LayerCaches] | None = None,
    ) -> torch.Tensor:
        """Forward pass through all decoder layers.

        Args:
            tgt: Target sequence [B, S, D].
            memory: Encoder output [B, T, D].
            tgt_mask: Causal mask [S, S].
            tgt_key_padding_mask: Padding mask [B, S].
            layer_caches: Per-layer KV caches for incremental decoding.

        Returns:
            Output tensor [B, S, D].
        """
        x = tgt
        for i, layer in enumerate(self.layers):
            cache = layer_caches[i] if layer_caches is not None else None
            x = layer(x, memory, tgt_mask=tgt_mask,
                      tgt_key_padding_mask=tgt_key_padding_mask, cache=cache)
        return x

    def new_caches(self) -> list[LayerCaches]:
        """Create fresh empty caches for all layers."""
        return [LayerCaches() for _ in range(self.num_layers)]
