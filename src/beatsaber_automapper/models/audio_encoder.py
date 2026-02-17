"""Shared audio encoder: CNN frontend + Transformer encoder.

Processes mel spectrograms into contextualized audio frame embeddings
used by all three pipeline stages. Trained end-to-end on Beat Saber data
to capture low-level rhythmic features (transients, beat subdivisions).

Architecture:
    Mel spectrogram [80, T] -> 2D CNN (3-4 layers) -> Transformer encoder
    (6-8 layers, 8 heads, d_model=512) -> frame embeddings [T', 512]
"""

from __future__ import annotations

import logging

import torch.nn as nn

logger = logging.getLogger(__name__)


class AudioEncoder(nn.Module):
    """CNN + Transformer encoder for audio feature extraction.

    Args:
        n_mels: Number of mel frequency bands in input spectrogram.
        d_model: Transformer model dimension.
        nhead: Number of attention heads.
        num_layers: Number of transformer encoder layers.
        dim_feedforward: Feed-forward network dimension.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        n_mels: int = 80,
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        raise NotImplementedError("AudioEncoder will be implemented in PR 3")
