"""Stage 1: Onset / beat prediction model.

Binary classification per audio frame — predicts whether a note should
appear at each time step. Uses a small 2-layer transformer decoder on
top of audio encoder embeddings, conditioned on difficulty level.

Training: Binary cross-entropy with Gaussian-smoothed labels.
Inference: Peak picking with configurable threshold.
"""

from __future__ import annotations

import logging

import torch.nn as nn

logger = logging.getLogger(__name__)


class OnsetModel(nn.Module):
    """Onset prediction head for Stage 1.

    Args:
        d_model: Input embedding dimension (from audio encoder).
        nhead: Number of attention heads.
        num_layers: Number of transformer decoder layers.
        num_difficulties: Number of difficulty levels (5: Easy-ExpertPlus).
        dropout: Dropout rate.
    """

    def __init__(
        self,
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 2,
        num_difficulties: int = 5,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        raise NotImplementedError("OnsetModel will be implemented in PR 3")
