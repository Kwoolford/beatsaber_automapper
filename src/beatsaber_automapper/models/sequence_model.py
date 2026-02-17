"""Stage 2: Note sequence generation model.

Transformer decoder that generates note token sequences at each onset
timestamp, conditioned on audio features and difficulty level. Uses
causal self-attention (autoregressive) and cross-attention to audio.

Generates all v3 note types: colorNotes, bombNotes, obstacles,
sliders (arcs), burstSliders (chains).
"""

from __future__ import annotations

import logging

import torch.nn as nn

logger = logging.getLogger(__name__)


class SequenceModel(nn.Module):
    """Autoregressive note sequence generator for Stage 2.

    Args:
        vocab_size: Size of the token vocabulary.
        d_model: Model dimension.
        nhead: Number of attention heads.
        num_layers: Number of transformer decoder layers.
        dim_feedforward: Feed-forward network dimension.
        num_difficulties: Number of difficulty levels.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        vocab_size: int = 256,
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 8,
        dim_feedforward: int = 2048,
        num_difficulties: int = 5,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        raise NotImplementedError("SequenceModel will be implemented in PR 4")
