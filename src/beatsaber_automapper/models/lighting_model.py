"""Stage 3: Lighting event generation model.

Transformer decoder conditioned on both audio features and the generated
note sequence to produce synchronized lighting events.

Output: basicBeatmapEvents, colorBoostBeatmapEvents,
lightColorEventBoxGroups, lightRotationEventBoxGroups.
"""

from __future__ import annotations

import logging

import torch.nn as nn

logger = logging.getLogger(__name__)


class LightingModel(nn.Module):
    """Lighting event generator for Stage 3.

    Args:
        d_model: Model dimension.
        nhead: Number of attention heads.
        num_layers: Number of transformer decoder layers.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        raise NotImplementedError("LightingModel will be implemented in PR 6")
