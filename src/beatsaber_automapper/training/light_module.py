"""Lightning module for Stage 3: Lighting generation training.

Wraps the AudioEncoder + LightingModel for training lighting event
generation conditioned on audio and note sequences.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


class LightingLitModule:
    """Lightning training module for lighting generation.

    Handles training step, validation step, optimizer configuration,
    and metric logging for Stage 3.
    """

    def __init__(self) -> None:
        raise NotImplementedError("LightingLitModule will be implemented in PR 6")
