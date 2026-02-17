"""Lightning module for Stage 1: Onset prediction training.

Wraps the AudioEncoder + OnsetModel for training with binary cross-entropy
loss on Gaussian-smoothed onset labels. Logs onset F1 score on validation.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


class OnsetLitModule:
    """Lightning training module for onset prediction.

    Handles training step, validation step, optimizer configuration,
    and metric logging for Stage 1.
    """

    def __init__(self) -> None:
        raise NotImplementedError("OnsetLitModule will be implemented in PR 3")
