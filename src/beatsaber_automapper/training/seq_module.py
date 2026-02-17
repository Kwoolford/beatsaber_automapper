"""Lightning module for Stage 2: Note sequence generation training.

Wraps the AudioEncoder + SequenceModel for teacher-forced training
with cross-entropy loss over the token vocabulary.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


class SequenceLitModule:
    """Lightning training module for note sequence generation.

    Handles training step, validation step, optimizer configuration,
    and metric logging for Stage 2.
    """

    def __init__(self) -> None:
        raise NotImplementedError("SequenceLitModule will be implemented in PR 4")
