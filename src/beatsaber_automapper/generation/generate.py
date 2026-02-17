"""End-to-end inference pipeline.

Orchestrates the full generation flow:
    Audio -> Audio Encoder -> Stage 1 (onsets) -> Stage 2 (notes) -> Stage 3 (lights) -> export
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def generate_level(
    audio_path: Path,
    output_path: Path,
    difficulty: str = "Expert",
) -> Path:
    """Generate a complete Beat Saber level from an audio file.

    Args:
        audio_path: Path to input audio file (.mp3, .ogg, .wav).
        output_path: Path for the output .zip file.
        difficulty: Difficulty level (Easy, Normal, Hard, Expert, ExpertPlus).

    Returns:
        Path to the generated .zip file.
    """
    raise NotImplementedError("Generation pipeline will be implemented in PR 5")
