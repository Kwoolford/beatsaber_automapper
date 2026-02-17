"""Export generated tokens to Beat Saber v3 format.

Converts model output tokens into v3 JSON beatmap data and packages
everything into a playable .zip file with Info.dat, song audio, and
difficulty .dat files.
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def tokens_to_beatmap(tokens: list[int], bpm: float, offset: float = 0.0) -> dict:
    """Convert token sequences to v3 beatmap JSON structure.

    Args:
        tokens: Token sequence from the sequence model.
        bpm: Song BPM for beat-time conversion.
        offset: Song time offset in seconds.

    Returns:
        Dictionary matching the v3 .dat JSON structure.
    """
    raise NotImplementedError("Token-to-beatmap export will be implemented in PR 5")


def package_level(
    beatmap_data: dict,
    audio_path: Path,
    output_path: Path,
    song_name: str = "Generated Level",
    bpm: float = 120.0,
) -> Path:
    """Package beatmap data and audio into a Beat Saber .zip.

    Args:
        beatmap_data: v3 beatmap JSON dictionary.
        audio_path: Path to the song audio file.
        output_path: Path for the output .zip file.
        song_name: Song title for Info.dat.
        bpm: Song BPM for Info.dat.

    Returns:
        Path to the generated .zip file.
    """
    raise NotImplementedError("Level packaging will be implemented in PR 5")
