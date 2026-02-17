"""CLI: Generate a Beat Saber level from an audio file.

Usage:
    bsa-generate song.mp3 --difficulty Expert --output level.zip
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def main() -> None:
    """Entry point for the bsa-generate CLI command."""
    parser = argparse.ArgumentParser(description="Generate a Beat Saber level from audio")
    parser.add_argument("audio", type=Path, help="Input audio file (.mp3, .ogg, .wav)")
    parser.add_argument("--difficulty", default="Expert", help="Difficulty level")
    parser.add_argument("--output", type=Path, default=Path("level.zip"), help="Output .zip path")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s: %(message)s"
    )
    logger.info("Generate CLI not yet implemented (PR 5)")
    logger.info("Would generate %s level from %s", args.difficulty, args.audio)


if __name__ == "__main__":
    main()
