"""CLI: Download Beat Saber maps from BeatSaver.

Usage:
    bsa-download --output data/raw --count 500
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def main() -> None:
    """Entry point for the bsa-download CLI command."""
    parser = argparse.ArgumentParser(description="Download Beat Saber maps from BeatSaver")
    parser.add_argument("--output", type=Path, default=Path("data/raw"), help="Output directory")
    parser.add_argument("--count", type=int, default=500, help="Number of maps to download")
    parser.add_argument("--min-rating", type=float, default=0.8, help="Minimum upvote ratio")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s: %(message)s"
    )
    logger.info("Download CLI not yet implemented (PR 2)")
    logger.info("Would download %d maps to %s", args.count, args.output)


if __name__ == "__main__":
    main()
