"""CLI: Preprocess downloaded maps into training tensors.

Usage:
    bsa-preprocess --input data/raw --output data/processed
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def main() -> None:
    """Entry point for the bsa-preprocess CLI command."""
    parser = argparse.ArgumentParser(description="Preprocess Beat Saber maps into tensors")
    parser.add_argument("--input", type=Path, default=Path("data/raw"), help="Raw data directory")
    parser.add_argument(
        "--output", type=Path, default=Path("data/processed"), help="Output directory"
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s: %(message)s"
    )
    logger.info("Preprocess CLI not yet implemented (PR 2)")
    logger.info("Would process maps from %s to %s", args.input, args.output)


if __name__ == "__main__":
    main()
