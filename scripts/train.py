"""CLI: Train a model stage.

Usage:
    bsa-train stage=onset
    bsa-train stage=sequence
    bsa-train stage=lighting
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def main() -> None:
    """Entry point for the bsa-train CLI command."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s: %(message)s"
    )
    logger.info("Training CLI not yet implemented (PR 3)")


if __name__ == "__main__":
    main()
