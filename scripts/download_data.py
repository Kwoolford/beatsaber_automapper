"""CLI: Download Beat Saber maps from BeatSaver.

Usage:
    bsa-download --output data/raw --count 500
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from beatsaber_automapper.data.download import download_maps

logger = logging.getLogger(__name__)


def main() -> None:
    """Entry point for the bsa-download CLI command."""
    parser = argparse.ArgumentParser(description="Download Beat Saber maps from BeatSaver")
    parser.add_argument("--output", type=Path, default=Path("data/raw"), help="Output directory")
    parser.add_argument("--count", type=int, default=500, help="Number of maps to download")
    parser.add_argument("--min-rating", type=float, default=0.8, help="Minimum upvote ratio")
    parser.add_argument("--max-nps", type=float, default=20.0, help="Maximum notes-per-second")
    parser.add_argument("--min-year", type=int, default=2022, help="Min upload year")
    parser.add_argument("--rate-limit", type=float, default=0.5, help="Seconds between requests")
    parser.add_argument(
        "--include-ai", action="store_true", help="Include AI/automapped maps (excluded by default)"
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s: %(message)s"
    )

    downloaded = download_maps(
        output_dir=args.output,
        count=args.count,
        min_rating=args.min_rating,
        max_nps=args.max_nps,
        min_year=args.min_year,
        rate_limit=args.rate_limit,
        exclude_ai=not args.include_ai,
    )
    logger.info("Done. Downloaded %d maps.", len(downloaded))


if __name__ == "__main__":
    main()
