"""CLI: Download Beat Saber maps from BeatSaver.

Usage:
    # With per-category quotas (recommended):
    bsa-download --quota vanilla:10000 --quota chroma:2000 --quota noodle:1000

    # Simple total count (legacy):
    bsa-download --output data/raw --count 500

Unspecified categories are downloaded opportunistically (no cap).
A manifest.json is maintained in the output directory tracking every map's
category, mod requirements, and download timestamp.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from beatsaber_automapper.data.download import download_maps

logger = logging.getLogger(__name__)


def _parse_quota(value: str) -> tuple[str, int]:
    """Parse a 'category:N' quota string.

    Args:
        value: String in the form "category:N", e.g. "vanilla:10000".

    Returns:
        Tuple of (category_name, count).

    Raises:
        argparse.ArgumentTypeError: If the format is invalid.
    """
    parts = value.split(":", 1)
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(
            f"Invalid quota format {value!r}. Expected 'category:N', e.g. 'vanilla:10000'"
        )
    category, n_str = parts
    try:
        n = int(n_str)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"Invalid quota count {n_str!r} in {value!r}. Must be an integer."
        )
    if n < 0:
        raise argparse.ArgumentTypeError(f"Quota count must be non-negative, got {n}")
    return category.strip(), n


def main() -> None:
    """Entry point for the bsa-download CLI command."""
    parser = argparse.ArgumentParser(
        description="Download Beat Saber maps from BeatSaver",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  bsa-download --quota vanilla:10000 --quota chroma:2000 --quota noodle:1000\n"
            "  bsa-download --count 500  # legacy: total count, no category quotas\n"
        ),
    )
    parser.add_argument("--output", type=Path, default=Path("data/raw"), help="Output directory")
    parser.add_argument(
        "--quota",
        metavar="CATEGORY:N",
        action="append",
        dest="quotas",
        default=[],
        help=(
            "Per-category download quota as 'category:N'. Repeatable. "
            "Valid categories: vanilla, chroma, noodle, mapping_extensions, vivify. "
            "Example: --quota vanilla:10000 --quota chroma:2000"
        ),
    )
    parser.add_argument(
        "--count",
        type=int,
        default=500,
        help="Total maps to download when no --quota flags are given (legacy fallback)",
    )
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

    # Parse --quota flags into a dict; None for omitted categories (opportunistic)
    quotas: dict[str, int | None] | None
    if args.quotas:
        quotas = {}
        for raw in args.quotas:
            category, n = _parse_quota(raw)
            if category in quotas:
                parser.error(f"Duplicate quota for category {category!r}")
            quotas[category] = n
        logger.info("Using per-category quotas: %s", quotas)
    else:
        quotas = None
        logger.info("No quotas specified — using total count: %d", args.count)

    downloaded = download_maps(
        output_dir=args.output,
        quotas=quotas,
        count=args.count,
        min_rating=args.min_rating,
        max_nps=args.max_nps,
        min_year=args.min_year,
        rate_limit=args.rate_limit,
        exclude_ai=not args.include_ai,
    )
    logger.info("Done. Downloaded %d new maps.", len(downloaded))


if __name__ == "__main__":
    main()
