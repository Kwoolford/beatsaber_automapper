"""Filter outlier maps from the training dataset.

Scans frame_index.json and .pt files to identify problematic maps, then writes
a blacklist file that datasets can use to skip them.

Usage:
    python scripts/filter_outliers.py --data-dir data/processed
    python scripts/filter_outliers.py --data-dir data/processed --dry-run
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def build_blacklist(data_dir: Path, dry_run: bool = False) -> dict[str, str]:
    """Scan dataset and identify outlier maps.

    Returns:
        Dict mapping song_id -> reason for blacklisting.
    """
    frame_index_path = data_dir / "frame_index.json"
    if not frame_index_path.exists():
        logger.error("frame_index.json not found in %s", data_dir)
        return {}

    with open(frame_index_path) as f:
        frame_index = json.load(f)

    blacklist: dict[str, str] = {}

    for song_id, entry in frame_index.items():
        n_frames = entry.get("n_frames", 0)
        cat = entry.get("category", "vanilla")
        diffs = entry.get("difficulties", [])

        # Tier 1: Short songs (< 15 seconds = ~652 frames at 23ms/frame)
        if n_frames < 652:
            blacklist[song_id] = f"short_song ({n_frames} frames)"
            continue

        # Tier 2: Modded maps that add noise
        if cat in ("noodle", "mapping_extensions"):
            blacklist[song_id] = f"modded ({cat})"
            continue

        # Tier 3: No Expert or ExpertPlus difficulty
        if "Expert" not in diffs and "ExpertPlus" not in diffs:
            blacklist[song_id] = "no_expert_difficulty"
            continue

    logger.info("Blacklist: %d maps out of %d total", len(blacklist), len(frame_index))

    # Summarize reasons
    reasons: dict[str, int] = {}
    for reason in blacklist.values():
        key = reason.split(" (")[0]
        reasons[key] = reasons.get(key, 0) + 1
    for reason, count in sorted(reasons.items(), key=lambda x: -x[1]):
        logger.info("  %s: %d", reason, count)

    return blacklist


def main() -> None:
    parser = argparse.ArgumentParser(description="Filter outlier maps from training data")
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--dry-run", action="store_true", help="Print stats without writing")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    blacklist = build_blacklist(args.data_dir)

    if args.dry_run:
        print(f"\nDry run: would blacklist {len(blacklist)} maps")
        return

    output_path = args.data_dir / "blacklist.json"
    with open(output_path, "w") as f:
        json.dump(blacklist, f, indent=2)
    print(f"\nWrote {len(blacklist)} entries to {output_path}")


if __name__ == "__main__":
    main()
