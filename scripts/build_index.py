"""Build frame_index.json for fast dataset initialization.

Scans all preprocessed .pt files once and records per-song metadata:
  n_frames, difficulty names, n_onsets/n_lights per diff, category, genre.

This file is read by OnsetDataset, SequenceDataset, and LightingDataset during
__init__ to avoid loading every .pt file just to count frames/onsets.

Usage:
    python scripts/build_index.py --data-dir data/processed
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import torch
from tqdm import tqdm

logger = logging.getLogger(__name__)


def build_frame_index(data_dir: Path) -> dict:
    """Scan all .pt files and build a metadata index.

    Args:
        data_dir: Directory containing preprocessed .pt files.

    Returns:
        Dict mapping song_id -> metadata dict.
    """
    index: dict = {}
    pt_files = sorted(data_dir.glob("*.pt"))
    logger.info("Scanning %d .pt files...", len(pt_files))

    for pt_path in tqdm(pt_files, desc="Building index", unit="file"):
        try:
            data = torch.load(pt_path, weights_only=False)
        except Exception as e:
            logger.warning("Skipping %s: %s", pt_path.name, e)
            continue

        mel = data.get("mel_spectrogram")
        if mel is None:
            continue

        mod = data.get("mod_requirements", {})
        diffs: dict[str, dict] = {}
        for diff_name, diff_data in data.get("difficulties", {}).items():
            diffs[diff_name] = {
                "n_onsets": len(diff_data.get("token_sequences", [])),
                "n_lights": len(diff_data.get("light_token_sequences", [])),
            }

        index[pt_path.stem] = {
            "n_frames": int(mel.shape[1]),
            "difficulties": diffs,
            "category": mod.get("category", "unknown"),
            "genre": mod.get("genre", "unknown"),
        }

    return index


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Build frame_index.json for fast dataset init")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/processed"),
        help="Directory containing preprocessed .pt files",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

    index = build_frame_index(args.data_dir)
    out_path = args.data_dir / "frame_index.json"
    with open(out_path, "w") as f:
        json.dump(index, f)

    logger.info("Wrote %d entries to %s", len(index), out_path)


if __name__ == "__main__":
    main()
