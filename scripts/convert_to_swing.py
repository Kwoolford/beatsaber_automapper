"""Convert V5-format .pt files to V6 by adding swing_tokens per difficulty.

Reads each .pt file, reconstructs the beatmap from the stored V5 chord tokens
using BeatmapTokenizer, then re-encodes with SwingEventTokenizer and writes
the result back into the same .pt file (adding a ``swing_tokens`` key per
difficulty without removing any existing V5 data).

Usage:
    # Convert all processed files (in-place):
    python scripts/convert_to_swing.py --data-dir data/processed

    # Convert a specific cohort directory:
    python scripts/convert_to_swing.py --data-dir data/cohorts/joetastic/processed

    # Dry run (validate conversion without writing):
    python scripts/convert_to_swing.py --data-dir data/processed --dry-run

    # Limit to N files (smoke test):
    python scripts/convert_to_swing.py --data-dir data/processed --limit 20
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import torch

# Allow running as a script without installing the package
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from beatsaber_automapper.data.swing_tokenizer import SwingEventTokenizer
from beatsaber_automapper.data.tokenizer import BeatmapTokenizer

logger = logging.getLogger(__name__)

# ~86 audio frames per second (sr=44100, hop=512)
_FRAMES_PER_SEC = 44100 / 512


def _reconstruct_and_encode(
    pt_data: dict,
    diff_name: str,
    swing_tok: SwingEventTokenizer,
    v5_tok: BeatmapTokenizer,
) -> list[int] | None:
    """Reconstruct a beatmap from V5 chord tokens and encode as V6 swing stream.

    Args:
        pt_data: Loaded .pt dict.
        diff_name: Difficulty name (e.g. "Expert").
        swing_tok: V6 SwingEventTokenizer instance.
        v5_tok: V5 BeatmapTokenizer instance.

    Returns:
        Flat swing token list, or None if conversion failed.
    """
    diff_data = pt_data.get("difficulties", {}).get(diff_name, {})
    onset_frames = diff_data.get("onset_frames")
    token_sequences = diff_data.get("token_sequences")
    bpm = float(pt_data.get("bpm", 120.0))

    if onset_frames is None or token_sequences is None:
        return None
    if len(onset_frames) == 0:
        return [swing_tok.bos_token, swing_tok.eos_token]

    frames_per_beat = _FRAMES_PER_SEC * 60.0 / bpm

    # Build per-beat token dict
    beat_tokens: dict[float, list[int]] = {}
    for i, (frame, seq) in enumerate(zip(onset_frames, token_sequences)):
        beat = float(int(frame.item())) / frames_per_beat
        beat_tokens[beat] = list(seq)

    beatmap = v5_tok.decode_beatmap(beat_tokens)
    return swing_tok.encode_beatmap(beatmap)


def convert_file(
    pt_path: Path,
    swing_tok: SwingEventTokenizer,
    v5_tok: BeatmapTokenizer,
    *,
    dry_run: bool = False,
    overwrite: bool = False,
) -> dict[str, int]:
    """Convert one .pt file in-place, adding swing_tokens per difficulty.

    Returns:
        Dict with keys: "converted", "skipped", "failed" (counts).
    """
    stats = {"converted": 0, "skipped": 0, "failed": 0}
    try:
        data = torch.load(pt_path, weights_only=False)
    except Exception as exc:
        logger.warning("Cannot load %s: %s", pt_path.name, exc)
        stats["failed"] += 1
        return stats

    changed = False
    for diff_name, diff_data in data.get("difficulties", {}).items():
        if not overwrite and "swing_tokens" in diff_data:
            stats["skipped"] += 1
            continue

        swing_tokens = _reconstruct_and_encode(data, diff_name, swing_tok, v5_tok)
        if swing_tokens is None:
            logger.warning("%s / %s: conversion returned None", pt_path.name, diff_name)
            stats["failed"] += 1
            continue

        diff_data["swing_tokens"] = swing_tokens
        changed = True
        stats["converted"] += 1

    if changed and not dry_run:
        torch.save(data, pt_path)

    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Add V6 swing_tokens to V5 .pt files.")
    parser.add_argument("--data-dir", type=Path, required=True,
                        help="Directory containing .pt files to convert.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate conversion without writing files.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Re-encode even if swing_tokens already present.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Stop after this many files (smoke test).")
    parser.add_argument("--workers", type=int, default=1,
                        help="Number of parallel conversion workers.")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    logging.basicConfig(level=args.log_level,
                        format="%(asctime)s %(levelname)s %(message)s")

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        logger.error("Data directory not found: %s", data_dir)
        sys.exit(1)

    pt_files = sorted(data_dir.glob("*.pt"))
    if args.limit:
        pt_files = pt_files[: args.limit]

    if not pt_files:
        logger.error("No .pt files found in %s", data_dir)
        sys.exit(1)

    logger.info("Converting %d files in %s (dry_run=%s)", len(pt_files), data_dir, args.dry_run)

    swing_tok = SwingEventTokenizer()
    v5_tok = BeatmapTokenizer()

    total = {"converted": 0, "skipped": 0, "failed": 0}

    if args.workers > 1:
        import functools
        from concurrent.futures import ProcessPoolExecutor, as_completed
        fn = functools.partial(
            convert_file,
            swing_tok=swing_tok,
            v5_tok=v5_tok,
            dry_run=args.dry_run,
            overwrite=args.overwrite,
        )
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futures = {ex.submit(fn, p): p for p in pt_files}
            for i, fut in enumerate(as_completed(futures), 1):
                p = futures[fut]
                try:
                    stats = fut.result()
                except Exception as exc:
                    logger.warning("Worker error on %s: %s", p.name, exc)
                    total["failed"] += 1
                    continue
                for k in total:
                    total[k] += stats.get(k, 0)
                if i % 100 == 0:
                    logger.info("[%d/%d] converted=%d skipped=%d failed=%d",
                                i, len(pt_files), total["converted"],
                                total["skipped"], total["failed"])
    else:
        for i, pt_path in enumerate(pt_files, 1):
            stats = convert_file(
                pt_path, swing_tok, v5_tok,
                dry_run=args.dry_run, overwrite=args.overwrite,
            )
            for k in total:
                total[k] += stats.get(k, 0)
            if i % 100 == 0 or i == len(pt_files):
                logger.info("[%d/%d] converted=%d skipped=%d failed=%d",
                            i, len(pt_files), total["converted"],
                            total["skipped"], total["failed"])

    logger.info(
        "Done. converted=%d skipped=%d failed=%d%s",
        total["converted"], total["skipped"], total["failed"],
        " (dry run — no files written)" if args.dry_run else "",
    )


if __name__ == "__main__":
    main()
