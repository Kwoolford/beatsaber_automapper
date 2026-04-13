"""Filter outlier maps from the training dataset.

Scans .pt files to compute quality scores based on map content, then writes
a blacklist file that datasets can use to skip low-quality maps.

Quality metrics:
    - Parity violation rate: how often swing direction alternation is broken
    - Color balance: ratio between red and blue notes (ideal ~0.5)
    - Grid entropy: variety of grid positions used (low = repetitive)
    - NPS variance: consistency of note density across the song
    - Empty onset ratio: fraction of onsets with zero notes

Usage:
    python scripts/filter_outliers.py --data-dir data/processed
    python scripts/filter_outliers.py --data-dir data/processed --dry-run
    python scripts/filter_outliers.py --data-dir data/processed --threshold 0.15
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from collections import Counter
from pathlib import Path

import torch
from tqdm import tqdm

from beatsaber_automapper.data.tokenizer import (
    COL_OFFSET,
    COLOR_OFFSET,
    DIR_OFFSET,
    EOS,
    NOTE,
    ROW_OFFSET,
    SEP,
)

logger = logging.getLogger(__name__)

# Direction pairs that form valid alternations (opposite swing directions)
# 0=up, 1=down, 2=left, 3=right, 4=up-left, 5=up-right, 6=down-left, 7=down-right
_OPPOSITE_DIRS = {
    0: {1, 6, 7},      # up -> down variants
    1: {0, 4, 5},      # down -> up variants
    2: {3, 5, 7},      # left -> right variants
    3: {2, 4, 6},      # right -> left variants
    4: {1, 3, 6, 7},   # up-left -> down or right variants
    5: {1, 2, 6, 7},   # up-right -> down or left variants
    6: {0, 3, 4, 5},   # down-left -> up or right variants
    7: {0, 2, 4, 5},   # down-right -> up or left variants
}


def _extract_notes_from_tokens(token_seq: list[int]) -> list[tuple[int, int, int, int]]:
    """Extract (color, col, row, direction) tuples from a token sequence."""
    notes = []
    i = 0
    while i < len(token_seq):
        if token_seq[i] == NOTE and i + 5 < len(token_seq):
            color = token_seq[i + 1] - COLOR_OFFSET
            col = token_seq[i + 2] - COL_OFFSET
            row = token_seq[i + 3] - ROW_OFFSET
            direction = token_seq[i + 4] - DIR_OFFSET
            if 0 <= color <= 1 and 0 <= col <= 3 and 0 <= row <= 2 and 0 <= direction <= 8:
                notes.append((color, col, row, direction))
            i += 6
        elif token_seq[i] in (EOS, SEP):
            i += 1
        else:
            # Skip other event types by advancing past them
            i += 1
    return notes


def compute_quality_scores(
    pt_path: Path,
) -> dict[str, dict[str, float]]:
    """Compute quality scores for each difficulty in a .pt file.

    Returns:
        Dict mapping difficulty name -> {metric_name: score}.
    """
    data = torch.load(pt_path, weights_only=False)
    difficulties = data.get("difficulties", {})
    n_frames = data.get("mel_spectrogram", torch.zeros(1, 1)).shape[1]

    # Approximate song duration in seconds
    hop_length = 512
    sample_rate = 44100
    duration = n_frames * hop_length / sample_rate

    scores: dict[str, dict[str, float]] = {}

    for diff_name, diff_data in difficulties.items():
        token_sequences = diff_data.get("token_sequences", [])
        n_onsets = len(token_sequences)

        if n_onsets == 0:
            scores[diff_name] = {"empty": 1.0}
            continue

        # Collect all notes across all onsets
        all_notes: list[tuple[int, int, int, int]] = []
        empty_onsets = 0
        per_onset_counts: list[int] = []

        for seq in token_sequences:
            seq = list(seq)
            notes = _extract_notes_from_tokens(seq)
            per_onset_counts.append(len(notes))
            if len(notes) == 0:
                empty_onsets += 1
            all_notes.extend(notes)

        total_notes = len(all_notes)
        if total_notes < 5:
            scores[diff_name] = {"too_few_notes": 1.0}
            continue

        # 1. Parity violation rate (per color)
        parity_violations = 0
        parity_checks = 0
        for color in (0, 1):
            color_notes = [n for n in all_notes if n[0] == color and n[3] != 8]  # skip dot notes
            for j in range(1, len(color_notes)):
                prev_dir = color_notes[j - 1][3]
                curr_dir = color_notes[j][3]
                if prev_dir in _OPPOSITE_DIRS:
                    parity_checks += 1
                    if curr_dir not in _OPPOSITE_DIRS.get(prev_dir, set()):
                        parity_violations += 1
        parity_rate = parity_violations / max(parity_checks, 1)

        # 2. Color balance (0.0 = perfectly balanced, 1.0 = all one color)
        red_count = sum(1 for n in all_notes if n[0] == 0)
        color_ratio = red_count / total_notes
        color_imbalance = abs(color_ratio - 0.5) * 2  # 0-1 scale

        # 3. Grid entropy (how many unique positions are used)
        # Max possible positions: 4 cols * 3 rows = 12
        positions = [(n[1], n[2]) for n in all_notes]
        position_counts = Counter(positions)
        n_unique_positions = len(position_counts)
        # Normalized entropy
        if n_unique_positions <= 1:
            grid_entropy = 0.0
        else:
            total_pos = sum(position_counts.values())
            entropy = -sum(
                (c / total_pos) * math.log2(c / total_pos) for c in position_counts.values()
            )
            max_entropy = math.log2(min(n_unique_positions, 12))
            grid_entropy = entropy / max_entropy if max_entropy > 0 else 0.0

        # 4. NPS variance (coefficient of variation of notes-per-onset)
        if len(per_onset_counts) > 1:
            mean_nps = sum(per_onset_counts) / len(per_onset_counts)
            variance = sum((c - mean_nps) ** 2 for c in per_onset_counts) / len(per_onset_counts)
            nps_cv = math.sqrt(variance) / max(mean_nps, 0.01)
        else:
            nps_cv = 0.0

        # 5. Empty onset ratio
        empty_ratio = empty_onsets / n_onsets

        # 6. Direction entropy (variety of swing directions)
        directions = [n[3] for n in all_notes if n[3] != 8]  # exclude dot
        if directions:
            dir_counts = Counter(directions)
            total_dirs = sum(dir_counts.values())
            dir_entropy = -sum(
                (c / total_dirs) * math.log2(c / total_dirs) for c in dir_counts.values()
            )
            max_dir_entropy = math.log2(8)  # 8 non-dot directions
            dir_entropy_norm = dir_entropy / max_dir_entropy
        else:
            dir_entropy_norm = 0.0

        # 7. Dot note ratio (too many dot notes = lazy mapping)
        dot_notes = sum(1 for n in all_notes if n[3] == 8)
        dot_ratio = dot_notes / total_notes

        # 8. Notes per second (sanity check)
        nps = total_notes / max(duration, 1.0)

        scores[diff_name] = {
            "parity_rate": parity_rate,
            "color_imbalance": color_imbalance,
            "grid_entropy": grid_entropy,
            "nps_cv": nps_cv,
            "empty_ratio": empty_ratio,
            "dir_entropy": dir_entropy_norm,
            "dot_ratio": dot_ratio,
            "nps": nps,
            "total_notes": float(total_notes),
            "n_onsets": float(n_onsets),
        }

    return scores


def compute_composite_score(metrics: dict[str, float]) -> float:
    """Compute a 0-1 composite quality score (higher = better).

    Weights penalize bad parity, low grid diversity, and extreme values.
    """
    if "empty" in metrics or "too_few_notes" in metrics:
        return 0.0

    score = 1.0

    # Heavy penalty for parity violations (>40% is very bad)
    score -= 0.30 * min(metrics["parity_rate"] / 0.5, 1.0)

    # Penalty for color imbalance (>0.8 means almost all one color)
    score -= 0.10 * min(metrics["color_imbalance"] / 0.8, 1.0)

    # Reward grid diversity (low entropy = repetitive placement)
    score -= 0.15 * (1.0 - metrics["grid_entropy"])

    # Penalty for low direction diversity
    score -= 0.15 * (1.0 - metrics["dir_entropy"])

    # Penalty for high dot note ratio (>30% dots)
    score -= 0.10 * min(metrics["dot_ratio"] / 0.3, 1.0)

    # Penalty for many empty onsets
    score -= 0.10 * min(metrics["empty_ratio"] / 0.2, 1.0)

    # Penalty for extreme NPS (>20 or <0.5)
    nps = metrics["nps"]
    if nps > 20:
        score -= 0.10
    elif nps < 0.5:
        score -= 0.10

    return max(0.0, min(1.0, score))


def build_blacklist(
    data_dir: Path,
    threshold: float = 0.15,
    dry_run: bool = False,
) -> dict[str, str]:
    """Scan dataset and identify low-quality maps.

    Args:
        data_dir: Path to processed data directory.
        threshold: Bottom N fraction to blacklist (0.15 = bottom 15%).
        dry_run: If True, don't write anything.

    Returns:
        Dict mapping song_id -> reason for blacklisting.
    """
    # Load frame_index for metadata checks
    frame_index_path = data_dir / "frame_index.json"
    frame_index: dict = {}
    if frame_index_path.exists():
        with open(frame_index_path) as f:
            frame_index = json.load(f)

    blacklist: dict[str, str] = {}
    song_scores: dict[str, float] = {}

    pt_files = sorted(data_dir.glob("*.pt"))
    if not pt_files:
        logger.error("No .pt files found in %s", data_dir)
        return {}

    logger.info("Scanning %d .pt files for quality...", len(pt_files))

    for pt_path in tqdm(pt_files, desc="Quality scoring"):
        song_id = pt_path.stem

        # Metadata-based filtering (fast, no content loading needed)
        if song_id in frame_index:
            entry = frame_index[song_id]
            n_frames = entry.get("n_frames", 0)
            cat = entry.get("category", "vanilla")
            diffs = entry.get("difficulties", [])

            # Short songs (< 15 seconds)
            if n_frames < 652:
                blacklist[song_id] = f"short_song ({n_frames} frames)"
                continue

            # Modded maps
            if cat in ("noodle", "mapping_extensions"):
                blacklist[song_id] = f"modded ({cat})"
                continue

            # No Expert/ExpertPlus
            if "Expert" not in diffs and "ExpertPlus" not in diffs:
                blacklist[song_id] = "no_expert_difficulty"
                continue

        # Content-based quality scoring
        try:
            diff_scores = compute_quality_scores(pt_path)
        except Exception as e:
            logger.warning("Failed to score %s: %s", song_id, e)
            blacklist[song_id] = f"scoring_error ({e})"
            continue

        # Use the best difficulty's score (Expert or ExpertPlus preferred)
        best_score = 0.0
        for diff_name, metrics in diff_scores.items():
            score = compute_composite_score(metrics)
            if score > best_score:
                best_score = score

        song_scores[song_id] = best_score

    # Determine quality threshold from percentile
    if song_scores:
        sorted_scores = sorted(song_scores.values())
        cutoff_idx = int(len(sorted_scores) * threshold)
        cutoff_score = sorted_scores[min(cutoff_idx, len(sorted_scores) - 1)]

        logger.info(
            "Quality score distribution: min=%.3f, p10=%.3f, p25=%.3f, median=%.3f, "
            "p75=%.3f, p90=%.3f, max=%.3f",
            sorted_scores[0],
            sorted_scores[int(len(sorted_scores) * 0.10)],
            sorted_scores[int(len(sorted_scores) * 0.25)],
            sorted_scores[len(sorted_scores) // 2],
            sorted_scores[int(len(sorted_scores) * 0.75)],
            sorted_scores[int(len(sorted_scores) * 0.90)],
            sorted_scores[-1],
        )
        logger.info("Cutoff at %.1f%% = %.3f", threshold * 100, cutoff_score)

        # Blacklist maps below the cutoff
        for song_id, score in song_scores.items():
            if score <= cutoff_score and song_id not in blacklist:
                blacklist[song_id] = f"low_quality (score={score:.3f})"

    logger.info("Total blacklisted: %d maps out of %d", len(blacklist), len(pt_files))

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
    parser.add_argument(
        "--threshold", type=float, default=0.15,
        help="Bottom fraction to blacklist (default: 0.15 = bottom 15%%)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    blacklist = build_blacklist(args.data_dir, threshold=args.threshold, dry_run=args.dry_run)

    if args.dry_run:
        print(f"\nDry run: would blacklist {len(blacklist)} maps")
        return

    output_path = args.data_dir / "blacklist.json"
    with open(output_path, "w") as f:
        json.dump(blacklist, f, indent=2)
    print(f"\nWrote {len(blacklist)} entries to {output_path}")


if __name__ == "__main__":
    main()
