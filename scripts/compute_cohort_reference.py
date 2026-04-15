"""Compute style-reference statistics for each cohort.

Scans a cohort's preprocessed .pt files, decodes the per-onset token sequences
into structured note events, and aggregates:
    - mean NPS (notes per second)
    - direction histogram (normalized over 0..8)
    - parity violation rate baseline (ground-truth maps *do* have small rates)
    - color balance (fraction red vs. blue)
    - n_maps, n_notes_total

Writes ``data/cohorts/{slug}/reference.json``. This is the style target that
``research/metrics.py:style_closeness()`` compares generated output against.

Usage:
    python scripts/compute_cohort_reference.py                    # all cohorts
    python scripts/compute_cohort_reference.py --cohort joetastic
    python scripts/compute_cohort_reference.py --bucket anime_jpop_flow
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from pathlib import Path

import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from beatsaber_automapper.data.tokenizer import tokens_to_structured  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("compute_cohort_reference")

FOREHAND = {1, 6, 7}
BACKHAND = {0, 4, 5}
NEUTRAL = {2, 3, 8}


def _parity_class(d: int) -> str:
    if d in FOREHAND:
        return "fore"
    if d in BACKHAND:
        return "back"
    return "neutral"


def compute_reference(data_dir: Path, difficulties: set[str] | None = None) -> dict:
    """Aggregate cohort statistics by scanning all .pt files in data_dir."""
    pt_files = sorted(data_dir.glob("*.pt"))
    if not pt_files:
        logger.warning("no .pt files in %s", data_dir)
        return {}

    n_maps = 0
    total_notes = 0
    total_song_sec = 0.0
    dir_counts: Counter[int] = Counter()
    parity_violations = 0
    parity_eligible = 0  # denominator — note pairs where both are non-neutral
    red = 0
    blue = 0

    for pt_path in tqdm(pt_files, desc=f"scan {data_dir.name}", unit="map"):
        try:
            data = torch.load(pt_path, weights_only=False)
        except Exception as e:
            logger.debug("skip %s: %s", pt_path.name, e)
            continue

        mel = data.get("mel_spectrogram")
        diffs = data.get("difficulties", {})
        if mel is None or not diffs:
            continue

        # Approximate song duration from mel shape (hop=512, sr=44100 → ~86 fps)
        song_sec = float(mel.shape[1]) * 512.0 / 44100.0

        # Pick one difficulty per map — prefer Expert, fall back to ExpertPlus
        chosen = None
        for name in ("Expert", "ExpertPlus", "Hard", "Normal", "Easy"):
            if difficulties is not None and name not in difficulties:
                continue
            if name in diffs and diffs[name].get("token_sequences"):
                chosen = diffs[name]
                break
        if chosen is None:
            continue

        n_maps += 1
        total_song_sec += song_sec

        # Per-color running last-direction for parity
        last_cls_per_color: dict[int, str] = {}
        for token_seq in chosen["token_sequences"]:
            if isinstance(token_seq, torch.Tensor):
                token_seq = token_seq.tolist()
            structured = tokens_to_structured(token_seq)
            for slot in structured.get("slots", []):
                if slot.get("event_type") != 0:
                    continue  # only standard notes contribute to style stats
                c = slot["color"]
                if c == 2:
                    continue  # none-color (bomb placeholder)
                d = slot["direction"]
                dir_counts[d] += 1
                total_notes += 1
                if c == 0:
                    red += 1
                elif c == 1:
                    blue += 1
                cls = _parity_class(d)
                prev = last_cls_per_color.get(c)
                if prev is not None and prev != "neutral" and cls != "neutral":
                    parity_eligible += 1
                    if cls == prev:
                        parity_violations += 1
                if cls != "neutral":
                    last_cls_per_color[c] = cls

    total_dir = sum(dir_counts.values()) or 1
    dir_hist = {d: dir_counts.get(d, 0) / total_dir for d in range(9)}
    mean_nps = total_notes / max(total_song_sec, 1.0)
    parity_baseline = parity_violations / max(parity_eligible, 1)
    color_balance = red / max(red + blue, 1)

    return {
        "n_maps": n_maps,
        "n_notes_total": total_notes,
        "total_song_sec": total_song_sec,
        "mean_nps": mean_nps,
        "direction_hist": dir_hist,
        "parity_rate_baseline": parity_baseline,
        "color_balance": color_balance,
        "red_count": red,
        "blue_count": blue,
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--mappers-json",
        type=Path,
        default=Path("data/reference/mappers.json"),
    )
    p.add_argument("--cohorts-root", type=Path, default=Path("data/cohorts"))
    p.add_argument("--cohort", help="Only this cohort slug")
    p.add_argument("--bucket", help="Only this bucket id")
    p.add_argument(
        "--difficulties",
        nargs="*",
        default=None,
        help="Restrict to these difficulties (default: all)",
    )
    args = p.parse_args()

    diffs = set(args.difficulties) if args.difficulties else None

    targets: list[tuple[str, Path]] = []  # (label, processed_dir)

    if args.cohort:
        targets.append((args.cohort, args.cohorts_root / args.cohort / "processed"))
    elif args.bucket:
        targets.append((f"bucket:{args.bucket}", args.cohorts_root / "_buckets" / args.bucket))
    else:
        # All cohorts in mappers.json
        import re

        def slugify(name: str) -> str:
            return re.sub(r"[^a-zA-Z0-9_-]+", "_", name).strip("_").lower()

        data = json.loads(args.mappers_json.read_text(encoding="utf-8"))
        for m in data["mappers"]:
            slug = slugify(m["display_name"])
            targets.append((slug, args.cohorts_root / slug / "processed"))

    for label, data_dir in targets:
        if not data_dir.exists():
            logger.warning("skip %s: %s does not exist", label, data_dir)
            continue
        logger.info("computing reference for %s", label)
        ref = compute_reference(data_dir, difficulties=diffs)
        if not ref:
            continue
        if args.bucket is None:
            out = data_dir.parent / "reference.json"
        else:
            out = data_dir / "reference.json"
        out.write_text(json.dumps(ref, indent=2), encoding="utf-8")
        logger.info(
            "[%s] n_maps=%d n_notes=%d mean_nps=%.2f parity_baseline=%.3f → %s",
            label,
            ref["n_maps"],
            ref["n_notes_total"],
            ref["mean_nps"],
            ref["parity_rate_baseline"],
            out,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
