#!/usr/bin/env python3
"""Density-correlation eval — the TASK-2 inference DoD metric.

Question it answers: does a generated map's *note density* track the song's
actual musical density, with the Stage-1 section gate OFF? The whole point of
feeding per-instrument layering features into Stage 1 is that the model should
learn to densify drops and thin out breakdowns on its own — instead of the flat
~8 NPS metronome the section-gate-fixed maps still produce.

Unlike ``eval_alignment.py``'s per-section breakdown (whose section boundaries
come from the energy detector we're trying to retire — circular), this bins both
the generated notes and the reference onsets into FIXED, uniform windows and
reports rank correlation. DoD: Spearman >= 0.41 (the structure-PoC bar from
`scripts/v8_poc_structure.py`, where per-instrument event density hit r=0.41
against human note density).

Reuses the audio/onset/map plumbing from ``eval_alignment.py``.
"""
from __future__ import annotations

import argparse
import json
import logging
import pathlib

import numpy as np

from eval_alignment import (  # same scripts/ dir
    _separate_stems,
    _detect_onsets_librosa,
    _load_generated_beatmap,
    _beat_to_seconds,
)

log = logging.getLogger("eval_density_corr")


def _bin_counts(times: np.ndarray, duration: float, win: float) -> np.ndarray:
    """Count events per uniform ``win``-second window over ``[0, duration)``."""
    n_bins = max(1, int(np.ceil(duration / win)))
    edges = np.arange(n_bins + 1) * win
    counts, _ = np.histogram(times, bins=edges)
    return counts.astype(np.float64)


def _rank(a: np.ndarray) -> np.ndarray:
    """Average-rank transform (ties → mean rank), for Spearman via Pearson."""
    order = a.argsort(kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(len(a), dtype=np.float64)
    # average tied ranks
    _, inv, counts = np.unique(a, return_inverse=True, return_counts=True)
    csum = np.cumsum(counts)
    start = csum - counts
    avg = (start + csum - 1) / 2.0
    return avg[inv]


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    if x.std() == 0 or y.std() == 0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    return _pearson(_rank(x), _rank(y))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--audio", type=pathlib.Path, required=True)
    p.add_argument("--map", type=pathlib.Path, required=True, help="generated .zip or dir")
    p.add_argument("--difficulty", default="Expert")
    p.add_argument("--window-sec", type=float, default=2.0)
    p.add_argument("--sr", type=int, default=44100)
    p.add_argument("--json", type=pathlib.Path, default=None)
    p.add_argument("--label", default=None, help="tag echoed into the JSON/stdout")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(name)s %(levelname)s: %(message)s")

    notes, bpm = _load_generated_beatmap(args.map, args.difficulty)
    gen_times = np.array(sorted(_beat_to_seconds(b, bpm) for b, _x, _c in notes),
                         dtype=np.float64)

    stems = _separate_stems(args.audio, args.sr)
    drum_on = _detect_onsets_librosa(stems.get("drums", np.zeros(1)), args.sr)
    other_on = _detect_onsets_librosa(stems.get("other", np.zeros(1)), args.sr)
    ref_times = np.union1d(drum_on, other_on)

    duration = float(max(gen_times.max() if len(gen_times) else 0.0,
                         ref_times.max() if len(ref_times) else 0.0))
    gen_d = _bin_counts(gen_times, duration, args.window_sec)
    ref_d = _bin_counts(ref_times, duration, args.window_sec)
    n = min(len(gen_d), len(ref_d))
    gen_d, ref_d = gen_d[:n], ref_d[:n]

    spear = _spearman(gen_d, ref_d)
    pear = _pearson(gen_d, ref_d)

    result = {
        "label": args.label,
        "audio": str(args.audio),
        "map": str(args.map),
        "difficulty": args.difficulty,
        "window_sec": args.window_sec,
        "n_windows": int(n),
        "n_generated_notes": int(len(gen_times)),
        "n_reference_onsets": int(len(ref_times)),
        "gen_density_mean": float(gen_d.mean()),
        "gen_density_cv": float(gen_d.std() / gen_d.mean()) if gen_d.mean() else 0.0,
        "ref_density_cv": float(ref_d.std() / ref_d.mean()) if ref_d.mean() else 0.0,
        "spearman": spear,
        "pearson": pear,
        "dod_pass": bool(spear >= 0.41),
    }

    print(f"\n=== density-corr  {args.label or args.map.name}  "
          f"(win={args.window_sec}s, {n} windows) ===")
    print(f"  gen notes={len(gen_times)}  ref onsets={len(ref_times)}")
    print(f"  gen density CV={result['gen_density_cv']:.3f}  "
          f"(flat metronome → ~0; tracking structure → higher)")
    print(f"  Spearman={spear:.4f}   Pearson={pear:.4f}   "
          f"DoD(>=0.41)={'PASS' if result['dod_pass'] else 'FAIL'}")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(result, indent=2))
        log.info("wrote %s", args.json)


if __name__ == "__main__":
    main()
