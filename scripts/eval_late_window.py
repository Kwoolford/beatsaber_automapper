#!/usr/bin/env python
"""Late-song / final-chorus collapse diagnostic.

The last untouched original complaint (after drop-@-13s fixed via section_gate and
flat-density fixed via density-select): generated maps thin out / die in the final
section while the song (final chorus) is still energetic — the "late-song collapse".

There was no metric for it. This one is deliberately simple and human-relative:
per song we compare the GENERATED note-time distribution against the HUMAN reference
onset distribution over the LATE tail of the song. A model that collapses late puts a
smaller share of its notes in the tail than the human map does.

Metrics (per song, then mean over songs):
  ref_late_frac  = fraction of reference onsets in the final `--tail` of the song
  gen_late_frac  = fraction of generated notes  in the final `--tail`
  late_gap       = ref_late_frac - gen_late_frac   (POSITIVE => gen under-produces
                   the tail relative to human = collapse; ~0 => tracks human)
  late_corr      = Spearman(gen_density, ref_density) computed ONLY over tail windows
                   (is late density still tracking the song, or flat/dead?)

DoD proposal for the fix: mean late_gap <= 0.03 (gen tail-share within 3 pts of human)
AND late_corr >= 0.30, while whole-song density_corr and parity/monotony hold.

Usage:
  python scripts/eval_late_window.py --arm prod            # score cached maps for an arm
  python scripts/eval_late_window.py --map path/to.zip --ref data/eval_songset/1f333.ref.npz
"""
from __future__ import annotations

import argparse
import glob
import os
import pathlib
import sys

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
from eval_alignment import _load_generated_beatmap, _beat_to_seconds  # noqa: E402
from eval_density_corr import _bin_counts, _spearman  # noqa: E402

CACHE = REPO / "outputs" / "eval_sweep_cache"
SONGSET = REPO / "data" / "eval_songset"
WIN_SEC = 2.0


def _gen_times(zip_path: str) -> np.ndarray:
    notes, bpm = _load_generated_beatmap(pathlib.Path(zip_path), "Expert")
    return np.array(sorted(_beat_to_seconds(b, bpm) for b, _x, _c in notes), dtype=np.float64)


def _late_stats(gen_times: np.ndarray, ref_times: np.ndarray, duration: float,
                tail: float) -> dict:
    dur = float(max(duration, gen_times.max() if len(gen_times) else 0.0,
                    ref_times.max() if len(ref_times) else 0.0))
    cut = (1.0 - tail) * dur
    ref_late = float((ref_times >= cut).mean()) if len(ref_times) else 0.0
    gen_late = float((gen_times >= cut).mean()) if len(gen_times) else 0.0
    # tail-only density correlation
    gen_d = _bin_counts(gen_times, dur, WIN_SEC)
    ref_d = _bin_counts(ref_times, dur, WIN_SEC)
    n = min(len(gen_d), len(ref_d))
    gen_d, ref_d = gen_d[:n], ref_d[:n]
    tail_start = int((1.0 - tail) * n)
    late_corr = _spearman(gen_d[tail_start:], ref_d[tail_start:]) if n - tail_start >= 3 else float("nan")
    return {
        "ref_late_frac": ref_late,
        "gen_late_frac": gen_late,
        "late_gap": ref_late - gen_late,
        "late_corr": late_corr,
        "dur": dur,
    }


def _ref_for(stem: str) -> pathlib.Path | None:
    f = SONGSET / f"{stem}.ref.npz"
    return f if f.exists() else None


def score_arm(arm: str, tail: float) -> None:
    maps = sorted(glob.glob(str(CACHE / f"{arm}__*.zip")))
    if not maps:
        print(f"no cached maps for arm '{arm}' under {CACHE}")
        return
    rows = []
    print(f"=== late-window collapse: arm '{arm}', tail=final {tail:.0%} ===")
    print(f"{'song':26s} {'ref_late':>8s} {'gen_late':>8s} {'late_gap':>8s} {'late_corr':>9s}")
    for m in maps:
        stem = os.path.basename(m)[len(arm) + 2:-4]  # strip 'arm__' ... '.zip'
        ref = _ref_for(stem)
        if ref is None:
            continue
        d = np.load(ref)
        s = _late_stats(_gen_times(m), d["ref_times"], float(d["duration"]), tail)
        rows.append(s)
        lc = f"{s['late_corr']:+.2f}" if s["late_corr"] == s["late_corr"] else "  n/a"
        print(f"{stem:26s} {s['ref_late_frac']:8.3f} {s['gen_late_frac']:8.3f} "
              f"{s['late_gap']:+8.3f} {lc:>9s}")
    if rows:
        mg = float(np.mean([r["late_gap"] for r in rows]))
        corrs = [r["late_corr"] for r in rows if r["late_corr"] == r["late_corr"]]
        mc = float(np.mean(corrs)) if corrs else float("nan")
        print("-" * 62)
        print(f"{'MEAN':26s} {'':8s} {'':8s} {mg:+8.3f} {mc:+9.2f}")
        print(f"\nverdict: mean late_gap {mg:+.3f} "
              f"({'COLLAPSE (gen under-produces tail)' if mg > 0.03 else 'tracks human'}); "
              f"mean late_corr {mc:+.2f} "
              f"({'tail density tracks song' if mc >= 0.30 else 'tail density weak/flat'})")


def main() -> None:
    ap = argparse.ArgumentParser(description="Late-song collapse diagnostic.")
    ap.add_argument("--arm", help="score all cached maps for this eval_sweep arm")
    ap.add_argument("--map", help="single generated map zip")
    ap.add_argument("--ref", help="ref .npz for --map")
    ap.add_argument("--tail", type=float, default=0.20, help="tail fraction of the song (default 0.20)")
    a = ap.parse_args()
    if a.arm:
        score_arm(a.arm, a.tail)
    elif a.map and a.ref:
        d = np.load(a.ref)
        s = _late_stats(_gen_times(a.map), d["ref_times"], float(d["duration"]), a.tail)
        for k, v in s.items():
            print(f"{k:14s} {v:+.4f}")
    else:
        ap.error("give --arm, or --map with --ref")


if __name__ == "__main__":
    main()
