#!/usr/bin/env python
"""Recover the song's TEMPO AND PHASE — the thing that actually breaks our timing.

Axis A8 (2026-08-02) traced Kyle's "the notes are off beat" to the note grid: our
detected bpm is exact on **1 of 21** eval songs (median error 0.74%, four songs at
2/3 tempo), Stage-1 places every note on a 1/4-beat slot grid built from it, and a
0.74% error slides that grid through every phase as the song plays. Handing the
generator the true tempo on 1f767 moved onset precision 0.803 -> 0.899 and scatter
11.7 -> 8.5ms, which is BETTER than the human map's 8.7ms.

So the fix is not a model change. It is estimating the grid properly.

`detect_bpm` calls `librosa.beat.beat_track` and keeps only the tempo scalar:

    tempo, _ = librosa.beat.beat_track(...)   # the beat POSITIONS are discarded

The discarded half is the useful half. Two things follow from it:

  beat_lsq   Least-squares fit of time = period * beat_index + phase over the
             detected beat positions. The slope averages out the per-beat jitter
             the tempogram's single scalar cannot, and the intercept is the PHASE
             — which the current pipeline does not estimate at all: the grid is
             anchored at t=0 regardless of where the music's first downbeat is.
  comb       Refine that by maximising how tightly the DETECTED ONSETS cluster on
             the resulting 1/4-beat grid (circular concentration R of the onset
             phases). We already compute those onsets for A8, so this is free, and
             it optimises the exact quantity A8 measures.

Reported against the human map's declared bpm, which is ground truth: a human
mapper synced the map to the song by hand.

CPU-only. Usage:
  python scripts/fit_tempo_grid.py
  python scripts/fit_tempo_grid.py --songs 1f767 --verbose
"""
from __future__ import annotations

import argparse
import json
import pathlib
import statistics
import sys

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

SONGSET = REPO / "data" / "eval_songset"
ONSETS = REPO / "outputs" / "onset_cache"
TRUE_BPM_PATH = REPO / "outputs" / "true_bpm_eval_songset.json"
AUDIO_EXTS = (".ogg", ".mp3", ".wav", ".egg")


def _audio_for(sid: str) -> pathlib.Path | None:
    for ext in AUDIO_EXTS:
        p = SONGSET / f"{sid}{ext}"
        if p.exists():
            return p
    return None


def librosa_baseline(y, sr) -> tuple[float, np.ndarray]:
    """What the pipeline does today — plus the beat positions it throws away."""
    import librosa

    tempo, beats = librosa.beat.beat_track(y=y, sr=sr, units="time")
    return float(np.atleast_1d(tempo)[0]), np.asarray(beats, dtype=np.float64)


def beat_lsq(beat_times: np.ndarray) -> tuple[float, float]:
    """Fit time = period * index + phase over detected beats -> (bpm, phase_s).

    Robust to the occasional dropped/extra beat by rounding each beat onto the
    running index implied by the median inter-beat interval before fitting.
    """
    if len(beat_times) < 8:
        return float("nan"), 0.0
    ibi = float(np.median(np.diff(beat_times)))
    if ibi <= 0:
        return float("nan"), 0.0
    idx = np.round((beat_times - beat_times[0]) / ibi)
    # drop duplicate indices caused by a spurious beat
    _, keep = np.unique(idx, return_index=True)
    idx, t = idx[keep], beat_times[keep]
    period, phase = np.polyfit(idx, t, 1)
    if period <= 0:
        return float("nan"), 0.0
    return 60.0 / float(period), float(phase)


def comb_refine(onsets: np.ndarray, bpm0: float, phase0: float, subdiv: int = 4,
                span: float = 0.03, steps: int = 601) -> tuple[float, float, float]:
    """Maximise onset concentration on the 1/4-beat grid near `bpm0`.

    Score is the circular resultant length R of the onset phases modulo the slot
    period: R = 1 means every onset sits exactly on a slot, R ~ 0 means the grid
    has no relationship to the music. Returns (bpm, phase_s, R).

    `span` is fractional (0.03 = +-3%), wide enough to cover the 0.74% median error
    without reaching the 2/3-tempo alternatives, which are a different failure and
    need the octave logic, not a local search.
    """
    if len(onsets) < 20 or not np.isfinite(bpm0) or bpm0 <= 0:
        return bpm0, phase0, float("nan")
    best = (bpm0, phase0, -1.0)
    for bpm in np.linspace(bpm0 * (1 - span), bpm0 * (1 + span), steps):
        slot = 60.0 / bpm / subdiv
        ang = 2.0 * np.pi * (onsets / slot)
        z = np.exp(1j * ang).mean()
        r = float(abs(z))
        if r > best[2]:
            # circular mean phase, expressed as a time offset into the slot
            phase = float((np.angle(z) / (2.0 * np.pi)) * slot)
            best = (float(bpm), phase, r)
    return best


# Metrical levels to try before the local search. A 2/3 error (four eval songs) is
# NOT reachable by a +-3% refinement: a wrong metrical level is a different mistake
# from a drifting one and has to be enumerated.
#
# ONLY RATIOS >= 1. Every one of librosa's errors on this set is an UNDER-estimate
# (-33% or -50%, never over), and the two directions are not symmetric for us: a
# grid finer than the music can express everything the music does, while a coarser
# one cannot represent the fast notes at all (a half-tempo map's 1/4-beat slot is
# twice as long in real time). So going finer is cheap insurance and going coarser
# is a real loss.
#
# The naive "maximise R over all ratios" is WRONG and was tried first: R rises on
# COARSER grids whenever the music emphasises every other slot, so it picked half
# tempo on 12 of 23 songs, including songs the plain local search had exactly right.
RATIOS = [1.0, 4.0 / 3.0, 3.0 / 2.0, 2.0, 3.0]
BPM_MIN, BPM_MAX = 60.0, 250.0
# How much better a coarser-to-finer move has to look before it is taken. Within
# this margin the levels are metrically equivalent and the FINEST is preferred.
R_NEAR = 0.9


def comb_multi(onsets: np.ndarray, bpm0: float, phase0: float,
               subdiv: int = 4) -> tuple[float, float, float]:
    """Try each metrical level, then refine — returns the best (bpm, phase, R).

    R (onset concentration on the resulting grid) is what picks the winner, and it
    doubles as a CONFIDENCE: on the eval set every song the plain local search got
    right scored R >= 0.177, while every song it got wrong by a 2/3 ratio scored
    R <= 0.050. A low R means "this grid has no relationship to the music" and is
    exactly the signal the current pipeline lacks — `librosa.beat_track` reports a
    tempo with no indication that it is nonsense.
    """
    cands = []
    for r in RATIOS:
        cand = bpm0 * r
        if not (BPM_MIN <= cand <= BPM_MAX):
            continue
        bpm, phase, rr = comb_refine(onsets, cand, phase0, subdiv=subdiv)
        if rr == rr:
            cands.append((bpm, phase, rr))
    if not cands:
        return bpm0, phase0, float("nan")
    best_r = max(c[2] for c in cands)
    # Among levels that fit essentially as well, take the finest grid.
    near = [c for c in cands if c[2] >= R_NEAR * best_r]
    return max(near, key=lambda c: c[0])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--songs", nargs="*")
    ap.add_argument("--json", help="write per-song results here")
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args()

    import librosa

    true_bpm = json.loads(TRUE_BPM_PATH.read_text()) if TRUE_BPM_PATH.exists() else {}
    ids = a.songs or sorted(true_bpm)
    print(f"{'song':10s}{'true':>8s}{'librosa':>9s}{'err%':>8s}"
          f"{'beat_lsq':>10s}{'err%':>8s}{'comb':>9s}{'err%':>8s}{'R':>7s}"
          f"{'multi':>9s}{'err%':>8s}{'R':>7s}")
    print("-" * 102)

    rows = []
    for sid in ids:
        audio = _audio_for(sid)
        tb = true_bpm.get(sid)
        if audio is None or tb is None:
            continue
        y, sr = librosa.load(str(audio), sr=None, mono=True)
        lb, beats = librosa_baseline(y, sr)
        bl, ph = beat_lsq(beats)
        f = ONSETS / f"{sid}.npz"
        on = np.load(f, allow_pickle=False)["onsets"] if f.exists() else np.array([])
        cb, cph, r = comb_refine(on, bl, ph)
        mb, mph, mr = comb_multi(on, bl, ph)

        def err(v):
            return (v - tb) / tb * 100 if v == v else float("nan")

        rows.append({"song": sid, "true": tb, "librosa": lb, "beat_lsq": bl,
                     "comb": cb, "comb_multi": mb, "phase_s": mph, "R": r, "R_multi": mr,
                     "err_librosa": err(lb), "err_lsq": err(bl), "err_comb": err(cb),
                     "err_multi": err(mb)})
        print(f"{sid[:9]:10s}{tb:8.1f}{lb:9.2f}{err(lb):+8.2f}"
              f"{bl:10.2f}{err(bl):+8.2f}{cb:9.2f}{err(cb):+8.2f}{r:7.3f}"
              f"{mb:9.2f}{err(mb):+8.2f}{mr:7.3f}", flush=True)

    if not rows:
        print("no songs scored"); raise SystemExit(2)

    print("\n=== ACCURACY (share of songs within a tolerance of the human bpm) ===")
    print(f"{'estimator':12s}{'<=0.1%':>9s}{'<=0.5%':>9s}{'<=1%':>8s}"
          f"{'median |err|':>14s}")
    n = len(rows)
    for name, key in [("librosa", "err_librosa"), ("beat_lsq", "err_lsq"),
                      ("comb", "err_comb"), ("comb_multi", "err_multi")]:
        e = [abs(r[key]) for r in rows if r[key] == r[key]]
        if not e:
            continue
        print(f"{name:12s}{sum(x <= 0.1 for x in e):6d}/{n:<3d}"
              f"{sum(x <= 0.5 for x in e):6d}/{n:<3d}{sum(x <= 1.0 for x in e):5d}/{n:<3d}"
              f"{statistics.median(e):13.2f}%")

    print("\nREAD: the pipeline uses `librosa` today. A tempo within 0.1% keeps the")
    print("grid inside a few ms of the music for a whole song; 1% slides it a full")
    print("1/8 note in ~25 seconds, which is what A8 measures as scattered timing.")
    print("Octave/two-thirds errors are a SEPARATE failure — the local search here")
    print("deliberately cannot cross them, so a song that starts at 2/3 tempo stays")
    print("wrong. Fix those with octave logic before reading this table as final.")

    if a.json:
        pathlib.Path(a.json).write_text(json.dumps(rows, indent=2))
        print(f"\nwrote {a.json}")


if __name__ == "__main__":
    main()
