#!/usr/bin/env python
"""Probe tempo-octave errors in BPM detection. ⚠️ THE FIX HERE FAILED — see below.

**RESULT 2026-07-27 — the correction is NOT adopted. The measurement stands.**

Measured against the BPM declared in the human map for each song (ground truth,
set by hand by the mapper):

    raw librosa detection            16/23 correct within 2%   (7/23 WRONG, 30%)
    + octave rescoring (this file)   10/23   -- fixed 3, BROKE 9
    + conservative double-only test  14/23   -- fixed 2, broke 4

Both correction attempts made things WORSE, so `data/audio.py::detect_bpm` is
left alone. The hypothesis behind them — that the true metrical level is the one
whose grid has balanced odd/even beat energy — is simply false: plenty of real
music has strong backbeat or downbeat asymmetry at its true tempo, so onset-energy
balance does not discriminate metrical level.

**What to do with this instead:**
1. **30% wrong BPM is a real, quantified, upstream defect.** BPM determines the
   beat grid, the layout slots and the achievable note resolution; at half tempo
   the finest slot is twice as coarse in real time and the fast notes cannot be
   represented at all. This needs proper work (a tempo model, or a tempogram-ratio
   method), not a heuristic patch.
2. **For EVALUATION, sidestep it**: the eval songset comes from `data/raw`, so the
   human-declared BPM is available. Passing it to the generator removes tempo
   detection as a confound from every quality measurement we make. That is
   legitimate for evaluation and is NOT a production fix — production has no
   human map to read a BPM from.

--- original probe description ---

Found 2026-07-27 by reading a generated map next to its human counterpart in
scripts/map_view.py: our map said 94 BPM, the human map said 188. Across the
24-song eval set, 5/23 songs are detected at the wrong tempo — 2 at exactly half,
3 at a 2:3 misread. BPM is upstream of EVERYTHING (the beat grid, the layout
slots, note density), and at half tempo the finest grid slot is twice as coarse
in real time, so the fast notes simply cannot be represented.

Cause: `librosa.beat.beat_track` defaults to `start_bpm=120`, a prior that pulls
estimates toward 120 and produces the classic octave/metrical-level error.

Fix under test: after detection, score the plausible metrical relatives
(x1/2, x2/3, x1, x3/2, x2) and pick the level whose implied beat grid the ONSETS
actually support. The discriminating test is whether the *in-between* beats carry
comparable onset energy: if a grid at 2x has strong onsets on every beat, 2x is
the real tempo; if the odd beats are weak, we are running at double the true
tempo and should halve.

Ground truth is the BPM declared in the human map for the same song, which the
mapper set by hand.

Usage:
  python scripts/bpm_octave_probe.py
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

SONGSET = REPO / "data" / "eval_songset"
RAW = REPO / "data" / "raw"
MIN_BPM, MAX_BPM = 60.0, 240.0
RATIOS = (0.5, 2.0 / 3.0, 1.0, 1.5, 2.0)


def grid_support(oenv: np.ndarray, sr: int, hop: int, bpm: float) -> float:
    """How well the onset envelope supports a beat grid at this tempo.

    Returns the *weakest-link* support: the mean onset strength on the weaker
    half of the grid (odd vs even beats), normalised. A tempo that is twice the
    truth has strong even beats and weak odd ones, so this score collapses; the
    true tempo has both halves comparably strong.
    """
    period = 60.0 / bpm * sr / hop
    if period < 2 or period * 4 >= len(oenv):
        return 0.0
    n = int((len(oenv) - 1) / period)
    if n < 8:
        return 0.0
    # best phase: try a few offsets, keep the strongest total
    best = 0.0
    for ph in np.linspace(0, period, 8, endpoint=False):
        idx = np.rint(ph + period * np.arange(n)).astype(int)
        idx = idx[idx < len(oenv)]
        if len(idx) < 8:
            continue
        vals = oenv[idx]
        ev, od = vals[0::2], vals[1::2]
        if len(ev) < 2 or len(od) < 2:
            continue
        # weakest half, normalised by overall onset energy
        score = min(ev.mean(), od.mean()) / (oenv.mean() + 1e-9)
        best = max(best, score)
    return float(best)


def corrected_bpm(y: np.ndarray, sr: int, raw_bpm: float,
                  hop: int = 512) -> tuple[float, dict]:
    import librosa
    oenv = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)
    scores = {}
    for r in RATIOS:
        cand = raw_bpm * r
        if not (MIN_BPM <= cand <= MAX_BPM):
            continue
        scores[cand] = grid_support(oenv, sr, hop, cand)
    if not scores:
        return raw_bpm, {}
    # Prefer the FASTEST level whose support is within 5% of the best — a slower
    # level always scores at least as well (its beats are a subset), so ties must
    # break toward the faster reading or we re-introduce the halving bias.
    best = max(scores.values())
    ok = [b for b, s in scores.items() if s >= 0.95 * best]
    return max(ok), scores


def main() -> None:
    import librosa
    sys.path.insert(0, str(REPO / "scripts"))
    from feel_disc_poc import _zip_bpm

    songs = sorted(p for p in SONGSET.glob("*") if p.suffix.lower() in (".ogg", ".mp3"))
    print(f"{'song':10s}{'truth':>7s}{'raw':>8s}{'fixed':>8s}   {'raw err':>8s}{'fix err':>8s}")
    print("-" * 58)
    raw_ok = fix_ok = n = 0
    for sp in songs:
        truth_zip = RAW / f"{sp.stem}.zip"
        if not truth_zip.exists():
            continue
        truth = _zip_bpm(str(truth_zip))
        if not truth:
            continue
        y, sr = librosa.load(str(sp), sr=None, mono=True)
        y = y[: sr * 90]                       # 90 s is plenty for tempo
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        raw = float(np.atleast_1d(tempo)[0])
        fixed, _sc = corrected_bpm(y, sr, raw)
        er, ef = abs(raw - truth) / truth, abs(fixed - truth) / truth
        n += 1
        raw_ok += er < 0.02
        fix_ok += ef < 0.02
        flag = ""
        if er >= 0.02 and ef < 0.02:
            flag = "  FIXED"
        elif er < 0.02 and ef >= 0.02:
            flag = "  BROKE"
        print(f"{sp.stem:10s}{truth:7.1f}{raw:8.1f}{fixed:8.1f}   "
              f"{er:8.2f}{ef:8.2f}{flag}")
    print("-" * 58)
    print(f"within 2% of the human-declared BPM:  raw {raw_ok}/{n}   fixed {fix_ok}/{n}")
    print("\nDoD: fixed >= 21/23 and no song BROKEn => adopt in data/audio.py::detect_bpm")


if __name__ == "__main__":
    main()
