#!/usr/bin/env python
"""P0.1 — THE MAIN BEAT: which pulse is the song actually built on?

Kyle, 2026-08-04: *"It feels like every couple main beat notes were mapped
instead of most of the main beats… Like it hits the main flow partially."*

Everything in the new suite rests on knowing what "the main beat" IS. A first
attempt assumed one metrical level — the fitted beat — and it was fragile: on
1fa48 the human map covered **9.7 %** of those beats and on 1fb44 **100 %**, a
spread that says the grid, not the mapper, was wrong. A human hears the pulse at
whatever level the song states it; a mapper on a slow song plays half-notes and
on a fast one eighths, and both feel like "the main beat".

**Method.** Score candidate grids `period = beat * r` for several ratios, each at
its best phase, against the CARRIER onsets (drums ∪ bass — the rhythm section,
not vocals or lead). Two quantities matter and they pull opposite ways:

    support = share of grid positions that carry a carrier onset
              (a too-FINE grid has many empty positions -> low)
    capture = share of carrier onsets that land on a grid position
              (a too-COARSE grid leaves onsets unexplained -> low)

`f1(support, capture)` picks the level that both is played and explains the
playing. ⚠️This is the same trap `data/tempo.py` documents from the other side:
a single one-sided score always prefers one extreme — there, maximising onset
concentration chose HALF tempo on 12 of 23 songs. Two-sided scoring is the fix.

Returns the chosen level, its phase, its confidence, and — deliberately — the
scores of **every** candidate, so a view can show when the map follows a
different level from the music.
"""

from __future__ import annotations

import pathlib
import sys
from dataclasses import dataclass, field

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[1]

# ⚠️0.25 (a 16th grid) is NOT a candidate: nobody hears sixteenths as "the main
# beat", and including it exposed the bug below. Eighths CAN be the main pulse in
# fast music, so 0.5 stays.
RATIOS = (0.5, 1.0, 2.0)
TOL = 0.070


def _tol(period: float) -> float:
    """⚠️THE TOLERANCE MUST SCALE WITH THE PERIOD.

    First version used a flat 70ms for every candidate. At a 16th grid (period
    0.08s) the spacing is SMALLER than 2*tol, so every onset is within tolerance
    of some grid point and `capture` is 1.000 BY CONSTRUCTION -- the score then
    picked the finest grid on 20 of 24 songs. A hit only means something if it is
    within a fraction of the period being tested.
    """
    return min(TOL, 0.25 * period)
MIN_RUN = 4


@dataclass(slots=True)
class MainBeat:
    period: float                    # seconds between main beats
    phase: float                     # seconds, offset of the first beat
    ratio: float                     # period / (60/bpm)
    support: float                   # share of grid positions that are played
    capture: float                   # share of carrier onsets explained
    f1: float
    grid: np.ndarray = field(repr=False)        # every main-beat position
    runs: np.ndarray = field(repr=False)        # positions inside a run of >=MIN_RUN
    candidates: list = field(default_factory=list, repr=False)

    @property
    def confidence(self) -> str:
        if self.f1 >= 0.55:
            return "high"
        return "medium" if self.f1 >= 0.40 else "LOW — treat the grid as unreliable"


def carrier_onsets(song_id: str) -> np.ndarray | None:
    f = REPO / "outputs" / "stem_onset_cache" / f"{song_id}.npz"
    if not f.exists():
        return None
    d = np.load(f, allow_pickle=True)
    parts = [d[f"onsets_{s}"] for s in ("drums", "bass") if f"onsets_{s}" in d.files]
    parts = [p for p in parts if len(p)]
    if not parts:
        return None
    return np.sort(np.concatenate(parts))


def _near(sorted_arr: np.ndarray, t: float, tol: float) -> bool:
    if len(sorted_arr) == 0:
        return False
    i = int(np.searchsorted(sorted_arr, t))
    return any(abs(t - sorted_arr[j]) <= tol
               for j in (i - 1, i) if 0 <= j < len(sorted_arr))


def _score(car: np.ndarray, period: float, phase: float, end: float) -> tuple:
    grid = np.arange(phase, end, period)
    if len(grid) < 8:
        return 0.0, 0.0, 0.0, grid
    tol = _tol(period)
    played = np.array([_near(car, t, tol) for t in grid])
    support = float(played.mean())
    # capture: onsets explained by SOME grid position
    idx = np.round((car - phase) / period)
    capture = float(np.mean(np.abs(car - (phase + idx * period)) <= tol))
    f1 = 0.0 if support + capture <= 0 else 2 * support * capture / (support + capture)
    return support, capture, f1, grid


def find_main_beat(song_id: str, bpm: float, end: float,
                   car: np.ndarray | None = None) -> MainBeat | None:
    if car is None:
        car = carrier_onsets(song_id)
    if car is None or len(car) < 40 or bpm <= 0:
        return None
    beat = 60.0 / bpm
    best = None
    cands = []
    for r in RATIOS:
        period = beat * r
        if period <= 0 or period > (end / 8):
            continue
        # best phase: try a fine sweep within one period
        bp, bs = 0.0, (-1.0, 0, 0, None)
        for ph in np.arange(0, period, max(period / 24, 0.01)):
            s, c, f, g = _score(car, period, float(ph), end)
            if f > bs[0]:
                bs, bp = (f, s, c, g), float(ph)
        f, s, c, g = bs
        cands.append(dict(ratio=r, period=period, phase=bp, support=s,
                          capture=c, f1=f))
        if best is None or f > best[0]:
            best = (f, r, period, bp, s, c, g)
    if best is None:
        return None
    f, r, period, phase, support, capture, grid = best

    played = np.array([_near(car, t, _tol(period)) for t in grid])
    runs = np.zeros(len(grid), dtype=bool)
    i = 0
    while i < len(played):
        if played[i]:
            j = i
            while j < len(played) and played[j]:
                j += 1
            if j - i >= MIN_RUN:
                runs[i:j] = True
            i = j
        else:
            i += 1
    return MainBeat(period=period, phase=phase, ratio=r, support=support,
                    capture=capture, f1=f, grid=grid, runs=grid[runs],
                    candidates=sorted(cands, key=lambda c: -c["f1"]))


def coverage(notes: np.ndarray, mb: MainBeat, use_runs: bool = True) -> dict:
    """How much of the main beat does a map actually play, and what else does it play?"""
    beats = mb.runs if use_runs and len(mb.runs) >= 20 else mb.grid
    notes = np.sort(np.asarray(notes, dtype=float))
    if len(beats) == 0 or len(notes) == 0:
        return {}
    tol = _tol(mb.period)
    hit = np.array([_near(notes, t, tol) for t in beats])
    on = np.array([_near(beats, t, tol) for t in notes])
    # consecutive coverage: humans hold the line, we drop in and out
    cont = 0.0
    if len(hit) > 1:
        prev = hit[:-1]
        cont = float(hit[1:][prev].mean()) if prev.any() else 0.0
    return {"main_covered": float(hit.mean()),
            "main_continuity": cont,
            "notes_on_main": float(on.mean()),
            "n_main_beats": int(len(beats))}


if __name__ == "__main__":
    import glob
    sys.path.insert(0, str(REPO / "src"))
    sys.path.insert(0, str(REPO / "scripts"))
    from beatsaber_automapper.evaluation import alignment, scorecard  # noqa: E402
    from calibrate_playfeel import load_expert_only  # noqa: E402

    print(f"{'song':9s}{'level':>7s}{'period':>8s}{'supp':>7s}{'capt':>7s}{'f1':>6s}"
          f"{'conf':>8s}   {'ourCov':>7s}{'humCov':>7s}{'ourCont':>8s}{'humCont':>8s}")
    for p in sorted(glob.glob(str(REPO / "outputs/eval_sweep_cache/tf_trim_ev03_rc05#s0__*.zip"))):
        sid = scorecard.song_id(pathlib.Path(p))
        L = scorecard._load_any(pathlib.Path(p))
        if not L:
            continue
        o = np.sort(np.asarray(alignment.note_times(L[0], L[1]), dtype=float))
        hz = REPO / "data" / "raw" / f"{sid}.zip"
        H = load_expert_only(hz) if hz.exists() else None
        h = np.sort(np.asarray(alignment.note_times(H[0], float(H[1])), dtype=float)) if H else None
        end = float(max(o.max(), h.max() if h is not None else 0))
        mb = find_main_beat(sid, float(L[1]), end)
        if mb is None:
            continue
        co = coverage(o, mb)
        ch = coverage(h, mb) if h is not None else {}
        print(f"{sid[:9]:9s}{mb.ratio:7.2f}{mb.period:8.3f}{mb.support:7.3f}"
              f"{mb.capture:7.3f}{mb.f1:6.3f}{mb.confidence.split()[0]:>8s}   "
              f"{co.get('main_covered', float('nan')):7.3f}"
              f"{ch.get('main_covered', float('nan')):7.3f}"
              f"{co.get('main_continuity', float('nan')):8.3f}"
              f"{ch.get('main_continuity', float('nan')):8.3f}")
