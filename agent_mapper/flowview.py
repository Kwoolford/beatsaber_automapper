#!/usr/bin/env python
"""V3 — THE FLOW VIEW: where the bursts are, and whether the song asked for them.

★**Why this exists.** Kyle's fifth defect is the only one nobody can act on:

> *"random bursts of really fast non flowy notes"*

It names a symptom with **no timestamp**. Every other defect points somewhere — a drop,
a vocal line, a tempo — but this one has never been locatable, so nothing has ever been
fixed for it. V1 (the notesheet) drew the song and V2 (the overlay) drew our notes
against it; neither draws what the *hands* are doing, which is what "flow" means.

**The DoD is not a number. It is: given his complaint, we can point at the bar.**

## The claim has two halves, and they are separable

*"random"* and *"non flowy"* are different accusations, and a burst can be guilty of
either alone:

| half | what it means | how it is measured here |
|---|---|---|
| **random** | the burst is not in the music — the song did not get busier | `motivation`: musical event rate inside the burst ÷ the song's own median rate |
| **non flowy** | the burst is uncomfortable to *play* | `harsh` (wrist rotations > 90°), `travel` (grid distance per second), and parity `resets` inside it |

A burst that is motivated **and** flowy is a **good** burst — a mapper's payoff on a
fill — and this module is careful to say so rather than flagging every fast passage.
The output ranks bursts by how *unmotivated* they are, because that is his word
("random"), and prints the flow cost beside it rather than blending the two into one
score. ⚠️**Blending them would be the mistake this project keeps making**: a single
number is not correctable by ear, and Kyle correcting the picture is the entire point.

## What counts as a burst
A maximal run of ≥`MIN_NOTES` notes whose consecutive gaps are all ≤ `GAP_BEATS` of a
beat. **Both hands together**, because a burst is heard as one event by the player, not
per hand — the hand-level view is what `travel`/`harsh` then report *inside* it.

⚠️**Reuses `swing_sim` and `evaluation.flow` for every play-level quantity** (swings,
parity, the parity-aware swing angle, travel). If the picture computed its own
ergonomics it could disagree with the flow axis, and then a disagreement between Kyle
and the axis would be unattributable.

Usage:
    python agent_mapper/flowview.py data/eval_songset/1f8d6.ogg --map <map.zip>
    python agent_mapper/flowview.py <audio> --map <zip> --human <human.zip>   # compare
"""

from __future__ import annotations

import argparse
import math
import pathlib
import statistics
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

# ★**THE THRESHOLD IS ADAPTIVE, AND A FIXED ONE WAS MEASURED WRONG FIRST.** The first
# version asked for gaps <= 0.30 beats (1/4-note streams) and found **zero bursts in
# every map, ours and human alike** — because nothing in this corpus is that fast.
# Measured over the standing songs: median gap **0.5-1.0 beats**, and the smallest
# non-zero gap anywhere is **0.25**. A player's "really fast" is relative to what the
# rest of the map does, so the threshold is the map's own fast rate:
#
#   thr = min( its 10th-percentile gap , its median gap / SPEEDUP )
#
# On a map whose median is a quarter note, a sustained run of eighths IS the burst.
#
# ⚠️**And it is measured on DISTINCT TIMES, not swings.** A double is two swings at one
# instant: it does not make a hand move faster, and counting it as speed would have
# reported the doubles defect (C5) a second time wearing a flow costume. On
# FallenKingdom_BEFORE that is 788 swings but only **497 distinct times** (a 37 %
# collapse) against the human's 692 -> 646 (**7 %**).
SPEEDUP = 1.6             # a burst runs at least this much faster than the map's median
MIN_NOTES = 5             # below this a run reads as an accent, not a burst
MAX_GAP_S = 0.30          # wall-clock floor: slower than this is nobody's "really fast"
MIN_GAP_BEATS = 0.20      # a grid finer than 1/16 is quantisation noise, not a stream
# "The song got busier here" is relative to the song's own median event rate, so a
# uniformly dense song does not read as one long burst.
MOTIVATED = 1.25          # >= this much of the median rate = the music asked for it
UNMOTIVATED = 0.90        # <= this = the music did not


def swings_of(map_zip: pathlib.Path, difficulty: str = "Expert") -> tuple[list, float]:
    """Every swing of both hands, in beat order, plus the map's bpm."""
    from beatsaber_automapper.evaluation import swing_sim as ss

    bm, bpm = ss._load_difficulty(map_zip, difficulty)
    card = ss.simulate(bm, bpm=bpm)
    sw = [s for h in card.per_hand.values() for s in h.swings]
    sw.sort(key=lambda s: (s.beat, s.color))
    return sw, float(bpm)


def _harsh_and_travel(swings: list, spb: float) -> tuple[float, float, int]:
    """(harsh fraction, median travel px/s, resets) over one run, per hand then pooled.

    ⚠️Consecutive swings **of the same hand** — the hand is what rotates and travels.
    Pooling the two hands' swings into one sequence would measure a quantity no wrist
    experiences.
    """
    from beatsaber_automapper.evaluation import flow as ef
    from beatsaber_automapper.evaluation import swing_sim as ss

    angles: list[float] = []
    travels: list[float] = []
    resets = 0
    for color in (0, 1):
        hand = [s for s in swings if s.color == color]
        resets += sum(1 for s in hand if s.is_reset and s.reset_kind == "violation")
        for prev, cur in zip(hand, hand[1:]):
            a0, a1 = ef._swing_angle(prev), ef._swing_angle(cur)
            if a0 is not None and a1 is not None:
                angles.append(ss._angle_delta(a0, a1))
            dt = (cur.beat - prev.end_beat) * spb
            if dt > 1e-3:
                travels.append(math.hypot(cur.x - prev.end_x, cur.y - prev.end_y) / dt)
    harsh = (sum(1 for a in angles if a > ss.ANGLE_FLOW_DEG) / len(angles)
             if angles else float("nan"))
    trav = statistics.median(travels) if travels else float("nan")
    return harsh, trav, resets


def burst_threshold(times: np.ndarray, spb: float) -> tuple[float, dict]:
    """The gap that counts as fast FOR THIS MAP, plus why."""
    g = np.diff(times)
    g = g[g > 1e-6]
    if len(g) < 8:
        return float("nan"), {"reason": "too few notes"}
    med = float(np.median(g))
    p10 = float(np.percentile(g, 10))
    thr = min(p10, med / SPEEDUP)
    thr = max(thr, MIN_GAP_BEATS)
    return thr, {"median_gap": round(med, 3), "p10_gap": round(p10, 3),
                 "thr_beats": round(thr, 3), "thr_s": round(thr * spb, 3),
                 "too_slow_to_be_a_burst": bool(thr * spb > MAX_GAP_S)}


def find_bursts(swings: list, bpm: float, thr_beats: float | None = None,
                min_notes: int = MIN_NOTES) -> tuple[list[dict], dict]:
    """Maximal fast runs over DISTINCT times, both hands together, with their cost.

    Both hands together because a burst is heard as one event by the player; the
    per-hand view is what `harsh`/`travel` then report *inside* it.
    """
    spb = 60.0 / bpm if bpm > 0 else 0.5
    if not swings:
        return [], {"reason": "no swings"}
    beats = np.array([s.beat for s in swings], dtype=float)
    times = np.unique(np.round(beats, 4))
    auto, why = burst_threshold(times, spb)
    thr = auto if thr_beats is None else thr_beats
    if thr != thr:
        return [], why
    why["thr_used"] = round(thr, 3)

    runs, cur = [], [times[0]]
    for prev, nxt in zip(times, times[1:]):
        if nxt - prev <= thr + 1e-9:
            cur.append(nxt)
        else:
            if len(cur) >= min_notes:
                runs.append(cur)
            cur = [nxt]
    if len(cur) >= min_notes:
        runs.append(cur)

    bursts = []
    for run in runs:
        b0, b1 = run[0], run[-1]
        inside = [s for s in swings if b0 - 1e-6 <= s.beat <= b1 + 1e-6]
        t0, t1 = b0 * spb, b1 * spb
        harsh, trav, resets = _harsh_and_travel(inside, spb)
        bursts.append({
            "t0": t0, "t1": t1, "beat0": float(b0), "beat1": float(b1),
            "n": len(run), "n_swings": len(inside), "dur": t1 - t0,
            "nps": len(inside) / max(t1 - t0, 1e-6),
            "harsh": harsh, "travel": trav, "resets": resets,
            "hands": len({s.color for s in inside}),
        })
    return bursts, why


def motivate(bursts: list[dict], d: dict, main_only: bool = False) -> list[dict]:
    """Did the MUSIC get busier where the map did? Adds `motivation` to each burst.

    The reference is the song's **own** median event rate over windows the same length
    as the burst — not a corpus constant. A song that is busy everywhere then has no
    unmotivated bursts, which is correct: nothing about it is a surprise to the player.
    """
    import overlay as _ov

    if main_only:
        ev = np.array([e["t"] for e in _ov.main_events(d)], dtype=float)
    else:
        ev = np.array(sorted([e["t"] for e in _ov.main_events(d)]
                             + [e["t"] for e in _ov._minor_events(d)]), dtype=float)
    dur = float(d["dur"])
    for b in bursts:
        w = max(b["dur"], 0.5)
        inside = int(np.sum((ev >= b["t0"]) & (ev <= b["t1"]))) / w
        # the song's own distribution of event rate at this window length
        starts = np.arange(0.0, max(dur - w, 0.1), w / 2.0)
        rates = np.array([np.sum((ev >= s) & (ev <= s + w)) / w for s in starts])
        med = float(np.median(rates)) if len(rates) else float("nan")
        b["ev_rate"] = inside
        b["song_rate"] = med
        b["motivation"] = inside / med if med > 1e-9 else float("nan")
        b["verdict"] = ("motivated" if b["motivation"] >= MOTIVATED else
                        "RANDOM" if b["motivation"] <= UNMOTIVATED else "neutral")
    return bursts


def bar_of(d: dict, t: float) -> int:
    import brief as _brief
    g = d["grid"]
    t0 = _brief.bar_time(g, 0)
    return int((t - t0) // g["bar_s"])


def _mmss(t: float) -> str:
    return f"{int(t // 60)}:{t % 60:05.2f}"


def report(audio: pathlib.Path, map_zip: pathlib.Path, d: dict,
           difficulty: str = "Expert") -> dict:
    sw, bpm = swings_of(map_zip, difficulty)
    bursts, why = find_bursts(sw, bpm)
    bursts = motivate(bursts, d)
    return {"bpm": bpm, "n_swings": len(sw), "bursts": bursts, "threshold": why,
            "map": map_zip.stem, "difficulty": difficulty}


def note_rows(map_zip: pathlib.Path, difficulty: str = "Expert") -> tuple[list[dict], float]:
    """Every swing as a drawable row: when, which hand, where on the grid, which way.

    ⚠️`x`/`y` are the grid cell of the swing's FIRST note and `end_x`/`end_y` its last:
    a slider travels during the swing, and drawing only the head would hide exactly the
    movement the flow view exists to show.
    """
    sw, bpm = swings_of(map_zip, difficulty)
    spb = 60.0 / bpm if bpm > 0 else 0.5
    rows = [{"t": s_.beat * spb, "t_end": s_.end_beat * spb, "hand": int(s_.color),
             "x": int(s_.x), "y": int(s_.y), "end_x": int(s_.end_x),
             "end_y": int(s_.end_y), "dir": int(s_.direction),
             "flex": bool(s_.flexible), "reset": bool(s_.is_reset),
             "kind": s_.reset_kind, "notes": int(s_.note_count)}
            for s_ in sw]
    return rows, bpm


def analyse(map_zip: pathlib.Path, d: dict, difficulty: str = "Expert") -> dict:
    """Everything the flow lane draws: swing rows + located bursts + the threshold."""
    rows, bpm = note_rows(map_zip, difficulty)
    sw, _ = swings_of(map_zip, difficulty)
    bursts, why = find_bursts(sw, bpm)
    bursts = motivate(bursts, d)
    for b in bursts:
        b["bar"] = bar_of(d, b["t0"])
    ndist = len({round(s_.beat, 4) for s_ in sw})
    return {"rows": rows, "bursts": bursts, "threshold": why, "bpm": bpm,
            "n_swings": len(sw), "n_distinct": ndist,
            "doubled": round(1 - ndist / max(len(sw), 1), 3)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("audio", type=pathlib.Path)
    ap.add_argument("--map", type=pathlib.Path, required=True)
    ap.add_argument("--human", type=pathlib.Path, default=None,
                    help="a human map of the same song, for the same table")
    ap.add_argument("--difficulty", default="Expert")
    ap.add_argument("--gap", type=float, default=None,
                    help="override the adaptive burst threshold, in beats")
    ap.add_argument("--min-notes", type=int, default=MIN_NOTES)
    a = ap.parse_args()

    import notesheet as _ns
    d = _ns.collect(a.audio)

    for label, zp in (("OURS", a.map), ("HUMAN", a.human)):
        if zp is None:
            continue
        sw, bpm = swings_of(zp, a.difficulty)
        bursts, why = find_bursts(sw, bpm, a.gap, a.min_notes)
        bursts = motivate(bursts, d)
        dur = d["dur"]
        ndist = len(np.unique(np.round([s.beat for s in sw], 4)))
        print(f"\n=== {label}: {zp.stem}  ({len(sw)} swings at {ndist} distinct times "
              f"= {1 - ndist/max(len(sw),1):.0%} doubled, bpm {bpm:.1f}) ===")
        print(f"  fast-for-this-map = gap <= {why.get('thr_used', float('nan')):.3f} beats "
              f"({why.get('thr_s', float('nan')):.3f} s); median gap "
              f"{why.get('median_gap', float('nan')):.3f}")
        if not bursts:
            print("  no bursts at this threshold")
            continue
        print(f"  {len(bursts)} bursts, "
              f"{sum(b['n'] for b in bursts)} of {len(sw)} swings "
              f"({sum(b['n'] for b in bursts)/max(len(sw),1):.0%} of the map), "
              f"{sum(b['dur'] for b in bursts)/dur:.0%} of its running time")
        n_rand = sum(1 for b in bursts if b["verdict"] == "RANDOM")
        print(f"  ★{n_rand} of {len(bursts)} are RANDOM "
              f"(the music did NOT get busier under them)")
        print(f"\n  {'when':>8} {'bar':>5} {'n':>3} {'nps':>5} {'motiv':>6} "
              f"{'verdict':<10} {'harsh':>6} {'travel':>7} {'resets':>6}")
        for b in sorted(bursts, key=lambda x: (x["motivation"]
                                               if x["motivation"] == x["motivation"]
                                               else 9e9)):
            print(f"  {_mmss(b['t0']):>8} {bar_of(d, b['t0']):>5} {b['n']:>3} "
                  f"{b['nps']:>5.1f} {b['motivation']:>6.2f} {b['verdict']:<10} "
                  f"{b['harsh']:>6.2f} {b['travel']:>7.2f} {b['resets']:>6}")
    print("\nRANDOM = motivation <= %.2f (event rate under the burst is at or below the "
          "song's own median).\nharsh = share of wrist rotations > 90 deg; travel = grid "
          "cells/s; resets = parity violations." % UNMOTIVATED)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
