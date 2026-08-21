#!/usr/bin/env python
"""How much of the audio does our 1/4-beat build grid REPRESENT?

⚠️**MISNAMED WHEN WRITTEN, kept for the record.** The number below is the share of
ONSETS whose nearest grid slot is within tolerance -- recall-side representability. It
is NOT an upper bound on `onset_precision`, which asks the opposite question (does
THIS note slot have an onset near it). Three songs scored ABOVE their "ceiling",
which is impossible for a real bound and is what exposed the error.

We sit at `onset_precision` 0.856 against a human 0.919 (14th percentile). The
remaining suspect is structural rather than a choice: `mapctl` snaps every note to
the build grid, and at 160 bpm a 1/4-beat slot is 93.75 ms while the alignment axis
matches at 50 ms. A snap can therefore push a note off its own onset **by
construction**.

★**This is answerable without building a map.** Take each song's cached audio onsets,
snap each to the nearest grid slot, and measure what fraction land within the
tolerance. That is the BEST any map on that grid could score -- a ceiling, not an
estimate. Then the question splits cleanly:

    ceiling ~= ours       we are AT the structural limit; the fix is a finer grid
    ceiling >> ours       the grid is innocent and note SELECTION is the defect

⚠️A ceiling is not a target. The human maps are NOT on our grid, so their 0.919 is
not evidence that 0.919 is reachable at SUBDIV 4 -- which is exactly what this
measures.

Also reports the ceiling at SUBDIV 8 and 16 to size the prize before anyone pays for
it. ⚠️Global SUBDIV 8 is REFUTED for the ML path (precision -0.127, TODO C4/landmines);
this says nothing about that, only what the grid permits.

Usage:
    python scripts/diag_grid_ceiling.py
"""
from __future__ import annotations

import argparse
import pathlib
import statistics
import sys

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "agent_mapper"))

TOL_S = 0.05          # the alignment axis' matching tolerance
BEATS_PER_BAR = 4


def onsets_for(sid: str):
    f = REPO / "outputs" / "onset_cache" / f"{sid}.npz"
    if not f.exists():
        return None
    z = np.load(f)
    return z[list(z.keys())[0]]


def ceiling(onsets, phase: float, bar_s: float, subdiv: int) -> tuple[float, float]:
    """(share of onsets reachable within TOL, median snap distance ms) at `subdiv`."""
    slot_s = bar_s / (BEATS_PER_BAR * subdiv)
    k = np.round((onsets - phase) / slot_s)
    snapped = phase + k * slot_s
    d = np.abs(onsets - snapped)
    return float((d <= TOL_S).mean()), float(np.median(d) * 1000.0)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--songs", nargs="*", default=None)
    a = ap.parse_args()

    import brief as B

    sids = a.songs or [p.stem for p in
                       sorted((REPO / "data" / "eval_songset").glob("*.ogg"))]
    rows = []
    for sid in sids:
        on = onsets_for(sid)
        audio = REPO / "data" / "eval_songset" / f"{sid}.ogg"
        if on is None or not audio.exists():
            continue
        try:
            an = B.analyse(audio)
            g = B.grid(an)
        except Exception as exc:  # noqa: BLE001
            print(f"  {sid}: grid failed ({exc})")
            continue
        r = {"sid": sid, "bpm": g["bpm"], "slot_ms": g["slot"] * 1000}
        for sd in (4, 8, 16):
            c, dm = ceiling(on, g["phase"], g["bar_s"], sd)
            r[f"c{sd}"] = c
            r[f"d{sd}"] = dm
        rows.append(r)

    print(f"\n{'song':<8}{'bpm':>7}{'1/4 slot':>10}"
          f"{'ceil@4':>9}{'ceil@8':>9}{'ceil@16':>9}")
    print("-" * 55)
    for r in rows:
        print(f"{r['sid']:<8}{r['bpm']:>7.1f}{r['slot_ms']:>9.1f}ms"
              f"{r['c4']:>9.3f}{r['c8']:>9.3f}{r['c16']:>9.3f}")
    if not rows:
        print("no songs scored")
        return 0
    m4 = statistics.median([r["c4"] for r in rows])
    m8 = statistics.median([r["c8"] for r in rows])
    m16 = statistics.median([r["c16"] for r in rows])
    print(f"\n{'MEDIAN':<8}{'':>7}{'':>10}{m4:>9.3f}{m8:>9.3f}{m16:>9.3f}")

    OURS, HUMAN = 0.856, 0.919
    print(f"\nours {OURS:.3f}   human {HUMAN:.3f}   ceiling at our SUBDIV 4: {m4:.3f}")
    if m4 < HUMAN:
        print(f"★THE GRID CANNOT REACH THE HUMAN: no map on a 1/4-beat grid can score "
              f"above {m4:.3f}, and the human sits at {HUMAN:.3f}. Their maps are not "
              f"on our grid, so that number was never reachable at SUBDIV 4.")
    if m4 - OURS < 0.03:
        print(f"⇒ WE ARE AT THE STRUCTURAL LIMIT (headroom {m4 - OURS:+.3f}). Note "
              f"SELECTION cannot fix this; only a finer grid can. Prize for SUBDIV 8: "
              f"{m8 - m4:+.3f}.")
    else:
        print(f"⇒ THE GRID IS NOT THE BINDING CONSTRAINT (headroom {m4 - OURS:+.3f} "
              f"still available at SUBDIV 4). Look at which onsets we choose, not at "
              f"the grid.")
    print("⚠️A ceiling is what the grid PERMITS, not what a good map scores. Do not "
          "read the gap to it as a to-do list.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
