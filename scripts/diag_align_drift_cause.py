#!/usr/bin/env python
"""K1 diagnosis — WHY does alignment degrade toward the end of a song?

`eval_align_drift.py` establishes THAT it does (for a subset of songs). This
script asks what causes it, and it exists because the TODO's proposed diagnosis
turned out to be wrong: the hypothesis was "fit tempo per-segment -- either the
song's tempo genuinely moves (answer: BPM events) or our single global fit
accumulates error (answer: piecewise fit)". Measured, it is **neither**.

Three tests, each of which can independently kill a hypothesis:

1. HUMAN CONTROL (the C2 lesson). Score the human map of the SAME song against
   the SAME cached onsets. Where the human map also degrades, the cause is at
   least partly the song or the onset detector, and "fixing" it would be fitting
   the detector -- the h_dist failure. Where the human sits flat and we drift,
   it is ours.

2. OFFSET RAMP. Median match offset per quintile. Accumulated tempo error MUST
   show up as a monotone ramp of tens of ms. A flat wobble means the notes that
   land are just as accurately placed at the end as at the start, and the lost
   precision is notes that match NOTHING -- a selection defect, not a timing one.

3. DENSITY TRACKING. Notes/s over onsets/s per time-fifth. If the ratio climbs
   as a song winds down, we are holding note density roughly constant while the
   music thins out, and the extra notes have no onset to land on.

Usage:
    python scripts/diag_align_drift_cause.py --arm tf_hl014_ds048
"""

from __future__ import annotations

import argparse
import glob
import pathlib
import statistics as st
import sys

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from beatsaber_automapper.evaluation import alignment, scorecard  # noqa: E402

NQ = 5


def _load(p) -> tuple | None:
    try:
        return scorecard._load_any(pathlib.Path(p))
    except Exception:  # noqa: BLE001
        return None


def _quintile_prec_and_lag(loaded) -> tuple[list, list, list] | None:
    bm, bpm, ons = loaded
    if ons is None or len(ons) == 0:
        return None
    times = alignment.note_times(bm, bpm)
    if len(times) < NQ * alignment.MIN_NOTES:
        return None
    ref = np.sort(np.asarray(ons, dtype=float))
    prec, lag, mad = [], [], []
    for chunk in np.array_split(np.asarray(times, dtype=float), NQ):
        matched, offs = alignment.match_offsets(list(chunk), ref)
        prec.append(matched / len(chunk))
        if offs:
            med = st.median(offs)
            lag.append(med * 1000.0)
            mad.append(st.median([abs(o - med) for o in offs]) * 1000.0)
        else:
            lag.append(float("nan"))
            mad.append(float("nan"))
    return prec, lag, mad


def _human_zip(sid: str) -> pathlib.Path | None:
    c = sorted((REPO / "data" / "raw").glob(f"{sid}*.zip"))
    return c[0] if c else None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", default="tf_hl014_ds048")
    ap.add_argument("--cache", default="outputs/eval_sweep_cache")
    a = ap.parse_args()

    paths = sorted(glob.glob(f"{a.cache}/{a.arm}__*.zip"))
    if not paths:
        sys.exit(f"no cached maps for arm {a.arm}")

    rows = {}
    for p in paths:
        sid = pathlib.Path(p).name.split("__", 1)[1][:-4]
        L = _load(p)
        if not L:
            continue
        r = _quintile_prec_and_lag(L)
        if r:
            rows[sid] = {"ours": r, "loaded": L}
    for sid in list(rows):
        hz = _human_zip(sid)
        if hz:
            L = _load(hz)
            if L:
                r = _quintile_prec_and_lag(L)
                if r:
                    rows[sid]["human"] = r

    # ---- TEST 1: human control -------------------------------------------
    print("=== TEST 1 — HUMAN CONTROL (same song, same onsets) ===")
    print("Where the human ALSO drifts, the cause is the song/detector, not our grid.")
    print(f"{'song':8s}{'ourDrift':>10s}{'humDrift':>10s}{'tail o/h':>10s}  verdict")
    ours_to_fix, shared = [], []
    HUM_P90 = 0.1451  # measured on the human corpus, eval_align_drift.py
    for sid in sorted(rows, key=lambda s: -(rows[s]["ours"][0][0] - rows[s]["ours"][0][-1])):
        r = rows[sid]
        od = r["ours"][0][0] - r["ours"][0][-1]
        if "human" not in r:
            continue
        hd = r["human"][0][0] - r["human"][0][-1]
        bm, bpm, ons = r["loaded"]
        t = alignment.note_times(bm, bpm)
        last = float(np.max(ons))
        on = sum(1 for x in t if x > last)
        if od <= HUM_P90:
            continue
        v = "shared with human" if hd > HUM_P90 else "OURS"
        (shared if hd > HUM_P90 else ours_to_fix).append(sid)
        print(f"{sid:8s}{od:>10.3f}{hd:>10.3f}{on:>10d}  {v}")
    print(f"\nours to fix: {ours_to_fix}\nshared:      {shared}")

    # ---- TEST 2: offset ramp ---------------------------------------------
    print("\n=== TEST 2 — OFFSET RAMP (median match offset, ms, per quintile) ===")
    print("Accumulated tempo error => a MONOTONE ramp of tens of ms. Flat => not tempo.")
    print(f"{'song':12s}" + "".join(f"{'q'+str(i+1):>8s}" for i in range(NQ)) + f"{'q5-q1':>9s}")
    for sid in ours_to_fix + shared:
        for who in ("ours", "human"):
            if who not in rows[sid]:
                continue
            lag = rows[sid][who][1]
            tag = sid if who == "ours" else f"{sid}(hum)"
            print(f"{tag:12s}" + "".join(f"{x:>8.1f}" for x in lag)
                  + f"{lag[-1]-lag[0]:>9.1f}")

    # ---- TEST 3: density tracking ----------------------------------------
    print("\n=== TEST 3 — DENSITY TRACKING (notes/s over onsets/s, time-fifths) ===")
    print("Ratio CLIMBING as a song winds down => we hold density while music thins.")
    print(f"{'song':12s}" + "".join(f"{'f'+str(i+1):>9s}" for i in range(NQ)))
    for sid in ours_to_fix + shared:
        bm, bpm, ons = rows[sid]["loaded"]
        t = np.asarray(alignment.note_times(bm, bpm), dtype=float)
        ref = np.sort(np.asarray(ons, dtype=float))
        T = float(max(t.max(), ref.max()))
        edges = np.linspace(0, T, NQ + 1)
        nn, _ = np.histogram(t, bins=edges)
        no, _ = np.histogram(ref, bins=edges)
        ratio = [nn[i] / no[i] if no[i] > 0 else float("nan") for i in range(NQ)]
        print(f"{sid:12s}" + "".join(f"{x:>9.3f}" if x == x else f"{'--':>9s}"
                                     for x in ratio))
        print(f"{'  onsets/s':12s}" + "".join(f"{no[i]/(T/NQ):>9.2f}" for i in range(NQ)))


if __name__ == "__main__":
    main()
