#!/usr/bin/env python
"""W1a — is the offbeat defect in Stage-1's PROBABILITIES or in the DECODE?

`eval_beat_phase.py` measured that we place a note half a beat off a
multi-instrument event **2.6x** more often than humans (0.245 vs 0.095). That has
two possible causes needing opposite fixes:

  DECODE   Stage-1 prefers the event slot, but selection / NMS / the density
           curve takes the offbeat one anyway  =>  a decode lever fixes it.

  STAGE-1  the probability field itself cannot tell the two apart  =>  **no
           decode lever can fix it**, because there is nothing to select on.
           That is Track B: `version_4` has only `drum_proj` + `mix_proj` and no
           instrument projection, so it cannot hear which instrument hit.

This reads a `BEAT_PROBS_DUMP` (raw Stage-1 probs, BEFORE thresholding/NMS/
density) and, for every k>=3 coincidence event, compares the probability at the
event's slot against the slot half a beat away.

The decisive statistic is `win_rate` -- how often the event slot beats the
offbeat slot. **0.5 is a coin flip and means Stage-1 is phase-blind.** The
`vs_random` ratio is reported beside it to show the model is not simply flat:
it can find the musically active region while still being unable to place the
downbeat inside it.

Usage:
    python scripts/eval_probs_phase.py --dumps outputs/probs_phase_2026-08-03
"""

from __future__ import annotations

import argparse
import json
import pathlib
import statistics as st
import sys

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from eval_coincidence import events_for  # noqa: E402


def analyse(npz: pathlib.Path, kmin: int = 3) -> dict | None:
    d = np.load(npz)
    P = d["beat_probs"]
    bpm, sub = float(d["bpm"]), int(d["beat_subdiv"])
    if sub < 2:
        return None
    slot = 60.0 / bpm / sub
    half = sub // 2                      # a half beat, in slots
    pmax = P.max(axis=1)                 # best of the two hands

    ev = events_for(npz.stem, 0.030)
    if ev is None:
        return None
    times, ks = ev
    e = times[ks >= kmin]
    si = np.round(e / slot).astype(int)
    keep = (si >= half) & (si < len(P) - half)
    si = si[keep]
    if len(si) < 50:
        return None

    on = pmax[si]
    off = np.maximum(pmax[si + half], pmax[si - half])
    rng = np.random.default_rng(0)
    rand = pmax[rng.integers(half, len(P) - half, size=max(len(si), 500))]

    return {
        "song": npz.stem,
        "n_events": int(len(si)),
        "p_on_event": round(float(np.median(on)), 4),
        "p_halfbeat": round(float(np.median(off)), 4),
        "win_rate": round(float(np.mean(on > off)), 4),
        "vs_random": round(float(np.median(on) / max(float(np.median(rand)), 1e-9)), 4),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dumps", required=True, help="dir of BEAT_PROBS_DUMP .npz files")
    ap.add_argument("--kmin", type=int, default=3)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()

    rows = []
    for f in sorted(pathlib.Path(a.dumps).glob("*.npz")):
        try:
            r = analyse(f, a.kmin)
        except Exception as exc:  # noqa: BLE001
            print(f"  {f.stem}: FAILED ({exc})")
            continue
        if r:
            rows.append(r)
            print(f"  {r['song']:10s} n={r['n_events']:5d}  on {r['p_on_event']:.4f}"
                  f"  half {r['p_halfbeat']:.4f}  win {r['win_rate']:.4f}"
                  f"  vs_random {r['vs_random']:.2f}x")
    if not rows:
        sys.exit("no dumps analysed")

    win = [r["win_rate"] for r in rows]
    vsr = [r["vs_random"] for r in rows]
    print(f"\n=== COHORT (n={len(rows)} songs) ===")
    print(f"  win_rate   median {st.median(win):.4f}   p10 {np.percentile(win,10):.4f}"
          f"   p90 {np.percentile(win,90):.4f}")
    print(f"  vs_random  median {st.median(vsr):.4f}")

    print("\n=== VERDICT ===")
    m = st.median(win)
    if m >= 0.60:
        print(f"  win_rate {m:.3f} >= 0.60  =>  DECODE. Stage-1 does prefer the event")
        print("  slot and something downstream is discarding that preference. Find it")
        print("  in threshold/NMS/density-select before touching the model.")
    elif m <= 0.55:
        print(f"  win_rate {m:.3f} ~ coin flip  =>  STAGE-1 IS PHASE-BLIND at")
        print("  multi-instrument events. No decode lever can fix this: there is no")
        print("  signal to select on. This is TRACK B -- the missing instrument")
        print("  projection -- and it makes W1 a retrain, not a knob.")
        print(f"  (vs_random {st.median(vsr):.2f}x says the model DOES find the active")
        print("   region; it just cannot place the downbeat inside it.)")
    else:
        print(f"  win_rate {m:.3f} is between the bands => partial preference.")
        print("  Report as PARTLY CONFIRMED and do not commit a GPU night either way.")
    print("\n  ⚠️ One song is not a cohort -- 1f333 is a documented probe trap.")
    print("     Weigh the p10/p90 spread, not just the median.")

    if a.json:
        pathlib.Path(a.json).write_text(json.dumps(
            {"win_rate_median": st.median(win), "vs_random_median": st.median(vsr),
             "rows": rows}, indent=2))
        print(f"\nwrote {a.json}")


if __name__ == "__main__":
    main()
