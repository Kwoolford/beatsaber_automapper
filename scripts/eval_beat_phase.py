#!/usr/bin/env python
"""W1 — OFFBEAT DISPLACEMENT at multi-instrument events.

> *"The notes are stubbornly not being placed on this tempo. They are being
> placed on all of the other little sounds."* — Kyle on SO TIRED ROCK

This is the measured form of that sentence, and it is **not** what A8 alignment
measures. A8 asks *"is this note on a real onset?"* — and a note sitting on one
of the "other little sounds" **passes A8**, because a lone-stem onset is still an
onset. A8 is blind to *which* onset we chose by construction.

This metric asks the next question: when several instruments hit together (a
`k>=3` event from `eval_coincidence.py`), how far off is our nearest note **in
beat phase**? The offset is wrapped into +-half a beat, so:

    phase ~ 0          we played that event
    phase ~ half beat  we played the OFFBEAT instead -- the note is on a real
                       audio event, just the wrong one, one eighth away

`halfbeat_rate` = the share of multi-instrument events whose nearest note falls
in the OUTER THIRD of the beat (|phase| >= 0.35 * beat). Humans have a floor here
and it is not zero: syncopation is real mapping, so the bar is the human cohort,
never zero.

Measured 2026-08-03 (link 30 ms, k>=3, strict-Expert human cohort):

    ours (tf_trim_ev03_rc05, 24 songs x 3 seeds)   0.245   p10 0.109  p90 0.310
    human (n=188)                                  0.095   p10 0.020  p90 0.189
    SO TIRED ROCK, ours                            0.316   <- past our own p90

⚠️**DIAGNOSTIC, not an axis.** It must clear `scripts/audit_eval_suite.py` before
it is allowed to steer the generator, like every metric since `h_dist`.

Usage:
    python scripts/eval_beat_phase.py --gen 'outputs/eval_sweep_cache/arm#s*__*.zip'
"""

from __future__ import annotations

import argparse
import glob
import json
import pathlib
import statistics as st
import sys

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from beatsaber_automapper.evaluation import alignment, scorecard  # noqa: E402
from calibrate_playfeel import load_expert_only  # noqa: E402
from eval_coincidence import events_for  # noqa: E402

OUTER = 0.35  # |phase| >= OUTER * beat counts as displaced


def phase_metrics(beatmap, bpm: float, ev, kmin: int = 3) -> dict | None:
    times, ks = ev
    beat = 60.0 / bpm
    notes = np.sort(np.asarray(alignment.note_times(beatmap, bpm), dtype=np.float64))
    if len(notes) < 100:
        return None
    e = times[ks >= kmin]
    if len(e) < 50:
        return None
    i = np.searchsorted(notes, e).clip(1, len(notes) - 1)
    # signed offset from the event to whichever neighbouring note is nearer
    d = np.where(np.abs(e - notes[i - 1]) < np.abs(e - notes[i]),
                 e - notes[i - 1], e - notes[i])
    ph = np.abs((d + beat / 2) % beat - beat / 2)
    return {"halfbeat_rate": round(float(np.mean(ph >= OUTER * beat)), 4),
            "phase_median": round(float(np.median(ph)), 4),
            "on_event_rate": round(float(np.mean(np.abs(d) <= 0.05)), 4),
            "n_events": int(len(e))}


def scan(paths, loader, label: str, kmin: int) -> list[dict]:
    rows = []
    for p in paths:
        pp = pathlib.Path(p)
        ev = events_for(scorecard.song_id(pp), 0.030)
        if ev is None:
            continue
        try:
            L = loader(pp)
        except Exception:  # noqa: BLE001
            continue
        if not L:
            continue
        r = phase_metrics(L[0], L[1], ev, kmin)
        if r:
            r["map"], r["song"] = pp.name, scorecard.song_id(pp)
            rows.append(r)
    print(f"{label}: {len(rows)} maps scored")
    return rows


def report(rows, label) -> dict:
    if not rows:
        return {}
    print(f"\n=== {label} (n={len(rows)}) ===")
    out = {"n": len(rows)}
    for k in ("halfbeat_rate", "phase_median", "on_event_rate"):
        v = [r[k] for r in rows]
        out[k] = {"median": round(st.median(v), 4),
                  "p10": round(float(np.percentile(v, 10)), 4),
                  "p90": round(float(np.percentile(v, 90)), 4)}
        print(f"  {k:15s} median {st.median(v):7.4f}   p10 {np.percentile(v,10):7.4f}"
              f"   p90 {np.percentile(v,90):7.4f}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gen", default="outputs/eval_sweep_cache/tf_trim_ev03_rc05#s*__*.zip")
    ap.add_argument("--human-n", type=int, default=200)
    ap.add_argument("--kmin", type=int, default=3)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()

    cached = {p.stem for p in (REPO / "outputs" / "stem_onset_cache").glob("*.npz")}
    human = [p for p in sorted((REPO / "data" / "raw").glob("*.zip"))
             if p.stem in cached][:a.human_n]

    g = scan(sorted(glob.glob(a.gen)), scorecard._load_any, "ours", a.kmin)
    h = scan(human, load_expert_only, "human", a.kmin)
    out = {"ours": report(g, "OURS"), "human": report(h, "HUMAN (strict Expert)")}

    if out["ours"] and out["human"]:
        o = out["ours"]["halfbeat_rate"]["median"]
        hh = out["human"]["halfbeat_rate"]["median"]
        print("\n=== READ ===")
        print(f"  halfbeat_rate  ours {o:.4f}   human {hh:.4f}   ratio {o/max(hh,1e-9):.2f}x")
        print("  Humans syncopate too, so the target is the human cohort, NOT zero.")
        print("  A lever fixing this must move halfbeat_rate toward the human median")
        print("  WITHOUT lowering on_event_rate -- otherwise it merely deleted the")
        print("  offbeat notes instead of moving them onto the event, which is the")
        print("  'fixed it by making everything smaller' failure BEAT_REACH was")
        print("  pre-registered against. Check both numbers, always.")

    if a.json:
        out["ours_rows"], out["human_rows"] = g, h
        pathlib.Path(a.json).write_text(json.dumps(out, indent=2))
        print(f"\nwrote {a.json}")


if __name__ == "__main__":
    main()
