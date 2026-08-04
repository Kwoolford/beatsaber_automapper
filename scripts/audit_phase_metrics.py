#!/usr/bin/env python
"""Control battery for the 2026-08-03 metrics — MUST pass before either steers anything.

Standing rule in this project since `h_dist`: a metric earns the right to select
a lever only after degenerate control maps show it discriminates. This runs the
same controls as `audit_eval_suite.py` against the two new diagnostics:

    halfbeat_rate    (eval_beat_phase.py)   share of k>=3 multi-instrument events
                                            whose nearest note is a half beat off
    coincidence lift (eval_coincidence.py)  P(hit | k>=3) / P(hit | k==1)
    share_over_1s    (eval_phrase_abandon)  share of sung phrases holding a >1s
                                            stretch with no notes

RESULTS (2026-08-04, n=12, this script's own cohort): BOTH halfbeat_rate AND
share_over_1s FAIL, for the SAME reason -- a METRONOME beats a human on each:

    halfbeat_rate    metronome 0.0362   human 0.0843
    share_over_1s    metronome 0.2000   human 0.2500
    (timing_random correctly fails both: 0.1944 and 0.6667)

A constant pulse covers the beat grid densely AND never leaves a hole, so
"minimise this metric" is satisfiable by becoming the for-sport degenerate.

★THE STRUCTURAL POINT: every metric that rewards REGULARITY is gameable by the
metronome, and both 2026-08-03/04 metrics reward regularity. Both stay valid as
DIAGNOSTICS against human maps at matched density -- which is all they have been
used for -- but neither may select a lever alone. Any lever aimed at either must
carry a metronome guard (rhythm A2 / pulse_stability) scored alongside it.

**Axis-aware expectation, declared BEFORE running** — the same reasoning A8 and
A2 are already audited under. Both metrics read only note TIMES, so:

  MUST discriminate (these move note times):
      metronome        constant interval, no relation to the music
      timing_random    note times randomised
      timing_jitter    note times perturbed
  BLIND BY CONSTRUCTION (these permute positions and keep human times):
      random, shuffled, zigzag

A metric that scored the timing controls as human-like would be useless. A metric
that "caught" the position-only controls would be measuring something other than
what it claims. Both outcomes are failures; only the declared pattern is a pass.

Usage:
    python scripts/audit_phase_metrics.py --n 12
"""

from __future__ import annotations

import argparse
import copy
import json
import pathlib
import random
import statistics as st
import sys

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from audit_eval_suite import CONTROLS  # noqa: E402
from calibrate_playfeel import load_expert_only  # noqa: E402
from eval_beat_phase import OUTER  # noqa: E402
from eval_coincidence import events_for  # noqa: E402
from eval_phrase_abandon import phrase_silence, vocal_phrases  # noqa: E402

TOL = 0.050
TIMING_CONTROLS = ("metronome", "timing_random", "timing_jitter")
POSITION_CONTROLS = ("random", "shuffled", "zigzag")


def metrics_from_times(notes, bpm: float, ev, song_id: str = "") -> dict | None:
    times, ks = ev
    beat = 60.0 / bpm
    spb = 60.0 / bpm
    nt = np.sort(np.unique(np.round([n.beat * spb for n in notes], 4)))
    if len(nt) < 100:
        return None

    def nearest_signed(t):
        i = int(np.searchsorted(nt, t))
        c = [nt[j] for j in (i - 1, i) if 0 <= j < len(nt)]
        return min(c, key=lambda x: abs(t - x)) - t if c else np.inf

    vph = vocal_phrases(song_id, 1.2, 2.0) if song_id else []
    e3 = times[ks >= 3]
    e1 = times[ks == 1]
    if len(e3) < 50 or len(e1) < 20:
        return None

    d3 = np.array([nearest_signed(t) for t in e3])
    hit3 = np.mean(np.abs(d3) <= TOL)
    hit1 = np.mean([abs(nearest_signed(t)) <= TOL for t in e1])
    out = {"halfbeat_rate": float(np.mean(np.abs((d3 + beat / 2) % beat - beat / 2)
                                          >= OUTER * beat)),
           "lift": float(hit3 / hit1) if hit1 > 0 else float("nan")}
    if vph:
        sil = phrase_silence(nt, vph)
        if sil:
            out["share_over_1s"] = sil["share_over_1s"]
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()

    cached = {p.stem for p in (REPO / "outputs" / "stem_onset_cache").glob("*.npz")}
    zips = [p for p in sorted((REPO / "data" / "raw").glob("*.zip")) if p.stem in cached]

    cohorts: dict[str, list[dict]] = {"human": []}
    for name in list(CONTROLS):
        cohorts[name] = []

    rng = random.Random(0)
    used = 0
    for zp in zips:
        if used >= a.n:
            break
        ev = events_for(zp.stem, 0.030)
        if ev is None:
            continue
        L = load_expert_only(zp)
        if not L:
            continue
        bm, bpm = L[0], float(L[1])
        base = metrics_from_times(bm.color_notes, bpm, ev, zp.stem)
        if not base:
            continue
        cohorts["human"].append(base)
        for name, fn in CONTROLS.items():
            try:
                ctrl = fn(copy.deepcopy(bm.color_notes), rng)
                m = metrics_from_times(ctrl, bpm, ev, zp.stem)
            except Exception:  # noqa: BLE001
                m = None
            if m:
                cohorts[name].append(m)
        used += 1
        print(f"  {zp.stem}: scored")

    print(f"\n=== CONTROL BATTERY (n={used} songs) ===")
    print(f"{'cohort':16s}{'halfbeat_rate':>16s}{'lift':>10s}{'share_over_1s':>16s}")
    med = {}
    for name, rows in cohorts.items():
        if not rows:
            print(f"{name:16s}{'(none)':>16s}")
            continue
        hb = st.median([r["halfbeat_rate"] for r in rows])
        lf = st.median([r["lift"] for r in rows if r["lift"] == r["lift"]])
        sv = [r["share_over_1s"] for r in rows if "share_over_1s" in r]
        so = st.median(sv) if len(sv) >= 4 else float("nan")
        med[name] = (hb, lf, so)
        print(f"{name:16s}{hb:16.4f}{lf:10.4f}{so:16.4f}")

    if "human" not in med:
        sys.exit("no human baseline scored")
    h_hb, h_lf, h_so = med["human"]

    print("\n=== VERDICT (expectations declared in the docstring, before the run) ===")
    ok = True
    for c in TIMING_CONTROLS:
        if c not in med:
            continue
        worse_hb = med[c][0] > h_hb
        worse_lf = med[c][1] < h_lf
        flag = "PASS" if (worse_hb and worse_lf) else "FAIL"
        ok &= (worse_hb and worse_lf)
        print(f"  {flag}  {c:14s} halfbeat {med[c][0]:.4f} vs human {h_hb:.4f}"
              f" | lift {med[c][1]:.3f} vs human {h_lf:.3f}   (must be worse on BOTH)")
    for c in POSITION_CONTROLS:
        if c not in med:
            continue
        same = abs(med[c][0] - h_hb) < 1e-6 and abs(med[c][1] - h_lf) < 1e-6
        print(f"  {'PASS' if same else 'NOTE'}  {c:14s} identical to human: {same}"
              f"   (blind BY CONSTRUCTION — these keep human note times)")

    if h_so == h_so:
        print("\n  share_over_1s (W4):")
        for c in ("metronome", "timing_random"):
            if c in med and med[c][2] == med[c][2]:
                worse = med[c][2] > h_so
                print(f"    {'PASS' if worse else 'FAIL'}  {c:14s} {med[c][2]:.4f} vs human {h_so:.4f}")
                ok &= worse
    print(f"\n  OVERALL: {'PASS — the metrics may steer levers' if ok else 'FAIL — do not steer with these'}")

    if a.json:
        pathlib.Path(a.json).write_text(json.dumps(
            {k: {"halfbeat_rate": v[0], "lift": v[1]} for k, v in med.items()}, indent=2))
        print(f"  wrote {a.json}")


if __name__ == "__main__":
    main()
