#!/usr/bin/env python
"""Could an AXIS-AWARE gate reject the `offbeat` control without failing humans?

**The problem** (PROGRESS 2026-08-21): `mapjudge` accepts **65 %** of maps shifted a
quarter-beat off the music. The alignment axis sees them perfectly -- `onset_precision`
AUC 0.898 while 21 of 23 metrics score exactly 0.500 -- but the verdict is a
mean/topk/max over all 23, and two moving among twenty-one silent does not shift it.

**This measures a candidate rule; it does not ship one.** The rule tested is
deliberately **metric-agnostic**:

    reject if ANY single metric sits beyond the q-th percentile of the human
    distribution, two-sided (below 1-q or above q)

🔴**Privileging `onset_precision` was rejected as a design**: tuning a weight until this
one control fails fits the gate to the control, which is the `h_dist` mistake. A rule
that names no metric cannot be fitted to one, so if it works it works for the right
reason -- and if it fails humans, that is the honest answer that an axis-aware gate
costs more than it buys.

⚠️**The threshold is swept, not chosen.** The output is a trade-off curve: human accept
against control accept at each q. Picking a point on it is Kyle's call, because it
changes what a PASS *means*.

Usage:
    python scripts/probe_axis_gate.py --n 120
"""

# 🔴🔴**PARITY IS PART OF THE VERDICT AND THESE PROBES ONCE IGNORED IT.**
# `JudgeResult.verdict()` returns FAIL on `viol > 0` BEFORE it ever looks at the
# p-value, so an accept rule of `p >= alpha` overstates acceptance for any cohort that
# breaks parity. It made `shuffled` look like it passed 20 % of the time when the real
# gate rejects 100 % of it (97.5 % of shuffled maps have violations), and that was
# filed as a defect before being caught. ★Cohorts with viol≈0 (`human`, `offbeat`,
# `timing_jitter`) are unaffected, which is why the P0.2 conclusions survived.

from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--q", nargs="+", type=float,
                    default=[0.990, 0.995, 0.998, 0.999])
    a = ap.parse_args()

    from beatsaber_automapper.evaluation import mapjudge as mj
    import audit_mapjudge as A

    ref = mj.load_reference()
    raws = A.corpus(0)[A.CORPUS_OFFSET:] if hasattr(A, "corpus") else None
    if raws is None:
        from calibrate_mapjudge import CORPUS_OFFSET, corpus
        raws = corpus(0)[CORPUS_OFFSET:]
    span = int(1100 * 1.25) + 40
    held = raws[2 * span:3 * span]

    controls = dict(A.CONTROLS)
    controls.update(A.EXTRA_CONTROLS)
    rng = __import__("random").Random(0)

    # cohort -> list of "worst two-sided percentile distance from 0.5" per map
    worst: dict[str, list[float]] = {"human": []}
    for c in controls:
        worst[c] = []

    scored = 0
    for zp in held:
        if scored >= a.n:
            break
        loaded = A._load_human(zp)
        if loaded is None:
            continue
        notes, bpm = loaded
        if len(notes) < 50 or not (30.0 < bpm < 400.0):
            continue
        f = REPO / "outputs" / "onset_cache" / f"{zp.stem}.npz"
        if not f.exists():
            continue
        z = np.load(f)
        on = np.asarray(z[list(z.keys())[0]], dtype=float)
        scored += 1

        variants = {"human": notes}
        for cname, fn in controls.items():
            variants[cname] = fn(list(notes), rng)
        for vname, vnotes in variants.items():
            rec = mj.map_record(vnotes, bpm, onsets=on)
            res = mj.judge(rec, ref, label=vname)
            # two-sided extremeness of the single most extreme metric
            ex = [abs((m.pct or 0.5) - 0.5) * 2 for m in res.metrics if m.pct is not None]
            worst[vname].append(max(ex) if ex else 0.0)

    print(f"\nscored {scored} held-out human maps x {1+len(controls)} variants\n")
    names = ["human"] + list(controls)
    print(f"{'q (two-sided)':<16}" + "".join(f"{n[:9]:>11}" for n in names))
    print("-" * (16 + 11 * len(names)))
    for q in a.q:
        # reject if the most extreme metric is beyond q
        row = []
        for n in names:
            v = worst[n]
            acc = sum(1 for x in v if x < q) / len(v) if v else float("nan")
            row.append(acc)
        print(f"{q:<16.3f}" + "".join(f"{r:>11.3f}" for r in row))
    print("\naccept rate per cohort. ★A usable rule needs human >=0.85 AND every "
          "control <=0.10 — including `offbeat`.")
    print("⚠️This rule names no metric, so it cannot be fitted to one control. If it "
          "cannot separate them, an axis-aware gate costs more than it buys and the "
          "honest answer is to say so.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
