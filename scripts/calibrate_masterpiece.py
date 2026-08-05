#!/usr/bin/env python
"""THE HUMAN BAR FOR THE MASTERPIECE AXES — and the exceedance readout that uses it.

Every other axis in this suite is calibrated: A8 has a human precision of 0.930 and
a bar at 0.39, A2/A3/A6 have medians and MADs from the corpus. The M-axes shipped
with a **paired delta** and nothing else, which answers "are we below the human"
but not "by how much, and on how many songs".

This computes the human distribution of each M-axis over the wide cohort's **149
human Expert maps** — median, MAD, p10 / p25 / p75 / p90 — and reports our
**exceedance**: the share of our maps that fall below the human **p10**, i.e. below
what 90 % of human maps clear.

★**Why exceedance and not distance-to-median.** `h_dist` failed this project by
scoring each map on its distance to a corpus median, which rewards being average and
saturates. And a cohort median cannot see a subset-of-songs defect — the project's
oldest lesson. Exceedance over a human percentile keeps the tail visible and names
the songs.

⚠️This is a **norm** bar, not an aspiration bar. Kyle's target is *the best mappers*,
so for an aspirational axis the corpus median is a **floor, not a target**, and
"the human cohort passes it" is not a validity check. Ask him norm-or-aspiration
before treating any of these as a goal.

Usage:
    python scripts/calibrate_masterpiece.py --json docs/eval_references/masterpiece_human.json
"""

from __future__ import annotations

import argparse
import json
import pathlib
import statistics as st
import sys

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import masterpiece_report as mr  # noqa: E402

AXES = [k for _, k in mr.REPORT_KEYS]

# ⚠️For most axes our defect is being BELOW the human, so exceedance is the share
# under the human p10. `double_share` is the opposite — we sit ABOVE the whole human
# range (0.646 vs a p90 of 0.308) — and reading it from the low tail printed a
# meaningless 0.0 %. A tail statistic has to be taken from the tail the defect is in.
HIGHER_IS_WORSE = {"double_share"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", default="prod")
    ap.add_argument("--seed", default="s0")
    ap.add_argument("--json", default="")
    a = ap.parse_args()

    rows = mr.collect(a.arm, a.seed, rebuild=False, wide=True)
    rows = [r for r in rows if r.get("human")]
    if len(rows) < 30:
        print(f"only {len(rows)} paired songs — build the wide cohort first")
        return

    verdicts = mr.steer_verdicts()
    print(f"\n{'='*104}")
    print(f"HUMAN BAR FOR THE MASTERPIECE AXES — {len(rows)} human Expert maps (wide cohort)")
    print(f"{'='*104}")
    print(f"{'axis':<20} {'p10':>9} {'median':>9} {'p90':>9} {'MAD':>8}   "
          f"{'ours med':>9} {'outside':>10}  steer?")
    out = {}
    for k in AXES:
        h = [r["human"][k] for r in rows if r["human"].get(k) is not None]
        o = [r["ours"][k] for r in rows if r["ours"].get(k) is not None]
        pairs = [(r["ours"][k], r["human"][k]) for r in rows
                 if r["ours"].get(k) is not None and r["human"].get(k) is not None]
        if len(h) < 30 or not pairs:
            continue
        p10, med, p90 = (float(np.quantile(h, q)) for q in (0.10, 0.5, 0.90))
        mad = float(np.median([abs(x - med) for x in h]))
        if k in HIGHER_IS_WORSE:
            below = float(np.mean([ours > p90 for ours, _ in pairs]))
        else:
            below = float(np.mean([ours < p10 for ours, _ in pairs]))
        v = verdicts.get(k)
        mark = {True: "MAY STEER", False: "diagnostic"}.get(v, "unaudited")
        out[k] = {"n": len(h), "p10": round(p10, 4), "median": round(med, 4),
                  "p90": round(p90, 4), "mad": round(mad, 4),
                  "ours_median": round(st.median(o), 4),
                  "outside_human_tail": round(below, 4),
                  "tail": "above p90" if k in HIGHER_IS_WORSE else "below p10",
                  "may_steer": v}
        print(f"{k:<20} {p10:>+9.4f} {med:>+9.4f} {p90:>+9.4f} {mad:>8.4f}   "
              f"{st.median(o):>+9.4f} {below:>10.1%}  {mark}")

    print("\nHOW TO READ: `outside` is the share of OUR maps outside the human tail the")
    print("defect lives in (below p10, or above p90 where higher is worse). 10 % is what a")
    print("cohort drawn from the same population would show, so far above 10 % is a defect")
    print("with a TAIL; near 10 % with a resolvable paired delta is a SHIFT inside the")
    print("normal human range, which is a much weaker claim and a much weaker lever.")
    print("⚠️This is a NORM bar. Kyle's target is the best mappers, so on an aspirational")
    print("axis the corpus median is a FLOOR, not a target.")

    if a.json:
        p = pathlib.Path(a.json)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({"n_songs": len(rows), "cohort": "wide",
                                 "axes": out}, indent=2))
        print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
