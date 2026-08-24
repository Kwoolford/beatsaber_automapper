#!/usr/bin/env python
"""Does reconciling the two onset detectors actually move `onset_precision`?

P0.7's DoD: the share of placed events within 50 ms of a SCORED onset rises above
0.97 and `onset_precision` moves with it. If precision does NOT move once the
detectors agree, the remaining gap is genuine note selection and the reconciliation
was the wrong fix.

★Reports the whole judged picture, not just the target metric — a change that buys
alignment by wrecking rhythm is not a fix, and `nps` guards that the snap did not
quietly change the note budget.

Usage:
    python scripts/sweep_snap.py
"""
from __future__ import annotations

import argparse
import pathlib
import statistics
import subprocess
import sys
import tempfile
import time

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[1]
AM = REPO / "agent_mapper"
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

WATCH = ("onset_precision", "offset_mad_ms", "pulse_stability", "nps",
         # ★Cut-direction axes added 2026-08-24. The snap cost 2 of 23 songs their
         # PASS, and BOTH failed on idiom/geometry while `onset_precision` ROSE --
         # `diagonal_share` at the 98th-99.8th human percentile, `vertical_share`
         # at the 3rd. That is the P1.2 direction (we sit vertical 0.773 / diagonal
         # 0.223 vs human 0.480 / 0.415) and P1.2 records the loss as living
         # "inside the sampler and NOT yet explained". Two songs propose; only the
         # cohort disposes -- this is here to make the cohort answer.
         "vertical_share", "diagonal_share", "idiom_coverage")


def onsets_for(sid):
    f = REPO / "outputs" / "onset_cache" / f"{sid}.npz"
    if not f.exists():
        return None
    z = np.load(f)
    return z[list(z.keys())[0]]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--songs", nargs="*", default=None)
    a = ap.parse_args()
    from beatsaber_automapper.evaluation import mapjudge as mj
    ref = mj.load_reference()

    # ⚠️`SNAP+SUB` retired: `--adaptive-subdiv` is REFUTED (`onset_precision` falls on
    # 10 of 10 affected songs, `pulse_stability` 0.591 -> 0.376), so keeping it here
    # spent a third of the sweep re-measuring a dead arm.
    arms = {"BASE": [], "SNAP": ["--snap-onsets"]}
    sids = a.songs or [p.stem for p in
                       sorted((REPO / "data" / "eval_songset").glob("*.ogg"))]
    tmp = pathlib.Path(tempfile.mkdtemp(prefix="snap_"))
    res = {k: [] for k in arms}
    times = {k: [] for k in arms}
    for sid in sids:
        audio = REPO / "data" / "eval_songset" / f"{sid}.ogg"
        on = onsets_for(sid)
        for arm, extra in arms.items():
            out = tmp / f"{arm}__{sid}.zip"
            t0 = time.time()
            subprocess.run(
                # ★`--lead-bias 0.20`, not 0.30: 0.30 was tuned against the SAMPLED
                # lead and overshoots to the 77.9th percentile under the `cyclic`
                # default. An operating point is not portable across a change in how
                # the knob works.
                [sys.executable, str(AM / "autobuild.py"), str(audio), "--pulse",
                 "--lead-bias", "0.2", "--name", f"sn_{sid}_{arm}",
                 "--out", str(out), *extra],
                capture_output=True, text=True, cwd=REPO)
            times[arm].append(time.time() - t0)
            if out.exists():
                try:
                    res[arm].append(mj.judge_zip(out, onsets=on, reference=ref))
                except Exception:  # noqa: BLE001
                    pass
    print(f"\n{'arm':<7}{'PASS':>8}{'p med':>8}" + "".join(f"{k[:14]:>17}" for k in WATCH))
    print("-" * 100)
    for arm in arms:
        rs = res[arm]
        if not rs:
            continue
        npass = sum(1 for r in rs if r.verdict() == "PASS")
        pmed = statistics.median([r.p_value for r in rs])
        cells = []
        for k in WATCH:
            vals = [(m.value, m.pct) for r in rs for m in r.metrics if m.name == k]
            if vals:
                v = statistics.median([x[0] for x in vals])
                pc = statistics.median([x[1] for x in vals if x[1] is not None])
                cells.append(f"{v:7.3f} ({100*pc:.0f}%)")
            else:
                cells.append("--")
        print(f"{arm:<7}{npass:>4}/{len(rs):<3}{pmed:>8.3f}"
              + "".join(f"{c:>17}" for c in cells))

    print(f"\n{'arm':<7}{'build s/song (median)':>24}")
    for arm in arms:
        if times[arm]:
            print(f"{arm:<7}{statistics.median(times[arm]):>24.1f}")
    if times["BASE"] and times["SNAP"]:
        d = statistics.median(times["SNAP"]) - statistics.median(times["BASE"])
        note = ("corpus cache is warm here, so this is the FLOOR" if d >= 0
                else "inside run-to-run noise")
        print(f"\nsnap costs {d:+.1f} s/song ({note})")

    print("\nVERDICT LOGIC — should --snap-onsets become the DEFAULT?")
    print("  The 2026-08-20 cost was `nps` 3.43 -> 3.29, but that predates the")
    print("  two-stream carrier fix, so it is re-measured here at current defaults.")
    print("  DEFAULT-WORTHY if: onset_precision rises, PASS count does not fall, and")
    print("     the nps cost is smaller than the density lever can trivially repay.")
    print("  NOT DEFAULT-WORTHY if: nps falls materially, or PASS falls -- the snap")
    print("     COLLAPSES events sharing an onset, and that is a note-budget change")
    print("     wearing an alignment change's clothes.")
    print("  ⚠️Build-time above is the CORPUS floor. A non-corpus song additionally")
    print("     pays a full 4-stem Demucs pass, which this sweep cannot see.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
