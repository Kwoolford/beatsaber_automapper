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

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[1]
AM = REPO / "agent_mapper"
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

WATCH = ("onset_precision", "offset_mad_ms", "pulse_stability", "role_asymmetry", "nps")


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

    arms = {"BASE": [], "SNAP": ["--snap-onsets"]}
    sids = a.songs or [p.stem for p in
                       sorted((REPO / "data" / "eval_songset").glob("*.ogg"))]
    tmp = pathlib.Path(tempfile.mkdtemp(prefix="snap_"))
    res = {k: [] for k in arms}
    for sid in sids:
        audio = REPO / "data" / "eval_songset" / f"{sid}.ogg"
        on = onsets_for(sid)
        for arm, extra in arms.items():
            out = tmp / f"{arm}__{sid}.zip"
            subprocess.run(
                [sys.executable, str(AM / "autobuild.py"), str(audio), "--pulse",
                 "--lead-bias", "0.3", "--name", f"sn_{sid}_{arm}",
                 "--out", str(out), *extra],
                capture_output=True, text=True, cwd=REPO)
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
