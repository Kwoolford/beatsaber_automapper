#!/usr/bin/env python
"""Pick the lead-bias operating point with the JUDGE, not by eyeballing four numbers.

The raw sweep disagrees with itself: at n=23, bias 0.3 lands `role_asymmetry` on the
human (0.109 vs 0.122) while bias 0.5 lands `role_swap_rate`, `role_run_len` and
`ebpm_burst` on it. Choosing between them by staring at a table is exactly the
distance-to-median reasoning that killed `h_dist`.

`mapjudge` is now calibrated on 23 metrics including the audio axis, so it can answer
this properly: every metric becomes a two-sided percentile against 1 100 human maps,
and the verdict is conformal at n=1.

★**Reported per arm**: PASS rate, median p, and the percentiles of the metrics this
change is supposed to move (`role_*`) and the ones it could break (`ebpm_burst`,
`nps`, `onset_precision`).
🔴**`rank_score` is NOT reported and must not be optimised** -- it is a
distance-from-typical, so minimising it Goodharts toward the average map.

Also re-judges everything on the NEW 23-metric reference, which the earlier 23/23
PASS predates.

Usage:
    python scripts/sweep_lead_judged.py --bias 0 0.3 0.4 0.5
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

WATCH = ("role_asymmetry", "role_swap_rate", "ebpm_burst", "nps", "onset_precision",
         "pulse_stability")


def onsets_for(sid: str):
    f = REPO / "outputs" / "onset_cache" / f"{sid}.npz"
    if not f.exists():
        return None
    z = np.load(f)
    return z[list(z.keys())[0]]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bias", nargs="+", type=float, default=[0.0, 0.3, 0.4, 0.5])
    ap.add_argument("--songs", nargs="*", default=None)
    a = ap.parse_args()

    from beatsaber_automapper.evaluation import mapjudge as mj
    ref = mj.load_reference()
    dists = ref["distributions"]

    sids = a.songs or [p.stem for p in
                       sorted((REPO / "data" / "eval_songset").glob("*.ogg"))]
    tmp = pathlib.Path(tempfile.mkdtemp(prefix="leadjudge_"))
    res: dict[float, list] = {b: [] for b in a.bias}
    for sid in sids:
        audio = REPO / "data" / "eval_songset" / f"{sid}.ogg"
        on = onsets_for(sid)
        for b in a.bias:
            out = tmp / f"lead{b}__{sid}.zip"
            subprocess.run(
                [sys.executable, str(AM / "autobuild.py"), str(audio), "--pulse",
                 "--name", f"lj_{sid}_{str(b).replace('.', '')}",
                 "--out", str(out), "--lead-bias", str(b)],
                capture_output=True, text=True, cwd=REPO)
            if not out.exists():
                continue
            try:
                res[b].append(mj.judge_zip(out, onsets=on, reference=ref))
            except Exception as exc:  # noqa: BLE001
                print(f"  {sid} b={b}: judge failed ({exc})")

    print(f"\n{'bias':<7}{'PASS':>8}{'p med':>8}" +
          "".join(f"{k[:13]:>15}" for k in WATCH))
    print("-" * 100)
    for b in a.bias:
        rs = res[b]
        if not rs:
            continue
        npass = sum(1 for r in rs if r.verdict() == "PASS")
        pmed = statistics.median([r.p_value for r in rs])
        cells = []
        for k in WATCH:
            vals = [m.pct for r in rs for m in r.metrics if m.name == k
                    and m.pct is not None]
            cells.append(f"{100 * statistics.median(vals):.1f}%" if vals else "--")
        print(f"{b:<7.2f}{npass:>4}/{len(rs):<3}{pmed:>8.3f}" +
              "".join(f"{c:>15}" for c in cells))
    print("\npercentiles are MEDIANS over songs, two-sided: 50% is the human median, "
          "and 95% is as wrong as 5%.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
