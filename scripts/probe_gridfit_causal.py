#!/usr/bin/env python
"""Does a WORSE grid cause worse `onset_precision`, or do the two share a cause?

`corr(fit r, onset_precision) = +0.575` over 23 songs, and the poorly-fitted half
scores 0.781 against 0.882. That is the strongest predictor of map quality measured
on this project — but it is correlational. A song whose onsets are hard to FIT may
simply be a song whose events are hard to HIT, in which case improving the fit would
buy nothing.

**This separates them by degrading the grid on purpose.** Same audio, same events,
same builder, same everything — only the session's `phase` is shifted, which slides
every bar line while leaving the music untouched. If precision follows the shift, the
grid is causal. If it does not, the correlation is a shared cause and chasing tempo
fitting is wasted work.

★The shift is a fraction of a BEAT so it scales with tempo, and it is applied AFTER
`init` writes the session, so the events the planner saw are identical across arms.
"""
from __future__ import annotations

import argparse
import pathlib
import statistics
import subprocess
import sys

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[1]
AM = REPO / "agent_mapper"
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))


def onsets_for(sid):
    f = REPO / "outputs" / "onset_cache" / f"{sid}.npz"
    if not f.exists():
        return None
    z = np.load(f)
    return z[list(z.keys())[0]]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--songs", nargs="+", default=["1f333", "1f767", "1f913", "1f8d6"])
    ap.add_argument("--shifts", nargs="+", type=float, default=[0.0, 0.10, 0.25])
    a = ap.parse_args()

    from beatsaber_automapper.evaluation import mapjudge as mj
    ref = mj.load_reference()
    out: dict[float, list] = {s: [] for s in a.shifts}

    for sid in a.songs:
        audio = REPO / "data" / "eval_songset" / f"{sid}.ogg"
        on = onsets_for(sid)
        for shift in a.shifts:
            zp = pathlib.Path(f"/tmp/gf_{sid}_{shift}.zip")
            subprocess.run(
                [sys.executable, str(AM / "autobuild.py"), str(audio), "--pulse",
                 "--lead-bias", "0.2", "--phase-shift", str(shift),
                 "--name", f"gf_{sid}_{str(shift).replace('.', '')}",
                 "--out", str(zp)],
                capture_output=True, text=True, cwd=REPO)
            if not zp.exists():
                continue
            try:
                res = mj.judge_zip(zp, onsets=on, reference=ref)
                v = [m.value for m in res.metrics if m.name == "onset_precision"]
                if v:
                    out[shift].append(v[0])
            except Exception:  # noqa: BLE001
                pass

    print(f"\n{'phase shift':<14}{'onset_precision':>18}{'n':>5}")
    print("-" * 40)
    for s in a.shifts:
        if out[s]:
            print(f"{s:<14.2f}{statistics.median(out[s]):>18.3f}{len(out[s]):>5}")
    print("\n★ if precision falls with the shift, the grid is CAUSAL.")
    print("  if it is flat, fit and hit-ability share a cause and tempo work is wasted.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
