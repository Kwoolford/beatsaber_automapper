#!/usr/bin/env python
"""Re-check P0.6's lead-hand result now that the seed actually reaches its RNG.

`--seed` used to reach `idiomize` only, never `mapctl auto`, so the lead-hand RNG ran
at seed 0 no matter what was asked. P0.6's headline -- `role_asymmetry` from the 1.3rd
to the 39.6th human percentile -- was therefore a **single-seed** reading of a
stochastic knob, on a 23-song cohort. This samples the variance that was never sampled.

★**Reports the per-seed SPREAD, not just the median.** The project's own standing rule
is that an axis moving more between seeds of one arm than between arms is not a
result; that rule was written for the ML path and has never been applied here, because
until now the seed could not move a hand-role metric at all.
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

WATCH = ("role_asymmetry", "role_swap_rate", "handedness")


def onsets_for(sid):
    f = REPO / "outputs" / "onset_cache" / f"{sid}.npz"
    if not f.exists():
        return None
    z = np.load(f)
    return z[list(z.keys())[0]]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--bias", nargs="+", type=float, default=[0.0, 0.3])
    a = ap.parse_args()
    from beatsaber_automapper.evaluation import mapjudge as mj
    ref = mj.load_reference()

    sids = [p.stem for p in
            sorted((REPO / "data" / "eval_songset").glob("*.ogg"))][: a.n]
    tmp = pathlib.Path(tempfile.mkdtemp(prefix="leadseed_"))
    # per (bias, seed) -> list over songs
    cell: dict[tuple, dict[str, list]] = {}
    for sid in sids:
        audio = REPO / "data" / "eval_songset" / f"{sid}.ogg"
        on = onsets_for(sid)
        for b in a.bias:
            for sd in a.seeds:
                out = tmp / f"b{b}_s{sd}__{sid}.zip"
                subprocess.run(
                    [sys.executable, str(AM / "autobuild.py"), str(audio), "--pulse",
                     "--lead-bias", str(b), "--seed", str(sd),
                     "--name", f"ls_{sid}_{str(b).replace('.','')}_{sd}",
                     "--out", str(out)],
                    capture_output=True, text=True, cwd=REPO)
                if not out.exists():
                    continue
                try:
                    r = mj.judge_zip(out, onsets=on, reference=ref)
                except Exception:  # noqa: BLE001
                    continue
                d = cell.setdefault((b, sd), {k: [] for k in WATCH})
                for m in r.metrics:
                    if m.name in WATCH and m.pct is not None:
                        d[m.name].append(m.pct)

    print(f"\nsongs {len(sids)}   seeds {a.seeds}   (percentile medians over songs)")
    print(f"\n{'metric':<18}{'bias':>6}" + "".join(f"{'seed ' + str(s):>10}"
                                                   for s in a.seeds)
          + f"{'spread':>9}{'arm gap':>10}")
    print("-" * 78)
    for k in WATCH:
        per_bias = {}
        for b in a.bias:
            vals = []
            for sd in a.seeds:
                v = cell.get((b, sd), {}).get(k) or []
                vals.append(statistics.median(v) if v else float("nan"))
            per_bias[b] = vals
        for b in a.bias:
            vals = per_bias[b]
            good = [v for v in vals if v == v]
            spread = (max(good) - min(good)) if len(good) > 1 else float("nan")
            print(f"{k if b == a.bias[0] else '':<18}{b:>6.2f}"
                  + "".join(f"{100*v:>10.1f}" for v in vals)
                  + f"{100*spread:>9.1f}", end="")
            if b == a.bias[-1] and len(a.bias) == 2:
                g0 = [v for v in per_bias[a.bias[0]] if v == v]
                g1 = [v for v in per_bias[a.bias[1]] if v == v]
                if g0 and g1:
                    gap = statistics.median(g1) - statistics.median(g0)
                    sp = max(max(g0) - min(g0), max(g1) - min(g1))
                    print(f"{100*gap:>10.1f}", end="")
                    print(f"   {'✅ gap > spread' if abs(gap) > sp else '🔴 GAP INSIDE SEED SPREAD'}",
                          end="")
            print()
    print("\n★An arm gap smaller than the seed spread of one arm is not a result.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
