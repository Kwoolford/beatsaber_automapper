#!/usr/bin/env python
"""Do dilution-resistant pooling statistics beat the current gate on `offbeat`?

Three candidates are already eliminated (per-metric bound; per-axis mean; per-axis
max) and the dilution is quantified: `onset_precision` alone separates `offbeat` at
**AUC 0.898** while the best existing gate term (`s_max`) manages **0.732**. The signal
the aggregate throws away is that `offbeat` has **two simultaneously-extreme RELATED
metrics** where a human has zero or one random one.

**This tests the two statistics designed for exactly that shape:**

  fisher   -2*sum(log p_i) -- pools evidence multiplicatively, so several moderate
           signals compound instead of being averaged away.
  hc       higher criticism: max_i sqrt(n)*(i/n - p_(i)) / sqrt(p_(i)(1-p_(i))) over
           the sorted p-values. Built to detect a FEW strong signals among many
           nulls, which is this defect exactly.

★**Calibrated, not assumed.** The metrics are correlated, so neither statistic's
textbook null holds here. Each is calibrated empirically on a **held-out half** of the
human maps; the other half measures human accept. ⚠️Same split discipline as the
per-axis probe -- fitting and evaluating on one set reports a threshold to itself.

★**The honesty check that killed the last candidate**: a winner must beat the current
gate on **every** control, not just `offbeat`. Anything that rescues only the control
I built to expose this defect has been fitted to it.

⚠️Measures candidates; ships nothing. What a PASS means is Kyle's decision.
"""
from __future__ import annotations

import argparse
import math
import pathlib
import random
import sys

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

EPS = 1e-6


def per_metric_p(res) -> list[float]:
    """Two-sided p per metric: how unusual its percentile is, in [0,1]."""
    out = []
    for m in res.metrics:
        if m.pct is None or m.pct != m.pct:
            continue
        out.append(max(EPS, min(1.0, 2.0 * min(m.pct, 1.0 - m.pct))))
    return out


def fisher(ps: list[float]) -> float:
    return -2.0 * sum(math.log(max(p, EPS)) for p in ps) if ps else 0.0


def higher_criticism(ps: list[float]) -> float:
    if not ps:
        return 0.0
    s = sorted(ps)
    n = len(s)
    best = 0.0
    for i, p in enumerate(s, start=1):
        den = math.sqrt(max(p * (1.0 - p), EPS))
        best = max(best, math.sqrt(n) * (i / n - p) / den)
    return best


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--alpha", type=float, default=0.10)
    a = ap.parse_args()

    from beatsaber_automapper.evaluation import mapjudge as mj
    import audit_mapjudge as A
    from calibrate_mapjudge import CORPUS_OFFSET, corpus

    ref = mj.load_reference()
    raws = corpus(0)[CORPUS_OFFSET:]
    span = int(1100 * 1.25) + 40
    held = raws[2 * span:3 * span]
    controls = dict(A.CONTROLS)
    controls.update(A.EXTRA_CONTROLS)
    rng = random.Random(0)

    stats = {"fisher": {}, "hc": {}}
    cur = {}
    for c in ["human"] + list(controls):
        stats["fisher"][c] = []
        stats["hc"][c] = []
        cur[c] = []

    scored = 0
    for zp in held:
        if scored >= a.n:
            break
        got = A._load_human(zp)
        if not got:
            continue
        notes, bpm = got
        if len(notes) < 50 or not (30.0 < bpm < 400.0):
            continue
        f = REPO / "outputs" / "onset_cache" / f"{zp.stem}.npz"
        if not f.exists():
            continue
        z = np.load(f)
        on = np.asarray(z[list(z.keys())[0]], dtype=float)
        scored += 1
        variants = {"human": notes}
        for cn, fn in controls.items():
            variants[cn] = fn(list(notes), rng)
        for vn, vnotes in variants.items():
            r = mj.judge(mj.map_record(vnotes, bpm, onsets=on), ref, label=vn)
            ps = per_metric_p(r)
            stats["fisher"][vn].append(fisher(ps))
            stats["hc"][vn].append(higher_criticism(ps))
            cur[vn].append(r.p_value)

    idx = list(range(len(stats["fisher"]["human"])))
    random.Random(1).shuffle(idx)
    cut = len(idx) // 2
    fit_i, ev_i = idx[:cut], idx[cut:]

    print(f"\nscored {scored} human maps x {1+len(controls)} variants   "
          f"alpha={a.alpha}   fitted on {len(fit_i)}, evaluated on {len(ev_i)}\n")
    names = ["human"] + list(controls)
    print(f"{'cohort':<15}{'fisher':>10}{'hc':>10}{'current':>10}")
    print("-" * 45)
    thr = {}
    for k in ("fisher", "hc"):
        vals = sorted(stats[k]["human"][i] for i in fit_i)
        # reject above the (1-alpha) quantile of the human statistic
        thr[k] = vals[min(len(vals) - 1, int((1 - a.alpha) * len(vals)))]
    for c in names:
        row = []
        for k in ("fisher", "hc"):
            v = stats[k][c]
            if c == "human":
                v = [v[i] for i in ev_i]
            row.append(sum(1 for x in v if x <= thr[k]) / max(len(v), 1))
        cu = cur[c]
        if c == "human":
            cu = [cu[i] for i in ev_i]
        row.append(sum(1 for p in cu if p >= a.alpha) / max(len(cu), 1))
        flag = "  ←" if c == "offbeat" else ""
        print(f"{c:<15}" + "".join(f"{x:>10.3f}" for x in row) + flag)
    print("\n★A winner needs human >=0.85, every control <=0.10, and must beat the "
          "current gate on ALL controls — not only `offbeat`.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
