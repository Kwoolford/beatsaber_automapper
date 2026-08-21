#!/usr/bin/env python
"""Would a PER-AXIS conformal gate reject `offbeat` without failing humans?

The per-METRIC rule is already ruled out (`probe_axis_gate.py`): with 23 metrics almost
every human map has one in a tail, so any bound tight enough to catch a real defect
rejects most humans -- an uncorrected multiple test over 23 hypotheses.

**The candidate here is the shape that finding pointed at.** Group the 23 metrics into
their six evaluation axes, give each axis its OWN conformal score calibrated on human
maps, and reject if any axis is extreme at a **Bonferroni-corrected** level. Six
hypotheses instead of twenty-three, each with a known false-rejection rate.

★**Why this can work where the per-metric rule cannot**: alignment is 2 metrics out of
23 and is drowned in the aggregate, but it is 1 axis out of 6 and cannot be drowned in
a per-axis minimum. **No metric is privileged by hand** -- alignment gets a voice
because it IS an axis, not because it was given a weight.

★**Honest calibration**: the human held-out maps are SPLIT -- one half sets each axis'
threshold, the other half measures human accept. Fitting and evaluating on the same
maps would report the threshold back to itself.

⚠️Measures a candidate; does not ship it. What a PASS means is Kyle's call.
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

AXES = {
    "flow":      ["angle_change", "angle_harsh_frac", "travel", "ebpm_burst",
                  "crossover", "handedness"],
    "rhythm":    ["pulse_stability", "ioi_cond_entropy", "ioi_switch_rate",
                  "dominant_share", "offgrid_frac"],
    "idiom":     ["idiom_coverage", "idiom_top50", "idiom_jsd", "idiom_local"],
    "handrole":  ["role_asymmetry", "role_swap_rate"],
    "playfeel":  ["nps", "peak_nps", "vertical_share", "diagonal_share"],
    "alignment": ["onset_precision", "offset_mad_ms"],
}


def axis_scores(res, agg: str = "max") -> dict[str, float]:
    """Nonconformity per axis (nan-safe); axes with no data are absent.

    ⚠️**`mean` DILUTES and was measured doing so.** Alignment is a 2-metric axis, so a
    map extreme on `onset_precision` alone has that halved -- the very drowning the
    per-axis idea exists to prevent, reintroduced one level down. Measured: with
    `mean` the per-axis gate is WORSE than the current one on every control
    (`offbeat` 0.800 vs 0.650). `max` keeps the extreme metric visible inside its axis.
    ★This is a structural choice between two aggregations, NOT a weight tuned per
    metric -- and the project already knows gating and ranking want different
    statistics (`max` saturates for ranking, which is why `s_mean` ranks).
    """
    u = {m.name: m.u for m in res.metrics if m.u == m.u}
    out = {}
    for ax, names in AXES.items():
        vals = [u[n] for n in names if n in u]
        if vals:
            out[ax] = max(vals) if agg == "max" else sum(vals) / len(vals)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--alpha", type=float, default=0.10)
    ap.add_argument("--agg", choices=("max", "mean"), default="max",
                    help="how metrics combine WITHIN an axis")
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

    human_axis: list[dict] = []
    ctrl_axis: dict[str, list[dict]] = {c: [] for c in controls}
    agg_human: list[float] = []
    agg_ctrl: dict[str, list[float]] = {c: [] for c in controls}

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

        res = mj.judge(mj.map_record(notes, bpm, onsets=on), ref, label="human")
        human_axis.append(axis_scores(res, a.agg))
        agg_human.append(res.p_value)
        for cname, fn in controls.items():
            r2 = mj.judge(mj.map_record(fn(list(notes), rng), bpm, onsets=on), ref,
                          label=cname)
            ctrl_axis[cname].append(axis_scores(r2, a.agg))
            agg_ctrl[cname].append(r2.p_value)

    # ---- split the humans: half calibrates the thresholds, half is evaluated ----
    idx = list(range(len(human_axis)))
    random.Random(1).shuffle(idx)
    cut = len(idx) // 2
    fit_i, ev_i = idx[:cut], idx[cut:]
    K = len(AXES)
    bar = 1.0 - (a.alpha / K)          # Bonferroni over the six axes

    thresh = {}
    for ax in AXES:
        vals = sorted(human_axis[i][ax] for i in fit_i if ax in human_axis[i])
        if len(vals) >= 20:
            thresh[ax] = vals[min(len(vals) - 1, int(bar * len(vals)))]
    if not thresh:
        print("no axis had enough calibration data")
        return 1

    def rejects(d: dict) -> bool:
        return any(ax in d and d[ax] > th for ax, th in thresh.items())

    print(f"\nscored {scored} human maps x {1+len(controls)} variants   "
          f"axes={K}  agg={a.agg}  alpha={a.alpha}  per-axis bar={bar:.4f}")
    print(f"thresholds fitted on {len(fit_i)} humans, evaluated on {len(ev_i)}\n")
    h_acc = sum(1 for i in ev_i if not rejects(human_axis[i])) / max(len(ev_i), 1)
    print(f"{'cohort':<14}{'per-axis accept':>17}{'current gate':>15}")
    print("-" * 48)
    print(f"{'human':<14}{h_acc:>17.3f}"
          f"{sum(1 for p in agg_human if p >= a.alpha)/max(len(agg_human),1):>15.3f}")
    for c in controls:
        acc = sum(1 for d in ctrl_axis[c] if not rejects(d)) / max(len(ctrl_axis[c]), 1)
        cur = sum(1 for p in agg_ctrl[c] if p >= a.alpha) / max(len(agg_ctrl[c]), 1)
        flag = "  ←" if c == "offbeat" else ""
        print(f"{c:<14}{acc:>17.3f}{cur:>15.3f}{flag}")
    ok_h = h_acc >= 0.85
    ok_c = all(sum(1 for d in ctrl_axis[c] if not rejects(d)) / max(len(ctrl_axis[c]), 1)
               <= 0.10 for c in controls)
    print(f"\nDoD (human >=0.85 AND every control <=0.10): "
          f"{'MET' if ok_h and ok_c else 'NOT MET'}")
    print("⚠️A candidate measurement, not a shipped gate.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
