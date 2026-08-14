#!/usr/bin/env python
"""OCTAVE DETECTION BY PERIODICITY — is there a pulse at HALF our detected beat?

**Why this exists.** Subdiv 8 lifts the half-tempo speed ceiling exactly (ebpm ratio
0.500 → 1.000 at n=28) and wrecks correctly-detected songs just as clearly
(1.000 → 2.000, precision −0.127). The whole value of the lever is now in telling the
two groups apart, so detection *is* the problem.

**Two routes are already closed:**
* 2026-07-27, `bpm_octave_probe.py` — onset-energy **balance** between on-beat and
  off-beat grid positions. Made detection WORSE (16/23 → 10/23, → 14/23): plenty of
  real music has strong backbeat asymmetry at its true tempo.
* 2026-08-14, `diag_grid_too_coarse.py` — share of onset gaps shorter than a slot.
  100 % recall at an **88.6 %** false-positive rate, because stem onsets are dense and
  the statistic measures onset DENSITY rather than grid coarseness.

★**THE REMAINING IDEA, and it is different from both**: ask about **PERIODICITY**, not
energy and not proximity. If our detected beat period is P and the music actually
pulses at P/2, then the onset train should be nearly as self-similar at lag P/2 as at
lag P. A backbeat makes alternate beats *unequal in energy* — which is what killed the
2026-07-27 statistic — but it does not stop them from RECURRING, so autocorrelation of
the onset train should still see them.

**The statistic**: `acf_ratio` = ACF(P/2) / ACF(P) on an onset-impulse train.
Near 1 ⇒ the half-period is as strong a recurrence as the beat ⇒ we are probably at
half tempo. Well below 1 ⇒ the detected beat is the real pulse.

🔴**RESULT 2026-08-14 — BEST OF THE THREE ROUTES, AND STILL NOT USABLE.** n=148:

    label     n   median    p10     p90
    same    105    0.810   0.471   1.113
    half     28    1.041   0.667   1.236

The groups separate in the right direction, and by more than the onset-gap route did
(separation 0.350 vs 0.114), but the best single threshold still only catches **75 %**
of half-tempo songs while firing on **40 %** of correct ones.

⚠️**A hypothesis of mine died here too**: I expected DRUMS-only onsets to sharpen it,
since the drums carry the pulse and the union train is noisy. Measured, drums are the
**worst** of the three (separation 0.248 vs bass 0.286 vs union 0.350) — the union's
extra events evidently reinforce the periodicity more than they blur it.

⇒**Octave detection remains OPEN after three attempts** (energy balance 2026-07-27,
onset-gap density and ACF periodicity 2026-08-14). ★What is now well established is
the **payoff**, which none of the earlier attempts had: a correct detector is worth
taking 28 of 149 songs from a hard 0.500× speed ceiling to 1.000×, and a false
positive costs 0.127 onset precision. **That price is what any future attempt should
be measured against**, and it argues for a method with a real model behind it rather
than a fourth summary statistic.

⚠️Scored against the oracle labels (our bpm vs the human's declared bpm). This asks
only whether the signal separates the groups; a shipped gate would need the threshold
priced against the control's **0.127 precision** cost per false positive.

Usage:
    python scripts/diag_tempo_octave_acf.py
"""

from __future__ import annotations

import argparse
import json
import pathlib
import statistics as st
import sys

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from beatsaber_automapper.evaluation import scorecard  # noqa: E402

BIN_S = 0.010          # 10 ms resolution for the onset train


def acf_ratio(onsets, bpm: float) -> float:
    """ACF(half period) / ACF(period) on an onset-impulse train."""
    o = np.asarray(sorted(onsets), dtype=np.float64)
    if len(o) < 16 or bpm <= 0:
        return float("nan")
    period = 60.0 / bpm
    dur = float(o[-1] - o[0])
    if dur < 4 * period:
        return float("nan")

    n = int(dur / BIN_S) + 1
    train = np.zeros(n, dtype=np.float64)
    idx = ((o - o[0]) / BIN_S).astype(int)
    train[np.clip(idx, 0, n - 1)] = 1.0
    train -= train.mean()

    full = np.correlate(train, train, mode="full")[n - 1:]
    if full[0] <= 0:
        return float("nan")
    full = full / full[0]

    def at(lag_s: float) -> float:
        # Take the best value in a +-30 ms neighbourhood: the grid is not exact and a
        # point sample would read the trough next to a real peak.
        c = int(round(lag_s / BIN_S))
        w = max(1, int(round(0.030 / BIN_S)))
        lo, hi = max(1, c - w), min(len(full), c + w + 1)
        return float(np.max(full[lo:hi])) if hi > lo else float("nan")

    a_half, a_full = at(period / 2.0), at(period)
    if not np.isfinite(a_half) or not np.isfinite(a_full) or a_full <= 0:
        return float("nan")
    return a_half / a_full


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", type=pathlib.Path, default=None)
    a = ap.parse_args()

    labels = json.loads(
        (REPO / "outputs" / "true_bpm_wide_cohort_labels.json").read_text())
    rows = []
    for r in labels:
        if r["label"] == "MISSING":
            continue
        zp = REPO / "outputs" / "wide_cohort" / f"{r['song']}.zip"
        if not zp.exists():
            continue
        on = scorecard.onsets_for(zp)
        if on is None or len(on) == 0:
            continue
        v = acf_ratio(on, float(r["ours"]))
        if v != v:
            continue
        rows.append({"song": r["song"], "label": r["label"], "acf_ratio": v})

    if not rows:
        print("no rows")
        return 2

    groups: dict[str, list[float]] = {}
    for r in rows:
        groups.setdefault(r["label"], []).append(r["acf_ratio"])

    print(f"ACF(P/2) / ACF(P) on the onset train, by tempo label (n={len(rows)})")
    print(f"  {'label':>14}{'n':>5}{'median':>9}{'p10':>8}{'p90':>8}")
    for lab in sorted(groups, key=lambda k: -len(groups[k])):
        v = sorted(groups[lab])
        q = lambda p: v[min(len(v) - 1, max(0, int(round(p * (len(v) - 1)))))]  # noqa: E731
        print(f"  {lab:>14}{len(v):>5}{st.median(v):>9.3f}{q(0.10):>8.3f}{q(0.90):>8.3f}")

    half = [r["acf_ratio"] for r in rows if r["label"] == "half"]
    same = [r["acf_ratio"] for r in rows if r["label"] == "same"]
    if half and same:
        best = None
        for t in np.arange(0.0, 2.0, 0.01):
            tp = sum(1 for v in half if v >= t) / len(half)
            fp = sum(1 for v in same if v >= t) / len(same)
            if best is None or (tp - fp) > best[1]:
                best = (float(t), tp - fp, tp, fp)
        t, sep, tp, fp = best
        print(f"\n  best single threshold: acf_ratio >= {t:.2f}")
        print(f"    catches  {tp:.1%} of half-tempo songs")
        print(f"    fires on {fp:.1%} of correct-tempo songs")
        print(f"    separation (TPR - FPR) = {sep:.3f}")
        # Price it. Each caught song gains the ceiling; each false positive costs
        # 0.127 precision. Report the trade rather than a bare verdict.
        print(f"\n  ⇒ {'USABLE — worth pricing against the 0.127 precision cost' if sep > 0.5 else 'NOT USABLE as a single threshold — the groups overlap'}")

    if a.json:
        out = a.json.resolve()
        out.write_text(json.dumps(rows, indent=1))
        print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
