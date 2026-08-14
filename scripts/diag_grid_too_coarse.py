#!/usr/bin/env python
"""CAN OUR GRID REPRESENT THIS MUSIC? — a detector that avoids the question that beat us.

**The problem this replaces.** Subdiv 8 lifts the half-tempo speed ceiling exactly
(ebpm ratio 0.500 → 1.000, n=28) and wrecks correctly-detected songs just as clearly
(1.000 → 2.000, precision −0.127). So the lever is worth having *if and only if* we
can tell the two apart. The obvious route — detect the true metrical level — is a
known dead end here: `bpm_octave_probe.py` tried two heuristics on 2026-07-27 and
**both made detection worse** (16/23 correct → 10/23, and → 14/23), because
onset-energy balance does not discriminate metrical level.

★**THE REFRAME.** We never actually needed the true tempo. We need to know whether
**our grid can represent this music** — and that is a property of the audio we
already hold. If a large share of the音 onsets fall closer together than one grid
slot, those events are physically unrepresentable no matter how good the model is.
That is measurable at generation time, needs no human map, and does not require
naming the metrical level. Same move that rescued `BEAT_GRID_PHASE`: **search the
thing you can measure instead of predicting the thing you cannot.**

**The statistic**: `unrepresentable` = share of consecutive stem-onset gaps shorter
than one slot (60/bpm/subdiv). Under a correct tempo this should be small; under a
half-tempo call the slot is twice as long and the share should jump.

🔴🔴**RESULT 2026-08-14 — NOT USABLE. The reframe was appealing and the instrument is
blunt.** n=149:

    label          n   median    p10     p90
    same         105    0.702   0.590   0.835
    half          28    0.885   0.773   0.965

The groups do separate in the median, but they **overlap heavily**: the best single
threshold catches 100 % of half-tempo songs only by firing on **88.6 %** of
correctly-detected ones — and the `same` control priced a false positive at
**−0.127 onset precision**. As a gate this is worse than doing nothing.

★**WHY, and it is the useful part**: stem onsets are DENSE. Even at a correct tempo
**70 %** of adjacent onset gaps are already shorter than one slot — the detector
fires several times per beat across four stems, on ornaments and noise as well as on
structural events. So this statistic is dominated by **onset density**, not by grid
coarseness, and the half-tempo signal rides on top of a much larger nuisance term.
⇒**Measuring "can the grid represent the audio" via raw onset gaps measures the wrong
thing.** The discriminator has to be about **periodicity** — is there a regular pulse
at a period shorter than our slot — not about how many events are close together.
**Next candidate (untried): a tempogram-ratio method**, which is what
`bpm_octave_probe.py`'s own postmortem recommended. ⚠️It is NOT the thing that failed
in 2026-07-27 — that was onset-energy *balance* between on- and off-beat positions,
a different statistic.

⚠️This is a DETECTOR PROTOTYPE scored against the oracle labels (our bpm vs the
human's declared bpm). Kept because a recorded negative with a named cause is what
stops the next session re-deriving it.

Usage:
    python scripts/diag_grid_too_coarse.py
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

SUBDIV = 4


def unrepresentable_share(onsets, bpm: float, subdiv: int = SUBDIV) -> float:
    """Share of adjacent onset gaps shorter than one grid slot."""
    o = np.sort(np.asarray(onsets, dtype=np.float64))
    if len(o) < 8 or bpm <= 0:
        return float("nan")
    gaps = np.diff(o)
    gaps = gaps[gaps > 0]
    if len(gaps) == 0:
        return float("nan")
    slot = 60.0 / bpm / subdiv
    return float(np.mean(gaps < slot))


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
        u = unrepresentable_share(on, float(r["ours"]))
        if u != u:
            continue
        rows.append({"song": r["song"], "label": r["label"], "ours": r["ours"],
                     "human": r["human"], "unrep": u})

    if not rows:
        print("no rows")
        return 2

    groups: dict[str, list[float]] = {}
    for r in rows:
        groups.setdefault(r["label"], []).append(r["unrep"])

    print(f"unrepresentable-onset share, by tempo label (n={len(rows)})")
    print(f"  {'label':>14}{'n':>5}{'median':>9}{'p10':>8}{'p90':>8}")
    for lab in sorted(groups, key=lambda k: -len(groups[k])):
        v = sorted(groups[lab])
        q = lambda p: v[min(len(v) - 1, max(0, int(round(p * (len(v) - 1)))))]  # noqa: E731
        print(f"  {lab:>14}{len(v):>5}{st.median(v):>9.3f}{q(0.10):>8.3f}{q(0.90):>8.3f}")

    half = [r["unrep"] for r in rows if r["label"] == "half"]
    same = [r["unrep"] for r in rows if r["label"] == "same"]
    if half and same:
        # A detector is only useful if a THRESHOLD separates the groups. Sweep it and
        # report the best, with both error types — a rule that catches every half-tempo
        # song by also firing on every correct one is worthless, and the `same` arm
        # showed a false positive costs 0.127 of precision.
        best = None
        for t in np.arange(0.0, 0.60, 0.005):
            tp = sum(1 for v in half if v >= t) / len(half)
            fp = sum(1 for v in same if v >= t) / len(same)
            if best is None or (tp - fp) > best[1]:
                best = (float(t), tp - fp, tp, fp)
        t, _, tp, fp = best
        print(f"\n  best single threshold: unrep >= {t:.3f}")
        print(f"    catches {tp:.1%} of half-tempo songs")
        print(f"    fires on {fp:.1%} of correct-tempo songs  "
              f"(each false positive costs ~0.127 precision)")
        print(f"\n  ⇒ {'USABLE — the groups separate' if tp - fp > 0.5 else 'NOT USABLE as a single threshold — the groups overlap'}")

    if a.json:
        out = a.json.resolve()
        out.write_text(json.dumps(rows, indent=1))
        print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
