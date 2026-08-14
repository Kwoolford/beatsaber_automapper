#!/usr/bin/env python
"""IS THE OCTAVE ERROR LEARNABLE AT ALL? — a tempogram VECTOR, cross-validated.

**The question this settles before anything gets built.** Detecting that our tempo is
an octave low is worth a lot — subdiv 8 takes those 28 songs from a hard 0.500× speed
ceiling to 1.000× — and a false positive costs 0.127 onset precision. But three
hand-designed statistics have now failed:

| attempt | separation (TPR − FPR) |
|---|---|
| onset-energy balance (2026-07-27) | made detection *worse* |
| onset-gap density (2026-08-14) | 0.114 |
| ACF ratio at P/2 (2026-08-14) | 0.350 |

The natural next move is "train a model on the corpus, where 5,373 map zips carry a
human-declared bpm as free supervision". ⚠️**That is hours of audio decoding and beat
tracking before the first result.** This script asks the cheap question first:
**given a richer representation than a single ratio, is the signal there at all?**

★If a cross-validated classifier on a tempogram VECTOR cannot separate the groups on
the 149 songs we have already labelled, then the corpus-scale version is not blocked
on sample size and scaling up would be building on nothing. If it can, the scale-up
is justified and this tells us which features carry it.

**Representation**: autocorrelation of the onset train sampled at lags that are
MULTIPLES OF THE DETECTED PERIOD (0.25P … 4P). Indexing by P rather than by seconds
makes the feature tempo-invariant, so the classifier cannot simply learn "slow songs
are the mislabelled ones" — which, given half-tempo songs have low detected bpm by
construction, is the obvious way to get a fake result. **`bpm` is deliberately NOT a
feature**, and the ablation below reports what happens when it is added.

⚠️n = 28 positives. Cross-validated AUC on 28 positives has a wide error bar; this is
a SCREEN for "is there signal", not a performance estimate.

Usage:
    python scripts/diag_octave_learnable.py
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from beatsaber_automapper.evaluation import scorecard  # noqa: E402

BIN_S = 0.010
# Lags as multiples of the detected beat period. 0.5 is the octave-error lag; the
# others give the classifier context (is 0.5 strong *relative to* its neighbours).
MULTIPLES = [0.25, 0.375, 0.5, 0.625, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 4.0]


def acf_features(onsets, bpm: float) -> np.ndarray | None:
    o = np.asarray(sorted(onsets), dtype=np.float64)
    if len(o) < 16 or bpm <= 0:
        return None
    period = 60.0 / bpm
    dur = float(o[-1] - o[0])
    if dur < 8 * period:
        return None
    n = int(dur / BIN_S) + 1
    train = np.zeros(n)
    train[np.clip(((o - o[0]) / BIN_S).astype(int), 0, n - 1)] = 1.0
    train -= train.mean()
    full = np.correlate(train, train, mode="full")[n - 1:]
    if full[0] <= 0:
        return None
    full = full / full[0]

    def at(lag_s: float) -> float:
        c = int(round(lag_s / BIN_S))
        w = max(1, int(round(0.030 / BIN_S)))
        lo, hi = max(1, c - w), min(len(full), c + w + 1)
        return float(np.max(full[lo:hi])) if hi > lo else 0.0

    return np.array([at(m * period) for m in MULTIPLES], dtype=np.float64)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", type=pathlib.Path, default=None)
    a = ap.parse_args()

    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score

    labels = json.loads(
        (REPO / "outputs" / "true_bpm_wide_cohort_labels.json").read_text())
    X, y, bpms, songs = [], [], [], []
    for r in labels:
        if r["label"] not in ("half", "same"):
            continue
        zp = REPO / "outputs" / "wide_cohort" / f"{r['song']}.zip"
        if not zp.exists():
            continue
        on = scorecard.onsets_for(zp)
        if on is None or len(on) == 0:
            continue
        f = acf_features(on, float(r["ours"]))
        if f is None:
            continue
        X.append(f)
        y.append(1 if r["label"] == "half" else 0)
        bpms.append(float(r["ours"]))
        songs.append(r["song"])

    X = np.array(X)
    y = np.array(y)
    print(f"n={len(y)}  half={int(y.sum())}  same={int((1 - y).sum())}  "
          f"features={X.shape[1]} (ACF at multiples of the DETECTED period)\n")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0))

    def evaluate(Xu, name: str) -> float:
        p = cross_val_predict(clf, Xu, y, cv=cv, method="predict_proba")[:, 1]
        auc = roc_auc_score(y, p)
        # A threshold-free AUC hides whether any operating point is usable, and the
        # cost here is asymmetric, so report the best separation too.
        best = max(((float(t), (p[y == 1] >= t).mean() - (p[y == 0] >= t).mean())
                    for t in np.unique(p)), key=lambda kv: kv[1])
        t, sep = best
        tpr = (p[y == 1] >= t).mean()
        fpr = (p[y == 0] >= t).mean()
        print(f"  {name:<34} AUC {auc:.3f}   best sep {sep:.3f} "
              f"(TPR {tpr:.0%}, FPR {fpr:.0%})")
        return auc

    print("CROSS-VALIDATED (5-fold, stratified):")
    evaluate(X, "tempogram vector (tempo-invariant)")

    # ⚠️THE CONFOUND CHECK. Half-tempo songs have a low DETECTED bpm by construction,
    # so a model given bpm can score well while learning nothing about metre. If
    # adding bpm helps a lot, the "signal" is mostly that confound.
    evaluate(np.column_stack([X, np.array(bpms)]), "+ detected bpm (CONFOUNDED)")
    evaluate(np.array(bpms).reshape(-1, 1), "detected bpm ALONE (the confound)")

    print("\n  ⇒ read the first row against the third: if bpm ALONE is as good as the")
    print("    tempogram, nothing has been learned about metre and a corpus-scale")
    print("    version of this would inherit the same confound at 40x the cost.")

    if a.json:
        out = a.json.resolve()
        out.write_text(json.dumps(
            [{"song": s, "y": int(v), "bpm": b} for s, v, b in zip(songs, y, bpms)],
            indent=1))
        print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
