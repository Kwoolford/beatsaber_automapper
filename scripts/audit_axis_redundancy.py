#!/usr/bin/env python
"""IS THE SUITE WELL-FORMED? — what each axis adds that the others do not.

A suite grows one axis at a time and nobody ever asks whether two of them are the
same measurement wearing different names. That matters twice over here:

  * if a new axis correlates ~1.0 with an old one, it is not a new finding, it is
    the old finding restated — and this project has already published a "new"
    result that turned out to be an old one (W3 was C5 wearing a hat);
  * if two axes inside the new set are the same, every summary that lists both is
    double-counting one defect.

**Method.** Score the same 149 wide-cohort maps on both the classic scorecard
(A1 flow, A2 rhythm, A3 idiom, A6 handrole, A8 alignment, playfeel) and the
masterpiece axes, then take **Spearman** correlations across songs — rank-based,
because several of these are bounded and skewed.

Read it as: *for this axis, what is the largest |r| against anything already in the
suite?* A value near 1 means drop it or merge it; a value near 0 means it is
carrying information nothing else does.

Usage:
    python scripts/audit_axis_redundancy.py --json outputs/axis_redundancy.json
"""

from __future__ import annotations

import argparse
import glob
import json
import pathlib
import sys

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

import masterpiece_report as mr  # noqa: E402
from beatsaber_automapper.evaluation import scorecard  # noqa: E402

CLASSIC = ["precision", "scatter_ms", "flow_dist", "angle_change", "travel",
           "pulse_stability", "ioi_entropy", "idiom_coverage", "idiom_jsd",
           "role_asymmetry", "nps", "peak_nps", "diagonal_share"]
M_AXES = [k for _, k in mr.REPORT_KEYS]


def spearman(a: np.ndarray, b: np.ndarray) -> float | None:
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 25:
        return None
    ra = np.argsort(np.argsort(a[ok])).astype(float)
    rb = np.argsort(np.argsort(b[ok])).astype(float)
    if ra.std() < 1e-9 or rb.std() < 1e-9:
        return None
    return float(np.corrcoef(ra, rb)[0, 1])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json", default="")
    ap.add_argument("--limit", type=int, default=200)
    a = ap.parse_args()

    rows = mr.collect("prod", "s0", rebuild=False, wide=True)
    by_song = {r["song"]: r["ours"] for r in rows}
    cols: dict[str, dict[str, float]] = {}

    files = sorted(glob.glob(str(REPO / "outputs/wide_cohort/*.zip")))[: a.limit]
    for f in files:
        song = pathlib.Path(f).stem
        if song not in by_song:
            continue
        p = pathlib.Path(f)
        L = scorecard._load_any(p)
        if not L:
            continue
        try:
            onsets = scorecard.onsets_for(p)
        except Exception:
            onsets = None
        rec = scorecard._metrics_for(L[0], float(L[1]), onsets)
        cols[song] = {k: v for k, v in rec.items()
                      if isinstance(v, (int, float)) and v is not None}
        cols[song].update({k: v for k, v in by_song[song].items()
                           if isinstance(v, (int, float)) and v is not None})
    if len(cols) < 30:
        print(f"only {len(cols)} songs scored")
        return
    songs = sorted(cols)
    print(f"{len(songs)} songs scored on both suites\n")

    def vec(k):
        return np.array([cols[s].get(k, np.nan) for s in songs], dtype=float)

    classic = [c for c in CLASSIC if np.isfinite(vec(c)).sum() >= 30]
    m_axes = [m for m in M_AXES if np.isfinite(vec(m)).sum() >= 30]

    print(f"{'='*96}\nWHAT EACH MASTERPIECE AXIS ADDS — |Spearman| against the classic suite")
    print(f"{'='*96}")
    print(f"{'axis':<20} {'max |r| vs classic':>19}  {'nearest classic':<18} "
          f"{'max |r| vs other M':>19}  nearest M")
    out = {}
    for m in m_axes:
        vm = vec(m)
        cr = {c: spearman(vm, vec(c)) for c in classic}
        cr = {c: r for c, r in cr.items() if r is not None}
        mr_ = {o: spearman(vm, vec(o)) for o in m_axes if o != m}
        mr_ = {o: r for o, r in mr_.items() if r is not None}
        bc = max(cr, key=lambda c: abs(cr[c])) if cr else None
        bm_ = max(mr_, key=lambda o: abs(mr_[o])) if mr_ else None
        out[m] = {"max_abs_r_classic": round(abs(cr[bc]), 3) if bc else None,
                  "nearest_classic": bc,
                  "max_abs_r_masterpiece": round(abs(mr_[bm_]), 3) if bm_ else None,
                  "nearest_masterpiece": bm_}
        print(f"{m:<20} {abs(cr[bc]) if bc else float('nan'):>19.3f}  {str(bc):<18} "
              f"{abs(mr_[bm_]) if bm_ else float('nan'):>19.3f}  {bm_}")

    print("\nHOW TO READ: a large |r| against the CLASSIC suite means the axis is an old")
    print("finding restated (this project has shipped one of those: W3 was C5 wearing a")
    print("hat). A large |r| against another M axis means any summary listing both is")
    print("double-counting one defect.")

    if a.json:
        pathlib.Path(a.json).write_text(json.dumps(
            {"n_songs": len(songs), "axes": out}, indent=2))
        print(f"\nwrote {a.json}")


if __name__ == "__main__":
    main()
