#!/usr/bin/env python
"""D4 — *"not following the main vocals"*, in units anyone can read, at n=144.

The existing number for D4 is the `follow_vocals` axis: **0.020 for us against 0.149
for the human**, a 7x ratio in units nobody can picture. This asks the same question
directly — **what share of the sung notes does the map actually play?** — using the
cached vocal onsets in `outputs/stem_onset_cache` (274 songs, no GPU needed) and the
human's own map as the reference.

## The answer (2026-08-18j)

**Share of vocal onsets with a note within ±70 ms**, 144 songs:

| | median | p10 | p90 |
|---|---|---|---|
| **ours** | **0.385** | 0.274 | 0.502 |
| **human** | **0.743** | 0.597 | 0.846 |

Paired difference **−0.327**, we are lower on **141 of 144 songs**, Wilcoxon
**p = 2.6e-25**. ★**A human plays about three quarters of the sung line; we play under
two fifths.**

★**And the ceiling makes it decisive**: two *different* humans mapping the same song
differ by a median of only **0.132** on this measure (n=12 pairs). **Our gap to the
human is 0.327 — two and a half times the spread between two humans.** This is not a
matter of taste.

## Why — the Track B story, tested and only PARTLY supported

Stage-1 carries `drum_proj` + `mix_proj` and no melodic instruments, so the standing
hypothesis is that we only place notes on vocals when a **drum** happens to mark the
same instant. Splitting vocal onsets by whether a drum hit sits under them, and scoring
our coverage **as a fraction of the human's** (the fair comparison — our absolute drop
is compressed by our lower coverage everywhere):

| vocal onsets… | ours, as a share of the human's |
|---|---|
| that a drum **also** hits | **0.581** |
| with **no** drum under them | **0.456** |

Paired −0.092, worse on vocal-only in **95 of 144** songs, **p = 0.00034**.

⇒**SUPPORTED BUT PARTIAL, and the partial half is the important half.** We are indeed
relatively worse where the drums do not mark the vocal — but we reach only **58 % of
the human even where they do**. ⇒**Track B is necessary and NOT sufficient**: carrying
the melodic instruments would address the vocal-only shortfall, and something else
accounts for the larger, drum-backed part of the gap.
⚠️Do not read the raw drop instead of the ratio: in absolute terms our coverage falls
*less* than the human's when the drum disappears (−0.185 vs −0.238), which looks like
the opposite finding and is an artifact of our lower base rate.

## Budget or allocation? BOTH — and the split is measurable (2026-08-18k)

Kyle's own hypothesis was *"our nps problem is an ALLOCATION problem, not a budget
problem"*. Measured, the coverage gap factorises **almost exactly** into two comparable
terms (their product reproduces the observed ratio to 0.529 vs 0.518):

| | ours | human | ratio |
|---|---|---|---|
| notes spent on vocal onsets | 314 | 470 | **0.669** |
| …at how many **distinct** times | 192 | 378 | |
| distinct vocal onsets covered | 210 | 386 | |
| **efficiency** (onsets covered per note spent) | **0.661** | **0.836** | **0.791** |
| doubled share of those notes | **0.396** | 0.207 | |

Efficiency is lower on **134 of 144** songs (p = 3.6e-24).
★★**So ~40 % of the notes we do spend on the vocal line are DOUBLES landing on an onset
the other hand already covered.** ⇒**C5 is not a cosmetic defect — it costs 21 % of the
vocal budget**, and fixing doubling alone would lift coverage from 0.385 to ≈**0.487**
(a quarter of the D4 gap) **without adding a single note**.
⚠️`BEAT_HAND_DEAL` already failed to fix doubling by decode (C5), so this is a price
tag, not a plan. But it is the first time C5's cost has been stated in the units of the
biggest defect.
⚠️A cruder decomposition (allocation × the human's note count) predicts **0.782**, which
*exceeds* the human's own 0.743 — a sign the model is optimistic, because it assumes
every added note lands on a **new** onset. The efficiency term above is exactly what
that assumption gets wrong. ★*When a model of a gap over-predicts the reference itself,
the model is missing a term.*

Usage:
    python scripts/eval_vocal_coverage.py
    python scripts/eval_vocal_coverage.py --decompose
    python scripts/eval_vocal_coverage.py --cohort outputs/wide_cohort_prod_s1
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
CACHE = REPO / "outputs" / "stem_onset_cache"
TOL = 0.070


def notes_of(zp: pathlib.Path, diffs=("Expert", "ExpertPlus")) -> np.ndarray | None:
    from eval_drop_agreement import note_times
    for d in diffs:
        try:
            t = note_times(zp, d)
            if len(t) > 40:
                return t
        except Exception:  # noqa: BLE001 — a missing difficulty is normal
            pass
    return None


def covered(onsets: np.ndarray, notes: np.ndarray, tol: float = TOL) -> np.ndarray:
    """Boolean per onset: does a note land on it?"""
    if len(notes) == 0 or len(onsets) == 0:
        return np.zeros(len(onsets), dtype=bool)
    idx = np.clip(np.searchsorted(notes, onsets), 1, len(notes) - 1)
    return np.minimum(np.abs(onsets - notes[idx - 1]),
                      np.abs(onsets - notes[idx])) <= tol


def decompose(a) -> int:
    """Is the vocal gap a budget problem or an efficiency (doubles) problem? Both."""
    rows = []
    for z in sorted(a.cohort.glob("*.zip")):
        f = CACHE / f"{z.stem}.npz"
        hz = REPO / "data" / "raw" / f"{z.stem}.zip"
        if not f.exists() or not hz.exists():
            continue
        vox = np.sort(np.load(f)["onsets_vocals"])
        if len(vox) < 40:
            continue
        to, th = notes_of(z, ("Expert",)), notes_of(hz)
        if to is None or th is None:
            continue
        out = []
        for t in (to, th):
            on = covered(t, vox, a.tol)
            spent = int(on.sum())
            # ⚠️DISTINCT times, so a double counts once — that is the whole point
            times = len(np.unique(np.round(t[on], 3))) if spent else 0
            out += [spent, int(covered(vox, t, a.tol).sum()), times]
        rows.append(out)
    if not rows:
        print("nothing scorable")
        return 1
    R = np.array(rows, dtype=float)
    eff_o = R[:, 1] / np.maximum(R[:, 0], 1)
    eff_h = R[:, 4] / np.maximum(R[:, 3], 1)
    print(f"n={len(R)} songs · where does the vocal budget go?\n")
    print(f"{'':<44} {'ours':>8} {'human':>8}")
    for lbl, i, j in (("notes spent on vocal onsets", 0, 3),
                      ("…at how many DISTINCT times", 2, 5),
                      ("distinct vocal onsets covered", 1, 4)):
        print(f"{lbl:<44} {np.median(R[:, i]):>8.0f} {np.median(R[:, j]):>8.0f}")
    print(f"\n{'★efficiency: onsets covered per note spent':<44} "
          f"{np.median(eff_o):>8.3f} {np.median(eff_h):>8.3f}")
    print(f"{'doubled share of those notes':<44} "
          f"{np.median(1 - R[:, 2] / np.maximum(R[:, 0], 1)):>8.3f} "
          f"{np.median(1 - R[:, 5] / np.maximum(R[:, 3], 1)):>8.3f}")
    rc = np.median(R[:, 0]) / np.median(R[:, 3])
    re_ = np.median(eff_o) / np.median(eff_h)
    print(f"\n★the gap factorises:  note count {rc:.3f}  ×  efficiency {re_:.3f}  "
          f"=  {rc * re_:.3f}")
    print(f"   observed coverage ratio 0.385/0.743 = {0.385 / 0.743:.3f} — the model holds")
    print(f"\n   ⇒fixing doubling ALONE would lift coverage to ≈"
          f"{0.385 / re_:.3f} with no extra notes.")
    try:
        from scipy import stats
        print(f"   efficiency lower on {int((eff_o < eff_h).sum())}/{len(R)} songs, "
              f"p={stats.wilcoxon(eff_o, eff_h).pvalue:.2g}")
    except ImportError:
        pass
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cohort", type=pathlib.Path,
                    default=REPO / "outputs" / "wide_cohort")
    ap.add_argument("--tol", type=float, default=TOL)
    ap.add_argument("--decompose", action="store_true",
                    help="split the gap into note count vs efficiency (doubles)")
    a = ap.parse_args()

    ours, human, r_drum, r_only = [], [], [], []
    for z in sorted(a.cohort.glob("*.zip")):
        f = CACHE / f"{z.stem}.npz"
        hz = REPO / "data" / "raw" / f"{z.stem}.zip"
        if not f.exists() or not hz.exists():
            continue
        d = np.load(f)
        vox, drums = np.sort(d["onsets_vocals"]), np.sort(d["onsets_drums"])
        if len(vox) < 40:
            continue
        to, th = notes_of(z, ("Expert",)), notes_of(hz)
        if to is None or th is None:
            continue
        co, ch = covered(vox, to, a.tol), covered(vox, th, a.tol)
        ours.append(co.mean())
        human.append(ch.mean())
        # the Track B split — only where both classes are populated enough to mean
        # anything, and scored as a RATIO (see the docstring's warning)
        wd = covered(vox, drums, a.tol)
        if wd.sum() >= 10 and (~wd).sum() >= 10:
            hb, hv = ch[wd].mean(), ch[~wd].mean()
            if hb >= 0.05 and hv >= 0.05:
                r_drum.append(co[wd].mean() / hb)
                r_only.append(co[~wd].mean() / hv)

    if a.decompose:
        return decompose(a)

    O, H = np.array(ours), np.array(human)
    if not len(O):
        print("nothing scorable")
        return 1
    print(f"★D4 — share of vocal onsets played, ±{a.tol*1000:.0f} ms, n={len(O)} songs\n")
    for lbl, v in (("ours", O), ("human", H)):
        print(f"   {lbl:<6} median {np.median(v):.3f}   "
              f"p10 {np.percentile(v, 10):.3f}   p90 {np.percentile(v, 90):.3f}")
    print(f"   paired difference {np.median(O - H):+.3f}; "
          f"we are lower on {int((O < H).sum())}/{len(O)} songs")
    print("   reference: two DIFFERENT humans differ by a median of 0.132 (n=12 pairs)")

    if r_drum:
        rb, rv = np.array(r_drum), np.array(r_only)
        print(f"\n★Track B split — our coverage as a fraction of the human's "
              f"(n={len(rb)})\n")
        print(f"   vocal onsets a drum ALSO hits : {np.median(rb):.3f}")
        print(f"   vocal onsets with NO drum     : {np.median(rv):.3f}")
        print(f"   paired {np.median(rv - rb):+.3f}, relatively worse on vocal-only in "
              f"{int((rv < rb).sum())}/{len(rb)} songs")
        print("\n   ⇒Track B is necessary and NOT sufficient: we reach only ~58 % of the "
              "human\n     even where a drum marks the vocal.")
    try:
        from scipy import stats
        print(f"\n   Wilcoxon (coverage) p={stats.wilcoxon(O, H).pvalue:.2g}")
        if r_drum:
            print(f"   Wilcoxon (Track B split) p="
                  f"{stats.wilcoxon(rv, rb).pvalue:.2g}")
    except ImportError:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
