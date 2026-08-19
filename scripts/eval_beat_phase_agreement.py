#!/usr/bin/env python
"""D2 — *"slightly off beat"*, tested against the human map instead of our own onsets.

★**Why not the alignment axis.** `BEAT_GRID_PHASE=search` passed the alignment axis on
149 songs (74 better, 0 worse) and Kyle **still** reported *"slightly off beat"*. That
axis scores our notes against **our own onset detector**, which the C2 landmine says
carries its own offset — so it can be satisfied without the map feeling any better.
This asks a different question with a different oracle: **do our notes land where a
human's notes land**, and does our beat-phase distribution look like a human's?

## What it found (2026-08-18g)

**1. Timing agreement — the share of our notes within ±30 ms of some note in the other
map.** ⚠️Do NOT use the median |delta|: over half our notes land *exactly* on a human
note time (same grid), so that statistic saturates at 0.0 ms and says nothing.

| | ±10 ms | ±30 ms | ±70 ms |
|---|---|---|---|
| ours vs human (n=147) | 0.685 | **0.719** | 0.743 |
| **two DIFFERENT humans, same song** (n=60) | 0.621 | **0.676** | 0.688 |
| ours with the phase destroyed (null) | 0.000 | 0.065 | 0.624 |

⇒**We agree with the human's note times slightly MORE than two humans agree with each
other.**

**2. On the four maps he actually played**, timing agreement is 0.87 (Fallen Kingdom),
0.92 (Hunger), 0.73-0.82 (アリスブルー), 0.67-0.71 (Digital Life Hacker) — all at or
above the human-human 0.676 — **and every one has exactly the human's bpm**
(138/188/160/160), with `_songTimeOffset` 0 on both sides.

**3. Beat phase, 100 songs where our bpm matches the human's:** we are **more** on-beat
than the human (0.580 vs 0.515) and place **fewer** notes on the 16th positions
(1/4 + 3/4: −0.054 paired, we exceed the human on only 30/100 songs, Wilcoxon
p = 0.0006).

🔴🔴**⇒D2 AS "OUR NOTES SIT OFF THE BEAT" IS REFUTED ON THE MAPS HE JUDGED.** Tempo is
right on all four, the offset is zero, and our note times match a human's better than
another human's would. **Stop pointing D2 at tempo for these songs** — the cohort-wide
"tempo right on only 70.5 %" is real but his four songs are inside the 70.5 %.

⚠️**A lead, explicitly UNTESTED**: the same numbers say we are *more rigidly quantised*
than a human — more on-beat, fewer 16ths. A map that puts everything squarely on the
grid while the music swings can feel wrong against the song while being mathematically
aligned. That is a hypothesis about **groove**, and nothing here tests it.
⚠️**A per-song lead that did NOT generalise, recorded so it is not rediscovered**: on
Hunger, 59.5 % of our *unmatched* notes sit at the 3/4 ("a") position where the human
puts 12.8 % overall — but across 100 songs we place *fewer* 16ths than humans, so this
is a property of that song, not of the generator.

Usage:
    python scripts/eval_beat_phase_agreement.py                # cohort + his four maps
    python scripts/eval_beat_phase_agreement.py --tol 0.05
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

HIS_MAPS = {
    "1f8d6": ("FallenKingdom", "Fallen Kingdom"),
    "1f333": ("Hunger", "Hunger"),
    "1f767": ("AliceBlue", "アリスブルー"),
    "1f913": ("DigitalLifeHacker", "Digital Life Hacker"),
}
HUMAN_HUMAN_30MS = 0.676     # measured, 60 independent-mapper pairs


def times(zp: pathlib.Path, diffs=("Expert", "ExpertPlus")) -> np.ndarray | None:
    from eval_drop_agreement import note_times
    for d in diffs:
        try:
            t = note_times(zp, d)
            if len(t) > 40:
                return t
        except Exception:  # noqa: BLE001 — a missing difficulty is normal
            pass
    return None


def coincidence(a: np.ndarray, b: np.ndarray, tol: float) -> float:
    """Share of `a` within tol of some element of `b`."""
    idx = np.clip(np.searchsorted(b, a), 1, len(b) - 1)
    lo, hi = b[idx - 1], b[idx]
    d = np.abs(np.where(np.abs(a - lo) < np.abs(a - hi), a - lo, a - hi))
    return float((d <= tol).mean())


def phase_bins(t: np.ndarray, bpm: float) -> tuple[float, float, float, float]:
    """Share of notes on the beat, and on each 16th subdivision of it."""
    ph = (t * bpm / 60.0) % 1.0
    return (float(np.mean((ph >= 0.875) | (ph < 0.125))),
            float(np.mean((ph >= 0.125) & (ph < 0.375))),
            float(np.mean((ph >= 0.375) & (ph < 0.625))),
            float(np.mean((ph >= 0.625) & (ph < 0.875))))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tol", type=float, default=0.030)
    ap.add_argument("--cohort", type=pathlib.Path,
                    default=REPO / "outputs" / "wide_cohort")
    a = ap.parse_args()
    from feel_disc_poc import _zip_bpm

    print(f"★THE FOUR MAPS HE PLAYED — our notes within ±{a.tol*1000:.0f} ms of a human note")
    print(f"  (two different humans manage {HUMAN_HUMAN_30MS:.3f} on the same song)\n")
    print(f"{'song':<22} {'arm':<11} {'agree':>7} {'our bpm':>8} {'human':>7}")
    for sid, (name, pretty) in HIS_MAPS.items():
        hz = REPO / "data" / "raw" / f"{sid}.zip"
        th = times(hz)
        hb = float(_zip_bpm(str(hz)) or 0)
        if th is None:
            continue
        for arm in ("BEFORE", "AFTER", "CROSSOVER", "BOTH"):
            z = REPO / "for_review" / "A_structure_crossover" / f"{name}_{arm}.zip"
            to = times(z, ("Expert",)) if z.exists() else None
            if to is None:
                continue
            ob = float(_zip_bpm(str(z)) or 0)
            print(f"{pretty if arm == 'BEFORE' else '':<22} {arm:<11} "
                  f"{coincidence(to, th, a.tol):>7.3f} {ob:>8.1f} {hb:>7.1f}"
                  f"{'   ⚠️bpm DISAGREES' if abs(ob - hb) > 0.6 else ''}")

    O, H = [], []
    for z in sorted(a.cohort.glob("*.zip")):
        hz = REPO / "data" / "raw" / f"{z.stem}.zip"
        if not hz.exists():
            continue
        to, th = times(z, ("Expert",)), times(hz)
        if to is None or th is None:
            continue
        ob, hb = float(_zip_bpm(str(z)) or 0), float(_zip_bpm(str(hz)) or 0)
        if ob <= 0 or hb <= 0 or abs(ob - hb) > 0.6:
            continue                      # phase is meaningless across different tempos
        O.append(phase_bins(to, ob))
        H.append(phase_bins(th, hb))
    if not O:
        return 0
    O, H = np.array(O), np.array(H)
    print(f"\n★BEAT PHASE over {len(O)} cohort songs where our bpm matches the human's\n")
    print(f"{'beat position':<22} {'ours':>8} {'human':>8} {'delta':>8}")
    for i, lbl in enumerate(("on the beat", "1/4 (the 'e')",
                             "1/2 (the '&')", "3/4 (the 'a')")):
        print(f"{lbl:<22} {np.median(O[:, i]):>8.3f} {np.median(H[:, i]):>8.3f} "
              f"{np.median(O[:, i]) - np.median(H[:, i]):>+8.3f}")
    try:
        from scipy import stats
        s16o, s16h = O[:, 1] + O[:, 3], H[:, 1] + H[:, 3]
        p = stats.wilcoxon(s16o, s16h).pvalue
        print(f"\nboth 16th positions: paired median {np.median(s16o - s16h):+.4f}, "
              f"we exceed the human on {int((s16o > s16h).sum())}/{len(O)}, p={p:.2g}")
    except ImportError:
        pass
    print("\n⇒We are MORE on-beat than a human and place FEWER 16ths. 'Off beat' is not "
          "where\n  our notes are. See the groove hypothesis in this file's docstring — "
          "it is untested.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
