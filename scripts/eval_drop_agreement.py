#!/usr/bin/env python
"""D3 — *"doing drops at the wrong time"*, made measurable WITHOUT Kyle in the loop.

★**The idea: the human mapper's own density jump is an oracle for where the drop is.**
A mapper marks a drop by moving the note rate, so the times at which a human map's
density steps up are that mapper's answer to "where does this song turn". We can ask
whether **our** map turns in the same places — no ear required, and no reliance on our
own section detector, which would make the test circular.

## What was ruled out first
Two cheaper explanations of D3 were measured and **both are refuted** (2026-08-18f):

1. *We fail to lift density at drops.* **No.** Density at detected `DROP`/`peak`
   sections against each map's own mean: ours **1.09 / 1.21 / 1.08 / 0.95**, the human
   **1.11 / 1.21 / 1.09 / 0.99** — the same lift, song for song. The old key-note
   "flat ~8 NPS ignores song structure" does not survive at this granularity.
2. *Our density jumps miss our own detected drops.* Also no: our best jump sits
   1.9 / 0.4 / 0.5 / 4.0 s from a detected `DROP`, the human's 3.2 / 0.3 / 0.0 / 3.5 s.
   Equally well aligned.

⇒What is left is the *placement of the moves themselves*, which is what this measures.

## The metric
Take each map's `--k` biggest sustained density step-ups (density over the next `win`
seconds minus the previous `win`, with a 20 s non-maximum suppression so one drop cannot
occupy every slot). A jump **agrees** if a jump of the other map sits within `--tol`.

⚠️**Scored against a permutation null**, because with a handful of jumps in a 4-minute
song, agreement "by eye" is worthless — uniformly random jump times already agree
surprisingly often at a loose tolerance. The number that matters is agreement **minus**
what chance gives, and this project has twice been fooled by a statistic that had none
(the detector alarm at 59.5 % against a permutation null of 37.8 %, p = 0.324).

## The answer, at n=144 songs (2026-08-18f)

| what | agreement | note |
|---|---|---|
| uniformly random jump times | **0.140** | the floor |
| **ours vs the human, 144 songs / 432 jumps** | **0.347** | 95 % CI [0.302, 0.392] |
| two DIFFERENT humans, same song | **0.49** (0.56 allowing one global offset) | 54 pairs |
| the same human, two difficulties | 1.00 | n=2 — same-author, so inflated |

★★**D3 CONFIRMED.** The CI excludes the null *and* the human-human band: we are not
random, and we are not human. **43 of 144 songs agree on NOTHING** — not one of our
three biggest density moves coincides with any of theirs.
★**The human-human number is the one that makes this readable.** Two humans mapping the
same song agree only about half the time, so 1.00 was never the target; without that
band, 0.347 could have been argued either way. ⚠️It comes from a *different* sample
(duplicate-song pairs) than our 144, so the comparison is across cohorts, not paired —
no human map exists twice for the songs we generate on.

🔴**This is a NEW metric and it has NOT passed `scripts/audit_eval_suite.py`.** It is a
diagnosis, not a steering signal, until it does — see the loop discipline in TODO.md.

Usage:
    python scripts/eval_drop_agreement.py
    python scripts/eval_drop_agreement.py --tol 4 --k 3 --perm 2000
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "agent_mapper"))

SONGS = {
    "1f8d6": ("FallenKingdom", "Fallen Kingdom", "Expert"),
    "1f333": ("Hunger", "Hunger", "Expert"),
    "1f767": ("AliceBlue", "アリスブルー", "Expert"),
    "1f913": ("DigitalLifeHacker", "Digital Life Hacker", "ExpertPlus"),
}


def note_times(zp: pathlib.Path, difficulty: str) -> np.ndarray:
    from eval_contour_follow import _load_notes_with_direction
    from feel_disc_poc import _zip_bpm

    recs = _load_notes_with_direction(zp, difficulty)
    bpm = float(_zip_bpm(str(zp)) or 120.0)
    return np.sort(np.array([r[0] for r in recs], dtype=float) * 60.0 / bpm)


def jumps(t: np.ndarray, dur: float, win: float = 6.0, k: int = 3,
          suppress: float = 20.0) -> list[tuple[float, float]]:
    """The k biggest sustained density step-ups, with non-maximum suppression."""
    step = 0.5
    grid = np.arange(win, max(dur - win, win + step), step)
    if len(grid) < 4:
        return []
    dens = np.array([(np.sum((t >= g) & (t < g + win))
                      - np.sum((t >= g - win) & (t < g))) / win for g in grid])
    out: list[tuple[float, float]] = []
    guard = int(suppress / step)
    for _ in range(k):
        i = int(np.argmax(dens))
        if dens[i] <= -1e8:
            break
        out.append((float(grid[i]), float(dens[i])))
        dens[max(0, i - guard):i + guard] = -1e9
    return out


def agree(a: list[float], b: list[float], tol: float) -> int:
    """How many of `a` have a partner in `b` within tol. Greedy, one partner each."""
    free = list(b)
    n = 0
    for x in a:
        if not free:
            break
        j = min(range(len(free)), key=lambda i: abs(free[i] - x))
        if abs(free[j] - x) <= tol:
            n += 1
            free.pop(j)
    return n


def null_agreement(n_a: int, b: list[float], dur: float, tol: float,
                   perm: int, rng: np.random.Generator) -> np.ndarray:
    """Agreement when OUR jumps are placed uniformly at random over the song."""
    return np.array([agree(sorted(rng.uniform(0, dur, n_a).tolist()), b, tol)
                     for _ in range(perm)], dtype=float)


def cohort(a) -> int:
    """Score a whole directory of our maps against their human counterparts."""
    rng = np.random.default_rng(a.seed)
    rows, nulls, skipped = [], [], 0
    for z in sorted(a.cohort.glob("*.zip")):
        human = REPO / "data" / "raw" / f"{z.stem}.zip"
        if not human.exists():
            skipped += 1
            continue
        try:
            to = note_times(z, "Expert")
            th = None
            for diff in ("Expert", "ExpertPlus"):
                try:
                    t = note_times(human, diff)
                    if len(t) > 40:
                        th = t
                        break
                except Exception:  # noqa: BLE001 — a missing difficulty is normal
                    pass
            if th is None:
                skipped += 1
                continue
        except Exception:  # noqa: BLE001
            skipped += 1
            continue
        # ⚠️Duration from the NOTES, not the audio: the cohort's audio lives inside the
        # zips and re-extracting 150 of them to learn a number we already have is waste.
        dur = float(min(to.max(), th.max()))
        if dur < 60:
            skipped += 1
            continue
        jo = [x for x, _ in jumps(to, dur, a.win, a.k)]
        jh = [x for x, _ in jumps(th, dur, a.win, a.k)]
        if len(jo) < a.k or len(jh) < a.k:
            skipped += 1
            continue
        rows.append((z.stem, agree(jo, jh, a.tol), len(jo)))
        nulls.append(float(null_agreement(len(jo), jh, dur, a.tol,
                                          max(a.perm // 10, 50), rng).mean()))
    if not rows:
        print("nothing scorable")
        return 1
    hit = sum(r[1] for r in rows)
    n = sum(r[2] for r in rows)
    share = hit / n
    se = (share * (1 - share) / n) ** 0.5
    zero = sum(1 for _, h, _ in rows if h == 0)
    print(f"{len(rows)} songs scored, {skipped} skipped\n")
    print(f"  ours vs human      {hit}/{n} = {share:.3f}  "
          f"95% CI [{share - 1.96 * se:.3f}, {share + 1.96 * se:.3f}]")
    print(f"  permutation null   {sum(nulls) / n:.3f}")
    print(f"  two humans, same song   0.49  (0.56 allowing one global offset, n=54 pairs)")
    print(f"\n  {zero} of {len(rows)} songs agree on NOTHING; "
          f"{sum(1 for _, h, k in rows if h >= 2 * k / 3)} agree on 2/3 or more")
    print("\nRead it against BOTH bands: above the null we are not random, below the "
          "human-human\nband we are not human. ⚠️That band comes from a different "
          "sample (duplicate-song pairs),\nso it is a cross-cohort reference, not a "
          "paired one.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tol", type=float, default=4.0, help="seconds; ~2 bars")
    ap.add_argument("--k", type=int, default=3, help="jumps per map")
    ap.add_argument("--win", type=float, default=6.0)
    ap.add_argument("--perm", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cohort", type=pathlib.Path, default=None,
                    help="a directory of OUR maps named <song id>.zip (e.g. "
                         "outputs/wide_cohort) — scores every one against its human "
                         "map in data/raw. This is how the n=144 number was got.")
    a = ap.parse_args()

    if a.cohort is not None:
        return cohort(a)

    import notesheet as ns
    rng = np.random.default_rng(a.seed)

    tot_hit = tot_n = 0
    null_pool: list[float] = []
    print(f"tolerance ±{a.tol:.0f}s, {a.k} jumps a map, {a.win:.0f}s window\n")
    print(f"{'song':<22} {'ours (m:ss)':<26} {'human (m:ss)':<26} {'agree':>6} {'null':>6}")
    for sid, (mapname, pretty, hdiff) in SONGS.items():
        audio = REPO / "data" / "eval_songset" / f"{sid}.ogg"
        ours_z = REPO / "for_review" / "A_structure_crossover" / f"{mapname}_BEFORE.zip"
        human_z = REPO / "data" / "raw" / f"{sid}.zip"
        if not (audio.exists() and ours_z.exists() and human_z.exists()):
            print(f"{pretty:<22} missing inputs")
            continue
        dur = ns.collect(audio)["dur"]
        jo = jumps(note_times(ours_z, "Expert"), dur, a.win, a.k)
        jh = jumps(note_times(human_z, hdiff), dur, a.win, a.k)
        to, th = [x for x, _ in jo], [x for x, _ in jh]
        hit = agree(to, th, a.tol)
        nul = null_agreement(len(to), th, dur, a.tol, a.perm, rng)
        tot_hit += hit
        tot_n += len(to)
        null_pool.append(float(nul.mean()))
        fmt = lambda xs: " ".join(f"{int(x//60)}:{x%60:04.1f}" for x in xs)
        print(f"{pretty:<22} {fmt(to):<26} {fmt(th):<26} "
              f"{hit}/{len(to):<4} {nul.mean():>6.2f}")

    exp = sum(null_pool)
    print(f"\nPOOLED: {tot_hit}/{tot_n} of our density jumps land within ±{a.tol:.0f}s of "
          f"one of the human's")
    print(f"        chance alone gives {exp:.2f}/{tot_n} "
          f"({exp / max(tot_n, 1):.2f} vs our {tot_hit / max(tot_n, 1):.2f})")
    print("\nVERDICT LOGIC: agreement at or below the null means our biggest density "
          "moves\nhappen where the human's do not — which is D3 stated as a number. "
          "Well above\nthe null means D3 is NOT about where the moves are, and the "
          "search moves on.")
    print("\n🔴Diagnosis only — this metric has not passed audit_eval_suite.py and must "
          "not steer\nthe generator until it does.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
