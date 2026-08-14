#!/usr/bin/env python
"""IS THE BASELINE'S ALIGNMENT FAILURE A GRID-PHASE DEFECT? — n=149, no GPU.

**The open problem.** Restoring A8 exposed that our promoted maps fail alignment at
n=149 (ours 0.8914 vs human 0.9492 paired). The failure is BIMODAL and, per the
seed test, **song-driven near-deterministically** (corr(Δs0, Δs1) = +0.981; 35 of
38 bad songs identical across seeds). Nothing checked predicts which songs: bpm,
our nps, human nps, density ratio, onset density all came back null.

★**Phase was never among the predictors checked**, and there is a mechanism sitting
in our own source. `generate.py` runs `estimate_tempo`, takes `_fit.bpm` — and
merely **logs** `_fit.phase_s`. The beat grid is anchored at **t=0** (`_slot_sec =
(arange(n_slots) / BEAT_SUBDIV) * (60/bpm)`), so a song whose first downbeat is not
at t=0 gets a grid offset by up to half a slot everywhere. That is:
  * a property of the AUDIO ⇒ song-driven and seed-invariant ✓ matches +0.981
  * invisible to bpm / nps / onset density ✓ matches every null predictor
  * already measured once, on the songset (2026-08-02): a global phase shift moved
    precision 0.887 → 0.906 median, but **rescued individual songs dramatically**
    (`1fa48` 0.614 → 0.975) — median |shift| 36.5 ms against a 93 ms slot.

**This script asks whether that generalises to the 149-song cohort, and it needs no
generation**: shifting every note by +δ is exactly equivalent to shifting every
onset by −δ, so the whole sweep is a re-score of maps we already have. (Shifting the
onsets also sidesteps the `copy.deepcopy` landmine — `_load_any`'s `_BM` holds
`color_notes` as a CLASS attribute, so a "copy" shares the note objects.)

🔴**PRE-REGISTERED READING — the human control splits the result in two, and only
half of it is ours to fix (the C2 lesson; a blanket shift is how `h_dist` failed):**

  OURS TO FIX   the bad songs recover materially under a shift AND their human map
                is already fine at δ=0 ⇒ our grid is genuinely misplaced ⇒ wire
                `_fit.phase_s` through as `BEAT_GRID_PHASE`.
  DETECTOR      the human map wants the SAME shift we do ⇒ the offset is in the
                onset detector, not our grid. Correcting it would be fitting the
                detector. Report it, do not build it.
  NOT PHASE     the bad songs do not recover under ANY shift ⇒ phase is exonerated
                and the subset defect is a selection problem. Also a real result:
                it closes the last cheap structural suspect for these 35 songs.

⚠️A per-song argmax over ~100 shift candidates will find SOME gain by chance. The
`same`-tempo, already-good songs act as the built-in null: whatever median gain they
show is the selection floor, and a real effect must clear it.

Usage:
    python scripts/diag_grid_phase.py                      # full cohort
    python scripts/diag_grid_phase.py --limit 20           # quick smoke
    python scripts/diag_grid_phase.py --json outputs/grid_phase_2026-08-13.json
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
from beatsaber_automapper.evaluation.alignment import alignment_metrics  # noqa: E402

COHORT = REPO / "outputs" / "wide_cohort"

# +-120 ms at 2.5 ms resolution. The tolerance is 50 ms and a slot is ~90-190 ms,
# so this covers a full slot either way without pretending to sub-ms precision.
SHIFTS_MS = np.arange(-120.0, 120.0 + 1e-9, 2.5)

# "Bad" = the subset definition already in PROGRESS.md, kept identical so this
# analysis lands on the same 38 songs rather than a fresh judgement call.
BAD_THRESHOLD = -0.10


def _precision_curve(bm, bpm: float, onsets: np.ndarray) -> np.ndarray:
    """Onset precision as a function of a global time shift applied to the MAP.

    Implemented by shifting the onsets the other way, which is identical for a
    nearest-match statistic and leaves the beatmap untouched.
    """
    out = np.empty(len(SHIFTS_MS), dtype=np.float64)
    for i, ms in enumerate(SHIFTS_MS):
        rep = alignment_metrics(bm, bpm=bpm, onsets=onsets - ms / 1000.0)
        p = rep.metrics.get("onset_precision", float("nan"))
        out[i] = p
    return out


def _best(curve: np.ndarray) -> tuple[float, float]:
    """(precision at the best shift, that shift in ms). Ties break toward 0 ms."""
    if not np.isfinite(curve).any():
        return float("nan"), float("nan")
    top = np.nanmax(curve)
    cand = np.flatnonzero(np.isclose(curve, top, atol=1e-12))
    pick = cand[np.argmin(np.abs(SHIFTS_MS[cand]))]
    return float(top), float(SHIFTS_MS[pick])


def _zero(curve: np.ndarray) -> float:
    return float(curve[int(np.argmin(np.abs(SHIFTS_MS)))])


def analyse_song(stem: str) -> dict | None:
    ours_zip = COHORT / f"{stem}.zip"
    human_zip = REPO / "data" / "raw" / f"{stem}.zip"
    if not ours_zip.exists() or not human_zip.exists():
        return None

    # ⚠️BOTH sides must be scored against the SAME onsets. `load_expert_only`
    # returns a 2-tuple and silently yields alignment = nan; that mistake made the
    # first run of the n=149 analysis return 0 scorable songs.
    onsets = scorecard.onsets_for(ours_zip)
    if onsets is None or len(onsets) == 0:
        return None

    ours = scorecard._load_any(ours_zip)
    human = scorecard._load_any(human_zip)
    if ours is None or human is None:
        return None
    o_bm, o_bpm, _ = ours
    h_bm, h_bpm, _ = human

    o_curve = _precision_curve(o_bm, o_bpm, onsets)
    h_curve = _precision_curve(h_bm, h_bpm, onsets)
    o_zero, h_zero = _zero(o_curve), _zero(h_curve)
    o_best, o_shift = _best(o_curve)
    h_best, h_shift = _best(h_curve)
    if not np.isfinite(o_zero) or not np.isfinite(h_zero):
        return None

    return {
        "song": stem,
        "ours_at0": o_zero, "ours_best": o_best, "ours_shift_ms": o_shift,
        "human_at0": h_zero, "human_best": h_best, "human_shift_ms": h_shift,
        "delta_at0": o_zero - h_zero,          # the defect as PROGRESS.md defines it
        "recovered": o_best - o_zero,          # what a perfect phase would buy us
        "residual": o_best - h_zero,           # gap to the human that a shift cannot close
        "bad": bool(o_zero - h_zero < BAD_THRESHOLD),
    }


def _med(rows: list[dict], key: str) -> float:
    vals = [r[key] for r in rows if np.isfinite(r[key])]
    return float(st.median(vals)) if vals else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="score only the first N songs")
    ap.add_argument("--json", type=pathlib.Path, default=None)
    a = ap.parse_args()

    stems = sorted(p.stem for p in COHORT.glob("*.zip"))
    if a.limit:
        stems = stems[:a.limit]

    rows: list[dict] = []
    for i, stem in enumerate(stems, 1):
        r = analyse_song(stem)
        if r:
            rows.append(r)
        if i % 20 == 0:
            print(f"  ... {i}/{len(stems)} ({len(rows)} scorable)", flush=True)

    if not rows:
        print("no scorable songs — check the onset cache")
        return 2

    bad = [r for r in rows if r["bad"]]
    good = [r for r in rows if not r["bad"]]

    print(f"\n=== GRID PHASE vs THE ALIGNMENT SUBSET DEFECT — n={len(rows)} ===")
    print(f"    bad (>{-BAD_THRESHOLD:.2f} below human at 0 ms): {len(bad)}    "
          f"rest: {len(good)}\n")

    hdr = f"{'group':>6} {'n':>4} {'ours@0':>8} {'ours@best':>10} {'recovered':>10} " \
          f"{'|shift|':>8} {'human@0':>8} {'|h shift|':>10} {'residual':>9}"
    print(hdr)
    print("-" * len(hdr))
    for name, grp in (("bad", bad), ("rest", good), ("all", rows)):
        if not grp:
            continue
        print(f"{name:>6} {len(grp):>4} "
              f"{_med(grp, 'ours_at0'):>8.4f} {_med(grp, 'ours_best'):>10.4f} "
              f"{_med(grp, 'recovered'):>10.4f} "
              f"{st.median([abs(r['ours_shift_ms']) for r in grp]):>8.1f} "
              f"{_med(grp, 'human_at0'):>8.4f} "
              f"{st.median([abs(r['human_shift_ms']) for r in grp]):>10.1f} "
              f"{_med(grp, 'residual'):>9.4f}")

    # ⚠️THE SELECTION FLOOR. An argmax over ~97 candidates finds gain by chance; the
    # `rest` group is the built-in null and a real effect must clear its median.
    floor = _med(good, "recovered")
    gain = _med(bad, "recovered")
    print(f"\n  selection floor (median gain on the NON-failing songs): {floor:+.4f}")
    print(f"  median gain on the failing songs:                        {gain:+.4f}")
    print(f"  gain above floor:                                        {gain - floor:+.4f}")

    # ★THE C2 SPLIT — whose defect is it? Only songs where the HUMAN is fine at zero
    # and WE are not are ours to fix; where both want the same shift it is the onset
    # detector, and correcting it would be fitting the detector.
    agree = [r for r in bad if abs(r["ours_shift_ms"] - r["human_shift_ms"]) <= 10.0]
    ours_only = [r for r in bad if abs(r["human_shift_ms"]) <= 10.0
                 and abs(r["ours_shift_ms"]) > 10.0]
    print(f"\n  of the {len(bad)} failing songs:")
    print(f"    {len(ours_only):>3}  human fine at 0, we are not  ->  OUR GRID (fixable)")
    print(f"    {len(agree):>3}  human wants the same shift    ->  ONSET DETECTOR (do not fit)")
    print(f"    {len(bad) - len(ours_only) - len(agree):>3}  neither                       "
          f"->  something else")

    worst = sorted(bad, key=lambda r: r["delta_at0"])[:10]
    print("\n  worst 10 by delta at 0 ms:")
    print(f"    {'song':<8} {'ours@0':>7} {'best':>7} {'shift':>7} "
          f"{'human@0':>8} {'h shift':>8}")
    for r in worst:
        print(f"    {r['song']:<8} {r['ours_at0']:>7.3f} {r['ours_best']:>7.3f} "
              f"{r['ours_shift_ms']:>7.1f} {r['human_at0']:>8.3f} "
              f"{r['human_shift_ms']:>8.1f}")

    if a.json:
        # `.resolve()` because the repo is reached through a symlink
        # (/home/kyle/repos -> /mnt/giga_speed/repos) and a cwd-relative path is
        # not under the resolved REPO, which made `relative_to` raise.
        out = a.json.resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(rows, indent=1))
        print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
