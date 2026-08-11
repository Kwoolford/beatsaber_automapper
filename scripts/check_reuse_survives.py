#!/usr/bin/env python
"""DID THE COPY SURVIVE POSTPROCESS? — step 1 of the M-E verdict logic, standalone.

The pre-registered reading of tonight's run starts with a fork: if `harm_place` does
not move, is that because the lever never fired, or because it fired and postprocess
ate the result? `fix_parity` rewrites a large share of cut directions and
`enforce_reachability` moves notes around, and M-E deliberately runs BEFORE both so
they can repair its seams — which is also exactly how a copy gets quietly undone.

This answers it without touching `masterpiece_report`'s cache. ⚠️That matters: the
report caches per (wide-dir, arm) and running it against a half-built arm directory
would poison tonight's eval with partial results.

**The measurement.** On the evaluator's own bar grid, take bar pairs the AUDIO says are
returns of each other, and ask how often the map plays the *same placement pattern* in
both. Compare arm against control on the same songs and the same pairs. A lever that
fired and survived raises it; one that fired and was repaired away does not.

★It reads the FINAL maps and rebuilds the pairs from the evaluator's grid rather than
trusting the generator's internal one, so it cannot inherit a bug from the thing it is
checking.
"""

from __future__ import annotations

import argparse
import glob
import pathlib
import sys

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))


def pattern(V: dict, bi: int) -> tuple:
    """A bar's placement pattern: which slots play, and where/how they cut."""
    r = V["rhythm"][bi]
    p = V["place"][bi]
    return tuple(np.round(np.concatenate([r, p.reshape(-1)]), 3).tolist())


def shared_slot_agreement(V: dict, tgt: int, src: int) -> tuple[int, int]:
    """Placement agreement over the slots BOTH bars play -> (agree, shared).

    🔴WHY THIS EXISTS, AND WHY THE FIRST VERSION OF THIS SCRIPT WAS THE WRONG
    INSTRUMENT. It first asked whether the two bars' patterns were IDENTICAL, and
    read control 0.0000 / arm 0.0006 — a null. But `place` mode never changes which
    slots play, so two bars can only be identical if their rhythms already coincided;
    the test was unreachable by construction for the arm it was pointed at. That is
    this project's signature failure (W4's `tail_ratio`, `harm_place` v1): **the null
    was the instrument.** Identity remains the right test for `full` mode, which does
    copy the rhythm.

    ⚠️AND THE OBVIOUS OBJECTION DOES NOT APPLY HERE. "Mean agreement over shared
    slots" is exactly the shape that broke `harm_place` v1 — it pays you for deleting
    notes, because dropping a note removes a slot where the bars disagreed. It cannot
    do that in this comparison: `place` mode is time-neutral, so the arm and the
    control have the *same notes at the same times*, hence the identical shared-slot
    set. The denominator is fixed across the two sides being compared.
    """
    agree = shared = 0
    for si in range(V["rhythm"].shape[1]):
        if V["rhythm"][tgt, si] and V["rhythm"][src, si]:
            shared += 1
            agree += int(np.allclose(V["place"][tgt, si], V["place"][src, si],
                                     atol=1e-6))
    return agree, shared


def score_dir(d: str, songs: list[str], pairs: dict) -> dict:
    import song_structure as ss
    from beatsaber_automapper.evaluation import scorecard
    import eval_motif_rhyme as m1

    out = {}
    for sid in songs:
        z = pathlib.Path(d) / f"{sid}.zip"
        if not z.exists() or sid not in pairs:
            continue
        L = scorecard._load_any(z)
        if not L:
            continue
        bm, bpm = L[0], float(L[1])
        B = ss.bars(sid, bpm, ss.song_end(sid))
        if B is None:
            continue
        V = ss.map_bar_vectors(m1.notes_xydc(bm, bpm), B)
        same = tot = 0
        ag = sh = 0
        for tgt, src in pairs[sid]:
            if tgt >= B.n or src >= B.n:
                continue
            if V["count"][tgt] < 3 or V["count"][src] < 3:
                continue
            tot += 1
            same += int(pattern(V, tgt) == pattern(V, src))
            a_, s_ = shared_slot_agreement(V, tgt, src)
            ag += a_
            sh += s_
        if tot >= 3 and sh >= 10:
            out[sid] = {"identical": same / tot, "shared_agree": ag / sh,
                        "n_shared": sh}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", required=True, help="arm map directory")
    ap.add_argument("--control", default="outputs/wide_cohort")
    ap.add_argument("--min-sim", type=float, default=0.6)
    ap.add_argument("--min-lag", type=int, default=4)
    ap.add_argument("--min-z", type=float, default=2.0)
    ap.add_argument("--diag", action="store_true",
                    help="rebuild the repeat pairs with the DIAGONAL planner. ⚠️Required "
                         "for any diag_* arm: scoring a diagonal arm against per-bar "
                         "pairs asks whether it reproduced a plan it never used, which "
                         "would understate it for a reason that has nothing to do with "
                         "the lever.")
    ap.add_argument("--min-run", type=int, default=4, help="--diag only")
    a = ap.parse_args()

    import song_structure as ss
    from beatsaber_automapper.evaluation import scorecard
    from beatsaber_automapper.generation.structure_reuse import (
        plan_reuse, plan_reuse_diagonal,
    )

    arm_ids = {pathlib.Path(p).stem for p in glob.glob(f"{a.arm}/*.zip")}
    ctl_ids = {pathlib.Path(p).stem for p in glob.glob(f"{a.control}/*.zip")}
    songs = sorted(arm_ids & ctl_ids)
    print(f"{len(songs)} songs present in BOTH the arm and the control\n")

    # Rebuild the audio's repeat pairs on the EVALUATOR's grid.
    pairs: dict[str, list] = {}
    for sid in songs:
        try:
            L = scorecard._load_any(pathlib.Path(a.control) / f"{sid}.zip")
            if not L:
                continue
            B = ss.bars(sid, float(L[1]), ss.song_end(sid))
            if B is None or B.n < 24:
                continue
            A = ss.bar_audio_matrix(sid, B)
            if A is None:
                continue
            S = {"harm": A["harm"], "rhy": A["rhy"], "energy": A["energy"]}
            if a.diag:
                pl = plan_reuse_diagonal(S, B.edges, min_sim=a.min_sim,
                                         min_lag=a.min_lag, min_run=a.min_run)
            else:
                pl = plan_reuse(S, B.edges, min_sim=a.min_sim, min_lag=a.min_lag,
                                min_z=a.min_z)
            if pl.n_copied:
                pairs[sid] = sorted(pl.source.items())
        except Exception:                                       # noqa: BLE001
            continue
    print(f"{len(pairs)} songs have at least one distinctive musical repeat "
          f"({'diagonal' if a.diag else 'per-bar'} planner)")

    A_ = score_dir(a.arm, songs, pairs)
    C_ = score_dir(a.control, songs, pairs)
    common = sorted(set(A_) & set(C_))
    if not common:
        print("no song scorable on both sides")
        return
    print(f"\nPAIRED over {len(common)} songs, on bar pairs the AUDIO calls a repeat.\n")
    for key, label in (("shared_agree", "placement agreement on slots BOTH bars play"),
                       ("identical", "whole bar pattern IDENTICAL")):
        da = np.array([A_[s][key] for s in common])
        dc = np.array([C_[s][key] for s in common])
        d = da - dc
        se = 2 * d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else float("inf")
        print(f"  -- {label} --")
        print(f"     control {dc.mean():.4f}   arm {da.mean():.4f}   "
              f"Δ {d.mean():+.4f}  (2se {se:.4f})")
        print(f"     resolvable: {'YES' if abs(d.mean()) > se else 'NO (inside noise)'}"
              f"   improved {int((d > 0).sum())}/{len(d)}")
    print("\n  ⚠️`identical` is unreachable for a place-mode arm by construction "
          "(it never\n     changes which slots play) — it is the test for `full`.")
    print("""
READING IT
  Δ clearly positive  the copy fired AND survived postprocess. If harm_place still
                      does not move, the axis is not seeing what the lever does —
                      look at the axis, not the lever.
  Δ ~ 0               the copy did NOT survive. fix_parity / enforce_reachability are
                      repairing it away; the lever would need to run after them, which
                      trades the safety net for the effect.
⚠️This is a manipulation check. It says whether the mechanism engaged. It says nothing
  about whether the map is better — that is Kyle's ear, per M-F.""")


if __name__ == "__main__":
    main()
