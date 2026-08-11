#!/usr/bin/env python
"""DOES ANY AXIS AGREE WITH KYLE? — the screen that should have run months ago.

Kyle, 2026-08-10: *"The metrics still don't capture the full picture. It may be time
for a significantly different approach."*

He is right, and M-F measured it before he said it: ranking the eval songset by the
mean gap over the steer-safe axes puts **Fallen Kingdom second-best** — the map he
called *"really empty"* — and **Hunger fifth-worst** — the map he graded **A+** and
told us to promote.

M-F stopped at the aggregate. This script asks the question that decides what to build
next: **is the whole suite anti-correlated with his ear, or only the aggregate of it?**
If some axes get his ordering right and others get it backwards, the aggregate is the
problem and the fix is a weighting. If essentially none do, no reweighting of these
axes will produce a quality metric and the source of truth has to change.

⚠️⚠️**THIS IS A SCREEN, NOT A RESULT, AND THE n IS 1.** One judged pair. This project
has written down twice what small n does here — *n=3 lies* (idiom's "resolvable" gain
vanished at n=5) and *n=13 inflates effect sizes 3–20×* — so a single pair cannot
confirm anything about any axis. What it CAN do is bound the hypothesis: an axis that
gets the one known pair backwards is not a candidate quality metric until a real
preference set says otherwise, and if ~half agree, that is exactly the coin-flip the
no-signal hypothesis predicts. **Report the count, do not rank the axes by it.**

★**WHY THE COMPARISON IS gap-to-THIS-SONG's-HUMAN AND NOT THE RAW AXIS.** Comparing a
raw axis value on Hunger against one on Fallen Kingdom is a cross-population
comparison — the mistake this project has made more than any other (W3's 6.5-vs-5.5,
the cohort-nps-vs-corpus-median, the added-notes base rate). Each song's own human map
is the only thing that makes two songs commensurable, so every axis is read as
|ours − that song's human| and "better" means "smaller".

Usage:
    python scripts/preference_screen.py
    python scripts/preference_screen.py --json outputs/preference_screen.json
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

# --------------------------------------------------------------------------------
# THE LEDGER. Every judgement Kyle has given on a specific map file, with the words
# he used. Keep the quote: "really empty" and "A+" are different KINDS of statement
# and a later reader needs to see that, not a number we invented for them.
#
# ⚠️`better` / `worse` must name the maps he ACTUALLY PLAYED. He judged the AFTER2
# (reach) build on 2026-08-03; scoring a different arm's map and calling it his
# verdict would be the ExpertPlus-contamination error in a new costume.
# --------------------------------------------------------------------------------
PAIRS = [
    {
        "id": "2026-08-03/hunger-vs-fallen-kingdom",
        "better": ("1f333", "outputs/kyle_review_2026-08-03/1f333_AFTER2_reach.zip"),
        "worse":  ("1f8d6", "outputs/kyle_review_2026-08-03/1f8d6_AFTER2_reach.zip"),
        "quote_better": "The vast majority of the 1f333 song is A+ and better than "
                        "what we had before so promote it.",
        "quote_worse":  "Fallen Kingdom feels really empty.",
        "note": "Different songs, so read every axis as a gap to THAT song's human. "
                "⚠️He may also be judging on different scales — 'A+ vs what we used "
                "to do' against 'empty vs what the song wants'. Four instruments "
                "already failed to explain 'empty' (PROGRESS.md, 2026-08-04).",
    },
]

NAMES = {"1f333": "Hunger", "1f8d6": "Fallen Kingdom",
         "1f913": "Digital Life Hacker", "1f767": "アリスブルー"}


def masterpiece_axes(song: str, zpath: pathlib.Path) -> dict | None:
    """Every M-axis for one map, plus the same axes for that song's human map."""
    import song_structure as ss
    import eval_motif_rhyme as m1
    import eval_rhythm_fidelity as m2
    import eval_arrangement as m4
    from beatsaber_automapper.evaluation import scorecard, alignment
    from calibrate_playfeel import load_expert_only
    from masterpiece_report import score_one

    L = scorecard._load_any(zpath)
    if not L:
        return None
    bm, bpm = L[0], float(L[1])
    t = np.asarray(alignment.note_times(bm, bpm), dtype=float)
    if len(t) < 100:
        return None
    B = ss.bars(song, bpm, ss.song_end(song, float(t.max())))
    if B is None:
        return None
    A = ss.bar_audio_matrix(song, B)
    stems = m2.stem_onsets(song)
    if A is None or len(stems) < 3:
        return None
    nov = m4.novelty(A)
    bnds = m4.boundaries(nov) if nov is not None else []
    ours = score_one(song, m1.notes_xydc(bm, bpm), B, A, stems, bnds)
    H = load_expert_only(REPO / "data" / "raw" / f"{song}.zip")
    if not H:
        return None
    human = score_one(song, m1.notes_xydc(H[0], float(H[1])), B, A, stems, bnds)
    return {"ours": ours, "human": human}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json", default="")
    a = ap.parse_args()

    out = []
    for pair in PAIRS:
        (sb, pb), (sw, pw) = pair["better"], pair["worse"]
        print(f"\n=== {pair['id']} ===")
        print(f"  HE PREFERRED : {NAMES.get(sb, sb)}  — \"{pair['quote_better']}\"")
        print(f"  HE CRITICISED: {NAMES.get(sw, sw)}  — \"{pair['quote_worse']}\"")
        print(f"  ⚠️{pair['note']}\n")

        B = masterpiece_axes(sb, REPO / pb)
        W = masterpiece_axes(sw, REPO / pw)
        if B is None or W is None:
            print("  could not score one side — skipped")
            continue

        rows = []
        # Only axes present on BOTH sides and on both cohorts. An axis can be
        # missing for a song (too few bars, no lead stem), and silently treating a
        # missing key as a zero gap would invent agreement.
        keys = (set(B["ours"]) & set(B["human"]) & set(W["ours"]) & set(W["human"]))
        for k in sorted(keys):
            if k.endswith(("_null", "_raw", "_global")) or k in (
                    "lead_stem", "n_boundaries", "n_events"):
                continue
            try:
                gb = abs(float(B["ours"][k]) - float(B["human"][k]))
                gw = abs(float(W["ours"][k]) - float(W["human"][k]))
            except (TypeError, ValueError):
                continue
            if not (np.isfinite(gb) and np.isfinite(gw)):
                continue
            rows.append({"axis": k, "gap_preferred": gb, "gap_criticised": gw,
                         "agrees": bool(gb < gw)})

        agree = [r for r in rows if r["agrees"]]
        print(f"  {'axis':22s}{'gap(A+)':>10s}{'gap(empty)':>12s}   verdict")
        print("  " + "-" * 58)
        for r in sorted(rows, key=lambda r: r["axis"]):
            v = "agrees with Kyle" if r["agrees"] else "BACKWARDS"
            print(f"  {r['axis']:22s}{r['gap_preferred']:10.4f}{r['gap_criticised']:12.4f}"
                  f"   {v}")
        n, k = len(rows), len(agree)
        print(f"\n  {k}/{n} axes rank the map he liked as the better one "
              f"({100*k/max(n,1):.0f}%).")
        print("  ⚠️n=1 pair. A coin flip lands near 50% by construction — this bounds"
              "\n    the hypothesis, it does not test any single axis.")
        print("  ⚠️⚠️AND THESE ARE NOT n INDEPENDENT JUDGEMENTS. `audit_axis_redundancy`"
              "\n    found follow_mean/follow_best/follow_drums correlate 0.65-0.84 —"
              "\n    one measurement wearing three names, and the x_* families are"
              "\n    built from shared parts too. The effective count is a small"
              "\n    single digit, so do not read this ratio as a proportion test.")
        out.append({"pair": pair["id"], "n_axes": n, "n_agree": k, "rows": rows})

    if out:
        n = sum(o["n_axes"] for o in out)
        k = sum(o["n_agree"] for o in out)
        print(f"\n=== OVERALL: {k}/{n} axis-judgements agree with Kyle ===")
        print("""
HOW TO READ THIS
  ~50%  the suite carries no signal about his preference in EITHER direction. No
        reweighting of these axes makes a quality metric; the source of truth has to
        change (structured A/B preference set — TODO P0, 2026-08-10).
  >>50% the aggregate was the problem, not the axes. Fit a weighting — but only on a
        real preference set, never on this one pair.
  <<50% the suite is actively anti-correlated, which would be the most useful result
        available: it means the axes measure something real and we have the SIGN of
        its relation to quality wrong.
⚠️Whatever it says, this is one pair. The instrument that settles it does not exist
  yet; it is a few listening sessions with him. That is the ask.""")
    if a.json:
        pathlib.Path(a.json).write_text(json.dumps(out, indent=1))
        print(f"\nwrote {a.json}")


if __name__ == "__main__":
    main()
