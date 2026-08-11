#!/usr/bin/env python
"""THE SENSITIVITY BATTERY — what can the suite NOT SEE AT ALL?

Kyle, 2026-08-11: *"keep prodding the visibility suite to identify more blind spots."*

★**THIS ASKS A DIFFERENT QUESTION FROM EVERY AUDIT ALREADY HERE, AND THE DIFFERENCE IS
THE WHOLE POINT.** `audit_eval_suite.py` and `audit_masterpiece.py` are **degeneracy**
batteries: they build maps nobody would call good — a metronome, random attributes, a
bar-rotated map — and check the suite ranks them BELOW a human. That tests whether a
metric can be *fooled*.

It does not test whether a metric can *see*. On 2026-08-11 the M-E lever rewrote the
position and cut direction of **25 % of every map's notes** and **twelve of the fifteen
masterpiece axes moved by exactly +0.0000**. Nothing was fooled; the suite simply had no
opinion. A lever operating in a dimension no axis measures is **unmeasurable, not
neutral** — and this project has already shipped one of those (`BEAT_ONSET_EVIDENCE`
degraded reachability and no axis noticed until Kyle's ear forced the metric to exist).

**The method.** Take our own production maps. Apply a perturbation a human would
obviously notice but which leaves the map broadly playable. Re-score. Any axis that does
not move is blind to that dimension.

    perturbation moves an axis          ⇒ the suite can see that dimension
    perturbation moves NOTHING          ⇒ 🔴BLIND SPOT: a lever here is unmeasurable

⚠️**A blind spot is not automatically a metric to build.** It is a warning label: it
says *"if a change lives here, this suite will report nothing, and silence will look
like safety."* Whether it matters is Kyle's ear — the axes already rank his A+ map fifth
worst, so more axes is not obviously the answer (see the 2026-08-10 P0).

⚠️Perturbations are applied to OUR maps, never the human corpus — human zips carry the
`ExpertPlus` loader landmine, and our generated maps only ever contain
`ExpertStandard.dat`.

Usage:
    python scripts/audit_sensitivity.py --n 40
    python scripts/audit_sensitivity.py --n 40 --json outputs/sensitivity.json
"""

from __future__ import annotations

import argparse
import copy
import glob
import json
import pathlib
import random
import sys

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

# cut direction -> its mirror across the vertical axis, and its opposite
MIRROR_DIR = {0: 0, 1: 1, 2: 3, 3: 2, 4: 5, 5: 4, 6: 7, 7: 6, 8: 8}
OPPOSITE_DIR = {0: 1, 1: 0, 2: 3, 3: 2, 4: 7, 5: 6, 6: 5, 7: 4, 8: 8}


# --------------------------------------------------------------- perturbations
# Each takes (notes, rng) and mutates in place. Every one is something a player
# would notice immediately; none of them turn the map into rubble.

def p_mirror_x(notes, rng):
    """Reflect the map left-right. Same shapes, opposite hand geography."""
    for n in notes:
        n.x = 3 - n.x
        n.direction = MIRROR_DIR.get(n.direction, n.direction)


def p_swap_colors(notes, rng):
    """Red becomes blue. Every hand-role convention in the map inverts."""
    for n in notes:
        n.color = 1 - n.color


def p_flatten_rows(notes, rng):
    """Collapse to one row. All vertical variety gone."""
    for n in notes:
        n.y = 1


def p_flatten_cols(notes, rng):
    """Collapse to two adjacent columns. All horizontal spread gone."""
    for n in notes:
        n.x = 1 if n.x <= 1 else 2


def p_all_dots(notes, rng):
    """Every note becomes a dot — all cut-direction information erased."""
    for n in notes:
        n.direction = 8


def p_reverse_dirs(notes, rng):
    """Every cut direction reversed. Parity and flow become nonsense."""
    for n in notes:
        n.direction = OPPOSITE_DIR.get(n.direction, n.direction)


def p_rows_random(notes, rng):
    """Randomise the row only. Column and timing untouched."""
    for n in notes:
        n.y = rng.randint(0, 2)


def p_cols_random(notes, rng):
    """Randomise the column only. Row and timing untouched."""
    for n in notes:
        n.x = rng.randint(0, 3)


def p_shift_20ms(notes, rng):
    _shift(notes, 0.020)


def p_shift_60ms(notes, rng):
    _shift(notes, 0.060)


def _shift(notes, secs):
    # applied in beats by the caller via bpm; stored here as a marker
    for n in notes:
        n.beat += secs * getattr(n, "_bps", 2.0)


def p_drop_double_partner(notes, rng):
    """Remove the second note of every simultaneous pair — doubles become singles."""
    seen = {}
    drop = []
    for i, n in enumerate(notes):
        k = round(n.beat, 4)
        if k in seen:
            drop.append(i)
        else:
            seen[k] = i
    for i in reversed(drop):
        notes.pop(i)


PERTURBATIONS = {
    "mirror_x": p_mirror_x,
    "swap_colors": p_swap_colors,
    "flatten_rows": p_flatten_rows,
    "flatten_cols": p_flatten_cols,
    "all_dots": p_all_dots,
    "reverse_dirs": p_reverse_dirs,
    "rows_random": p_rows_random,
    "cols_random": p_cols_random,
    "drop_double_partner": p_drop_double_partner,
}
TIME_PERTURBATIONS = {"shift_20ms": 0.020, "shift_60ms": 0.060}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--src", default="outputs/wide_cohort")
    ap.add_argument("--json", default="")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    from beatsaber_automapper.evaluation import scorecard

    files = sorted(glob.glob(f"{a.src}/*.zip"))[: a.n]

    def fresh():
        """Re-read every map from disk.

        🔴**LANDMINE, AND IT INVALIDATED THIS SCRIPT'S FIRST RESULTS.**
        `scorecard._load_any` builds a *local* class `_BM` whose `color_notes` is a
        **class attribute** closed over the parsed list. So `copy.deepcopy(bm)` returns
        a new `_BM` instance that **still shares the same note list and the same note
        objects** — verified: `deepcopy(bm).color_notes is bm.color_notes` is True.
        Mutating "a copy" therefore corrupts the original, and in a loop over
        perturbations every row after the first is contaminated by all the rows before
        it. The first table this script printed had exactly that signature — the last
        three perturbations agreeing to three decimals, which this project's own rule
        calls a construction rather than a result.
        Each `_load_any` call defines a NEW `_BM`, so re-reading is the isolation.
        """
        out = []
        for f in files:
            L = scorecard._load_any(pathlib.Path(f))
            if L:
                out.append(L)
        return out

    base = fresh()
    print(f"loaded {len(base)} maps from {a.src}\n")

    ref = scorecard.score_cohort(fresh(), "unperturbed")
    ref_gaps = {ax.name: ax.gap for ax in ref["axes"]}
    print("unperturbed baseline:")
    for k, v in ref_gaps.items():
        print(f"    {k:10s} {v:.4f}")
    print()

    rows = []
    all_names = list(PERTURBATIONS) + list(TIME_PERTURBATIONS)
    for name in all_names:
        rng = random.Random(a.seed)
        cohort = []
        for L in fresh():                       # ⚠️re-read; see `fresh()` above
            bm = L[0]
            bpm = float(L[1])
            if name in TIME_PERTURBATIONS:
                d_beats = TIME_PERTURBATIONS[name] * bpm / 60.0
                for n in bm.color_notes:
                    n.beat += d_beats
            else:
                PERTURBATIONS[name](bm.color_notes, rng)
            cohort.append((bm, bpm, L[2] if len(L) > 2 else None))
        n_notes = sum(len(c[0].color_notes) for c in cohort)
        res = scorecard.score_cohort(cohort, name)
        gaps = {ax.name: ax.gap for ax in res["axes"]}
        moved = {}
        for k, v in gaps.items():
            r = ref_gaps.get(k, float("nan"))
            if not (np.isfinite(v) and np.isfinite(r)):
                moved[k] = None
                continue
            moved[k] = abs(v - r)
        rows.append({"perturbation": name, "gaps": gaps, "delta": moved,
                     "n_notes": n_notes})

    # ⚠️SANITY GATE. `drop_double_partner` silently removed nothing in the first run
    # (the shared-list bug), and a no-op perturbation looks exactly like a blind spot.
    base_notes = sum(len(L[0].color_notes) for L in base)
    for r in rows:
        if r["perturbation"] == "drop_double_partner" and r["n_notes"] >= base_notes:
            print(f"⚠️{r['perturbation']} removed NO notes ({r['n_notes']} vs "
                  f"{base_notes}) — it is a no-op, not a blind spot. Fix it before "
                  f"reading its row.\n")

    axes = [ax.name for ax in ref["axes"]]
    print(f"{'perturbation':22s}" + "".join(f"{ax[:9]:>10s}" for ax in axes))
    print("-" * (22 + 10 * len(axes)))
    for r in rows:
        line = f"{r['perturbation']:22s}"
        for ax in axes:
            d = r["delta"].get(ax)
            line += f"{'  n/a':>10s}" if d is None else f"{d:10.3f}"
        print(line)

    print("\n🔴 BLIND SPOTS — perturbations no axis notices (|Δgap| < 0.02 everywhere):")
    any_blind = False
    for r in rows:
        ds = [d for d in r["delta"].values() if d is not None]
        if ds and max(ds) < 0.02:
            print(f"    {r['perturbation']:22s} max |Δ| = {max(ds):.4f}")
            any_blind = True
    if not any_blind:
        print("    (none at this threshold — every perturbation moved at least one axis)")

    print("""
HOW TO READ THIS
  A large Δ means the suite SEES that dimension; a lever there is measurable.
  A Δ near zero everywhere means the suite is BLIND: a lever operating in that
  dimension would report as "no regression on any axis" while changing the map
  substantially. That is not safety, it is silence.
⚠️This says nothing about whether a dimension MATTERS to a player. It says whether we
  could tell. Kyle's ear decides the first question; this decides the second.""")

    if a.json:
        pathlib.Path(a.json).write_text(json.dumps(
            {"baseline": ref_gaps, "rows": rows}, indent=1, default=float))
        print(f"\nwrote {a.json}")


if __name__ == "__main__":
    main()
