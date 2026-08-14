#!/usr/bin/env python
"""THE CROSSOVER GUARD — calibrate the human band, and score a cohort against it.

**Why a guard and not an axis.** `flow.py` computes `crossover` and deliberately
keeps it OUT of the `flow_dist` composite: it is computed from note attributes
independent of ORDER, so it is unchanged by the `shuffled` control and including
it would only dilute that axis's ability to detect destroyed sequencing. The
comment there says it is *"still reported, as guards"* — **and nothing ever
guarded it.** That omission cost the project a whole missing capability:

| | crossover share |
|---|---|
| human (n=150 strict Expert) | median **0.183**, and **0 of 150 maps have none** |
| ours (n=149 wide cohort) | **0.0000 on every single map** |

`enforce_color_separation` with `COLOR_SEP_MODE=full` moves every wrong-side note,
and its own docstring says outright *"this is why our maps measure crossover ==
0.000"*. No axis ever noticed, because the one metric that could see it was wired
into nothing. This file is the guard that TODO's P0 asks for.

★**It is TWO-SIDED, and the lower bound is the important one.** Zero crossovers is
the *non-human* state — no human map in the cohort has none — so a guard that only
caught excess would pass the exact defect we shipped. The upper bound still
matters: a map with random hand assignment sits near 0.5.

⚠️**Calibrated through `load_expert_only`, never `scorecard._load_any`** — `_load_any`
prefers ExpertPlus, and ExpertPlus is denser by construction. That is a standing
methodology rule here, and it is why this does not simply reuse
`flow_human_reference.json`'s crossover entry (median 0.218, n=200 mixed
Standard/Expert from the 2026-07-26 calibration).

⚠️**Ask "norm or aspiration?"** — Kyle's target is the best mappers, so a human
median is a FLOOR for aspirational axes. Crossover is treated as a **norm** axis
here: the defect is that we emit categorically zero, and reaching the human band is
the whole ask. Raising the bar above the human median would need his input.

Usage:
    python scripts/calibrate_crossover.py --n 200          # write the reference
    python scripts/calibrate_crossover.py --score 'outputs/wide_cohort/*.zip'
"""

from __future__ import annotations

import argparse
import glob
import json
import pathlib
import statistics as st
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

REFERENCE = REPO / "docs" / "eval_references" / "crossover_human_reference.json"


def crossover_of(beatmap) -> float:
    """Share of notes whose hand is on the far side of the grid.

    Red (0) belongs in columns 0-1, blue (1) in columns 2-3. Kept identical to
    `flow.py`'s definition so the guard and the reported metric cannot drift apart.
    """
    notes = list(beatmap.color_notes)
    if not notes:
        return float("nan")
    wrong = sum(1 for n in notes
                if (n.color == 0 and n.x >= 2) or (n.color == 1 and n.x <= 1))
    return wrong / len(notes)


def human_values(n: int) -> list[float]:
    from calibrate_playfeel import load_expert_only

    vals = []
    for zp in sorted(pathlib.Path(REPO / "data" / "raw").glob("*.zip")):
        r = load_expert_only(zp)
        if r is None:
            continue
        vals.append(crossover_of(r[0]))
        if len(vals) >= n:
            break
    return vals


def _band(vals: list[float]) -> dict:
    vals = sorted(v for v in vals if v == v)

    def q(p: float) -> float:
        i = min(len(vals) - 1, max(0, int(round(p * (len(vals) - 1)))))
        return vals[i]

    med = st.median(vals)
    return {
        "median": med,
        "mad": st.median([abs(v - med) for v in vals]),
        "p10": q(0.10), "p90": q(0.90),
        "min": vals[0], "max": vals[-1],
        "n": len(vals),
        "n_zero": sum(1 for v in vals if v == 0.0),
    }


def load_reference() -> dict | None:
    if REFERENCE.exists():
        try:
            return json.loads(REFERENCE.read_text())
        except Exception:  # noqa: BLE001
            return None
    return None


def guard(cohort_median: float, ref: dict | None = None) -> tuple[bool, str]:
    """PASS/FAIL plus a reason. The cohort median is the statistic, as elsewhere here."""
    ref = ref or load_reference()
    if ref is None:
        return True, "no reference — guard inactive"
    if cohort_median != cohort_median:
        return True, "crossover not scorable"
    lo, hi = ref["p10"], ref["p90"]
    if cohort_median < lo:
        extra = ("  <- ZERO crossovers is the non-human state; "
                 f"{ref['n_zero']}/{ref['n']} human maps have none"
                 if cohort_median == 0.0 else "")
        return False, f"crossover {cohort_median:.4f} BELOW human p10 {lo:.4f}{extra}"
    if cohort_median > hi:
        return False, f"crossover {cohort_median:.4f} ABOVE human p90 {hi:.4f}"
    return True, f"crossover {cohort_median:.4f} inside human band [{lo:.4f}, {hi:.4f}]"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200, help="human maps to calibrate over")
    ap.add_argument("--score", default=None, metavar="GLOB",
                    help="score a cohort of maps against the stored band")
    a = ap.parse_args()

    if a.score:
        from beatsaber_automapper.evaluation import scorecard
        vals = []
        for m in sorted(glob.glob(a.score)):
            r = scorecard._load_any(pathlib.Path(m))
            if r:
                vals.append(crossover_of(r[0]))
        vals = [v for v in vals if v == v]
        if not vals:
            print("no scorable maps")
            return 2
        med = st.median(vals)
        ok, why = guard(med)
        b = _band(vals)
        print(f"cohort n={len(vals)}  median {med:.4f}  "
              f"p10 {b['p10']:.4f}  p90 {b['p90']:.4f}  zeros {b['n_zero']}/{b['n']}")
        print(f"{'PASS' if ok else 'FAIL'} — {why}")
        return 0 if ok else 1

    vals = human_values(a.n)
    if not vals:
        print("no human maps loaded")
        return 2
    band = _band(vals)
    band["loader"] = "calibrate_playfeel.load_expert_only (strict ExpertStandard)"
    REFERENCE.parent.mkdir(parents=True, exist_ok=True)
    REFERENCE.write_text(json.dumps(band, indent=1))
    print(f"human crossover, n={band['n']} strict-Expert maps")
    for k in ("median", "mad", "p10", "p90", "min", "max", "n_zero"):
        print(f"  {k:>8}  {band[k]}")
    print(f"\nwrote {REFERENCE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
