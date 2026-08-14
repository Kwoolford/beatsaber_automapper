#!/usr/bin/env python
"""Build a `BEAT_BPM_ORACLE` table for the 149-song wide cohort, and characterise
the tempo disagreement it is meant to remove.

**Why this exists.** On 2026-08-11 the baseline was found to FAIL alignment at
n=149 (ours 0.8914 vs human 0.9492, paired, resolvable) and the failure is
BIMODAL and SONG-DRIVEN: we beat the human on 26% of songs and sit >0.10 below on
another 26%, with corr(seed0 delta, seed1 delta) = +0.981. Separately, our
detected BPM disagrees with the human's on 44/149 songs (30%), 28 of them at
exactly half tempo, and a disagreeing song fails alignment at 41% vs 19%.

That is a CORRELATION. `BEAT_BPM_ORACLE` (already in `generate.py`) turns it into
an INTERVENTION: hand the generator the tempo the mapper declared and re-ask.

⚠️This is an EVALUATION instrument, not a production fix — production has no human
map to read a BPM from. What it buys is an **upper bound**: whatever alignment a
perfect tempo recovers is the most any tempo model could ever be worth, and if it
recovers nothing, a tempo model is not the thing to build.

★The pre-registration is already written into `generate.py`'s own docstring: the
oracle fixes **only the tempo**, and the grid is still anchored at t=0, so *"if
alignment does not recover with a perfect tempo, phase is the remaining suspect."*

Usage:
    python scripts/build_true_bpm_wide.py            # write the table + report
    python scripts/build_true_bpm_wide.py --report   # report only, no write
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from feel_disc_poc import _zip_bpm  # noqa: E402

COHORT = REPO / "outputs" / "wide_cohort"
OUT = REPO / "outputs" / "true_bpm_wide_cohort.json"

# A tempo ratio counts as "the same tempo" inside this band. 2% is the band
# bpm_octave_probe.py used against the same ground truth, kept identical so the
# 30%-wrong figure stays comparable across cohorts.
TOL = 0.02


def _ratio_label(ours: float, human: float) -> str:
    """Name the metrical relationship, not just 'wrong'.

    A half-tempo error and a 2:3 misread are different defects: at half tempo the
    finest grid slot is twice as coarse in real time and fast notes cannot be
    represented at all, whereas a 2:3 misread puts the beats in the wrong PLACES.
    Lumping them loses the mechanism.
    """
    r = ours / human
    for name, target in (("same", 1.0), ("half", 0.5), ("double", 2.0),
                         ("two_thirds", 2 / 3), ("three_halves", 1.5),
                         ("quarter", 0.25), ("quadruple", 4.0)):
        if abs(r / target - 1.0) <= TOL:
            return name
    return "other"


def collect() -> list[dict]:
    rows = []
    for zp in sorted(COHORT.glob("*.zip")):
        stem = zp.stem
        raw = REPO / "data" / "raw" / f"{stem}.zip"
        human = _zip_bpm(raw) if raw.exists() else None
        ours = _zip_bpm(zp)
        if not human or not ours:
            # Say which side is missing; a silent drop here would shrink the
            # cohort invisibly, which is exactly how A8 went dead for two nights.
            rows.append({"song": stem, "ours": ours, "human": human,
                         "label": "MISSING"})
            continue
        rows.append({"song": stem, "ours": float(ours), "human": float(human),
                     "ratio": float(ours) / float(human),
                     "label": _ratio_label(float(ours), float(human))})
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", action="store_true",
                    help="print the breakdown without writing the oracle table")
    a = ap.parse_args()

    rows = collect()
    n = len(rows)
    counts: dict[str, int] = {}
    for r in rows:
        counts[r["label"]] = counts.get(r["label"], 0) + 1

    print(f"wide cohort: {n} songs with a generated map")
    for label in sorted(counts, key=lambda k: -counts[k]):
        print(f"  {label:>14}  {counts[label]:3d}  ({counts[label] / n:.1%})")
    disagree = n - counts.get("same", 0) - counts.get("MISSING", 0)
    print(f"\n  DISAGREEING: {disagree}/{n} "
          f"({disagree / n:.1%}) — of which {counts.get('half', 0)} at exactly half tempo")

    if a.report:
        return 0

    table = {r["song"]: r["human"] for r in rows if r["label"] != "MISSING"}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(table, indent=1, sort_keys=True))
    print(f"\nwrote {OUT.relative_to(REPO)}  ({len(table)} entries)")

    side = OUT.with_name("true_bpm_wide_cohort_labels.json")
    side.write_text(json.dumps(rows, indent=1))
    print(f"wrote {side.relative_to(REPO)}  (per-song labels, for splitting the eval)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
