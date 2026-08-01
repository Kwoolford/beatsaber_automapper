#!/usr/bin/env python
"""Decompose each v2 axis's `min_spread` into WHICH sub-metric collapsed.

Why this exists (2026-08-01): both Track A candidates clear 4/5 axes and fail the
5th on SPREAD, not gap -- `ds055` fails handrole (0.27 < 0.35) and `ar_xy_ds055`
fails idiom (0.30 < 0.35). `scorecard.py` reports only `min_spread`, the MINIMUM
spread over an axis's sequence keys, so a whole axis can fail on one collapsed
sub-metric while every other key sits in the human range. The scorecard never says
which key it was, and "the generator is mode-collapsed on idiom" is not actionable
while "the generator emits a near-constant `idiom_top50`" is.

The control matters as much as the arms. A cohort spread near 1.0 means "as varied
between songs as human maps are"; well below 1.0 means the generator emits much the
same value whatever the song. But a sub-metric can also be *intrinsically* narrow --
if the HUMAN cohort also comes out low on that key, the 0.35 bar is miscalibrated
for it and the axis is failing on a metric artifact, not on a real collapse. That is
exactly the h_dist trap (docs/eval_suite_v2.md) in a new place, so always read an
arm's column next to the `human` column, never on its own.

Usage:
  python scripts/eval_spread_breakdown.py --arms ds055,ar_xy_ds055,prod --human 24
"""
from __future__ import annotations

import argparse
import pathlib
import random
import statistics
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from beatsaber_automapper.evaluation import scorecard  # noqa: E402

CACHE = REPO / "outputs" / "eval_sweep_cache"
HUMAN_DIR = REPO / "data" / "raw"


def _load_cohort(paths: list[pathlib.Path]) -> list[tuple]:
    out = []
    for p in paths:
        try:
            r = scorecard._load_any(p)
        except Exception:  # noqa: BLE001
            r = None
        if r:
            out.append(r)
    return out


def _records(paths: list[pathlib.Path]) -> list[dict]:
    return [scorecard._metrics_for(bm, bpm) for bm, bpm in _load_cohort(paths)]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arms", default="ds055,ar_xy_ds055,prod",
                    help="comma-separated eval_sweep_cache arm names")
    ap.add_argument("--human", type=int, default=24,
                    help="how many human maps from data/raw to use as the control "
                         "cohort (0 disables)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    cohorts: dict[str, list[dict]] = {}

    if a.human > 0:
        humans = sorted(HUMAN_DIR.glob("*.zip"))
        random.Random(a.seed).shuffle(humans)
        recs = _records(humans[: a.human])
        if recs:
            cohorts["human"] = recs
        print(f"human control: {len(recs)} maps loaded", flush=True)

    for arm in [x.strip() for x in a.arms.split(",") if x.strip()]:
        zips = sorted(CACHE.glob(f"{arm}__*.zip"))
        if not zips:
            print(f"-- {arm}: NO CACHED MAPS, skipping")
            continue
        recs = _records(zips)
        if recs:
            cohorts[arm] = recs
        print(f"{arm}: {len(recs)} maps loaded", flush=True)

    if not cohorts:
        raise SystemExit("no cohorts could be loaded")

    names = list(cohorts)
    for axis_name, mod, gap_key, gap_bar, spread_bar in scorecard.AXES:
        ccs = {n: mod.cohort_comparison(cohorts[n]) for n in names}
        keys = [k for k in mod.SEQUENCE_KEYS if any(k in ccs[n] for n in names)]
        if not keys:
            continue

        print(f"\n=== {axis_name}  (spread bar {spread_bar:.2f}, "
              f"gap bar {gap_bar:.2f}) ===")
        head = f"{'sub-metric':22s}" + "".join(f"{n:>16s}" for n in names)
        print(head)
        print("-" * len(head))
        # shift over spread, per key: spread is what fails the axis, but a key can
        # only be called collapsed if its shift is not also miles off.
        for k in keys:
            row = f"{k:22s}"
            for n in names:
                e = ccs[n].get(k)
                row += ("  " + "n/a".rjust(14)) if e is None else \
                       f"  {e['spread']:6.2f} /{e['shift']:+6.2f}"
            print(row)
        print(f"{'':22s}" + "".join(f"{'spread / shift':>16s}" for _ in names))

        summary = f"{'AXIS min_spread':22s}"
        for n in names:
            s = ccs[n].get("_summary", {})
            ms = s.get("min_spread", float("nan"))
            worst = min((k for k in keys if k in ccs[n]),
                        key=lambda k: ccs[n][k]["spread"], default=None)
            flag = "" if ms >= spread_bar else " FAIL"
            summary += f"  {ms:6.2f}{flag:>8s}"
        print(summary)
        drivers = f"{'  driven by':22s}"
        for n in names:
            worst = min((k for k in keys if k in ccs[n]),
                        key=lambda k: ccs[n][k]["spread"], default=None)
            drivers += f"  {(worst or '-')[:14]:>14s}"
        print(drivers)

    print("\nREAD: spread < 0.35 fails the axis. Compare every arm against the "
          "`human` column —\na key where HUMAN is also low is a miscalibrated bar, "
          "not a collapsed generator.")


if __name__ == "__main__":
    main()
