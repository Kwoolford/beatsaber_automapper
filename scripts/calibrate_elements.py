#!/usr/bin/env python
"""THE HUMAN REFERENCE FOR WALLS, ARCS AND CHAINS — so the agent can judge them.

`agent_mapper/elements.py` made the three non-note elements readable for the first
time. Reading them is not the same as JUDGING them: *"89 walls, wall duty 4.2 %, 19
dodge windows under half a second"* means nothing without knowing what human mappers
do. Quoting a number with no yardstick is the mistake `READING.md` names first.

★**Kyle, 2026-08-24:** the agent should *"evaluate the map correctly and not need to
rely on me to audit."* For notes that is `mapjudge`'s 1 100-map reference. For the other
three elements there has never been one -- **the 23 metrics move by exactly 0.000 when
elements are added**, so no calibration existed to build on.

This builds it from `data/raw` (5 373 maps). ⚠️**No audio and no Demucs**: everything
here comes from the map JSON, so the whole corpus is affordable rather than a sample.

**Percentiles, not means.** The eval suite's own lesson: a cohort statistic is what
lets you say *"this map is at the 3rd human percentile on wall duty"*, which is a
judgement; a mean lets you say *"lower than average"*, which is not.

⚠️Records **presence separately from amount**. 96 % of human maps have walls and 50 %
of v3 maps have chains, so pooling zero-chain maps into a chain-density percentile
would report our 16 chains as unusually high while most maps simply have none.
⇒`*_present` gates, then percentiles over the maps that USE the element.

Usage:
    python scripts/calibrate_elements.py --limit 1500 --json outputs/element_reference.json
"""
from __future__ import annotations

import argparse
import json
import pathlib
import random
import statistics as st
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "agent_mapper"))

import elements as EL  # noqa: E402

# Quantities a player actually perceives, and which the agent must be able to place
# against the corpus.
KEYS = ("walls", "arcs", "chains", "wall_duty", "walls_per_min",
        "arc_share_of_notes", "chain_segments", "notes_in_walls",
        "tight_dodges_lt_0p5s", "min_dodge_s")


def pcts(vals: list[float]) -> dict:
    v = sorted(x for x in vals if x is not None)
    if not v:
        return {}
    q = lambda p: v[min(int(p / 100 * len(v)), len(v) - 1)]  # noqa: E731
    return {"n": len(v), "p5": q(5), "p25": q(25), "median": q(50),
            "p75": q(75), "p95": q(95), "mean": round(st.fmean(v), 4)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", default="")
    a = ap.parse_args()

    zips = sorted((REPO / "data" / "raw").glob("*.zip"))
    random.Random(a.seed).shuffle(zips)
    rows, failed = [], 0
    for zp in zips:
        if len(rows) >= a.limit:
            break
        try:
            e = EL.load_elements(zp)
            if len(e["notes"]) < 100:
                continue
            s = EL.summary(e)
        except Exception:  # noqa: BLE001
            failed += 1
            continue
        s["song"] = zp.stem
        rows.append(s)
        if len(rows) % 250 == 0:
            print(f"  {len(rows)} maps…", flush=True)

    if not rows:
        print("no maps read")
        return 1

    n = len(rows)
    present = {
        "walls": sum(1 for r in rows if r["walls"] > 0) / n,
        "arcs": sum(1 for r in rows if r["arcs"] > 0) / n,
        "chains": sum(1 for r in rows if r["chains"] > 0) / n,
    }
    print(f"\nread {n} human maps ({failed} unreadable)\n")
    print("PRESENCE — what share of human maps use each element at all")
    for k, v in present.items():
        print(f"  {k:8s} {v:6.1%}")

    # ⚠️Percentiles computed over the maps that USE the element, never over all maps.
    ref = {"n_maps": n, "present": present, "dist": {}}
    print("\nAMOUNT — percentiles over the maps that USE the element")
    print(f"  {'key':22s}{'n':>6}{'p5':>9}{'p25':>9}{'median':>9}{'p75':>9}{'p95':>9}")
    for k in KEYS:
        gate = {"arcs": "arcs", "arc_share_of_notes": "arcs",
                "chains": "chains", "chain_segments": "chains"}.get(k, "walls")
        vals = [r[k] for r in rows if r.get(gate, 0) > 0]
        d = pcts(vals)
        if not d:
            continue
        ref["dist"][k] = d
        print(f"  {k:22s}{d['n']:>6}{d['p5']:>9.3g}{d['p25']:>9.3g}"
              f"{d['median']:>9.3g}{d['p75']:>9.3g}{d['p95']:>9.3g}")

    bad = [r for r in rows if r["notes_in_walls"] > 0]
    print(f"\n🔴 human maps with notes trapped inside walls: {len(bad)}/{n} "
          f"({len(bad)/n:.1%})")
    print("   ⇒ if that share is ~0, ANY collision in our map is a defect, not a style.")

    if a.json:
        p = pathlib.Path(a.json)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(ref, indent=2))
        print(f"\nwrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
