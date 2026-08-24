#!/usr/bin/env python
"""HUMAN PER-TRANSITION EFFORT — so "hard" means harder than humans ask for.

`agent_mapper/effort.py` computes what the wrists do between consecutive same-hand
notes. Without a reference those are physics, not judgement: 10.3 grid-units/s means
nothing until you know human mappers routinely ask for it.

★**Per-TRANSITION percentiles, not per-map.** The question is *"would a player notice
THIS transition"*, so the distribution has to be over transitions pooled across maps.
⚠️That makes n very large (millions), so this samples maps and reports exact quantiles
over the pooled sample.

⚠️**No audio needed** -- everything comes from the map JSON, so a large sample is cheap.

Usage:
    python scripts/calibrate_effort.py --limit 400 --json outputs/effort_reference.json
"""
from __future__ import annotations

import argparse
import json
import pathlib
import random
import sys

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "agent_mapper"))

import effort as EF  # noqa: E402
import elements as EL  # noqa: E402


def q(v: np.ndarray) -> dict:
    return {"n": int(v.size),
            "p5": float(np.percentile(v, 5)), "p25": float(np.percentile(v, 25)),
            "median": float(np.percentile(v, 50)), "p75": float(np.percentile(v, 75)),
            "p95": float(np.percentile(v, 95)), "p99": float(np.percentile(v, 99))}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", default="")
    a = ap.parse_args()

    zips = sorted((REPO / "data" / "raw").glob("*.zip"))
    random.Random(a.seed).shuffle(zips)
    sp, ro, rs, nmaps, failed = [], [], [], 0, 0
    for zp in zips:
        if nmaps >= a.limit:
            break
        try:
            e = EL.load_elements(zp)
            if len(e["notes"]) < 100:
                continue
            tr = EF.transitions(e)
        except Exception:  # noqa: BLE001
            failed += 1
            continue
        if not tr:
            continue
        sp.extend(r["speed"] for r in tr)
        ro.extend(r["rotation"] for r in tr)
        rs.append(float(np.mean([r["reset"] for r in tr])))
        nmaps += 1
        if nmaps % 100 == 0:
            print(f"  {nmaps} maps, {len(sp)} transitions…", flush=True)

    if not sp:
        print("nothing read")
        return 1
    ref = {"n_maps": nmaps, "speed": q(np.array(sp)), "rotation": q(np.array(ro)),
           "reset_share_per_map": q(np.array(rs))}
    print(f"\n{nmaps} human maps · {len(sp):,} transitions ({failed} unreadable)\n")
    for k in ("speed", "rotation", "reset_share_per_map"):
        d = ref[k]
        print(f"  {k:22s}p5 {d['p5']:8.3g}  p50 {d['median']:8.3g}  "
              f"p95 {d['p95']:8.3g}  p99 {d['p99']:8.3g}")
    print("\n★ 'hard' = above the human p95. Humans put 5 % of their OWN transitions")
    print("  there by definition, so a map far above 0.05 is asking more than they do.")
    if a.json:
        p = pathlib.Path(a.json)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(ref, indent=2))
        print(f"\nwrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
