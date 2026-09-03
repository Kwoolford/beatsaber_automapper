#!/usr/bin/env python
"""Calibrate the UNDILUTED alignment floor for `mapjudge` (P0.2, 2026-09-02).

**Why a floor exists at all.** The judge's verdict pools 23 metrics; alignment is two of
them. A map shifted a quarter-beat off the music is extreme on exactly those two and
median on the other twenty-one, and the pooled gate accepted **65 %** of such maps
(PROGRESS 2026-08-21). Every pooled alternative measured worse. What works is the one
thing pooling cannot do: leave alignment undiluted.

**What the floor is**: a raw minimum on `onset_precision` at the q-th percentile of the
human REFERENCE distribution (the same 811-map distribution every other metric's
percentile is read from). Below it the map FAILs regardless of the pooled p-value.
One-sided, one metric -- two-sided or with `offset_mad_ms` folded in under a max was
measured worse on `offbeat` at the same human cost (38 % / 22 % vs ~9 % through).

The script writes the minimum into the reference under `align_floor` and reports the
price on held-out humans and their `offbeat` twins (floor AND pooled gate together),
so the trade is always logged next to the number. Re-run after any change to
`map_record`'s alignment metrics or the reference distributions.

Usage:
    python scripts/calibrate_align_floor.py [--n 300] [--q 0.10] [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import pathlib
import random
import sys

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=300, help="held-out human maps to price on")
    ap.add_argument("--q", type=float, default=0.10,
                    help="human reference percentile the floor sits at (0.10 = the "
                         "worst-aligned tenth of humans fail)")
    ap.add_argument("--dry-run", action="store_true", help="print, do not write")
    a = ap.parse_args()

    from beatsaber_automapper.evaluation import mapjudge as mj
    import audit_mapjudge as A
    from calibrate_mapjudge import CORPUS_OFFSET, corpus

    ref = mj.load_reference()
    dist = np.asarray(ref["distributions"][mj.ALIGN_FLOOR_METRIC], dtype=float)
    floor = float(np.percentile(dist, 100.0 * a.q))
    print(f"{mj.ALIGN_FLOOR_METRIC}: human reference n={len(dist)}  "
          f"median {np.median(dist):.3f}  q{a.q:.2f} = {floor:.3f}  <- floor")

    # Price it on the same held-out slice the P0.2 probes used: never the
    # calibration humans, never the reference distribution itself.
    raws = corpus(0)[CORPUS_OFFSET:]
    span = int(1100 * 1.25) + 40
    held = raws[2 * span:3 * span]
    rng = random.Random(0)
    rows: dict[str, list[tuple[float, bool]]] = {"human": [], "offbeat": []}
    for zp in held:
        if len(rows["human"]) >= a.n:
            break
        loaded = A._load_human(zp)
        if loaded is None:
            continue
        notes, bpm = loaded
        if len(notes) < 50 or not (30.0 < bpm < 400.0):
            continue
        f = REPO / "outputs" / "onset_cache" / f"{zp.stem}.npz"
        if not f.exists():
            continue
        z = np.load(f)
        on = np.asarray(z[list(z.keys())[0]], dtype=float)
        for name, nn in (("human", notes), ("offbeat", A.make_offbeat(list(notes), rng))):
            rec = mj.map_record(nn, bpm, onsets=on)
            r = mj.judge(rec, ref, label=name, align_floor=False)
            rows[name].append((float(rec[mj.ALIGN_FLOOR_METRIC]),
                               r.verdict() == "PASS"))

    n = len(rows["human"])
    if n < 50:
        print(f"only {n} held-out maps with alignment -- refusing to calibrate")
        return 1
    price = {}
    for name, rs in rows.items():
        pooled = sum(1 for _v, ok in rs if ok) / len(rs)
        both = sum(1 for v, ok in rs if ok and v >= floor) / len(rs)
        price[name] = {"pooled_gate": round(pooled, 3), "with_floor": round(both, 3)}
        print(f"  {name:<8} accept: pooled gate {pooled:.3f} -> with floor {both:.3f}")
    print(f"  (n={n} held-out humans and their offbeat twins)")

    if a.dry_run:
        return 0
    ref["align_floor"] = {"metric": mj.ALIGN_FLOOR_METRIC, "min": floor, "q": a.q,
                          "priced_on": n, "accept": price,
                          "written": "2026-09-02 scripts/calibrate_align_floor.py"}
    mj.QUANTILE_PATH.write_text(json.dumps(ref))
    print(f"wrote align_floor to {mj.QUANTILE_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
