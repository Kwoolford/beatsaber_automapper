#!/usr/bin/env python
"""Build the human idiom VOCABULARY and the A3 metric reference (eval suite v2).

Two artifacts, from two DISJOINT slices of the human corpus — this matters:
if the vocabulary were mined from the same maps the reference is computed on,
those maps' own idioms would be in the vocabulary and their coverage would be
inflated toward 1.0, making the human reference unreachable by anything else.

  outputs/idiom_vocab_human.json      idiom -> count   (the vocabulary itself)
  outputs/idiom_human_reference.json  median/MAD of the A3 metrics

The vocabulary is also the deliverable behind the project goal: a rule-based
mapper can sample from it directly. Top-200 idioms cover ~75% of everything human
mappers do; top-500 covers ~90%.

Usage:
  python scripts/calibrate_idiom.py --n 200
"""
from __future__ import annotations

import argparse
import json
import pathlib
import random
import sys
from collections import Counter

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from beatsaber_automapper.evaluation import idiom  # noqa: E402
from calibrate_flow import load_human  # noqa: E402

RAW = REPO / "data" / "raw"




def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=200, help="maps per slice")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip", type=int, default=32,
                    help="keep both slices clear of the audit's human cohort")
    a = ap.parse_args()

    raws = sorted(RAW.glob("*.zip"))
    random.Random(a.seed).shuffle(raws)
    vocab_slice = raws[a.skip:a.skip + a.n]
    ref_slice = raws[a.skip + a.n:a.skip + 2 * a.n]

    # ---- 1. mine the vocabulary ----
    counts: Counter = Counter()
    n_vocab = 0
    for zp in vocab_slice:
        loaded = load_human(zp)
        if loaded is None:
            continue
        counts.update(idiom.idioms_of(loaded[0]))
        n_vocab += 1
    tot = sum(counts.values())
    idiom.VOCAB_PATH.parent.mkdir(parents=True, exist_ok=True)
    idiom.VOCAB_PATH.write_text(json.dumps(
        {str(k): v for k, v in counts.most_common(4000)}, indent=0))
    idiom._VOCAB_CACHE = None  # force reload with the new vocabulary

    print(f"vocabulary: {n_vocab} maps, {tot} transitions, {len(counts)} distinct idioms")
    for k in (50, 200, 500, 1000):
        cov = sum(v for _, v in counts.most_common(k)) / max(tot, 1)
        print(f"  top {k:5d} cover {cov:.3f}")
    print(f"  -> {idiom.VOCAB_PATH}")

    # ---- 2. metric reference, on the DISJOINT slice ----
    records = []
    for zp in ref_slice:
        loaded = load_human(zp)
        if loaded is None:
            continue
        # load_human already returns a parsed beatmap, not a note list
        records.append(idiom.idiom_metrics(loaded[0]).metrics)
    ref = idiom.calibrate(records)
    idiom.REFERENCE_PATH.write_text(json.dumps(ref, indent=2))

    print(f"\nreference: {len(records)} held-out maps -> {idiom.REFERENCE_PATH}\n")
    print(f"{'metric':20s} {'median':>10s} {'MAD':>10s} {'n':>6s}")
    for k, v in ref.items():
        print(f"{k:20s} {v['median']:10.3f} {v['mad']:10.3f} {v['n']:6d}")


if __name__ == "__main__":
    main()
