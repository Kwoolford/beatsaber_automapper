#!/usr/bin/env python
"""THE HUMAN REFERENCE FOR EMPTINESS — how much of a song do human mappers answer?

`agent_mapper/coverage.py` measures the direction nothing measured before: not *"was
this note motivated?"* but *"was this musical moment answered?"* -- which is what
*"it feels really empty"* means. Reading it is not judging it.

🔴**A low answered-share is NOT automatically a defect.** A song offers far more onsets
than any map should play (C1 measured **4.5 onsets available per note we emit**), so
answering 70 % might be generous or stingy and nothing in this repo could say which.
This builds the reference that decides.

★**No Demucs and no audio decoding**: onsets are already cached for **1 956** songs that
also have a human map in `data/raw`, so the whole reference is affordable.

⚠️**Uses the SAME onset set for both sides.** Both the human map and ours are scored
against `outputs/onset_cache`, which is the fixed point every alignment number in this
repo is measured against. Scoring the human against a different detector would repeat
the `h_dist` mistake -- the failure that hid for months was precisely a human control
that was never run on the same footing.

Usage:
    python scripts/calibrate_coverage.py --limit 1200 --json outputs/coverage_reference.json
"""
from __future__ import annotations

import argparse
import json
import pathlib
import random
import statistics as st
import sys

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "agent_mapper"))

import coverage as CV  # noqa: E402
import elements as EL  # noqa: E402

KEYS = ("answered_overall", "answered_busy_windows", "n_gaps_over_2s",
        "longest_gap_s", "gap_share_of_song")


def pcts(vals) -> dict:
    v = sorted(x for x in vals if x is not None)
    if not v:
        return {}
    q = lambda p: v[min(int(p / 100 * len(v)), len(v) - 1)]  # noqa: E731
    return {"n": len(v), "p5": q(5), "p25": q(25), "median": q(50),
            "p75": q(75), "p95": q(95), "mean": round(st.fmean(v), 4)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=1200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", default="")
    a = ap.parse_args()

    cache = REPO / "outputs" / "onset_cache"
    cands = []
    for f in cache.glob("*.npz"):
        sid = f.stem
        if sid.startswith("audio_"):
            continue
        zp = REPO / "data" / "raw" / f"{sid}.zip"
        if zp.exists():
            cands.append((sid, zp, f))
    random.Random(a.seed).shuffle(cands)
    print(f"{len(cands)} songs have BOTH cached onsets and a human map")

    rows, failed = [], 0
    for sid, zp, f in cands:
        if len(rows) >= a.limit:
            break
        try:
            z = np.load(f)
            on = np.sort(np.asarray(z[list(z.keys())[0]], dtype=float))
            e = EL.load_elements(zp)
            if len(e["notes"]) < 100 or not len(on):
                continue
            s = CV.summary(e, on)
        except Exception:  # noqa: BLE001
            failed += 1
            continue
        s.pop("worst_gaps", None)
        s["song"] = sid
        rows.append(s)
        if len(rows) % 200 == 0:
            print(f"  {len(rows)}…", flush=True)

    if not rows:
        print("no rows")
        return 1
    ref = {"n_maps": len(rows), "dist": {}}
    print(f"\nread {len(rows)} human maps ({failed} unreadable)\n")
    print(f"  {'key':24s}{'n':>6}{'p5':>9}{'p25':>9}{'median':>9}{'p75':>9}{'p95':>9}")
    for k in KEYS:
        d = pcts([r.get(k) for r in rows])
        if not d:
            continue
        ref["dist"][k] = d
        print(f"  {k:24s}{d['n']:>6}{d['p5']:>9.3g}{d['p25']:>9.3g}"
              f"{d['median']:>9.3g}{d['p75']:>9.3g}{d['p95']:>9.3g}")

    print("\n★ HOW TO READ THIS")
    m = ref["dist"].get("answered_overall", {}).get("median")
    if m is not None:
        print(f"  Human mappers answer a MEDIAN {m:.1%} of a song's detected onsets.")
        print("  ⇒ answering everything is not the goal and never was; the song offers")
        print("     several times more events than a map should play.")
    g = ref["dist"].get("longest_gap_s", {})
    if g:
        print(f"  A human map's longest note-free stretch WHILE THE SONG PLAYS runs a")
        print(f"  median {g['median']:.1f}s (p95 {g['p95']:.1f}s) — silence is normal;")
        print(f"  being far above p95 is what 'empty' means.")

    if a.json:
        p = pathlib.Path(a.json)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(ref, indent=2, default=float))
        print(f"\nwrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
