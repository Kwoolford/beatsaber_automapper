#!/usr/bin/env python
"""Re-rank every cached arm now that the suite can hear the music (axis A8).

Step 5 of the A8 task (TODO.md, 2026-08-01): the suite gained an axis, so every
verdict it has ever issued is provisional until re-scored. The specific thing this
answers is the one that made A8 necessary — **five configurations passed all five
axes and one of them still sounded off the beat to Kyle.** If A8 is doing its job,
those arms drop to 5/6 and the leaderboard reorders.

Reports, per arm: the six axis gaps, the pass count, and — the column that matters —
`onset_precision`, the share of that arm's notes landing on a real audio onset.
Human maps sit at ~0.97; the best arm measured before this script existed sat at
0.82.

Two arms are singled out because Kyle listened to them and ranked them:
`hl014_ds055` ("a noticeable step up") above `b1_e17_ds055` ("a lot wrong"). Any
re-ranking that inverts that pair is wrong no matter how principled it looks.

CPU-only — safe to run alongside a GPU job.

Usage:
  python scripts/rerank_with_alignment.py --json outputs/rerank_a8_2026-08-02.json
  python scripts/rerank_with_alignment.py --arms prod,ds055,hl014_ds055,b1_e17_ds055
"""
from __future__ import annotations

import argparse
import json
import pathlib
import statistics
import sys
import time

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from beatsaber_automapper.evaluation import scorecard  # noqa: E402

CACHE = REPO / "outputs" / "eval_sweep_cache"
# Kyle's ordering, 2026-08-01 — better first.
KYLE_PAIR = ("hl014_ds055", "b1_e17_ds055")


def arms_in_cache() -> list[str]:
    return sorted({p.stem.split("__")[0] for p in CACHE.glob("*__*.zip")})


def score_arm(arm: str, min_maps: int) -> dict | None:
    loaded = []
    for zp in sorted(CACHE.glob(f"{arm}__*.zip")):
        try:
            r = scorecard._load_any(zp)
        except Exception:  # noqa: BLE001
            r = None
        if r:
            loaded.append(r)
    if len(loaded) < min_maps:
        return None
    res = scorecard.score_cohort(loaded, arm)
    axes = {ax.name: ax for ax in res["axes"]}
    prec = [r["onset_precision"] for r in res["records"]
            if r.get("onset_precision") is not None
            and r["onset_precision"] == r["onset_precision"]]
    return {
        "arm": arm,
        "n_maps": len(loaded),
        "gaps": {n: ax.gap for n, ax in axes.items()},
        "spreads": {n: ax.min_spread for n, ax in axes.items()},
        "passed": {n: ax.passed for n, ax in axes.items()},
        "n_pass": sum(1 for ax in axes.values() if ax.passed),
        "viol": res["total_viol"],
        "overall": res["passed"],
        "onset_precision": statistics.median(prec) if prec else float("nan"),
        # What the arm's verdict was BEFORE A8 existed — the comparison that shows
        # what the new axis actually changed.
        "passed_old5": all(ax.passed for n, ax in axes.items() if n != "alignment")
        and res["total_viol"] == 0,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arms", help="comma-separated (default: every arm in the cache)")
    ap.add_argument("--min-maps", type=int, default=20)
    ap.add_argument("--json", help="write full results here")
    a = ap.parse_args()

    arms = [x for x in a.arms.split(",") if x] if a.arms else arms_in_cache()
    print(f"re-scoring {len(arms)} arms on SIX axes (A8 = audio alignment)\n", flush=True)

    rows = []
    t0 = time.time()
    for i, arm in enumerate(arms, 1):
        r = score_arm(arm, a.min_maps)
        if r is None:
            continue
        rows.append(r)
        print(f"  [{i}/{len(arms)}] {arm:24s} {r['n_pass']}/6 "
              f"align_gap={r['gaps'].get('alignment', float('nan')):6.2f} "
              f"prec={r['onset_precision']:.3f}", flush=True)
    print(f"\nscored {len(rows)} arms in {time.time() - t0:.0f}s\n")

    rows.sort(key=lambda r: (r["gaps"].get("alignment", float("inf"))))
    names = ["flow", "rhythm", "idiom", "handrole", "playfeel", "alignment"]
    print("=== LEADERBOARD, best AUDIO ALIGNMENT first ===")
    print(f"{'arm':26s}" + "".join(f"{n[:6]:>8s}" for n in names)
          + f"{'prec':>8s}{'pass':>6s}  old5")
    print("-" * 100)
    for r in rows:
        line = f"{r['arm'][:25]:26s}"
        for n in names:
            g = r["gaps"].get(n, float("nan"))
            line += f"{g:8.2f}"
        line += f"{r['onset_precision']:8.3f}{r['n_pass']:5d}/6"
        line += "   PASSED-5" if r["passed_old5"] else ""
        print(line)

    old5 = [r for r in rows if r["passed_old5"]]
    new6 = [r for r in rows if r["overall"]]
    print(f"\n--- WHAT A8 CHANGED ---")
    print(f"  arms passing the OLD five axes : {len(old5)}"
          f"  ({', '.join(r['arm'] for r in old5) or 'none'})")
    print(f"  arms passing ALL SIX           : {len(new6)}"
          f"  ({', '.join(r['arm'] for r in new6) or 'none'})")
    demoted = [r["arm"] for r in old5 if not r["overall"]]
    if demoted:
        print(f"  DEMOTED by alignment           : {', '.join(demoted)}")
        print("  => these are the maps Kyle called off-beat. The suite now agrees.")
    elif old5:
        print("  Nothing was demoted. Either A8's bar is too loose, or the arms really")
        print("  are aligned and Kyle's complaint is about something else — check the")
        print("  onset_precision column against the human ~0.97 before believing it.")

    by_arm = {r["arm"]: r for r in rows}
    if all(k in by_arm for k in KYLE_PAIR):
        ga, gb = (by_arm[k]["gaps"].get("alignment", float("nan")) for k in KYLE_PAIR)
        ok = ga < gb
        print(f"\n--- KYLE-ORDERING CHECK ---\n  {KYLE_PAIR[0]} {ga:.2f} "
              f"{'<' if ok else '>='} {KYLE_PAIR[1]} {gb:.2f}  -> "
              f"{'AGREES with his ear' if ok else 'DISAGREES — the axis is not ready'}")

    if a.json:
        pathlib.Path(a.json).write_text(json.dumps(rows, indent=2))
        print(f"\nwrote {a.json}")


if __name__ == "__main__":
    main()
