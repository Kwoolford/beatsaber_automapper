#!/usr/bin/env python
"""Is our missing pulse a VOCABULARY problem or an ORDERING problem?

`diag_pulse_union.py` measured both, and the IOI histogram came back a surprise:
our maps use **16 distinct intervals** where the human uses 347, and our top-3
intervals cover **0.756** of all gaps against the human's 0.732. We are more
concentrated than the human, not less -- so "our intervals are smeared / need
quantising" is REFUTED. Yet `pulse_stability` is 0.329 against their 0.514.

`pulse_stability` is the fraction of CONSECUTIVE IOI pairs that are equal, so it
can only be low for one of two reasons: the vocabulary is too wide (refuted), or
the same intervals are emitted in the wrong ORDER.

This separates them with the one control that does it cleanly: **shuffle each
map's own IOI sequence.** A shuffle preserves the histogram exactly and destroys
the ordering, so the shuffled score is what that map would get from its own
vocabulary with no rhythmic memory at all.

    map >> its own shuffle   the map HOLDS intervals -- there is a pulse
    map ~= its own shuffle   the map emits its own vocabulary in random order

★If the human sits far above their shuffle and we sit on ours, the defect is
ordering, and no amount of re-quantising or thinning can touch it.

Reads the IOI sequences `diag_pulse_union.py` already saved -- no rebuild, no GPU.

Usage:
    python scripts/diag_pulse_ordering.py --json outputs/pulse_union_2026-08-20.json
"""
from __future__ import annotations

import argparse
import json
import pathlib
import random
import statistics

ARMS = ("DRUMS", "CARRIER", "UNION", "HUMAN")


def pulse(d: list[float]) -> float:
    """`rhythm.pulse_stability`: share of consecutive IOI pairs that are equal."""
    if len(d) < 2:
        return float("nan")
    return statistics.fmean([1.0 if abs(a - b) < 1e-9 else 0.0
                             for a, b in zip(d, d[1:])])


def shuffled_pulse(d: list[float], reps: int, rng: random.Random) -> float:
    """Mean pulse over `reps` shuffles: the same histogram, no ordering."""
    if len(d) < 2:
        return float("nan")
    out = []
    work = list(d)
    for _ in range(reps):
        rng.shuffle(work)
        out.append(pulse(work))
    return statistics.fmean(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json", type=pathlib.Path,
                    default=pathlib.Path("outputs/pulse_union_2026-08-20.json"))
    ap.add_argument("--reps", type=int, default=40)
    ap.add_argument("--out", type=pathlib.Path)
    a = ap.parse_args()

    data = json.loads(a.json.read_text())
    rng = random.Random(0)
    rows = {k: {"obs": [], "null": [], "lift": []} for k in ARMS}
    for song in data["songs"]:
        for k in ARMS:
            rec = song.get(k)
            if not rec or not rec.get("_iois"):
                continue
            d = [float(x) for x in rec["_iois"]]
            o, n = pulse(d), shuffled_pulse(d, a.reps, rng)
            if o != o or n != n:
                continue
            rows[k]["obs"].append(o)
            rows[k]["null"].append(n)
            rows[k]["lift"].append(o - n)

    print(f"{'arm':<10}{'observed':>10}{'shuffled':>10}{'lift':>9}"
          f"{'songs>null':>12}{'n':>5}")
    print("-" * 56)
    summary = {}
    for k in ARMS:
        r = rows[k]
        if not r["obs"]:
            continue
        n_up = sum(1 for x in r["lift"] if x > 0)
        summary[k] = {"obs": statistics.median(r["obs"]),
                      "null": statistics.median(r["null"]),
                      "lift": statistics.median(r["lift"]),
                      "n_up": n_up, "n": len(r["obs"])}
        s = summary[k]
        print(f"{k:<10}{s['obs']:>10.3f}{s['null']:>10.3f}{s['lift']:>+9.3f}"
              f"{n_up:>9}/{len(r['obs'])}{len(r['obs']):>5}")

    # ---- verdict: an ADDITIVE split, not a winner ----
    # The shuffle gives each map a null built from its OWN histogram, so the total
    # gap decomposes exactly:
    #     gap = (human_null - our_null)   <- vocabulary: how concentrated per song
    #         + (human_lift - our_lift)   <- ordering:   how well it is held
    # Reporting only the lift would call this "ordering" and hide the other half.
    if "UNION" in summary and "HUMAN" in summary:
        hu, us = summary["HUMAN"], summary["UNION"]
        gap = hu["obs"] - us["obs"]
        voc = hu["null"] - us["null"]
        order = hu["lift"] - us["lift"]
        print()
        print(f"pulse gap {gap:+.3f}  =  vocabulary {voc:+.3f}  +  ordering {order:+.3f}")
        print(f"  vocabulary: per-song IOI concentration (human dominant share is the "
              f"driver, not a wider grid)")
        print(f"  ordering:   human holds {hu['lift']:+.3f} above its own shuffle, "
              f"we hold {us['lift']:+.3f}")
        share = order / gap if gap else float("nan")
        if 0.35 <= share <= 0.65:
            print(f"⇒ BOTH, ROUGHLY EVENLY (ordering is {share:.0%} of the gap). A fix "
                  f"must BOTH commit to fewer intervals within a section AND hold each "
                  f"one across consecutive notes. Doing either alone closes about half.")
        elif share > 0.65:
            print(f"⇒ MOSTLY ORDERING ({share:.0%}). Our intervals are human; the "
                  f"SEQUENCE is not.")
        else:
            print(f"⇒ MOSTLY VOCABULARY ({share:.0%}). We spread over more intervals "
                  f"per section than the human does.")
        if hu["null"] > us["obs"]:
            print(f"★A human map with its rhythm RANDOMLY SHUFFLED ({hu['null']:.3f}) "
                  f"still holds a pulse as well as our map does in its intended order "
                  f"({us['obs']:.3f}).")
        print("⚠️A shuffle preserves the histogram EXACTLY, so the lift term is "
              "ordering and nothing else. It says where the defect is, not that a fix "
              "is easy.")
    if a.out:
        a.out.write_text(json.dumps(summary, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
