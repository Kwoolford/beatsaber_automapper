#!/usr/bin/env python
"""DoD harness for the swing simulator (P1-1).

Runs evaluation.swing_sim over a set of human Expert maps and a set of raw
PRE-postprocess V7 maps and prints the verdict:

    DoD MET  <=>  0 violations across ALL human maps  AND  >0 on raw V7 output.

Usage:
    python scripts/eval_swing_sim.py \
        --human data/raw --human-n 40 \
        --v7 outputs/2026-06-07/prepost --v7-glob '*_pre.zip'
"""
from __future__ import annotations

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[0].parent / "src"))

from beatsaber_automapper.evaluation import swing_sim as ss  # noqa: E402


def _run(path: pathlib.Path, difficulty: str) -> dict | None:
    try:
        bm, bpm = ss._load_difficulty(path, difficulty)
    except Exception as e:  # noqa: BLE001
        return {"name": path.name, "error": str(e)}
    card = ss.simulate(bm, bpm=bpm)
    d = card.as_dict()
    d["name"] = path.name
    d["bpm"] = round(bpm, 1)
    d["reset_rate"] = round(card.resets / max(card.n_swings, 1), 3)
    return d


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--human", type=pathlib.Path, default=pathlib.Path("data/raw"))
    ap.add_argument("--human-n", type=int, default=40)
    ap.add_argument("--v7", type=pathlib.Path,
                    default=pathlib.Path("outputs/2026-06-07/prepost"))
    ap.add_argument("--v7-glob", default="*_pre.zip")
    ap.add_argument("--difficulty", default="Expert")
    args = ap.parse_args()

    human_maps = sorted(args.human.glob("*.zip"))[: args.human_n]
    v7_maps = sorted(args.v7.glob(args.v7_glob))

    print(f"=== HUMAN ({len(human_maps)} maps, {args.difficulty}) ===")
    human_viol = 0
    human_offenders = []
    rates = []
    for p in human_maps:
        d = _run(p, args.difficulty)
        if d is None or "error" in d:
            print(f"  SKIP {p.name}: {d.get('error') if d else 'none'}")
            continue
        rates.append(d["reset_rate"])
        if d["violations"] > 0:
            human_viol += d["violations"]
            human_offenders.append((d["name"], d["violations"], d["violation_beats"][:5]))
    avg_rate = sum(rates) / len(rates) if rates else 0.0
    print(f"  total human violations: {human_viol}  (avg reset-rate {avg_rate:.3f})")
    for name, v, beats in human_offenders:
        print(f"    !! {name}: {v} violations @ {beats}")

    print(f"\n=== RAW V7 PRE ({len(v7_maps)} maps) ===")
    v7_viol = 0
    for p in v7_maps:
        d = _run(p, args.difficulty)
        if d is None or "error" in d:
            print(f"  SKIP {p.name}: {d.get('error') if d else 'none'}")
            continue
        v7_viol += d["violations"]
        print(f"  {d['name']:28s} swings={d['n_swings']:5d} "
              f"resets={d['resets']:5d} rate={d['reset_rate']:.2f} "
              f"VIOLATIONS={d['violations']}")

    met = (human_viol == 0) and (v7_viol > 0)
    print("\n" + "=" * 60)
    print(f"DoD: human_violations={human_viol} (need 0), "
          f"v7_violations={v7_viol} (need >0)")
    print(f"VERDICT: {'DoD MET ✓' if met else 'DoD NOT MET ✗'}")
    sys.exit(0 if met else 1)


if __name__ == "__main__":
    main()
