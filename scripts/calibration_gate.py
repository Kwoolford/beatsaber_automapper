#!/usr/bin/env python
"""Blind calibration gate for the perception channel (Phase 1, TASK P1-3).

Renders N human + N V7 maps with ANONYMIZED titles ("Sample 03") in shuffled
order, and writes the true labels to a key file that the judge must NOT read
until after committing a ranking. The judge (Claude vision) views the panels,
ranks them by human-likeness, and states reasons; then `--score key.json
ranking.json` checks the DoD:

  human/V7 separate in the ranking  AND  reasons cite the known complaints
  (diagonals, monotony, dead drops).

Usage:
  python scripts/calibration_gate.py render --n 5 --out outputs/calib
  # ... judge views outputs/calib/sample_*.png, writes ranking.json ...
  python scripts/calibration_gate.py score --key outputs/calib/key.json \
        --ranking outputs/calib/ranking.json
"""
from __future__ import annotations

import argparse
import json
import pathlib
import random
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from render_map import render_map  # noqa: E402


def cmd_render(args: argparse.Namespace) -> None:
    rng = random.Random(args.seed)
    human = sorted(pathlib.Path(args.human).glob("*.zip"))
    v7 = sorted(pathlib.Path(args.v7).glob("*.zip"))
    rng.shuffle(human)
    rng.shuffle(v7)

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tmpdir = out / "_staging"
    tmpdir.mkdir(exist_ok=True)

    # Render into staging first (so skips don't shorten the set), THEN shuffle and
    # rename to anonymous sample_NN.png so the order carries no source signal.
    staged: list[tuple[str, pathlib.Path, pathlib.Path]] = []
    for src, pool in (("human", human), ("v7", v7)):
        got = 0
        for p in pool:
            if got >= args.n:
                break
            staging_png = tmpdir / f"{src}_{got}.png"
            try:
                render_map(p, args.difficulty, staging_png,
                           n_panels=args.panels, panel_beats=args.panel_beats,
                           title="__PLACEHOLDER__", with_audio=not args.no_audio)
            except Exception as e:  # noqa: BLE001
                print(f"  SKIP {p.name}: {e}")
                continue
            staged.append((src, p, staging_png))
            got += 1
        if got < args.n:
            print(f"  WARNING: only {got}/{args.n} {src} maps rendered")

    rng.shuffle(staged)
    key = {}
    for i, (src, p, _staging) in enumerate(staged, 1):
        label = f"sample_{i:02d}"
        # re-render with the correct anonymous title now that order is fixed
        render_map(p, args.difficulty, out / f"{label}.png",
                   n_panels=args.panels, panel_beats=args.panel_beats,
                   title=f"Sample {i:02d}", with_audio=not args.no_audio)
        key[label] = {"source": src, "map": p.name}
    import shutil
    shutil.rmtree(tmpdir, ignore_errors=True)
    (out / "key.json").write_text(json.dumps(key, indent=2))
    print(f"rendered {len(staged)} samples -> {out}/sample_*.png")
    print(f"key (DO NOT READ until after ranking) -> {out}/key.json")


def cmd_score(args: argparse.Namespace) -> None:
    key = json.loads(pathlib.Path(args.key).read_text())
    ranking = json.loads(pathlib.Path(args.ranking).read_text())
    # ranking: {"order": [labels most->least human], "reasons": "..."}
    order = ranking["order"]
    n = len(order)
    half = n // 2
    top_half = set(order[:half])  # predicted most human

    # DoD 1: separation. Count humans in the top half (should be ~all).
    humans_in_top = sum(1 for lbl in top_half if key[lbl]["source"] == "human")
    n_human = sum(1 for v in key.values() if v["source"] == "human")
    clean = humans_in_top == n_human and humans_in_top == half

    # DoD 2: reasons cite known complaints
    reasons = ranking.get("reasons", "").lower()
    complaints = {
        "diagonals": any(w in reasons for w in ("diagonal", "for-sport", "for sport")),
        "monotony": any(w in reasons for w in ("monoton", "flat", "repetit", "uniform", "same")),
        "dead drops": any(w in reasons for w in ("dead drop", "dead-drop", "empty", "silence", "gap")),
    }

    print("=== calibration gate score ===")
    print(f"ranking (most->least human): {order}")
    for lbl in order:
        print(f"  {lbl}: {key[lbl]['source']:5s}  {key[lbl]['map']}")
    print(f"\nDoD-1 separation: humans_in_top_half={humans_in_top}/{n_human} "
          f"-> {'CLEAN' if clean else 'NOT clean'}")
    print(f"DoD-2 reasons cite complaints: {complaints}")
    passed = clean and all(complaints.values())
    print(f"\nVERDICT: {'GATE PASSED ✓' if passed else 'GATE FAILED ✗'}")
    sys.exit(0 if passed else 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("render")
    r.add_argument("--n", type=int, default=5, help="maps per source")
    r.add_argument("--human", default="data/raw")
    r.add_argument("--v7", default="outputs/v7_cohort_2026-06-10")
    r.add_argument("--out", default="outputs/calib")
    r.add_argument("--difficulty", default="Expert")
    r.add_argument("--panels", type=int, default=4)
    r.add_argument("--panel-beats", type=float, default=8.0)
    r.add_argument("--seed", type=int, default=20260615)
    r.add_argument("--no-audio", action="store_true")
    r.set_defaults(func=cmd_render)

    s = sub.add_parser("score")
    s.add_argument("--key", required=True)
    s.add_argument("--ranking", required=True)
    s.set_defaults(func=cmd_score)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
