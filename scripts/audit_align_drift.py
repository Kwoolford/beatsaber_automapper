#!/usr/bin/env python
"""Control battery for the K1 within-song drift metric.

Every new metric in this project passes the degenerate-control battery before it
is allowed to steer the generator (`docs/eval_suite_v2.md`). Drift needs a
battery of its own, because it is a **conditional** metric and the standard one
would flatter it: a randomised map has uniformly low precision, so its
*first-fifth minus last-fifth* is ~0 and it looks pristine. Passing "random"
here would mean nothing.

So this battery asks two questions, and a metric must answer both:

  NEGATIVE — does it stay quiet on degenerate maps? Not because it is sharp, but
             because it is blind to them. That is fine PROVIDED drift is never
             read on its own; A8's precision gate is what rejects those maps.
             This run measures how blind, so the dependency is documented rather
             than assumed.

  POSITIVE — does it FIRE on the defect it claims to detect? A metric that never
             produces a large value on a map built to have the defect is not a
             metric. `decay_*` takes a human map and progressively displaces the
             last fifth of its notes off the grid, which is precisely "timing
             degrades toward the end of the song" and nothing else.

Usage:
    python scripts/audit_align_drift.py --n 20
"""

from __future__ import annotations

import argparse
import pathlib
import random
import statistics as st
import sys

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from beatsaber_automapper.data.beatmap import ColorNote  # noqa: E402
from beatsaber_automapper.evaluation import alignment, scorecard  # noqa: E402

from eval_align_drift import drift_metrics  # noqa: E402


def _decay(notes: list[ColorNote], frac: float, amount_beats: float,
           rng: random.Random) -> list[ColorNote]:
    """Displace the last `frac` of notes by up to +-amount_beats. The POSITIVE control."""
    if not notes:
        return []
    k = int(len(notes) * (1.0 - frac))
    out = []
    for i, n in enumerate(notes):
        b = n.beat
        if i >= k and amount_beats > 0:
            b = b + rng.uniform(-amount_beats, amount_beats)
        out.append(ColorNote(beat=b, x=n.x, y=n.y, color=n.color,
                             direction=n.direction))
    return out


def _rand_times(notes: list[ColorNote], rng: random.Random) -> list[ColorNote]:
    if not notes:
        return []
    lo, hi = notes[0].beat, notes[-1].beat
    beats = sorted(rng.uniform(lo, hi) for _ in notes)
    return [ColorNote(beat=b, x=n.x, y=n.y, color=n.color, direction=n.direction)
            for b, n in zip(beats, notes)]


def _jitter(notes: list[ColorNote], rng: random.Random) -> list[ColorNote]:
    return [ColorNote(beat=n.beat + rng.uniform(-0.04, 0.04), x=n.x, y=n.y,
                      color=n.color, direction=n.direction) for n in notes]


class _BM:
    def __init__(self, notes):
        self.color_notes = notes
        self.bomb_notes: list = []


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=20, help="human maps to use")
    a = ap.parse_args()

    rng = random.Random(0)
    zips = sorted((REPO / "data" / "raw").glob("*.zip"))
    rows: dict[str, list[dict]] = {}
    prec: dict[str, list[float]] = {}
    used = 0

    variants = {
        "human": lambda ns: ns,
        "timing_random": lambda ns: _rand_times(ns, rng),
        "timing_jitter": lambda ns: _jitter(ns, rng),
        "decay_0.25b": lambda ns: _decay(ns, 0.2, 0.25, rng),
        "decay_0.50b": lambda ns: _decay(ns, 0.2, 0.50, rng),
        "decay_1.00b": lambda ns: _decay(ns, 0.2, 1.00, rng),
    }

    for zp in zips:
        if used >= a.n:
            break
        try:
            loaded = scorecard._load_any(zp)
        except Exception:  # noqa: BLE001
            continue
        if not loaded:
            continue
        bm, bpm, ons = loaded
        if ons is None or len(ons) == 0:
            continue
        base = list(bm.color_notes)
        if len(base) < 200:
            continue
        ok = False
        for name, fn in variants.items():
            r = drift_metrics(_BM(fn(base)), bpm=bpm, onsets=ons)
            if not r:
                continue
            rows.setdefault(name, []).append(r)
            p = alignment.alignment_metrics(_BM(fn(base)), bpm=bpm,
                                            onsets=ons).metrics["onset_precision"]
            prec.setdefault(name, []).append(p)
            ok = True
        used += int(ok)

    if not rows:
        sys.exit("nothing scored — is the onset cache populated?")

    print(f"=== DRIFT CONTROL BATTERY (n={used} human maps) ===\n")
    print(f"{'variant':16s}{'precision':>11s}{'drift med':>11s}{'drift p90':>11s}"
          f"{'share>0.145':>13s}")
    print("-" * 62)
    for name in variants:
        if name not in rows:
            continue
        d = [r["drift_q1_q5"] for r in rows[name]]
        share = sum(1 for x in d if x > 0.1451) / len(d)
        print(f"{name:16s}{st.mean(prec[name]):>11.3f}{st.median(d):>11.4f}"
              f"{float(np.percentile(d, 90)):>11.4f}{share:>13.1%}")

    print("\n--- READ ---")
    print("NEGATIVE controls (timing_random / timing_jitter): drift should stay")
    print("  SMALL while precision collapses. That is the metric being blind, not")
    print("  sharp -- drift is conditional and must never be read without A8's")
    print("  precision gate beside it.")
    print("POSITIVE controls (decay_*): drift MUST rise with the displacement, and")
    print("  the share past the human p90 must climb toward 100%. If it does not,")
    print("  the metric cannot see the defect it was built for and nothing")
    print("  measured with it stands.")


if __name__ == "__main__":
    main()
