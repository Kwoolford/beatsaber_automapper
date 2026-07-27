#!/usr/bin/env python
"""A Beat Saber mapper with NO machine learning, built only from the eval suite.

This is the test of the project's stated goal: *"an evaluation suite so good I
could give an agent a set of instructions to build a mapper by itself without
machine learning."* If the suite is genuinely prescriptive, that mapper is
buildable from its rules alone — and if it is not buildable, the suite is still
descriptive and we know what is missing.

Every decision below comes from a measured rule in docs/eval_suite_v2.md. No
model, no checkpoint, no learned weights; the only inputs are onset times, a BPM,
and the mined human idiom vocabulary.

  A2 rhythm   Play both hands together on ~17.5% of beats, not 86% — hand
              LOCKSTEP is what makes a map metronomic. Otherwise alternate hands.
  A3 idiom    Choose each note by sampling the human idiom vocabulary conditioned
              on this hand's previous note and the time gap. 2,510 idioms cover
              human mapping; the top 500 cover ~90%.
  A1 flow     Prefer idioms whose implied hand travel is near the human median
              (~4 grid-units/sec) and whose wrist rotation is small.
  parity      Alternate forehand/backhand per hand; a same-parity repeat needs
              >= 0.30 s (swing_sim's calibrated wrist-break floor).
  crossover   Keep red left / blue right, except ~20% deliberate crossovers.

Usage:
  python scripts/rule_mapper.py --onsets-from <map.zip> --out rule.zip
  python scripts/rule_mapper.py --audio song.ogg --bpm 128 --out rule.zip
"""
from __future__ import annotations

import argparse
import pathlib
import random
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from beatsaber_automapper.data.beatmap import ColorNote  # noqa: E402
from beatsaber_automapper.evaluation import idiom as idm  # noqa: E402
from beatsaber_automapper.evaluation import swing_sim as ss  # noqa: E402

# --- constants lifted straight from the measured human reference ---
SIMULTANEITY = 0.175      # A2: share of beats with both hands
CROSSOVER_RATE = 0.20     # A1 guard: human median 0.218
TRAVEL_TARGET = 4.0       # A1: grid-units per second
HARD_RESET_SEC = ss.HARD_RESET_SEC
DOWN_DIRS = (1, 6, 7)
UP_DIRS = (0, 4, 5)


def _parity_of(direction: int) -> int | None:
    if direction in DOWN_DIRS:
        return 0
    if direction in UP_DIRS:
        return 1
    return None


class HandState:
    def __init__(self, color: int, home_cols: tuple[int, ...]):
        self.color = color
        self.home = home_cols
        self.x = home_cols[0]
        self.y = 1
        self.direction = 1
        self.parity = 0
        self.beat = -99.0


def _candidates(vocab_ranked, hand: HandState, dt_beats: float, spb: float,
                rng: random.Random, top_k: int):
    """Idioms that are legal AND comfortable from this hand's current state."""
    cls = idm.dt_class(dt_beats)
    dt_sec = dt_beats * spb
    out = []
    for (dx, dy, d_from, d_to, c) in vocab_ranked[:top_k]:
        if c != cls or d_from != hand.direction:
            continue
        nx, ny = hand.x + dx, hand.y + dy
        if not (0 <= nx <= 3 and 0 <= ny <= 2):
            continue
        # crossover rule: mostly stay on our own side
        own_side = nx in hand.home
        if not own_side and rng.random() > CROSSOVER_RATE:
            continue
        # parity rule: alternate, unless there is time for a real reset
        p = _parity_of(d_to)
        if p is not None and p == hand.parity and dt_sec < HARD_RESET_SEC:
            continue
        # flow rule: prefer travel near the human median
        dist = (dx * dx + dy * dy) ** 0.5
        speed = dist / dt_sec if dt_sec > 0 else 999.0
        out.append(((dx, dy, d_from, d_to, c), abs(speed - TRAVEL_TARGET)))
    out.sort(key=lambda t: t[1])
    return [o[0] for o in out]


def build_map(onset_beats: list[float], bpm: float, seed: int = 0,
              top_k: int = 500, width: int = 6) -> list[ColorNote]:
    rng = random.Random(seed)
    _, ranked, _ = idm.load_vocab()
    if not ranked:
        raise SystemExit("no idiom vocabulary — run scripts/calibrate_idiom.py first")
    spb = 60.0 / bpm if bpm > 0 else 0.5

    hands = {0: HandState(0, (0, 1)), 1: HandState(1, (2, 3))}
    notes: list[ColorNote] = []
    turn = 0
    for b in sorted(onset_beats):
        # A2: alternate hands, with occasional real doubles at the human rate
        both = rng.random() < SIMULTANEITY
        picks = (0, 1) if both else (turn % 2,)
        turn += 1
        for color in picks:
            h = hands[color]
            dt = b - h.beat
            if dt <= 0:
                continue
            cands = _candidates(ranked, h, min(dt, 2.0), spb, rng, top_k)
            if cands:
                # sample from the best few so the map is not deterministic
                dx, dy, _df, d_to, _c = cands[rng.randrange(min(width, len(cands)))]
                nx, ny = h.x + dx, h.y + dy
            else:
                # nothing idiomatic fits: fall back to a plain parity alternation
                d_to = 0 if h.parity == 0 else 1
                nx, ny = h.x, h.y
            notes.append(ColorNote(beat=b, x=nx, y=ny, color=color, direction=d_to))
            p = _parity_of(d_to)
            h.x, h.y, h.direction, h.beat = nx, ny, d_to, b
            if p is not None:
                h.parity = p
            else:
                h.parity ^= 1
    return notes


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--onsets-from", help="take note TIMES from this map zip "
                                          "(isolates pattern quality from onset detection)")
    ap.add_argument("--bpm", type=float)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--top-k", type=int, default=500)
    ap.add_argument("--width", type=int, default=6,
                    help="sample from this many best candidates; higher = more varied "
                         "maps. The suite prescribes what a human map looks like on "
                         "average but not how much maps should DIFFER, so this knob "
                         "is the one thing not derivable from the rules.")
    ap.add_argument("--score", action="store_true", help="score the result and exit")
    a = ap.parse_args()

    if not a.onsets_from:
        ap.error("--onsets-from is required in this PoC")

    from audit_eval_suite import _load_generated, _load_human
    p = pathlib.Path(a.onsets_from)
    loaded = _load_generated(p) or _load_human(p)
    if loaded is None:
        ap.error(f"could not read {p}")
    src_notes, bpm = loaded
    bpm = a.bpm or bpm
    beats = sorted({round(n.beat, 4) for n in src_notes})

    notes = build_map(beats, bpm, seed=a.seed, top_k=a.top_k, width=a.width)
    print(f"rule-based map: {len(notes)} notes from {len(beats)} onsets @ {bpm:.1f} bpm")

    if a.score:
        from beatsaber_automapper.evaluation import scorecard as sc

        class _BM:
            color_notes = sorted(notes, key=lambda n: n.beat)
            bomb_notes: list = []

        print(sc.report(sc.score_cohort([(_BM(), bpm)], "rule-based (1 map)")))


if __name__ == "__main__":
    main()
