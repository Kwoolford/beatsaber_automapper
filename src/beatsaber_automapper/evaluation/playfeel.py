"""Difficulty and direction idiom — axis A7 of the v2 evaluation suite.

Built 2026-07-27 after Kyle played the maps and found them "busy, unmusical, and
unplayable as Expert". Three of his complaints turned out to be measurable, and
**none of them was gated by any existing axis** — the suite was scoring rhythm,
flow, idiom and hand-role while the map was a difficulty tier too dense and made
of diagonals.

    complaint                       measurement        human    ours
    "this is Expert, not Expert+"   nps                 4.46     6.18
    "obsessed with 45-degree notes" diagonal share      0.370    0.513
                                    up/down share       0.562    0.468
    (unnoted, also true)            dot-note share      0.042    0.001

**The direction finding is a self-inflicted wound.** The anti-repeat lever
promoted 2026-07-23, and every push on `dir_entropy` and `grid_coverage` before
it, rewarded spreading probability across all nine cut directions. Nine
directions means mostly diagonals, so "more diverse" drove us away from the human
idiom, which leads with up/down and uses diagonals as *deviation*. Kyle's
original "for-sport diagonals" complaint was never fixed; it was made worse and
recorded as progress. This axis exists so that cannot happen silently again.

Metrics:
  nps              notes per second. Scored against the human **Expert**
                   distribution specifically — difficulty is the whole point, so
                   pooling Expert with ExpertPlus would defeat it.
                   RULE: an Expert map runs at roughly 4.5 notes/second.
  peak_nps         95th-percentile per-2s-window density. A map can have a human
                   mean and still spike into unplayable bursts.
                   RULE: peaks stay inside what an Expert player sustains.
  vertical_share   fraction of cuts that are up or down.
  diagonal_share   fraction that are one of the four diagonals.
                   RULE: lead with the vertical axis; diagonals are punctuation,
                   not the default.
  dot_share        fraction of any-direction (dot) notes; humans use them ~4% of
                   the time and we essentially never do (guard).

All five are per-note or per-window *proportions* rather than sequence
statistics, so unlike A1/A2/A3/A6 this axis is deliberately NOT sequence-aware —
it is a difficulty-and-vocabulary gate, and the `random` control passes parts of
it by construction. That is expected: it is the axis that catches "technically
varied but a tier too hard", which the sequence axes cannot see.
"""
from __future__ import annotations

import json
import pathlib
import statistics
from collections import Counter
from dataclasses import dataclass, field

from beatsaber_automapper.evaluation import _dist

KEYS = ["nps", "peak_nps", "vertical_share", "diagonal_share", "dot_share"]
# All five gate real, separately-observed defects, so all five drive the gap.
SEQUENCE_KEYS = ["nps", "peak_nps", "vertical_share", "diagonal_share"]

REFERENCE_PATH = (
    pathlib.Path(__file__).resolve().parents[3] / "outputs" / "playfeel_human_reference.json"
)

VERTICAL = (0, 1)            # up, down
DIAGONAL = (4, 5, 6, 7)
DOT = 8
WIN_SEC = 2.0


@dataclass(slots=True)
class PlayFeelReport:
    metrics: dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {"metrics": {k: round(v, 4) for k, v in self.metrics.items()}}


def playfeel_metrics(beatmap, *, bpm: float) -> PlayFeelReport:
    notes = sorted(beatmap.color_notes, key=lambda n: n.beat)
    rep = PlayFeelReport()
    if len(notes) < 40 or bpm <= 0:
        rep.metrics = {k: float("nan") for k in KEYS}
        return rep

    spb = 60.0 / bpm
    times = [n.beat * spb for n in notes]
    dur = max(times[-1], 1e-6)

    # per-window density, for the peak
    nwin = max(int(dur / WIN_SEC) + 1, 1)
    counts = [0] * nwin
    for t in times:
        counts[min(int(t / WIN_SEC), nwin - 1)] += 1
    dens = sorted(c / WIN_SEC for c in counts)
    peak = dens[min(int(0.95 * (len(dens) - 1)), len(dens) - 1)]

    c = Counter(n.direction for n in notes)
    n = len(notes)
    rep.metrics = {
        "nps": n / dur,
        "peak_nps": peak,
        "vertical_share": sum(c[d] for d in VERTICAL) / n,
        "diagonal_share": sum(c[d] for d in DIAGONAL) / n,
        "dot_share": c[DOT] / n,
    }
    return rep


def load_reference() -> dict[str, tuple[float, float]]:
    if REFERENCE_PATH.exists():
        try:
            raw = json.loads(REFERENCE_PATH.read_text())
            return {k: (float(v["median"]), float(v["mad"])) for k, v in raw.items()}
        except Exception:  # noqa: BLE001
            pass
    return {}


def cohort_comparison(records: list[dict], reference: dict | None = None) -> dict:
    ref = reference if reference is not None else load_reference()
    return _dist.cohort_comparison(records, ref, KEYS, SEQUENCE_KEYS,
                                   gap_name="playfeel_gap")


def calibrate(records: list[dict]) -> dict[str, dict]:
    return _dist.calibrate(records, KEYS)
