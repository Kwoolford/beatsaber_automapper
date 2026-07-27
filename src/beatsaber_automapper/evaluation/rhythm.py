"""Rhythm / beat-grid metrics — axis A2 of the v2 evaluation suite.

See ``docs/eval_suite_v2.md``. Nothing in the suite measured RHYTHM, and the
evidence says that was the largest remaining blind spot. Measured over the
inter-onset intervals (IOIs) of 40 human maps vs 12 of our production maps:

    IOI (beats)   0.5      0.25     1.0
    human         0.41     0.20     0.16      pulse_stability 0.529
    ours          0.75     0.06     0.13      pulse_stability 0.765

**Our maps put 75% of their notes at a constant 1/8-note spacing** where humans
spread across 1/16, 1/8, 1/4 and dotted values, and our consecutive intervals
repeat far more often than a human's. That metronomic pulse is very likely a real
component of the original "for-sport" complaint, and no existing metric could see
it — every scorecard metric is computed over note *attributes*, none over note
*times*.

Note what is NOT worth measuring: on-grid purity. V7 emits on a 1/16 BPM grid by
construction and human maps are 94-99% on that same grid (the V8-0 finding), so
"is it on the grid" is trivially satisfied by both and cannot discriminate.
`offgrid_frac` is kept only as a guard.

Metrics (the first three are sequence-aware — computed over *consecutive*
intervals, so they cannot be satisfied by matching the IOI histogram alone):

  pulse_stability   fraction of consecutive IOI pairs that are equal. A stream of
                    even notes scores 1.0. Humans sit near 0.53 — they hold a
                    pulse about half the time and break it the rest.
                    RULE: hold a pulse, but break it; do not run one spacing for
                    the whole song.
  ioi_cond_entropy  normalised conditional entropy H(next IOI | current IOI).
                    Low = the rhythm is predictable from its own past (a machine
                    stream); high = incoherent. Humans sit in between.
                    RULE: the next interval should be mostly, not entirely,
                    predictable from the current one.
  ioi_switch_rate   rate of changes in the local dominant IOI, per 100 notes —
                    how often the map changes rhythmic gear.
                    RULE: change subdivision at musical boundaries, not never and
                    not constantly.
  dominant_share    share of the single most common IOI (guard; ours 0.75 vs
                    human 0.41).
  ioi_entropy       normalised entropy of the IOI distribution (guard).
  offgrid_frac      fraction of notes off the 1/16 grid (guard).

Calibrate with::

    python scripts/calibrate_rhythm.py --n 200
"""
from __future__ import annotations

import json
import math
import pathlib
import statistics
from collections import Counter
from dataclasses import dataclass, field

from beatsaber_automapper.evaluation import _dist

KEYS = ["pulse_stability", "ioi_cond_entropy", "ioi_switch_rate",
        "dominant_share", "ioi_entropy", "offgrid_frac"]

# Only these enter the `rhythm_gap` composite. The rest are marginal statistics
# over the IOI histogram (order-invariant), kept as guards — the same lesson axis
# A1 learned about crossover/handedness.
SEQUENCE_KEYS = ["pulse_stability", "ioi_cond_entropy", "ioi_switch_rate"]

REFERENCE_PATH = (
    pathlib.Path(__file__).resolve().parents[3] / "outputs" / "rhythm_human_reference.json"
)

GRID = 16.0        # V7's 1/16-beat quantisation
GRID_TOL = 0.01    # beats
IOI_ROUND = 3      # beats, for binning intervals into symbols
MAX_IOI = 4.0      # intervals longer than this are rests, not rhythm


@dataclass(slots=True)
class RhythmReport:
    metrics: dict[str, float] = field(default_factory=dict)
    rhythm_dist: float = float("nan")

    def as_dict(self) -> dict:
        return {"metrics": {k: round(v, 4) for k, v in self.metrics.items()},
                "rhythm_dist": round(self.rhythm_dist, 4)}


def _norm_entropy(counts) -> float:
    tot = sum(counts)
    if tot <= 0:
        return float("nan")
    ps = [c / tot for c in counts if c > 0]
    if len(ps) < 2:
        return 0.0
    h = -sum(p * math.log(p) for p in ps)
    return h / math.log(len(ps))


def rhythm_metrics(beatmap, *, bpm: float | None = None) -> RhythmReport:
    """Rhythm metrics for a parsed ``DifficultyBeatmap``. BPM-free (beat domain)."""
    beats = sorted({round(n.beat, 4) for n in beatmap.color_notes})
    rep = RhythmReport()
    if len(beats) < 20:
        rep.metrics = {k: float("nan") for k in KEYS}
        return rep

    offgrid = sum(1 for b in beats
                  if abs(b * GRID - round(b * GRID)) > GRID_TOL * GRID) / len(beats)

    d = [round(b - a, IOI_ROUND) for a, b in zip(beats, beats[1:])]
    d = [x for x in d if 0 < x <= MAX_IOI]
    if len(d) < 10:
        rep.metrics = {k: float("nan") for k in KEYS}
        rep.metrics["offgrid_frac"] = offgrid
        return rep

    # --- sequence-aware ---
    pulse = statistics.fmean([1.0 if abs(a - b) < 1e-9 else 0.0
                              for a, b in zip(d, d[1:])])

    # conditional entropy H(next | current), averaged over current-symbol mass
    trans: dict[float, Counter] = {}
    for a, b in zip(d, d[1:]):
        trans.setdefault(a, Counter())[b] += 1
    tot = sum(sum(c.values()) for c in trans.values())
    cond = sum((sum(c.values()) / tot) * _norm_entropy(list(c.values()))
               for c in trans.values() if sum(c.values()) > 0) if tot else float("nan")

    # local dominant-IOI switches, per 100 notes
    win, switches, prev_dom = 8, 0, None
    for i in range(0, len(d) - win + 1):
        dom = Counter(d[i:i + win]).most_common(1)[0][0]
        if prev_dom is not None and dom != prev_dom:
            switches += 1
        prev_dom = dom
    switch_rate = 100.0 * switches / max(len(d), 1)

    # --- guards (marginal over the IOI histogram) ---
    counts = Counter(d)
    dominant = counts.most_common(1)[0][1] / len(d)

    rep.metrics = {
        "pulse_stability": pulse,
        "ioi_cond_entropy": cond,
        "ioi_switch_rate": switch_rate,
        "dominant_share": dominant,
        "ioi_entropy": _norm_entropy(list(counts.values())),
        "offgrid_frac": offgrid,
    }
    rep.rhythm_dist = _dist.score_map(rep.metrics, load_reference(), SEQUENCE_KEYS)
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
                                   gap_name="rhythm_gap")


def calibrate(records: list[dict]) -> dict[str, dict]:
    return _dist.calibrate(records, KEYS)
