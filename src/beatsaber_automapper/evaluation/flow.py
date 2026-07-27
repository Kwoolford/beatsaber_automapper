"""Flow / ergonomics metrics — axis A1 of the v2 evaluation suite.

See ``docs/eval_suite_v2.md``. The swing simulator (``swing_sim``) answers
*"is this map parity-LEGAL?"*. It does not answer *"is this map COMFORTABLE?"* —
and that difference is most of what separates a map that merely passes from a
map that is fun to play. A map can have zero parity violations and still be
exhausting nonsense (the ``shuffled`` control in ``scripts/audit_eval_suite.py``
is exactly that).

Every metric here is **sequence-aware**: it is computed from *consecutive* swings,
so it cannot be satisfied by matching marginal distributions. That is the property
the current scorecard lacks — see eval_suite_v2.md §1 Finding 2.

Each metric is stated so a mapper could be built against it (design principle 3):

  angle_change     Wrist rotation between consecutive swings of one hand, in the
                   PARITY-AWARE frame (a forehand 'down' and a backhand 'up' are
                   the same physical heading, so a clean up/down stream reads as
                   0 deg of rotation). Human mapping keeps this small and only
                   occasionally spikes.
                   RULE: successive swings of a hand should mostly continue the
                   current swing plane; large rotations are punctuation, not the
                   default.

  travel           Grid distance a hand moves between consecutive swings, per
                   second. Human hands stay in a region and move deliberately.
                   RULE: do not teleport a hand across the grid at speed.

  crossover        Fraction of notes where a hand is on the far side of the grid
                   (red/left hand in columns 2-3, blue/right in columns 0-1).
                   Human maps use crossovers deliberately and sparingly; a map
                   with random hand assignment sits near 0.5.
                   RULE: keep red left and blue right by default; cross over as a
                   deliberate, low-frequency device.

  handedness       Absolute imbalance between the two hands' note counts.
                   RULE: both hands work; neither idles.

  ebpm_burst       95th-percentile burst swing rate, in swings per MINUTE. Note
                   swing_sim reports this per BEAT, which is tempo-blind — 2
                   swings/beat is relaxed at 120 BPM and brutal at 220 — so it is
                   converted to wall-clock here. Comfort is wall-clock, the same
                   lesson swing_sim learned for reset timing.
                   RULE: burst speed stays inside what a human sustains.

Scoring (design principle 4 — distributions, not point targets): each raw metric
is converted to a robust z-score against the HUMAN corpus distribution
(median / MAD), and ``flow_dist`` is the mean absolute z. Lower = more human-like.
Scoring against the human *spread* rather than a point target is what stops the
"more extreme than human is better" failure that saturated ``h_dist``.

Calibrate the human reference with::

    python -m beatsaber_automapper.evaluation.flow calibrate --n 200
"""
from __future__ import annotations

import json
import math
import pathlib
import statistics
from dataclasses import dataclass, field

from beatsaber_automapper.evaluation import swing_sim as ss

# Raw metric keys, in report order.
KEYS = ["angle_change", "angle_harsh_frac", "travel", "crossover", "handedness",
        "ebpm_burst"]

# Only these enter the `flow_dist` composite. `crossover` and `handedness` are
# computed from note attributes independent of ORDER, so they are unchanged by
# the `shuffled` control (verified: both identical to human) — including them
# only dilutes the composite's ability to detect destroyed sequencing, which is
# the whole point of this axis. They are still reported, as guards.
SEQUENCE_KEYS = ["angle_change", "angle_harsh_frac", "travel", "ebpm_burst"]

# Human reference (median, MAD) per key, filled by `calibrate`. The defaults are
# the 2026-07-26 calibration over 200 human Standard/Expert maps so the module is
# usable without a fresh run; `load_reference()` prefers the cached JSON.
REFERENCE_PATH = (
    pathlib.Path(__file__).resolve().parents[3] / "outputs" / "flow_human_reference.json"
)
_FALLBACK_REFERENCE: dict[str, tuple[float, float]] = {}


@dataclass(slots=True)
class FlowReport:
    per_hand: dict[int, dict] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)
    flow_dist: float = float("nan")

    def as_dict(self) -> dict:
        return {
            "metrics": {k: round(v, 4) for k, v in self.metrics.items()},
            "flow_dist": round(self.flow_dist, 4),
            "per_hand": self.per_hand,
        }


def _swing_angle(sw: ss.Swing) -> float | None:
    """Physical saber heading of a swing, in degrees, accounting for parity.

    Dot-only ("flexible") swings have no committed heading — they are excluded
    rather than assigned a fake angle. That is the same lesson swing_sim learned:
    giving all-dot swings a geometric direction was its single biggest source of
    false positives.
    """
    if sw.flexible or sw.direction == 8:
        return None
    table = ss.FOREHAND_ANGLE if sw.parity is ss.Parity.FOREHAND else ss.BACKHAND_ANGLE
    return float(table.get(sw.direction, 0))


def _hand_flow(swings: list[ss.Swing], spb: float) -> dict:
    """Sequential flow stats for one hand."""
    angles: list[float] = []
    travels: list[float] = []
    for prev, cur in zip(swings, swings[1:]):
        a0, a1 = _swing_angle(prev), _swing_angle(cur)
        if a0 is not None and a1 is not None:
            angles.append(ss._angle_delta(a0, a1))
        dt = (cur.beat - prev.end_beat) * spb
        if dt > 1e-3:
            dist = math.hypot(cur.x - prev.end_x, cur.y - prev.end_y)
            travels.append(dist / dt)
    return {
        "angle_change": statistics.fmean(angles) if angles else float("nan"),
        # "harsh" = a rotation of more than a right angle between consecutive
        # swings; the wrist has to be re-seated rather than carried through.
        "angle_harsh_frac": (
            sum(a > ss.ANGLE_FLOW_DEG for a in angles) / len(angles) if angles else float("nan")
        ),
        "travel": statistics.median(travels) if travels else float("nan"),
        "n_swings": len(swings),
    }


def flow_metrics(beatmap, *, bpm: float) -> FlowReport:
    """Flow/ergonomics metrics for a parsed ``DifficultyBeatmap``."""
    card = ss.simulate(beatmap, bpm=bpm)
    spb = 60.0 / bpm if bpm > 0 else 0.5
    notes = list(beatmap.color_notes)
    rep = FlowReport()

    per_hand = {}
    for color, hand in card.per_hand.items():
        per_hand[color] = _hand_flow(hand.swings, spb)
        # swing_sim reports swings-per-BEAT; scale to swings-per-minute so the
        # metric reflects what the wrist actually has to do at this tempo.
        per_hand[color]["ebpm_burst"] = hand.swing_ebpm_p95 * bpm
    rep.per_hand = per_hand

    def _avg(key: str) -> float:
        vals = [h[key] for h in per_hand.values()
                if h.get(key) is not None and not _isnan(h[key])]
        return statistics.fmean(vals) if vals else float("nan")

    # crossover: red (0) belongs left (cols 0-1), blue (1) belongs right (cols 2-3)
    if notes:
        wrong = sum(1 for n in notes
                    if (n.color == 0 and n.x >= 2) or (n.color == 1 and n.x <= 1))
        crossover = wrong / len(notes)
        n_red = sum(1 for n in notes if n.color == 0)
        handedness = abs(n_red - (len(notes) - n_red)) / len(notes)
    else:
        crossover = handedness = float("nan")

    rep.metrics = {
        "angle_change": _avg("angle_change"),
        "angle_harsh_frac": _avg("angle_harsh_frac"),
        "travel": _avg("travel"),
        "crossover": crossover,
        "handedness": handedness,
        "ebpm_burst": _avg("ebpm_burst"),
    }
    rep.flow_dist = score_against_reference(rep.metrics)
    return rep


def _isnan(v) -> bool:
    return isinstance(v, float) and math.isnan(v)


def load_reference() -> dict[str, tuple[float, float]]:
    if REFERENCE_PATH.exists():
        try:
            raw = json.loads(REFERENCE_PATH.read_text())
            return {k: (float(v["median"]), float(v["mad"])) for k, v in raw.items()}
        except Exception:  # noqa: BLE001
            pass
    return _FALLBACK_REFERENCE


def score_against_reference(metrics: dict[str, float],
                            reference: dict | None = None) -> float:
    """Mean |robust z| of the metrics vs the human corpus. Lower = more human.

    Uses median/MAD rather than mean/sd because a few pathological human maps
    would otherwise widen the reference enough to admit anything.
    """
    ref = reference if reference is not None else load_reference()
    if not ref:
        return float("nan")
    zs = []
    for k in SEQUENCE_KEYS:
        if k not in ref:
            continue
        med, mad = ref[k]
        v = metrics.get(k)
        if v is None or _isnan(v) or mad <= 0:
            continue
        zs.append(abs(v - med) / mad)
    return statistics.fmean(zs) if zs else float("nan")


def cohort_comparison(records: list[dict[str, float]],
                      reference: dict | None = None) -> dict:
    """Compare a COHORT of maps against the human distribution.

    Why this exists, and why it is the statistic to rank generators by:
    ``flow_dist`` is a per-map distance from the human *median*, and any such
    distance rewards a generator that always emits the average — a mode-collapsed
    cohort sits closer to the median than typical human maps do, and so scores
    "better than human". That is exactly how ``h_dist`` saturated
    (docs/eval_suite_v2.md §1 Finding 1), and it reappears here if you are not
    careful: our production maps score flow_dist 1.37 vs human 1.54 while being
    measurably *less* human on the underlying quantities.

    So compare distributions, not points. Per metric we report:
      shift   (cohort median - human median) / human MAD  — signed, in human units
      spread  cohort MAD / human MAD — < 1 means under-dispersed (mode collapse),
              which a distance-to-median score cannot see at all.

    A cohort is human-like when every |shift| is small AND spread is near 1.
    """
    ref = reference if reference is not None else load_reference()
    out: dict[str, dict] = {}
    for k in KEYS:
        if k not in ref:
            continue
        hmed, hmad = ref[k]
        vals = [r[k] for r in records if r.get(k) is not None and not _isnan(r[k])]
        if len(vals) < 3 or hmad <= 0:
            continue
        med = statistics.median(vals)
        mad = statistics.median([abs(v - med) for v in vals])
        out[k] = {
            "median": med,
            "shift": (med - hmed) / hmad,
            "spread": mad / hmad,
            "n": len(vals),
        }
    seq = [abs(out[k]["shift"]) for k in SEQUENCE_KEYS if k in out]
    out["_summary"] = {
        "flow_gap": statistics.fmean(seq) if seq else float("nan"),
        "min_spread": min((out[k]["spread"] for k in SEQUENCE_KEYS if k in out),
                          default=float("nan")),
    }
    return out


def calibrate(records: list[dict[str, float]]) -> dict[str, dict]:
    """Build the human reference (median + MAD per key) from raw metric dicts."""
    out: dict[str, dict] = {}
    for k in KEYS:
        vals = [r[k] for r in records if r.get(k) is not None and not _isnan(r[k])]
        if len(vals) < 5:
            continue
        med = statistics.median(vals)
        mad = statistics.median([abs(v - med) for v in vals])
        if mad <= 0:  # degenerate; fall back to sd so the key stays usable
            mad = statistics.pstdev(vals) or 1e-6
        out[k] = {"median": med, "mad": mad, "n": len(vals)}
    return out
