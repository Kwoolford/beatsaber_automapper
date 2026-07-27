"""Pattern-idiom vocabulary — axis A3 of the v2 evaluation suite.

See ``docs/eval_suite_v2.md``. This is the axis that makes Kyle's stated goal
concrete: *"an evaluation suite so good I could give an agent a set of
instructions to build a mapper by itself without machine learning."* The idiom
vocabulary IS that instruction set — a rule-based mapper can sample from it.

The premise, measured over 181 human maps (`scripts/calibrate_idiom.py`):

    130,395 per-hand transitions  ->  2,510 distinct idioms
      top   50 idioms cover 41.2% of everything human mappers do
      top  200 idioms cover 74.8%
      top  500 idioms cover 89.5%
      top 1000 idioms cover 96.4%

Human mapping is a **small vocabulary deployed deliberately**, not a
high-entropy search over the 4x3x9 space. This is the direct rebuttal to the
assumption that saturated the old scorecard, where a uniform-random map beat
human maps on `grid_coverage` and `dir_entropy` (§1 Finding 3): variety is not
the goal, *idiomatic* variety is.

An idiom is one hand's transition between consecutive notes:

    (dx, dy, dir_from, dir_to, dt_class)

with dt bucketed to {stack, 1/16, 1/8, 1/4, slow} — because the same geometric
move is a different pattern at different speeds. It is inherently sequence-aware,
so the `shuffled` control destroys it.

Metrics:
  idiom_coverage   fraction of a map's transitions drawn from the human top-K
                   vocabulary. LOW = the map is doing things human mappers do not
                   do. This is the one that correctly ranks a uniform-random map
                   below both human and our maps.
                   RULE: build patterns from the known vocabulary.
  idiom_top50      fraction drawn from the top-50 core idioms. Humans lean on the
                   core heavily; a map that never uses it is not idiomatic, and
                   one that uses ONLY it is monotonous.
                   RULE: lean on the core, but do not live there.
  idiom_jsd        Jensen-Shannon divergence between the map's idiom distribution
                   and the human one (0 = identical mix). Catches using the right
                   vocabulary in the wrong proportions.
  idiom_entropy    normalised entropy over the map's idiom usage (guard).

Calibrate with::

    python scripts/calibrate_idiom.py --n 200
"""
from __future__ import annotations

import ast
import json
import math
import pathlib
import statistics
from collections import Counter
from dataclasses import dataclass, field

from beatsaber_automapper.evaluation import _dist

KEYS = ["idiom_coverage", "idiom_top50", "idiom_jsd", "idiom_entropy", "idiom_local"]
SEQUENCE_KEYS = ["idiom_coverage", "idiom_top50", "idiom_jsd", "idiom_local"]

LOCAL_WINDOW = 16   # consecutive transitions

VOCAB_PATH = (
    pathlib.Path(__file__).resolve().parents[3] / "outputs" / "idiom_vocab_human.json"
)
REFERENCE_PATH = (
    pathlib.Path(__file__).resolve().parents[3] / "outputs" / "idiom_human_reference.json"
)

TOP_K = 500      # vocabulary size for `idiom_coverage` (89.5% human coverage)
CORE_K = 50      # "core" vocabulary for `idiom_top50`
MAX_DT = 2.0     # beats; longer than this is a rest, not a transition


def dt_class(dt: float) -> int:
    """Bucket an inter-note gap. The same move is a different idiom at speed."""
    if dt <= 0.126:
        return 0    # stack / slider (same swing)
    if dt <= 0.26:
        return 1    # ~1/16
    if dt <= 0.51:
        return 2    # ~1/8
    if dt <= 1.01:
        return 3    # ~1/4
    return 4        # slow


def idioms_of(beatmap) -> list[tuple]:
    """All per-hand transitions of a map, as idiom tuples."""
    out: list[tuple] = []
    notes = list(beatmap.color_notes)
    for color in (0, 1):
        ns = sorted((n for n in notes if n.color == color), key=lambda n: n.beat)
        for a, b in zip(ns, ns[1:]):
            dt = round(b.beat - a.beat, 3)
            if dt <= 0 or dt > MAX_DT:
                continue
            out.append((b.x - a.x, b.y - a.y, a.direction, b.direction, dt_class(dt)))
    return out


@dataclass(slots=True)
class IdiomReport:
    metrics: dict[str, float] = field(default_factory=dict)
    idiom_dist: float = float("nan")

    def as_dict(self) -> dict:
        return {"metrics": {k: round(v, 4) for k, v in self.metrics.items()},
                "idiom_dist": round(self.idiom_dist, 4)}


_VOCAB_CACHE: tuple[dict, list, dict] | None = None


def load_vocab() -> tuple[dict, list, dict]:
    """(counts, ranked idiom list, human probability distribution)."""
    global _VOCAB_CACHE
    if _VOCAB_CACHE is not None:
        return _VOCAB_CACHE
    if not VOCAB_PATH.exists():
        _VOCAB_CACHE = ({}, [], {})
        return _VOCAB_CACHE
    raw = json.loads(VOCAB_PATH.read_text())
    counts = {ast.literal_eval(k): v for k, v in raw.items()}
    ranked = [k for k, _ in sorted(counts.items(), key=lambda kv: -kv[1])]
    tot = sum(counts.values()) or 1
    probs = {k: v / tot for k, v in counts.items()}
    _VOCAB_CACHE = (counts, ranked, probs)
    return _VOCAB_CACHE


def _jsd(p: dict, q: dict) -> float:
    """Jensen-Shannon divergence between two sparse distributions, base 2."""
    keys = set(p) | set(q)
    if not keys:
        return float("nan")
    s = 0.0
    for k in keys:
        pk, qk = p.get(k, 0.0), q.get(k, 0.0)
        mk = 0.5 * (pk + qk)
        if pk > 0:
            s += 0.5 * pk * math.log2(pk / mk)
        if qk > 0:
            s += 0.5 * qk * math.log2(qk / mk)
    return s


def idiom_metrics(beatmap) -> IdiomReport:
    _counts, ranked, human_p = load_vocab()
    rep = IdiomReport()
    seq = idioms_of(beatmap)
    if len(seq) < 20 or not ranked:
        rep.metrics = {k: float("nan") for k in KEYS}
        return rep

    top = set(ranked[:TOP_K])
    core = set(ranked[:CORE_K])
    n = len(seq)
    c = Counter(seq)
    map_p = {k: v / n for k, v in c.items()}

    # LOCAL vocabulary breadth: distinct idioms inside a sliding window of
    # consecutive transitions. Added 2026-07-27 after READING a map showed the
    # right hand alternating between exactly two idioms (#51/#50) for bars at a
    # time. Whole-map counts miss this entirely — our maps use MORE distinct
    # idioms overall than human ones (238 vs 219) while recycling a handful
    # locally (0.703 vs 0.861 distinct per 16-transition window).
    #
    # This is the third instance of the same failure shape: globally right,
    # locally wrong. The same is true of hand balance (identical globally,
    # 4x off locally) and of sequencing itself (h_dist histograms pass while a
    # shuffled map scores like a human one). Global statistics are where this
    # generator looks good; local structure is where it is broken.
    w = LOCAL_WINDOW
    if len(seq) >= w * 3:
        local = statistics.fmean(
            len(set(seq[i:i + w])) / w for i in range(0, len(seq) - w, w // 2))
    else:
        local = float("nan")

    rep.metrics = {
        "idiom_coverage": sum(1 for s in seq if s in top) / n,
        "idiom_top50": sum(1 for s in seq if s in core) / n,
        "idiom_jsd": _jsd(map_p, human_p),
        "idiom_entropy": _dist_entropy(list(c.values())),
        "idiom_local": local,
    }
    rep.idiom_dist = _dist.score_map(rep.metrics, load_reference(), SEQUENCE_KEYS)
    return rep


def _dist_entropy(counts: list[int]) -> float:
    tot = sum(counts)
    if tot <= 0:
        return float("nan")
    ps = [v / tot for v in counts if v > 0]
    if len(ps) < 2:
        return 0.0
    return -sum(p * math.log(p) for p in ps) / math.log(len(ps))


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
                                   gap_name="idiom_gap")


def calibrate(records: list[dict]) -> dict[str, dict]:
    return _dist.calibrate(records, KEYS)
