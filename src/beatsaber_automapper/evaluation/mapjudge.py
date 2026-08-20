"""Judge ONE map, at n=1, without a cohort and without Kyle playing it.

**Why this exists.** `scorecard.py` is a *cohort* statistic by design: every axis is a
median shift + spread against the human distribution, which is the right construction
for ranking generators and the wrong one for asking "is this particular map any good?".
Measured 2026-08-19: all six axes return `nan` below n=5, and the same maps score
`flow` 1.260 at n=5 and 0.446 at n=50 — the bars do not even transfer across cohort
size. So the suite could never answer the question an agent authoring one map needs
answered.

**Why this is not the `h_dist` failure again.** `h_dist` was a *ranking* scalar: mean
relative distance to the human median. Being AT the median maximised it, so a
mode-collapsed generator scored "more human than human" (docs/eval_suite_v2.md §1
Finding 1). This module never ranks by closeness. It asks a different question —
*is this map inside the human distribution?* — and answers it with a **conformal
p-value**: the fraction of held-out human maps that are at least as extreme as this
one. Sitting at the median is a PASS here, and that is correct: a map at the human
median is not defective. Mode collapse is a property of a COHORT, not of a map, and
it stays `scorecard.py`'s job. The two modules answer two different questions and
conflating them is what made the suite unable to answer either.

**The construction.**

1. Every metric is scored as its **empirical percentile** in a human corpus slice,
   so no metric needs a hand-chosen bar and no metric can be Goodharted by matching a
   point target.
2. Percentile becomes a two-sided nonconformity `u = 2|p - 0.5|`: 0 at the human
   median, 1 in either tail.
3. A map's score aggregates `u` over metrics; its **p-value** is the fraction of
   *calibration* human maps scoring at least as extreme. `PASS` means "this map is
   not more unusual than a human map".
4. **Three aggregates, because GATING and RANKING are different jobs** -- the single
   most expensive lesson in this repo, and the reason `h_dist` had to be retired.
   `mean` catches a map that is subtly off everywhere; `max` catches one metric
   broken outright; `topk` (mean of the K most extreme) sits between them. The
   **gate** fires if any of the three does, re-conformalised so the multiplicity is
   paid for. The **ranking** number is `s_mean` alone, exposed as `rank_score`.
   ★Both are needed and neither substitutes for the other, measured directly:
   with `max` in the gate, `timing_jitter` is rejected 1.000 but every production
   map ties at p=0.127 (all of them have `crossover` fully outside the human range,
   so the max pins at 1.0); with `topk` instead, the ordering comes back but
   `timing_jitter` acceptance jumps to 0.204, because a defect living in ONE metric
   is diluted by averaging. Gate on all three; steer on the mean.
5. Parity violations are a **hard gate**, not a percentile — an illegal map is
   unplayable regardless of how human its statistics look.

★**The held-out human accept rate is NOT evidence that this works.** It is ~1-alpha by
construction — that is what conformal calibration guarantees. The only evidence is the
**control rejection rate**: `scripts/audit_mapjudge.py` scores the six degenerate
controls at n=1 and a metric that cannot separate them is dead weight and gets dropped.
"""
from __future__ import annotations

import json
import math
import pathlib
from dataclasses import dataclass, field

QUANTILE_PATH = (
    pathlib.Path(__file__).resolve().parents[3]
    / "docs" / "eval_references" / "mapjudge_human_quantiles.json"
)

# Metrics the judge scores, grouped by the axis they came from. Each entry is
# (metric, axis, note) where `note` is the plain-English reading used in the
# prescriptive report -- the judge has to say WHAT IS WRONG, not just that
# something is (docs/eval_suite_v2.md design principle 3).
#
# `tail` says which side is a defect:
#   "both" - either extreme is wrong (travel: static hands OR flailing)
#   "high" - only the upper tail is a defect
#   "low"  - only the lower tail is a defect
# A one-sided metric contributes u = 0 on its safe side. Being one-sided is a
# CLAIM about the metric and every one of them is checked by the control battery.
CANDIDATES: list[tuple[str, str, str, str]] = [
    # --- A1 flow / ergonomics -------------------------------------------------
    ("angle_change",     "flow",     "both", "wrist rotation between swings"),
    ("angle_harsh_frac", "flow",     "high", "share of >90 degree transitions"),
    ("travel",           "flow",     "both", "how far a hand moves per second"),
    ("ebpm_burst",       "flow",     "high", "peak per-hand swing rate"),
    ("crossover",        "flow",     "both", "share of notes played across the body"),
    ("handedness",       "flow",     "both", "left/right note imbalance"),
    # --- A2 rhythm ------------------------------------------------------------
    ("pulse_stability",  "rhythm",   "both", "how steadily the map holds a pulse"),
    ("ioi_cond_entropy", "rhythm",   "both", "how predictable the next gap is"),
    ("ioi_switch_rate",  "rhythm",   "both", "how often the map changes rhythmic gear"),
    ("dominant_share",   "rhythm",   "high", "share of gaps at the single commonest value"),
    ("offgrid_frac",     "rhythm",   "high", "share of notes off the 1/16 grid"),
    # --- A3 idiom vocabulary --------------------------------------------------
    ("idiom_coverage",   "idiom",    "low",  "share of moves drawn from the human top-500"),
    ("idiom_top50",      "idiom",    "both", "share drawn from the 50-idiom core"),
    ("idiom_jsd",        "idiom",    "high", "how different the idiom MIX is from human"),
    ("idiom_local",      "idiom",    "low",  "idiom variety inside a 16-note window"),
    # --- A6 hand role ---------------------------------------------------------
    ("role_asymmetry",   "handrole", "both", "how lopsided the hands are within a passage"),
    ("role_swap_rate",   "handrole", "both", "how often the lead hand changes"),
    # --- A7 playfeel ----------------------------------------------------------
    ("nps",              "playfeel", "both", "notes per second"),
    ("peak_nps",         "playfeel", "high", "densest stretch"),
    ("vertical_share",   "playfeel", "both", "share of up/down swings"),
    ("diagonal_share",   "playfeel", "both", "share of diagonal swings"),
    # --- A8 alignment (needs audio; skipped when onsets are unavailable) -------
    ("onset_precision",  "alignment", "low",  "share of notes on something audible"),
    ("offset_mad_ms",    "alignment", "high", "timing scatter against the music"),
]

# How many of the most extreme metrics the second aggregate averages. 3 was chosen
# because `u` saturates at 1.0 and a plain max therefore ties every map that has any
# metric fully outside the human range -- which is all of ours (crossover = 0.000).
TOPK = 3

# Metrics dropped by the control battery live here so the reason survives.
# Populated by scripts/audit_mapjudge.py --write-drops.
DROPPED: dict[str, str] = {}


def active_metrics(exclude_axes: set[str] | None = None) -> list[tuple[str, str, str, str]]:
    """The candidate metrics minus anything the control battery retired."""
    ex = exclude_axes or set()
    return [c for c in CANDIDATES if c[0] not in DROPPED and c[1] not in ex]


# --------------------------------------------------------------------------
# per-map raw metrics
# --------------------------------------------------------------------------
class _BM:
    """Minimal DifficultyBeatmap shim (same one audit_eval_suite.py uses)."""

    def __init__(self, notes):
        self.color_notes = sorted(notes, key=lambda n: n.beat)
        self.bomb_notes = []


def map_record(notes, bpm: float, onsets=None) -> dict[str, float]:
    """Every candidate metric for one map, plus the parity hard gate.

    Missing metrics are simply absent from the dict -- never filled with a
    placeholder. A silently-defaulted metric is how `alignment` went missing from
    every scorecard for two nights (agent_mapper/README.md rule 3).
    """
    from beatsaber_automapper.evaluation import (
        alignment, flow, handrole, idiom, playfeel, rhythm, swing_sim,
    )

    bm = _BM(notes)
    rec: dict[str, float] = {}

    for fn, kwargs in (
        (flow.flow_metrics, {"bpm": bpm}),
        (rhythm.rhythm_metrics, {}),
        (idiom.idiom_metrics, {}),
        (handrole.handrole_metrics, {}),
        (playfeel.playfeel_metrics, {"bpm": bpm}),
    ):
        try:
            rec.update(fn(bm, **kwargs).metrics)
        except Exception:  # noqa: BLE001
            pass

    if onsets is not None and len(onsets):
        try:
            rec.update(alignment.alignment_metrics(bm, bpm=bpm, onsets=onsets).metrics)
        except Exception:  # noqa: BLE001
            pass

    try:
        rec["viol"] = float(swing_sim.simulate(bm, bpm=bpm).violations)
    except Exception:  # noqa: BLE001
        pass

    rec["n_notes"] = float(len(bm.color_notes))
    return rec


# --------------------------------------------------------------------------
# percentiles + conformal scoring
# --------------------------------------------------------------------------
def percentile_of(value: float, sorted_vals: list[float]) -> float:
    """Empirical percentile of `value` within `sorted_vals`, in [0, 1].

    Midpoint convention (ties split) so an exactly-median value scores 0.5 rather
    than drifting with the number of duplicates -- `angle_harsh_frac` is exactly
    0.0 on a large minority of human maps and a naive `<` count puts all of them
    in the lower tail.
    """
    n = len(sorted_vals)
    if n == 0 or value is None or (isinstance(value, float) and math.isnan(value)):
        return float("nan")
    lo = _bisect_left(sorted_vals, value)
    hi = _bisect_right(sorted_vals, value)
    return (lo + hi) / 2.0 / n


def _bisect_left(a: list[float], x: float) -> int:
    import bisect
    return bisect.bisect_left(a, x)


def _bisect_right(a: list[float], x: float) -> int:
    import bisect
    return bisect.bisect_right(a, x)


def nonconformity(value: float, sorted_vals: list[float], tail: str) -> float:
    """Two-sided (or one-sided) extremeness in [0, 1]. 0 = human median."""
    p = percentile_of(value, sorted_vals)
    if math.isnan(p):
        return float("nan")
    if tail == "high":
        return max(0.0, 2.0 * (p - 0.5))
    if tail == "low":
        return max(0.0, 2.0 * (0.5 - p))
    return abs(2.0 * (p - 0.5))


@dataclass(slots=True)
class MetricScore:
    name: str
    axis: str
    value: float
    pct: float           # human percentile, 0-1
    u: float             # nonconformity, 0-1
    note: str

    @property
    def flag(self) -> str:
        if math.isnan(self.u):
            return "?"
        if self.u >= 0.98:
            return "!!"
        if self.u >= 0.90:
            return "!"
        return ""


@dataclass(slots=True)
class JudgeResult:
    label: str
    metrics: list[MetricScore] = field(default_factory=list)
    s_mean: float = float("nan")
    s_topk: float = float("nan")
    s_max: float = float("nan")
    p_mean: float = float("nan")
    p_topk: float = float("nan")
    p_max: float = float("nan")
    viol: float | None = None
    n_notes: float = 0.0
    n_scored: int = 0
    missing: list[str] = field(default_factory=list)

    p_combined: float = float("nan")

    @property
    def p_min(self) -> float:
        """The more extreme of the two aggregate p-values. NOT a calibrated p-value."""
        vals = [v for v in (self.p_mean, self.p_topk, self.p_max)
                if not math.isnan(v)]
        return min(vals) if vals else float("nan")

    @property
    def p_value(self) -> float:
        """The map's calibrated overall p-value.

        ★**Why this is not `min(p_mean, p_max)`.** Firing on whichever of two
        p-values is smaller is an uncorrected multiple test: each is calibrated at
        alpha on its own, but their minimum rejects at somewhere between alpha and
        2*alpha. Measured directly -- with alpha 0.10 the min-rule accepted
        **0.844** of held-out human maps where conformal calibration guarantees
        0.90. So the minimum is re-conformalised against the same statistic
        computed on the calibration humans, which restores the guarantee while
        keeping both failure modes (subtly-off-everywhere, and one-axis-broken).
        """
        return self.p_combined

    def verdict(self, alpha: float = 0.10) -> str:
        if self.viol is not None and self.viol > 0:
            return "FAIL"
        if math.isnan(self.p_value):
            return "UNSCORED"
        return "PASS" if self.p_value >= alpha else "FAIL"

    @property
    def rank_score(self) -> float:
        """The number to ORDER maps by (lower = more human).

        ★Deliberately **not** the gate. The gate includes a max term that saturates
        on any fully-out-of-range metric, which ties every map we generate; the mean
        keeps ordering them. Ranking by the gate is how `h_dist` died.
        """
        return self.s_mean

    def worst(self, k: int = 5) -> list[MetricScore]:
        ok = [m for m in self.metrics if not math.isnan(m.u)]
        return sorted(ok, key=lambda m: -m.u)[:k]


def load_reference(path: pathlib.Path | None = None) -> dict:
    p = path or QUANTILE_PATH
    if not p.exists():
        raise FileNotFoundError(
            f"no mapjudge reference at {p} -- run scripts/calibrate_mapjudge.py first"
        )
    return json.loads(p.read_text())


def judge(record: dict[str, float], reference: dict, *, label: str = "map",
          exclude_axes: set[str] | None = None) -> JudgeResult:
    """Score one map's raw metrics against the human reference."""
    dists: dict[str, list[float]] = reference["distributions"]
    calib_mean: list[float] = sorted(reference["calib_scores"]["mean"])
    calib_topk: list[float] = sorted(reference["calib_scores"]["topk"])
    calib_max: list[float] = sorted(reference["calib_scores"]["max"])
    calib_pmin: list[float] = sorted(reference["calib_scores"].get("pmin", []))

    res = JudgeResult(label=label)
    res.viol = record.get("viol")
    res.n_notes = record.get("n_notes", 0.0)

    us: list[float] = []
    for name, axis, tail, note in active_metrics(exclude_axes):
        if name not in dists:
            continue
        if name not in record:
            res.missing.append(name)
            continue
        val = float(record[name])
        u = nonconformity(val, dists[name], tail)
        if math.isnan(u):
            res.missing.append(name)
            continue
        res.metrics.append(MetricScore(name, axis, val,
                                       percentile_of(val, dists[name]), u, note))
        us.append(u)

    res.n_scored = len(us)
    if not us:
        return res

    res.s_mean = sum(us) / len(us)
    top = sorted(us, reverse=True)[:TOPK]
    res.s_topk = sum(top) / len(top)
    res.s_max = max(us)
    res.p_mean = _conformal_p(res.s_mean, calib_mean)
    res.p_topk = _conformal_p(res.s_topk, calib_topk)
    res.p_max = _conformal_p(res.s_max, calib_max)
    # Re-conformalise the minimum so the overall bar keeps its guarantee. When the
    # calibration slice is absent (the provisional pass inside the calibrator) fall
    # back to the raw minimum, which is what that pass is measuring.
    if calib_pmin:
        n = len(calib_pmin)
        le = _bisect_right(calib_pmin, res.p_min)
        res.p_combined = (1.0 + le) / (n + 1.0)
    else:
        res.p_combined = res.p_min
    return res


def _conformal_p(score: float, calib_sorted: list[float]) -> float:
    """(1 + #{calib >= score}) / (n + 1) -- the standard conformal p-value."""
    n = len(calib_sorted)
    if n == 0 or math.isnan(score):
        return float("nan")
    ge = n - _bisect_left(calib_sorted, score)
    return (1.0 + ge) / (n + 1.0)


def report(res: JudgeResult, *, alpha: float = 0.10, top: int = 8) -> str:
    """A prescriptive per-map report: what is off, which way, and by how much."""
    v = res.verdict(alpha)
    out = [
        f"{res.label}: {v}   p={res.p_value:.3f} (bar {alpha:.2f})   "
        f"notes={int(res.n_notes)}  scored {res.n_scored} metrics",
        f"  rank score (mean extremeness) {res.s_mean:.3f}  p={res.p_mean:.3f}"
        f"   | worst-{TOPK} {res.s_topk:.3f} p={res.p_topk:.3f}"
        f"   | worst {res.s_max:.3f} p={res.p_max:.3f}",
    ]
    if res.viol is not None:
        vtxt = "clean" if res.viol == 0 else f"{int(res.viol)} PARITY VIOLATIONS - unplayable"
        out.append(f"  parity: {vtxt}")
    if res.missing:
        out.append(f"  ⚠️ not scored ({len(res.missing)}): {', '.join(res.missing)}")

    out.append("  furthest from human:")
    for m in res.worst(top):
        side = "high" if m.pct > 0.5 else "low"
        out.append(f"    {m.flag:<2} {m.name:<18} {m.value:>9.3f}  "
                   f"human pct {m.pct*100:5.1f}%  ({side}) - {m.note}")
    return "\n".join(out)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def judge_zip(path, *, difficulty: str = "Expert", onsets=None,
              reference: dict | None = None, alpha: float = 0.10) -> JudgeResult:
    """Judge one map zip -- ours or a human's -- at n=1."""
    import pathlib as _pl
    import sys as _sys

    path = _pl.Path(path)
    repo = _pl.Path(__file__).resolve().parents[3]
    _sys.path.insert(0, str(repo / "scripts"))

    notes = bpm = None
    # Our generated zips and human corpus zips have different layouts; try the
    # human loader first and fall back, rather than guessing from the filename
    # (the filename convention is exactly what silently broke `alignment`).
    try:
        from audit_eval_suite import _load_generated, _load_human  # noqa: PLC0415
        for loader in (_load_human, _load_generated):
            got = loader(path)
            if got:
                notes, bpm = got
                break
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"cannot load {path}: {exc}") from exc
    if not notes:
        raise RuntimeError(f"no notes found in {path}")

    rec = map_record(notes, bpm, onsets=onsets)
    return judge(rec, reference or load_reference(), label=path.stem)


def main() -> int:
    import argparse
    import pathlib as _pl

    ap = argparse.ArgumentParser(
        description="Judge one map (or several) at n=1 against the human corpus.")
    ap.add_argument("maps", nargs="+", type=_pl.Path)
    ap.add_argument("--alpha", type=float, default=0.10)
    ap.add_argument("--top", type=int, default=8)
    ap.add_argument("--audio", type=_pl.Path, default=None,
                    help="song audio, to score the alignment axis too")
    ap.add_argument("--brief", action="store_true", help="one line per map")
    a = ap.parse_args()

    ref = load_reference()
    onsets = None
    if a.audio is not None:
        from beatsaber_automapper.evaluation import scorecard  # noqa: PLC0415
        onsets = scorecard.onsets_for(a.audio)

    rc = 0
    for mp in a.maps:
        try:
            res = judge_zip(mp, onsets=onsets, reference=ref, alpha=a.alpha)
        except Exception as exc:  # noqa: BLE001
            print(f"{mp.name}: ERROR {exc}")
            rc = 1
            continue
        if a.brief:
            w = res.worst(1)
            print(f"{res.verdict(a.alpha):<8} p={res.p_value:5.3f}  "
                  f"{res.label:<34} worst: {w[0].name if w else '-'}")
        else:
            print(report(res, alpha=a.alpha, top=a.top))
            print()
        if res.verdict(a.alpha) == "FAIL":
            rc = max(rc, 2)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
