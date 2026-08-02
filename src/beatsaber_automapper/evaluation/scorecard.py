"""The single evaluation entry point — one scoring system, one verdict.

This is the consolidation step of docs/eval_suite_v2.md (§1 Finding 4 recorded
that "how good is this map?" had three different answers in this repo), and it is
the thing that lets the suite stand in for a human judge: give it a cohort of
maps, get back a per-axis verdict and one overall PASS/FAIL.

Design rules it enforces, all learned the hard way (see the doc):

* **Judge a COHORT, not a map.** Every axis is scored by median shift + spread
  against the human distribution. A per-map distance to the human median rewards
  mode collapse — a generator that always emits the average scores *better than
  real human maps*, which is exactly how the old `h_dist` saturated.
* **Spread is a first-class failure.** An arm that matches every human median but
  emits the same map every time is not human-like. `min_spread` well below 1.0
  fails, independently of the gap.
* **No single scalar.** The axes measure different things and a map can be good
  at one and broken at another; collapsing them into one weighted number is how
  `research/metrics.py::composite_score` became untrustworthy. Report all axes,
  and let the overall verdict be the AND of them.

Thresholds are set from the control battery (`scripts/audit_eval_suite.py`): a
held-out human cohort scores A1 0.21 / A2 0.31 / A3 0.50, and the degenerate
controls score several times higher. The bars below sit at roughly twice the
human cohort's own gap — inside normal human variation, far below any control.

CLI::

    python -m beatsaber_automapper.evaluation.scorecard outputs/eval_sweep_cache/prod__*.zip
"""
from __future__ import annotations

import pathlib
import sys
from dataclasses import dataclass, field

from beatsaber_automapper.evaluation import (
    alignment, flow, handrole, idiom, playfeel, rhythm, swing_sim,
)

# A8's bars, set by scripts/calibrate_alignment.py from the held-out human cohort
# (same rule as every other axis: ~2x the human cohort's own gap). Kept as named
# constants because they were measured, not chosen — see docs/eval_suite_v2.md.
ALIGN_GAP_BAR = 1.00
ALIGN_SPREAD_BAR = 0.35

# (module, gap key, gap bar, min-spread bar). Bars are ~2x the human cohort's own
# gap; spread bar catches mode collapse independently of the gap.
AXES = [
    ("flow",     flow,     "flow_gap",     0.50, 0.35),
    ("rhythm",   rhythm,   "rhythm_gap",   0.70, 0.35),
    ("idiom",    idiom,    "idiom_gap",    1.00, 0.35),
    # A6 has a looser bar because the human cohort's own gap is higher here
    # (0.96 on 12 maps) — role division varies a lot between human mappers, which
    # is itself the point. Our maps sit at 3.07, worse than a uniformly random
    # map, so the bar is nowhere near the binding constraint.
    ("handrole", handrole, "handrole_gap", 2.00, 0.35),
    # A7 difficulty + direction idiom. Added 2026-07-27 after Kyle found the maps
    # unplayable as Expert: nothing in the scorecard gated note density or the
    # up/down-vs-diagonal balance, so the suite was scoring rhythm and flow while
    # the map was a difficulty tier too dense and made of diagonals.
    ("playfeel", playfeel, "playfeel_gap", 1.00, 0.35),
    # A8 audio alignment. Added 2026-08-01, the day Kyle played the first maps to
    # pass the five axes above and said the notes were off the beat. He was right
    # and NONE of those five could see it: not one of them loads the audio. This
    # is the only axis that measures where the notes sit relative to the MUSIC
    # rather than relative to each other. Bars set from the human corpus by
    # scripts/calibrate_alignment.py.
    ("alignment", alignment, "alignment_gap", ALIGN_GAP_BAR, ALIGN_SPREAD_BAR),
]


@dataclass(slots=True)
class AxisResult:
    name: str
    gap: float
    min_spread: float
    gap_bar: float
    spread_bar: float
    detail: dict = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        ok_gap = self.gap == self.gap and self.gap <= self.gap_bar
        ok_spread = self.min_spread == self.min_spread and self.min_spread >= self.spread_bar
        return bool(ok_gap and ok_spread)

    @property
    def reason(self) -> str:
        if self.gap != self.gap:
            return "not scored"
        bits = []
        if self.gap > self.gap_bar:
            bits.append(f"gap {self.gap:.2f} > {self.gap_bar:.2f}")
        if self.min_spread < self.spread_bar:
            bits.append(f"spread {self.min_spread:.2f} < {self.spread_bar:.2f} (collapsed)")
        return ", ".join(bits) if bits else "within human range"


def _metrics_for(bm, bpm: float, onsets=None) -> dict:
    rec: dict = {}
    # A8 is the only axis that needs anything beyond the notes themselves. When
    # `onsets` is None the axis reports NaN and the axis verdict becomes
    # "not scored" — which FAILS. That is deliberate: a cohort whose alignment was
    # never measured has not passed, it has been un-audited, and treating those
    # two as the same thing is the mistake this axis exists to correct.
    try:
        rec.update(alignment.alignment_metrics(bm, bpm=bpm, onsets=onsets).metrics)
    except Exception:  # noqa: BLE001
        pass
    try:
        rec.update(flow.flow_metrics(bm, bpm=bpm).metrics)
    except Exception:  # noqa: BLE001
        pass
    try:
        rec.update(rhythm.rhythm_metrics(bm).metrics)
    except Exception:  # noqa: BLE001
        pass
    try:
        rec.update(idiom.idiom_metrics(bm).metrics)
    except Exception:  # noqa: BLE001
        pass
    try:
        rec.update(handrole.handrole_metrics(bm).metrics)
    except Exception:  # noqa: BLE001
        pass
    try:
        rec.update(playfeel.playfeel_metrics(bm, bpm=bpm).metrics)
    except Exception:  # noqa: BLE001
        pass
    try:
        rec["viol"] = swing_sim.simulate(bm, bpm=bpm).violations
    except Exception:  # noqa: BLE001
        rec["viol"] = None
    return rec


def score_cohort(maps: list[tuple], label: str = "cohort") -> dict:
    """Score a cohort. `maps` is a list of (beatmap, bpm) or (beatmap, bpm, onsets).

    The 2-tuple form is still accepted so every pre-A8 caller keeps working, but it
    cannot score alignment — `_load_any` returns the 3-tuple and should be used.
    """
    records = [_metrics_for(*(m if len(m) == 3 else (*m, None))) for m in maps]
    results = []
    for name, mod, gap_key, gap_bar, spread_bar in AXES:
        cc = mod.cohort_comparison(records)
        s = cc.get("_summary", {})
        results.append(AxisResult(
            name=name, gap=s.get(gap_key, float("nan")),
            min_spread=s.get("min_spread", float("nan")),
            gap_bar=gap_bar, spread_bar=spread_bar,
            detail={k: v for k, v in cc.items() if k != "_summary"},
        ))
    viols = [r["viol"] for r in records if r.get("viol") is not None]
    total_viol = int(sum(viols)) if viols else None
    return {
        "label": label,
        "n_maps": len(records),
        "axes": results,
        "total_viol": total_viol,
        # Playability is a hard gate, not an axis: a map with wrist-breaks is
        # unplayable regardless of how human its statistics look.
        "passed": all(a.passed for a in results) and (total_viol == 0),
        "records": records,
    }


def report(res: dict) -> str:
    lines = [f"=== scorecard: {res['label']} ({res['n_maps']} maps) ===", ""]
    lines.append(f"{'axis':10s}{'gap':>8s}{'bar':>8s}{'spread':>9s}{'bar':>7s}  verdict")
    lines.append("-" * 62)
    for a in res["axes"]:
        mark = "PASS" if a.passed else "FAIL"
        lines.append(f"{a.name:10s}{a.gap:8.2f}{a.gap_bar:8.2f}"
                     f"{a.min_spread:9.2f}{a.spread_bar:7.2f}  {mark} — {a.reason}")
    v = res["total_viol"]
    lines.append(f"{'parity':10s}{'':8s}{'':8s}{'':9s}{'':7s}  "
                 f"{'PASS' if v == 0 else 'FAIL'} — {v} swing violations")
    lines.append("")
    lines.append(f"OVERALL: {'PASS' if res['passed'] else 'FAIL'}")
    if not res["passed"]:
        worst = max((a for a in res["axes"] if not a.passed),
                    key=lambda a: a.gap if a.gap == a.gap else -1, default=None)
        if worst is not None:
            lines.append(f"worst axis: {worst.name} ({worst.reason})")
    return "\n".join(lines)


ONSET_CACHE = pathlib.Path(__file__).resolve().parents[3] / "outputs" / "onset_cache"


def song_id(path: pathlib.Path) -> str:
    """Song id from a map path. `<arm>__1f767.zip` and `1f767.zip` both -> `1f767`.

    Generated maps are cached as `<arm>__<song>.zip` by eval_sweep.py; human maps
    live at `data/raw/<song>.zip`. Both resolve to the same id so both are scored
    against the same onsets — the shared footing A8 depends on.
    """
    stem = path.stem
    return stem.split("__")[-1] if "__" in stem else stem


_ONSETS: dict[str, object] = {}


def onsets_for(path: pathlib.Path):
    """Cached onsets for a map's song, or None if the song was never cached.

    Never computes onsets on the fly: `scripts/build_onset_cache.py` owns that, so
    that every map of a song is scored against byte-identical detections.
    """
    import numpy as np

    sid = song_id(path)
    if sid not in _ONSETS:
        f = ONSET_CACHE / f"{sid}.npz"
        try:
            _ONSETS[sid] = np.load(f, allow_pickle=False)["onsets"] if f.exists() else None
        except Exception:  # noqa: BLE001
            _ONSETS[sid] = None
    return _ONSETS[sid]


def _load_any(path: pathlib.Path):
    """Load a generated or human map zip as (beatmap, bpm, onsets).

    Returns a 3-tuple as of A8 (2026-08-01); `onsets` is None for songs with no
    cached audio onsets. `score_cohort` accepts both shapes, so pre-A8 callers that
    just forward this value keep working.
    """
    repo = pathlib.Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(repo / "scripts"))
    from eval_contour_follow import _load_notes_with_direction
    from feel_disc_poc import _zip_bpm

    from beatsaber_automapper.data.beatmap import ColorNote

    recs = _load_notes_with_direction(path, "Expert")
    if not recs:
        return None
    notes = [ColorNote(beat=b, x=int(x), y=int(y), color=int(c), direction=int(d))
             for (b, x, y, c, d) in recs]

    class _BM:
        color_notes = notes
        bomb_notes: list = []

    return _BM(), float(_zip_bpm(str(path)) or 120.0), onsets_for(path)


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description="Score a cohort of maps on every v2 axis.")
    ap.add_argument("maps", nargs="+")
    ap.add_argument("--label", default="cohort")
    a = ap.parse_args()

    loaded = []
    for m in a.maps:
        try:
            r = _load_any(pathlib.Path(m))
        except Exception:  # noqa: BLE001
            r = None
        if r:
            loaded.append(r)
    if not loaded:
        print("no maps could be loaded")
        raise SystemExit(2)
    res = score_cohort(loaded, a.label)
    print(report(res))
    raise SystemExit(0 if res["passed"] else 1)


if __name__ == "__main__":
    main()
