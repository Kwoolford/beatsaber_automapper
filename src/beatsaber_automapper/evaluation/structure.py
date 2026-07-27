"""Structural self-consistency — axis A5. ⚠️ DORMANT: DOES NOT DISCRIMINATE.

**Result 2026-07-27: this axis is NOT SHIPPED. Do not add it to the scorecard.**
Kept as a documented negative so the idea is not re-attempted the same way.

The premise below — that human maps echo themselves at bar-aligned lags — is
FALSE as measured. Probed on 15 held-out human maps, 12 of ours, and the
degenerate controls, with three different similarity tokens (rhythm-only,
rhythm+hand, full note tuple), `struct_lift` came out ≈ 0 for **every** cohort
including human:

    token         human lift   prod lift   random lift   metronome lift
    rhythm          +0.001      -0.009       +0.001         -0.002
    rhythm+hand     +0.007      -0.005       +0.003         -0.002
    full            +0.001      -0.004       +0.003         -0.002

Human maps are no more self-similar at 8/16/32-bar lags than at 5/11/21-bar
lags. Why the shortcut fails: you cannot assume where repeated sections *are*.
Song structure does not sit at fixed bar multiples across genres, so a fixed-lag
probe cannot find it. **A5 done properly needs audio-derived section boundaries**
— identify which sections actually repeat, then ask whether the map echoes them.
We already have that machinery (`detect_sections`, phrase boundaries in
generation); re-spec A5 on top of it rather than on fixed lags.

One corroborating observation worth keeping: with the rhythm+hand token, our
maps' `struct_recall` is 0.587 against a human 0.329 — our maps DO repeat
themselves far more than humans. But since lift is ~0, that repetition is
uniform rather than structural, which makes it a restatement of the A2 rhythm
finding (we are metronomic) rather than a new axis. It would not earn its place.

--- original design notes below ---

See ``docs/eval_suite_v2.md``. A1-A3 all score a map by its *local* behaviour:
flow between consecutive swings, rhythm between consecutive onsets, idioms
between consecutive notes of a hand. A map can be locally perfect and still have
no long-range structure at all — the second chorus bearing no relation to the
first, an eight-bar phrase never echoed. Human mappers repeat themselves
deliberately: a returning musical section gets a recognisably related pattern,
varied rather than duplicated.

Measured map-only (no audio): compare bar-aligned windows of the note stream and
ask how self-similar the map is at *musical* lags versus arbitrary ones.

  struct_recall    best similarity found at a bar-aligned lag (8/16/32 bars),
                   averaged over windows. High = the map echoes itself at
                   musical distances.
                   RULE: when a section returns, echo the pattern you used before.
  struct_lift      bar-aligned similarity MINUS similarity at deliberately
                   non-aligned lags. This is the discriminating quantity: a
                   metronomic map is trivially self-similar at *every* lag and so
                   scores high recall but ~zero lift, while a map that genuinely
                   tracks song structure is more similar at musical distances than
                   at arbitrary ones.
                   RULE: repetition must be tied to musical position, not constant.
  struct_novelty   fraction of windows whose best bar-aligned match is below a
                   similarity floor — sections that are genuinely new material.
                   RULE: repeat, but keep introducing new material too.

`struct_lift` is the composite driver, for the same reason `flow_gap` excludes
crossover: a metric that a degenerate map can max out is not measuring the thing
we care about.
"""
from __future__ import annotations

import json
import pathlib
import statistics
from dataclasses import dataclass, field

from beatsaber_automapper.evaluation import _dist

KEYS = ["struct_recall", "struct_lift", "struct_novelty"]
SEQUENCE_KEYS = ["struct_lift", "struct_novelty"]

REFERENCE_PATH = (
    pathlib.Path(__file__).resolve().parents[3] / "outputs" / "structure_human_reference.json"
)

BEATS_PER_BAR = 4.0
WIN_BARS = 4.0                     # comparison window
ALIGNED_LAGS = (8.0, 16.0, 32.0)   # bars — where musical repeats live
OFFSET_LAGS = (5.0, 11.0, 21.0)    # bars — deliberately NOT section-aligned
NOVELTY_FLOOR = 0.35


@dataclass(slots=True)
class StructureReport:
    metrics: dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {"metrics": {k: round(v, 4) for k, v in self.metrics.items()}}


def _cells(beatmap, win_beats: float) -> dict[int, set]:
    """Window index -> set of (beat-offset-in-window, x, y, color) tokens."""
    out: dict[int, set] = {}
    for n in beatmap.color_notes:
        w = int(n.beat // win_beats)
        off = round((n.beat - w * win_beats) * 4) / 4.0   # quantise to 1/4 beat
        out.setdefault(w, set()).add((off, n.x, n.y, n.color))
    return out


def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 0.0
    u = len(a | b)
    return len(a & b) / u if u else 0.0


def structure_metrics(beatmap) -> StructureReport:
    win_beats = WIN_BARS * BEATS_PER_BAR
    cells = _cells(beatmap, win_beats)
    rep = StructureReport()
    if len(cells) < 8:
        rep.metrics = {k: float("nan") for k in KEYS}
        return rep

    win_per_bar = 1.0 / WIN_BARS
    aligned = [int(round(l * win_per_bar)) for l in ALIGNED_LAGS]
    offset = [int(round(l * win_per_bar)) for l in OFFSET_LAGS]
    aligned = [l for l in aligned if l > 0]
    offset = [l for l in offset if l > 0 and l not in aligned]

    best_aligned, best_offset = [], []
    for w, cur in cells.items():
        if not cur:
            continue
        ba = max((_jaccard(cur, cells[w - l]) for l in aligned if (w - l) in cells),
                 default=None)
        bo = max((_jaccard(cur, cells[w - l]) for l in offset if (w - l) in cells),
                 default=None)
        if ba is not None:
            best_aligned.append(ba)
        if bo is not None:
            best_offset.append(bo)

    if len(best_aligned) < 3:
        rep.metrics = {k: float("nan") for k in KEYS}
        return rep

    recall = statistics.fmean(best_aligned)
    off = statistics.fmean(best_offset) if best_offset else 0.0
    rep.metrics = {
        "struct_recall": recall,
        "struct_lift": recall - off,
        "struct_novelty": sum(1 for v in best_aligned if v < NOVELTY_FLOOR) / len(best_aligned),
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
                                   gap_name="structure_gap")


def calibrate(records: list[dict]) -> dict[str, dict]:
    return _dist.calibrate(records, KEYS)
