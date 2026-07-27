"""Hand-role division — axis A6 of the v2 evaluation suite.

Discovered 2026-07-27 by *reading* a map next to its human counterpart in
`scripts/map_view.py`, not by any statistic. In the human map, within a passage
**one hand carries a sustained run while the other punctuates sparsely**, and the
two swap that job between passages. Our maps run both hands at identical density
throughout, with no role division at all.

The signature, measured over 24 held-out human maps vs 24 of ours:

    metric                     human    ours
    same-hand run length        1.41    1.05
    local asymmetry (2 bars)    0.113   0.031
    dominant-hand swap rate     0.411   0.269

The important part is **globally balanced but locally lopsided**. Both cohorts
are near-perfectly balanced over a whole song (`flow.handedness` ~0.012 for
both), so the existing hand metric sees nothing. Human maps get that global
balance by giving one hand the lead for a stretch and then swapping; ours get it
by splitting every bar down the middle. Balance at *every* scale is the
unnatural thing, and no axis measured it.

Metrics:
  role_asymmetry   mean |L-R| / (L+R) within a 2-bar window. How lopsided the
                   hand split is locally. Human ~0.11, ours ~0.03.
                   RULE: let one hand lead a passage; do not split every bar
                   evenly.
  role_swap_rate   fraction of consecutive windows where the dominant hand
                   changes. Guards the obvious failure of the rule above — a map
                   where the LEFT hand always leads is lopsided but not human.
                   RULE: swap which hand leads, roughly every other passage.
  role_run_len     mean length of same-hand runs in the time-ordered note stream.
                   Reported as a GUARD, not a composite driver: notes that fall
                   on the same beat are ordered L-then-R, so a map whose hands
                   fire simultaneously has run length ~1.0 by construction. That
                   makes it largely a restatement of the A2 simultaneity finding
                   rather than independent evidence of role division.

Calibrate with::

    python scripts/calibrate_handrole.py --n 200
"""
from __future__ import annotations

import json
import pathlib
import statistics
from dataclasses import dataclass, field

from beatsaber_automapper.evaluation import _dist

KEYS = ["role_asymmetry", "role_swap_rate", "role_run_len"]
# run_len is entangled with simultaneity (see docstring), so it stays a guard.
SEQUENCE_KEYS = ["role_asymmetry", "role_swap_rate"]

REFERENCE_PATH = (
    pathlib.Path(__file__).resolve().parents[3] / "outputs" / "handrole_human_reference.json"
)

WINDOW_BEATS = 8.0     # 2 bars — long enough for a hand to "carry" something
MIN_NOTES_PER_WINDOW = 4


@dataclass(slots=True)
class HandRoleReport:
    metrics: dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {"metrics": {k: round(v, 4) for k, v in self.metrics.items()}}


def handrole_metrics(beatmap) -> HandRoleReport:
    notes = sorted(beatmap.color_notes, key=lambda n: (n.beat, n.color))
    rep = HandRoleReport()
    if len(notes) < 40:
        rep.metrics = {k: float("nan") for k in KEYS}
        return rep

    # same-hand runs in the time-ordered stream (guard)
    runs, cur = [], 1
    for a, b in zip(notes, notes[1:]):
        if a.color == b.color:
            cur += 1
        else:
            runs.append(cur)
            cur = 1
    runs.append(cur)

    # local asymmetry + which hand leads each window
    by_win: dict[int, list[int]] = {}
    for n in notes:
        by_win.setdefault(int(n.beat // WINDOW_BEATS), []).append(n.color)
    asym, dom = [], []
    for _w, cs in sorted(by_win.items()):
        if len(cs) < MIN_NOTES_PER_WINDOW:
            continue
        left = cs.count(0)
        right = len(cs) - left
        asym.append(abs(left - right) / len(cs))
        dom.append(0 if left > right else 1)

    if len(asym) < 3:
        rep.metrics = {k: float("nan") for k in KEYS}
        return rep

    swaps = ([1.0 if a != b else 0.0 for a, b in zip(dom, dom[1:])]
             if len(dom) > 1 else [])
    rep.metrics = {
        "role_asymmetry": statistics.fmean(asym),
        "role_swap_rate": statistics.fmean(swaps) if swaps else float("nan"),
        "role_run_len": statistics.fmean(runs),
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
                                   gap_name="handrole_gap")


def calibrate(records: list[dict]) -> dict[str, dict]:
    return _dist.calibrate(records, KEYS)
