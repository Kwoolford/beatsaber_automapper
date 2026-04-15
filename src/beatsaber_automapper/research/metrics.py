"""Evaluation metrics for auto-researcher experiments.

Three classes of metrics:
    1. Playability — objective checks on a generated map (parity, collisions, etc).
    2. Style-closeness — how similar generation is to the cohort's real maps.
    3. Composite score — weighted combination for leaderboard ranking.

Designed to run against a generated Beat Saber .zip + the cohort's reference
distribution (precomputed from training data).
"""

from __future__ import annotations

import json
import math
import zipfile
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

FOREHAND = {1, 6, 7}  # down, down-left, down-right
BACKHAND = {0, 4, 5}  # up, up-left, up-right
NEUTRAL = {2, 3, 8}  # left, right, any


@dataclass
class PlayabilityReport:
    n_notes: int = 0
    n_same_color_same_beat: int = 0  # same-color notes sharing a beat
    n_parity_violations: int = 0
    n_collisions: int = 0  # two notes same (x,y) same beat
    n_walls: int = 0
    n_arcs: int = 0
    n_chains: int = 0
    n_bombs: int = 0
    notes_per_sec: float = 0.0
    direction_histogram: dict[int, int] = field(default_factory=dict)
    color_balance: float = 0.5  # red/(red+blue), 0.5 = perfect

    @property
    def parity_rate(self) -> float:
        return self.n_parity_violations / max(self.n_notes - 1, 1)

    @property
    def collision_rate(self) -> float:
        return self.n_collisions / max(self.n_notes, 1)

    def as_dict(self) -> dict[str, Any]:
        return {
            "n_notes": self.n_notes,
            "n_same_color_same_beat": self.n_same_color_same_beat,
            "n_parity_violations": self.n_parity_violations,
            "n_collisions": self.n_collisions,
            "n_walls": self.n_walls,
            "n_arcs": self.n_arcs,
            "n_chains": self.n_chains,
            "n_bombs": self.n_bombs,
            "notes_per_sec": self.notes_per_sec,
            "parity_rate": self.parity_rate,
            "collision_rate": self.collision_rate,
            "direction_histogram": self.direction_histogram,
            "color_balance": self.color_balance,
        }


def _parity_class(direction: int) -> str:
    if direction in FOREHAND:
        return "fore"
    if direction in BACKHAND:
        return "back"
    return "neutral"


def analyze_beatmap(dat: dict[str, Any], song_duration_sec: float) -> PlayabilityReport:
    """Run objective playability checks against a parsed v3 beatmap dict."""
    notes = dat.get("colorNotes", [])
    bombs = dat.get("bombNotes", [])
    walls = dat.get("obstacles", [])
    arcs = dat.get("sliders", [])
    chains = dat.get("burstSliders", [])

    rpt = PlayabilityReport(
        n_notes=len(notes),
        n_walls=len(walls),
        n_arcs=len(arcs),
        n_chains=len(chains),
        n_bombs=len(bombs),
    )

    if not notes:
        return rpt

    notes_sorted = sorted(notes, key=lambda n: n.get("b", 0.0))

    # Group by beat for same-beat checks
    by_beat: dict[float, list[dict[str, Any]]] = {}
    for n in notes_sorted:
        by_beat.setdefault(round(n.get("b", 0.0), 3), []).append(n)

    for _, group in by_beat.items():
        color_counts = Counter(n.get("c", 0) for n in group)
        for _, count in color_counts.items():
            if count > 1:
                rpt.n_same_color_same_beat += count - 1
        pos_counts = Counter((n.get("x", 0), n.get("y", 0)) for n in group)
        rpt.n_collisions += sum(c - 1 for c in pos_counts.values() if c > 1)

    # Per-color parity over time
    last_dir_per_color: dict[int, str] = {}
    dir_hist: Counter[int] = Counter()
    red = blue = 0
    for n in notes_sorted:
        c = n.get("c", 0)
        d = n.get("d", 8)
        dir_hist[d] += 1
        if c == 0:
            red += 1
        else:
            blue += 1
        cls = _parity_class(d)
        prev = last_dir_per_color.get(c)
        if prev is not None and prev == cls and cls != "neutral":
            rpt.n_parity_violations += 1
        if cls != "neutral":
            last_dir_per_color[c] = cls

    rpt.direction_histogram = dict(dir_hist)
    rpt.color_balance = red / max(red + blue, 1)
    if song_duration_sec > 0:
        rpt.notes_per_sec = len(notes) / song_duration_sec
    return rpt


def analyze_generated_zip(zip_path: Path, song_duration_sec: float) -> PlayabilityReport:
    """Parse a generated Beat Saber .zip and analyze its Expert beatmap."""
    with zipfile.ZipFile(zip_path, "r") as zf:
        # Prefer ExpertStandard.dat, fall back to any difficulty .dat
        candidates = [n for n in zf.namelist() if n.lower().endswith(".dat")]
        candidates.sort(key=lambda n: 0 if "expert" in n.lower() else 1)
        for name in candidates:
            if "info" in name.lower():
                continue
            try:
                dat = json.loads(zf.read(name).decode("utf-8"))
            except Exception:
                continue
            if "colorNotes" in dat:
                return analyze_beatmap(dat, song_duration_sec)
    return PlayabilityReport()


# ------------------- Style-closeness -------------------


@dataclass
class CohortReference:
    """Aggregated statistics of a cohort's real maps — used as style target."""

    n_maps: int
    mean_nps: float
    direction_hist: dict[int, float]  # normalized
    parity_rate_baseline: float
    color_balance: float


def kl_divergence(p: dict[int, float], q: dict[int, float], eps: float = 1e-6) -> float:
    """KL(p || q) over direction bins 0..8. Smoothed."""
    kl = 0.0
    for d in range(9):
        pi = max(p.get(d, 0.0), eps)
        qi = max(q.get(d, 0.0), eps)
        kl += pi * math.log(pi / qi)
    return kl


def style_closeness(rpt: PlayabilityReport, ref: CohortReference) -> dict[str, float]:
    """Compute style-distance metrics between a generated map and cohort reference."""
    total = sum(rpt.direction_histogram.values()) or 1
    p_dir = {d: c / total for d, c in rpt.direction_histogram.items()}
    kl = kl_divergence(p_dir, ref.direction_hist)
    nps_gap = abs(rpt.notes_per_sec - ref.mean_nps)
    parity_gap = abs(rpt.parity_rate - ref.parity_rate_baseline)
    color_gap = abs(rpt.color_balance - ref.color_balance)
    return {
        "direction_kl": kl,
        "nps_gap": nps_gap,
        "parity_rate_gap": parity_gap,
        "color_balance_gap": color_gap,
    }


# ------------------- Composite score -------------------

# Default weights — tuned against known failure modes:
#   * v15 (1 note + 1572 walls) → penalized by density + wall sanity.
#   * averaged-mapper output    → penalized by style_nps + style_dir.
#   * beam-search loopholes     → penalized by parity.
# Sum = 1.0. 60% playability, 40% style.
DEFAULT_COMPOSITE_WEIGHTS: dict[str, float] = {
    "parity": 0.25,
    "collision": 0.15,
    "density": 0.10,
    "walls": 0.10,
    "style_dir": 0.15,
    "style_nps": 0.15,
    "style_parity": 0.05,
    "style_color": 0.05,
}


def _density_score(n_notes: int) -> float:
    """1.0 inside the plausible 40–2000 range; ramps linearly outside."""
    if n_notes < 40:
        return max(0.0, n_notes / 40.0)
    if n_notes > 2000:
        return max(0.0, 1.0 - (n_notes - 2000) / 2000.0)
    return 1.0


def composite_score(
    playability: PlayabilityReport,
    style: dict[str, float] | None,
    *,
    weights: dict[str, float] | None = None,
) -> float:
    """Single scalar in [0, 1]. Higher is better.

    Eight components combined linearly:
        playability (0.60 total):
            - parity        : 1 - parity_rate
            - collision     : 1 - collision_rate
            - density       : 1 inside 40-2000 notes, ramps to 0 outside
            - walls         : 1 - walls_per_note (clipped) — kills wall-spam runs
        style (0.40 total, vs cohort reference):
            - style_dir     : exp(-direction_kl)
            - style_nps     : exp(-nps_gap / 3)  — 0.37 at 3 nps off
            - style_parity  : 1 - 5*parity_rate_gap (clipped)
            - style_color   : 1 - 2*color_balance_gap (clipped)

    When no cohort reference is available, style terms default to 0.5 each
    (neutral — no bonus, no penalty for unknown-cohort experiments).
    """
    w = weights or DEFAULT_COMPOSITE_WEIGHTS

    s_parity = max(0.0, 1.0 - min(playability.parity_rate, 1.0))
    s_coll = max(0.0, 1.0 - min(playability.collision_rate, 1.0))
    s_density = _density_score(playability.n_notes)
    walls_per_note = playability.n_walls / max(playability.n_notes, 1)
    s_walls = max(0.0, 1.0 - min(walls_per_note, 1.0))

    if style is not None:
        s_style_dir = math.exp(-style.get("direction_kl", 5.0))
        s_style_nps = math.exp(-style.get("nps_gap", 5.0) / 3.0)
        s_style_parity = max(
            0.0, 1.0 - min(style.get("parity_rate_gap", 1.0) * 5.0, 1.0)
        )
        s_style_color = max(
            0.0, 1.0 - min(style.get("color_balance_gap", 0.5) * 2.0, 1.0)
        )
    else:
        s_style_dir = s_style_nps = s_style_parity = s_style_color = 0.5

    return (
        w["parity"] * s_parity
        + w["collision"] * s_coll
        + w["density"] * s_density
        + w["walls"] * s_walls
        + w["style_dir"] * s_style_dir
        + w["style_nps"] * s_style_nps
        + w["style_parity"] * s_style_parity
        + w["style_color"] * s_style_color
    )
