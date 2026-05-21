"""Quantitative quality metrics for generated Beat Saber maps.

Provides three metric families:

    density_metrics  — NPS, note count, bomb ratio vs real-map distributions
    flow_metrics     — hand balance, parity alternation rate, direction entropy
    coverage_metrics — how evenly notes are spread across the song

Call ``evaluate_map()`` for a combined report, or use the individual
functions when you need a specific metric in a training loop.

All functions accept a v3 beatmap dict (``colorNotes``, ``bombNotes``, etc.)
and song metadata. The ``reference_stats`` parameter lets you compare against
a pre-computed distribution from training data (see ``compute_reference_stats``).
"""

from __future__ import annotations

import math
from collections import Counter


# ---------------------------------------------------------------------------
# Density
# ---------------------------------------------------------------------------

def density_metrics(
    notes: list[dict],
    bombs: list[dict],
    song_duration_secs: float,
    bpm: float,
) -> dict[str, float]:
    """Compute note density and bomb ratio metrics.

    Args:
        notes: colorNotes list from a v3 beatmap dict.
        bombs: bombNotes list from a v3 beatmap dict.
        song_duration_secs: Full song length in seconds.
        bpm: Song BPM.

    Returns:
        Dictionary with:
            nps            — notes per second (over full song duration)
            note_count     — total color notes
            bomb_count     — total bomb notes
            bomb_ratio     — bombs / (notes + bombs), 0 if no objects
            active_density — NPS over the "active" window (first→last note)
            notes_per_beat — note_count / total_beats
    """
    n = len(notes)
    b = len(bombs)
    total_beats = song_duration_secs * bpm / 60.0

    nps = n / max(song_duration_secs, 1e-6)
    bomb_ratio = b / max(n + b, 1)
    notes_per_beat = n / max(total_beats, 1e-6)

    active_density = nps
    if notes:
        beats_sorted = sorted(note["b"] for note in notes)
        active_beats = beats_sorted[-1] - beats_sorted[0]
        active_secs = active_beats / bpm * 60.0
        active_density = n / max(active_secs, 1e-6)

    return {
        "nps": round(nps, 3),
        "note_count": n,
        "bomb_count": b,
        "bomb_ratio": round(bomb_ratio, 3),
        "active_density": round(active_density, 3),
        "notes_per_beat": round(notes_per_beat, 3),
    }


# ---------------------------------------------------------------------------
# Flow
# ---------------------------------------------------------------------------

# Parity groups (matching postprocess / playability conventions)
_FOREHAND = frozenset({1, 6, 7})   # down, down-left, down-right
_BACKHAND = frozenset({0, 4, 5})   # up, up-left, up-right


def _parity(direction: int) -> str:
    if direction in _FOREHAND:
        return "forehand"
    if direction in _BACKHAND:
        return "backhand"
    return "neutral"


def flow_metrics(notes: list[dict], bpm: float) -> dict[str, float]:
    """Compute flow quality metrics.

    Args:
        notes: colorNotes list (each note has 'b' beat, 'c' color, 'd' direction).
        bpm: Song BPM (used to compute timing constraints).

    Returns:
        Dictionary with:
            hand_balance        — 0 = perfectly balanced, 1 = all one hand
            parity_alternation  — fraction of same-hand consecutive pairs with
                                  forehand↔backhand switch (higher = more natural)
            direction_entropy   — Shannon entropy of direction distribution [0, 1]
                                  normalised by log2(9) (9 possible directions)
            dot_fraction        — fraction of notes using dot (direction=8)
    """
    if not notes:
        return {
            "hand_balance": 0.0,
            "parity_alternation": 0.0,
            "direction_entropy": 0.0,
            "dot_fraction": 0.0,
        }

    # Separate by hand (color: 0=left/red, 1=right/blue)
    left = [n for n in notes if n.get("c", 0) == 0]
    right = [n for n in notes if n.get("c", 1) == 1]
    n_total = len(notes)

    # Hand balance: 0 = equal, 1 = completely one-sided
    hand_balance = abs(len(left) - len(right)) / max(n_total, 1)

    # Parity alternation: for each hand separately
    alternations = 0
    total_pairs = 0
    for hand_notes in (left, right):
        sorted_hand = sorted(hand_notes, key=lambda n: n["b"])
        for i in range(1, len(sorted_hand)):
            p_prev = _parity(sorted_hand[i - 1].get("d", 8))
            p_curr = _parity(sorted_hand[i].get("d", 8))
            if p_prev != "neutral" and p_curr != "neutral":
                total_pairs += 1
                if p_prev != p_curr:
                    alternations += 1
    parity_alternation = alternations / max(total_pairs, 1)

    # Direction entropy
    dir_counts = Counter(n.get("d", 8) for n in notes)
    total_dirs = sum(dir_counts.values())
    entropy = 0.0
    for count in dir_counts.values():
        p = count / total_dirs
        if p > 0:
            entropy -= p * math.log2(p)
    direction_entropy = entropy / math.log2(9)  # normalise to [0, 1]

    # Dot note fraction
    dot_fraction = dir_counts.get(8, 0) / max(n_total, 1)

    return {
        "hand_balance": round(hand_balance, 3),
        "parity_alternation": round(parity_alternation, 3),
        "direction_entropy": round(direction_entropy, 3),
        "dot_fraction": round(dot_fraction, 3),
    }


# ---------------------------------------------------------------------------
# Coverage
# ---------------------------------------------------------------------------

def coverage_metrics(
    notes: list[dict],
    song_duration_secs: float,
    bpm: float,
    n_bins: int = 8,
) -> dict[str, float]:
    """Compute how evenly notes are spread across the song.

    Args:
        notes: colorNotes list.
        song_duration_secs: Full song duration in seconds.
        bpm: Song BPM.
        n_bins: Number of equal-length time bins to divide the song into.

    Returns:
        Dictionary with:
            coverage_cv        — coefficient of variation of per-bin note counts
                                 (std / mean). Lower = more uniform.
            empty_bin_fraction — fraction of bins with zero notes.
            gini               — Gini coefficient of note distribution [0, 1].
                                 0 = perfectly uniform, 1 = all notes in one bin.
    """
    if not notes:
        return {"coverage_cv": float("inf"), "empty_bin_fraction": 1.0, "gini": 1.0}

    total_beats = song_duration_secs * bpm / 60.0
    bin_width = total_beats / n_bins

    counts = [0] * n_bins
    for note in notes:
        beat = note["b"]
        idx = min(int(beat / bin_width), n_bins - 1)
        counts[idx] += 1

    mean = sum(counts) / n_bins
    if mean == 0:
        return {"coverage_cv": float("inf"), "empty_bin_fraction": 1.0, "gini": 1.0}

    std = math.sqrt(sum((c - mean) ** 2 for c in counts) / n_bins)
    cv = std / mean

    empty_bins = sum(1 for c in counts if c == 0)

    # Gini coefficient
    sorted_counts = sorted(counts)
    n = len(sorted_counts)
    gini_num = sum((2 * (i + 1) - n - 1) * sorted_counts[i] for i in range(n))
    gini = gini_num / (n * sum(sorted_counts))

    return {
        "coverage_cv": round(cv, 3),
        "empty_bin_fraction": round(empty_bins / n_bins, 3),
        "gini": round(gini, 3),
    }


# ---------------------------------------------------------------------------
# Reference distribution
# ---------------------------------------------------------------------------

def compute_reference_stats(
    nps_samples: list[float],
) -> dict[str, float]:
    """Summarise a distribution of NPS values from real maps.

    Args:
        nps_samples: List of NPS values, one per real training map.

    Returns:
        Dictionary with p10, p25, p50, p75, p90, mean, std.
    """
    if not nps_samples:
        return {}
    s = sorted(nps_samples)
    n = len(s)

    def _pct(p: float) -> float:
        return s[min(int(p * n), n - 1)]

    mean = sum(s) / n
    variance = sum((x - mean) ** 2 for x in s) / n
    return {
        "p10": round(_pct(0.10), 3),
        "p25": round(_pct(0.25), 3),
        "p50": round(_pct(0.50), 3),
        "p75": round(_pct(0.75), 3),
        "p90": round(_pct(0.90), 3),
        "mean": round(mean, 3),
        "std": round(math.sqrt(variance), 3),
    }


def nps_percentile(nps: float, reference: dict[str, float]) -> float:
    """Estimate what percentile the given NPS falls at in the reference distribution.

    Uses linear interpolation between known percentile breakpoints.
    Returns a value in [0, 1].
    """
    if not reference:
        return 0.5
    breakpoints = [
        (0.0, 0.0),
        (reference["p10"], 0.10),
        (reference["p25"], 0.25),
        (reference["p50"], 0.50),
        (reference["p75"], 0.75),
        (reference["p90"], 0.90),
        (reference["p90"] * 2, 1.0),  # extrapolate beyond p90
    ]
    for i in range(1, len(breakpoints)):
        x0, p0 = breakpoints[i - 1]
        x1, p1 = breakpoints[i]
        if nps <= x1:
            if x1 == x0:
                return p0
            t = (nps - x0) / (x1 - x0)
            return round(p0 + t * (p1 - p0), 3)
    return 1.0


# ---------------------------------------------------------------------------
# Combined report
# ---------------------------------------------------------------------------

def evaluate_map(
    beatmap_data: dict,
    song_duration_secs: float,
    bpm: float,
    reference_stats: dict[str, float] | None = None,
    n_coverage_bins: int = 8,
) -> dict[str, object]:
    """Run all quality metrics on a generated map.

    Args:
        beatmap_data: v3 beatmap dict (colorNotes, bombNotes, ...).
        song_duration_secs: Full song duration in seconds.
        bpm: Song BPM.
        reference_stats: Optional output of ``compute_reference_stats``.
            If provided, adds ``nps_percentile`` to the report.
        n_coverage_bins: Number of bins for coverage uniformity.

    Returns:
        Flat dict of all metrics, prefixed by family:
            density.*  flow.*  coverage.*  (nps_percentile if reference given)
    """
    notes = beatmap_data.get("colorNotes", [])
    bombs = beatmap_data.get("bombNotes", [])

    d = density_metrics(notes, bombs, song_duration_secs, bpm)
    f = flow_metrics(notes, bpm)
    c = coverage_metrics(notes, song_duration_secs, bpm, n_bins=n_coverage_bins)

    report: dict[str, object] = {}
    for k, v in d.items():
        report[f"density.{k}"] = v
    for k, v in f.items():
        report[f"flow.{k}"] = v
    for k, v in c.items():
        report[f"coverage.{k}"] = v

    if reference_stats:
        report["nps_percentile"] = nps_percentile(d["nps"], reference_stats)

    return report
