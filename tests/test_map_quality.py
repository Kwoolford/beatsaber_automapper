"""Tests for evaluation/map_quality.py metrics."""

import math
import pytest

from beatsaber_automapper.evaluation.map_quality import (
    compute_reference_stats,
    coverage_metrics,
    density_metrics,
    evaluate_map,
    flow_metrics,
    nps_percentile,
)


def _note(beat: float, color: int = 0, direction: int = 1) -> dict:
    return {"b": beat, "c": color, "d": direction, "x": 1, "y": 1}


def _bomb(beat: float) -> dict:
    return {"b": beat, "x": 0, "y": 0}


# ---------------------------------------------------------------------------
# density_metrics
# ---------------------------------------------------------------------------

class TestDensityMetrics:
    def test_basic_nps(self):
        notes = [_note(i * 0.5) for i in range(20)]  # 20 notes over 10 beats
        result = density_metrics(notes, [], song_duration_secs=10.0, bpm=120.0)
        assert result["nps"] == pytest.approx(2.0, rel=0.01)
        assert result["note_count"] == 20
        assert result["bomb_count"] == 0
        assert result["bomb_ratio"] == 0.0

    def test_bomb_ratio(self):
        notes = [_note(i) for i in range(3)]
        bombs = [_bomb(i + 0.5) for i in range(7)]
        result = density_metrics(notes, bombs, song_duration_secs=10.0, bpm=120.0)
        assert result["bomb_ratio"] == pytest.approx(7 / 10)
        assert result["bomb_count"] == 7

    def test_empty_map(self):
        result = density_metrics([], [], song_duration_secs=60.0, bpm=120.0)
        assert result["nps"] == 0.0
        assert result["bomb_ratio"] == 0.0

    def test_notes_per_beat(self):
        notes = [_note(i) for i in range(10)]  # 10 notes, song is 120 beats
        result = density_metrics(notes, [], song_duration_secs=60.0, bpm=120.0)
        assert result["notes_per_beat"] == pytest.approx(10 / 120, rel=0.01)


# ---------------------------------------------------------------------------
# flow_metrics
# ---------------------------------------------------------------------------

class TestFlowMetrics:
    def test_perfect_balance(self):
        # Alternating left/right
        notes = [_note(i, color=i % 2) for i in range(10)]
        result = flow_metrics(notes, bpm=120.0)
        assert result["hand_balance"] == pytest.approx(0.0, abs=0.01)

    def test_one_sided(self):
        notes = [_note(i, color=0) for i in range(10)]
        result = flow_metrics(notes, bpm=120.0)
        assert result["hand_balance"] == pytest.approx(1.0, abs=0.01)

    def test_parity_alternation_perfect(self):
        # Alternates forehand (d=1) and backhand (d=0) for left hand
        notes = [_note(i * 0.5, color=0, direction=1 if i % 2 == 0 else 0) for i in range(10)]
        result = flow_metrics(notes, bpm=120.0)
        assert result["parity_alternation"] > 0.8

    def test_parity_no_alternation(self):
        # All same direction — no alternation
        notes = [_note(i * 0.5, color=0, direction=1) for i in range(10)]
        result = flow_metrics(notes, bpm=120.0)
        assert result["parity_alternation"] == pytest.approx(0.0, abs=0.01)

    def test_direction_entropy_uniform(self):
        # All 9 directions equally represented
        notes = [_note(i, color=0, direction=i % 9) for i in range(90)]
        result = flow_metrics(notes, bpm=120.0)
        assert result["direction_entropy"] == pytest.approx(1.0, rel=0.05)

    def test_direction_entropy_single(self):
        notes = [_note(i, color=0, direction=1) for i in range(10)]
        result = flow_metrics(notes, bpm=120.0)
        assert result["direction_entropy"] == pytest.approx(0.0, abs=0.01)

    def test_dot_fraction(self):
        notes = [_note(i, direction=8 if i % 2 == 0 else 1) for i in range(10)]
        result = flow_metrics(notes, bpm=120.0)
        assert result["dot_fraction"] == pytest.approx(0.5, rel=0.01)

    def test_empty(self):
        result = flow_metrics([], bpm=120.0)
        assert result["hand_balance"] == 0.0
        assert result["parity_alternation"] == 0.0


# ---------------------------------------------------------------------------
# coverage_metrics
# ---------------------------------------------------------------------------

class TestCoverageMetrics:
    def test_perfect_uniform(self):
        # 8 notes exactly one per bin in a 60s song (120 BPM = 120 beats)
        beats_per_bin = 120 / 8  # 15 beats per bin
        notes = [_note(i * beats_per_bin + 0.1) for i in range(8)]
        result = coverage_metrics(notes, song_duration_secs=60.0, bpm=120.0, n_bins=8)
        assert result["empty_bin_fraction"] == 0.0
        assert result["coverage_cv"] == pytest.approx(0.0, abs=0.01)
        assert result["gini"] == pytest.approx(0.0, abs=0.01)

    def test_all_in_one_bin(self):
        notes = [_note(0.1 * i) for i in range(10)]  # all in first bin
        result = coverage_metrics(notes, song_duration_secs=60.0, bpm=120.0, n_bins=8)
        assert result["empty_bin_fraction"] == pytest.approx(7 / 8)
        assert result["gini"] > 0.8

    def test_empty_map(self):
        result = coverage_metrics([], song_duration_secs=60.0, bpm=120.0)
        assert result["empty_bin_fraction"] == 1.0
        assert result["gini"] == 1.0


# ---------------------------------------------------------------------------
# reference stats + percentile
# ---------------------------------------------------------------------------

class TestReferenceStats:
    def test_basic_percentiles(self):
        samples = list(range(1, 101))  # 1..100
        stats = compute_reference_stats([float(x) for x in samples])
        assert stats["p50"] == pytest.approx(50.0, abs=1.0)
        assert stats["p10"] == pytest.approx(10.0, abs=1.0)
        assert stats["p90"] == pytest.approx(90.0, abs=1.0)

    def test_percentile_lookup_below_p10(self):
        stats = compute_reference_stats([float(i) for i in range(1, 101)])
        pct = nps_percentile(5.0, stats)
        assert pct < 0.10

    def test_percentile_lookup_median(self):
        stats = compute_reference_stats([float(i) for i in range(1, 101)])
        pct = nps_percentile(50.0, stats)
        assert 0.45 < pct < 0.55


# ---------------------------------------------------------------------------
# evaluate_map (integration)
# ---------------------------------------------------------------------------

class TestEvaluateMap:
    def test_full_report_keys(self):
        notes = [_note(i * 0.5, color=i % 2, direction=i % 9) for i in range(40)]
        bombs = [_bomb(i * 2.0) for i in range(5)]
        beatmap = {"colorNotes": notes, "bombNotes": bombs}
        result = evaluate_map(beatmap, song_duration_secs=60.0, bpm=120.0)

        assert "density.nps" in result
        assert "density.bomb_ratio" in result
        assert "flow.hand_balance" in result
        assert "flow.parity_alternation" in result
        assert "flow.direction_entropy" in result
        assert "coverage.coverage_cv" in result
        assert "coverage.empty_bin_fraction" in result
        assert "coverage.gini" in result
        assert "nps_percentile" not in result  # not provided reference

    def test_with_reference(self):
        notes = [_note(i * 0.5, color=i % 2) for i in range(20)]
        beatmap = {"colorNotes": notes, "bombNotes": []}
        ref = compute_reference_stats([1.0, 2.0, 3.0, 4.0, 5.0])
        result = evaluate_map(beatmap, song_duration_secs=10.0, bpm=120.0, reference_stats=ref)
        assert "nps_percentile" in result
        assert 0.0 <= result["nps_percentile"] <= 1.0
