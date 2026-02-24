"""Tests for the post-processing pipeline."""

from __future__ import annotations

from beatsaber_automapper.data.beatmap import ColorNote, DifficultyBeatmap
from beatsaber_automapper.generation.postprocess import (
    deduplicate_patterns,
    diversify_directions,
    enforce_nps,
    expand_grid_coverage,
    fix_parity,
    postprocess_beatmap,
    rebalance_colors,
)


def _make_beatmap(notes: list[ColorNote] | None = None) -> DifficultyBeatmap:
    """Helper to create a simple beatmap."""
    return DifficultyBeatmap(version="3.3.0", color_notes=notes or [])


def _note(beat: float, x: int = 1, y: int = 0, color: int = 0, direction: int = 1) -> ColorNote:
    """Shorthand for creating a ColorNote."""
    return ColorNote(beat=beat, x=x, y=y, color=color, direction=direction)


class TestEnforceNPS:
    def test_no_thinning_when_under_target(self):
        # 10 notes over 10 seconds = 1 NPS, well under Expert max of 8
        notes = [_note(beat=i * 1.0) for i in range(10)]
        bm = _make_beatmap(notes)
        result = enforce_nps(bm, difficulty="Expert", bpm=60.0, song_duration_secs=10.0)
        assert len(result.color_notes) == 10

    def test_thinning_when_over_target(self):
        # 100 notes over 10 seconds = 10 NPS, above Expert max of 8
        notes = [_note(beat=i * 0.5) for i in range(100)]
        bm = _make_beatmap(notes)
        result = enforce_nps(bm, difficulty="Expert", bpm=120.0, song_duration_secs=10.0)
        # Should thin to around target_nps=6.0, ~60 notes
        assert len(result.color_notes) < 100
        assert len(result.color_notes) > 30

    def test_empty_beatmap(self):
        bm = _make_beatmap([])
        result = enforce_nps(bm, difficulty="Expert", bpm=120.0)
        assert len(result.color_notes) == 0


class TestRebalanceColors:
    def test_already_balanced(self):
        notes = [_note(i, color=i % 2) for i in range(20)]
        bm = _make_beatmap(notes)
        result = rebalance_colors(bm)
        red = sum(1 for n in result.color_notes if n.color == 0)
        assert 0.40 <= red / len(result.color_notes) <= 0.60

    def test_heavily_skewed_red(self):
        # 90% red
        notes = [_note(i, color=0) for i in range(90)] + [_note(90 + i, color=1) for i in range(10)]
        bm = _make_beatmap(notes)
        result = rebalance_colors(bm)
        red = sum(1 for n in result.color_notes if n.color == 0)
        # Should be pushed toward 50%
        assert red / len(result.color_notes) < 0.65

    def test_too_few_notes_skipped(self):
        notes = [_note(0, color=0), _note(1, color=0)]
        bm = _make_beatmap(notes)
        result = rebalance_colors(bm)
        # < 10 notes, should skip
        assert all(n.color == 0 for n in result.color_notes)


class TestDiversifyDirections:
    def test_all_same_direction(self):
        # 50 notes all direction=1 (down)
        notes = [_note(i, direction=1) for i in range(50)]
        bm = _make_beatmap(notes)
        result = diversify_directions(bm)
        dir_counts = {}
        for n in result.color_notes:
            dir_counts[n.direction] = dir_counts.get(n.direction, 0) + 1
        # No direction should be > 40%
        for d, c in dir_counts.items():
            assert c <= 0.40 * len(result.color_notes) + 1  # +1 for rounding

    def test_already_diverse(self):
        # Evenly spread across 4 directions
        notes = [_note(i, direction=i % 4) for i in range(40)]
        bm = _make_beatmap(notes)
        result = diversify_directions(bm)
        # 10/40 = 25% each, all under 40%, no changes needed
        assert len(result.color_notes) == 40


class TestExpandGridCoverage:
    def test_low_coverage_expanded(self):
        # All notes in cell (1, 0)
        notes = [_note(i, x=1, y=0) for i in range(30)]
        bm = _make_beatmap(notes)
        result = expand_grid_coverage(bm, min_coverage=6)
        cells = {(n.x, n.y) for n in result.color_notes}
        assert len(cells) > 1

    def test_good_coverage_unchanged(self):
        # Spread across 10 cells
        cells_list = [(x, y) for x in range(4) for y in range(3)][:10]
        notes = [_note(i, x=cells_list[i % 10][0], y=cells_list[i % 10][1]) for i in range(30)]
        bm = _make_beatmap(notes)
        result = expand_grid_coverage(bm, min_coverage=8)
        assert len(result.color_notes) == 30


class TestDeduplicatePatterns:
    def test_long_run_varied(self):
        # 15 identical notes in a row
        notes = [_note(i, x=1, y=0, color=0, direction=1) for i in range(15)]
        bm = _make_beatmap(notes)
        result = deduplicate_patterns(bm, max_repeats=5)
        # After first 5, some should have different directions
        dirs = [n.direction for n in result.color_notes[5:]]
        assert any(d != 1 for d in dirs)

    def test_short_run_unchanged(self):
        notes = [_note(i, x=1, y=0, color=0, direction=1) for i in range(4)]
        bm = _make_beatmap(notes)
        result = deduplicate_patterns(bm, max_repeats=5)
        assert all(n.direction == 1 for n in result.color_notes)


class TestFixParity:
    def test_consecutive_forehand_fixed(self):
        # Two consecutive down swings for same color = parity violation
        notes = [
            _note(0, color=0, direction=1),  # down (forehand)
            _note(1, color=0, direction=1),  # down (forehand) - violation
        ]
        bm = _make_beatmap(notes)
        result = fix_parity(bm)
        # Second note should be flipped to backhand
        assert result.color_notes[1].direction in {0, 4, 5}

    def test_alternating_parity_unchanged(self):
        notes = [
            _note(0, color=0, direction=1),  # down (forehand)
            _note(1, color=0, direction=0),  # up (backhand)
            _note(2, color=0, direction=1),  # down (forehand)
        ]
        bm = _make_beatmap(notes)
        result = fix_parity(bm)
        assert result.color_notes[0].direction == 1
        assert result.color_notes[1].direction == 0
        assert result.color_notes[2].direction == 1

    def test_different_colors_independent(self):
        # Each color has its own parity stream
        notes = [
            _note(0, color=0, direction=1),
            _note(0.5, color=1, direction=1),  # different color, not a violation
            _note(1, color=0, direction=0),  # correct alternation for red
            _note(1.5, color=1, direction=0),  # correct alternation for blue
        ]
        bm = _make_beatmap(notes)
        result = fix_parity(bm)
        # No changes needed
        assert result.color_notes[0].direction == 1
        assert result.color_notes[2].direction == 0


class TestFullPipeline:
    def test_postprocess_beatmap_runs(self):
        notes = [_note(i * 0.5, color=0, direction=1) for i in range(50)]
        bm = _make_beatmap(notes)
        result = postprocess_beatmap(bm, difficulty="Expert", bpm=120.0, seed=42)
        assert len(result.color_notes) > 0

    def test_empty_beatmap_safe(self):
        bm = _make_beatmap([])
        result = postprocess_beatmap(bm, difficulty="Expert", bpm=120.0)
        assert len(result.color_notes) == 0
