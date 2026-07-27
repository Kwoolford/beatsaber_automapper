"""Tests for the pattern-idiom metrics (eval suite v2, axis A3)."""
from __future__ import annotations

import pytest

from beatsaber_automapper.data.beatmap import ColorNote
from beatsaber_automapper.evaluation import idiom


class _BM:
    def __init__(self, notes):
        self.color_notes = notes
        self.bomb_notes = []


def test_dt_class_buckets_by_speed():
    """The same geometric move is a different idiom at different speeds."""
    assert idiom.dt_class(0.05) == 0     # stack
    assert idiom.dt_class(0.25) == 1     # 1/16
    assert idiom.dt_class(0.5) == 2      # 1/8
    assert idiom.dt_class(1.0) == 3      # 1/4
    assert idiom.dt_class(1.9) == 4      # slow


def test_idioms_are_per_hand():
    """A transition is between consecutive notes of the SAME hand.

    Interleaved hands must not produce phantom idioms across the two.
    """
    notes = [
        ColorNote(beat=0.0, x=0, y=0, color=0, direction=1),
        ColorNote(beat=0.25, x=3, y=2, color=1, direction=0),
        ColorNote(beat=0.5, x=1, y=0, color=0, direction=0),
    ]
    out = idiom.idioms_of(_BM(notes))
    # red 0->1 only; blue has a single note so contributes nothing
    assert out == [(1, 0, 1, 0, 2)]


def test_idioms_skip_rests():
    notes = [
        ColorNote(beat=0.0, x=0, y=0, color=0, direction=1),
        ColorNote(beat=9.0, x=1, y=0, color=0, direction=0),   # > MAX_DT
    ]
    assert idiom.idioms_of(_BM(notes)) == []


def test_idiom_transition_is_order_sensitive():
    """Reversing a sequence must change its idioms — this is what makes A3
    catch the `shuffled` control that the marginal metrics cannot."""
    fwd = [ColorNote(beat=i * 0.5, x=i % 4, y=0, color=0, direction=1)
           for i in range(8)]
    rev = [ColorNote(beat=n.beat, x=x, y=0, color=0, direction=1)
           for n, x in zip(fwd, [n.x for n in fwd][::-1])]
    assert idiom.idioms_of(_BM(fwd)) != idiom.idioms_of(_BM(rev))


def test_jsd_is_zero_for_identical_distributions_and_positive_otherwise():
    p = {"a": 0.5, "b": 0.5}
    assert idiom._jsd(p, p) == pytest.approx(0.0, abs=1e-12)
    assert idiom._jsd({"a": 1.0}, {"b": 1.0}) == pytest.approx(1.0)


def test_short_map_yields_nan_not_a_fake_score():
    notes = [ColorNote(beat=i * 0.5, x=0, y=0, color=0, direction=1) for i in range(5)]
    m = idiom.idiom_metrics(_BM(notes)).metrics
    assert all(v != v for v in m.values())


def test_metrics_run_against_the_real_vocabulary():
    """Smoke test against the mined artifact, if it has been calibrated."""
    if not idiom.VOCAB_PATH.exists():
        pytest.skip("idiom vocabulary not calibrated in this checkout")
    notes = []
    for i in range(60):
        notes.append(ColorNote(beat=i * 0.5, x=i % 2, y=0, color=0,
                               direction=1 if i % 2 else 0))
    m = idiom.idiom_metrics(_BM(notes)).metrics
    assert 0.0 <= m["idiom_coverage"] <= 1.0
    assert 0.0 <= m["idiom_top50"] <= 1.0
    assert m["idiom_jsd"] >= 0.0
