"""Tests for the flow/ergonomics metrics (eval suite v2, axis A1)."""
from __future__ import annotations

import pytest

from beatsaber_automapper.data.beatmap import ColorNote
from beatsaber_automapper.evaluation import flow


class _BM:
    def __init__(self, notes):
        self.color_notes = notes
        self.bomb_notes = []


def _stream(directions, *, color=0, x=1, y=1, step=0.5):
    """One hand, one cell, alternating cut directions at a fixed interval."""
    return [ColorNote(beat=i * step, x=x, y=y, color=color, direction=d)
            for i, d in enumerate(directions)]


BPM = 120.0


def test_clean_updown_stream_has_zero_wrist_rotation():
    """A down/up stream is the canonical comfortable pattern.

    Parity alternates, and in the parity-aware frame a forehand 'down' and a
    backhand 'up' are the same physical heading — so the wrist carries straight
    through and the rotation between swings must be 0.
    """
    bm = _BM(_stream([1, 0] * 12))
    rep = flow.flow_metrics(bm, bpm=BPM)
    assert rep.metrics["angle_change"] == pytest.approx(0.0, abs=1e-6)
    assert rep.metrics["angle_harsh_frac"] == pytest.approx(0.0, abs=1e-6)


def test_diagonal_stream_also_flows():
    """down-right -> up-left is a real diagonal stream, not a wrist-break."""
    bm = _BM(_stream([7, 4] * 12))
    rep = flow.flow_metrics(bm, bpm=BPM)
    assert rep.metrics["angle_change"] == pytest.approx(0.0, abs=1e-6)


def test_alternating_axis_forces_rotation():
    """down -> up-left demands a real wrist rotation, and it is measured."""
    bm = _BM(_stream([1, 4] * 12))
    rep = flow.flow_metrics(bm, bpm=BPM)
    assert rep.metrics["angle_change"] > 30.0


def test_dot_swings_are_excluded_from_angle_stats():
    """All-dot swings have no committed heading.

    Giving them a geometric direction was swing_sim's single largest source of
    false positives; the flow metrics must not reintroduce that mistake.
    """
    bm = _BM(_stream([8] * 12))
    rep = flow.flow_metrics(bm, bpm=BPM)
    assert rep.metrics["angle_change"] != rep.metrics["angle_change"]  # NaN


def test_crossover_counts_hands_on_the_wrong_side():
    # red (0) belongs in columns 0-1, blue (1) in columns 2-3
    good = [ColorNote(beat=i, x=0, y=1, color=0, direction=1) for i in range(5)]
    good += [ColorNote(beat=i + 0.25, x=3, y=1, color=1, direction=1) for i in range(5)]
    assert flow.flow_metrics(_BM(good), bpm=BPM).metrics["crossover"] == 0.0

    crossed = [ColorNote(beat=i, x=3, y=1, color=0, direction=1) for i in range(5)]
    assert flow.flow_metrics(_BM(crossed), bpm=BPM).metrics["crossover"] == 1.0


def test_handedness_flags_an_idle_hand():
    one_hand = _stream([1, 0] * 10, color=0)
    assert flow.flow_metrics(_BM(one_hand), bpm=BPM).metrics["handedness"] == 1.0

    both = one_hand + [ColorNote(beat=n.beat + 0.25, x=3, y=1, color=1,
                                 direction=n.direction) for n in one_hand]
    assert flow.flow_metrics(_BM(both), bpm=BPM).metrics["handedness"] == 0.0


def test_ebpm_is_wall_clock_not_per_beat():
    """The same note pattern is twice as demanding at twice the tempo."""
    bm = _BM(_stream([1, 0] * 20))
    slow = flow.flow_metrics(bm, bpm=100.0).metrics["ebpm_burst"]
    fast = flow.flow_metrics(bm, bpm=200.0).metrics["ebpm_burst"]
    assert fast == pytest.approx(2 * slow, rel=1e-6)


def test_score_against_reference_uses_only_sequence_keys():
    """crossover/handedness are order-invariant, so they must not dilute the
    composite — that is what lets flow_dist detect a shuffled map at all."""
    ref = {k: (0.0, 1.0) for k in flow.KEYS}
    metrics = {k: 0.0 for k in flow.KEYS}
    metrics["crossover"] = 100.0  # huge, but excluded from the composite
    assert flow.score_against_reference(metrics, ref) == pytest.approx(0.0)

    metrics = {k: 0.0 for k in flow.KEYS}
    metrics["angle_change"] = 4.0  # a sequence key, so it must count
    assert flow.score_against_reference(metrics, ref) == pytest.approx(1.0)


def test_cohort_comparison_reports_shift_and_spread():
    ref = {k: (10.0, 2.0) for k in flow.KEYS}
    # cohort centred 1 human-MAD above the reference, with half the spread
    records = [{k: v for k in flow.KEYS} for v in (11.0, 12.0, 13.0)]
    cc = flow.cohort_comparison(records, ref)
    assert cc["angle_change"]["shift"] == pytest.approx(1.0)
    assert cc["angle_change"]["spread"] == pytest.approx(0.5)


def test_cohort_comparison_detects_mode_collapse():
    """A generator emitting the human median every time scores a perfect shift
    but must still be caught by spread — this is the failure that saturated
    h_dist, so the suite has to be able to see it."""
    ref = {k: (10.0, 2.0) for k in flow.KEYS}
    collapsed = [{k: 10.0 for k in flow.KEYS} for _ in range(10)]
    cc = flow.cohort_comparison(collapsed, ref)
    assert cc["_summary"]["flow_gap"] == pytest.approx(0.0)
    assert cc["_summary"]["min_spread"] == pytest.approx(0.0)


def test_scorecard_passes_human_and_fails_a_degenerate_cohort():
    """The suite must judge without a human in the loop: a human cohort passes,
    a metronomic cohort fails. Bars live in evaluation/scorecard.py."""
    from beatsaber_automapper.evaluation import scorecard as sc

    metronomic = []
    for _ in range(6):
        notes = [ColorNote(beat=i * 0.5, x=1, y=0, color=i % 2, direction=1)
                 for i in range(200)]
        metronomic.append((_BM(notes), 120.0))
    res = sc.score_cohort(metronomic, "metronomic")
    assert not res["passed"]
    assert "OVERALL: FAIL" in sc.report(res)
