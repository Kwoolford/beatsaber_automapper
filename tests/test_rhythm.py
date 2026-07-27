"""Tests for the rhythm/beat-grid metrics (eval suite v2, axis A2)."""
from __future__ import annotations

import random

import pytest

from beatsaber_automapper.data.beatmap import ColorNote
from beatsaber_automapper.evaluation import rhythm


class _BM:
    def __init__(self, notes):
        self.color_notes = notes
        self.bomb_notes = []


def _at(beats):
    return _BM([ColorNote(beat=b, x=1, y=1, color=0, direction=1) for b in beats])


def test_constant_pulse_is_maximally_stable():
    """A single unbroken spacing is the metronomic degenerate case."""
    bm = _at([i * 0.5 for i in range(80)])
    m = rhythm.rhythm_metrics(bm).metrics
    assert m["pulse_stability"] == pytest.approx(1.0)
    assert m["ioi_cond_entropy"] == pytest.approx(0.0)
    assert m["dominant_share"] == pytest.approx(1.0)


def test_varied_rhythm_breaks_the_pulse():
    beats, t = [], 0.0
    for i in range(80):
        t += (0.25, 0.5, 1.0)[i % 3]
        beats.append(t)
    m = rhythm.rhythm_metrics(_at(beats)).metrics
    assert m["pulse_stability"] < 0.1        # spacing changes every note
    assert m["dominant_share"] < 0.5


def test_offgrid_guard_detects_jitter():
    on = rhythm.rhythm_metrics(_at([i * 0.25 for i in range(80)])).metrics
    assert on["offgrid_frac"] == pytest.approx(0.0)

    rng = random.Random(0)
    off = rhythm.rhythm_metrics(
        _at([i * 0.25 + rng.uniform(0.02, 0.04) for i in range(80)])).metrics
    assert off["offgrid_frac"] > 0.9


def test_rhythm_is_tempo_independent():
    """Metrics are computed in the beat domain, so BPM must not matter."""
    bm = _at([i * 0.5 for i in range(80)])
    a = rhythm.rhythm_metrics(bm, bpm=100.0).metrics
    b = rhythm.rhythm_metrics(bm, bpm=220.0).metrics
    assert a == b


def test_switch_rate_counts_gear_changes():
    steady = _at([i * 0.5 for i in range(80)])
    assert rhythm.rhythm_metrics(steady).metrics["ioi_switch_rate"] == pytest.approx(0.0)

    # alternating blocks of 1/8 and 1/16 spacing = repeated gear changes
    beats, t = [], 0.0
    for blk in range(10):
        step = 0.5 if blk % 2 == 0 else 0.25
        for _ in range(8):
            t += step
            beats.append(t)
    assert rhythm.rhythm_metrics(_at(beats)).metrics["ioi_switch_rate"] > 0.0


def test_too_few_notes_yields_nan_not_a_fake_score():
    m = rhythm.rhythm_metrics(_at([0.0, 0.5, 1.0])).metrics
    assert all(v != v for k, v in m.items() if k in rhythm.SEQUENCE_KEYS)


def test_cohort_comparison_exposes_our_known_collapse():
    """A cohort that is uniformly metronomic must show up as shifted AND collapsed."""
    ref = {k: (0.55, 0.10) for k in rhythm.KEYS}
    collapsed = [{k: 0.95 for k in rhythm.KEYS} for _ in range(10)]
    cc = rhythm.cohort_comparison(collapsed, ref)
    assert cc["_summary"]["rhythm_gap"] == pytest.approx(4.0)
    assert cc["_summary"]["min_spread"] == pytest.approx(0.0)
