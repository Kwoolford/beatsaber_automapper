"""Axis A8 — does the map land on the MUSIC?

The properties pinned here are the ones the axis is only useful if it has. A8 was
added because five axes agreed a map was human while Kyle could hear it was off the
beat; a metric introduced to fix a blind spot has to be shown not to have its own.

The load-bearing ones:
  * a perfectly-aligned map scores 1.0 and a randomly-timed one scores near 0,
  * doubles are ONE event (we emit 4x too many; if they counted twice, the largest
    known structural defect would inflate the new axis),
  * one onset absorbs at most one note, so note spam cannot manufacture precision,
  * a map that hits nothing scores 0.0 rather than NaN — "not scored" must never be
    reachable by being maximally bad.
"""
from __future__ import annotations

import numpy as np
import pytest

from beatsaber_automapper.data.beatmap import ColorNote
from beatsaber_automapper.evaluation import alignment

BPM = 120.0
SPB = 60.0 / BPM


class _BM:
    def __init__(self, notes):
        self.color_notes = notes
        self.bomb_notes = []


def _map_at(times_s, colors=None):
    """A map whose notes sit at the given times in seconds."""
    colors = colors or [i % 2 for i in range(len(times_s))]
    return _BM([ColorNote(beat=t / SPB, x=1, y=1, color=c, direction=1)
                for t, c in zip(times_s, colors)])


def test_perfect_alignment_scores_one():
    onsets = np.arange(0.0, 60.0, 0.5)
    m = alignment.alignment_metrics(_map_at(list(onsets)), bpm=BPM, onsets=onsets)
    assert m.metrics["onset_precision"] == pytest.approx(1.0)
    assert m.metrics["offset_mad_ms"] == pytest.approx(0.0, abs=1e-6)


def test_random_timing_scores_near_zero():
    """The decisive control: human note count, no relationship to the music."""
    onsets = np.arange(0.0, 60.0, 0.5)
    rng = np.random.default_rng(0)
    times = sorted(rng.uniform(0.0, 60.0, size=120).tolist())
    m = alignment.alignment_metrics(_map_at(times), bpm=BPM, onsets=onsets)
    # onsets every 500ms with a 50ms window => ~20% hit by chance; must be far
    # below the ~0.97 a human map scores.
    assert m.metrics["onset_precision"] < 0.35


def test_jitter_inflates_scatter_before_it_kills_precision():
    """Kyle's actual complaint: right notes, sloppily placed.

    A8 has to move on this, otherwise it only catches maps that are wrong and not
    maps that are merely loose — and 'loose' is what we ship.
    """
    onsets = np.arange(0.0, 60.0, 0.5)
    rng = np.random.default_rng(1)
    jittered = [float(t + rng.uniform(-0.03, 0.03)) for t in onsets]
    tight = alignment.alignment_metrics(_map_at(list(onsets)), bpm=BPM, onsets=onsets)
    loose = alignment.alignment_metrics(_map_at(jittered), bpm=BPM, onsets=onsets)
    assert loose.metrics["offset_mad_ms"] > tight.metrics["offset_mad_ms"] + 5.0
    assert loose.metrics["onset_precision"] > 0.9


def test_a_double_is_one_musical_event():
    """Two hands on the same beat must not count as two hits on one onset.

    We emit ~4x too many doubles (0.77 vs human 0.231). If a double scored twice,
    that defect would quietly raise this axis instead of being neutral to it.
    """
    onsets = np.arange(0.0, 60.0, 0.5)
    times = [float(t) for t in onsets]
    single = alignment.alignment_metrics(_map_at(times), bpm=BPM, onsets=onsets)
    doubled = alignment.alignment_metrics(
        _map_at(times + times, colors=[0] * len(times) + [1] * len(times)),
        bpm=BPM, onsets=onsets)
    assert single.n_notes == doubled.n_notes
    assert doubled.metrics["onset_precision"] == pytest.approx(
        single.metrics["onset_precision"])


def test_one_onset_absorbs_only_one_note():
    """Precision must not be gameable by stacking hits on a single loud event."""
    onsets = np.array([1.0, 30.0])
    times = [1.0 + 0.001 * i for i in range(60)]
    m = alignment.alignment_metrics(_map_at(times), bpm=BPM, onsets=onsets)
    assert m.metrics["onset_precision"] < 0.05


def test_recall_is_reported_but_not_gated():
    """Humans ignore most onsets on purpose — gating recall would reward note spam."""
    assert "onset_recall" in alignment.KEYS
    assert "onset_recall" not in alignment.SEQUENCE_KEYS
    assert "onset_lag_ms" not in alignment.SEQUENCE_KEYS


def test_missing_audio_is_not_scored_rather_than_passed():
    onsets = np.arange(0.0, 60.0, 0.5)
    m = alignment.alignment_metrics(_map_at(list(onsets)), bpm=BPM, onsets=None)
    assert all(np.isnan(v) for v in m.metrics.values())


def test_total_miss_scores_zero_not_nan():
    """Maximally bad must be a FAIL, never a 'not scored'."""
    onsets = np.array([100.0, 200.0])
    m = alignment.alignment_metrics(_map_at([float(i) * 0.5 for i in range(80)]),
                                    bpm=BPM, onsets=onsets)
    assert m.metrics["onset_precision"] == 0.0
    assert not np.isnan(m.metrics["offset_mad_ms"])


def test_constant_lag_shows_as_lag_not_scatter():
    """A map that is uniformly 25ms late is a sync bug, not a musicality bug."""
    onsets = np.arange(0.0, 60.0, 0.5)
    late = [float(t) + 0.025 for t in onsets]
    m = alignment.alignment_metrics(_map_at(late), bpm=BPM, onsets=onsets)
    assert m.metrics["onset_lag_ms"] == pytest.approx(25.0, abs=1.0)
    assert m.metrics["offset_mad_ms"] < 1.0
    assert m.metrics["onset_precision"] == pytest.approx(1.0)


def test_short_maps_are_not_scored():
    onsets = np.arange(0.0, 60.0, 0.5)
    m = alignment.alignment_metrics(_map_at([0.5, 1.0, 1.5]), bpm=BPM, onsets=onsets)
    assert all(np.isnan(v) for v in m.metrics.values())
