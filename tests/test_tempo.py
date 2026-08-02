"""Tempo + phase fitting — the grid our notes get placed on.

Axis A8 traced "the notes are off beat" to this: the pipeline's tempo was exact on
1 of 21 eval songs, and the grid PHASE was never estimated at all (anchored at
t=0). These tests pin the behaviours that made the new fitter work, including the
two that were wrong on the first attempt and are easy to reintroduce:

  * ratios are >= 1 only — maximising the fit over ALL ratios picks half tempo,
  * a metrically ambiguous fit resolves to the FINER grid, never the coarser one.
"""
from __future__ import annotations

import numpy as np
import pytest

from beatsaber_automapper.data import tempo as T


def _grid_onsets(bpm: float, phase: float = 0.0, subdiv: int = 4,
                 n: int = 400, keep=None) -> np.ndarray:
    """Onsets sitting exactly on a 1/subdiv-beat grid."""
    slot = 60.0 / bpm / subdiv
    idx = np.arange(n) if keep is None else np.array(keep)
    return phase + idx * slot


def test_recovers_an_exact_grid():
    on = _grid_onsets(160.0)
    fit = T.fit_tempo(on, 160.0)
    assert fit.bpm == pytest.approx(160.0, abs=0.2)
    assert fit.r > 0.99
    assert fit.trusted


def test_recovers_tempo_from_a_wrong_starting_point():
    """The real failure: librosa hands us 161.5 for a 160 song."""
    on = _grid_onsets(160.0)
    fit = T.fit_tempo(on, 161.5)
    assert fit.bpm == pytest.approx(160.0, abs=0.2)


def test_crosses_a_two_thirds_error_which_refinement_cannot():
    """Four eval songs come in at 2/3 of the true tempo (168 -> 112)."""
    on = _grid_onsets(168.0)
    fit = T.fit_tempo(on, 112.0)
    assert fit.bpm == pytest.approx(168.0, rel=0.01)


def test_crosses_a_half_tempo_error():
    on = _grid_onsets(188.0)
    fit = T.fit_tempo(on, 94.0)
    assert fit.bpm == pytest.approx(188.0, rel=0.01)


def test_never_returns_a_grid_coarser_than_the_starting_estimate():
    """The bug that made the first version worse than doing nothing.

    Unrestricted ratio search maximises onset concentration, which is HIGHER on
    coarser grids whenever the music emphasises every other slot — it chose half
    tempo on 12 of 23 songs, including ones already exact.
    """
    # every other slot carries an onset: a half-tempo grid fits it "better"
    on = _grid_onsets(160.0, keep=range(0, 400, 2))
    fit = T.fit_tempo(on, 160.0)
    assert fit.bpm >= 160.0 - 0.5


def test_ambiguous_levels_resolve_to_the_finer_grid():
    """A finer grid expresses everything a coarser one can; the reverse is false."""
    on = _grid_onsets(120.0)
    fit = T.fit_tempo(on, 120.0)
    # 120 and 240 fit an exact grid equally well; taking 240 is safe, 60 is not
    assert fit.bpm >= 120.0 - 0.5


def test_phase_is_estimated_not_assumed_zero():
    """The pipeline anchored every grid at t=0 regardless of the music."""
    slot = 60.0 / 160.0 / 4
    on = _grid_onsets(160.0, phase=0.37 * slot)
    fit = T.fit_tempo(on, 160.0)
    off = (fit.phase_s - 0.37 * slot) % slot
    assert min(off, slot - off) < 0.15 * slot


def test_unrelated_onsets_score_untrusted():
    """A weak fit must be reported as weak — librosa reports nonsense silently."""
    rng = np.random.default_rng(0)
    on = np.sort(rng.uniform(0, 200, size=2000))
    fit = T.fit_tempo(on, 128.0)
    assert not fit.trusted


def test_beat_lsq_recovers_period_and_phase():
    beats = 0.31 + np.arange(60) * (60.0 / 174.0)
    bpm, phase = T.beat_lsq(beats)
    assert bpm == pytest.approx(174.0, rel=1e-3)
    assert phase == pytest.approx(0.31, abs=1e-3)


def test_beat_lsq_survives_a_dropped_beat():
    beats = 0.0 + np.arange(60) * (60.0 / 150.0)
    beats = np.delete(beats, [17, 41])
    bpm, _ = T.beat_lsq(beats)
    assert bpm == pytest.approx(150.0, rel=1e-3)


def test_too_few_onsets_is_not_scored():
    fit = T.fit_tempo(np.array([1.0, 2.0, 3.0]), 128.0)
    assert not fit.trusted
