"""P1.0 phase-outlier gate (2026-09-17): fires on a real disagreement, never on a half-slot lock."""
import sys
import pathlib

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "agent_mapper"))
import mapctl  # noqa: E402

SPB = 60.0 / 93.0
Q = SPB / 4


def _onsets(true_phase, n=400):
    # onsets on the true grid, read EARLY by the detector bias, with a little jitter
    rng = np.random.default_rng(0)
    k = np.arange(n)
    return true_phase + k * Q + mapctl.ONSET_GRID_BIAS_S + rng.normal(0, 0.004, n)


def test_agrees_keeps_phase():
    assert mapctl.phase_outlier(0.010, SPB, _onsets(0.010)) is None


def test_real_disagreement_moves_to_the_onset_grid():
    got = mapctl.phase_outlier(-0.034, SPB, _onsets(-0.086))
    assert got is not None and abs(got - (-0.086)) < 0.006


def test_half_slot_lock_is_never_trusted():
    # a disagreement of exactly half a slot is the onset grid's known failure mode
    assert mapctl.phase_outlier(0.0, SPB, _onsets(Q / 2)) is None
