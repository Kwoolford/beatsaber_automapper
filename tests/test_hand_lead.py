"""BEAT_HAND_LEAD: per-window budget multipliers that give one hand the lead.

The property that matters is not "asymmetry went up" — `_assign_hand_roles` could
do that too, by deleting a quarter of the notes. It is that asymmetry goes up
*while every hand keeps its total budget*, so the map stays balanced globally and
loses nothing. These tests pin that, plus the failure mode the 2026-07-27 review
found: a lead that sits on one hand, or flips every window, is not what human
mappers do (measured swap rate 0.461 per 2-bar block).
"""
from __future__ import annotations

import numpy as np
import pytest

from beatsaber_automapper.generation.generate import _lead_multipliers

BPM = 128.0
WIN_SEC = 2.0
HUMAN_ASYM = 0.115
HUMAN_SWAP = 0.461


def test_hands_are_exactly_complementary():
    """One hand is up by exactly as much as the other is down, in every window."""
    left, right = _lead_multipliers(200, WIN_SEC, BPM, HUMAN_ASYM, HUMAN_SWAP)
    assert np.allclose(left + right, 2.0)


def test_totals_stay_balanced_so_no_hand_idles():
    """Neither hand gets systematically more of the song.

    This is what separates this lever from the failed `_assign_hand_roles`: the
    per-window share moves, the total does not, so no note is deleted.
    """
    left, right = _lead_multipliers(400, WIN_SEC, BPM, HUMAN_ASYM, HUMAN_SWAP)
    assert left.mean() == pytest.approx(1.0, abs=0.05)
    assert right.mean() == pytest.approx(1.0, abs=0.05)


def test_asymmetry_magnitude_is_the_requested_target():
    for asym in (0.05, 0.115, 0.30):
        left, _ = _lead_multipliers(200, WIN_SEC, BPM, asym, HUMAN_SWAP)
        assert set(np.round(np.abs(left - 1.0), 6)) == {round(asym, 6)}


def test_lead_persists_over_a_block_and_still_swaps():
    """Human role division is a 2-bar lead, not per-window flapping (run length
    ~1.36 notes but the LEAD itself carries a passage), and it must not stick on
    one hand for the whole song.
    """
    left, _ = _lead_multipliers(300, WIN_SEC, BPM, HUMAN_ASYM, HUMAN_SWAP)
    flips = int((np.diff(left) != 0).sum())
    assert flips > 5, "lead never swaps — one hand carries the whole song"
    assert flips < 150, "lead flips almost every window — no passage-level role"
    frac_left = float((left > 1.0).mean())
    assert 0.3 < frac_left < 0.7, f"lead is stuck on one hand ({frac_left:.0%})"


def test_zero_asymmetry_is_a_no_op():
    """The OFF path must leave the budget untouched — `prod` stays byte-identical."""
    left, right = _lead_multipliers(120, WIN_SEC, BPM, 0.0, HUMAN_SWAP)
    assert np.allclose(left, 1.0) and np.allclose(right, 1.0)


def test_deterministic_for_a_given_seed():
    a, _ = _lead_multipliers(120, WIN_SEC, BPM, HUMAN_ASYM, HUMAN_SWAP, seed=7)
    b, _ = _lead_multipliers(120, WIN_SEC, BPM, HUMAN_ASYM, HUMAN_SWAP, seed=7)
    c, _ = _lead_multipliers(120, WIN_SEC, BPM, HUMAN_ASYM, HUMAN_SWAP, seed=8)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


@pytest.mark.parametrize("bpm", [94.0, 120.0, 175.0, 220.0])
def test_block_length_tracks_tempo_not_wall_clock(bpm):
    """A 2-bar lead is a musical length, so the block must be derived from BPM.

    Guards the same lesson swing_sim learned in reverse: comfort is wall-clock,
    but musical STRUCTURE is beats.
    """
    left, _ = _lead_multipliers(400, WIN_SEC, bpm, HUMAN_ASYM, HUMAN_SWAP)
    # runs of constant lead, in windows. Drop the last one: the final block is
    # truncated by n_win, so its length says nothing about the block size.
    changes = np.flatnonzero(np.diff(left) != 0)
    runs = np.diff(np.concatenate(([0], changes + 1, [len(left)])))[:-1]
    expected_block = max(round(8.0 * (60.0 / bpm) / WIN_SEC), 1)
    assert min(runs) >= expected_block - 1e-9
    assert all(r % expected_block == 0 for r in runs)
