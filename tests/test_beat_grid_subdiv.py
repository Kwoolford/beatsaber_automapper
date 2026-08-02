"""BEAT_GRID_SUBDIV — the knob on the thing that manufactures our timing scatter.

`_quantize_to_beat_grid` snaps every Stage-1 onset to a beat subdivision. At the
default 1/8 the grid spacing is ~46ms, so it displaces notes by up to +-23ms
uniformly — predicted MAD 11.6ms, measured 11.7ms on axis A8, against a human
8.7ms whose offsets are a *peak* rather than a flat band. Stage-1's own frames are
11.6ms apart, so this function is throwing away timing the model had.

These tests pin the displacement bound at each setting, because that bound IS the
hypothesis: if the sweep shows alignment improving as the bound shrinks, the
scatter was quantisation; if it does not, the scatter is the model's and this lever
is a dead end. Either answer is worth having, but only if the lever does what it
claims.
"""
from __future__ import annotations

import os

import pytest

from beatsaber_automapper.generation.generate import _quantize_to_beat_grid

SR, HOP, BPM = 44100, 512, 161.5
MS_PER_FRAME = HOP / SR * 1000.0
FRAMES = [10, 23, 47, 88, 131, 200, 317, 404, 517]


@pytest.fixture(autouse=True)
def _clean_env():
    old = os.environ.pop("BEAT_GRID_SUBDIV", None)
    yield
    os.environ.pop("BEAT_GRID_SUBDIV", None)
    if old is not None:
        os.environ["BEAT_GRID_SUBDIV"] = old


def _max_displacement_ms(subdiv: str | None) -> float:
    if subdiv is not None:
        os.environ["BEAT_GRID_SUBDIV"] = subdiv
    out = _quantize_to_beat_grid(FRAMES, bpm=BPM, sample_rate=SR, hop_length=HOP)
    assert len(out) == len(FRAMES), "quantisation must not drop or merge onsets here"
    return max(abs(a - b) * MS_PER_FRAME for a, b in zip(FRAMES, out))


def test_default_is_unchanged_eighth_note_snapping():
    """Default behaviour is prior behaviour — the lever ships OFF."""
    assert _max_displacement_ms(None) == pytest.approx(23.2, abs=1.0)


def test_sixteenths_halve_the_displacement():
    assert _max_displacement_ms("16") == pytest.approx(11.6, abs=1.0)


def test_zero_disables_snapping_entirely():
    """Off means the model's own frame timing survives, unmodified."""
    os.environ["BEAT_GRID_SUBDIV"] = "0"
    out = _quantize_to_beat_grid(FRAMES, bpm=BPM, sample_rate=SR, hop_length=HOP)
    assert out == sorted(set(FRAMES))


def test_finer_grids_are_monotonically_gentler():
    """The bound must shrink with the subdivision, or the hypothesis is untestable."""
    bounds = [_max_displacement_ms(s) for s in ("4", "8", "16", "32")]
    assert bounds == sorted(bounds, reverse=True)


def test_disabled_snapping_still_dedupes_and_sorts():
    os.environ["BEAT_GRID_SUBDIV"] = "0"
    out = _quantize_to_beat_grid([50, 10, 50, 23], bpm=BPM, sample_rate=SR,
                                 hop_length=HOP)
    assert out == [10, 23, 50]


def test_unknown_bpm_is_left_alone():
    assert _quantize_to_beat_grid(FRAMES, bpm=0.0, sample_rate=SR,
                                  hop_length=HOP) == FRAMES
