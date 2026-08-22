"""A wall a note sits inside is unplayable, and NO metric in the suite can see it.

Adding 84 walls + 48 arcs + 16 chains moves every evaluation axis by exactly 0.000 —
the suite scores notes and nothing else (PROGRESS 2026-08-19g). So wall placement has
no safety net except this test.

★The specific bug this pins: `walls.py` chooses lanes no note occupies, but `idiomize`
REDRAWS every note's column afterwards. Placing walls before idiomize produced **12
trapped notes** while the lane, duration and width statistics all still matched the
human idiom perfectly — the map looked right on every number available and was
unplayable.
"""
from __future__ import annotations

import numpy as np

from agent_mapper import walls as W  # noqa: F401  (import path check)


def _plan(note_beats, note_x, n_walls=40, seed=0):
    import agent_mapper.walls as mod
    rng = np.random.default_rng(seed)
    return mod.plan_walls(np.asarray(note_beats, dtype=float),
                          np.asarray(note_x, dtype=int),
                          (0.0, 200.0), n_walls, rng)


def test_no_wall_traps_a_note():
    # a dense map using every lane, so a careless placement WILL collide
    beats = np.arange(0.0, 200.0, 0.5)
    xs = np.tile([0, 1, 2, 3], len(beats) // 4 + 1)[: len(beats)]
    plan = _plan(beats, xs)
    for w in plan:
        lanes = set(range(w["x"], w["x"] + w["w"]))
        lo, hi = w["b"], w["b"] + w["d"]
        for b, x in zip(beats, xs):
            if lo <= b <= hi and x in lanes:
                raise AssertionError(
                    f"wall at beat {w['b']} lane {w['x']} traps a note at {b}"
                )


def test_walls_stay_in_outer_lanes():
    """93 % of human walls are in an outer lane; ours should never be central."""
    beats = np.arange(0.0, 200.0, 4.0)
    xs = np.ones(len(beats), dtype=int)  # notes only in lane 1
    for w in _plan(beats, xs, n_walls=30):
        assert w["x"] in (0, 3), f"wall in central lane {w['x']}"


def test_no_walls_when_every_lane_is_busy():
    """A map with a note everywhere should get few or no walls, not unplayable ones."""
    beats = np.repeat(np.arange(0.0, 200.0, 0.25), 4)
    xs = np.tile([0, 1, 2, 3], len(beats) // 4 + 1)[: len(beats)]
    plan = _plan(beats, xs, n_walls=50)
    for w in plan:
        lanes = set(range(w["x"], w["x"] + w["w"]))
        lo, hi = w["b"], w["b"] + w["d"]
        assert not any(lo <= b <= hi and x in lanes for b, x in zip(beats, xs))
