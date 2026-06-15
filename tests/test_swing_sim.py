"""Unit tests for the swing simulator (evaluation/swing_sim.py, P1-1).

Tiny hand-authored maps with known parity properties pin down each rule:
clean alternation passes, true wrist-breaks are flagged, and the playable
exceptions (dot-absorbed flips, angle rolls, bomb resets, lone doubles) do not
false-positive. These are the mechanics the DoD sweep on real maps exercises at
scale (scripts/eval_swing_sim.py); here we lock them in deterministically.
"""

from __future__ import annotations

from beatsaber_automapper.data.beatmap import BombNote, ColorNote, DifficultyBeatmap
from beatsaber_automapper.evaluation import swing_sim as ss

# direction codes: 0=up 1=down 2=left 3=right 4=up-left 5=up-right
#                  6=down-left 7=down-right 8=dot
BPM = 120.0  # 1 beat = 0.5 s


def _map(notes, bombs=None):
    return DifficultyBeatmap(
        version="3.0.0",
        color_notes=notes,
        bomb_notes=bombs or [],
    )


def _blue(beat, direction, x=1, y=1):
    return ColorNote(beat=beat, x=x, y=y, color=1, direction=direction)


def test_clean_alternation_has_no_resets():
    # down, up, down, up ... a beat apart: textbook forehand/backhand alternation.
    notes = [_blue(i, 1 if i % 2 == 0 else 0) for i in range(8)]
    card = ss.simulate(_map(notes), bpm=BPM)
    assert card.resets == 0
    assert card.violations == 0


def test_fast_same_direction_stream_is_violation():
    # Four down-cuts an eighth-note apart (0.125 s at 120 BPM): a sustained reset
    # run the wrist cannot recover from -> violations (the V7 chaos signature).
    notes = [_blue(i * 0.25, 1) for i in range(4)]
    card = ss.simulate(_map(notes), bpm=BPM)
    assert card.resets >= 2
    assert card.violations >= 1


def test_lone_fast_double_is_not_a_violation():
    # up, down, DOWN, up: a single forehand double surrounded by clean alternation
    # is a playable "double", not a violation.
    notes = [_blue(0, 0), _blue(1, 1), _blue(1.25, 1), _blue(2, 0)]
    card = ss.simulate(_map(notes), bpm=BPM)
    assert card.resets == 1
    assert card.violations == 0


def test_dot_absorbs_parity_flip():
    # up, dot, up: the dot is parity-free and absorbs the flip, so the second up is
    # NOT a reset even though both are backhand.
    notes = [_blue(0, 0), _blue(0.5, 8), _blue(1.0, 0)]
    card = ss.simulate(_map(notes), bpm=BPM)
    assert card.resets == 0
    assert card.violations == 0


def test_diagonal_roll_is_not_a_reset():
    # down-left then down-right: same (fore) parity but a 90 degrees wrist roll, the
    # player sweeps through -> not a reset.
    notes = [_blue(0, 6), _blue(0.5, 7), _blue(1.0, 6), _blue(1.5, 7)]
    card = ss.simulate(_map(notes), bpm=BPM)
    assert card.resets == 0
    assert card.violations == 0


def test_bomb_makes_reset_intentional():
    # A fast up-stream (run of resets) is a violation on its own...
    notes = [_blue(0, 0), _blue(0.25, 0), _blue(0.5, 0)]
    card = ss.simulate(_map(notes), bpm=BPM)
    assert card.violations >= 1

    # ...but with bombs laced through it, it reads as a deliberate bomb reset.
    bombs = [BombNote(beat=0.125, x=1, y=0), BombNote(beat=0.375, x=1, y=0)]
    card_bomb = ss.simulate(_map(notes, bombs), bpm=BPM)
    assert card_bomb.violations == 0


def test_slow_reset_is_intentional_not_violation():
    # up, then up again a full beat later (0.5 s): enough time to re-cock -> the
    # reset is intentional, not a wrist-break.
    notes = [_blue(0, 0), _blue(1.0, 0)]
    card = ss.simulate(_map(notes), bpm=BPM)
    assert card.resets == 1
    assert card.violations == 0


def test_hands_are_independent():
    # A red down-stream and a clean blue alternation: only red should accrue resets.
    red = [ColorNote(beat=i * 0.25, x=1, y=1, color=0, direction=1) for i in range(4)]
    blue = [_blue(i, 1 if i % 2 == 0 else 0) for i in range(4)]
    card = ss.simulate(_map(red + blue), bpm=BPM)
    assert card.per_hand[0].violations >= 1
    assert card.per_hand[1].violations == 0


def test_seam_hand_states_reports_parity():
    notes = [_blue(0, 1), _blue(1, 0), _blue(2, 1), _blue(3, 0)]
    card = ss.simulate(_map(notes), bpm=BPM)
    seams = ss.seam_hand_states(card, section_beats=[1.5])
    assert seams[0]["beat"] == 1.5
    assert seams[0]["hands"][1]["exit_parity"] in ("FOREHAND", "BACKHAND")
    assert seams[0]["hands"][1]["enter_parity"] in ("FOREHAND", "BACKHAND")
