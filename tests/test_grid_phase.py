"""BEAT_GRID_PHASE — the lever that puts the beat grid on the music's downbeat.

Covers the three things this project has been bitten by on every previous lever:
a default-OFF lever must be a true no-op when off, a rigid translation must not
change anything except the times, and a shift must not silently eat notes.
"""

from __future__ import annotations

import pytest

from beatsaber_automapper.generation import grid_phase


class _Note:
    __slots__ = ("beat", "x", "y", "color", "direction")

    def __init__(self, beat, x=0, y=0, color=0, direction=1):
        self.beat, self.x, self.y = beat, x, y
        self.color, self.direction = color, direction


class _BM:
    def __init__(self, beats, bombs=()):
        self.color_notes = [_Note(b) for b in beats]
        self.bomb_notes = [_Note(b) for b in bombs]


def test_wrap_to_slot_folds_a_whole_slot_to_zero():
    # A whole slot of offset is the SAME grid, so it must wrap away entirely.
    bpm, subdiv = 120.0, 4
    slot = 60.0 / bpm / subdiv          # 0.125 s
    assert grid_phase.wrap_to_slot(slot, bpm, subdiv) == pytest.approx(0.0, abs=1e-12)
    assert grid_phase.wrap_to_slot(3 * slot, bpm, subdiv) == pytest.approx(0.0, abs=1e-12)
    # ...and anything else folds into +-half a slot.
    for raw in (0.03, -0.03, 0.4, -1.7, 12.34):
        w = grid_phase.wrap_to_slot(raw, bpm, subdiv)
        assert -slot / 2 - 1e-12 <= w <= slot / 2 + 1e-12


def test_wrap_preserves_the_grid_it_describes():
    bpm, subdiv = 137.0, 4
    slot = 60.0 / bpm / subdiv
    raw = 0.071
    w = grid_phase.wrap_to_slot(raw, bpm, subdiv)
    # The wrapped phase differs from the raw one by a whole number of slots.
    assert (raw - w) / slot == pytest.approx(round((raw - w) / slot), abs=1e-9)


def test_off_by_default_is_a_true_noop(monkeypatch):
    monkeypatch.delenv("BEAT_GRID_PHASE", raising=False)
    bm = _BM([0.0, 1.0, 2.5])
    before = [n.beat for n in bm.color_notes]
    assert grid_phase.maybe_apply(bm, bpm=120.0, phase_s=0.05, subdiv=4) is False
    assert [n.beat for n in bm.color_notes] == before


def test_shift_translates_every_note_by_the_same_amount():
    bm = _BM([1.0, 2.0, 4.5], bombs=[3.0])
    bpm, phase = 120.0, 0.05          # 0.05 s at 120 bpm = 0.1 beats
    dropped = grid_phase.shift_beatmap(bm, bpm=bpm, phase_s=phase)
    assert dropped == 0
    assert [n.beat for n in bm.color_notes] == pytest.approx([1.1, 2.1, 4.6])
    assert [n.beat for n in bm.bomb_notes] == pytest.approx([3.1])


def test_shift_preserves_every_interval():
    """A rigid translation must not change the rhythm, only where it starts."""
    beats = [0.5, 1.0, 1.75, 3.0, 3.25]
    bm = _BM(beats)
    gaps_before = [b - a for a, b in zip(beats, beats[1:])]
    grid_phase.shift_beatmap(bm, bpm=150.0, phase_s=-0.02)
    after = [n.beat for n in bm.color_notes]
    gaps_after = [b - a for a, b in zip(after, after[1:])]
    assert gaps_after == pytest.approx(gaps_before)


def test_a_negative_shift_drops_only_what_falls_before_the_song():
    # 0.25 s at 120 bpm = 0.5 beats, so the note at 0.25 goes negative and is
    # dropped; clamping it to 0 instead would stack it onto the note at 0.75 and
    # manufacture a chord that was never generated.
    bm = _BM([0.25, 0.75, 2.0])
    dropped = grid_phase.shift_beatmap(bm, bpm=120.0, phase_s=-0.25)
    assert dropped == 1
    assert [n.beat for n in bm.color_notes] == pytest.approx([0.25, 1.5])


def test_zero_phase_changes_nothing():
    bm = _BM([1.0, 2.0])
    assert grid_phase.shift_beatmap(bm, bpm=120.0, phase_s=0.0) == 0
    assert [n.beat for n in bm.color_notes] == [1.0, 2.0]


@pytest.mark.parametrize("bpm,subdiv", [(0.0, 4), (-120.0, 4), (120.0, 0)])
def test_degenerate_inputs_are_refused_not_applied(monkeypatch, bpm, subdiv):
    """A bad bpm or subdiv must refuse loudly, not silently report success."""
    monkeypatch.setenv("BEAT_GRID_PHASE", "1")
    bm = _BM([1.0, 2.0])
    assert grid_phase.maybe_apply(bm, bpm=bpm, phase_s=0.4, subdiv=subdiv) is False
    assert [n.beat for n in bm.color_notes] == [1.0, 2.0]


def test_a_whole_slot_of_phase_is_the_same_grid_and_does_not_shift(monkeypatch):
    monkeypatch.setenv("BEAT_GRID_PHASE", "1")
    bm = _BM([1.0, 2.0])
    slot = 60.0 / 120.0 / 4          # 0.125 s
    assert grid_phase.maybe_apply(bm, bpm=120.0, phase_s=slot, subdiv=4) is False
    assert [n.beat for n in bm.color_notes] == [1.0, 2.0]


def test_enabled_applies_once_and_is_not_idempotent_by_accident(monkeypatch):
    """Applying twice must shift twice — silent idempotence would hide a double call."""
    monkeypatch.setenv("BEAT_GRID_PHASE", "1")
    bm = _BM([1.0])
    assert grid_phase.maybe_apply(bm, bpm=120.0, phase_s=0.05, subdiv=4) is True
    once = bm.color_notes[0].beat
    assert once == pytest.approx(1.1)
    grid_phase.maybe_apply(bm, bpm=120.0, phase_s=0.05, subdiv=4)
    assert bm.color_notes[0].beat == pytest.approx(1.2)
