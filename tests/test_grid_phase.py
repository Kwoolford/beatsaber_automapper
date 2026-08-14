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


# --------------------------------------------------------------------------- #
# mode `search` — find the shift instead of predicting it
# --------------------------------------------------------------------------- #
def _map_at_times(times, bpm):
    """A map whose notes land at the given SECONDS."""
    return _BM([t * bpm / 60.0 for t in times])


def test_search_finds_a_planted_offset():
    bpm = 120.0
    onsets = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5]
    # 80 ms is OUTSIDE the 50 ms tolerance, so the notes are unmatched at zero and
    # the match rate actually has somewhere to go. A plant SMALLER than the
    # tolerance would leave the objective flat — see the sub-tolerance test below.
    bm = _map_at_times([t + 0.080 for t in onsets], bpm)
    shift, gain = grid_phase.search_shift(bm, bpm=bpm, onsets=onsets)
    assert shift == pytest.approx(-0.080, abs=0.003)
    assert gain > 0.0


def test_a_sub_tolerance_offset_yields_no_rate_gain_so_nothing_is_applied(monkeypatch):
    """The conservative half of the design, and it is deliberate.

    A 20 ms offset is inside the 50 ms tolerance: every note already matches, so
    the match rate cannot improve and `MIN_GAIN` refuses the shift. The tie-break
    would have centred it, but re-introducing "shift every song on a scatter-only
    signal" is exactly what made mode `1` destructive. Requiring a RATE gain is the
    guard; the scatter tie-break only chooses AMONG shifts once we have decided to
    move at all.
    """
    monkeypatch.setenv("BEAT_GRID_PHASE", "search")
    bpm = 120.0
    onsets = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5]
    bm = _map_at_times([t + 0.020 for t in onsets], bpm)
    before = [n.beat for n in bm.color_notes]
    assert grid_phase.maybe_apply(bm, bpm=bpm, phase_s=0.0, subdiv=4,
                                  onsets=onsets) is False
    assert [n.beat for n in bm.color_notes] == pytest.approx(before)


def test_the_scatter_tiebreak_centres_among_equally_matching_shifts():
    """Among shifts with the same match rate, the tighter scatter must win."""
    bpm = 120.0
    onsets = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5]
    bm = _map_at_times([t + 0.020 for t in onsets], bpm)
    shift, gain = grid_phase.search_shift(bm, bpm=bpm, onsets=onsets)
    assert gain == pytest.approx(0.0)          # rate had nowhere to go
    assert shift == pytest.approx(-0.020, abs=0.003)   # ...but it still centred


def test_search_leaves_an_already_aligned_map_alone(monkeypatch):
    """The failure mode that killed mode `1`: shifting songs that were fine."""
    monkeypatch.setenv("BEAT_GRID_PHASE", "search")
    bpm = 120.0
    onsets = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5]
    bm = _map_at_times(onsets, bpm)
    before = [n.beat for n in bm.color_notes]
    assert grid_phase.maybe_apply(bm, bpm=bpm, phase_s=0.07, subdiv=4,
                                  onsets=onsets) is False
    assert [n.beat for n in bm.color_notes] == pytest.approx(before)


def test_search_ignores_the_fitted_phase_entirely(monkeypatch):
    """`search` must not fall back to the refuted `phase_s` estimate."""
    monkeypatch.setenv("BEAT_GRID_PHASE", "search")
    bpm = 120.0
    onsets = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5]
    bm = _map_at_times([t + 0.080 for t in onsets], bpm)
    # A wildly wrong phase_s is supplied; the search must override it.
    assert grid_phase.maybe_apply(bm, bpm=bpm, phase_s=+0.090, subdiv=4,
                                  onsets=onsets) is True
    got = [n.beat * 60.0 / bpm for n in bm.color_notes]
    assert got[0] == pytest.approx(1.0, abs=0.005)


def test_search_without_onsets_refuses_loudly(monkeypatch):
    monkeypatch.setenv("BEAT_GRID_PHASE", "search")
    bm = _BM([1.0, 2.0])
    assert grid_phase.maybe_apply(bm, bpm=120.0, phase_s=0.05, subdiv=4,
                                  onsets=None) is False
    assert [n.beat for n in bm.color_notes] == [1.0, 2.0]


def test_an_unknown_mode_is_off(monkeypatch):
    monkeypatch.setenv("BEAT_GRID_PHASE", "yes")
    bm = _BM([1.0, 2.0])
    assert grid_phase.maybe_apply(bm, bpm=120.0, phase_s=0.05, subdiv=4) is False
    assert [n.beat for n in bm.color_notes] == [1.0, 2.0]
