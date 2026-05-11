"""Tests for the V6 swing-event tokenizer and saber-state extractor."""

from __future__ import annotations

import math

import pytest
import torch

from beatsaber_automapper.data.beatmap import (
    BombNote,
    BurstSlider,
    ColorNote,
    DifficultyBeatmap,
    Slider,
)
from beatsaber_automapper.data.saber_state import (
    PARITY_RESET_BEATS,
    compute_saber_states,
    compute_saber_states_from_beatmap,
)
from beatsaber_automapper.data.swing_tokenizer import (
    ARC_HEAD,
    ARC_TAIL,
    BOS,
    BOMB,
    CHAIN_HEAD,
    CHAIN_TAIL,
    DT_BASE,
    DT_COUNT,
    EOS,
    EVENT_LENGTHS,
    HAND_LEFT,
    HAND_RIGHT,
    HAND_NONE,
    KIND_BASE,
    KIND_LENGTHS,
    NOTE,
    PAD,
    VOCAB_SIZE,
    SwingEventTokenizer,
    _DT_BINS,
    _nearest_bin,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _bm(**kwargs) -> DifficultyBeatmap:
    defaults = dict(
        version="3.3.0",
        color_notes=[],
        bomb_notes=[],
        obstacles=[],
        sliders=[],
        burst_sliders=[],
        basic_events=[],
        color_boost_events=[],
    )
    defaults.update(kwargs)
    return DifficultyBeatmap(**defaults)


def _tok() -> SwingEventTokenizer:
    return SwingEventTokenizer()


def _round_trip(beatmap: DifficultyBeatmap) -> DifficultyBeatmap:
    t = _tok()
    return t.decode_beatmap(t.encode_beatmap(beatmap))


def _beats_close(a: float, b: float, tol: float = 0.1) -> bool:
    return abs(a - b) <= tol


# ---------------------------------------------------------------------------
# Vocabulary basics
# ---------------------------------------------------------------------------


def test_vocab_size() -> None:
    assert VOCAB_SIZE == 118


def test_special_tokens_distinct() -> None:
    assert len({PAD, BOS, EOS}) == 3


def test_kind_lengths_sane() -> None:
    # KIND_LENGTHS = tokens from the KIND token onwards (excluding HAND + Δt)
    assert KIND_LENGTHS[NOTE] == 5        # KIND X Y DIR ANGLE
    assert KIND_LENGTHS[ARC_HEAD] == 5    # KIND X Y DIR MU
    assert KIND_LENGTHS[ARC_TAIL] == 5    # KIND X Y DIR MU
    assert KIND_LENGTHS[CHAIN_HEAD] == 5  # KIND X Y DIR SLICE
    assert KIND_LENGTHS[CHAIN_TAIL] == 4  # KIND X Y SQUISH
    assert KIND_LENGTHS[BOMB] == 3        # KIND X Y


def test_event_lengths_include_hand_and_dt() -> None:
    for kind, kl in KIND_LENGTHS.items():
        assert EVENT_LENGTHS[kind] == kl + 2


def test_dt_bins_ordered() -> None:
    for i in range(len(_DT_BINS) - 1):
        assert _DT_BINS[i] <= _DT_BINS[i + 1]


def test_dt_bins_count() -> None:
    assert len(_DT_BINS) == DT_COUNT


def test_dt_base_token_range() -> None:
    assert DT_BASE + DT_COUNT <= VOCAB_SIZE


def test_nearest_bin_exact() -> None:
    bins = [0.0, 0.5, 1.0]
    assert _nearest_bin(0.0, bins) == 0
    assert _nearest_bin(0.5, bins) == 1
    assert _nearest_bin(1.0, bins) == 2


def test_nearest_bin_between() -> None:
    bins = [0.0, 1.0, 2.0]
    assert _nearest_bin(0.4, bins) == 0
    assert _nearest_bin(0.6, bins) == 1
    assert _nearest_bin(1.9, bins) == 2


def test_tokenizer_properties() -> None:
    t = _tok()
    assert t.vocab_size == VOCAB_SIZE
    assert t.pad_token == PAD
    assert t.bos_token == BOS
    assert t.eos_token == EOS
    assert "NOTE" in t.kind_tokens
    assert "BOMB" in t.kind_tokens


# ---------------------------------------------------------------------------
# Encoding structure
# ---------------------------------------------------------------------------


def test_empty_beatmap_stream() -> None:
    tokens = _tok().encode_beatmap(_bm())
    assert tokens == [BOS, EOS]


def test_stream_starts_with_bos_ends_with_eos() -> None:
    bm = _bm(color_notes=[ColorNote(beat=1.0, x=1, y=0, color=0, direction=1)])
    tokens = _tok().encode_beatmap(bm)
    assert tokens[0] == BOS
    assert tokens[-1] == EOS


def test_single_note_token_count() -> None:
    bm = _bm(color_notes=[ColorNote(beat=2.0, x=0, y=0, color=0, direction=0)])
    tokens = _tok().encode_beatmap(bm)
    # BOS + [HAND Δt NOTE X Y DIR ANGLE] + EOS = 9 tokens
    assert len(tokens) == 9


def test_bomb_token_count() -> None:
    bm = _bm(bomb_notes=[BombNote(beat=1.0, x=1, y=1)])
    tokens = _tok().encode_beatmap(bm)
    # BOS + [NONE Δt BOMB X Y] + EOS = 7 tokens
    assert len(tokens) == 7


def test_all_tokens_in_vocab() -> None:
    bm = _bm(
        color_notes=[
            ColorNote(beat=1.0, x=0, y=0, color=0, direction=0, angle_offset=-15),
            ColorNote(beat=1.0, x=3, y=2, color=1, direction=7, angle_offset=30),
        ],
        bomb_notes=[BombNote(beat=2.0, x=2, y=1)],
        sliders=[Slider(color=0, beat=3.0, x=1, y=0, direction=0, mu=1.0,
                        tail_beat=4.0, tail_x=1, tail_y=2, tail_direction=1, tail_mu=1.0)],
        burst_sliders=[BurstSlider(color=1, beat=5.0, x=2, y=1, direction=3,
                                   tail_beat=5.5, tail_x=3, tail_y=1,
                                   slice_count=4, squish=0.5)],
    )
    tokens = _tok().encode_beatmap(bm)
    for tok in tokens:
        assert 0 <= tok < VOCAB_SIZE, f"Token {tok} out of range"


def test_hand_assignment_red_is_left_blue_is_right() -> None:
    bm = _bm(color_notes=[
        ColorNote(beat=1.0, x=0, y=0, color=0, direction=1),  # red
        ColorNote(beat=1.0, x=3, y=0, color=1, direction=0),  # blue
    ])
    tokens = _tok().encode_beatmap(bm)
    # First event after BOS: HAND_LEFT for red
    assert tokens[1] == HAND_LEFT
    # Second event: HAND_RIGHT for blue
    second_event_start = 1 + EVENT_LENGTHS[NOTE]
    assert tokens[second_event_start] == HAND_RIGHT


def test_chord_second_event_dt_zero() -> None:
    bm = _bm(color_notes=[
        ColorNote(beat=4.0, x=0, y=0, color=0, direction=1),
        ColorNote(beat=4.0, x=3, y=0, color=1, direction=0),
    ])
    tokens = _tok().encode_beatmap(bm)
    # Second event Δt token should be DT_BASE + 0 (Δt=0, bin 0)
    second_dt_pos = 1 + EVENT_LENGTHS[NOTE] + 1  # after BOS, after first event, skip HAND
    assert tokens[second_dt_pos] == DT_BASE  # bin 0 = Δt 0.0


# ---------------------------------------------------------------------------
# Round-trip: color notes
# ---------------------------------------------------------------------------


def test_round_trip_single_red_note() -> None:
    bm = _bm(color_notes=[ColorNote(beat=2.0, x=1, y=0, color=0, direction=1, angle_offset=0)])
    out = _round_trip(bm)
    assert len(out.color_notes) == 1
    n = out.color_notes[0]
    assert n.color == 0
    assert n.x == 1
    assert n.y == 0
    assert n.direction == 1
    assert _beats_close(n.beat, 2.0)


def test_round_trip_single_blue_note() -> None:
    bm = _bm(color_notes=[ColorNote(beat=3.5, x=2, y=2, color=1, direction=5, angle_offset=15)])
    out = _round_trip(bm)
    assert len(out.color_notes) == 1
    n = out.color_notes[0]
    assert n.color == 1
    assert n.x == 2
    assert n.y == 2
    assert n.direction == 5
    assert _beats_close(n.beat, 3.5)


def test_round_trip_all_directions() -> None:
    notes = [ColorNote(beat=float(i), x=0, y=0, color=0, direction=i) for i in range(9)]
    bm = _bm(color_notes=notes)
    out = _round_trip(bm)
    assert len(out.color_notes) == 9
    dirs = {n.direction for n in out.color_notes}
    assert dirs == set(range(9))


def test_round_trip_angle_offsets() -> None:
    angles = [-45, -30, -15, 0, 15, 30, 45]
    notes = [ColorNote(beat=float(i), x=1, y=0, color=0, direction=1, angle_offset=a)
             for i, a in enumerate(angles)]
    bm = _bm(color_notes=notes)
    out = _round_trip(bm)
    assert len(out.color_notes) == len(angles)
    out_angles = sorted(n.angle_offset for n in out.color_notes)
    assert out_angles == sorted(angles)


def test_round_trip_chord() -> None:
    bm = _bm(color_notes=[
        ColorNote(beat=4.0, x=1, y=0, color=0, direction=1),
        ColorNote(beat=4.0, x=2, y=0, color=1, direction=0),
    ])
    out = _round_trip(bm)
    assert len(out.color_notes) == 2
    reds = [n for n in out.color_notes if n.color == 0]
    blues = [n for n in out.color_notes if n.color == 1]
    assert len(reds) == 1 and len(blues) == 1
    assert _beats_close(reds[0].beat, 4.0)
    assert _beats_close(blues[0].beat, 4.0)


def test_round_trip_many_notes_preserves_count() -> None:
    notes = [
        ColorNote(beat=float(i) * 0.5, x=i % 4, y=i % 3, color=i % 2, direction=i % 9)
        for i in range(32)
    ]
    bm = _bm(color_notes=notes)
    out = _round_trip(bm)
    assert len(out.color_notes) == 32


def test_round_trip_preserves_beat_ordering() -> None:
    bm = _bm(color_notes=[
        ColorNote(beat=8.0, x=0, y=0, color=0, direction=1),
        ColorNote(beat=4.0, x=1, y=0, color=1, direction=0),
        ColorNote(beat=2.0, x=2, y=0, color=0, direction=3),
    ])
    out = _round_trip(bm)
    beats = [n.beat for n in out.color_notes]
    assert beats == sorted(beats)


# ---------------------------------------------------------------------------
# Round-trip: bombs
# ---------------------------------------------------------------------------


def test_round_trip_bomb() -> None:
    bm = _bm(bomb_notes=[BombNote(beat=3.0, x=2, y=1)])
    out = _round_trip(bm)
    assert len(out.bomb_notes) == 1
    b = out.bomb_notes[0]
    assert b.x == 2 and b.y == 1
    assert _beats_close(b.beat, 3.0)


def test_round_trip_notes_and_bombs_coexist() -> None:
    bm = _bm(
        color_notes=[ColorNote(beat=2.0, x=0, y=0, color=0, direction=1)],
        bomb_notes=[BombNote(beat=2.5, x=1, y=1)],
    )
    out = _round_trip(bm)
    assert len(out.color_notes) == 1
    assert len(out.bomb_notes) == 1


# ---------------------------------------------------------------------------
# Round-trip: arcs/sliders
# ---------------------------------------------------------------------------


def test_round_trip_arc() -> None:
    s = Slider(color=0, beat=2.0, x=1, y=0, direction=0, mu=1.0,
               tail_beat=4.0, tail_x=1, tail_y=2, tail_direction=1, tail_mu=1.0)
    bm = _bm(sliders=[s])
    out = _round_trip(bm)
    assert len(out.sliders) == 1
    r = out.sliders[0]
    assert r.color == 0
    assert r.x == 1 and r.y == 0
    assert r.tail_x == 1 and r.tail_y == 2
    assert _beats_close(r.beat, 2.0)
    assert _beats_close(r.tail_beat, 4.0)


def test_round_trip_arc_mu_bins() -> None:
    mus = [0.0, 0.5, 1.0, 1.5, 2.0]
    sliders = [
        Slider(color=0, beat=float(i), x=0, y=0, direction=0, mu=m,
               tail_beat=float(i) + 1.0, tail_x=0, tail_y=2, tail_direction=1, tail_mu=m)
        for i, m in enumerate(mus)
    ]
    bm = _bm(sliders=sliders)
    out = _round_trip(bm)
    assert len(out.sliders) == len(mus)


def test_round_trip_two_arcs_same_color() -> None:
    sliders = [
        Slider(color=0, beat=1.0, x=0, y=0, direction=0, mu=1.0,
               tail_beat=2.0, tail_x=0, tail_y=2, tail_direction=1, tail_mu=1.0),
        Slider(color=0, beat=3.0, x=1, y=0, direction=5, mu=0.5,
               tail_beat=4.0, tail_x=2, tail_y=1, tail_direction=6, tail_mu=0.5),
    ]
    bm = _bm(sliders=sliders)
    out = _round_trip(bm)
    assert len(out.sliders) == 2


def test_round_trip_mixed_color_arcs() -> None:
    sliders = [
        Slider(color=0, beat=1.0, x=1, y=0, direction=0, mu=1.0,
               tail_beat=3.0, tail_x=1, tail_y=2, tail_direction=1, tail_mu=1.0),
        Slider(color=1, beat=2.0, x=2, y=0, direction=0, mu=0.5,
               tail_beat=4.0, tail_x=2, tail_y=2, tail_direction=1, tail_mu=0.5),
    ]
    bm = _bm(sliders=sliders)
    out = _round_trip(bm)
    assert len(out.sliders) == 2
    assert {s.color for s in out.sliders} == {0, 1}


# ---------------------------------------------------------------------------
# Round-trip: chains/burst sliders
# ---------------------------------------------------------------------------


def test_round_trip_chain() -> None:
    bs = BurstSlider(color=1, beat=4.0, x=2, y=1, direction=3,
                     tail_beat=4.5, tail_x=3, tail_y=1, slice_count=4, squish=0.5)
    bm = _bm(burst_sliders=[bs])
    out = _round_trip(bm)
    assert len(out.burst_sliders) == 1
    r = out.burst_sliders[0]
    assert r.color == 1
    assert r.x == 2 and r.y == 1
    assert r.tail_x == 3 and r.tail_y == 1
    assert _beats_close(r.beat, 4.0)
    assert _beats_close(r.tail_beat, 4.5)
    assert r.slice_count == 4


def test_round_trip_chain_slice_range() -> None:
    slices = [2, 5, 10, 20, 32]
    chains = [
        BurstSlider(color=0, beat=float(i), x=0, y=0, direction=0,
                    tail_beat=float(i) + 0.5, tail_x=1, tail_y=0,
                    slice_count=sc, squish=0.5)
        for i, sc in enumerate(slices)
    ]
    bm = _bm(burst_sliders=chains)
    out = _round_trip(bm)
    assert len(out.burst_sliders) == len(slices)


# ---------------------------------------------------------------------------
# Round-trip: obstacles excluded
# ---------------------------------------------------------------------------


def test_obstacles_excluded_from_stream() -> None:
    from beatsaber_automapper.data.beatmap import Obstacle
    bm = _bm(
        color_notes=[ColorNote(beat=1.0, x=0, y=0, color=0, direction=1)],
        obstacles=[Obstacle(beat=0.0, duration=2.0, x=0, y=0, width=1, height=5)],
    )
    tokens = _tok().encode_beatmap(bm)
    out = _tok().decode_beatmap(tokens)
    assert len(out.obstacles) == 0   # walls not encoded
    assert len(out.color_notes) == 1


# ---------------------------------------------------------------------------
# Ordering: canonical sort at same beat
# ---------------------------------------------------------------------------


def test_ordering_note_before_bomb_same_beat() -> None:
    bm = _bm(
        color_notes=[ColorNote(beat=2.0, x=0, y=0, color=0, direction=1)],
        bomb_notes=[BombNote(beat=2.0, x=1, y=1)],
    )
    tokens = _tok().encode_beatmap(bm)
    # After BOS, first KIND token should be NOTE
    assert tokens[3] == NOTE  # BOS HAND DT [KIND]


def test_ordering_left_before_right_same_beat() -> None:
    bm = _bm(color_notes=[
        ColorNote(beat=1.0, x=0, y=0, color=1, direction=0),  # blue (RIGHT) first in list
        ColorNote(beat=1.0, x=3, y=0, color=0, direction=1),  # red (LEFT) second in list
    ])
    tokens = _tok().encode_beatmap(bm)
    # HAND of first event should be LEFT despite list order
    assert tokens[1] == HAND_LEFT


def test_ordering_arc_head_before_tail() -> None:
    s = Slider(color=0, beat=2.0, x=1, y=0, direction=0, mu=1.0,
               tail_beat=4.0, tail_x=1, tail_y=2, tail_direction=1, tail_mu=1.0)
    tokens = _tok().encode_beatmap(_bm(sliders=[s]))
    events = _tok().decode_events(tokens)
    kinds = [e.kind for e in events]
    assert kinds.index(ARC_HEAD) < kinds.index(ARC_TAIL)


# ---------------------------------------------------------------------------
# Arc/chain self-connect: orphan handling
# ---------------------------------------------------------------------------


def test_orphan_arc_tail_dropped() -> None:
    # Emit an ARC_TAIL with no preceding ARC_HEAD — should be dropped
    bm = _bm(sliders=[
        Slider(color=0, beat=2.0, x=0, y=0, direction=0, mu=1.0,
               tail_beat=1.0,  # tail beat BEFORE head beat — results in tail before head
               tail_x=1, tail_y=1, tail_direction=1, tail_mu=1.0)
    ])
    out = _round_trip(bm)
    # With tail_beat < beat, tail comes first in the stream (beat 1.0 < 2.0)
    # ARC_TAIL appears before ARC_HEAD → no match → slider dropped
    assert len(out.sliders) == 0


def test_two_arcs_same_hand_match_fifo() -> None:
    sliders = [
        Slider(color=0, beat=1.0, x=0, y=0, direction=0, mu=1.0,
               tail_beat=3.0, tail_x=0, tail_y=1, tail_direction=1, tail_mu=0.5),
        Slider(color=0, beat=5.0, x=1, y=0, direction=5, mu=0.75,
               tail_beat=7.0, tail_x=2, tail_y=2, tail_direction=6, tail_mu=1.0),
    ]
    bm = _bm(sliders=sliders)
    out = _round_trip(bm)
    assert len(out.sliders) == 2
    # Both arcs should be matched correctly (FIFO by hand)
    beats = sorted(s.beat for s in out.sliders)
    assert _beats_close(beats[0], 1.0)
    assert _beats_close(beats[1], 5.0)


# ---------------------------------------------------------------------------
# decode_events: raw event extraction
# ---------------------------------------------------------------------------


def test_decode_events_count() -> None:
    bm = _bm(
        color_notes=[ColorNote(beat=1.0, x=0, y=0, color=0, direction=1)],
        bomb_notes=[BombNote(beat=2.0, x=1, y=1)],
    )
    tokens = _tok().encode_beatmap(bm)
    events = _tok().decode_events(tokens)
    assert len(events) == 2


def test_decode_events_beat_accumulation() -> None:
    bm = _bm(color_notes=[
        ColorNote(beat=0.5, x=0, y=0, color=0, direction=1),
        ColorNote(beat=1.0, x=1, y=0, color=0, direction=0),
    ])
    tokens = _tok().encode_beatmap(bm)
    events = _tok().decode_events(tokens)
    assert len(events) == 2
    assert _beats_close(events[0].beat, 0.5)
    assert _beats_close(events[1].beat, 1.0)


# ---------------------------------------------------------------------------
# Saber state: basic properties
# ---------------------------------------------------------------------------


def test_saber_state_shape() -> None:
    bm = _bm(color_notes=[
        ColorNote(beat=1.0, x=1, y=0, color=0, direction=1),
        ColorNote(beat=2.0, x=2, y=0, color=1, direction=0),
    ])
    events, states = compute_saber_states_from_beatmap(bm)
    assert states.shape == (len(events), 12)


def test_saber_state_initial_is_finite() -> None:
    bm = _bm(color_notes=[ColorNote(beat=1.0, x=0, y=0, color=0, direction=1)])
    events, states = compute_saber_states_from_beatmap(bm)
    assert torch.isfinite(states).all()


def test_saber_state_depends_only_on_prior_events() -> None:
    """State at position i should be the same whether or not future events exist."""
    bm2 = _bm(color_notes=[
        ColorNote(beat=1.0, x=1, y=0, color=0, direction=1),
        ColorNote(beat=2.0, x=2, y=0, color=1, direction=0),
    ])
    bm1 = _bm(color_notes=[
        ColorNote(beat=1.0, x=1, y=0, color=0, direction=1),
    ])
    _, states2 = compute_saber_states_from_beatmap(bm2)
    _, states1 = compute_saber_states_from_beatmap(bm1)
    # State at event 0 should be the same regardless of subsequent events
    assert torch.allclose(states2[0], states1[0])


def test_saber_state_left_right_independent() -> None:
    """Right-hand events don't affect left-hand state columns (0-5) and vice versa."""
    bm = _bm(color_notes=[
        ColorNote(beat=1.0, x=0, y=0, color=1, direction=0),  # RIGHT only
        ColorNote(beat=2.0, x=3, y=2, color=1, direction=1),  # RIGHT only
        ColorNote(beat=3.0, x=0, y=0, color=0, direction=5),  # LEFT
    ])
    events, states = compute_saber_states_from_beatmap(bm)
    # Before event 0 (no prior events), left state should be neutral (dt col = 1.0 = max_log)
    # Before event 2 (two right events, no left events yet), left dt should still be at max
    left_dt_col = 4
    assert states[2, left_dt_col].item() == pytest.approx(1.0, abs=0.01)


def test_saber_state_position_updates_after_swing() -> None:
    bm = _bm(color_notes=[
        ColorNote(beat=1.0, x=0, y=0, color=0, direction=1),
        ColorNote(beat=2.0, x=2, y=2, color=0, direction=0),
    ])
    events, states = compute_saber_states_from_beatmap(bm)
    # Before event 1, left hand should be at x=0,y=0 (from event 0)
    assert states[1, 0].item() == pytest.approx(0.0 / 3.0, abs=0.01)  # L_x = 0/3
    assert states[1, 1].item() == pytest.approx(0.0 / 2.0, abs=0.01)  # L_y = 0/2


def test_saber_state_parity_reset_on_long_gap() -> None:
    gap = PARITY_RESET_BEATS + 1.0
    bm = _bm(color_notes=[
        ColorNote(beat=1.0, x=0, y=0, color=0, direction=1),  # forehand (down), parity=-1
        ColorNote(beat=1.0 + gap, x=1, y=0, color=0, direction=0),  # after big gap
    ])
    events, states = compute_saber_states_from_beatmap(bm)
    # Before event 1: parity should be reset to 0 due to gap > PARITY_RESET_BEATS
    left_parity_col = 5
    assert states[1, left_parity_col].item() == pytest.approx(0.0, abs=0.01)


def test_saber_state_parity_no_reset_short_gap() -> None:
    bm = _bm(color_notes=[
        ColorNote(beat=1.0, x=0, y=0, color=0, direction=1),  # forehand (down)
        ColorNote(beat=2.0, x=1, y=0, color=0, direction=0),  # 1 beat gap
    ])
    events, states = compute_saber_states_from_beatmap(bm)
    left_parity_col = 5
    # Before event 1: parity from event 0 (forehand=down => -1) should be preserved
    assert states[1, left_parity_col].item() == pytest.approx(-1.0, abs=0.01)


def test_saber_state_neutral_direction_does_not_update_parity() -> None:
    bm = _bm(color_notes=[
        ColorNote(beat=1.0, x=0, y=0, color=0, direction=1),  # forehand, parity = -1
        ColorNote(beat=2.0, x=1, y=0, color=0, direction=2),  # left (neutral)
        ColorNote(beat=3.0, x=2, y=0, color=0, direction=0),  # another note
    ])
    events, states = compute_saber_states_from_beatmap(bm)
    left_parity_col = 5
    # Before event 2 (after the neutral swing): parity should still be -1
    assert states[2, left_parity_col].item() == pytest.approx(-1.0, abs=0.01)


def test_saber_state_bombs_do_not_update_saber_state() -> None:
    bm = _bm(
        color_notes=[ColorNote(beat=1.0, x=1, y=0, color=0, direction=1)],
        bomb_notes=[BombNote(beat=2.0, x=3, y=2)],
    )
    events, states = compute_saber_states_from_beatmap(bm)
    bomb_idx = next(i for i, e in enumerate(events) if e.kind == BOMB)
    note_idx = next(i for i, e in enumerate(events) if e.kind == NOTE)
    assert bomb_idx > note_idx  # bomb is second (beat 2.0 > beat 1.0)
    # State before bomb should reflect note at beat 1.0, not the bomb
    # Left hand position should be x=1,y=0 (from note)
    assert states[bomb_idx, 0].item() == pytest.approx(1.0 / 3.0, abs=0.01)


def test_saber_state_dt_normalized_range() -> None:
    bm = _bm(color_notes=[
        ColorNote(beat=0.0, x=0, y=0, color=0, direction=1),
        ColorNote(beat=100.0, x=1, y=0, color=0, direction=0),
    ])
    events, states = compute_saber_states_from_beatmap(bm)
    # All dt values should be in [0, 1]
    assert (states[:, 4] >= 0).all()
    assert (states[:, 4] <= 1).all()
    assert (states[:, 10] >= 0).all()
    assert (states[:, 10] <= 1).all()


# ---------------------------------------------------------------------------
# compute_saber_states directly
# ---------------------------------------------------------------------------


def test_compute_saber_states_empty() -> None:
    states = compute_saber_states([])
    assert states.shape == (0, 12)


def test_compute_saber_states_dtype() -> None:
    bm = _bm(color_notes=[ColorNote(beat=1.0, x=0, y=0, color=0, direction=1)])
    events, states = compute_saber_states_from_beatmap(bm)
    assert states.dtype == torch.float32
