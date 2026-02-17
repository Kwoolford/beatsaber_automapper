"""Tests for the Beat Saber note event tokenizer."""

from __future__ import annotations

from beatsaber_automapper.data.beatmap import (
    BombNote,
    BurstSlider,
    ColorNote,
    DifficultyBeatmap,
    Obstacle,
    Slider,
)
from beatsaber_automapper.data.tokenizer import (
    ARC_END,
    ARC_START,
    BOMB,
    BOS,
    CHAIN,
    EOS,
    NOTE,
    PAD,
    SEP,
    VOCAB_SIZE,
    WALL,
    BeatmapTokenizer,
)


def _make_beatmap(**kwargs) -> DifficultyBeatmap:
    """Helper to create a DifficultyBeatmap with specified fields."""
    defaults = {
        "version": "3.3.0",
        "color_notes": [],
        "bomb_notes": [],
        "obstacles": [],
        "sliders": [],
        "burst_sliders": [],
        "basic_events": [],
        "color_boost_events": [],
    }
    defaults.update(kwargs)
    return DifficultyBeatmap(**defaults)


# ---------------------------------------------------------------------------
# Basic setup
# ---------------------------------------------------------------------------


def test_tokenizer_init() -> None:
    tok = BeatmapTokenizer()
    assert tok.vocab_size == VOCAB_SIZE
    assert tok.pad_token == PAD
    assert tok.eos_token == EOS
    assert tok.sep_token == SEP
    assert tok.bos_token == BOS


def test_special_and_event_tokens() -> None:
    tok = BeatmapTokenizer()
    assert tok.special_tokens == {"PAD": 0, "EOS": 1, "SEP": 2, "BOS": 3}
    assert "NOTE" in tok.event_tokens
    assert "CHAIN" in tok.event_tokens


def test_vocab_size_reasonable() -> None:
    """Vocab should be ~167 tokens per plan."""
    assert 150 <= VOCAB_SIZE <= 200


# ---------------------------------------------------------------------------
# Round-trip per event type
# ---------------------------------------------------------------------------


def test_round_trip_color_note() -> None:
    tok = BeatmapTokenizer()
    bm = _make_beatmap(
        color_notes=[ColorNote(beat=10.0, x=1, y=0, color=0, direction=1, angle_offset=0)]
    )
    encoded = tok.encode_beatmap(bm)
    assert 10.0 in encoded
    tokens = encoded[10.0]
    assert tokens[0] == NOTE
    assert tokens[-1] == EOS

    decoded = tok.decode_beatmap(encoded)
    assert len(decoded.color_notes) == 1
    n = decoded.color_notes[0]
    assert n.beat == 10.0
    assert n.x == 1
    assert n.y == 0
    assert n.color == 0
    assert n.direction == 1
    assert n.angle_offset == 0


def test_round_trip_color_note_with_angle() -> None:
    tok = BeatmapTokenizer()
    bm = _make_beatmap(
        color_notes=[ColorNote(beat=5.0, x=2, y=1, color=1, direction=3, angle_offset=30)]
    )
    encoded = tok.encode_beatmap(bm)
    decoded = tok.decode_beatmap(encoded)
    assert decoded.color_notes[0].angle_offset == 30


def test_round_trip_bomb() -> None:
    tok = BeatmapTokenizer()
    bm = _make_beatmap(bomb_notes=[BombNote(beat=11.0, x=0, y=1)])
    encoded = tok.encode_beatmap(bm)
    tokens = encoded[11.0]
    assert tokens[0] == BOMB
    assert tokens[-1] == EOS

    decoded = tok.decode_beatmap(encoded)
    assert len(decoded.bomb_notes) == 1
    b = decoded.bomb_notes[0]
    assert b.beat == 11.0
    assert b.x == 0
    assert b.y == 1


def test_round_trip_obstacle() -> None:
    tok = BeatmapTokenizer()
    bm = _make_beatmap(obstacles=[Obstacle(beat=14.0, duration=2.5, x=0, y=2, width=2, height=3)])
    encoded = tok.encode_beatmap(bm)
    tokens = encoded[14.0]
    assert tokens[0] == WALL

    decoded = tok.decode_beatmap(encoded)
    assert len(decoded.obstacles) == 1
    o = decoded.obstacles[0]
    assert o.beat == 14.0
    assert o.x == 0
    assert o.y == 2
    assert o.width == 2
    assert o.height == 3
    assert o.duration == 2.5


def test_round_trip_slider() -> None:
    tok = BeatmapTokenizer()
    bm = _make_beatmap(
        sliders=[
            Slider(
                color=0,
                beat=15.0,
                x=1,
                y=0,
                direction=1,
                mu=1.0,
                tail_beat=17.0,
                tail_x=2,
                tail_y=2,
                tail_direction=0,
                tail_mu=1.0,
                mid_anchor_mode=0,
            )
        ]
    )
    encoded = tok.encode_beatmap(bm)
    # Should have ARC_START at beat 15.0 and ARC_END at beat 17.0
    assert 15.0 in encoded
    assert 17.0 in encoded
    assert encoded[15.0][0] == ARC_START
    assert encoded[17.0][0] == ARC_END

    decoded = tok.decode_beatmap(encoded)
    assert len(decoded.sliders) == 1
    s = decoded.sliders[0]
    assert s.color == 0
    assert s.beat == 15.0
    assert s.x == 1
    assert s.y == 0
    assert s.direction == 1
    assert s.mu == 1.0
    assert s.tail_beat == 17.0
    assert s.tail_x == 2
    assert s.tail_y == 2
    assert s.tail_direction == 0
    assert s.tail_mu == 1.0
    assert s.mid_anchor_mode == 0


def test_round_trip_burst_slider() -> None:
    tok = BeatmapTokenizer()
    bm = _make_beatmap(
        burst_sliders=[
            BurstSlider(
                color=1,
                beat=20.0,
                x=3,
                y=1,
                direction=2,
                tail_beat=21.0,
                tail_x=1,
                tail_y=1,
                slice_count=5,
                squish=0.5,
            )
        ]
    )
    encoded = tok.encode_beatmap(bm)
    tokens = encoded[20.0]
    assert tokens[0] == CHAIN

    decoded = tok.decode_beatmap(encoded)
    assert len(decoded.burst_sliders) == 1
    bs = decoded.burst_sliders[0]
    assert bs.color == 1
    assert bs.beat == 20.0
    assert bs.x == 3
    assert bs.y == 1
    assert bs.direction == 2
    assert bs.tail_x == 1
    assert bs.tail_y == 1
    assert bs.slice_count == 5
    assert bs.squish == 0.5


# ---------------------------------------------------------------------------
# Multiple events at same timestamp
# ---------------------------------------------------------------------------


def test_sep_between_events_at_same_beat() -> None:
    tok = BeatmapTokenizer()
    bm = _make_beatmap(
        color_notes=[
            ColorNote(beat=10.0, x=1, y=0, color=0, direction=1, angle_offset=0),
            ColorNote(beat=10.0, x=2, y=1, color=1, direction=0, angle_offset=0),
        ]
    )
    encoded = tok.encode_beatmap(bm)
    tokens = encoded[10.0]
    assert SEP in tokens
    assert tokens[-1] == EOS
    # Two NOTE events with one SEP and one EOS: 6 + 1 + 6 + 1 = 14
    assert len(tokens) == 14


def test_canonical_ordering() -> None:
    """Events should be ordered: type priority, then x, then y."""
    tok = BeatmapTokenizer()
    bm = _make_beatmap(
        color_notes=[
            # Higher x first (should appear second)
            ColorNote(beat=10.0, x=3, y=0, color=1, direction=0, angle_offset=0),
            # Lower x first (should appear first)
            ColorNote(beat=10.0, x=1, y=0, color=0, direction=1, angle_offset=0),
        ],
        bomb_notes=[
            # Bomb has lower priority than note, so appears after notes
            BombNote(beat=10.0, x=2, y=1),
        ],
    )
    encoded = tok.encode_beatmap(bm)
    tokens = encoded[10.0]

    # Find event type tokens
    event_types = [t for t in tokens if t in (NOTE, BOMB, WALL, ARC_START, ARC_END, CHAIN)]
    assert event_types == [NOTE, NOTE, BOMB]


# ---------------------------------------------------------------------------
# Mixed timestamps
# ---------------------------------------------------------------------------


def test_multiple_timestamps() -> None:
    tok = BeatmapTokenizer()
    bm = _make_beatmap(
        color_notes=[
            ColorNote(beat=10.0, x=1, y=0, color=0, direction=1, angle_offset=0),
            ColorNote(beat=20.0, x=2, y=1, color=1, direction=0, angle_offset=0),
        ],
        bomb_notes=[
            BombNote(beat=15.0, x=0, y=1),
        ],
    )
    encoded = tok.encode_beatmap(bm)
    assert len(encoded) == 3
    assert sorted(encoded.keys()) == [10.0, 15.0, 20.0]


def test_full_round_trip_mixed() -> None:
    """Encode and decode a beatmap with multiple event types."""
    tok = BeatmapTokenizer()
    bm = _make_beatmap(
        color_notes=[
            ColorNote(beat=10.0, x=1, y=0, color=0, direction=1, angle_offset=0),
            ColorNote(beat=10.0, x=2, y=1, color=1, direction=0, angle_offset=15),
        ],
        bomb_notes=[BombNote(beat=11.0, x=0, y=1)],
        obstacles=[Obstacle(beat=14.0, duration=2.0, x=0, y=2, width=1, height=3)],
        sliders=[
            Slider(
                color=0,
                beat=15.0,
                x=1,
                y=0,
                direction=1,
                mu=1.5,
                tail_beat=17.0,
                tail_x=2,
                tail_y=2,
                tail_direction=0,
                tail_mu=0.75,
                mid_anchor_mode=1,
            )
        ],
        burst_sliders=[
            BurstSlider(
                color=1,
                beat=20.0,
                x=3,
                y=1,
                direction=2,
                tail_beat=21.0,
                tail_x=1,
                tail_y=1,
                slice_count=5,
                squish=0.5,
            )
        ],
    )
    encoded = tok.encode_beatmap(bm)
    decoded = tok.decode_beatmap(encoded)

    assert len(decoded.color_notes) == 2
    assert len(decoded.bomb_notes) == 1
    assert len(decoded.obstacles) == 1
    assert len(decoded.sliders) == 1
    assert len(decoded.burst_sliders) == 1

    # Check quantized values match
    assert decoded.sliders[0].mu == 1.5
    assert decoded.sliders[0].tail_mu == 0.75
    assert decoded.sliders[0].mid_anchor_mode == 1


# ---------------------------------------------------------------------------
# Quantization edge cases
# ---------------------------------------------------------------------------


def test_angle_quantization_nearest() -> None:
    """Angle 20 should snap to 15."""
    tok = BeatmapTokenizer()
    bm = _make_beatmap(
        color_notes=[ColorNote(beat=1.0, x=0, y=0, color=0, direction=0, angle_offset=20)]
    )
    encoded = tok.encode_beatmap(bm)
    decoded = tok.decode_beatmap(encoded)
    assert decoded.color_notes[0].angle_offset == 15


def test_mu_quantization() -> None:
    """mu=1.3 should snap to 1.25."""
    tok = BeatmapTokenizer()
    bm = _make_beatmap(
        sliders=[
            Slider(
                color=0,
                beat=1.0,
                x=0,
                y=0,
                direction=0,
                mu=1.3,
                tail_beat=2.0,
                tail_x=1,
                tail_y=1,
                tail_direction=0,
                tail_mu=1.0,
                mid_anchor_mode=0,
            )
        ]
    )
    encoded = tok.encode_beatmap(bm)
    decoded = tok.decode_beatmap(encoded)
    assert decoded.sliders[0].mu == 1.25


def test_empty_beatmap() -> None:
    tok = BeatmapTokenizer()
    bm = _make_beatmap()
    encoded = tok.encode_beatmap(bm)
    assert encoded == {}


def test_all_tokens_in_vocab_range() -> None:
    """All generated tokens should be in [0, VOCAB_SIZE)."""
    tok = BeatmapTokenizer()
    bm = _make_beatmap(
        color_notes=[ColorNote(beat=1.0, x=3, y=2, color=1, direction=8, angle_offset=45)],
        bomb_notes=[BombNote(beat=2.0, x=3, y=2)],
        obstacles=[Obstacle(beat=3.0, duration=64.75, x=3, y=2, width=4, height=5)],
        sliders=[
            Slider(
                color=1,
                beat=4.0,
                x=3,
                y=2,
                direction=8,
                mu=2.0,
                tail_beat=5.0,
                tail_x=3,
                tail_y=2,
                tail_direction=8,
                tail_mu=2.0,
                mid_anchor_mode=2,
            )
        ],
        burst_sliders=[
            BurstSlider(
                color=1,
                beat=6.0,
                x=3,
                y=2,
                direction=8,
                tail_beat=7.0,
                tail_x=3,
                tail_y=2,
                slice_count=32,
                squish=1.0,
            )
        ],
    )
    encoded = tok.encode_beatmap(bm)
    for tokens in encoded.values():
        for t in tokens:
            assert 0 <= t < VOCAB_SIZE, f"Token {t} out of range [0, {VOCAB_SIZE})"
