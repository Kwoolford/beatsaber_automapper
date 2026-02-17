"""Tests for the Beat Saber v3 beatmap parser."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from beatsaber_automapper.data.beatmap import (
    BasicEvent,
    BeatmapInfo,
    BombNote,
    BurstSlider,
    ColorBoostEvent,
    ColorNote,
    DifficultyBeatmap,
    Obstacle,
    Slider,
    parse_difficulty_dat,
    parse_difficulty_dat_json,
    parse_info_dat,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_INFO_DAT = {
    "_version": "2.1.0",
    "_songName": "Test Song",
    "_songSubName": "Remix",
    "_songAuthorName": "Test Artist",
    "_levelAuthorName": "TestMapper",
    "_beatsPerMinute": 128,
    "_shuffle": 0,
    "_shufflePeriod": 0.5,
    "_previewStartTime": 12,
    "_previewDuration": 10,
    "_songFilename": "song.ogg",
    "_coverImageFilename": "cover.png",
    "_environmentName": "BigMirrorEnvironment",
    "_songTimeOffset": 0,
    "_difficultyBeatmapSets": [
        {
            "_beatmapCharacteristicName": "Standard",
            "_difficultyBeatmaps": [
                {
                    "_difficulty": "Expert",
                    "_difficultyRank": 7,
                    "_beatmapFilename": "ExpertStandard.dat",
                    "_noteJumpMovementSpeed": 16,
                    "_noteJumpStartBeatOffset": 0,
                },
                {
                    "_difficulty": "ExpertPlus",
                    "_difficultyRank": 9,
                    "_beatmapFilename": "ExpertPlusStandard.dat",
                    "_noteJumpMovementSpeed": 18,
                    "_noteJumpStartBeatOffset": 0.5,
                },
            ],
        }
    ],
}


SAMPLE_V3_DAT = {
    "version": "3.3.0",
    "colorNotes": [
        {"b": 10.0, "x": 1, "y": 0, "c": 0, "d": 1, "a": 0},
        {"b": 10.0, "x": 2, "y": 1, "c": 1, "d": 0, "a": 15},
        {"b": 12.5, "x": 3, "y": 2, "c": 0, "d": 3},
    ],
    "bombNotes": [
        {"b": 11.0, "x": 0, "y": 1},
    ],
    "obstacles": [
        {"b": 14.0, "d": 2.0, "x": 0, "y": 2, "w": 1, "h": 3},
    ],
    "sliders": [
        {
            "c": 0,
            "b": 15.0,
            "x": 1,
            "y": 0,
            "d": 1,
            "mu": 1.5,
            "tb": 17.0,
            "tx": 2,
            "ty": 2,
            "tc": 0,
            "tmu": 0.8,
            "m": 1,
        },
    ],
    "burstSliders": [
        {
            "c": 1,
            "b": 20.0,
            "x": 3,
            "y": 1,
            "d": 2,
            "tb": 21.0,
            "tx": 1,
            "ty": 1,
            "sc": 5,
            "s": 0.7,
        },
    ],
    "basicBeatmapEvents": [
        {"b": 10.0, "et": 1, "i": 3, "f": 0.8},
    ],
    "colorBoostBeatmapEvents": [
        {"b": 10.0, "o": True},
    ],
}


SAMPLE_V2_DAT = {
    "_version": "2.6.0",
    "_notes": [{"_time": 10.0, "_lineIndex": 1, "_lineLayer": 0, "_type": 0, "_cutDirection": 1}],
}


# ---------------------------------------------------------------------------
# Info.dat tests
# ---------------------------------------------------------------------------


def test_parse_info_dat_from_file() -> None:
    """Parse Info.dat from a file on disk."""
    with tempfile.TemporaryDirectory() as tmpdir:
        info_path = Path(tmpdir) / "Info.dat"
        info_path.write_text(json.dumps(SAMPLE_INFO_DAT), encoding="utf-8")

        info = parse_info_dat(info_path)

    assert info is not None
    assert isinstance(info, BeatmapInfo)
    assert info.song_name == "Test Song"
    assert info.song_sub_name == "Remix"
    assert info.song_author == "Test Artist"
    assert info.level_author == "TestMapper"
    assert info.bpm == 128.0
    assert info.song_filename == "song.ogg"
    assert info.cover_filename == "cover.png"
    assert info.environment_name == "BigMirrorEnvironment"
    assert info.preview_start == 12.0
    assert info.preview_duration == 10.0

    assert len(info.difficulties) == 2
    expert = info.difficulties[0]
    assert expert.difficulty == "Expert"
    assert expert.difficulty_rank == 7
    assert expert.filename == "ExpertStandard.dat"
    assert expert.note_jump_speed == 16.0

    exp_plus = info.difficulties[1]
    assert exp_plus.difficulty == "ExpertPlus"
    assert exp_plus.note_jump_speed == 18.0
    assert exp_plus.note_jump_offset == 0.5


# ---------------------------------------------------------------------------
# Difficulty .dat v3 tests
# ---------------------------------------------------------------------------


def test_parse_v3_color_notes() -> None:
    """Parse colorNotes from v3 dat."""
    result = parse_difficulty_dat_json(SAMPLE_V3_DAT)
    assert result is not None
    assert isinstance(result, DifficultyBeatmap)
    assert result.version == "3.3.0"
    assert len(result.color_notes) == 3

    n = result.color_notes[0]
    assert isinstance(n, ColorNote)
    assert n.beat == 10.0
    assert n.x == 1
    assert n.y == 0
    assert n.color == 0
    assert n.direction == 1
    assert n.angle_offset == 0

    # Second note has angle offset
    assert result.color_notes[1].angle_offset == 15

    # Third note missing 'a' field — defaults to 0
    assert result.color_notes[2].angle_offset == 0


def test_parse_v3_bomb_notes() -> None:
    result = parse_difficulty_dat_json(SAMPLE_V3_DAT)
    assert result is not None
    assert len(result.bomb_notes) == 1
    b = result.bomb_notes[0]
    assert isinstance(b, BombNote)
    assert b.beat == 11.0
    assert b.x == 0
    assert b.y == 1


def test_parse_v3_obstacles() -> None:
    result = parse_difficulty_dat_json(SAMPLE_V3_DAT)
    assert result is not None
    assert len(result.obstacles) == 1
    o = result.obstacles[0]
    assert isinstance(o, Obstacle)
    assert o.beat == 14.0
    assert o.duration == 2.0
    assert o.x == 0
    assert o.y == 2
    assert o.width == 1
    assert o.height == 3


def test_parse_v3_sliders() -> None:
    result = parse_difficulty_dat_json(SAMPLE_V3_DAT)
    assert result is not None
    assert len(result.sliders) == 1
    s = result.sliders[0]
    assert isinstance(s, Slider)
    assert s.color == 0
    assert s.beat == 15.0
    assert s.x == 1
    assert s.y == 0
    assert s.direction == 1
    assert s.mu == 1.5
    assert s.tail_beat == 17.0
    assert s.tail_x == 2
    assert s.tail_y == 2
    assert s.tail_direction == 0
    assert s.tail_mu == 0.8
    assert s.mid_anchor_mode == 1


def test_parse_v3_burst_sliders() -> None:
    result = parse_difficulty_dat_json(SAMPLE_V3_DAT)
    assert result is not None
    assert len(result.burst_sliders) == 1
    bs = result.burst_sliders[0]
    assert isinstance(bs, BurstSlider)
    assert bs.color == 1
    assert bs.beat == 20.0
    assert bs.x == 3
    assert bs.y == 1
    assert bs.direction == 2
    assert bs.tail_beat == 21.0
    assert bs.tail_x == 1
    assert bs.tail_y == 1
    assert bs.slice_count == 5
    assert bs.squish == 0.7


def test_parse_v3_basic_events() -> None:
    result = parse_difficulty_dat_json(SAMPLE_V3_DAT)
    assert result is not None
    assert len(result.basic_events) == 1
    e = result.basic_events[0]
    assert isinstance(e, BasicEvent)
    assert e.beat == 10.0
    assert e.event_type == 1
    assert e.value == 3
    assert e.float_value == 0.8


def test_parse_v3_color_boost_events() -> None:
    result = parse_difficulty_dat_json(SAMPLE_V3_DAT)
    assert result is not None
    assert len(result.color_boost_events) == 1
    cb = result.color_boost_events[0]
    assert isinstance(cb, ColorBoostEvent)
    assert cb.beat == 10.0
    assert cb.boost is True


def test_parse_v3_empty_collections() -> None:
    """Dat with no objects should produce empty lists."""
    data = {"version": "3.3.0"}
    result = parse_difficulty_dat_json(data)
    assert result is not None
    assert result.color_notes == []
    assert result.bomb_notes == []
    assert result.obstacles == []
    assert result.sliders == []
    assert result.burst_sliders == []
    assert result.basic_events == []
    assert result.color_boost_events == []


# ---------------------------------------------------------------------------
# v2 detection
# ---------------------------------------------------------------------------


def test_v2_returns_none() -> None:
    """v2 format beatmaps should return None."""
    result = parse_difficulty_dat_json(SAMPLE_V2_DAT)
    assert result is None


def test_no_version_returns_none() -> None:
    """Data with no version field should return None."""
    result = parse_difficulty_dat_json({"colorNotes": []})
    assert result is None


# ---------------------------------------------------------------------------
# File-based parsing
# ---------------------------------------------------------------------------


def test_parse_difficulty_dat_from_file() -> None:
    """Parse difficulty dat from a file on disk."""
    with tempfile.TemporaryDirectory() as tmpdir:
        dat_path = Path(tmpdir) / "ExpertStandard.dat"
        dat_path.write_text(json.dumps(SAMPLE_V3_DAT), encoding="utf-8")

        result = parse_difficulty_dat(dat_path)

    assert result is not None
    assert len(result.color_notes) == 3
    assert len(result.sliders) == 1


def test_default_values() -> None:
    """Fields with defaults should be populated when missing from JSON."""
    data = {
        "version": "3.3.0",
        "colorNotes": [{"b": 1.0, "x": 0, "y": 0, "c": 0, "d": 0}],
        "sliders": [
            {
                "c": 0,
                "b": 2.0,
                "x": 0,
                "y": 0,
                "d": 0,
                "mu": 1.0,
                "tb": 3.0,
                "tx": 0,
                "ty": 0,
                "tc": 0,
                "tmu": 1.0,
            }
        ],
        "basicBeatmapEvents": [{"b": 1.0, "et": 0, "i": 0}],
    }
    result = parse_difficulty_dat_json(data)
    assert result is not None
    assert result.color_notes[0].angle_offset == 0
    assert result.sliders[0].mid_anchor_mode == 0
    assert result.basic_events[0].float_value == 1.0
