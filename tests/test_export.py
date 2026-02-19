"""Tests for generation/export.py — v3 JSON export and .zip packaging."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

from beatsaber_automapper.data.beatmap import (
    BasicEvent,
    BombNote,
    BurstSlider,
    ColorBoostEvent,
    ColorNote,
    DifficultyBeatmap,
    Obstacle,
    Slider,
)
from beatsaber_automapper.generation.export import (
    beatmap_to_v3_dict,
    build_info_dat,
    package_level,
    tokens_to_beatmap,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def simple_beatmap() -> DifficultyBeatmap:
    """Minimal beatmap with one of each object type."""
    return DifficultyBeatmap(
        version="3.3.0",
        color_notes=[ColorNote(beat=1.0, x=1, y=0, color=0, direction=1, angle_offset=0)],
        bomb_notes=[BombNote(beat=2.0, x=2, y=1)],
        obstacles=[Obstacle(beat=3.0, duration=2.0, x=0, y=0, width=1, height=5)],
        sliders=[
            Slider(
                color=1,
                beat=4.0,
                x=1,
                y=1,
                direction=0,
                mu=1.0,
                tail_beat=5.0,
                tail_x=2,
                tail_y=2,
                tail_direction=0,
                tail_mu=1.0,
                mid_anchor_mode=0,
            )
        ],
        burst_sliders=[
            BurstSlider(
                color=0,
                beat=6.0,
                x=0,
                y=0,
                direction=1,
                tail_beat=7.0,
                tail_x=3,
                tail_y=0,
                slice_count=5,
                squish=0.5,
            )
        ],
        basic_events=[BasicEvent(beat=1.0, event_type=1, value=3, float_value=1.0)],
        color_boost_events=[ColorBoostEvent(beat=2.0, boost=True)],
    )


@pytest.fixture
def empty_beatmap() -> DifficultyBeatmap:
    return DifficultyBeatmap(version="3.3.0")


@pytest.fixture
def tmp_audio(tmp_path: Path) -> Path:
    """Create a tiny valid WAV file for packaging tests."""
    import wave

    wav_path = tmp_path / "song.wav"
    with wave.open(str(wav_path), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(44100)
        # 0.1 seconds of silence
        wf.writeframes(b"\x00\x00" * 4410)
    return wav_path


# ---------------------------------------------------------------------------
# beatmap_to_v3_dict
# ---------------------------------------------------------------------------


class TestBeatmapToV3Dict:
    def test_version_field(self, simple_beatmap):
        d = beatmap_to_v3_dict(simple_beatmap)
        assert d["version"] == "3.3.0"

    def test_color_notes(self, simple_beatmap):
        d = beatmap_to_v3_dict(simple_beatmap)
        assert len(d["colorNotes"]) == 1
        n = d["colorNotes"][0]
        assert n["b"] == 1.0
        assert n["x"] == 1
        assert n["y"] == 0
        assert n["c"] == 0
        assert n["d"] == 1
        assert n["a"] == 0

    def test_bomb_notes(self, simple_beatmap):
        d = beatmap_to_v3_dict(simple_beatmap)
        assert len(d["bombNotes"]) == 1
        b = d["bombNotes"][0]
        assert b["b"] == 2.0
        assert b["x"] == 2
        assert b["y"] == 1

    def test_obstacles(self, simple_beatmap):
        d = beatmap_to_v3_dict(simple_beatmap)
        assert len(d["obstacles"]) == 1
        o = d["obstacles"][0]
        assert o["b"] == 3.0
        assert o["d"] == 2.0
        assert o["w"] == 1
        assert o["h"] == 5

    def test_sliders(self, simple_beatmap):
        d = beatmap_to_v3_dict(simple_beatmap)
        assert len(d["sliders"]) == 1
        s = d["sliders"][0]
        assert s["c"] == 1
        assert s["b"] == 4.0
        assert s["tb"] == 5.0
        assert s["mu"] == 1.0
        assert s["tmu"] == 1.0
        assert s["m"] == 0

    def test_burst_sliders(self, simple_beatmap):
        d = beatmap_to_v3_dict(simple_beatmap)
        assert len(d["burstSliders"]) == 1
        bs = d["burstSliders"][0]
        assert bs["c"] == 0
        assert bs["b"] == 6.0
        assert bs["sc"] == 5
        assert bs["s"] == 0.5

    def test_basic_events(self, simple_beatmap):
        d = beatmap_to_v3_dict(simple_beatmap)
        assert len(d["basicBeatmapEvents"]) == 1
        e = d["basicBeatmapEvents"][0]
        assert e["b"] == 1.0
        assert e["et"] == 1
        assert e["i"] == 3

    def test_color_boost_events(self, simple_beatmap):
        d = beatmap_to_v3_dict(simple_beatmap)
        assert len(d["colorBoostBeatmapEvents"]) == 1
        assert d["colorBoostBeatmapEvents"][0]["o"] is True

    def test_empty_beatmap(self, empty_beatmap):
        d = beatmap_to_v3_dict(empty_beatmap)
        assert d["colorNotes"] == []
        assert d["bombNotes"] == []
        assert d["obstacles"] == []
        assert d["sliders"] == []
        assert d["burstSliders"] == []

    def test_json_serializable(self, simple_beatmap):
        d = beatmap_to_v3_dict(simple_beatmap)
        # Should not raise
        json_str = json.dumps(d)
        assert len(json_str) > 0

    def test_required_v3_keys(self, empty_beatmap):
        d = beatmap_to_v3_dict(empty_beatmap)
        for key in ("version", "colorNotes", "bombNotes", "obstacles", "sliders", "burstSliders"):
            assert key in d, f"Missing key: {key}"


# ---------------------------------------------------------------------------
# build_info_dat
# ---------------------------------------------------------------------------


class TestBuildInfoDat:
    def test_required_fields(self):
        info = build_info_dat(
            song_name="Test Song",
            song_author="Test Artist",
            bpm=120.0,
            difficulties=["Expert"],
        )
        assert info["_songName"] == "Test Song"
        assert info["_songAuthorName"] == "Test Artist"
        assert info["_beatsPerMinute"] == 120.0
        assert info["_levelAuthorName"] == "beatsaber_automapper"

    def test_single_difficulty(self):
        info = build_info_dat("S", "A", 120.0, ["Expert"])
        sets = info["_difficultyBeatmapSets"]
        assert len(sets) == 1
        maps = sets[0]["_difficultyBeatmaps"]
        assert len(maps) == 1
        assert maps[0]["_difficulty"] == "Expert"
        assert maps[0]["_beatmapFilename"] == "ExpertStandard.dat"
        assert maps[0]["_difficultyRank"] == 7

    def test_multiple_difficulties(self):
        info = build_info_dat("S", "A", 120.0, ["Easy", "Expert", "ExpertPlus"])
        maps = info["_difficultyBeatmapSets"][0]["_difficultyBeatmaps"]
        assert len(maps) == 3
        names = [m["_difficulty"] for m in maps]
        assert "Easy" in names
        assert "ExpertPlus" in names

    def test_expert_plus_rank(self):
        info = build_info_dat("S", "A", 120.0, ["ExpertPlus"])
        maps = info["_difficultyBeatmapSets"][0]["_difficultyBeatmaps"]
        assert maps[0]["_difficultyRank"] == 9

    def test_json_serializable(self):
        info = build_info_dat("S", "A", 128.5, ["Normal", "Hard"])
        json_str = json.dumps(info)
        assert len(json_str) > 0


# ---------------------------------------------------------------------------
# tokens_to_beatmap
# ---------------------------------------------------------------------------


class TestTokensToBeatmap:
    def test_empty_returns_beatmap(self):
        bm = tokens_to_beatmap({})
        assert bm is not None
        assert bm.color_notes == []

    def test_decode_note_tokens(self):
        from beatsaber_automapper.data.tokenizer import (
            ANGLE_OFFSET_OFFSET,
            COL_OFFSET,
            COLOR_OFFSET,
            DIR_OFFSET,
            EOS,
            NOTE,
            ROW_OFFSET,
        )

        # Encode a single note token sequence manually
        tokens = [
            NOTE,
            COLOR_OFFSET + 0,  # red
            COL_OFFSET + 1,    # x=1
            ROW_OFFSET + 0,    # y=0
            DIR_OFFSET + 1,    # down
            ANGLE_OFFSET_OFFSET + 3,  # angle bin 3 = 0°
            EOS,
        ]
        bm = tokens_to_beatmap({1.0: tokens})
        assert len(bm.color_notes) == 1
        n = bm.color_notes[0]
        assert n.beat == 1.0
        assert n.color == 0
        assert n.x == 1
        assert n.y == 0
        assert n.direction == 1


# ---------------------------------------------------------------------------
# package_level
# ---------------------------------------------------------------------------


class TestPackageLevel:
    def test_creates_zip(self, simple_beatmap, tmp_audio, tmp_path):
        out = tmp_path / "level.zip"
        result = package_level(
            beatmaps={"Expert": simple_beatmap},
            audio_path=tmp_audio,
            output_path=out,
            song_name="Test Song",
            song_author="Test Artist",
            bpm=120.0,
        )
        assert result == out
        assert out.exists()

    def test_zip_contains_info_dat(self, simple_beatmap, tmp_audio, tmp_path):
        out = tmp_path / "level.zip"
        package_level(
            beatmaps={"Expert": simple_beatmap},
            audio_path=tmp_audio,
            output_path=out,
        )
        with zipfile.ZipFile(out) as zf:
            names = zf.namelist()
        assert "Info.dat" in names

    def test_zip_contains_difficulty_dat(self, simple_beatmap, tmp_audio, tmp_path):
        out = tmp_path / "level.zip"
        package_level(
            beatmaps={"Expert": simple_beatmap},
            audio_path=tmp_audio,
            output_path=out,
        )
        with zipfile.ZipFile(out) as zf:
            names = zf.namelist()
        assert "ExpertStandard.dat" in names

    def test_zip_contains_audio(self, simple_beatmap, tmp_audio, tmp_path):
        out = tmp_path / "level.zip"
        package_level(
            beatmaps={"Expert": simple_beatmap},
            audio_path=tmp_audio,
            output_path=out,
        )
        with zipfile.ZipFile(out) as zf:
            names = zf.namelist()
        # Audio file should be present (with original extension)
        audio_files = [n for n in names if n.startswith("song.")]
        assert len(audio_files) == 1

    def test_info_dat_is_valid_json(self, simple_beatmap, tmp_audio, tmp_path):
        out = tmp_path / "level.zip"
        package_level(
            beatmaps={"Expert": simple_beatmap},
            audio_path=tmp_audio,
            output_path=out,
        )
        with zipfile.ZipFile(out) as zf:
            info_bytes = zf.read("Info.dat")
        info = json.loads(info_bytes)
        assert info["_songName"] is not None
        assert "_difficultyBeatmapSets" in info

    def test_difficulty_dat_is_valid_json(self, simple_beatmap, tmp_audio, tmp_path):
        out = tmp_path / "level.zip"
        package_level(
            beatmaps={"Expert": simple_beatmap},
            audio_path=tmp_audio,
            output_path=out,
        )
        with zipfile.ZipFile(out) as zf:
            dat_bytes = zf.read("ExpertStandard.dat")
        dat = json.loads(dat_bytes)
        assert dat["version"] == "3.3.0"
        assert "colorNotes" in dat

    def test_multiple_difficulties(self, simple_beatmap, empty_beatmap, tmp_audio, tmp_path):
        out = tmp_path / "level.zip"
        package_level(
            beatmaps={"Expert": simple_beatmap, "Hard": empty_beatmap},
            audio_path=tmp_audio,
            output_path=out,
        )
        with zipfile.ZipFile(out) as zf:
            names = zf.namelist()
        assert "ExpertStandard.dat" in names
        assert "HardStandard.dat" in names

    def test_creates_parent_dirs(self, simple_beatmap, tmp_audio, tmp_path):
        out = tmp_path / "subdir" / "nested" / "level.zip"
        package_level(
            beatmaps={"Expert": simple_beatmap},
            audio_path=tmp_audio,
            output_path=out,
        )
        assert out.exists()
