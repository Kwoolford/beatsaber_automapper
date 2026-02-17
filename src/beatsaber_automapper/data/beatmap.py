"""Beat Saber v3 beatmap parser.

Parses Info.dat and difficulty .dat files, extracting all v3 object types:
colorNotes, bombNotes, obstacles, sliders (arcs), burstSliders (chains),
basicBeatmapEvents, colorBoostBeatmapEvents.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataclasses — fields mirror v3 JSON shorthand
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class ColorNote:
    """A standard directional note (v3 colorNotes)."""

    beat: float  # b
    x: int  # column 0-3
    y: int  # row 0-2
    color: int  # c: 0=red, 1=blue
    direction: int  # d: 0-8
    angle_offset: int = 0  # a


@dataclass(slots=True)
class BombNote:
    """A bomb note (v3 bombNotes)."""

    beat: float  # b
    x: int  # column 0-3
    y: int  # row 0-2


@dataclass(slots=True)
class Obstacle:
    """A wall/obstacle (v3 obstacles)."""

    beat: float  # b
    duration: float  # d
    x: int  # column 0-3
    y: int  # row 0-2
    width: int  # w
    height: int  # h (1-5)


@dataclass(slots=True)
class Slider:
    """An arc/slider (v3 sliders)."""

    color: int  # c
    beat: float  # b  (head)
    x: int  # head column
    y: int  # head row
    direction: int  # d  head direction
    mu: float  # head curvature multiplier
    tail_beat: float  # tb
    tail_x: int  # tx
    tail_y: int  # ty
    tail_direction: int  # tc
    tail_mu: float  # tmu
    mid_anchor_mode: int = 0  # m


@dataclass(slots=True)
class BurstSlider:
    """A burst slider / chain (v3 burstSliders)."""

    color: int  # c
    beat: float  # b  (head)
    x: int  # head column
    y: int  # head row
    direction: int  # d  head direction
    tail_beat: float  # tb
    tail_x: int  # tx
    tail_y: int  # ty
    slice_count: int  # sc
    squish: float  # s


@dataclass(slots=True)
class BasicEvent:
    """A basic beatmap event (v3 basicBeatmapEvents)."""

    beat: float  # b
    event_type: int  # et
    value: int  # i
    float_value: float = 1.0  # f


@dataclass(slots=True)
class ColorBoostEvent:
    """A color boost event (v3 colorBoostBeatmapEvents)."""

    beat: float  # b
    boost: bool  # o


@dataclass(slots=True)
class DifficultyInfo:
    """Metadata for one difficulty within a beatmap set."""

    difficulty: str  # e.g. "Expert"
    difficulty_rank: int  # e.g. 7
    filename: str  # e.g. "ExpertStandard.dat"
    note_jump_speed: float = 16.0
    note_jump_offset: float = 0.0


@dataclass(slots=True)
class BeatmapInfo:
    """Parsed Info.dat metadata."""

    song_name: str
    song_sub_name: str
    song_author: str
    level_author: str
    bpm: float
    song_filename: str
    cover_filename: str
    environment_name: str
    song_time_offset: float
    preview_start: float
    preview_duration: float
    difficulties: list[DifficultyInfo] = field(default_factory=list)


@dataclass(slots=True)
class DifficultyBeatmap:
    """All parsed objects from a single difficulty .dat file."""

    version: str
    color_notes: list[ColorNote] = field(default_factory=list)
    bomb_notes: list[BombNote] = field(default_factory=list)
    obstacles: list[Obstacle] = field(default_factory=list)
    sliders: list[Slider] = field(default_factory=list)
    burst_sliders: list[BurstSlider] = field(default_factory=list)
    basic_events: list[BasicEvent] = field(default_factory=list)
    color_boost_events: list[ColorBoostEvent] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------


def parse_info_dat(path: Path | str) -> BeatmapInfo | None:
    """Parse a Beat Saber Info.dat file.

    Args:
        path: Path to Info.dat file.

    Returns:
        BeatmapInfo with song metadata and difficulty list, or None on error.
    """
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))

    difficulties: list[DifficultyInfo] = []
    for bset in data.get("_difficultyBeatmapSets", []):
        for diff in bset.get("_difficultyBeatmaps", []):
            difficulties.append(
                DifficultyInfo(
                    difficulty=diff.get("_difficulty", ""),
                    difficulty_rank=diff.get("_difficultyRank", 0),
                    filename=diff.get("_beatmapFilename", ""),
                    note_jump_speed=diff.get("_noteJumpMovementSpeed", 16.0),
                    note_jump_offset=diff.get("_noteJumpStartBeatOffset", 0.0),
                )
            )

    return BeatmapInfo(
        song_name=data.get("_songName", ""),
        song_sub_name=data.get("_songSubName", ""),
        song_author=data.get("_songAuthorName", ""),
        level_author=data.get("_levelAuthorName", ""),
        bpm=float(data.get("_beatsPerMinute", 120)),
        song_filename=data.get("_songFilename", "song.ogg"),
        cover_filename=data.get("_coverImageFilename", ""),
        environment_name=data.get("_environmentName", "DefaultEnvironment"),
        song_time_offset=float(data.get("_songTimeOffset", 0)),
        preview_start=float(data.get("_previewStartTime", 0)),
        preview_duration=float(data.get("_previewDuration", 10)),
        difficulties=difficulties,
    )


def parse_info_dat_json(data: dict[str, Any]) -> BeatmapInfo | None:
    """Parse Info.dat from an already-loaded JSON dict.

    Args:
        data: Parsed JSON dictionary from Info.dat.

    Returns:
        BeatmapInfo with song metadata and difficulty list, or None on error.
    """
    difficulties: list[DifficultyInfo] = []
    for bset in data.get("_difficultyBeatmapSets", []):
        for diff in bset.get("_difficultyBeatmaps", []):
            difficulties.append(
                DifficultyInfo(
                    difficulty=diff.get("_difficulty", ""),
                    difficulty_rank=diff.get("_difficultyRank", 0),
                    filename=diff.get("_beatmapFilename", ""),
                    note_jump_speed=diff.get("_noteJumpMovementSpeed", 16.0),
                    note_jump_offset=diff.get("_noteJumpStartBeatOffset", 0.0),
                )
            )

    return BeatmapInfo(
        song_name=data.get("_songName", ""),
        song_sub_name=data.get("_songSubName", ""),
        song_author=data.get("_songAuthorName", ""),
        level_author=data.get("_levelAuthorName", ""),
        bpm=float(data.get("_beatsPerMinute", 120)),
        song_filename=data.get("_songFilename", "song.ogg"),
        cover_filename=data.get("_coverImageFilename", ""),
        environment_name=data.get("_environmentName", "DefaultEnvironment"),
        song_time_offset=float(data.get("_songTimeOffset", 0)),
        preview_start=float(data.get("_previewStartTime", 0)),
        preview_duration=float(data.get("_previewDuration", 10)),
        difficulties=difficulties,
    )


def parse_difficulty_dat(path: Path | str) -> DifficultyBeatmap | None:
    """Parse a Beat Saber v3 difficulty .dat file.

    Args:
        path: Path to difficulty .dat file (e.g., ExpertStandard.dat).

    Returns:
        DifficultyBeatmap with all parsed objects, or None if v2 format.
    """
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    return parse_difficulty_dat_json(data)


def parse_difficulty_dat_json(data: dict[str, Any]) -> DifficultyBeatmap | None:
    """Parse a difficulty .dat from an already-loaded JSON dict.

    Args:
        data: Parsed JSON dictionary from a difficulty .dat file.

    Returns:
        DifficultyBeatmap with all parsed objects, or None if v2 format.
    """
    version = data.get("version", "")
    if not version or version.startswith("2"):
        # v2 format not supported
        if version.startswith("2"):
            logger.warning("Skipping v2 beatmap (version=%s)", version)
        else:
            # Check for _version field (v2 indicator)
            v2_version = data.get("_version", "")
            if v2_version:
                logger.warning("Skipping v2 beatmap (_version=%s)", v2_version)
                return None
            logger.warning("No version field found, skipping beatmap")
        return None

    return DifficultyBeatmap(
        version=version,
        color_notes=[_parse_color_note(n) for n in data.get("colorNotes", [])],
        bomb_notes=[_parse_bomb_note(n) for n in data.get("bombNotes", [])],
        obstacles=[_parse_obstacle(o) for o in data.get("obstacles", [])],
        sliders=[_parse_slider(s) for s in data.get("sliders", [])],
        burst_sliders=[_parse_burst_slider(bs) for bs in data.get("burstSliders", [])],
        basic_events=[_parse_basic_event(e) for e in data.get("basicBeatmapEvents", [])],
        color_boost_events=[_parse_color_boost(e) for e in data.get("colorBoostBeatmapEvents", [])],
    )


# ---------------------------------------------------------------------------
# Internal parsers for each object type
# ---------------------------------------------------------------------------


def _parse_color_note(d: dict[str, Any]) -> ColorNote:
    return ColorNote(
        beat=float(d.get("b", 0)),
        x=int(d.get("x", 0)),
        y=int(d.get("y", 0)),
        color=int(d.get("c", 0)),
        direction=int(d.get("d", 0)),
        angle_offset=int(d.get("a", 0)),
    )


def _parse_bomb_note(d: dict[str, Any]) -> BombNote:
    return BombNote(
        beat=float(d.get("b", 0)),
        x=int(d.get("x", 0)),
        y=int(d.get("y", 0)),
    )


def _parse_obstacle(d: dict[str, Any]) -> Obstacle:
    return Obstacle(
        beat=float(d.get("b", 0)),
        duration=float(d.get("d", 0)),
        x=int(d.get("x", 0)),
        y=int(d.get("y", 0)),
        width=int(d.get("w", 1)),
        height=int(d.get("h", 1)),
    )


def _parse_slider(d: dict[str, Any]) -> Slider:
    return Slider(
        color=int(d.get("c", 0)),
        beat=float(d.get("b", 0)),
        x=int(d.get("x", 0)),
        y=int(d.get("y", 0)),
        direction=int(d.get("d", 0)),
        mu=float(d.get("mu", 1.0)),
        tail_beat=float(d.get("tb", 0)),
        tail_x=int(d.get("tx", 0)),
        tail_y=int(d.get("ty", 0)),
        tail_direction=int(d.get("tc", 0)),
        tail_mu=float(d.get("tmu", 1.0)),
        mid_anchor_mode=int(d.get("m", 0)),
    )


def _parse_burst_slider(d: dict[str, Any]) -> BurstSlider:
    return BurstSlider(
        color=int(d.get("c", 0)),
        beat=float(d.get("b", 0)),
        x=int(d.get("x", 0)),
        y=int(d.get("y", 0)),
        direction=int(d.get("d", 0)),
        tail_beat=float(d.get("tb", 0)),
        tail_x=int(d.get("tx", 0)),
        tail_y=int(d.get("ty", 0)),
        slice_count=int(d.get("sc", 3)),
        squish=float(d.get("s", 0.5)),
    )


def _parse_basic_event(d: dict[str, Any]) -> BasicEvent:
    return BasicEvent(
        beat=float(d.get("b", 0)),
        event_type=int(d.get("et", 0)),
        value=int(d.get("i", 0)),
        float_value=float(d.get("f", 1.0)),
    )


def _parse_color_boost(d: dict[str, Any]) -> ColorBoostEvent:
    return ColorBoostEvent(
        beat=float(d.get("b", 0)),
        boost=bool(d.get("o", False)),
    )
