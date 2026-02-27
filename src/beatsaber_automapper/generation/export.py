"""Export generated tokens to Beat Saber v3 format.

Converts model output tokens into v3 JSON beatmap data and packages
everything into a playable .zip file with Info.dat, song audio, and
difficulty .dat files.
"""

from __future__ import annotations

import json
import logging
import zipfile
from pathlib import Path
from typing import Any

from beatsaber_automapper.data.beatmap import DifficultyBeatmap
from beatsaber_automapper.data.tokenizer import BeatmapTokenizer

logger = logging.getLogger(__name__)

# Difficulty metadata: name -> (rank, njs)
_DIFFICULTY_META: dict[str, tuple[int, float]] = {
    "Easy": (1, 10.0),
    "Normal": (3, 12.0),
    "Hard": (5, 14.0),
    "Expert": (7, 16.0),
    "ExpertPlus": (9, 18.0),
}


def beatmap_to_v3_dict(
    beatmap: DifficultyBeatmap,
    chroma_events: list[dict] | None = None,
) -> dict[str, Any]:
    """Convert a DifficultyBeatmap to a v3 .dat JSON dictionary.

    Args:
        beatmap: Parsed or generated difficulty beatmap.
        chroma_events: Optional list of lighting events with Chroma _customData.
            If provided, replaces the basic_events from beatmap with these
            Chroma-enhanced events.

    Returns:
        Dictionary matching the Beat Saber v3 .dat JSON structure.
    """
    color_notes = [
        {"b": n.beat, "x": n.x, "y": n.y, "c": n.color, "d": n.direction, "a": n.angle_offset}
        for n in beatmap.color_notes
    ]
    bomb_notes = [
        {"b": n.beat, "x": n.x, "y": n.y}
        for n in beatmap.bomb_notes
    ]
    obstacles = [
        {"b": o.beat, "d": o.duration, "x": o.x, "y": o.y, "w": o.width, "h": o.height}
        for o in beatmap.obstacles
    ]
    sliders = [
        {
            "c": s.color,
            "b": s.beat,
            "x": s.x,
            "y": s.y,
            "d": s.direction,
            "mu": s.mu,
            "tb": s.tail_beat,
            "tx": s.tail_x,
            "ty": s.tail_y,
            "tc": s.tail_direction,
            "tmu": s.tail_mu,
            "m": s.mid_anchor_mode,
        }
        for s in beatmap.sliders
    ]
    burst_sliders = [
        {
            "c": bs.color,
            "b": bs.beat,
            "x": bs.x,
            "y": bs.y,
            "d": bs.direction,
            "tb": bs.tail_beat,
            "tx": bs.tail_x,
            "ty": bs.tail_y,
            "sc": bs.slice_count,
            "s": bs.squish,
        }
        for bs in beatmap.burst_sliders
    ]

    # Use Chroma-enhanced events if provided, otherwise standard events
    if chroma_events is not None:
        basic_events = chroma_events
    else:
        basic_events = [
            {"b": e.beat, "et": e.event_type, "i": e.value, "f": e.float_value}
            for e in beatmap.basic_events
        ]

    color_boost_events = [
        {"b": e.beat, "o": e.boost}
        for e in beatmap.color_boost_events
    ]

    return {
        "version": "3.3.0",
        "colorNotes": color_notes,
        "bombNotes": bomb_notes,
        "obstacles": obstacles,
        "sliders": sliders,
        "burstSliders": burst_sliders,
        "basicBeatmapEvents": basic_events,
        "colorBoostBeatmapEvents": color_boost_events,
        "lightColorEventBoxGroups": [],
        "lightRotationEventBoxGroups": [],
        "lightTranslationEventBoxGroups": [],
        "vfxEventBoxGroups": [],
        "_fxEventsCollection": {"_fl": [], "_il": []},
        "useNormalEventsAsCompatibilityEvents": True,
    }


def tokens_to_beatmap(
    beat_tokens: dict[float, list[int]],
) -> DifficultyBeatmap:
    """Decode per-beat token sequences to a DifficultyBeatmap.

    This is a thin wrapper around BeatmapTokenizer.decode_beatmap.

    Args:
        beat_tokens: Dict mapping beat (float) -> list of tokens for that beat.

    Returns:
        DifficultyBeatmap with reconstructed note objects.
    """
    tokenizer = BeatmapTokenizer()
    return tokenizer.decode_beatmap(beat_tokens)


def build_info_dat(
    song_name: str,
    song_author: str,
    bpm: float,
    difficulties: list[str],
    song_filename: str = "song.ogg",
    cover_filename: str = "cover.png",
    environment_name: str = "DefaultEnvironment",
    song_time_offset: float = 0.0,
    preview_start: float = 12.0,
    preview_duration: float = 10.0,
    chroma: bool = True,
) -> dict[str, Any]:
    """Build an Info.dat dictionary for a generated level.

    Args:
        song_name: Title of the song.
        song_author: Artist name.
        bpm: Beats per minute.
        difficulties: List of difficulty names to include (e.g. ["Expert"]).
        song_filename: Filename for the audio within the zip.
        cover_filename: Filename for the cover image within the zip.
        environment_name: Beat Saber environment name.
        song_time_offset: Song time offset in seconds.
        preview_start: Preview start time in seconds.
        preview_duration: Preview duration in seconds.

    Returns:
        Info.dat dictionary.
    """
    beatmap_list = []
    for diff in difficulties:
        rank, njs = _DIFFICULTY_META.get(diff, (7, 16.0))
        beatmap_list.append(
            {
                "_difficulty": diff,
                "_difficultyRank": rank,
                "_beatmapFilename": f"{diff}Standard.dat",
                "_noteJumpMovementSpeed": njs,
                "_noteJumpStartBeatOffset": 0,
                "_customData": {},
            }
        )

    info_dat: dict[str, Any] = {
        "_version": "2.1.0",
        "_songName": song_name,
        "_songSubName": "",
        "_songAuthorName": song_author,
        "_levelAuthorName": "beatsaber_automapper",
        "_beatsPerMinute": bpm,
        "_shuffle": 0,
        "_shufflePeriod": 0.5,
        "_previewStartTime": preview_start,
        "_previewDuration": preview_duration,
        "_songFilename": song_filename,
        "_coverImageFilename": cover_filename,
        "_environmentName": environment_name,
        "_songTimeOffset": song_time_offset,
        "_difficultyBeatmapSets": [
            {
                "_beatmapCharacteristicName": "Standard",
                "_difficultyBeatmaps": beatmap_list,
            }
        ],
    }

    # Add Chroma as a suggestion for graceful degradation
    if chroma:
        info_dat["_customData"] = {"_suggestions": ["Chroma"]}

    return info_dat


def package_level(
    beatmaps: dict[str, DifficultyBeatmap],
    audio_path: Path,
    output_path: Path,
    song_name: str = "Generated Level",
    song_author: str = "Unknown Artist",
    bpm: float = 120.0,
    song_time_offset: float = 0.0,
    cover_path: Path | None = None,
    environment_name: str = "DefaultEnvironment",
    chroma_events: dict[str, list[dict] | None] | None = None,
) -> Path:
    """Package beatmap data and audio into a Beat Saber .zip.

    Creates a complete, loadable Beat Saber level package containing
    Info.dat, one difficulty .dat per entry in beatmaps, the song audio,
    and an optional cover image.

    Args:
        beatmaps: Dict mapping difficulty name -> DifficultyBeatmap.
        audio_path: Path to the song audio file.
        output_path: Path for the output .zip file.
        song_name: Song title for Info.dat.
        song_author: Song artist name for Info.dat.
        bpm: Song BPM for Info.dat.
        song_time_offset: Song time offset in seconds.
        cover_path: Optional path to a cover image (PNG/JPG).
        environment_name: Beat Saber environment name.

    Returns:
        Path to the generated .zip file.
    """
    import tempfile

    from beatsaber_automapper.data.audio import convert_to_ogg

    audio_path = Path(audio_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cover_in_zip = "cover.png" if cover_path is not None else ""

    # Convert audio to .ogg for best Beat Saber compatibility
    audio_suffix = audio_path.suffix.lower()
    if audio_suffix in (".ogg", ".egg"):
        zip_audio_name = "song.ogg"
        audio_to_pack = audio_path
        tmp_ogg = None
    else:
        zip_audio_name = "song.ogg"
        tmp_ogg = Path(tempfile.mktemp(suffix=".ogg"))
        convert_to_ogg(audio_path, tmp_ogg)
        audio_to_pack = tmp_ogg

    try:
        # Build Info.dat
        has_chroma = chroma_events is not None and any(v is not None for v in chroma_events.values())
        info = build_info_dat(
            song_name=song_name,
            song_author=song_author,
            bpm=bpm,
            difficulties=list(beatmaps.keys()),
            song_filename=zip_audio_name,
            cover_filename=cover_in_zip,
            environment_name=environment_name,
            song_time_offset=song_time_offset,
            chroma=has_chroma,
        )

        with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("Info.dat", json.dumps(info, indent=2))

            for diff_name, beatmap in beatmaps.items():
                dat_name = f"{diff_name}Standard.dat"
                diff_chroma = (
                    chroma_events.get(diff_name) if chroma_events else None
                )
                dat_dict = beatmap_to_v3_dict(beatmap, chroma_events=diff_chroma)
                zf.writestr(dat_name, json.dumps(dat_dict))
                logger.info(
                    "Packed %s: %d notes, %d bombs, %d walls, %d arcs, %d chains",
                    dat_name,
                    len(beatmap.color_notes),
                    len(beatmap.bomb_notes),
                    len(beatmap.obstacles),
                    len(beatmap.sliders),
                    len(beatmap.burst_sliders),
                )

            zf.write(audio_to_pack, zip_audio_name)

            if cover_path is not None:
                cover_path = Path(cover_path)
                zf.write(cover_path, "cover.png")
    finally:
        if tmp_ogg is not None and tmp_ogg.exists():
            tmp_ogg.unlink(missing_ok=True)

    logger.info("Wrote Beat Saber level to %s", output_path)
    return output_path
