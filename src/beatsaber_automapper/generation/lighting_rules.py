"""Rule-based lighting event generator.

Replaces the ML-based lighting model with a deterministic system that
classifies song sections using energy features and maps them to lighting
templates. Produces v3 basicBeatmapEvents and colorBoostBeatmapEvents.

Section types:
    intro    — first ~15 seconds, gradually rising energy
    calm     — low RMS, low onset strength
    buildup  — rising RMS over 4-8 beat window
    drop     — high RMS + high bass energy (sustained)
    breakdown — sudden RMS drop after a drop
    outro    — last ~15 seconds, declining energy

Each section maps to a lighting pattern with density, event types, and
boost on/off settings.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import StrEnum

import torch

from beatsaber_automapper.data.beatmap import BasicEvent, ColorBoostEvent

logger = logging.getLogger(__name__)


class SectionType(StrEnum):
    INTRO = "intro"
    CALM = "calm"
    BUILDUP = "buildup"
    DROP = "drop"
    BREAKDOWN = "breakdown"
    OUTRO = "outro"


@dataclass
class SectionLabel:
    """A labeled section of the song."""

    start_beat: float
    end_beat: float
    section_type: SectionType


# v3 event types for basicBeatmapEvents
# 0=back lasers, 1=ring lights, 2=left lasers, 3=right lasers, 4=center lights
# 8=ring rotation, 9=ring zoom, 12=left laser speed, 13=right laser speed
_LIGHT_GROUPS = [0, 1, 2, 3, 4]
_RING_ROTATION = 8
_RING_ZOOM = 9
_LEFT_SPEED = 12
_RIGHT_SPEED = 13

# Event values: 0=off, 1=on, 2=flash, 3=fade, 5=on(blue), 6=flash(blue), 7=fade(blue)
_OFF = 0
_ON_RED = 1
_FLASH_RED = 2
_FADE_RED = 3
_ON_BLUE = 5
_FLASH_BLUE = 6
_FADE_BLUE = 7


def classify_sections(
    structure_features: torch.Tensor,
    bpm: float,
    sample_rate: int = 44100,
    hop_length: int = 512,
) -> list[SectionLabel]:
    """Classify each beat region into a song section type.

    Uses RMS energy, onset strength, and bass energy from structure_features
    to detect intro, calm, buildup, drop, breakdown, and outro.

    Args:
        structure_features: [6, T] features (RMS, onset_strength, bass, mid, high, centroid).
        bpm: Song BPM.
        sample_rate: Audio sample rate.
        hop_length: Spectrogram hop length.

    Returns:
        List of SectionLabel covering the full song duration.
    """
    total_frames = structure_features.shape[1]
    frames_per_second = sample_rate / hop_length
    frames_per_beat = (60.0 / max(bpm, 1.0)) * frames_per_second
    total_beats = total_frames / max(frames_per_beat, 1.0)

    # Extract and smooth features
    rms = structure_features[0].cpu().numpy()
    bass = structure_features[2].cpu().numpy()

    # Smooth with ~4-beat window
    smooth_window = max(1, int(frames_per_beat * 4))
    import numpy as np

    def _smooth(arr: np.ndarray, window: int) -> np.ndarray:
        if len(arr) < window:
            return arr
        kernel = np.ones(window) / window
        return np.convolve(arr, kernel, mode="same")

    smooth_rms = _smooth(rms, smooth_window)
    smooth_bass = _smooth(bass, smooth_window)

    # Normalize to 0-1
    def _norm(arr: np.ndarray) -> np.ndarray:
        mn, mx = arr.min(), arr.max()
        if mx > mn:
            return (arr - mn) / (mx - mn)
        return np.full_like(arr, 0.5)

    norm_rms = _norm(smooth_rms)
    norm_bass = _norm(smooth_bass)

    # Intro/outro detection: first/last ~15 seconds
    intro_frames = int(15.0 * frames_per_second)
    outro_frames = int(15.0 * frames_per_second)

    # Build per-beat labels
    sections: list[SectionLabel] = []
    n_beats = int(total_beats) + 1

    # Compute per-beat energy
    beat_energies = []
    for b in range(n_beats):
        frame = int(b * frames_per_beat)
        frame = min(frame, total_frames - 1)
        beat_energies.append((norm_rms[frame], norm_bass[frame]))

    # Classify beats
    prev_type = SectionType.INTRO
    current_start = 0.0

    for b in range(n_beats):
        frame = int(b * frames_per_beat)
        e_rms, e_bass = beat_energies[b]

        # Intro: first ~15 seconds
        if frame < intro_frames:
            section_type = SectionType.INTRO
        # Outro: last ~15 seconds
        elif frame > total_frames - outro_frames:
            section_type = SectionType.OUTRO
        # Drop: high RMS and high bass
        elif e_rms > 0.65 and e_bass > 0.5:
            section_type = SectionType.DROP
        # Buildup: mid-high RMS with rising trend (check 4-beat slope)
        elif 0.4 < e_rms < 0.7 and b >= 4:
            prev_e = beat_energies[b - 4][0]
            if e_rms > prev_e + 0.1:
                section_type = SectionType.BUILDUP
            else:
                section_type = SectionType.CALM
        # Breakdown: low energy right after a drop
        elif e_rms < 0.4 and prev_type == SectionType.DROP:
            section_type = SectionType.BREAKDOWN
        # Calm: low energy
        elif e_rms < 0.4:
            section_type = SectionType.CALM
        else:
            section_type = SectionType.CALM

        # Extend current section or start new one
        if section_type != prev_type:
            if b > 0:
                sections.append(SectionLabel(
                    start_beat=current_start,
                    end_beat=float(b),
                    section_type=prev_type,
                ))
            current_start = float(b)
            prev_type = section_type

    # Final section
    sections.append(SectionLabel(
        start_beat=current_start,
        end_beat=float(n_beats),
        section_type=prev_type,
    ))

    # Merge very short sections (< 4 beats) into neighbors
    merged: list[SectionLabel] = []
    for s in sections:
        if merged and (s.end_beat - s.start_beat) < 4.0:
            merged[-1].end_beat = s.end_beat
        else:
            merged.append(s)

    logger.info(
        "Section classification: %s",
        [(s.section_type.value, f"{s.start_beat:.0f}-{s.end_beat:.0f}") for s in merged],
    )
    return merged


def _generate_section_events(
    section: SectionLabel,
    bpm: float,
) -> tuple[list[BasicEvent], list[ColorBoostEvent]]:
    """Generate lighting events for a single section.

    Args:
        section: The section to generate events for.
        bpm: Song BPM.

    Returns:
        Tuple of (basic_events, color_boost_events) for this section.
    """
    basic_events: list[BasicEvent] = []
    boost_events: list[ColorBoostEvent] = []

    st = section.section_type
    start = section.start_beat
    end = section.end_beat
    duration = end - start

    if duration <= 0:
        return basic_events, boost_events

    # Density: events per beat
    density_map = {
        SectionType.INTRO: 0.25,
        SectionType.CALM: 0.5,
        SectionType.BUILDUP: 0.75,
        SectionType.DROP: 1.0,
        SectionType.BREAKDOWN: 0.5,
        SectionType.OUTRO: 0.25,
    }
    density = density_map.get(st, 0.5)
    beat_step = 1.0 / max(density, 0.1)

    # Color boost: on during drops
    use_boost = st in (SectionType.DROP, SectionType.BUILDUP)
    boost_events.append(ColorBoostEvent(beat=round(start, 4), boost=use_boost))

    # Generate events based on section type
    beat = start
    event_idx = 0
    while beat < end:
        if st == SectionType.INTRO:
            # Slow fade-in across different light groups
            group = _LIGHT_GROUPS[event_idx % len(_LIGHT_GROUPS)]
            value = _FADE_RED if event_idx % 2 == 0 else _FADE_BLUE
            basic_events.append(BasicEvent(
                beat=round(beat, 4), event_type=group, value=value, float_value=0.5,
            ))

        elif st == SectionType.CALM:
            # Gentle pulse alternating colors
            group = _LIGHT_GROUPS[event_idx % len(_LIGHT_GROUPS)]
            if event_idx % 4 < 2:
                value = _ON_BLUE
                brightness = 0.6
            else:
                value = _FADE_RED
                brightness = 0.4
            basic_events.append(BasicEvent(
                beat=round(beat, 4), event_type=group, value=value, float_value=brightness,
            ))

        elif st == SectionType.BUILDUP:
            # Accelerating flashes + ring rotation
            group = _LIGHT_GROUPS[event_idx % len(_LIGHT_GROUPS)]
            value = _FLASH_RED if event_idx % 2 == 0 else _FLASH_BLUE
            brightness = min(1.0, 0.5 + (beat - start) / max(duration, 1) * 0.5)
            basic_events.append(BasicEvent(
                beat=round(beat, 4), event_type=group, value=value, float_value=brightness,
            ))
            # Add ring rotation periodically
            if event_idx % 4 == 0:
                basic_events.append(BasicEvent(
                    beat=round(beat, 4), event_type=_RING_ROTATION, value=1, float_value=1.0,
                ))
            # Accelerate: reduce beat step as we progress
            progress = (beat - start) / max(duration, 1)
            beat_step = max(0.25, (1.0 / density) * (1.0 - progress * 0.5))

        elif st == SectionType.DROP:
            # Full intensity: strobe, ring rotation, all groups
            for g_idx, group in enumerate(_LIGHT_GROUPS):
                if event_idx % 2 == 0:
                    value = _FLASH_RED if g_idx % 2 == 0 else _FLASH_BLUE
                else:
                    value = _ON_BLUE if g_idx % 2 == 0 else _ON_RED
                basic_events.append(BasicEvent(
                    beat=round(beat, 4), event_type=group, value=value, float_value=1.0,
                ))
            # Ring effects every beat
            basic_events.append(BasicEvent(
                beat=round(beat, 4), event_type=_RING_ROTATION, value=1, float_value=1.0,
            ))
            if event_idx % 2 == 0:
                basic_events.append(BasicEvent(
                    beat=round(beat, 4), event_type=_RING_ZOOM, value=1, float_value=1.0,
                ))

        elif st == SectionType.BREAKDOWN:
            # Fade down from drop intensity
            group = _LIGHT_GROUPS[event_idx % len(_LIGHT_GROUPS)]
            progress = (beat - start) / max(duration, 1)
            brightness = max(0.2, 1.0 - progress * 0.8)
            value = _FADE_BLUE if event_idx % 2 == 0 else _FADE_RED
            basic_events.append(BasicEvent(
                beat=round(beat, 4), event_type=group, value=value, float_value=brightness,
            ))

        elif st == SectionType.OUTRO:
            # Slow fade out
            group = _LIGHT_GROUPS[event_idx % len(_LIGHT_GROUPS)]
            progress = (beat - start) / max(duration, 1)
            brightness = max(0.1, 0.5 * (1.0 - progress))
            value = _FADE_BLUE
            basic_events.append(BasicEvent(
                beat=round(beat, 4), event_type=group, value=value, float_value=brightness,
            ))

        beat += beat_step
        event_idx += 1

    return basic_events, boost_events


def generate_lighting_events(
    structure_features: torch.Tensor,
    bpm: float,
    sample_rate: int = 44100,
    hop_length: int = 512,
) -> tuple[list[BasicEvent], list[ColorBoostEvent]]:
    """Generate complete lighting events from song structure features.

    This is the main entry point replacing the ML lighting model.

    Args:
        structure_features: [6, T] song structure features.
        bpm: Song BPM.
        sample_rate: Audio sample rate.
        hop_length: Spectrogram hop length.

    Returns:
        Tuple of (basic_events, color_boost_events) for the full song.
    """
    sections = classify_sections(
        structure_features, bpm, sample_rate=sample_rate, hop_length=hop_length,
    )

    all_basic: list[BasicEvent] = []
    all_boost: list[ColorBoostEvent] = []

    for section in sections:
        basic, boost = _generate_section_events(section, bpm)
        all_basic.extend(basic)
        all_boost.extend(boost)

    # Sort by beat
    all_basic.sort(key=lambda e: e.beat)
    all_boost.sort(key=lambda e: e.beat)

    logger.info(
        "Rule-based lighting: %d basic events, %d boost events across %d sections",
        len(all_basic), len(all_boost), len(sections),
    )
    return all_basic, all_boost
