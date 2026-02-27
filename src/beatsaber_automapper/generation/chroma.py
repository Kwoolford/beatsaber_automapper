"""Rule-based Chroma RGB color generator for lighting events.

Maps song energy/structure features to color palettes for Chroma-enhanced
lighting. This is a post-processing step applied after lighting events are
generated — the lighting model outputs vanilla events, and this module adds
_customData._color fields for Chroma-enabled environments.

Chroma is listed as a _suggestion in Info.dat, so maps gracefully degrade
to standard lighting when the mod is not installed.

Palettes:
    - synthwave: Purple/cyan/pink neon (default for electronic music)
    - edm: High-energy reds/whites/oranges (high bass energy)
    - chill: Cool blues/greens/soft purples (low energy)
    - rock: Warm reds/ambers/yellows (mid energy)
    - ambient: Deep blues/teals/soft whites (very low energy)
    - monochrome: White/grey with subtle blue tint (fallback)
"""

from __future__ import annotations

import logging
import math

import torch

logger = logging.getLogger(__name__)

# Each palette is a list of (r, g, b, a) tuples representing color stops
# from low energy to high energy. Colors are interpolated based on energy level.
PALETTES: dict[str, list[tuple[float, float, float, float]]] = {
    "synthwave": [
        (0.15, 0.05, 0.35, 1.0),  # deep purple (calm)
        (0.40, 0.10, 0.60, 1.0),  # purple (low)
        (0.20, 0.60, 0.80, 1.0),  # cyan (mid)
        (0.90, 0.20, 0.60, 1.0),  # hot pink (high)
        (1.00, 0.40, 0.80, 1.0),  # bright pink (max)
    ],
    "edm": [
        (0.10, 0.05, 0.20, 1.0),  # dark purple (calm)
        (0.80, 0.20, 0.10, 1.0),  # red (building)
        (1.00, 0.50, 0.10, 1.0),  # orange (mid-high)
        (1.00, 0.90, 0.30, 1.0),  # yellow (high)
        (1.00, 1.00, 1.00, 1.0),  # white (drop)
    ],
    "chill": [
        (0.05, 0.10, 0.25, 1.0),  # deep blue (calm)
        (0.10, 0.30, 0.50, 1.0),  # blue (low)
        (0.15, 0.50, 0.45, 1.0),  # teal (mid)
        (0.30, 0.60, 0.80, 1.0),  # light blue (mid-high)
        (0.60, 0.40, 0.70, 1.0),  # soft purple (high)
    ],
    "rock": [
        (0.20, 0.08, 0.02, 1.0),  # dark brown (calm)
        (0.60, 0.15, 0.05, 1.0),  # dark red (low)
        (0.80, 0.40, 0.10, 1.0),  # amber (mid)
        (1.00, 0.70, 0.20, 1.0),  # gold (high)
        (1.00, 0.90, 0.50, 1.0),  # bright yellow (max)
    ],
    "ambient": [
        (0.02, 0.05, 0.15, 1.0),  # near-black blue (calm)
        (0.05, 0.15, 0.30, 1.0),  # deep blue (low)
        (0.10, 0.25, 0.35, 1.0),  # ocean blue (mid)
        (0.15, 0.40, 0.40, 1.0),  # teal (mid-high)
        (0.30, 0.50, 0.60, 1.0),  # soft blue-grey (high)
    ],
    "monochrome": [
        (0.05, 0.05, 0.08, 1.0),  # near black
        (0.20, 0.20, 0.25, 1.0),  # dark grey
        (0.50, 0.50, 0.55, 1.0),  # mid grey
        (0.80, 0.80, 0.85, 1.0),  # light grey
        (1.00, 1.00, 1.00, 1.0),  # white
    ],
}


def select_palette(
    structure_features: torch.Tensor | None,
    genre: str = "unknown",
) -> str:
    """Auto-select a color palette based on song energy profile and genre.

    Args:
        structure_features: Song structure features [6, T] or None.
        genre: Genre string from metadata.

    Returns:
        Palette name key into PALETTES dict.
    """
    # Genre-based hints
    genre_lower = genre.lower()
    if any(g in genre_lower for g in ("electronic", "edm", "dubstep", "drum")):
        return "edm"
    if any(g in genre_lower for g in ("ambient", "classical", "soundtrack")):
        return "ambient"
    if any(g in genre_lower for g in ("rock", "metal", "punk")):
        return "rock"
    if any(g in genre_lower for g in ("chill", "lofi", "jazz")):
        return "chill"

    if structure_features is None:
        return "synthwave"  # default

    # Energy-based selection from structure features
    # structure_features[0] = RMS energy, [2] = bass energy
    mean_rms = structure_features[0].mean().item()
    mean_bass = structure_features[2].mean().item()

    if mean_rms > 0.5 and mean_bass > 0.5:
        return "edm"
    elif mean_rms < 0.2:
        return "ambient"
    elif mean_bass > 0.4:
        return "rock"
    elif mean_rms < 0.35:
        return "chill"
    else:
        return "synthwave"


def _lerp_color(
    c1: tuple[float, float, float, float],
    c2: tuple[float, float, float, float],
    t: float,
) -> list[float]:
    """Linearly interpolate between two RGBA colors."""
    return [
        c1[0] + (c2[0] - c1[0]) * t,
        c1[1] + (c2[1] - c1[1]) * t,
        c1[2] + (c2[2] - c1[2]) * t,
        c1[3] + (c2[3] - c1[3]) * t,
    ]


def _energy_to_color(
    energy: float,
    palette: list[tuple[float, float, float, float]],
) -> list[float]:
    """Map a 0-1 energy value to an RGBA color via palette interpolation."""
    energy = max(0.0, min(1.0, energy))
    n = len(palette) - 1
    idx = energy * n
    lo = int(math.floor(idx))
    hi = min(lo + 1, n)
    t = idx - lo
    return _lerp_color(palette[lo], palette[hi], t)


def _hue_shift(color: list[float], shift: float) -> list[float]:
    """Apply a slight hue rotation to an RGB color for variety.

    Uses a simplified rotation in RGB space (not true HSV) for speed.
    shift should be small (-0.1 to 0.1).
    """
    r, g, b, a = color
    # Rotate towards adjacent color channels
    return [
        max(0, min(1, r + shift * 0.5)),
        max(0, min(1, g - shift * 0.3)),
        max(0, min(1, b + shift * 0.2)),
        a,
    ]


def add_chroma_colors(
    events: list[dict],
    structure_features: torch.Tensor | None,
    bpm: float,
    sample_rate: int = 44100,
    hop_length: int = 512,
    palette_name: str = "auto",
    genre: str = "unknown",
) -> list[dict]:
    """Add Chroma _customData._color to vanilla lighting events.

    Each event gets an RGB color based on the song's energy at that beat
    position, interpolated within the selected color palette.

    Args:
        events: List of lighting event dicts with "b" (beat) key.
        structure_features: Song structure features [6, T] or None.
            Features: [rms, onset_strength, bass, mid, high, centroid].
        bpm: Song BPM.
        sample_rate: Audio sample rate.
        hop_length: Spectrogram hop length.
        palette_name: Palette name or "auto" for energy-based selection.
        genre: Genre string for palette auto-selection.

    Returns:
        Same events list with _customData._color added to each event.
    """
    if palette_name == "auto":
        palette_name = select_palette(structure_features, genre)

    palette = PALETTES.get(palette_name, PALETTES["synthwave"])
    logger.info("Using Chroma palette: %s", palette_name)

    if structure_features is None:
        # No structure features — use uniform mid-energy colors
        for event in events:
            event["_customData"] = {"_color": _energy_to_color(0.5, palette)}
        return events

    n_frames = structure_features.shape[1]
    frames_per_second = sample_rate / hop_length

    for event in events:
        beat = event.get("b", 0.0)
        # Convert beat to frame index
        time_sec = beat * 60.0 / max(bpm, 1.0)
        frame = int(round(time_sec * frames_per_second))
        frame = max(0, min(frame, n_frames - 1))

        # Get energy at this frame (RMS energy is feature 0)
        energy = structure_features[0, frame].item()

        # Get event type for hue variation (different light groups get different hues)
        et = event.get("et", 0)
        hue_shift = (et % 5) * 0.04 - 0.08  # range: -0.08 to +0.08

        color = _energy_to_color(energy, palette)
        color = _hue_shift(color, hue_shift)

        # Brightness modulation: scale alpha by onset strength
        onset_str = structure_features[1, frame].item()
        color[3] = max(0.3, min(1.0, 0.5 + onset_str * 0.5))

        event["_customData"] = {"_color": [round(c, 4) for c in color]}

    return events
