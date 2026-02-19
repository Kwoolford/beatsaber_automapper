"""Token vocabulary for Beat Saber note events.

Converts between beatmap objects and token sequences for model training.

Encoding per event type:
    NOTE:      [NOTE] [COLOR] [COL] [ROW] [DIR] [ANGLE]       (6 tokens)
    BOMB:      [BOMB] [COL] [ROW]                               (3 tokens)
    WALL:      [WALL] [COL] [ROW] [W] [H] [DUR_INT] [DUR_FRAC] (7 tokens)
    ARC_START: [ARC_START] [COLOR] [COL] [ROW] [DIR] [MU]       (6 tokens)
    ARC_END:   [ARC_END] [COLOR] [COL] [ROW] [DIR] [MU] [MID]   (7 tokens)
    CHAIN:     [CHAIN] [COLOR] [COL] [ROW] [DIR] [TAIL_COL] [TAIL_ROW] [SLICE] [SQUISH] (9 tokens)

Special tokens: PAD=0, EOS=1, SEP=2, BOS=3.
Canonical ordering within a timestamp: left-to-right, bottom-to-top;
type priority: NOTE > BOMB > WALL > ARC_START > ARC_END > CHAIN.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass

from beatsaber_automapper.data.beatmap import (
    BombNote,
    BurstSlider,
    ColorNote,
    DifficultyBeatmap,
    Obstacle,
    Slider,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Token layout constants
# ---------------------------------------------------------------------------

# Special tokens
PAD = 0
EOS = 1
SEP = 2
BOS = 3

# Event type tokens (offset 4)
_EVENT_BASE = 4
NOTE = _EVENT_BASE + 0  # 4
BOMB = _EVENT_BASE + 1  # 5
WALL = _EVENT_BASE + 2  # 6
ARC_START = _EVENT_BASE + 3  # 7
ARC_END = _EVENT_BASE + 4  # 8
CHAIN = _EVENT_BASE + 5  # 9

# Attribute token ranges (each gets its own offset range)
_ATTR_BASE = 10

# COLOR: 2 values (0=red, 1=blue)
COLOR_OFFSET = _ATTR_BASE  # 10-11
COLOR_COUNT = 2

# COL: 4 values (0-3)
COL_OFFSET = COLOR_OFFSET + COLOR_COUNT  # 12-15
COL_COUNT = 4

# ROW: 3 values (0-2)
ROW_OFFSET = COL_OFFSET + COL_COUNT  # 16-18
ROW_COUNT = 3

# DIR: 9 values (0-8)
DIR_OFFSET = ROW_OFFSET + ROW_COUNT  # 19-27
DIR_COUNT = 9

# ANGLE_OFFSET: 7 bins at 15° steps: -45, -30, -15, 0, 15, 30, 45
ANGLE_OFFSET_OFFSET = DIR_OFFSET + DIR_COUNT  # 28-34
ANGLE_OFFSET_COUNT = 7
_ANGLE_BINS = [-45, -30, -15, 0, 15, 30, 45]

# MU (curvature): 9 bins at 0.25 steps over 0.0-2.0
MU_OFFSET = ANGLE_OFFSET_OFFSET + ANGLE_OFFSET_COUNT  # 35-43
MU_COUNT = 9
_MU_BINS = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]

# MID_ANCHOR: 3 values (0-2)
MID_ANCHOR_OFFSET = MU_OFFSET + MU_COUNT  # 44-46
MID_ANCHOR_COUNT = 3

# SLICE_COUNT: 31 values (2-32)
SLICE_OFFSET = MID_ANCHOR_OFFSET + MID_ANCHOR_COUNT  # 47-77
SLICE_COUNT = 31
_SLICE_MIN = 2
_SLICE_MAX = 32

# SQUISH: 11 bins at 0.1 steps over 0.0-1.0
SQUISH_OFFSET = SLICE_OFFSET + SLICE_COUNT  # 78-88
SQUISH_COUNT = 11
_SQUISH_BINS = [round(i * 0.1, 1) for i in range(11)]

# WALL WIDTH: 4 values (1-4)
WIDTH_OFFSET = SQUISH_OFFSET + SQUISH_COUNT  # 89-92
WIDTH_COUNT = 4

# WALL HEIGHT: 5 values (1-5)
HEIGHT_OFFSET = WIDTH_OFFSET + WIDTH_COUNT  # 93-97
HEIGHT_COUNT = 5

# WALL DURATION INT: 65 values (0-64)
DUR_INT_OFFSET = HEIGHT_OFFSET + HEIGHT_COUNT  # 98-162
DUR_INT_COUNT = 65

# WALL DURATION FRAC: 4 bins (0, 0.25, 0.5, 0.75)
DUR_FRAC_OFFSET = DUR_INT_OFFSET + DUR_INT_COUNT  # 163-166
DUR_FRAC_COUNT = 4
_DUR_FRAC_BINS = [0.0, 0.25, 0.5, 0.75]

VOCAB_SIZE = DUR_FRAC_OFFSET + DUR_FRAC_COUNT  # 167

# Type priority for canonical ordering (lower = higher priority)
_TYPE_PRIORITY = {
    NOTE: 0,
    BOMB: 1,
    WALL: 2,
    ARC_START: 3,
    ARC_END: 4,
    CHAIN: 5,
}


# ---------------------------------------------------------------------------
# Internal event representation
# ---------------------------------------------------------------------------


@dataclass
class _TokenEvent:
    """Internal representation of an event for sorting."""

    type_token: int
    x: int  # primary column for sorting
    y: int  # primary row for sorting
    tokens: list[int]


# ---------------------------------------------------------------------------
# Quantization helpers
# ---------------------------------------------------------------------------


def _quantize_to_bin(value: float, bins: list[float]) -> int:
    """Find nearest bin index for a value."""
    best_idx = 0
    best_dist = abs(value - bins[0])
    for i, b in enumerate(bins[1:], 1):
        dist = abs(value - b)
        if dist < best_dist:
            best_dist = dist
            best_idx = i
    return best_idx


def _quantize_angle(angle: int) -> int:
    """Quantize angle offset to nearest 15° bin."""
    return _quantize_to_bin(float(angle), [float(b) for b in _ANGLE_BINS])


def _clamp(value: int, lo: int, hi: int) -> int:
    """Clamp value to [lo, hi] inclusive."""
    return max(lo, min(hi, value))


def _dequantize_angle(bin_idx: int) -> int:
    return _ANGLE_BINS[_clamp(bin_idx, 0, ANGLE_OFFSET_COUNT - 1)]


def _quantize_mu(mu: float) -> int:
    return _quantize_to_bin(mu, _MU_BINS)


def _dequantize_mu(bin_idx: int) -> float:
    return _MU_BINS[_clamp(bin_idx, 0, MU_COUNT - 1)]


def _quantize_squish(squish: float) -> int:
    return _quantize_to_bin(squish, _SQUISH_BINS)


def _dequantize_squish(bin_idx: int) -> float:
    return _SQUISH_BINS[_clamp(bin_idx, 0, SQUISH_COUNT - 1)]


def _quantize_dur_frac(frac: float) -> int:
    return _quantize_to_bin(frac, _DUR_FRAC_BINS)


def _dequantize_dur_frac(bin_idx: int) -> float:
    return _DUR_FRAC_BINS[_clamp(bin_idx, 0, DUR_FRAC_COUNT - 1)]


# ---------------------------------------------------------------------------
# BeatmapTokenizer
# ---------------------------------------------------------------------------


class BeatmapTokenizer:
    """Tokenizer for Beat Saber note events.

    Encodes beatmap objects into integer token sequences and decodes
    token sequences back to beatmap objects. Supports round-trip
    encode -> decode without information loss (within quantization limits).
    """

    def __init__(self) -> None:
        self.vocab_size = VOCAB_SIZE
        self.pad_token = PAD
        self.eos_token = EOS
        self.sep_token = SEP
        self.bos_token = BOS

    def encode_beatmap(
        self,
        beatmap: DifficultyBeatmap,
    ) -> dict[float, list[int]]:
        """Encode a full difficulty beatmap into per-beat token sequences.

        Sliders are split into ARC_START at the head beat and ARC_END at
        the tail beat. Multiple events at the same beat are SEP-separated
        in canonical order, ending with EOS.

        Args:
            beatmap: Parsed difficulty beatmap data.

        Returns:
            Dict mapping beat (float) -> list of tokens for that beat.
        """
        # Collect all events grouped by beat
        beat_events: dict[float, list[_TokenEvent]] = defaultdict(list)

        for n in beatmap.color_notes:
            tokens = [
                NOTE,
                COLOR_OFFSET + n.color,
                COL_OFFSET + n.x,
                ROW_OFFSET + n.y,
                DIR_OFFSET + n.direction,
                ANGLE_OFFSET_OFFSET + _quantize_angle(n.angle_offset),
            ]
            beat_events[n.beat].append(_TokenEvent(NOTE, n.x, n.y, tokens))

        for b in beatmap.bomb_notes:
            tokens = [
                BOMB,
                COL_OFFSET + b.x,
                ROW_OFFSET + b.y,
            ]
            beat_events[b.beat].append(_TokenEvent(BOMB, b.x, b.y, tokens))

        for o in beatmap.obstacles:
            dur_int = min(int(o.duration), 64)
            dur_frac = o.duration - dur_int
            tokens = [
                WALL,
                COL_OFFSET + o.x,
                ROW_OFFSET + o.y,
                WIDTH_OFFSET + (o.width - 1),
                HEIGHT_OFFSET + (o.height - 1),
                DUR_INT_OFFSET + dur_int,
                DUR_FRAC_OFFSET + _quantize_dur_frac(dur_frac),
            ]
            beat_events[o.beat].append(_TokenEvent(WALL, o.x, o.y, tokens))

        for s in beatmap.sliders:
            # ARC_START at head beat
            head_tokens = [
                ARC_START,
                COLOR_OFFSET + s.color,
                COL_OFFSET + s.x,
                ROW_OFFSET + s.y,
                DIR_OFFSET + s.direction,
                MU_OFFSET + _quantize_mu(s.mu),
            ]
            beat_events[s.beat].append(_TokenEvent(ARC_START, s.x, s.y, head_tokens))

            # ARC_END at tail beat
            tail_tokens = [
                ARC_END,
                COLOR_OFFSET + s.color,
                COL_OFFSET + s.tail_x,
                ROW_OFFSET + s.tail_y,
                DIR_OFFSET + s.tail_direction,
                MU_OFFSET + _quantize_mu(s.tail_mu),
                MID_ANCHOR_OFFSET + s.mid_anchor_mode,
            ]
            beat_events[s.tail_beat].append(_TokenEvent(ARC_END, s.tail_x, s.tail_y, tail_tokens))

        for bs in beatmap.burst_sliders:
            sc = max(_SLICE_MIN, min(_SLICE_MAX, bs.slice_count))
            tokens = [
                CHAIN,
                COLOR_OFFSET + bs.color,
                COL_OFFSET + bs.x,
                ROW_OFFSET + bs.y,
                DIR_OFFSET + bs.direction,
                COL_OFFSET + bs.tail_x,
                ROW_OFFSET + bs.tail_y,
                SLICE_OFFSET + (sc - _SLICE_MIN),
                SQUISH_OFFSET + _quantize_squish(bs.squish),
            ]
            beat_events[bs.beat].append(_TokenEvent(CHAIN, bs.x, bs.y, tokens))

        # Sort events within each beat: type priority, then left-to-right, bottom-to-top
        result: dict[float, list[int]] = {}
        for beat in sorted(beat_events.keys()):
            events = beat_events[beat]
            events.sort(key=lambda e: (_TYPE_PRIORITY.get(e.type_token, 99), e.x, e.y))
            token_list: list[int] = []
            for i, ev in enumerate(events):
                if i > 0:
                    token_list.append(SEP)
                token_list.extend(ev.tokens)
            token_list.append(EOS)
            result[beat] = token_list

        return result

    def decode_beatmap(
        self,
        beat_tokens: dict[float, list[int]],
    ) -> DifficultyBeatmap:
        """Decode per-beat token sequences back to beatmap objects.

        ARC_START and ARC_END tokens are matched by color to reconstruct
        Slider objects. Unmatched arc ends are discarded.

        Args:
            beat_tokens: Dict mapping beat (float) -> list of tokens.

        Returns:
            DifficultyBeatmap with reconstructed objects.
        """
        color_notes: list[ColorNote] = []
        bomb_notes: list[BombNote] = []
        obstacles: list[Obstacle] = []
        burst_sliders: list[BurstSlider] = []

        # Collect arc starts/ends for matching
        arc_starts: list[tuple[float, int, int, int, int, float]] = []  # beat, color, x, y, dir, mu
        arc_ends: list[
            tuple[float, int, int, int, int, float, int]
        ] = []  # beat, color, x, y, dir, mu, mid

        for beat in sorted(beat_tokens.keys()):
            tokens = beat_tokens[beat]
            pos = 0
            while pos < len(tokens):
                if tokens[pos] == EOS:
                    break
                if tokens[pos] == SEP:
                    pos += 1
                    continue

                event_type = tokens[pos]
                remaining = len(tokens) - pos
                if event_type == NOTE:
                    if remaining < 6:
                        break
                    color_notes.append(
                        ColorNote(
                            beat=beat,
                            color=tokens[pos + 1] - COLOR_OFFSET,
                            x=tokens[pos + 2] - COL_OFFSET,
                            y=tokens[pos + 3] - ROW_OFFSET,
                            direction=tokens[pos + 4] - DIR_OFFSET,
                            angle_offset=_dequantize_angle(tokens[pos + 5] - ANGLE_OFFSET_OFFSET),
                        )
                    )
                    pos += 6
                elif event_type == BOMB:
                    if remaining < 3:
                        break
                    bomb_notes.append(
                        BombNote(
                            beat=beat,
                            x=tokens[pos + 1] - COL_OFFSET,
                            y=tokens[pos + 2] - ROW_OFFSET,
                        )
                    )
                    pos += 3
                elif event_type == WALL:
                    if remaining < 7:
                        break
                    dur_int = tokens[pos + 5] - DUR_INT_OFFSET
                    dur_frac = _dequantize_dur_frac(tokens[pos + 6] - DUR_FRAC_OFFSET)
                    obstacles.append(
                        Obstacle(
                            beat=beat,
                            x=tokens[pos + 1] - COL_OFFSET,
                            y=tokens[pos + 2] - ROW_OFFSET,
                            width=(tokens[pos + 3] - WIDTH_OFFSET) + 1,
                            height=(tokens[pos + 4] - HEIGHT_OFFSET) + 1,
                            duration=dur_int + dur_frac,
                        )
                    )
                    pos += 7
                elif event_type == ARC_START:
                    if remaining < 6:
                        break
                    arc_starts.append(
                        (
                            beat,
                            tokens[pos + 1] - COLOR_OFFSET,
                            tokens[pos + 2] - COL_OFFSET,
                            tokens[pos + 3] - ROW_OFFSET,
                            tokens[pos + 4] - DIR_OFFSET,
                            _dequantize_mu(tokens[pos + 5] - MU_OFFSET),
                        )
                    )
                    pos += 6
                elif event_type == ARC_END:
                    if remaining < 7:
                        break
                    arc_ends.append(
                        (
                            beat,
                            tokens[pos + 1] - COLOR_OFFSET,
                            tokens[pos + 2] - COL_OFFSET,
                            tokens[pos + 3] - ROW_OFFSET,
                            tokens[pos + 4] - DIR_OFFSET,
                            _dequantize_mu(tokens[pos + 5] - MU_OFFSET),
                            tokens[pos + 6] - MID_ANCHOR_OFFSET,
                        )
                    )
                    pos += 7
                elif event_type == CHAIN:
                    if remaining < 9:
                        break
                    burst_sliders.append(
                        BurstSlider(
                            beat=beat,
                            color=tokens[pos + 1] - COLOR_OFFSET,
                            x=tokens[pos + 2] - COL_OFFSET,
                            y=tokens[pos + 3] - ROW_OFFSET,
                            direction=tokens[pos + 4] - DIR_OFFSET,
                            tail_x=tokens[pos + 5] - COL_OFFSET,
                            tail_y=tokens[pos + 6] - ROW_OFFSET,
                            tail_beat=beat,  # chains don't store separate tail beat in tokens
                            slice_count=(tokens[pos + 7] - SLICE_OFFSET) + _SLICE_MIN,
                            squish=_dequantize_squish(tokens[pos + 8] - SQUISH_OFFSET),
                        )
                    )
                    pos += 9
                else:
                    # Unknown token, skip
                    pos += 1

        # Match arc starts with arc ends by color (FIFO per color)
        sliders: list[Slider] = []
        end_by_color: dict[int, list[tuple[float, int, int, int, float, int]]] = defaultdict(list)
        for end in arc_ends:
            end_by_color[end[1]].append((end[0], end[2], end[3], end[4], end[5], end[6]))

        for start in arc_starts:
            s_beat, s_color, s_x, s_y, s_dir, s_mu = start
            ends = end_by_color.get(s_color, [])
            if ends:
                e_beat, e_x, e_y, e_dir, e_mu, e_mid = ends.pop(0)
                sliders.append(
                    Slider(
                        color=s_color,
                        beat=s_beat,
                        x=s_x,
                        y=s_y,
                        direction=s_dir,
                        mu=s_mu,
                        tail_beat=e_beat,
                        tail_x=e_x,
                        tail_y=e_y,
                        tail_direction=e_dir,
                        tail_mu=e_mu,
                        mid_anchor_mode=e_mid,
                    )
                )

        return DifficultyBeatmap(
            version="3.3.0",
            color_notes=color_notes,
            bomb_notes=bomb_notes,
            obstacles=obstacles,
            sliders=sliders,
            burst_sliders=burst_sliders,
        )

    @property
    def special_tokens(self) -> dict[str, int]:
        """Return mapping of special token names to IDs."""
        return {"PAD": PAD, "EOS": EOS, "SEP": SEP, "BOS": BOS}

    @property
    def event_tokens(self) -> dict[str, int]:
        """Return mapping of event type names to IDs."""
        return {
            "NOTE": NOTE,
            "BOMB": BOMB,
            "WALL": WALL,
            "ARC_START": ARC_START,
            "ARC_END": ARC_END,
            "CHAIN": CHAIN,
        }


# ---------------------------------------------------------------------------
# Convenience
# ---------------------------------------------------------------------------

# Difficulty name -> integer mapping for embedding layers
DIFFICULTY_MAP: dict[str, int] = {
    "Easy": 0,
    "Normal": 1,
    "Hard": 2,
    "Expert": 3,
    "ExpertPlus": 4,
}


# ---------------------------------------------------------------------------
# Lighting token vocabulary
# ---------------------------------------------------------------------------
# Shared special tokens (same IDs as note tokenizer for consistency)
LIGHT_PAD = 0
LIGHT_EOS = 1
LIGHT_SEP = 2
LIGHT_BOS = 3

# Event-type tokens
LIGHT_BASIC = 4  # basicBeatmapEvent: followed by [ET][VAL][BRIGHT]
LIGHT_BOOST = 5  # colorBoostBeatmapEvent: followed by [ONOFF]

# ET (event type): Beat Saber event types 0-14
LIGHT_ET_OFFSET = 6
LIGHT_ET_COUNT = 15  # tokens 6-20
_LIGHT_ET_MAX = LIGHT_ET_COUNT - 1

# VAL (event value): 0-7
# 0=off, 1=blue-on, 2=blue-flash, 3=blue-fade, 5=red-on, 6=red-flash, 7=red-fade
LIGHT_VAL_OFFSET = LIGHT_ET_OFFSET + LIGHT_ET_COUNT  # 21
LIGHT_VAL_COUNT = 8  # tokens 21-28

# BRIGHT (brightness / float_value): 4 bins
LIGHT_BRIGHT_OFFSET = LIGHT_VAL_OFFSET + LIGHT_VAL_COUNT  # 29
LIGHT_BRIGHT_COUNT = 4  # tokens 29-32
_LIGHT_BRIGHT_BINS: list[float] = [0.0, 0.33, 0.67, 1.0]

# ONOFF (boost on/off): 2 values
LIGHT_ONOFF_OFFSET = LIGHT_BRIGHT_OFFSET + LIGHT_BRIGHT_COUNT  # 33
LIGHT_ONOFF_COUNT = 2  # tokens 33-34

LIGHT_VOCAB_SIZE = LIGHT_ONOFF_OFFSET + LIGHT_ONOFF_COUNT  # 35


def _quantize_brightness(value: float) -> int:
    """Quantize float brightness to nearest bin index."""
    return _quantize_to_bin(value, _LIGHT_BRIGHT_BINS)


def _dequantize_brightness(bin_idx: int) -> float:
    return _LIGHT_BRIGHT_BINS[_clamp(bin_idx, 0, LIGHT_BRIGHT_COUNT - 1)]


class LightingTokenizer:
    """Tokenizer for Beat Saber lighting events.

    Encodes basicBeatmapEvents and colorBoostBeatmapEvents into per-beat
    token sequences and decodes them back to event objects.

    Per-beat encoding:
        BasicEvent:      [LIGHT_BASIC][ET_TOKEN][VAL_TOKEN][BRIGHT_TOKEN]
        ColorBoostEvent: [LIGHT_BOOST][ONOFF_TOKEN]
    Multiple events at the same beat are SEP-separated, ending with LIGHT_EOS.
    """

    def __init__(self) -> None:
        self.vocab_size = LIGHT_VOCAB_SIZE
        self.pad_token = LIGHT_PAD
        self.eos_token = LIGHT_EOS
        self.sep_token = LIGHT_SEP
        self.bos_token = LIGHT_BOS

    def encode_lighting(
        self,
        beatmap: DifficultyBeatmap,  # noqa: F821
    ) -> dict[float, list[int]]:
        """Encode lighting events from a beatmap into per-beat token sequences.

        Args:
            beatmap: Parsed difficulty beatmap containing basic_events and
                color_boost_events.

        Returns:
            Dict mapping beat (float) -> list of tokens, ending with LIGHT_EOS.
        """
        from collections import defaultdict as _defaultdict

        beat_events: dict[float, list[list[int]]] = _defaultdict(list)

        for e in beatmap.basic_events:
            et = _clamp(e.event_type, 0, _LIGHT_ET_MAX)
            val = _clamp(e.value, 0, LIGHT_VAL_COUNT - 1)
            bright = _quantize_brightness(e.float_value)
            beat_events[e.beat].append(
                [LIGHT_BASIC, LIGHT_ET_OFFSET + et, LIGHT_VAL_OFFSET + val,
                 LIGHT_BRIGHT_OFFSET + bright]
            )

        for e in beatmap.color_boost_events:
            onoff = 1 if e.boost else 0
            beat_events[e.beat].append([LIGHT_BOOST, LIGHT_ONOFF_OFFSET + onoff])

        result: dict[float, list[int]] = {}
        for beat in sorted(beat_events.keys()):
            events = beat_events[beat]
            token_list: list[int] = []
            for i, ev_tokens in enumerate(events):
                if i > 0:
                    token_list.append(LIGHT_SEP)
                token_list.extend(ev_tokens)
            token_list.append(LIGHT_EOS)
            result[beat] = token_list

        return result

    def decode_lighting(
        self,
        beat_tokens: dict[float, list[int]],
    ) -> tuple[list[BasicEvent], list[ColorBoostEvent]]:  # noqa: F821
        """Decode per-beat token sequences back to lighting event objects.

        Args:
            beat_tokens: Dict mapping beat (float) -> list of tokens.

        Returns:
            Tuple of (basic_events, color_boost_events).
        """
        from beatsaber_automapper.data.beatmap import BasicEvent, ColorBoostEvent

        basic_events: list[BasicEvent] = []
        color_boost_events: list[ColorBoostEvent] = []

        for beat in sorted(beat_tokens.keys()):
            tokens = beat_tokens[beat]
            pos = 0
            while pos < len(tokens):
                if tokens[pos] == LIGHT_EOS:
                    break
                if tokens[pos] == LIGHT_SEP:
                    pos += 1
                    continue

                event_type = tokens[pos]
                remaining = len(tokens) - pos

                if event_type == LIGHT_BASIC:
                    if remaining < 4:
                        break
                    et = _clamp(tokens[pos + 1] - LIGHT_ET_OFFSET, 0, _LIGHT_ET_MAX)
                    val = _clamp(tokens[pos + 2] - LIGHT_VAL_OFFSET, 0, LIGHT_VAL_COUNT - 1)
                    bright_idx = _clamp(
                        tokens[pos + 3] - LIGHT_BRIGHT_OFFSET, 0, LIGHT_BRIGHT_COUNT - 1
                    )
                    basic_events.append(
                        BasicEvent(
                            beat=beat,
                            event_type=et,
                            value=val,
                            float_value=_dequantize_brightness(bright_idx),
                        )
                    )
                    pos += 4
                elif event_type == LIGHT_BOOST:
                    if remaining < 2:
                        break
                    onoff = _clamp(tokens[pos + 1] - LIGHT_ONOFF_OFFSET, 0, 1)
                    color_boost_events.append(
                        ColorBoostEvent(beat=beat, boost=bool(onoff))
                    )
                    pos += 2
                else:
                    pos += 1

        return basic_events, color_boost_events
