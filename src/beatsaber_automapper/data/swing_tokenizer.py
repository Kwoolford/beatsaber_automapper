"""V6 Swing-Event Tokenizer for Stage 2.

Replaces the V5 chord-at-timestamp grammar with a single globally-ordered
stream of per-hand swing events. Parity is structural: consecutive same-hand
events alternate forehand/backhand by construction of the data.

Grammar (see docs/swing_event_grammar.md for the locked spec):

  SWING event (7 tokens):       [HAND] [Δt] [KIND] [X] [Y] [DIR] [FIELD_D]
  CHAIN_TAIL event (6 tokens):  [HAND] [Δt] [CHAIN_TAIL] [X] [Y] [SQUISH]
  BOMB event (5 tokens):        [HAND=NONE] [Δt] [BOMB] [X] [Y]

Walls are excluded from the stream and handled by rule-based postprocessing.
Arc mid_anchor_mode is always decoded as 0 (rare in practice, saves a token).

Vocab size: 118 tokens.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from beatsaber_automapper.data.beatmap import (
    BombNote,
    BurstSlider,
    ColorNote,
    DifficultyBeatmap,
    Slider,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Vocabulary constants (locked — do not change without full re-preprocess)
# ---------------------------------------------------------------------------

PAD = 0
BOS = 1
EOS = 2

# HAND tokens
HAND_LEFT = 3   # red saber, color=0
HAND_RIGHT = 4  # blue saber, color=1
HAND_NONE = 5   # bombs

# Δt bins (32 values, IDs 6-37)
DT_BASE = 6
DT_COUNT = 32
_DT_BINS: list[float] = [
    0.0,    0.0625, 0.125,  0.1875,
    0.25,   0.3125, 0.375,  0.4375,
    0.5,    0.5625, 0.625,  0.6875,
    0.75,   0.8125, 0.875,  0.9375,
    1.0,    1.5,    2.0,    2.5,
    3.0,    3.5,    4.0,    4.5,
    5.0,    6.0,    7.0,    8.0,
    12.0,   16.0,   32.0,   64.0,
]
assert len(_DT_BINS) == DT_COUNT

# KIND tokens (IDs 38-43)
KIND_BASE = 38
NOTE = KIND_BASE          # 38
ARC_HEAD = KIND_BASE + 1  # 39
ARC_TAIL = KIND_BASE + 2  # 40
CHAIN_HEAD = KIND_BASE + 3  # 41
CHAIN_TAIL = KIND_BASE + 4  # 42
BOMB = KIND_BASE + 5      # 43
KIND_COUNT = 6

# Grid X (IDs 44-47)
X_BASE = 44
X_COUNT = 4

# Grid Y (IDs 48-50)
Y_BASE = 48
Y_COUNT = 3

# Direction (IDs 51-59)
DIR_BASE = 51
DIR_COUNT = 9

# Angle offset bins (IDs 60-66), 7 bins at -45° to +45°
ANGLE_BASE = 60
ANGLE_COUNT = 7
_ANGLE_BINS: list[float] = [-45.0, -30.0, -15.0, 0.0, 15.0, 30.0, 45.0]
assert len(_ANGLE_BINS) == ANGLE_COUNT

# Arc curvature bins (IDs 67-75), 9 bins 0.0-2.0
MU_BASE = 67
MU_COUNT = 9
_MU_BINS: list[float] = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
assert len(_MU_BINS) == MU_COUNT

# Chain slice count bins (IDs 76-106), slice counts 2-32
SLICE_BASE = 76
SLICE_COUNT = 31
_SLICE_MIN = 2
_SLICE_MAX = 32

# Chain squish factor bins (IDs 107-117), 11 bins 0.0-1.0
SQUISH_BASE = 107
SQUISH_COUNT = 11
_SQUISH_BINS: list[float] = [round(i * 0.1, 1) for i in range(11)]
assert len(_SQUISH_BINS) == SQUISH_COUNT

VOCAB_SIZE = SQUISH_BASE + SQUISH_COUNT  # 118

# Full event length in tokens (HAND + Δt + KIND + fields).
# Used by grammar-constrained decoding and offset arithmetic.
EVENT_LENGTHS: dict[int, int] = {
    NOTE: 7,
    ARC_HEAD: 7,
    ARC_TAIL: 7,
    CHAIN_HEAD: 7,
    CHAIN_TAIL: 6,
    BOMB: 5,
}

# Tokens consumed starting from the KIND token (excluding HAND and Δt prefix).
KIND_LENGTHS: dict[int, int] = {k: v - 2 for k, v in EVENT_LENGTHS.items()}
# i.e.: NOTE/ARC_HEAD/ARC_TAIL/CHAIN_HEAD = 5, CHAIN_TAIL = 4, BOMB = 3


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _nearest_bin(value: float, bins: list[float]) -> int:
    """Return index of the nearest bin to value."""
    best = 0
    best_dist = abs(value - bins[0])
    for i, b in enumerate(bins[1:], 1):
        d = abs(value - b)
        if d < best_dist:
            best_dist = d
            best = i
    return best


def _clamp(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, v))


def _hand_from_color(color: int) -> int:
    return HAND_LEFT if color == 0 else HAND_RIGHT


def _color_from_hand(hand: int) -> int:
    return 0 if hand == HAND_LEFT else 1


# ---------------------------------------------------------------------------
# Internal event representation for sorting
# ---------------------------------------------------------------------------

# Kind-priority order for events at the same beat (lower = earlier)
_KIND_PRIORITY = {NOTE: 0, ARC_HEAD: 1, CHAIN_HEAD: 2, ARC_TAIL: 3, CHAIN_TAIL: 4, BOMB: 5}


@dataclass
class _SwingEvent:
    """Intermediate representation before tokenization."""
    beat: float
    hand: int         # HAND_LEFT / HAND_RIGHT / HAND_NONE
    kind: int         # NOTE / ARC_HEAD / … / BOMB
    x: int
    y: int
    direction: int    # 0-8; ignored for BOMB
    field_d: int      # ANGLE_bin / MU_bin / SLICE_bin / SQUISH_bin; ignored for BOMB/CHAIN_TAIL
    tail_x: int = 0  # CHAIN_TAIL tail X
    tail_y: int = 0  # CHAIN_TAIL tail Y

    def sort_key(self) -> tuple[float, int, int]:
        return (self.beat, _KIND_PRIORITY.get(self.kind, 99), self.hand)


# ---------------------------------------------------------------------------
# SwingEventTokenizer
# ---------------------------------------------------------------------------


class SwingEventTokenizer:
    """V6 per-hand swing-event tokenizer.

    Encodes a DifficultyBeatmap into a flat integer token list, and decodes
    a token list back to a DifficultyBeatmap.

    Grammar:
        SWING event (7 tokens): [HAND][Δt][KIND][X][Y][DIR][FIELD_D]
        CHAIN_TAIL (6 tokens):  [HAND][Δt][CHAIN_TAIL][X][Y][SQUISH]
        BOMB (5 tokens):        [HAND=NONE][Δt][BOMB][X][Y]

    Walls are excluded; handled by rule-based postprocessing.
    """

    vocab_size: int = VOCAB_SIZE
    pad_token: int = PAD
    bos_token: int = BOS
    eos_token: int = EOS

    def encode_beatmap(self, beatmap: DifficultyBeatmap) -> list[int]:
        """Encode a full beatmap into a flat swing-event token stream.

        The stream is bookended with BOS … EOS. Walls and lighting events
        are silently skipped.

        Args:
            beatmap: Parsed v3 difficulty beatmap.

        Returns:
            Flat list of integer token IDs starting with BOS, ending with EOS.
        """
        events: list[_SwingEvent] = []

        for n in beatmap.color_notes:
            events.append(_SwingEvent(
                beat=n.beat,
                hand=_hand_from_color(n.color),
                kind=NOTE,
                x=_clamp(n.x, 0, 3),
                y=_clamp(n.y, 0, 2),
                direction=_clamp(n.direction, 0, 8),
                field_d=_nearest_bin(float(n.angle_offset), _ANGLE_BINS),
            ))

        for s in beatmap.sliders:
            events.append(_SwingEvent(
                beat=s.beat,
                hand=_hand_from_color(s.color),
                kind=ARC_HEAD,
                x=_clamp(s.x, 0, 3),
                y=_clamp(s.y, 0, 2),
                direction=_clamp(s.direction, 0, 8),
                field_d=_nearest_bin(s.mu, _MU_BINS),
            ))
            events.append(_SwingEvent(
                beat=s.tail_beat,
                hand=_hand_from_color(s.color),
                kind=ARC_TAIL,
                x=_clamp(s.tail_x, 0, 3),
                y=_clamp(s.tail_y, 0, 2),
                direction=_clamp(s.tail_direction, 0, 8),
                field_d=_nearest_bin(s.tail_mu, _MU_BINS),
            ))

        for bs in beatmap.burst_sliders:
            sc = _clamp(bs.slice_count, _SLICE_MIN, _SLICE_MAX)
            events.append(_SwingEvent(
                beat=bs.beat,
                hand=_hand_from_color(bs.color),
                kind=CHAIN_HEAD,
                x=_clamp(bs.x, 0, 3),
                y=_clamp(bs.y, 0, 2),
                direction=_clamp(bs.direction, 0, 8),
                field_d=sc - _SLICE_MIN,  # bin index; token ID = SLICE_BASE + field_d
            ))
            events.append(_SwingEvent(
                beat=bs.tail_beat,
                hand=_hand_from_color(bs.color),
                kind=CHAIN_TAIL,
                x=_clamp(bs.tail_x, 0, 3),
                y=_clamp(bs.tail_y, 0, 2),
                direction=0,  # unused for chain tail
                field_d=_nearest_bin(bs.squish, _SQUISH_BINS),
                tail_x=_clamp(bs.tail_x, 0, 3),
                tail_y=_clamp(bs.tail_y, 0, 2),
            ))

        for b in beatmap.bomb_notes:
            events.append(_SwingEvent(
                beat=b.beat,
                hand=HAND_NONE,
                kind=BOMB,
                x=_clamp(b.x, 0, 3),
                y=_clamp(b.y, 0, 2),
                direction=0,
                field_d=0,
            ))

        events.sort(key=lambda e: e.sort_key())

        tokens: list[int] = [BOS]
        prev_beat = 0.0
        for evt in events:
            dt = max(0.0, evt.beat - prev_beat)
            dt_bin = _nearest_bin(dt, _DT_BINS)
            prev_beat = evt.beat

            tokens.append(evt.hand)
            tokens.append(DT_BASE + dt_bin)
            tokens.append(evt.kind)
            tokens.append(X_BASE + evt.x)
            tokens.append(Y_BASE + evt.y)

            if evt.kind == BOMB:
                pass  # 5-token event ends here
            elif evt.kind == CHAIN_TAIL:
                tokens.append(SQUISH_BASE + evt.field_d)
            else:
                # NOTE, ARC_HEAD, ARC_TAIL, CHAIN_HEAD — all 7 tokens
                tokens.append(DIR_BASE + evt.direction)
                if evt.kind == NOTE:
                    tokens.append(ANGLE_BASE + evt.field_d)
                elif evt.kind in (ARC_HEAD, ARC_TAIL):
                    tokens.append(MU_BASE + evt.field_d)
                else:  # CHAIN_HEAD
                    slice_bin = _clamp(evt.field_d, 0, SLICE_COUNT - 1)
                    tokens.append(SLICE_BASE + slice_bin)

        tokens.append(EOS)
        return tokens

    def decode_beatmap(self, tokens: list[int]) -> DifficultyBeatmap:
        """Decode a flat swing-event token stream back to a DifficultyBeatmap.

        Arc heads are matched to the next arc tail of the same hand (FIFO).
        Chain heads are matched to the next chain tail of the same hand (FIFO).
        Unmatched heads and tails are dropped.

        Args:
            tokens: Flat list of integer token IDs (may include BOS/EOS/PAD).

        Returns:
            DifficultyBeatmap with reconstructed color_notes, sliders,
            burst_sliders, and bomb_notes. Obstacles and lighting are empty.
        """
        color_notes: list[ColorNote] = []
        bomb_notes: list[BombNote] = []

        # Pending arc / chain heads awaiting their tails {hand: [(beat, x, y, dir, mu)]}
        arc_heads: dict[int, list[tuple[float, int, int, int, float]]] = {
            HAND_LEFT: [], HAND_RIGHT: [],
        }
        chain_heads: dict[int, list[tuple[float, int, int, int, int]]] = {
            HAND_LEFT: [], HAND_RIGHT: [],
        }
        sliders: list[Slider] = []
        burst_sliders: list[BurstSlider] = []

        pos = 0
        n = len(tokens)
        current_beat = 0.0

        while pos < n:
            tok = tokens[pos]

            if tok in (PAD, BOS):
                pos += 1
                continue
            if tok == EOS:
                break

            # Expect a HAND token at the start of each event
            if tok not in (HAND_LEFT, HAND_RIGHT, HAND_NONE):
                pos += 1
                continue

            hand = tok
            remaining = n - pos

            # Need at least: HAND + Δt + KIND + X + Y = 5 tokens (BOMB minimum)
            if remaining < 5:
                break

            dt_tok = tokens[pos + 1]
            if not (DT_BASE <= dt_tok < DT_BASE + DT_COUNT):
                pos += 1
                continue
            dt = _DT_BINS[dt_tok - DT_BASE]
            current_beat += dt

            kind_tok = tokens[pos + 2]
            if not (KIND_BASE <= kind_tok < KIND_BASE + KIND_COUNT):
                pos += 1
                continue

            kind = kind_tok

            x_tok = tokens[pos + 3] if pos + 3 < n else X_BASE
            y_tok = tokens[pos + 4] if pos + 4 < n else Y_BASE
            x = _clamp(x_tok - X_BASE, 0, 3)
            y = _clamp(y_tok - Y_BASE, 0, 2)

            if kind == BOMB:
                bomb_notes.append(BombNote(beat=current_beat, x=x, y=y))
                pos += 5
                continue

            if kind == CHAIN_TAIL:
                if remaining < 6:
                    pos += 1
                    continue
                sq_tok = tokens[pos + 5] if pos + 5 < n else SQUISH_BASE
                squish_bin = _clamp(sq_tok - SQUISH_BASE, 0, SQUISH_COUNT - 1)
                squish = _SQUISH_BINS[squish_bin]
                tail_x, tail_y = x, y

                if hand in (HAND_LEFT, HAND_RIGHT) and chain_heads[hand]:
                    hb, hx, hy, hdir, hsc = chain_heads[hand].pop(0)
                    burst_sliders.append(BurstSlider(
                        color=_color_from_hand(hand),
                        beat=hb,
                        x=hx,
                        y=hy,
                        direction=hdir,
                        tail_beat=current_beat,
                        tail_x=tail_x,
                        tail_y=tail_y,
                        slice_count=hsc,
                        squish=squish,
                    ))
                # else: orphan tail, drop it
                pos += 6
                continue

            # All remaining kinds need position 5 (DIR) and position 6 (FIELD_D)
            if remaining < 7:
                pos += 1
                continue

            dir_tok = tokens[pos + 5] if pos + 5 < n else DIR_BASE
            fd_tok = tokens[pos + 6] if pos + 6 < n else ANGLE_BASE
            direction = _clamp(dir_tok - DIR_BASE, 0, 8)

            if kind == NOTE:
                angle_bin = _clamp(fd_tok - ANGLE_BASE, 0, ANGLE_COUNT - 1)
                angle = int(_ANGLE_BINS[angle_bin])
                color = _color_from_hand(hand)
                color_notes.append(ColorNote(
                    beat=current_beat,
                    x=x, y=y,
                    color=color,
                    direction=direction,
                    angle_offset=angle,
                ))
                pos += 7

            elif kind == ARC_HEAD:
                mu_bin = _clamp(fd_tok - MU_BASE, 0, MU_COUNT - 1)
                mu = _MU_BINS[mu_bin]
                if hand in (HAND_LEFT, HAND_RIGHT):
                    arc_heads[hand].append((current_beat, x, y, direction, mu))
                pos += 7

            elif kind == ARC_TAIL:
                mu_bin = _clamp(fd_tok - MU_BASE, 0, MU_COUNT - 1)
                tail_mu = _MU_BINS[mu_bin]
                if hand in (HAND_LEFT, HAND_RIGHT) and arc_heads[hand]:
                    hb, hx, hy, hdir, hmu = arc_heads[hand].pop(0)
                    sliders.append(Slider(
                        color=_color_from_hand(hand),
                        beat=hb,
                        x=hx, y=hy,
                        direction=hdir,
                        mu=hmu,
                        tail_beat=current_beat,
                        tail_x=x,
                        tail_y=y,
                        tail_direction=direction,
                        tail_mu=tail_mu,
                        mid_anchor_mode=0,
                    ))
                pos += 7

            elif kind == CHAIN_HEAD:
                slice_bin = _clamp(fd_tok - SLICE_BASE, 0, SLICE_COUNT - 1)
                sc = slice_bin + _SLICE_MIN
                if hand in (HAND_LEFT, HAND_RIGHT):
                    chain_heads[hand].append((current_beat, x, y, direction, sc))
                pos += 7

            else:
                pos += 1

        return DifficultyBeatmap(
            version="3.3.0",
            color_notes=color_notes,
            bomb_notes=bomb_notes,
            obstacles=[],
            sliders=sliders,
            burst_sliders=burst_sliders,
        )

    def decode_events(self, tokens: list[int]) -> list[_SwingEvent]:
        """Decode tokens into a list of _SwingEvent structs (for analysis).

        Does not do arc/chain matching — returns raw events with beat positions.

        Args:
            tokens: Flat list of token IDs.

        Returns:
            List of _SwingEvent in stream order with absolute beat positions.
        """
        events: list[_SwingEvent] = []
        pos = 0
        n = len(tokens)
        current_beat = 0.0

        while pos < n:
            tok = tokens[pos]
            if tok in (PAD, BOS):
                pos += 1
                continue
            if tok == EOS:
                break
            if tok not in (HAND_LEFT, HAND_RIGHT, HAND_NONE):
                pos += 1
                continue

            hand = tok
            remaining = n - pos
            if remaining < 5:
                break

            dt_tok = tokens[pos + 1]
            if not (DT_BASE <= dt_tok < DT_BASE + DT_COUNT):
                pos += 1
                continue
            dt = _DT_BINS[dt_tok - DT_BASE]
            current_beat += dt

            kind_tok = tokens[pos + 2]
            if not (KIND_BASE <= kind_tok < KIND_BASE + KIND_COUNT):
                pos += 1
                continue
            kind = kind_tok

            x = _clamp((tokens[pos + 3] if pos + 3 < n else X_BASE) - X_BASE, 0, 3)
            y = _clamp((tokens[pos + 4] if pos + 4 < n else Y_BASE) - Y_BASE, 0, 2)

            if kind == BOMB:
                events.append(_SwingEvent(beat=current_beat, hand=hand, kind=BOMB,
                                          x=x, y=y, direction=0, field_d=0))
                pos += 5
            elif kind == CHAIN_TAIL:
                if remaining < 6:
                    pos += 1
                    continue
                sq_bin = _clamp((tokens[pos + 5] if pos + 5 < n else SQUISH_BASE) - SQUISH_BASE,
                                0, SQUISH_COUNT - 1)
                events.append(_SwingEvent(beat=current_beat, hand=hand, kind=CHAIN_TAIL,
                                          x=x, y=y, direction=0, field_d=sq_bin))
                pos += 6
            else:
                if remaining < 7:
                    pos += 1
                    continue
                direction = _clamp((tokens[pos + 5] if pos + 5 < n else DIR_BASE) - DIR_BASE, 0, 8)
                fd_raw = tokens[pos + 6] if pos + 6 < n else ANGLE_BASE
                if kind == NOTE:
                    fd = _clamp(fd_raw - ANGLE_BASE, 0, ANGLE_COUNT - 1)
                elif kind in (ARC_HEAD, ARC_TAIL):
                    fd = _clamp(fd_raw - MU_BASE, 0, MU_COUNT - 1)
                else:  # CHAIN_HEAD
                    fd = _clamp(fd_raw - SLICE_BASE, 0, SLICE_COUNT - 1)
                events.append(_SwingEvent(beat=current_beat, hand=hand, kind=kind,
                                          x=x, y=y, direction=direction, field_d=fd))
                pos += 7

        return events

    @staticmethod
    def event_length(kind: int) -> int:
        """Full event length in tokens (including HAND and Δt prefix)."""
        return EVENT_LENGTHS.get(kind, 9)

    @property
    def special_tokens(self) -> dict[str, int]:
        return {"PAD": PAD, "BOS": BOS, "EOS": EOS}

    @property
    def kind_tokens(self) -> dict[str, int]:
        return {
            "NOTE": NOTE, "ARC_HEAD": ARC_HEAD, "ARC_TAIL": ARC_TAIL,
            "CHAIN_HEAD": CHAIN_HEAD, "CHAIN_TAIL": CHAIN_TAIL, "BOMB": BOMB,
        }


# ---------------------------------------------------------------------------
# Convenience re-exports for direct import
# ---------------------------------------------------------------------------

__all__ = [
    "SwingEventTokenizer",
    "PAD", "BOS", "EOS",
    "HAND_LEFT", "HAND_RIGHT", "HAND_NONE",
    "DT_BASE", "DT_COUNT", "_DT_BINS",
    "NOTE", "ARC_HEAD", "ARC_TAIL", "CHAIN_HEAD", "CHAIN_TAIL", "BOMB",
    "KIND_BASE", "KIND_COUNT", "KIND_LENGTHS", "EVENT_LENGTHS",
    "X_BASE", "X_COUNT", "Y_BASE", "Y_COUNT",
    "DIR_BASE", "DIR_COUNT",
    "ANGLE_BASE", "ANGLE_COUNT", "_ANGLE_BINS",
    "MU_BASE", "MU_COUNT", "_MU_BINS",
    "SLICE_BASE", "SLICE_COUNT", "_SLICE_MIN", "_SLICE_MAX",
    "SQUISH_BASE", "SQUISH_COUNT", "_SQUISH_BINS",
    "VOCAB_SIZE",
]
