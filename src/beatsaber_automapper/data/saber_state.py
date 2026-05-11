"""V6 Saber-state extractor.

Computes the 12-dim physical saber state at every position in a swing-event
stream. State at position i represents the physical configuration of both
sabers BEFORE event i is emitted — i.e., it's derived from events 0..i-1.

State vector layout (12 floats):
    [0]  L_x          left saber last grid column, normalized (x / 3.0)
    [1]  L_y          left saber last grid row, normalized (y / 2.0)
    [2]  L_dx         left saber last swing unit vector, x component
    [3]  L_dy         left saber last swing unit vector, y component
    [4]  L_dt         beats since last left swing (log-normalized, clamped)
    [5]  L_parity     left swing parity (+1=backhand/up, -1=forehand/down, 0=neutral)
    [6]  R_x          right saber last grid column, normalized
    [7]  R_y          right saber last grid row, normalized
    [8]  R_dx         right saber last swing unit vector, x component
    [9]  R_dy         right saber last swing unit vector, y component
    [10] R_dt         beats since last right swing (log-normalized, clamped)
    [11] R_parity     right swing parity

Parity convention (matching the playability checker):
    Forehand (natural downward swings): directions 1, 6, 7 → parity = -1
    Backhand (upward swings):           directions 0, 4, 5 → parity = +1
    Neutral (left/right/any):           directions 2, 3, 8 → parity = 0

Parity resets to 0 when dt > PARITY_RESET_BEATS (3.0) — a long gap means the
player has reset their saber to a natural resting position.

L_dt and R_dt are log-normalized: log(1 + dt) / log(1 + MAX_DT), clamped to [0, 1].
Initial state (before any events): all zeros.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch

from beatsaber_automapper.data.swing_tokenizer import (
    ARC_HEAD,
    ARC_TAIL,
    CHAIN_HEAD,
    CHAIN_TAIL,
    HAND_LEFT,
    HAND_RIGHT,
    NOTE,
    _SwingEvent,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# 9-direction unit vectors [x_right_positive, y_up_positive] indexed by direction id.
# 0=up 1=down 2=left 3=right 4=up-left 5=up-right 6=down-left 7=down-right 8=any
_DIR_UNIT: list[tuple[float, float]] = [
    (0.0,   1.0),    # 0 up
    (0.0,  -1.0),    # 1 down
    (-1.0,  0.0),    # 2 left
    (1.0,   0.0),    # 3 right
    (-0.707, 0.707), # 4 up-left
    (0.707,  0.707), # 5 up-right
    (-0.707,-0.707), # 6 down-left
    (0.707, -0.707), # 7 down-right
    (0.0,   0.0),    # 8 any — neutral
]

# Directions that update saber state with meaningful parity info
_FOREHAND_DIRS = frozenset({1, 6, 7})  # downward swings
_BACKHAND_DIRS = frozenset({0, 4, 5})  # upward swings
_NEUTRAL_DIRS  = frozenset({2, 3, 8})  # lateral or dot — don't update parity

# Saber kinds that carry meaningful swing information (not bombs)
_SWING_KINDS = frozenset({NOTE, ARC_HEAD, ARC_TAIL, CHAIN_HEAD, CHAIN_TAIL})

# After this many beats without a swing, parity resets to 0
PARITY_RESET_BEATS = 3.0

# Log-normalization ceiling for dt
MAX_DT = 32.0
_LOG_DENOM = math.log(1.0 + MAX_DT)


def _log_norm_dt(dt: float) -> float:
    return math.log(1.0 + min(dt, MAX_DT)) / _LOG_DENOM


def _parity(direction: int) -> float:
    if direction in _FOREHAND_DIRS:
        return -1.0
    if direction in _BACKHAND_DIRS:
        return 1.0
    return 0.0


# ---------------------------------------------------------------------------
# Core function
# ---------------------------------------------------------------------------


def compute_saber_states(
    events: Sequence[_SwingEvent],
) -> torch.Tensor:
    """Compute saber state at every event position in a swing-event stream.

    State at index i represents the physical configuration BEFORE event i.
    State at index 0 is all zeros (no prior events).

    Args:
        events: Ordered list of _SwingEvent objects (from SwingEventTokenizer.decode_events
                or built during encode_beatmap). Must be sorted by beat ascending.

    Returns:
        Float tensor of shape [N, 12] where N = len(events). Each row is the
        12-dim saber state vector before that event is processed.
    """
    n = len(events)
    out = torch.zeros(n, 12, dtype=torch.float32)

    # Per-hand mutable state
    state = {
        HAND_LEFT:  _HandState(),
        HAND_RIGHT: _HandState(),
    }

    for i, evt in enumerate(events):
        # Write state BEFORE this event
        ls = state[HAND_LEFT]
        rs = state[HAND_RIGHT]

        l_dt_since = evt.beat - ls.last_beat if ls.last_beat >= 0 else MAX_DT
        r_dt_since = evt.beat - rs.last_beat if rs.last_beat >= 0 else MAX_DT

        l_parity = 0.0 if l_dt_since > PARITY_RESET_BEATS else ls.parity
        r_parity = 0.0 if r_dt_since > PARITY_RESET_BEATS else rs.parity

        out[i, 0] = ls.x / 3.0
        out[i, 1] = ls.y / 2.0
        out[i, 2] = ls.dx
        out[i, 3] = ls.dy
        out[i, 4] = _log_norm_dt(l_dt_since)
        out[i, 5] = l_parity
        out[i, 6] = rs.x / 3.0
        out[i, 7] = rs.y / 2.0
        out[i, 8] = rs.dx
        out[i, 9] = rs.dy
        out[i, 10] = _log_norm_dt(r_dt_since)
        out[i, 11] = r_parity

        # Update state for this event (only for swing kinds, not bombs)
        if evt.kind in _SWING_KINDS and evt.hand in (HAND_LEFT, HAND_RIGHT):
            h = state[evt.hand]
            h.x = evt.x
            h.y = evt.y
            dx, dy = _DIR_UNIT[min(evt.direction, 8)]
            h.dx = dx
            h.dy = dy
            h.last_beat = evt.beat
            # Neutral directions don't update parity
            p = _parity(evt.direction)
            if p != 0.0:
                h.parity = p

    return out


# ---------------------------------------------------------------------------
# Internal mutable state per hand
# ---------------------------------------------------------------------------


class _HandState:
    """Mutable saber state for one hand."""

    __slots__ = ("x", "y", "dx", "dy", "last_beat", "parity")

    def __init__(self) -> None:
        self.x: float = 1.5   # grid centre-left
        self.y: float = 0.0   # ground row
        self.dx: float = 0.0
        self.dy: float = 0.0
        self.last_beat: float = -MAX_DT  # sentinel: no prior event
        self.parity: float = 0.0


# ---------------------------------------------------------------------------
# Convenience: compute from raw beatmap (no pre-encoded events needed)
# ---------------------------------------------------------------------------


def compute_saber_states_from_beatmap(
    beatmap,  # DifficultyBeatmap — avoid circular import typing
) -> tuple[list[_SwingEvent], torch.Tensor]:
    """Encode a beatmap to swing events and compute saber states in one step.

    Args:
        beatmap: DifficultyBeatmap.

    Returns:
        Tuple of (events, saber_states) where saber_states has shape [N, 12].
    """
    from beatsaber_automapper.data.swing_tokenizer import SwingEventTokenizer
    tok = SwingEventTokenizer()
    tokens = tok.encode_beatmap(beatmap)
    events = tok.decode_events(tokens)
    states = compute_saber_states(events)
    return events, states
