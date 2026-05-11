"""V6 grammar-constrained sequence decoder for swing-event streams.

Replaces the V5 per-onset chord grammar with a flat, globally-ordered
stream of swing events. The grammar is self-describing from the HAND token:

    event := HAND Δt KIND [X Y DIR FIELD_D | X Y SQUISH | X Y]

States cycle: EXPECT_HAND → EXPECT_DT → EXPECT_KIND → EXPECT_X → EXPECT_Y
→ [EXPECT_DIR → EXPECT_FIELD_D | EXPECT_SQUISH | (done)] → EXPECT_HAND

Saber state is maintained incrementally during decoding and passed to the
model at every step as a 12-dim conditioning signal.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum, auto

import torch

from beatsaber_automapper.data.saber_state import (
    _DIR_UNIT,
    MAX_DT,
    PARITY_RESET_BEATS,
    _log_norm_dt,
    _parity,
)
from beatsaber_automapper.data.swing_tokenizer import (
    _DT_BINS,
    ANGLE_BASE,
    ANGLE_COUNT,
    ARC_HEAD,
    ARC_TAIL,
    BOMB,
    BOS,
    CHAIN_HEAD,
    CHAIN_TAIL,
    DIR_BASE,
    DIR_COUNT,
    DT_BASE,
    DT_COUNT,
    EOS,
    HAND_LEFT,
    HAND_NONE,
    HAND_RIGHT,
    KIND_BASE,
    KIND_COUNT,
    MU_BASE,
    MU_COUNT,
    NOTE,
    PAD,
    SLICE_BASE,
    SLICE_COUNT,
    SQUISH_BASE,
    SQUISH_COUNT,
    VOCAB_SIZE,
    X_BASE,
    X_COUNT,
    Y_BASE,
    Y_COUNT,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Grammar state machine
# ---------------------------------------------------------------------------


class _Phase(Enum):
    EXPECT_HAND = auto()
    EXPECT_DT = auto()
    EXPECT_KIND = auto()
    EXPECT_X = auto()
    EXPECT_Y = auto()
    EXPECT_DIR = auto()
    EXPECT_FIELD_D = auto()
    EXPECT_SQUISH = auto()
    DONE = auto()


# Kinds where HAND_NONE is required (bombs); all others require LEFT or RIGHT
_BOMB_ONLY_HAND = frozenset({HAND_NONE})
_SWING_HANDS = frozenset({HAND_LEFT, HAND_RIGHT})

# FIELD_D token range by kind: (base, count)
_FIELD_D_RANGE: dict[int, tuple[int, int]] = {
    NOTE:       (ANGLE_BASE, ANGLE_COUNT),
    ARC_HEAD:   (MU_BASE,    MU_COUNT),
    ARC_TAIL:   (MU_BASE,    MU_COUNT),
    CHAIN_HEAD: (SLICE_BASE, SLICE_COUNT),
}

# Kinds that are 7-token events (have DIR + FIELD_D)
_SEVEN_TOKEN_KINDS = frozenset({NOTE, ARC_HEAD, ARC_TAIL, CHAIN_HEAD})

# Inf mask applied to all blocked token IDs
_NEG_INF = float("-inf")


@dataclass
class _GrammarState:
    """Per-sequence grammar state for one decoding stream."""
    phase: _Phase = _Phase.EXPECT_HAND
    current_hand: int = HAND_LEFT   # the HAND token of the event being decoded
    current_kind: int = NOTE        # the KIND token of the event being decoded
    # Saber state (12 floats): L_x, L_y, L_dx, L_dy, L_dt, L_parity, R...
    saber: list[float] = field(default_factory=lambda: [0.0] * 12)
    # Track the last beat per hand for dt computation
    last_beat: dict[int, float] = field(default_factory=lambda: {
        HAND_LEFT: -MAX_DT, HAND_RIGHT: -MAX_DT
    })
    current_beat: float = 0.0
    current_event_dt: float = 0.0   # decoded dt for the current event
    # Scratch fields for x/y/dir decoded within the current event (safe defaults)
    last_x: int = 0
    last_y: int = 0
    last_dir: int = 8  # default to "any" direction

    def saber_tensor(self, device: torch.device) -> torch.Tensor:
        t = torch.tensor(self.saber, dtype=torch.float32, device=device)
        return t.unsqueeze(0).unsqueeze(0)  # [1, 1, 12]


def _build_mask(phase: _Phase, hand: int, kind: int, vocab_size: int) -> torch.Tensor:
    """Return a logit additive mask (0 = allowed, -inf = blocked) for the given state."""
    mask = torch.full((vocab_size,), _NEG_INF)

    if phase == _Phase.EXPECT_HAND:
        mask[HAND_LEFT] = 0.0
        mask[HAND_RIGHT] = 0.0
        mask[HAND_NONE] = 0.0
        mask[EOS] = 0.0

    elif phase == _Phase.EXPECT_DT:
        for i in range(DT_COUNT):
            mask[DT_BASE + i] = 0.0

    elif phase == _Phase.EXPECT_KIND:
        if hand == HAND_NONE:
            mask[BOMB] = 0.0
        else:
            for k in range(KIND_COUNT):
                tok = KIND_BASE + k
                if tok != BOMB:
                    mask[tok] = 0.0

    elif phase == _Phase.EXPECT_X:
        for i in range(X_COUNT):
            mask[X_BASE + i] = 0.0

    elif phase == _Phase.EXPECT_Y:
        for i in range(Y_COUNT):
            mask[Y_BASE + i] = 0.0

    elif phase == _Phase.EXPECT_DIR:
        for i in range(DIR_COUNT):
            mask[DIR_BASE + i] = 0.0

    elif phase == _Phase.EXPECT_FIELD_D:
        base, count = _FIELD_D_RANGE.get(kind, (ANGLE_BASE, ANGLE_COUNT))
        for i in range(count):
            mask[base + i] = 0.0

    elif phase == _Phase.EXPECT_SQUISH:
        for i in range(SQUISH_COUNT):
            mask[SQUISH_BASE + i] = 0.0

    return mask


def _transition(state: _GrammarState, token: int) -> _GrammarState:
    """Return the next grammar state after sampling ``token``."""
    p = state.phase

    if p == _Phase.EXPECT_HAND:
        if token == EOS:
            state.phase = _Phase.DONE
        else:
            state.current_hand = token
            state.phase = _Phase.EXPECT_DT

    elif p == _Phase.EXPECT_DT:
        dt_bin = token - DT_BASE
        dt = _DT_BINS[dt_bin] if 0 <= dt_bin < DT_COUNT else 0.0
        state.current_event_dt = dt
        state.current_beat += dt
        state.phase = _Phase.EXPECT_KIND

    elif p == _Phase.EXPECT_KIND:
        state.current_kind = token
        state.phase = _Phase.EXPECT_X

    elif p == _Phase.EXPECT_X:
        state.phase = _Phase.EXPECT_Y

    elif p == _Phase.EXPECT_Y:
        kind = state.current_kind
        if kind == BOMB:
            state.phase = _Phase.EXPECT_HAND     # 5-token event done
        elif kind == CHAIN_TAIL:
            state.phase = _Phase.EXPECT_SQUISH   # 6-token event, next = squish
        else:
            state.phase = _Phase.EXPECT_DIR      # 7-token event, next = dir

    elif p == _Phase.EXPECT_DIR:
        state.phase = _Phase.EXPECT_FIELD_D

    elif p == _Phase.EXPECT_FIELD_D:
        # 7-token event complete — update saber state
        _update_saber(state, token)
        state.phase = _Phase.EXPECT_HAND

    elif p == _Phase.EXPECT_SQUISH:
        # 6-token (chain tail) event complete — update saber state (position only, not direction)
        _update_saber_chain_tail(state)
        state.phase = _Phase.EXPECT_HAND

    return state


def _update_saber(state: _GrammarState, field_d_token: int) -> None:
    """Update saber state after a 7-token swing event completes."""
    hand = state.current_hand
    if hand not in (HAND_LEFT, HAND_RIGHT):
        return

    x = state.last_x
    y = state.last_y
    direction = state.last_dir  # already clamped to [0, 8] by _record_field

    offset = 0 if hand == HAND_LEFT else 6

    state.saber[offset + 0] = x / 3.0
    state.saber[offset + 1] = y / 2.0
    dx, dy = _DIR_UNIT[direction]
    state.saber[offset + 2] = dx
    state.saber[offset + 3] = dy

    dt_since = state.current_beat - state.last_beat.get(hand, -MAX_DT)
    state.saber[offset + 4] = _log_norm_dt(dt_since)
    p = _parity(direction)
    if p != 0.0:
        state.saber[offset + 5] = p
    elif dt_since > PARITY_RESET_BEATS:
        state.saber[offset + 5] = 0.0

    state.last_beat[hand] = state.current_beat


def _update_saber_chain_tail(state: _GrammarState) -> None:
    """Update position only for chain-tail events (no direction info)."""
    hand = state.current_hand
    if hand not in (HAND_LEFT, HAND_RIGHT):
        return
    offset = 0 if hand == HAND_LEFT else 6
    state.saber[offset + 0] = state.last_x / 3.0
    state.saber[offset + 1] = state.last_y / 2.0
    dt_since = state.current_beat - state.last_beat.get(hand, -MAX_DT)
    state.saber[offset + 4] = _log_norm_dt(dt_since)
    state.last_beat[hand] = state.current_beat


def _record_field(state: _GrammarState, token: int, phase: _Phase) -> None:
    """Record x/y/dir fields as they are decoded (needed for saber update)."""
    if phase == _Phase.EXPECT_X:
        state.last_x = max(0, min(token - X_BASE, 3))
    elif phase == _Phase.EXPECT_Y:
        state.last_y = max(0, min(token - Y_BASE, 2))
    elif phase == _Phase.EXPECT_DIR:
        state.last_dir = max(0, min(token - DIR_BASE, 8))


# ---------------------------------------------------------------------------
# Public decoding API
# ---------------------------------------------------------------------------


@torch.no_grad()
def nucleus_sampling_v6(
    model: object,
    audio_features: torch.Tensor,
    difficulty: torch.Tensor,
    genre: torch.Tensor,
    max_events: int = 512,
    max_tokens: int = 4096,
    temperature: float = 0.9,
    top_p: float = 0.9,
    device: torch.device | None = None,
    mapper_id: torch.Tensor | None = None,
    phrase_emb: torch.Tensor | None = None,
) -> list[int]:
    """Generate a V6 swing-event token stream using nucleus sampling.

    Uses a grammar state machine to enforce V6 token grammar at every step.
    Saber state is updated incrementally and passed to the model as conditioning.

    Args:
        model: V6 SequenceModel with decode_step_cached().
        audio_features: Full-song or chunked audio features [1, T, d_model].
        difficulty: Difficulty index [1].
        genre: Genre index [1].
        max_events: Stop after this many complete events.
        max_tokens: Hard token cap (safety).
        temperature: Sampling temperature.
        top_p: Nucleus mass threshold.
        device: Torch device.
        mapper_id: Optional mapper/cohort index [1].
        phrase_emb: Optional phrase embedding [1, d_model].

    Returns:
        Flat token list (without BOS; EOS stripped). Empty if nothing generated.
    """
    if device is None:
        device = audio_features.device

    audio_features = audio_features.to(device)
    difficulty = difficulty.to(device)
    genre = genre.to(device)

    grammar = _GrammarState()
    layer_caches = model.new_caches()

    # Initialise with BOS
    current_token = torch.tensor([[BOS]], dtype=torch.long, device=device)
    logits = model.decode_step_cached(
        current_token, audio_features, difficulty, genre,
        layer_caches, step=0,
        saber_state_step=grammar.saber_tensor(device),
        phrase_emb=phrase_emb,
        mapper_id=mapper_id,
    )

    generated: list[int] = []
    event_count = 0
    step = 1

    while step < max_tokens and event_count < max_events:
        if grammar.phase == _Phase.DONE:
            break

        # Apply grammar mask (nan_to_num first so NaN can't bypass -inf mask)
        mask = _build_mask(grammar.phase, grammar.current_hand, grammar.current_kind,
                           VOCAB_SIZE).to(device)
        masked_logits = logits.nan_to_num(nan=0.0, posinf=1e4, neginf=-1e4) + mask

        # Sample from nucleus
        tok = _nucleus_sample(masked_logits.squeeze(0), temperature, top_p)
        generated.append(tok)

        if tok == EOS:
            break

        # Record position/direction fields before transitioning
        _record_field(grammar, tok, grammar.phase)

        # Advance grammar state
        _transition(grammar, tok)

        if grammar.phase == _Phase.EXPECT_HAND:
            event_count += 1

        # Next decode step
        token_tensor = torch.tensor([[tok]], dtype=torch.long, device=device)
        saber_step = grammar.saber_tensor(device)
        logits = model.decode_step_cached(
            token_tensor, audio_features, difficulty, genre,
            layer_caches, step=step,
            saber_state_step=saber_step,
            phrase_emb=phrase_emb,
            mapper_id=mapper_id,
        )
        step += 1

    # Strip BOS/EOS from output
    return [t for t in generated if t not in (BOS, EOS, PAD)]


def _nucleus_sample(logits: torch.Tensor, temperature: float, top_p: float) -> int:
    """Sample one token from a nucleus (top-p) distribution.

    Only considers tokens with strictly positive probability, so grammar masks
    that set logits to -inf are always respected even with top_p=1.0.
    """
    logits = logits.nan_to_num(nan=0.0, posinf=1e4, neginf=-1e4)
    logits = logits / max(temperature, 1e-6)
    probs = torch.softmax(logits, dim=-1)

    sorted_probs, sorted_indices = torch.sort(probs, descending=True)

    # Only consider tokens with strictly positive probability (respects -inf masks)
    pos_mask = sorted_probs > 0
    sorted_probs = sorted_probs[pos_mask]
    sorted_indices = sorted_indices[pos_mask]

    if len(sorted_probs) == 0:
        # Fallback: return the highest-logit token unconditionally
        return int(logits.argmax().item())

    cumulative = torch.cumsum(sorted_probs, dim=0)
    nucleus = sorted_indices[cumulative - sorted_probs <= top_p]
    if len(nucleus) == 0:
        nucleus = sorted_indices[:1]

    idx = torch.randint(len(nucleus), (1,)).item()
    return int(nucleus[idx].item())
