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
    _SwingEvent,
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
    """Per-sequence grammar state for one decoding stream.

    Cheap to copy (used to chain saber state across windowed inference calls).
    Beats are tracked in song-absolute coordinates so events can be stitched.
    """
    phase: _Phase = _Phase.EXPECT_HAND
    current_hand: int = HAND_LEFT
    current_kind: int = NOTE
    saber: list[float] = field(default_factory=lambda: [0.0] * 12)
    last_beat: dict[int, float] = field(default_factory=lambda: {
        HAND_LEFT: -MAX_DT, HAND_RIGHT: -MAX_DT
    })
    current_beat: float = 0.0
    current_event_dt: float = 0.0
    # Scratch fields for the in-progress event (cleared after each event completes)
    last_x: int = 0
    last_y: int = 0
    last_dir: int = 8
    last_field_d: int = 0

    def saber_tensor(self, device: torch.device) -> torch.Tensor:
        t = torch.tensor(self.saber, dtype=torch.float32, device=device)
        return t.unsqueeze(0).unsqueeze(0)  # [1, 1, 12]

    def clone_for_resume(self) -> _GrammarState:
        """Return a copy with phase reset to EXPECT_HAND for window chaining."""
        return _GrammarState(
            phase=_Phase.EXPECT_HAND,
            current_hand=self.current_hand,
            current_kind=self.current_kind,
            saber=list(self.saber),
            last_beat=dict(self.last_beat),
            current_beat=self.current_beat,
        )


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
    """Record x/y/dir/field_d as they are decoded (needed for event reconstruction)."""
    if phase == _Phase.EXPECT_X:
        state.last_x = max(0, min(token - X_BASE, 3))
    elif phase == _Phase.EXPECT_Y:
        state.last_y = max(0, min(token - Y_BASE, 2))
    elif phase == _Phase.EXPECT_DIR:
        state.last_dir = max(0, min(token - DIR_BASE, 8))
    elif phase == _Phase.EXPECT_FIELD_D:
        # Field offset depends on the current kind
        kind = state.current_kind
        if kind == NOTE:
            state.last_field_d = max(0, min(token - ANGLE_BASE, ANGLE_COUNT - 1))
        elif kind in (ARC_HEAD, ARC_TAIL):
            state.last_field_d = max(0, min(token - MU_BASE, MU_COUNT - 1))
        else:  # CHAIN_HEAD
            state.last_field_d = max(0, min(token - SLICE_BASE, SLICE_COUNT - 1))
    elif phase == _Phase.EXPECT_SQUISH:
        state.last_field_d = max(0, min(token - SQUISH_BASE, SQUISH_COUNT - 1))


def _emit_event(state: _GrammarState) -> _SwingEvent:
    """Snapshot a completed event from the current grammar state.

    Called when an event finalises (phase returns to EXPECT_HAND).
    Beat is song-absolute (whatever state.current_beat holds).
    """
    direction = 0 if state.current_kind in (BOMB, CHAIN_TAIL) else state.last_dir
    field_d = 0 if state.current_kind == BOMB else state.last_field_d
    return _SwingEvent(
        beat=state.current_beat,
        hand=state.current_hand,
        kind=state.current_kind,
        x=state.last_x,
        y=state.last_y,
        direction=direction,
        field_d=field_d,
    )


# ---------------------------------------------------------------------------
# Public decoding API
# ---------------------------------------------------------------------------


@dataclass
class SamplingResult:
    """Output of one V6 sampling call.

    tokens: flat token list (BOS/EOS stripped).
    events: completed swing events with song-absolute beats.
    final_state: grammar state at end of call (use .clone_for_resume() to chain).
    """
    tokens: list[int]
    events: list[_SwingEvent]
    final_state: _GrammarState


@torch.no_grad()
def sample_swing_events(
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
    initial_state: _GrammarState | None = None,
    stop_at_beat: float | None = None,
    activity_probs: torch.Tensor | None = None,
    activity_beat_start: float = 0.0,
    activity_beat_width: float = 8.0,
    activity_threshold: float = 0.5,
    song_pos_frac: torch.Tensor | None = None,
    section_id: torch.Tensor | None = None,
) -> SamplingResult:
    """Sample V6 swing events with grammar constraints and saber-state tracking.

    Stops on the first of: EOS sampled, max_events reached, max_tokens reached,
    or (if stop_at_beat is set) current_beat exceeds stop_at_beat.

    Args:
        model: V6 SequenceModel with decode_step_cached().
        audio_features: Audio encoder output [1, T, d_model] for this window.
        difficulty: Difficulty index [1].
        genre: Genre index [1].
        max_events: Max complete events to emit before stopping.
        max_tokens: Hard token cap (safety).
        temperature: Sampling temperature.
        top_p: Nucleus mass threshold.
        device: Torch device. Defaults to audio_features.device.
        mapper_id: Optional cohort index [1].
        phrase_emb: Optional phrase embedding [1, d_model].
        initial_state: Optional grammar state to resume from (saber state +
            song-absolute current_beat). Phase will be reset to EXPECT_HAND.
        stop_at_beat: If set, stop as soon as current_beat >= this after an
            event completes. Used by windowed full-song generation.
        activity_probs: Optional per-beat activity probabilities [N_BEATS].
            When a beat slot's probability exceeds activity_threshold, EOS is
            suppressed so the model keeps generating rather than going silent.
        activity_beat_start: Song-absolute beat at the start of the activity
            window (aligns activity_probs to current_beat).
        activity_beat_width: Beat range covered by activity_probs.
        activity_threshold: Probability above which EOS is suppressed (0.5).
        song_pos_frac: Optional song position fraction [1] passed to the model.
        section_id: Optional section type index [1] passed to the model.

    Returns:
        SamplingResult with tokens, decoded events, and final grammar state.
    """
    if device is None:
        device = audio_features.device

    audio_features = audio_features.to(device)
    difficulty = difficulty.to(device)
    genre = genre.to(device)
    if song_pos_frac is not None:
        song_pos_frac = song_pos_frac.to(device)
    if section_id is not None:
        section_id = section_id.to(device)

    if initial_state is None:
        grammar = _GrammarState()
    else:
        grammar = initial_state.clone_for_resume()
    layer_caches = model.new_caches()

    # Initialise with BOS
    current_token = torch.tensor([[BOS]], dtype=torch.long, device=device)
    logits = model.decode_step_cached(
        current_token, audio_features, difficulty, genre,
        layer_caches, step=0,
        saber_state_step=grammar.saber_tensor(device),
        phrase_emb=phrase_emb,
        mapper_id=mapper_id,
        song_pos_frac=song_pos_frac,
        section_id=section_id,
    )

    generated: list[int] = []
    events: list[_SwingEvent] = []
    event_count = 0
    step = 1

    while step < max_tokens and event_count < max_events:
        if grammar.phase == _Phase.DONE:
            break

        mask = _build_mask(grammar.phase, grammar.current_hand, grammar.current_kind,
                           VOCAB_SIZE).to(device)

        # V6-7: suppress EOS when ActivityPredictor says current beat is active
        if (activity_probs is not None and grammar.phase == _Phase.EXPECT_HAND):
            beat_frac = (
                (grammar.current_beat - activity_beat_start) / max(activity_beat_width, 1e-6)
            )
            slot = max(0, min(int(beat_frac * len(activity_probs)), len(activity_probs) - 1))
            if float(activity_probs[slot]) > activity_threshold:
                mask[EOS] = _NEG_INF

        masked_logits = logits.nan_to_num(nan=0.0, posinf=1e4, neginf=-1e4) + mask

        tok = _nucleus_sample(masked_logits.squeeze(0), temperature, top_p)
        generated.append(tok)

        if tok == EOS:
            break

        _record_field(grammar, tok, grammar.phase)
        _transition(grammar, tok)

        if grammar.phase == _Phase.EXPECT_HAND:
            # Event just completed — snapshot it
            events.append(_emit_event(grammar))
            event_count += 1
            # Honour stop_at_beat
            if stop_at_beat is not None and grammar.current_beat >= stop_at_beat:
                break

        token_tensor = torch.tensor([[tok]], dtype=torch.long, device=device)
        saber_step = grammar.saber_tensor(device)
        logits = model.decode_step_cached(
            token_tensor, audio_features, difficulty, genre,
            layer_caches, step=step,
            saber_state_step=saber_step,
            phrase_emb=phrase_emb,
            mapper_id=mapper_id,
            song_pos_frac=song_pos_frac,
            section_id=section_id,
        )
        step += 1

    tokens_clean = [t for t in generated if t not in (BOS, EOS, PAD)]
    return SamplingResult(tokens=tokens_clean, events=events, final_state=grammar)


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
    """Backward-compatible token-only V6 sampler. See sample_swing_events for
    the full result (events + chainable state)."""
    result = sample_swing_events(
        model, audio_features, difficulty, genre,
        max_events=max_events, max_tokens=max_tokens,
        temperature=temperature, top_p=top_p,
        device=device, mapper_id=mapper_id, phrase_emb=phrase_emb,
    )
    return result.tokens


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
