"""Beam search and nucleus sampling for Stage 2 note sequence generation.

Provides two decoding strategies:
    - beam_search_decode: Deterministic beam search with length normalization
    - nucleus_sampling_decode: Top-p sampling for creative diversity

Both support KV caching for ~10x faster inference when available.

Grammar-constrained decoding enforces valid Beat Saber note sequences:
    - Max notes per beat (Expert=2, ExpertPlus=3)
    - No same-color duplicates beyond limit
    - No grid collisions (two notes at same col+row)
    - Parity enforcement (forehand/backhand alternation)
    - Color separation bias (red=left, blue=right)
    - Token grammar enforcement (correct attribute order)
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field

import torch
from torch.nn import functional as F  # noqa: N812

from beatsaber_automapper.data.tokenizer import (
    ANGLE_OFFSET_COUNT,
    ANGLE_OFFSET_OFFSET,
    ARC_END,
    ARC_START,
    BOMB,
    BOS,
    CHAIN,
    CHAIN_TAIL_BEAT_COUNT,
    CHAIN_TAIL_BEAT_OFFSET,
    COL_COUNT,
    COL_OFFSET,
    COLOR_COUNT,
    COLOR_OFFSET,
    DIR_COUNT,
    DIR_OFFSET,
    DUR_FRAC_COUNT,
    DUR_FRAC_OFFSET,
    DUR_INT_COUNT,
    DUR_INT_OFFSET,
    EOS,
    HEIGHT_COUNT,
    HEIGHT_OFFSET,
    MID_ANCHOR_COUNT,
    MID_ANCHOR_OFFSET,
    MU_COUNT,
    MU_OFFSET,
    NOTE,
    ROW_COUNT,
    ROW_OFFSET,
    SEP,
    SLICE_COUNT,
    SLICE_OFFSET,
    SQUISH_COUNT,
    SQUISH_OFFSET,
    WALL,
    WIDTH_COUNT,
    WIDTH_OFFSET,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Parity direction groups
# ---------------------------------------------------------------------------

_FOREHAND_DIRS = {1, 6, 7}  # down, down-left, down-right
_BACKHAND_DIRS = {0, 4, 5}  # up, up-left, up-right
_NEUTRAL_DIRS = {2, 3, 8}   # left, right, any (resets parity)

# ---------------------------------------------------------------------------
# Event grammar: each event type -> list of (attr_name, token_offset, token_count)
# ---------------------------------------------------------------------------

_EVENT_GRAMMAR: dict[int, list[tuple[str, int, int]]] = {
    NOTE: [
        ("note_color", COLOR_OFFSET, COLOR_COUNT),
        ("note_col", COL_OFFSET, COL_COUNT),
        ("note_row", ROW_OFFSET, ROW_COUNT),
        ("note_dir", DIR_OFFSET, DIR_COUNT),
        ("note_angle", ANGLE_OFFSET_OFFSET, ANGLE_OFFSET_COUNT),
    ],
    BOMB: [
        ("bomb_col", COL_OFFSET, COL_COUNT),
        ("bomb_row", ROW_OFFSET, ROW_COUNT),
    ],
    WALL: [
        ("wall_col", COL_OFFSET, COL_COUNT),
        ("wall_row", ROW_OFFSET, ROW_COUNT),
        ("wall_width", WIDTH_OFFSET, WIDTH_COUNT),
        ("wall_height", HEIGHT_OFFSET, HEIGHT_COUNT),
        ("wall_dur_int", DUR_INT_OFFSET, DUR_INT_COUNT),
        ("wall_dur_frac", DUR_FRAC_OFFSET, DUR_FRAC_COUNT),
    ],
    ARC_START: [
        ("arc_color", COLOR_OFFSET, COLOR_COUNT),
        ("arc_col", COL_OFFSET, COL_COUNT),
        ("arc_row", ROW_OFFSET, ROW_COUNT),
        ("arc_dir", DIR_OFFSET, DIR_COUNT),
        ("arc_mu", MU_OFFSET, MU_COUNT),
    ],
    ARC_END: [
        ("arc_color", COLOR_OFFSET, COLOR_COUNT),
        ("arc_col", COL_OFFSET, COL_COUNT),
        ("arc_row", ROW_OFFSET, ROW_COUNT),
        ("arc_dir", DIR_OFFSET, DIR_COUNT),
        ("arc_mu", MU_OFFSET, MU_COUNT),
        ("arc_mid", MID_ANCHOR_OFFSET, MID_ANCHOR_COUNT),
    ],
    CHAIN: [
        ("chain_color", COLOR_OFFSET, COLOR_COUNT),
        ("chain_col", COL_OFFSET, COL_COUNT),
        ("chain_row", ROW_OFFSET, ROW_COUNT),
        ("chain_dir", DIR_OFFSET, DIR_COUNT),
        ("chain_tail_col", COL_OFFSET, COL_COUNT),
        ("chain_tail_row", ROW_OFFSET, ROW_COUNT),
        ("chain_slice", SLICE_OFFSET, SLICE_COUNT),
        ("chain_squish", SQUISH_OFFSET, SQUISH_COUNT),
        ("chain_tail_beat", CHAIN_TAIL_BEAT_OFFSET, CHAIN_TAIL_BEAT_COUNT),
    ],
}


# ---------------------------------------------------------------------------
# Constraint state
# ---------------------------------------------------------------------------


@dataclass
class ConstraintState:
    """Tracks grammar and semantic constraints during decoding.

    Manages two concerns:
    1. Grammar enforcement: ensures tokens follow the correct event structure
       (EVENT_TYPE -> attributes -> SEP/EOS -> EVENT_TYPE -> ...)
    2. Semantic constraints: max notes, no collisions, parity, color separation
    """

    # Grammar state
    phase: str = "event_type"  # "event_type", "in_event", "between_events"
    current_event: int = 0     # Event type token being generated
    attr_idx: int = 0          # Index into _EVENT_GRAMMAR[current_event]

    # Onset-level tracking (reset per onset)
    note_count: int = 0
    color_counts: dict[int, int] = field(default_factory=dict)
    positions_used: set[tuple[int, int]] = field(default_factory=set)

    # Current note being built
    current_color: int = -1
    current_col: int = -1

    # Difficulty settings
    max_notes: int = 2
    max_per_color: int = 1

    # Cross-onset parity tracking (persists across onsets)
    last_dir: dict[int, int] = field(default_factory=dict)  # color -> last direction


def init_constraints(
    difficulty: str = "Expert",
    prev_last_dirs: dict[int, int] | None = None,
) -> ConstraintState:
    """Create a fresh constraint state for one onset.

    Args:
        difficulty: "Expert" (max 2 notes, 1/color) or "ExpertPlus" (max 3, 2/color).
        prev_last_dirs: Last swing direction per color from previous onsets.

    Returns:
        Fresh ConstraintState with parity info carried over.
    """
    is_expert_plus = difficulty == "ExpertPlus"
    return ConstraintState(
        max_notes=3 if is_expert_plus else 2,
        max_per_color=2 if is_expert_plus else 1,
        last_dir=dict(prev_last_dirs) if prev_last_dirs else {},
    )


def apply_constraints(logits: torch.Tensor, state: ConstraintState) -> torch.Tensor:
    """Apply grammar and semantic constraints to logits.

    Modifies logits in-place by masking invalid tokens to -inf and applying
    soft biases for preferences (color separation).

    Args:
        logits: Raw model logits [VOCAB_SIZE].
        state: Current constraint state.

    Returns:
        Modified logits tensor.
    """
    mask = torch.full_like(logits, float("-inf"))

    if state.phase == "event_type":
        # Valid: event type tokens or EOS
        mask[EOS] = 0.0
        for et in (NOTE, BOMB, WALL, ARC_START, ARC_END, CHAIN):
            mask[et] = 0.0

        # Hard constraint: cap total notes
        if state.note_count >= state.max_notes:
            mask[NOTE] = float("-inf")

        # Hard constraint: both colors maxed out
        both_maxed = all(
            state.color_counts.get(c, 0) >= state.max_per_color for c in (0, 1)
        )
        if both_maxed:
            mask[NOTE] = float("-inf")

        # Soft: boost EOS if we already have notes (prefer shorter sequences)
        if state.note_count > 0:
            logits[EOS] += 1.0

        logits = logits + mask

    elif state.phase == "between_events":
        # After completing an event: SEP (more events) or EOS (done)
        mask[SEP] = 0.0
        mask[EOS] = 0.0

        # Boost EOS if at note limit
        if state.note_count >= state.max_notes:
            logits[EOS] += 5.0

        logits = logits + mask

    elif state.phase == "in_event":
        grammar = _EVENT_GRAMMAR.get(state.current_event, [])
        if state.attr_idx < len(grammar):
            attr_name, offset, count = grammar[state.attr_idx]

            # Base grammar: allow only tokens in the correct attribute range
            for i in range(count):
                mask[offset + i] = 0.0

            # --- NOTE-specific semantic constraints ---
            if attr_name == "note_color":
                # Mask colors that are maxed out
                for c in range(COLOR_COUNT):
                    if state.color_counts.get(c, 0) >= state.max_per_color:
                        mask[COLOR_OFFSET + c] = float("-inf")
                # Fallback: if all masked, allow all (shouldn't happen with event_type guard)
                if all(mask[COLOR_OFFSET + c] == float("-inf") for c in range(COLOR_COUNT)):
                    for c in range(COLOR_COUNT):
                        mask[COLOR_OFFSET + c] = 0.0

            elif attr_name == "note_col":
                # Soft bias: red prefers cols 0-1, blue prefers cols 2-3
                if state.current_color == 0:  # red
                    logits[COL_OFFSET + 0] += 2.0
                    logits[COL_OFFSET + 1] += 2.0
                    logits[COL_OFFSET + 2] -= 1.0
                    logits[COL_OFFSET + 3] -= 1.0
                elif state.current_color == 1:  # blue
                    logits[COL_OFFSET + 0] -= 1.0
                    logits[COL_OFFSET + 1] -= 1.0
                    logits[COL_OFFSET + 2] += 2.0
                    logits[COL_OFFSET + 3] += 2.0

            elif attr_name == "note_row":
                # Hard constraint: no grid collisions
                for r in range(ROW_COUNT):
                    if (state.current_col, r) in state.positions_used:
                        mask[ROW_OFFSET + r] = float("-inf")
                # Fallback: if all rows blocked for this col, allow all
                if all(
                    mask[ROW_OFFSET + r] == float("-inf") for r in range(ROW_COUNT)
                ):
                    for r in range(ROW_COUNT):
                        mask[ROW_OFFSET + r] = 0.0

            elif attr_name == "note_dir":
                # Soft: penalize direction 8 (any/dot) — model overuses it
                logits[DIR_OFFSET + 8] -= 3.0

                # Soft: boost straight up/down over diagonals
                logits[DIR_OFFSET + 0] += 1.0  # up
                logits[DIR_OFFSET + 1] += 1.0  # down

                # Hard constraint: parity enforcement
                last_d = state.last_dir.get(state.current_color)
                if last_d is not None and last_d not in _NEUTRAL_DIRS:
                    if last_d in _FOREHAND_DIRS:
                        # Last was forehand -> mask forehand, allow backhand+neutral
                        for d in _FOREHAND_DIRS:
                            mask[DIR_OFFSET + d] = float("-inf")
                    elif last_d in _BACKHAND_DIRS:
                        # Last was backhand -> mask backhand, allow forehand+neutral
                        for d in _BACKHAND_DIRS:
                            mask[DIR_OFFSET + d] = float("-inf")
                    # Fallback: if all masked, allow all
                    if all(
                        mask[DIR_OFFSET + d] == float("-inf")
                        for d in range(DIR_COUNT)
                    ):
                        for d in range(DIR_COUNT):
                            mask[DIR_OFFSET + d] = 0.0

            logits = logits + mask
        else:
            # Should not happen if state machine is correct
            # Allow SEP/EOS as fallback
            mask[SEP] = 0.0
            mask[EOS] = 0.0
            logits = logits + mask

    return logits


def update_constraints(state: ConstraintState, token: int) -> None:
    """Update constraint state after generating a token.

    Args:
        state: Current constraint state (modified in-place).
        token: The token that was just generated.
    """
    if token == EOS:
        # Sequence complete, no more updates needed
        return

    if state.phase == "event_type":
        if token in _EVENT_GRAMMAR:
            state.current_event = token
            state.attr_idx = 0
            state.phase = "in_event"
            state.current_color = -1
            state.current_col = -1
        # else: unexpected token, stay in event_type

    elif state.phase == "between_events":
        if token == SEP:
            state.phase = "event_type"
        # EOS handled above

    elif state.phase == "in_event":
        grammar = _EVENT_GRAMMAR.get(state.current_event, [])
        if state.attr_idx < len(grammar):
            attr_name, offset, _count = grammar[state.attr_idx]

            # Track NOTE attributes for semantic constraints
            if attr_name == "note_color":
                state.current_color = max(0, token - offset)
            elif attr_name == "note_col":
                state.current_col = max(0, token - offset)
            elif attr_name == "note_row":
                row = max(0, token - offset)
                state.positions_used.add((state.current_col, row))
            elif attr_name == "note_dir":
                direction = max(0, token - offset)
                state.last_dir[state.current_color] = direction

            state.attr_idx += 1

            # Check if event is complete
            if state.attr_idx >= len(grammar):
                # Event complete
                if state.current_event == NOTE:
                    state.note_count += 1
                    state.color_counts[state.current_color] = (
                        state.color_counts.get(state.current_color, 0) + 1
                    )
                state.phase = "between_events"


# ---------------------------------------------------------------------------
# Beam search (unchanged, no constraints)
# ---------------------------------------------------------------------------


def beam_search_decode(
    model: object,
    audio_features: torch.Tensor,
    difficulty: torch.Tensor,
    genre: torch.Tensor,
    beam_size: int = 8,
    max_length: int = 64,
    temperature: float = 1.0,
    prev_tokens: torch.Tensor | None = None,
    min_length: int = 3,
    constraints: ConstraintState | None = None,
    plan_vector: torch.Tensor | None = None,
) -> list[int]:
    """Run beam search decoding on the sequence model.

    Maintains beam_size hypotheses, expands by top-k tokens at each step,
    and scores by length-normalized log probability.

    Uses KV caching when available (model.decode_step_cached) for 10x
    faster inference. Falls back to non-cached decode_step otherwise.

    Args:
        model: The SequenceModel instance (must have decode_step method).
        audio_features: Encoded audio frame embeddings [1, T, d_model].
        difficulty: Difficulty index tensor [1].
        genre: Genre index tensor [1].
        beam_size: Number of beams to maintain.
        max_length: Maximum output token sequence length.
        temperature: Temperature for logit scaling (>1 = more diverse).
        prev_tokens: Optional previous onset tokens [1, K, S] for inter-onset context.
        min_length: Minimum tokens before EOS is allowed (prevents empty sequences).
        constraints: Optional constraint state for grammar-constrained decoding.

    Returns:
        Best token sequence from beam search (without BOS).
    """
    device = audio_features.device
    use_cache = hasattr(model, "decode_step_cached") and hasattr(model, "new_caches")

    if use_cache:
        return _beam_search_cached(
            model, audio_features, difficulty, genre,
            beam_size, max_length, temperature,
            prev_tokens=prev_tokens, min_length=min_length,
            constraints=constraints, plan_vector=plan_vector,
        )

    # Fallback: non-cached beam search (original implementation)
    beams: list[tuple[float, list[int], ConstraintState | None]] = [
        (0.0, [BOS], copy.deepcopy(constraints) if constraints else None)
    ]
    completed: list[tuple[float, list[int]]] = []

    for step_idx in range(max_length):
        candidates: list[tuple[float, list[int], ConstraintState | None]] = []

        for log_prob, tokens, beam_cs in beams:
            token_tensor = torch.tensor([tokens], dtype=torch.long, device=device)
            logits = model.decode_step(
                token_tensor, audio_features, difficulty, genre,
                prev_tokens=prev_tokens, plan_vector=plan_vector,
            )
            logits = logits.squeeze(0) / temperature

            # Suppress EOS before min_length
            if step_idx < min_length:
                logits[EOS] = float("-inf")

            # Apply constraints
            if beam_cs is not None:
                logits = apply_constraints(logits, beam_cs)

            log_probs = F.log_softmax(logits, dim=-1)

            topk_log_probs, topk_indices = log_probs.topk(beam_size)

            for k in range(beam_size):
                new_token = topk_indices[k].item()
                new_log_prob = log_prob + topk_log_probs[k].item()
                new_tokens = tokens + [new_token]

                if new_token == EOS:
                    seq_len = len(new_tokens) - 1
                    normalized_score = new_log_prob / max(1, seq_len)
                    completed.append((normalized_score, new_tokens))
                else:
                    new_cs = copy.deepcopy(beam_cs) if beam_cs else None
                    if new_cs is not None:
                        update_constraints(new_cs, new_token)
                    candidates.append((new_log_prob, new_tokens, new_cs))

        if not candidates:
            break

        candidates.sort(key=lambda x: x[0], reverse=True)
        beams = candidates[:beam_size]

        if len(completed) >= beam_size:
            break

    if not completed:
        completed = [(lp / max(1, len(t) - 1), t) for lp, t, _cs in beams]

    completed.sort(key=lambda x: x[0], reverse=True)
    best_tokens = completed[0][1]
    return [t for t in best_tokens if t != BOS and t != EOS]


def _beam_search_cached(
    model: object,
    audio_features: torch.Tensor,
    difficulty: torch.Tensor,
    genre: torch.Tensor,
    beam_size: int,
    max_length: int,
    temperature: float,
    prev_tokens: torch.Tensor | None = None,
    min_length: int = 3,
    constraints: ConstraintState | None = None,
    plan_vector: torch.Tensor | None = None,
) -> list[int]:
    """KV-cached beam search implementation.

    Each beam maintains its own set of KV caches and constraint state.
    """
    device = audio_features.device

    # Each beam: (log_prob, token_list, layer_caches, constraint_state)
    initial_caches = model.new_caches()
    initial_cs = copy.deepcopy(constraints)
    beams: list[tuple[float, list[int], list, ConstraintState | None]] = [
        (0.0, [BOS], initial_caches, initial_cs)
    ]
    completed: list[tuple[float, list[int]]] = []

    # Process BOS token first to populate caches
    bos_tensor = torch.tensor([[BOS]], dtype=torch.long, device=device)
    logits = model.decode_step_cached(
        bos_tensor, audio_features, difficulty, genre,
        initial_caches, step=0, prev_tokens=prev_tokens,
        plan_vector=plan_vector,
    )
    logits = logits.squeeze(0) / temperature

    # Suppress EOS at step 0 (before min_length)
    if min_length > 0:
        logits[EOS] = float("-inf")

    # Apply constraints at first step
    if initial_cs is not None:
        logits = apply_constraints(logits, initial_cs)

    log_probs = F.log_softmax(logits, dim=-1)

    # Expand BOS into beam_size initial beams
    topk_log_probs, topk_indices = log_probs.topk(beam_size)
    beams = []
    for k in range(beam_size):
        new_token = topk_indices[k].item()
        new_log_prob = topk_log_probs[k].item()

        if new_token == EOS:
            completed.append((new_log_prob, [BOS, new_token]))
            continue

        beam_caches = _clone_caches(initial_caches)
        beam_cs = copy.deepcopy(initial_cs)
        if beam_cs is not None:
            update_constraints(beam_cs, new_token)
        beams.append((new_log_prob, [BOS, new_token], beam_caches, beam_cs))

    # Continue from step 1 onward
    for step in range(1, max_length):
        if not beams:
            break

        candidates: list[tuple[float, list[int], list, ConstraintState | None]] = []

        for log_prob, tokens, caches, beam_cs in beams:
            last_token = torch.tensor([[tokens[-1]]], dtype=torch.long, device=device)
            logits = model.decode_step_cached(
                last_token, audio_features, difficulty, genre,
                caches, step=step, prev_tokens=prev_tokens,
                plan_vector=plan_vector,
            )
            logits = logits.squeeze(0) / temperature

            # Suppress EOS before min_length
            if step < min_length:
                logits[EOS] = float("-inf")

            # Apply constraints
            if beam_cs is not None:
                logits = apply_constraints(logits, beam_cs)

            step_log_probs = F.log_softmax(logits, dim=-1)

            topk_lp, topk_idx = step_log_probs.topk(beam_size)

            for k in range(beam_size):
                new_token = topk_idx[k].item()
                new_log_prob = log_prob + topk_lp[k].item()
                new_tokens = tokens + [new_token]

                if new_token == EOS:
                    seq_len = len(new_tokens) - 1
                    normalized_score = new_log_prob / max(1, seq_len)
                    completed.append((normalized_score, new_tokens))
                else:
                    if k == 0:
                        new_cs = beam_cs
                        candidates.append(
                            (new_log_prob, new_tokens, caches, new_cs)
                        )
                    else:
                        new_cs = copy.deepcopy(beam_cs)
                        candidates.append(
                            (new_log_prob, new_tokens, _clone_caches(caches), new_cs)
                        )
                    if candidates[-1][3] is not None:
                        update_constraints(candidates[-1][3], new_token)

        if not candidates:
            break

        candidates.sort(key=lambda x: x[0], reverse=True)
        beams = candidates[:beam_size]

        if len(completed) >= beam_size:
            break

    if not completed:
        completed = [
            (lp / max(1, len(t) - 1), t) for lp, t, _caches, _cs in beams
        ]

    completed.sort(key=lambda x: x[0], reverse=True)
    best_tokens = completed[0][1]
    return [t for t in best_tokens if t != BOS and t != EOS]


def _clone_caches(caches: list) -> list:
    """Deep clone KV caches for beam forking."""
    from beatsaber_automapper.models.components import KVCache, LayerCaches

    cloned = []
    for lc in caches:
        new_self = KVCache(
            k=lc.self_attn.k.clone() if lc.self_attn.k is not None else None,
            v=lc.self_attn.v.clone() if lc.self_attn.v is not None else None,
        )
        new_cross = KVCache(
            k=lc.cross_attn.k.clone() if lc.cross_attn.k is not None else None,
            v=lc.cross_attn.v.clone() if lc.cross_attn.v is not None else None,
        )
        cloned.append(LayerCaches(self_attn=new_self, cross_attn=new_cross))
    return cloned


# ---------------------------------------------------------------------------
# Nucleus sampling with constraints
# ---------------------------------------------------------------------------


def nucleus_sampling_decode(
    model: object,
    audio_features: torch.Tensor,
    difficulty: torch.Tensor,
    genre: torch.Tensor,
    max_length: int = 64,
    temperature: float = 1.0,
    top_p: float = 0.9,
    prev_tokens: torch.Tensor | None = None,
    min_length: int = 3,
    repetition_penalty: float = 1.0,
    constraints: ConstraintState | None = None,
    plan_vector: torch.Tensor | None = None,
) -> list[int]:
    """Run nucleus (top-p) sampling on the sequence model.

    Uses KV caching when available for faster inference.
    Supports grammar-constrained decoding when constraints are provided.

    Args:
        model: The SequenceModel instance (must have decode_step method).
        audio_features: Encoded audio frame embeddings [1, T, d_model].
        difficulty: Difficulty index tensor [1].
        genre: Genre index tensor [1].
        max_length: Maximum output token sequence length.
        temperature: Temperature for logit scaling.
        top_p: Cumulative probability threshold for nucleus filtering.
        prev_tokens: Optional previous onset tokens [1, K, S] for inter-onset context.
        min_length: Minimum tokens before EOS is allowed.
        repetition_penalty: Divide logits of recently generated tokens by this factor.
        constraints: Optional ConstraintState for grammar-constrained decoding.

    Returns:
        Sampled token sequence (without BOS or EOS).
    """
    device = audio_features.device
    use_cache = hasattr(model, "decode_step_cached") and hasattr(model, "new_caches")

    if use_cache:
        return _nucleus_cached(
            model, audio_features, difficulty, genre,
            max_length, temperature, top_p,
            prev_tokens=prev_tokens, min_length=min_length,
            repetition_penalty=repetition_penalty,
            constraints=constraints, plan_vector=plan_vector,
        )

    # Fallback: non-cached nucleus sampling
    tokens = [BOS]
    cs = copy.deepcopy(constraints)

    for step in range(max_length):
        token_tensor = torch.tensor([tokens], dtype=torch.long, device=device)
        logits = model.decode_step(
            token_tensor, audio_features, difficulty, genre,
            prev_tokens=prev_tokens, plan_vector=plan_vector,
        )
        logits = logits.squeeze(0) / temperature

        # Suppress EOS before min_length
        if step < min_length:
            logits[EOS] = float("-inf")

        # Repetition penalty: penalize recently generated tokens
        if repetition_penalty > 1.0 and len(tokens) > 1:
            for prev_token in set(tokens[-12:]):
                if logits[prev_token] > 0:
                    logits[prev_token] /= repetition_penalty
                else:
                    logits[prev_token] *= repetition_penalty

        # Apply grammar + semantic constraints
        if cs is not None:
            logits = apply_constraints(logits, cs)

        probs = F.softmax(logits, dim=-1)
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

        mask = cumulative_probs - sorted_probs > top_p
        sorted_probs[mask] = 0.0
        sorted_probs = sorted_probs / sorted_probs.sum()

        idx = torch.multinomial(sorted_probs, num_samples=1).item()
        next_token = sorted_indices[idx].item()

        if next_token == EOS:
            break

        if cs is not None:
            update_constraints(cs, next_token)
        tokens.append(next_token)

    return tokens[1:]


def _nucleus_cached(
    model: object,
    audio_features: torch.Tensor,
    difficulty: torch.Tensor,
    genre: torch.Tensor,
    max_length: int,
    temperature: float,
    top_p: float,
    prev_tokens: torch.Tensor | None = None,
    min_length: int = 3,
    repetition_penalty: float = 1.0,
    constraints: ConstraintState | None = None,
    plan_vector: torch.Tensor | None = None,
) -> list[int]:
    """KV-cached nucleus sampling with optional grammar constraints."""
    device = audio_features.device
    caches = model.new_caches()
    tokens = [BOS]
    cs = copy.deepcopy(constraints)

    for step in range(max_length):
        token_tensor = torch.tensor([[tokens[-1]]], dtype=torch.long, device=device)
        logits = model.decode_step_cached(
            token_tensor, audio_features, difficulty, genre,
            caches, step=step, prev_tokens=prev_tokens,
            plan_vector=plan_vector,
        )
        logits = logits.squeeze(0) / temperature

        # Suppress EOS before min_length
        if step < min_length:
            logits[EOS] = float("-inf")

        # Repetition penalty: penalize recently generated tokens
        if repetition_penalty > 1.0 and len(tokens) > 1:
            for prev_token in set(tokens[-12:]):
                if logits[prev_token] > 0:
                    logits[prev_token] /= repetition_penalty
                else:
                    logits[prev_token] *= repetition_penalty

        # Apply grammar + semantic constraints
        if cs is not None:
            logits = apply_constraints(logits, cs)

        probs = F.softmax(logits, dim=-1)
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

        mask = cumulative_probs - sorted_probs > top_p
        sorted_probs[mask] = 0.0
        sorted_probs = sorted_probs / sorted_probs.sum()

        idx = torch.multinomial(sorted_probs, num_samples=1).item()
        next_token = sorted_indices[idx].item()

        if next_token == EOS:
            break

        if cs is not None:
            update_constraints(cs, next_token)
        tokens.append(next_token)

    return tokens[1:]
