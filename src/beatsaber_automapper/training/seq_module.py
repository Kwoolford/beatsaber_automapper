"""Lightning module for Stage 2: Note sequence generation training.

Wraps the AudioEncoder + SequenceModel for teacher-forced training
with cross-entropy loss over the token vocabulary. BOS is prepended
to create decoder input; original tokens serve as targets.

Features rhythm token weighting (3x weight on timing-sensitive tokens
like EVENT_TYPE, SEP, EOS) from Mapperatorinator research — timing is
the hardest and most important thing to learn.
"""

from __future__ import annotations

import logging
import math

import lightning
import torch
import torch.nn as nn

from beatsaber_automapper.data.tokenizer import (
    ARC_END,
    ARC_START,
    BOMB,
    BOS,
    CHAIN,
    COL_OFFSET,
    COLOR_OFFSET,
    DIR_COUNT,
    DIR_OFFSET,
    EOS,
    NOTE,
    PAD,
    ROW_OFFSET,
    SEP,
    WALL,
)
from beatsaber_automapper.models.audio_encoder import AudioEncoder
from beatsaber_automapper.models.sequence_model import SequenceModel

logger = logging.getLogger(__name__)

# Timing-sensitive tokens — these control WHEN and WHAT type of note appears.
# They get higher weight in the loss because getting timing right is the most
# critical part of beatmap generation (from Mapperatorinator research).
_RHYTHM_TOKENS = frozenset({EOS, SEP, NOTE, BOMB, WALL, ARC_START, ARC_END, CHAIN})


def _build_token_weights(
    vocab_size: int,
    rhythm_weight: float = 3.0,
    eos_weight: float = 1.0,
    rare_event_weight: float = 1.0,
    bomb_weight: float = 1.0,
) -> torch.Tensor:
    """Build per-token loss weights with higher weight on rhythm + rare tokens.

    Args:
        vocab_size: Size of token vocabulary.
        rhythm_weight: Weight multiplier for timing-sensitive tokens.
        eos_weight: Weight for EOS token.
        rare_event_weight: Extra multiplier for ARC_START/ARC_END/CHAIN (V4).
        bomb_weight: Extra multiplier for BOMB (V4 — slightly less than arcs/chains).

    Returns:
        Weight tensor [vocab_size].
    """
    weights = torch.ones(vocab_size)
    for token_id in _RHYTHM_TOKENS:
        if 0 <= token_id < vocab_size:
            weights[token_id] = rhythm_weight
    # V4: up-weight rare event types so the model actually learns to emit them
    for token_id in (ARC_START, ARC_END, CHAIN):
        if 0 <= token_id < vocab_size:
            weights[token_id] = rhythm_weight * rare_event_weight
    if 0 <= BOMB < vocab_size:
        weights[BOMB] = rhythm_weight * bomb_weight
    if 0 <= EOS < vocab_size:
        weights[EOS] = eos_weight
    weights[PAD] = 0.0
    return weights


# Direction parity classes for flow-aware loss
# Forehand swings: up, down-left, down-right (directions 0, 6, 7 — but parity is
# based on hand movement, so we group by swing direction)
_FOREHAND_DIRS = frozenset({1, 6, 7})  # down, down-left, down-right
_BACKHAND_DIRS = frozenset({0, 4, 5})  # up, up-left, up-right

# 9-direction unit vectors (x_right_positive, y_up_positive). Index = direction id.
# 0=up, 1=down, 2=left, 3=right, 4=up-left, 5=up-right, 6=down-left, 7=down-right, 8=any
_DIR_VECTORS = torch.tensor([
    [0.0,  1.0],   # 0 up
    [0.0, -1.0],   # 1 down
    [-1.0, 0.0],   # 2 left
    [1.0,  0.0],   # 3 right
    [-0.707,  0.707],  # 4 up-left
    [0.707,   0.707],  # 5 up-right
    [-0.707, -0.707],  # 6 down-left
    [0.707,  -0.707],  # 7 down-right
    [0.0,  0.0],   # 8 any/dot — neutral
], dtype=torch.float32)


def _compute_flow_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    prev_tokens: torch.Tensor | None,
    time_gap: torch.Tensor | None,
) -> torch.Tensor:
    """Compute differentiable flow parity loss using logit probabilities.

    Unlike the previous version (which used `.detach()` on argmax predictions,
    producing zero gradients), this version works with soft direction probabilities
    to provide actual gradient signal for parity learning.

    **Per-color parity:** Each color (red=0, blue=1) has independent parity history.
    The loss only penalizes same-parity violations within the same color, since red
    and blue hands swing independently.

    For each note in the target sequence:
    1. Identify the note's color from the target tokens
    2. Look up the previous direction for THAT color from the previous onset
    3. Penalize probability mass on same-parity directions at the predicted position

    Also penalizes direction 8 (any/dot) overuse with a small auxiliary term.

    Args:
        logits: Model output logits [B, S, V] — NOT detached.
        target: Target token sequence [B, S].
        prev_tokens: Previous onset tokens [B, K, S] or None.
        time_gap: Seconds since previous onset [B] or None.

    Returns:
        Differentiable scalar loss (parity violation penalty + dot penalty).
    """
    if prev_tokens is None or time_gap is None:
        return torch.tensor(0.0, device=logits.device, requires_grad=False)

    b, s, v = logits.shape
    device = logits.device

    # Build masks for forehand/backhand/dot direction token IDs
    forehand_ids = [DIR_OFFSET + d for d in (1, 6, 7)]  # down, down-left, down-right
    backhand_ids = [DIR_OFFSET + d for d in (0, 4, 5)]  # up, up-left, up-right
    dot_id = DIR_OFFSET + 8  # "any" direction

    parity_loss_sum = torch.tensor(0.0, device=device)
    dot_loss_sum = torch.tensor(0.0, device=device)
    count = 0

    for i in range(b):
        # Skip if time gap > 3 seconds (parity resets)
        if time_gap[i].item() > 3.0:
            continue

        # Extract per-color last direction from the most recent previous onset
        last_prev = prev_tokens[i, -1]  # [S]
        prev_dirs_by_color = _extract_directions_by_color(last_prev)
        if not prev_dirs_by_color:
            continue

        # Find ALL notes in the target and check parity for each
        target_list = target[i].tolist()
        for pos in range(len(target_list)):
            if target_list[pos] != NOTE or pos + 4 >= s:
                continue

            # Extract this note's color and direction position
            note_color = target_list[pos + 1] - COLOR_OFFSET
            dir_pos = pos + 4

            # Look up previous direction for this specific color
            prev_dir = prev_dirs_by_color.get(note_color)
            if prev_dir is None or prev_dir in (2, 3, 8):
                # No parity history for this color, or neutral/dot — skip
                # Still penalize dot overuse
                if dot_id < v and dir_pos < s:
                    dir_probs = torch.softmax(logits[i, dir_pos, :], dim=-1)
                    dot_loss_sum = dot_loss_sum + dir_probs[dot_id]
                    count += 1
                continue

            if dir_pos >= s:
                continue

            prev_is_forehand = prev_dir in _FOREHAND_DIRS

            # Get soft probabilities at the direction position
            dir_probs = torch.softmax(logits[i, dir_pos, :], dim=-1)  # [V]

            # Sum probability mass on same-parity directions (= violation prob)
            if prev_is_forehand:
                same_parity_ids = forehand_ids
            else:
                same_parity_ids = backhand_ids

            same_parity_prob = sum(dir_probs[tid] for tid in same_parity_ids)
            parity_loss_sum = parity_loss_sum + same_parity_prob

            # Dot note penalty: penalize probability mass on direction 8
            if dot_id < v:
                dot_loss_sum = dot_loss_sum + dir_probs[dot_id]

            count += 1

    if count == 0:
        return torch.tensor(0.0, device=device, requires_grad=False)

    # Average parity violation probability + weighted dot penalty
    return (parity_loss_sum + 0.5 * dot_loss_sum) / count


def _extract_directions_by_color(tokens: torch.Tensor) -> dict[int, int]:
    """Extract the last direction for each color from a token sequence.

    Scans the full sequence to find all NOTE events and records
    the direction for each color (red=0, blue=1). Returns the LAST
    direction seen for each color, which represents the most recent
    swing parity state.

    Args:
        tokens: Token sequence [S].

    Returns:
        Dict mapping color (0 or 1) to direction (0-8).
    """
    tokens_list = tokens.tolist()
    dirs_by_color: dict[int, int] = {}
    i = 0
    while i < len(tokens_list):
        if tokens_list[i] == NOTE and i + 4 < len(tokens_list):
            color = tokens_list[i + 1] - COLOR_OFFSET
            direction = tokens_list[i + 4] - DIR_OFFSET
            if 0 <= color <= 1 and 0 <= direction <= 8:
                dirs_by_color[color] = direction
            i += 6  # Skip past this note's tokens
        else:
            i += 1
    return dirs_by_color


def _compute_ergo_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """Compute differentiable ergonomic loss for color-side preference.

    Penalizes the model for predicting columns on the wrong side for each color:
    - Red (color=0) should prefer columns 0-1 (left side)
    - Blue (color=1) should prefer columns 2-3 (right side)

    Uses soft column probabilities from the model's logits at positions where
    the TARGET has a column token (after a NOTE + COLOR).

    Args:
        logits: Model output logits [B, S, V].
        target: Target token sequence [B, S].

    Returns:
        Differentiable scalar loss (average wrong-side probability).
    """
    b, s, v = logits.shape
    device = logits.device

    # Red prefers cols 0,1; blue prefers cols 2,3
    red_wrong = [COL_OFFSET + 2, COL_OFFSET + 3]
    blue_wrong = [COL_OFFSET + 0, COL_OFFSET + 1]

    ergo_sum = torch.tensor(0.0, device=device)
    count = 0

    for i in range(b):
        target_list = target[i].tolist()
        for pos in range(len(target_list)):
            if target_list[pos] != NOTE or pos + 2 >= s:
                continue
            # pos+1 is COLOR, pos+2 is COL
            color = target_list[pos + 1] - COLOR_OFFSET
            col_pos = pos + 2
            if col_pos >= s:
                continue

            col_probs = torch.softmax(logits[i, col_pos, :], dim=-1)

            if color == 0:  # red
                wrong_prob = sum(col_probs[tid] for tid in red_wrong)
            elif color == 1:  # blue
                wrong_prob = sum(col_probs[tid] for tid in blue_wrong)
            else:
                continue

            ergo_sum = ergo_sum + wrong_prob
            count += 1

    if count == 0:
        return torch.tensor(0.0, device=device, requires_grad=False)

    return ergo_sum / count


def _extract_notes_with_position(tokens: list[int]) -> list[tuple[int, int, int, int, int]]:
    """Extract NOTE events as (token_pos, color, col, row, dir) tuples.

    Scans a token list and returns one entry per NOTE event found, with
    the token position of the direction slot (for indexing logits) and
    the physical (color, col, row, direction) values.

    Args:
        tokens: Flat list of token IDs.

    Returns:
        List of (dir_token_pos, color, col, row, direction) tuples.
    """
    out: list[tuple[int, int, int, int, int]] = []
    i = 0
    n = len(tokens)
    while i < n:
        if tokens[i] == NOTE and i + 4 < n:
            color = tokens[i + 1] - COLOR_OFFSET
            col = tokens[i + 2] - COL_OFFSET
            row = tokens[i + 3] - ROW_OFFSET
            direction = tokens[i + 4] - DIR_OFFSET
            if 0 <= color <= 1 and 0 <= col <= 3 and 0 <= row <= 2 and 0 <= direction <= 8:
                out.append((i + 4, color, col, row, direction))
            i += 6
        else:
            i += 1
    return out


def _compute_follow_through_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    prev_tokens: torch.Tensor | None,
    time_gap: torch.Tensor | None,
) -> torch.Tensor:
    """Differentiable penalty on swing direction vs. movement-vector alignment.

    The saber's swing at the current note should be roughly aligned with the
    direction of travel from the previous same-color note to the current one.
    Large misalignment = 2D teleport / unplayable follow-through.

    For each consecutive same-color note pair (prev → curr) within a distance
    of >= 0.5 grid units:

        movement = normalize(curr_pos - prev_pos)
        alignment[d] = movement . dir_vec[d]  for each direction d
        loss contribution = sum_d prob[d] * (1 - alignment[d]) / 2

    Alignment is in [-1, 1]; (1 - alignment)/2 is in [0, 1] (1 = opposite dirs).
    Soft direction probabilities give gradient flow.

    Applies both:
    - Cross-onset: last prev-onset note of each color → first same-color note of target
    - Intra-onset: consecutive same-color notes inside target

    Args:
        logits: Model output logits [B, S, V].
        target: Target token sequence [B, S].
        prev_tokens: Previous onset tokens [B, K, S] or None.
        time_gap: Seconds since previous onset [B] or None.

    Returns:
        Differentiable scalar loss (average misalignment, in [0, 1]).
    """
    b, s, v = logits.shape
    device = logits.device
    dir_vecs = _DIR_VECTORS.to(device)  # [9, 2]

    total = torch.tensor(0.0, device=device)
    count = 0

    for i in range(b):
        target_list = target[i].tolist()
        target_notes = _extract_notes_with_position(target_list)

        # Build last-note-per-color from the immediately previous onset
        prev_last_by_color: dict[int, tuple[int, int, int]] = {}
        use_prev = (
            prev_tokens is not None
            and time_gap is not None
            and time_gap[i].item() <= 3.0
        )
        if use_prev:
            last_prev = prev_tokens[i, -1].tolist()
            for _, c, x, y, d in _extract_notes_with_position(last_prev):
                prev_last_by_color[c] = (x, y, d)

        # Track most recent note per color as we walk through target
        curr_last_by_color: dict[int, tuple[int, int, int]] = dict(prev_last_by_color)

        for dir_pos, color, col, row, _ in target_notes:
            prev_pos = curr_last_by_color.get(color)
            curr_last_by_color[color] = (col, row, 0)  # dir updated below implicitly

            if prev_pos is None:
                continue
            px, py, _ = prev_pos
            dx = float(col - px)
            dy = float(row - py)
            dist = (dx * dx + dy * dy) ** 0.5
            if dist < 0.5:
                # Same position or tiny movement — no follow-through penalty (any dir OK)
                continue

            # Normalized movement vector
            inv = 1.0 / dist
            move = torch.tensor([dx * inv, dy * inv], device=device, dtype=dir_vecs.dtype)

            # Alignment scores per direction: cosine similarity in [-1, 1]
            alignment = dir_vecs @ move  # [9]
            # Map to loss per direction: 0 (aligned) → 1 (opposite)
            dir_penalty = (1.0 - alignment) * 0.5  # [9]
            # Neutral/dot gets a fixed mild penalty (encourage directional choices)
            dir_penalty = dir_penalty.clone()
            dir_penalty[8] = 0.5

            if dir_pos >= s:
                continue
            dir_probs = torch.softmax(logits[i, dir_pos, :], dim=-1)
            # Weight penalties by probability mass on each direction token
            weighted = sum(
                dir_probs[DIR_OFFSET + d] * dir_penalty[d] for d in range(DIR_COUNT)
            )
            total = total + weighted
            count += 1

    if count == 0:
        return torch.tensor(0.0, device=device, requires_grad=False)
    return total / count


def _compute_intra_onset_parity_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """Penalize same-parity consecutive same-color notes within a single onset.

    The inter-onset flow loss already handles parity across onsets. This loss
    closes the gap for chord-like onsets with multiple same-color notes.

    Args:
        logits: Model output logits [B, S, V].
        target: Target token sequence [B, S].

    Returns:
        Differentiable scalar loss.
    """
    b, s, v = logits.shape
    device = logits.device
    forehand_ids = [DIR_OFFSET + d for d in (1, 6, 7)]
    backhand_ids = [DIR_OFFSET + d for d in (0, 4, 5)]

    total = torch.tensor(0.0, device=device)
    count = 0

    for i in range(b):
        target_list = target[i].tolist()
        target_notes = _extract_notes_with_position(target_list)

        last_parity_by_color: dict[int, str] = {}
        for dir_pos, color, _, _, direction in target_notes:
            if dir_pos >= s:
                continue
            dir_probs = torch.softmax(logits[i, dir_pos, :], dim=-1)
            last = last_parity_by_color.get(color)

            if last == "F":
                same_ids = forehand_ids
            elif last == "B":
                same_ids = backhand_ids
            else:
                same_ids = None

            if same_ids is not None:
                same_prob = sum(dir_probs[tid] for tid in same_ids)
                total = total + same_prob
                count += 1

            if direction in _FOREHAND_DIRS:
                last_parity_by_color[color] = "F"
            elif direction in _BACKHAND_DIRS:
                last_parity_by_color[color] = "B"
            # Neutral directions don't update parity

    if count == 0:
        return torch.tensor(0.0, device=device, requires_grad=False)
    return total / count


def _find_first_direction_position(tokens: torch.Tensor) -> int | None:
    """Find the sequence position of the first direction token.

    In the token grammar, a NOTE event is: [NOTE, COLOR, COL, ROW, DIR, ANGLE].
    So DIR is at position NOTE_pos + 4.

    Args:
        tokens: Token sequence [S].

    Returns:
        Position index of the first direction token, or None.
    """
    tokens_list = tokens.tolist()
    for i, t in enumerate(tokens_list):
        if t == NOTE and i + 4 < len(tokens_list):
            return i + 4
    return None


def _extract_first_direction(tokens: torch.Tensor) -> int | None:
    """Extract the first direction value from a token sequence.

    Args:
        tokens: Token sequence [S].

    Returns:
        Direction value (0-8) or None if no NOTE found.
    """
    tokens_list = tokens.tolist()
    for i, t in enumerate(tokens_list):
        if t == NOTE and i + 4 < len(tokens_list):
            direction = tokens_list[i + 4] - DIR_OFFSET
            if 0 <= direction <= 8:
                return direction
    return None


class SequenceLitModule(lightning.LightningModule):
    """Lightning training module for note sequence generation.

    Handles training step, validation step, optimizer configuration,
    and metric logging for Stage 2.

    Args:
        n_mels: Number of mel bands for audio encoder.
        encoder_d_model: Audio encoder model dimension.
        encoder_nhead: Audio encoder attention heads.
        encoder_num_layers: Audio encoder transformer layers.
        encoder_dim_feedforward: Audio encoder FFN dimension.
        encoder_dropout: Audio encoder dropout.
        vocab_size: Token vocabulary size.
        seq_d_model: Sequence model dimension.
        seq_nhead: Sequence model attention heads.
        seq_num_layers: Sequence model transformer layers.
        seq_dim_feedforward: Sequence model FFN dimension.
        seq_num_difficulties: Number of difficulty levels.
        seq_num_genres: Number of genre classes.
        seq_dropout: Sequence model dropout.
        conditioning_dropout: Dropout probability for difficulty/genre embeddings.
        label_smoothing: Label smoothing for cross-entropy loss.
        rhythm_weight: Weight multiplier for timing-sensitive tokens (3.0 = 3x).
        learning_rate: Peak learning rate.
        weight_decay: AdamW weight decay.
        warmup_steps: Linear warmup steps.
        freeze_encoder: Whether to freeze audio encoder weights.
    """

    def __init__(
        self,
        # Audio encoder params
        n_mels: int = 80,
        encoder_d_model: int = 512,
        encoder_nhead: int = 8,
        encoder_num_layers: int = 6,
        encoder_dim_feedforward: int = 2048,
        encoder_dropout: float = 0.1,
        # Sequence model params
        vocab_size: int = 183,
        seq_d_model: int = 512,
        seq_nhead: int = 8,
        seq_num_layers: int = 8,
        seq_dim_feedforward: int = 2048,
        seq_num_difficulties: int = 5,
        seq_num_genres: int = 11,
        seq_dropout: float = 0.1,
        # Inter-onset context
        prev_context_k: int = 0,
        # Conditioning dropout for CFG
        conditioning_dropout: float = 0.0,
        # Training params
        label_smoothing: float = 0.1,
        rhythm_weight: float = 3.0,
        eos_weight: float = 0.3,
        rare_event_weight: float = 1.0,  # V4: extra multiplier for ARC/CHAIN/BOMB
        bomb_weight: float = 1.0,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        warmup_steps: int = 2000,
        lr_min_ratio: float = 0.01,
        token_dropout: float = 0.0,
        freeze_encoder: bool = False,
        # Flow-aware auxiliary loss
        flow_loss_alpha: float = 0.0,
        # Ergonomic auxiliary loss (color-side preference)
        ergo_loss_alpha: float = 0.0,
        # Follow-through loss (V4: grid-position vs. swing-direction alignment)
        follow_through_alpha: float = 0.0,
        # Intra-onset parity (V4: same-color notes within an onset should alternate)
        intra_onset_parity_alpha: float = 0.0,
        # Onset planner
        use_planner: bool = False,
        planner_layers: int = 4,
        planner_heads: int = 8,
        # Structure features
        n_structure_features: int = 8,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.audio_encoder = AudioEncoder(
            n_mels=n_mels,
            d_model=encoder_d_model,
            nhead=encoder_nhead,
            num_layers=encoder_num_layers,
            dim_feedforward=encoder_dim_feedforward,
            dropout=encoder_dropout,
            n_structure_features=n_structure_features,
        )
        self.sequence_model = SequenceModel(
            vocab_size=vocab_size,
            d_model=seq_d_model,
            nhead=seq_nhead,
            num_layers=seq_num_layers,
            dim_feedforward=seq_dim_feedforward,
            num_difficulties=seq_num_difficulties,
            num_genres=seq_num_genres,
            dropout=seq_dropout,
            conditioning_dropout=conditioning_dropout,
            prev_context_k=prev_context_k,
        )

        # Optional onset planner for song-level context
        self.onset_planner = None
        if use_planner:
            from beatsaber_automapper.models.onset_planner import OnsetPlanner
            self.onset_planner = OnsetPlanner(
                d_model=seq_d_model,
                nhead=planner_heads,
                num_layers=planner_layers,
                dim_feedforward=seq_dim_feedforward,
                dropout=seq_dropout,
            )

        if freeze_encoder:
            for param in self.audio_encoder.parameters():
                param.requires_grad = False

        # Build per-token loss weights with rhythm emphasis + EOS downweighting
        token_weights = _build_token_weights(
            vocab_size,
            rhythm_weight,
            eos_weight=eos_weight,
            rare_event_weight=rare_event_weight,
            bomb_weight=bomb_weight,
        )
        self.register_buffer("token_weights", token_weights)

        self.loss_fn = nn.CrossEntropyLoss(
            weight=token_weights,
            ignore_index=PAD,
            label_smoothing=label_smoothing,
        )

    def _prepare_teacher_forcing(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Prepare decoder input and targets for teacher forcing.

        Prepends BOS to tokens[:-1] as input; original tokens as target.

        Args:
            tokens: Raw token sequences [B, S].

        Returns:
            Tuple of (decoder_input [B, S], target [B, S]).
        """
        b, s = tokens.shape
        bos = torch.full((b, 1), BOS, dtype=tokens.dtype, device=tokens.device)
        decoder_input = torch.cat([bos, tokens[:, :-1]], dim=1)  # [B, S]
        target = tokens  # [B, S]
        return decoder_input, target

    def forward(
        self,
        mel: torch.Tensor,
        tokens: torch.Tensor,
        difficulty: torch.Tensor,
        genre: torch.Tensor,
        structure: torch.Tensor | None = None,
        prev_tokens: torch.Tensor | None = None,
        plan_vector: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass: mel -> audio features -> token logits.

        Args:
            mel: Mel spectrogram [B, n_mels, T].
            tokens: Decoder input tokens [B, S] (already BOS-prepended).
            difficulty: Difficulty indices [B].
            genre: Genre indices [B].
            structure: Optional structure features [B, 8, T].
            prev_tokens: Optional previous onset tokens [B, K, S] for inter-onset context.
            plan_vector: Optional plan vector from OnsetPlanner [B, 1, d_model].

        Returns:
            Token logits [B, S, vocab_size].
        """
        audio_features = self.audio_encoder(mel, structure_features=structure)

        # If planner is active and no plan_vector provided, compute one from audio center
        if self.onset_planner is not None and plan_vector is None:
            # Use center frame embedding as a single-onset approximation
            # Full song-level planning is done via SongBatchSampler (Phase 2C)
            center = audio_features.shape[1] // 2
            onset_emb = audio_features[:, center : center + 1, :]  # [B, 1, d_model]
            # Extract section features at center frame if structure has 8 channels
            onset_sec_ids = None
            onset_sec_progress = None
            if structure is not None and structure.shape[1] >= 8:
                sec_id_norm = structure[:, 6, center]  # [B]
                onset_sec_ids = (sec_id_norm * 5.0).round().long().clamp(0, 5).unsqueeze(1)
                onset_sec_progress = structure[:, 7, center].unsqueeze(1)  # [B, 1]
            plan_vector = self.onset_planner(
                onset_emb, section_ids=onset_sec_ids, section_progress=onset_sec_progress,
            )  # [B, 1, d_model]

        return self.sequence_model(
            tokens, audio_features, difficulty, genre,
            prev_tokens=prev_tokens, plan_vector=plan_vector,
        )

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        decoder_input, target = self._prepare_teacher_forcing(batch["tokens"])

        # Token dropout: replace random tokens with random vocab IDs to reduce
        # exposure bias and force reliance on audio context over token copying
        token_dropout = self.hparams.token_dropout
        if token_dropout > 0 and self.training:
            mask = torch.rand_like(decoder_input.float()) < token_dropout
            mask[:, 0] = False  # Never mask BOS/first token
            decoder_input = decoder_input.clone()
            decoder_input[mask] = torch.randint(
                1, self.hparams.vocab_size, (mask.sum().item(),),
                device=decoder_input.device,
            )

        structure = batch.get("structure", None)
        prev_tokens = batch.get("prev_tokens", None)
        logits = self(
            batch["mel"], decoder_input, batch["difficulty"], batch["genre"],
            structure=structure, prev_tokens=prev_tokens,
        )
        # logits: [B, S, V], target: [B, S]
        ce_loss = self.loss_fn(logits.reshape(-1, logits.size(-1)), target.reshape(-1))

        # Flow-aware auxiliary loss (differentiable via soft direction probabilities)
        loss = ce_loss
        alpha = self.hparams.flow_loss_alpha
        if alpha > 0 and prev_tokens is not None:
            time_gap = batch.get("time_gap", None)
            flow_loss = _compute_flow_loss(logits, target, prev_tokens, time_gap)
            loss = loss + alpha * flow_loss
            self.log("train_flow_loss", flow_loss, prog_bar=False)

        # Ergonomic auxiliary loss (color-side preference)
        ergo_alpha = self.hparams.ergo_loss_alpha
        if ergo_alpha > 0:
            ergo_loss = _compute_ergo_loss(logits, target)
            loss = loss + ergo_alpha * ergo_loss
            self.log("train_ergo_loss", ergo_loss, prog_bar=False)

        # V4: follow-through loss (grid-position vs. swing-dir alignment)
        ft_alpha = self.hparams.follow_through_alpha
        if ft_alpha > 0:
            time_gap = batch.get("time_gap", None)
            ft_loss = _compute_follow_through_loss(logits, target, prev_tokens, time_gap)
            loss = loss + ft_alpha * ft_loss
            self.log("train_follow_through_loss", ft_loss, prog_bar=False)

        # V4: intra-onset parity (same-color notes in chord must alternate)
        iop_alpha = self.hparams.intra_onset_parity_alpha
        if iop_alpha > 0:
            iop_loss = _compute_intra_onset_parity_loss(logits, target)
            loss = loss + iop_alpha * iop_loss
            self.log("train_intra_onset_parity_loss", iop_loss, prog_bar=False)

        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> None:
        decoder_input, target = self._prepare_teacher_forcing(batch["tokens"])
        structure = batch.get("structure", None)
        prev_tokens = batch.get("prev_tokens", None)
        logits = self(
            batch["mel"], decoder_input, batch["difficulty"], batch["genre"],
            structure=structure, prev_tokens=prev_tokens,
        )
        loss = self.loss_fn(logits.reshape(-1, logits.size(-1)), target.reshape(-1))
        self.log("val_loss", loss, prog_bar=True, sync_dist=True)

        # Token accuracy (ignoring PAD)
        preds = logits.argmax(dim=-1)  # [B, S]
        mask = target != PAD
        correct = (preds == target) & mask
        if mask.sum() > 0:
            acc = correct.sum().float() / mask.sum().float()
            self.log("val_token_acc", acc, prog_bar=True, sync_dist=True)

        # EOS accuracy: how often we predict EOS where target is EOS
        eos_mask = target == EOS
        if eos_mask.sum() > 0:
            eos_correct = (preds == EOS) & eos_mask
            eos_acc = eos_correct.sum().float() / eos_mask.sum().float()
            self.log("val_eos_acc", eos_acc, sync_dist=True)

    def configure_optimizers(self) -> dict:
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )

        warmup_steps = self.hparams.warmup_steps
        lr_min_ratio = self.hparams.lr_min_ratio

        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return step / max(1, warmup_steps)
            total = self.trainer.estimated_stepping_batches - warmup_steps
            progress = (step - warmup_steps) / max(1, total)
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            # Floor at lr_min_ratio so LR never decays to zero
            return max(lr_min_ratio, cosine)

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }
