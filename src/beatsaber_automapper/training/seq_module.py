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
    DIR_OFFSET,
    EOS,
    NOTE,
    PAD,
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
    vocab_size: int, rhythm_weight: float = 3.0, eos_weight: float = 0.3,
) -> torch.Tensor:
    """Build per-token loss weights with higher weight on rhythm tokens.

    Args:
        vocab_size: Size of token vocabulary.
        rhythm_weight: Weight multiplier for timing-sensitive tokens.
        eos_weight: Weight for EOS token (low to combat over-prediction).

    Returns:
        Weight tensor [vocab_size] with 1.0 for most tokens,
        rhythm_weight for timing-sensitive tokens, and eos_weight for EOS.
    """
    weights = torch.ones(vocab_size)
    for token_id in _RHYTHM_TOKENS:
        if 0 <= token_id < vocab_size:
            weights[token_id] = rhythm_weight
    # EOS weight: training data has no empty onsets (preprocessing filters them),
    # so eos_weight=1.0 is appropriate. min_length at inference prevents premature EOS.
    if 0 <= EOS < vocab_size:
        weights[EOS] = eos_weight
    # PAD should be ignored entirely (via ignore_index), but set to 0 for safety
    weights[PAD] = 0.0
    return weights


# Direction parity classes for flow-aware loss
# Forehand swings: up, down-left, down-right (directions 0, 6, 7 — but parity is
# based on hand movement, so we group by swing direction)
_FOREHAND_DIRS = frozenset({1, 6, 7})  # down, down-left, down-right
_BACKHAND_DIRS = frozenset({0, 4, 5})  # up, up-left, up-right


def _compute_flow_loss(
    pred_tokens: torch.Tensor,
    prev_tokens: torch.Tensor | None,
    time_gap: torch.Tensor | None,
) -> torch.Tensor:
    """Compute flow parity auxiliary loss on predicted tokens.

    Rewards alternating forehand/backhand swings (good flow) and penalizes
    same-parity consecutive swings (parity violation). Time gaps > 3 seconds
    reset the parity check.

    Args:
        pred_tokens: Argmax-predicted tokens [B, S].
        prev_tokens: Previous onset tokens [B, K, S] or None.
        time_gap: Seconds since previous onset [B] or None.

    Returns:
        Scalar loss (penalty for parity violations).
    """
    if prev_tokens is None or time_gap is None:
        return torch.tensor(0.0, device=pred_tokens.device)

    b = pred_tokens.shape[0]
    violations = 0.0
    count = 0

    for i in range(b):
        # Skip if time gap > 3 seconds (sequence restart)
        if time_gap[i].item() > 3.0:
            continue

        # Get last previous onset's first direction
        last_prev = prev_tokens[i, -1]  # [S] — last of K previous onsets
        prev_dir = _extract_first_direction(last_prev)
        if prev_dir is None:
            continue

        # Get current onset's first direction
        curr_dir = _extract_first_direction(pred_tokens[i])
        if curr_dir is None:
            continue

        # Check parity
        prev_is_forehand = prev_dir in _FOREHAND_DIRS
        curr_is_forehand = curr_dir in _FOREHAND_DIRS

        # "any" direction (8) — skip
        if prev_dir == 8 or curr_dir == 8:
            continue

        # Horizontal (left=2, right=3) — neutral, skip
        if prev_dir in (2, 3) or curr_dir in (2, 3):
            continue

        count += 1
        if prev_is_forehand == curr_is_forehand:
            violations += 1.0  # Same parity = bad flow

    if count == 0:
        return torch.tensor(0.0, device=pred_tokens.device)

    return torch.tensor(violations / count, device=pred_tokens.device)


def _extract_first_direction(tokens: torch.Tensor) -> int | None:
    """Extract the first direction value from a token sequence.

    Looks for NOTE token (token 3) and extracts direction at offset +4.

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
        vocab_size: int = 167,
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
        learning_rate: float = 3e-4,
        weight_decay: float = 0.01,
        warmup_steps: int = 1000,
        freeze_encoder: bool = False,
        # Flow-aware auxiliary loss
        flow_loss_alpha: float = 0.0,
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

        if freeze_encoder:
            for param in self.audio_encoder.parameters():
                param.requires_grad = False

        # Build per-token loss weights with rhythm emphasis + EOS downweighting
        token_weights = _build_token_weights(vocab_size, rhythm_weight, eos_weight=eos_weight)
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
    ) -> torch.Tensor:
        """Forward pass: mel -> audio features -> token logits.

        Args:
            mel: Mel spectrogram [B, n_mels, T].
            tokens: Decoder input tokens [B, S] (already BOS-prepended).
            difficulty: Difficulty indices [B].
            genre: Genre indices [B].
            structure: Optional structure features [B, 6, T].
            prev_tokens: Optional previous onset tokens [B, K, S] for inter-onset context.

        Returns:
            Token logits [B, S, vocab_size].
        """
        audio_features = self.audio_encoder(mel, structure_features=structure)
        return self.sequence_model(
            tokens, audio_features, difficulty, genre, prev_tokens=prev_tokens
        )

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        decoder_input, target = self._prepare_teacher_forcing(batch["tokens"])
        structure = batch.get("structure", None)
        prev_tokens = batch.get("prev_tokens", None)
        logits = self(
            batch["mel"], decoder_input, batch["difficulty"], batch["genre"],
            structure=structure, prev_tokens=prev_tokens,
        )
        # logits: [B, S, V], target: [B, S]
        ce_loss = self.loss_fn(logits.reshape(-1, logits.size(-1)), target.reshape(-1))

        # Flow-aware auxiliary loss
        alpha = self.hparams.flow_loss_alpha
        if alpha > 0 and prev_tokens is not None:
            preds = logits.argmax(dim=-1).detach()
            time_gap = batch.get("time_gap", None)
            flow_loss = _compute_flow_loss(preds, prev_tokens, time_gap)
            loss = ce_loss + alpha * flow_loss
            self.log("train_flow_loss", flow_loss, prog_bar=False)
        else:
            loss = ce_loss

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

        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return step / max(1, warmup_steps)
            total = self.trainer.estimated_stepping_batches - warmup_steps
            progress = (step - warmup_steps) / max(1, total)
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }
