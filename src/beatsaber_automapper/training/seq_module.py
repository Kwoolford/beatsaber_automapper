"""Lightning module for Stage 2 (V6): swing-event sequence generation.

Wraps AudioEncoder + SequenceModel for teacher-forced training with
cross-entropy loss over the V6 swing-event vocabulary.

V6 changes vs V5:
- Default vocab_size: 183 → 118 (swing-event grammar)
- Default BOS token: 3 → 1 (V6 swing_tokenizer.BOS)
- Removed aux losses: flow, ergo, follow-through, intra-onset-parity.
  These bandaids existed only because the chord grammar hid physics.
  Parity is now structural; follow-through is naturally enforced by the
  per-hand swing stream.
- Added: saber_state conditioning, phrase_emb conditioning, mapper_id emb.
- Added: phrase_energy_alpha for phrase-level density matching (V6-4).
- HAND tokens (3, 4, 5) are rhythm-weighted like event-type tokens.
"""

from __future__ import annotations

import logging
import math

import lightning
import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812

from beatsaber_automapper.data.swing_tokenizer import (
    BOS,
    EOS,
    HAND_LEFT,
    HAND_NONE,
    HAND_RIGHT,
    PAD,
)
from beatsaber_automapper.data.swing_tokenizer import (
    VOCAB_SIZE as SWING_VOCAB_SIZE,
)
from beatsaber_automapper.models.audio_encoder import AudioEncoder
from beatsaber_automapper.models.sequence_model import ActivityPredictor, SequenceModel

logger = logging.getLogger(__name__)

# V6 rhythm tokens: HAND tokens control WHEN and WHICH HAND fires — highest
# information density, so they get elevated loss weight.
_V6_RHYTHM_TOKENS = frozenset({EOS, HAND_LEFT, HAND_RIGHT, HAND_NONE})

# HAND token IDs that represent actual saber cuts (left + right, not NONE)
_SWING_HAND_IDS = [HAND_LEFT, HAND_RIGHT]


def _compute_dt_density_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    target_p_zero: float = 0.20,
) -> torch.Tensor:
    """Hinge penalty on excess P(Δt=0) at timing token positions.

    In training data, Δt=0 legitimately occurs for same-beat hand pairs (chords).
    But the model tends to over-apply Δt=0, producing event bursts at a single beat
    instead of spreading notes across the song.  This loss penalises whenever the
    model's predicted P(Δt=0) exceeds ``target_p_zero`` at DT token positions.

    Args:
        logits: Model output logits [B, S, V].
        target: Ground-truth target token IDs [B, S].
        target_p_zero: Maximum allowed expected rate of Δt=0 events (default 0.20).

    Returns:
        Scalar hinge loss (mean over DT positions; 0 when all positions are below cap).
    """
    from beatsaber_automapper.data.swing_tokenizer import DT_BASE, DT_COUNT

    B, S, V = logits.shape
    dt_mask = (target >= DT_BASE) & (target < DT_BASE + DT_COUNT)  # [B, S]
    n_dt = dt_mask.sum().item()
    if n_dt == 0:
        return torch.tensor(0.0, device=logits.device)

    flat_logits = logits.reshape(-1, V)
    flat_mask = dt_mask.reshape(-1)
    dt_logits = flat_logits[flat_mask][:, DT_BASE : DT_BASE + DT_COUNT]  # [N_dt, 32]
    dt_probs = F.softmax(dt_logits.float(), dim=-1)
    p_zero = dt_probs[:, 0]  # P(Δt=0) per position
    excess = (p_zero - target_p_zero).clamp(min=0.0)
    return excess.mean()


def _compute_phrase_energy_loss(
    logits: torch.Tensor,
    audio_signal: torch.Tensor,
    n_bins: int = 4,
) -> torch.Tensor:
    """KL divergence between predicted swing density and audio energy per time bin.

    Divides the token sequence and the audio time axis into ``n_bins`` equal
    segments and compares them as distributions. Encourages the model to
    concentrate swing events where the audio is energetic, not arbitrarily.

    Args:
        logits: Model output logits [B, S, V].
        audio_signal: Either a mel spectrogram [B, n_mels, T] (energy = mean
            across mel bands) or a structure-feature tensor [B, C, T] where
            channel 0 is RMS energy. Auto-detected from shape.
        n_bins: Number of time bins to compare (default 4).

    Returns:
        Scalar KL-divergence loss (batchmean reduction).
    """
    b, s, v = logits.shape
    t = audio_signal.shape[2]

    # Soft probability of emitting a swing token (LEFT or RIGHT) at each position
    probs = torch.softmax(logits, dim=-1)
    swing_prob = probs[..., HAND_LEFT] + probs[..., HAND_RIGHT]  # [B, S]

    # Predicted swing density per token segment
    seg_s = max(1, s // n_bins)
    pred_segs = torch.stack(
        [swing_prob[:, i * seg_s : (i + 1) * seg_s].mean(dim=1) for i in range(n_bins)],
        dim=1,
    )  # [B, n_bins]

    # Audio energy curve: for mel input use mean across bands; for structure
    # features use channel 0 (RMS). Structure has N_STRUCTURE_FEATURES=8 channels;
    # mel has n_mels (default 80). Use a threshold that's safe for any n_mels >= 16.
    if audio_signal.shape[1] > 8:
        energy = audio_signal.mean(dim=1)  # [B, T]
    else:
        energy = audio_signal[:, 0, :]  # structure: channel 0 = RMS

    seg_t = max(1, t // n_bins)
    audio_segs = torch.stack(
        [energy[:, i * seg_t : (i + 1) * seg_t].mean(dim=1) for i in range(n_bins)],
        dim=1,
    )  # [B, n_bins]

    pred_dist = F.softmax(pred_segs + 1e-8, dim=1)
    audio_dist = F.softmax(audio_segs.detach() + 1e-8, dim=1)

    return F.kl_div(pred_dist.log(), audio_dist, reduction="batchmean")


def _build_token_weights(
    vocab_size: int,
    rhythm_weight: float = 3.0,
    eos_weight: float = 1.0,
    bomb_hand_weight: float = 1.0,
) -> torch.Tensor:
    """Build per-token loss weights with higher weight on rhythm tokens.

    For V6, rhythm tokens are EOS + the three HAND tokens (LEFT/RIGHT/NONE).
    These tokens control the timing and hand assignment of each swing event.
    HAND_NONE (bombs) gets a separate weight because bombs are rare in Expert
    maps; letting it share rhythm_weight with HAND_LEFT/RIGHT causes the model
    to over-generate bombs as a low-effort way to satisfy the grammar.
    """
    weights = torch.ones(vocab_size)
    for token_id in _V6_RHYTHM_TOKENS:
        if 0 <= token_id < vocab_size:
            weights[token_id] = rhythm_weight
    if 0 <= EOS < vocab_size:
        weights[EOS] = eos_weight
    if 0 <= HAND_NONE < vocab_size:
        weights[HAND_NONE] = bomb_hand_weight
    weights[PAD] = 0.0
    return weights


class SequenceLitModule(lightning.LightningModule):
    """Lightning training module for V6 swing-event sequence generation.

    Handles training step, validation step, optimizer configuration,
    and metric logging for Stage 2.

    Args:
        n_mels: Number of mel bands for audio encoder.
        encoder_d_model: Audio encoder model dimension.
        encoder_nhead: Audio encoder attention heads.
        encoder_num_layers: Audio encoder transformer layers.
        encoder_dim_feedforward: Audio encoder FFN dimension.
        encoder_dropout: Audio encoder dropout.
        vocab_size: Token vocabulary size (118 for V6 swing grammar).
        seq_d_model: Sequence model dimension.
        seq_nhead: Sequence model attention heads.
        seq_num_layers: Sequence model transformer layers.
        seq_dim_feedforward: Sequence model FFN dimension.
        seq_num_difficulties: Number of difficulty levels.
        seq_num_genres: Number of genre classes.
        seq_num_mappers: Number of mapper/cohort IDs (0 = disabled).
        seq_dropout: Sequence model dropout.
        conditioning_dropout: Dropout on discrete conditioning embeddings.
        bos_token_id: BOS token ID (1 for V6 swing grammar).
        label_smoothing: Label smoothing for cross-entropy loss.
        rhythm_weight: Loss weight multiplier for HAND + EOS tokens.
        eos_weight: Loss weight for EOS token.
        learning_rate: Peak learning rate.
        weight_decay: AdamW weight decay.
        warmup_steps: Linear LR warmup steps.
        lr_min_ratio: Minimum LR ratio after cosine decay.
        token_dropout: Probability of replacing an input token with a random
            one during teacher forcing (reduces exposure bias).
        freeze_encoder: Freeze audio encoder weights.
        phrase_energy_alpha: Weight for phrase-energy KL aux loss (V6-4).
            0.0 = disabled (default until V6-4 is implemented).
        n_structure_features: Per-frame structure feature channels.
        prev_context_k: Legacy V5 inter-onset context (0 = disabled).
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
        vocab_size: int = SWING_VOCAB_SIZE,
        seq_d_model: int = 512,
        seq_nhead: int = 8,
        seq_num_layers: int = 8,
        seq_dim_feedforward: int = 2048,
        seq_num_difficulties: int = 5,
        seq_num_genres: int = 11,
        seq_num_mappers: int = 0,
        seq_dropout: float = 0.1,
        # V6 conditioning
        bos_token_id: int = BOS,
        conditioning_dropout: float = 0.0,
        # Training params
        label_smoothing: float = 0.1,
        rhythm_weight: float = 3.0,
        eos_weight: float = 1.0,
        bomb_hand_weight: float = 1.0,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        warmup_steps: int = 2000,
        lr_min_ratio: float = 0.01,
        token_dropout: float = 0.0,
        freeze_encoder: bool = False,
        # V6 aux losses
        phrase_energy_alpha: float = 0.0,  # V6-4: phrase density KL loss
        dt_density_alpha: float = 0.0,     # V6-8: Δt=0 over-concentration penalty
        activity_alpha: float = 0.0,       # V6-7: activity prediction aux loss
        # Section conditioning
        n_section_types: int = 6,
        # Structure features
        n_structure_features: int = 8,
        # Legacy V5 compat
        prev_context_k: int = 0,
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
            num_mappers=seq_num_mappers,
            num_sections=n_section_types,
            dropout=seq_dropout,
            conditioning_dropout=conditioning_dropout,
            prev_context_k=prev_context_k,
        )

        # V6-7: activity predictor — predicts which beat slots should have notes
        self.activity_predictor: ActivityPredictor | None = (
            ActivityPredictor(d_model=seq_d_model) if activity_alpha > 0 else None
        )

        if freeze_encoder:
            for param in self.audio_encoder.parameters():
                param.requires_grad = False

        token_weights = _build_token_weights(vocab_size, rhythm_weight, eos_weight, bomb_hand_weight)
        self.register_buffer("token_weights", token_weights)

        self.loss_fn = nn.CrossEntropyLoss(
            weight=token_weights,
            ignore_index=PAD,
            label_smoothing=label_smoothing,
        )

    def _prepare_teacher_forcing(
        self, tokens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Split a BOS-prefixed sequence into decoder input and target.

        Dataset tokens are [BOS, t0, t1, ..., tN, PAD...].  Standard LM shift:
          decoder_input = tokens[:, :-1]  = [BOS, t0, t1, ..., tN]
          target        = tokens[:, 1:]   = [t0,  t1, ..., tN, PAD]

        The old implementation prepended an extra BOS, producing double-BOS at
        position 0 and shifting saber_state out of alignment with decoder_input.
        """
        return tokens[:, :-1], tokens[:, 1:]

    def forward(
        self,
        mel: torch.Tensor,
        tokens: torch.Tensor,
        difficulty: torch.Tensor,
        genre: torch.Tensor,
        structure: torch.Tensor | None = None,
        saber_state: torch.Tensor | None = None,
        phrase_mel: torch.Tensor | None = None,
        mapper_id: torch.Tensor | None = None,
        song_pos_frac: torch.Tensor | None = None,
        section_id: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass: mel → audio features → swing-event logits.

        Args:
            mel: Mel spectrogram [B, n_mels, T].
            tokens: Decoder input tokens [B, S] (already BOS-prepended).
            difficulty: Difficulty indices [B].
            genre: Genre indices [B].
            structure: Optional per-frame structure features [B, 8, T].
            saber_state: Optional saber state per step [B, S, 12].
            phrase_mel: Optional wide-context mel for phrase embedding [B, n_mels, T_phrase].
            mapper_id: Optional mapper/cohort index [B].
            song_pos_frac: Optional song position fraction [B] in [0, 1].
            section_id: Optional section type index [B] in [0, n_section_types-1].

        Returns:
            Logits [B, S, vocab_size].
        """
        audio_features = self.audio_encoder(mel, structure_features=structure)

        # Phrase embedding: mean-pool wide audio context → d_model
        phrase_emb: torch.Tensor | None = None
        if phrase_mel is not None:
            phrase_audio = self.audio_encoder(phrase_mel)   # [B, T_phrase, d_model]
            phrase_emb = phrase_audio.mean(dim=1)           # [B, d_model]

        return self.sequence_model(
            tokens, audio_features, difficulty, genre,
            saber_state=saber_state,
            phrase_emb=phrase_emb,
            mapper_id=mapper_id,
            song_pos_frac=song_pos_frac,
            section_id=section_id,
        )

    def _encode_audio(
        self, batch: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Encode mel + structure once and compute phrase embedding."""
        audio_features = self.audio_encoder(
            batch["mel"], structure_features=batch.get("structure")
        )
        phrase_emb: torch.Tensor | None = None
        if batch.get("phrase_mel") is not None:
            phrase_audio = self.audio_encoder(batch["phrase_mel"])
            phrase_emb = phrase_audio.mean(dim=1)
        return audio_features, phrase_emb

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        decoder_input, target = self._prepare_teacher_forcing(batch["tokens"])

        # saber_state is [B, max_swing_len, 12]; slice to match decoder_input length.
        saber_state = batch.get("saber_state")
        if saber_state is not None:
            saber_state = saber_state[:, :-1, :]

        # Token dropout: replace random input tokens to reduce exposure bias
        if self.hparams.token_dropout > 0 and self.training:
            dropout_mask = torch.rand_like(decoder_input.float()) < self.hparams.token_dropout
            dropout_mask[:, 0] = False  # never mask BOS
            decoder_input = decoder_input.clone()
            decoder_input[dropout_mask] = torch.randint(
                1, self.hparams.vocab_size, (int(dropout_mask.sum().item()),),
                device=decoder_input.device,
            )

        # Encode audio once — reused by both sequence model and activity predictor
        audio_features, phrase_emb = self._encode_audio(batch)

        logits = self.sequence_model(
            decoder_input, audio_features, batch["difficulty"], batch["genre"],
            saber_state=saber_state,
            phrase_emb=phrase_emb,
            mapper_id=batch.get("mapper_id"),
            song_pos_frac=batch.get("song_pos_frac"),
            section_id=batch.get("section_id"),
        )

        loss = self.loss_fn(logits.reshape(-1, logits.size(-1)), target.reshape(-1))

        # V6-8: Δt=0 density hinge loss (penalise event bursts at a single beat)
        if self.hparams.dt_density_alpha > 0:
            dt_loss = _compute_dt_density_loss(logits, target)
            loss = loss + self.hparams.dt_density_alpha * dt_loss
            self.log("train_dt_density_loss", dt_loss, prog_bar=False)

        # V6-4: phrase-energy KL loss
        if self.hparams.phrase_energy_alpha > 0:
            audio_signal = batch.get("phrase_mel")
            if audio_signal is None:
                audio_signal = batch.get("structure")
            if audio_signal is not None:
                pe_loss = _compute_phrase_energy_loss(logits, audio_signal)
                loss = loss + self.hparams.phrase_energy_alpha * pe_loss
                self.log("train_phrase_energy_loss", pe_loss, prog_bar=False)

        # V6-7: activity prediction aux loss
        if self.activity_predictor is not None and self.hparams.activity_alpha > 0:
            act_labels = batch.get("activity_labels")
            if act_labels is not None:
                act_logits = self.activity_predictor(audio_features)  # [B, N_BEATS]
                act_loss = F.binary_cross_entropy_with_logits(act_logits, act_labels)
                loss = loss + self.hparams.activity_alpha * act_loss
                self.log("train_activity_loss", act_loss, prog_bar=False)

        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> None:
        decoder_input, target = self._prepare_teacher_forcing(batch["tokens"])
        saber_state = batch.get("saber_state")
        if saber_state is not None:
            saber_state = saber_state[:, :-1, :]

        audio_features, phrase_emb = self._encode_audio(batch)

        logits = self.sequence_model(
            decoder_input, audio_features, batch["difficulty"], batch["genre"],
            saber_state=saber_state,
            phrase_emb=phrase_emb,
            mapper_id=batch.get("mapper_id"),
            song_pos_frac=batch.get("song_pos_frac"),
            section_id=batch.get("section_id"),
        )
        loss = self.loss_fn(logits.reshape(-1, logits.size(-1)), target.reshape(-1))
        self.log("val_loss", loss, prog_bar=True, sync_dist=True)

        preds = logits.argmax(dim=-1)
        mask = target != PAD
        if mask.sum() > 0:
            acc = (preds == target)[mask].float().mean()
            self.log("val_token_acc", acc, prog_bar=True, sync_dist=True)

        eos_mask = target == EOS
        if eos_mask.sum() > 0:
            eos_acc = ((preds == EOS) & eos_mask).float().sum() / eos_mask.float().sum()
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
            return max(lr_min_ratio, cosine)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda),
                             "interval": "step", "frequency": 1},
        }
