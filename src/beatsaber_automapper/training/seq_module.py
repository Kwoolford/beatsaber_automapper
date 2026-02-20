"""Lightning module for Stage 2: Note sequence generation training.

Wraps the AudioEncoder + SequenceModel for teacher-forced training
with cross-entropy loss over the token vocabulary. BOS is prepended
to create decoder input; original tokens serve as targets.
"""

from __future__ import annotations

import logging
import math

import lightning
import torch
import torch.nn as nn

from beatsaber_automapper.data.tokenizer import BOS, PAD
from beatsaber_automapper.models.audio_encoder import AudioEncoder
from beatsaber_automapper.models.sequence_model import SequenceModel

logger = logging.getLogger(__name__)


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
        label_smoothing: Label smoothing for cross-entropy loss.
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
        # Training params
        label_smoothing: float = 0.1,
        learning_rate: float = 3e-4,
        weight_decay: float = 0.01,
        warmup_steps: int = 1000,
        freeze_encoder: bool = False,
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
        )

        if freeze_encoder:
            for param in self.audio_encoder.parameters():
                param.requires_grad = False

        self.loss_fn = nn.CrossEntropyLoss(ignore_index=PAD, label_smoothing=label_smoothing)

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
    ) -> torch.Tensor:
        """Forward pass: mel -> audio features -> token logits.

        Args:
            mel: Mel spectrogram [B, n_mels, T].
            tokens: Decoder input tokens [B, S] (already BOS-prepended).
            difficulty: Difficulty indices [B].
            genre: Genre indices [B].

        Returns:
            Token logits [B, S, vocab_size].
        """
        audio_features = self.audio_encoder(mel)
        return self.sequence_model(tokens, audio_features, difficulty, genre)

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        decoder_input, target = self._prepare_teacher_forcing(batch["tokens"])
        logits = self(batch["mel"], decoder_input, batch["difficulty"], batch["genre"])
        # logits: [B, S, V], target: [B, S]
        loss = self.loss_fn(logits.reshape(-1, logits.size(-1)), target.reshape(-1))
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> None:
        decoder_input, target = self._prepare_teacher_forcing(batch["tokens"])
        logits = self(batch["mel"], decoder_input, batch["difficulty"], batch["genre"])
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
        from beatsaber_automapper.data.tokenizer import EOS

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
