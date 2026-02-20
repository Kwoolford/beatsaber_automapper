"""Lightning module for Stage 3: Lighting generation training.

Wraps the AudioEncoder + LightingModel for teacher-forced training
with cross-entropy loss over the lighting token vocabulary. BOS is
prepended to the lighting token sequence for the decoder input;
original tokens serve as targets.
"""

from __future__ import annotations

import logging
import math

import lightning
import torch
import torch.nn as nn

from beatsaber_automapper.data.tokenizer import LIGHT_BOS, LIGHT_PAD, LIGHT_VOCAB_SIZE, VOCAB_SIZE
from beatsaber_automapper.models.audio_encoder import AudioEncoder
from beatsaber_automapper.models.lighting_model import LightingModel

logger = logging.getLogger(__name__)


class LightingLitModule(lightning.LightningModule):
    """Lightning training module for lighting event generation.

    Handles training step, validation step, optimizer configuration,
    and metric logging for Stage 3.

    Args:
        n_mels: Number of mel bands for audio encoder.
        encoder_d_model: Audio encoder model dimension.
        encoder_nhead: Audio encoder attention heads.
        encoder_num_layers: Audio encoder transformer layers.
        encoder_dim_feedforward: Audio encoder FFN dimension.
        encoder_dropout: Audio encoder dropout.
        light_vocab_size: Size of the lighting token vocabulary.
        note_vocab_size: Size of the note token vocabulary.
        light_d_model: Lighting model dimension.
        light_nhead: Lighting model attention heads.
        light_num_layers: Lighting model transformer layers.
        light_dim_feedforward: Lighting model FFN dimension.
        light_num_genres: Number of genre classes.
        light_dropout: Lighting model dropout.
        label_smoothing: Label smoothing for cross-entropy loss.
        learning_rate: Peak learning rate.
        weight_decay: AdamW weight decay.
        warmup_steps: Linear warmup steps.
        freeze_encoder: Whether to freeze audio encoder weights during training.
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
        # Lighting model params
        light_vocab_size: int = LIGHT_VOCAB_SIZE,
        note_vocab_size: int = VOCAB_SIZE,
        light_d_model: int = 512,
        light_nhead: int = 8,
        light_num_layers: int = 4,
        light_dim_feedforward: int = 2048,
        light_num_genres: int = 11,
        light_dropout: float = 0.1,
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
        self.lighting_model = LightingModel(
            light_vocab_size=light_vocab_size,
            note_vocab_size=note_vocab_size,
            d_model=light_d_model,
            nhead=light_nhead,
            num_layers=light_num_layers,
            dim_feedforward=light_dim_feedforward,
            num_genres=light_num_genres,
            dropout=light_dropout,
        )

        if freeze_encoder:
            for param in self.audio_encoder.parameters():
                param.requires_grad = False

        self.loss_fn = nn.CrossEntropyLoss(
            ignore_index=LIGHT_PAD, label_smoothing=label_smoothing
        )

    def _prepare_teacher_forcing(
        self, tokens: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Prepend BOS to tokens[:-1] as decoder input; original tokens as target.

        Args:
            tokens: Raw lighting token sequences [B, S].

        Returns:
            Tuple of (decoder_input [B, S], target [B, S]).
        """
        b, s = tokens.shape
        bos = torch.full((b, 1), LIGHT_BOS, dtype=tokens.dtype, device=tokens.device)
        decoder_input = torch.cat([bos, tokens[:, :-1]], dim=1)
        return decoder_input, tokens

    def forward(
        self,
        mel: torch.Tensor,
        light_tokens: torch.Tensor,
        note_tokens: torch.Tensor,
        genre: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass: mel -> audio features -> lighting logits.

        Args:
            mel: Mel spectrogram [B, n_mels, T].
            light_tokens: Decoder input lighting tokens [B, S] (BOS-prepended).
            note_tokens: Note token context [B, N].
            genre: Genre indices [B].

        Returns:
            Lighting logits [B, S, light_vocab_size].
        """
        audio_features = self.audio_encoder(mel)
        return self.lighting_model(light_tokens, audio_features, note_tokens, genre)

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        decoder_input, target = self._prepare_teacher_forcing(batch["light_tokens"])
        logits = self(batch["mel"], decoder_input, batch["note_tokens"], batch["genre"])
        loss = self.loss_fn(logits.reshape(-1, logits.size(-1)), target.reshape(-1))
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> None:
        decoder_input, target = self._prepare_teacher_forcing(batch["light_tokens"])
        logits = self(batch["mel"], decoder_input, batch["note_tokens"], batch["genre"])
        loss = self.loss_fn(logits.reshape(-1, logits.size(-1)), target.reshape(-1))
        self.log("val_loss", loss, prog_bar=True, sync_dist=True)

        # Token accuracy (ignoring PAD)
        preds = logits.argmax(dim=-1)
        mask = target != LIGHT_PAD
        correct = (preds == target) & mask
        if mask.sum() > 0:
            acc = correct.sum().float() / mask.sum().float()
            self.log("val_token_acc", acc, prog_bar=True, sync_dist=True)

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
