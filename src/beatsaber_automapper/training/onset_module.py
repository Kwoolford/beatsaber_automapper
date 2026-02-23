"""Lightning module for Stage 1: Onset prediction training.

Wraps the AudioEncoder + OnsetModel for training with binary cross-entropy
loss on Gaussian-smoothed onset labels. Logs onset F1 score on validation.
"""

from __future__ import annotations

import logging
import math

import lightning
import torch
import torch.nn as nn

from beatsaber_automapper.evaluation.metrics import onset_f1_framewise
from beatsaber_automapper.models.audio_encoder import AudioEncoder
from beatsaber_automapper.models.components import peak_picking
from beatsaber_automapper.models.onset_model import OnsetModel

logger = logging.getLogger(__name__)


class OnsetLitModule(lightning.LightningModule):
    """Lightning training module for onset prediction.

    Handles training step, validation step, optimizer configuration,
    and metric logging for Stage 1.

    Args:
        n_mels: Number of mel bands for audio encoder.
        encoder_d_model: Audio encoder model dimension.
        encoder_nhead: Audio encoder attention heads.
        encoder_num_layers: Audio encoder transformer layers.
        encoder_dim_feedforward: Audio encoder FFN dimension.
        encoder_dropout: Audio encoder dropout.
        onset_d_model: Onset model dimension (must match encoder_d_model).
        onset_nhead: Onset model attention heads.
        onset_num_layers: Onset model transformer layers.
        onset_num_difficulties: Number of difficulty levels.
        onset_num_genres: Number of genre classes.
        onset_dropout: Onset model dropout.
        pos_weight: Positive class weight for BCE loss.
        learning_rate: Peak learning rate.
        weight_decay: AdamW weight decay.
        warmup_steps: Linear warmup steps.
        onset_threshold: Peak picking threshold for validation.
        min_onset_distance: Minimum frames between predicted onsets.
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
        # Onset model params
        onset_d_model: int = 512,
        onset_nhead: int = 8,
        onset_num_layers: int = 2,
        onset_num_difficulties: int = 5,
        onset_num_genres: int = 11,
        onset_dropout: float = 0.1,
        # Training params
        pos_weight: float = 5.0,
        learning_rate: float = 3e-4,
        weight_decay: float = 0.01,
        warmup_steps: int = 1000,
        # Inference params
        onset_threshold: float = 0.5,
        min_onset_distance: int = 5,
        # Memory optimization
        use_gradient_checkpointing: bool = False,
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
            use_checkpoint=use_gradient_checkpointing,
        )
        self.onset_model = OnsetModel(
            d_model=onset_d_model,
            nhead=onset_nhead,
            num_layers=onset_num_layers,
            num_difficulties=onset_num_difficulties,
            num_genres=onset_num_genres,
            dropout=onset_dropout,
            use_checkpoint=use_gradient_checkpointing,
        )

        self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]))

    def forward(
        self, mel: torch.Tensor, difficulty: torch.Tensor, genre: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass: mel -> audio features -> onset logits.

        Args:
            mel: Mel spectrogram [B, n_mels, T].
            difficulty: Difficulty indices [B].
            genre: Genre indices [B].

        Returns:
            Onset logits [B, T].
        """
        audio_features = self.audio_encoder(mel)
        return self.onset_model(audio_features, difficulty, genre)

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        logits = self(batch["mel"], batch["difficulty"], batch["genre"])
        # Cast to float32: BCEWithLogitsLoss is numerically unstable with bf16 logits
        # (unlike CrossEntropyLoss). NaN gradients → CUDNN_STATUS_EXECUTION_FAILED.
        loss = self.loss_fn(logits.float(), batch["labels"])
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> None:
        logits = self(batch["mel"], batch["difficulty"], batch["genre"])
        loss = self.loss_fn(logits.float(), batch["labels"])
        self.log("val_loss", loss, prog_bar=True, sync_dist=True)

        # Compute onset F1 on this batch
        probs = torch.sigmoid(logits)
        threshold = self.hparams.onset_threshold
        min_dist = self.hparams.min_onset_distance

        f1_sum = 0.0
        precision_sum = 0.0
        recall_sum = 0.0
        count = 0

        for i in range(probs.shape[0]):
            pred_frames = peak_picking(probs[i], threshold=threshold, min_distance=min_dist)
            true_frames = (batch["labels"][i] > 0.5).nonzero(as_tuple=True)[0]
            metrics = onset_f1_framewise(pred_frames, true_frames, tolerance_frames=3)
            f1_sum += metrics["f1"]
            precision_sum += metrics["precision"]
            recall_sum += metrics["recall"]
            count += 1

        if count > 0:
            self.log("val_f1", f1_sum / count, prog_bar=True, sync_dist=True)
            self.log("val_precision", precision_sum / count, sync_dist=True)
            self.log("val_recall", recall_sum / count, sync_dist=True)

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
            # Cosine decay after warmup
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
