"""V7-5: Lightning module for Stage 2 LayoutModel training.

Cross-entropy loss over spatial tokens only (KIND, X, Y, DIR, FIELD_D).
The HAND and Δt tokens are absent — timing is provided externally by Stage 1.
"""

from __future__ import annotations

import math
import logging

import lightning
import torch
import torch.nn as nn
import torch.nn.functional as F

from beatsaber_automapper.models.layout_model import LayoutModel, LAYOUT_PAD

logger = logging.getLogger(__name__)


class LayoutLitModule(lightning.LightningModule):
    """Lightning module for Stage 2 spatial layout generation.

    Args:
        vocab_size:      Full swing-event vocab size (118).
        d_model:         Transformer hidden size.
        n_heads:         Attention heads.
        n_layers:        Decoder layers.
        dim_feedforward: FFN width.
        max_len:         Max sequence length.
        dropout:         Dropout.
        num_difficulties: Number of difficulty levels.
        num_genres:       Number of genre classes.
        learning_rate:   Peak LR.
        weight_decay:    AdamW weight decay.
        warmup_steps:    LR warmup steps.
        label_smoothing: CE label smoothing.
    """

    def __init__(
        self,
        vocab_size: int = 118,
        d_model: int = 512,
        n_heads: int = 8,
        n_layers: int = 4,
        dim_feedforward: int = 2048,
        max_len: int = 64,
        dropout: float = 0.1,
        num_difficulties: int = 5,
        num_genres: int = 11,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        warmup_steps: int = 1000,
        label_smoothing: float = 0.1,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.model = LayoutModel(
            vocab_size=vocab_size,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            dim_feedforward=dim_feedforward,
            max_len=max_len,
            dropout=dropout,
            num_difficulties=num_difficulties,
            num_genres=num_genres,
        )

        self.loss_fn = nn.CrossEntropyLoss(
            ignore_index=LAYOUT_PAD,
            label_smoothing=label_smoothing,
        )

    def _forward_batch(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
        tokens      = batch["layout_tokens"]    # [B, max_len]
        dec_input   = tokens[:, :-1]
        target      = tokens[:, 1:]

        logits = self.model(
            decoder_input=dec_input,
            local_mert=  batch["local_mert"],
            song_emb=    batch["song_emb"],
            section_emb= batch["section_emb"],
            saber_state= batch["saber_state"],
            phrase_feat= batch["phrase_feat"],
            difficulty=  batch["difficulty"],
            genre=       batch["genre"],
        )   # [B, S-1, vocab_size]

        loss = self.loss_fn(
            logits.reshape(-1, logits.size(-1)),
            target.reshape(-1),
        )
        return logits, loss

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        _, loss = self._forward_batch(batch)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch: dict, batch_idx: int) -> None:
        logits, loss = self._forward_batch(batch)
        self.log("val_loss", loss, prog_bar=True, sync_dist=True)

        # Token accuracy (excluding PAD)
        target = batch["layout_tokens"][:, 1:]
        preds  = logits.argmax(dim=-1)
        mask   = target != LAYOUT_PAD
        if mask.sum() > 0:
            acc = (preds == target)[mask].float().mean()
            self.log("val_token_acc", acc, prog_bar=True, sync_dist=True)

    def configure_optimizers(self) -> dict:
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )
        warmup = self.hparams.warmup_steps

        def lr_lambda(step: int) -> float:
            if step < warmup:
                return step / max(1, warmup)
            total = max(1, self.trainer.estimated_stepping_batches - warmup)
            progress = (step - warmup) / total
            return max(0.01, 0.5 * (1.0 + math.cos(math.pi * progress)))

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda),
                             "interval": "step"},
        }
