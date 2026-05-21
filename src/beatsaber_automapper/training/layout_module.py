"""V7-5b: Lightning module for phrase-level Stage 2 training.

CE loss over the next-token target sequence (PAD positions masked via
IGNORE_INDEX in the target tensor). Logs token accuracy and role-specific
accuracy so we can see *which kind of token* the model is bad at (KIND? DIR?
FIELD_D?) rather than just an aggregate number.
"""

from __future__ import annotations

import logging
import math

import lightning
import torch
import torch.nn as nn
import torch.nn.functional as F

from beatsaber_automapper.data.layout_dataset import (
    IGNORE_INDEX, LAYOUT_PAD, LAYOUT_VOCAB_SIZE,
    N_ROLES, ROLE_DIR, ROLE_FIELD_D, ROLE_KIND, ROLE_X, ROLE_Y,
)
from beatsaber_automapper.models.layout_model import LayoutPhraseModel

logger = logging.getLogger(__name__)


class LayoutPhraseLitModule(lightning.LightningModule):
    """Lightning module for phrase-level layout training.

    Args:
        vocab_size, d_model, n_heads, n_enc_layers, n_dec_layers,
        dim_feedforward, max_layout_len, max_phrase_slots, num_difficulties,
        num_genres, dropout: forwarded to LayoutPhraseModel.
        learning_rate, weight_decay, warmup_steps, label_smoothing: optim hparams.
    """

    def __init__(
        self,
        vocab_size:        int = LAYOUT_VOCAB_SIZE,
        d_model:           int = 384,
        n_heads:           int = 6,
        n_enc_layers:      int = 3,
        n_dec_layers:      int = 4,
        dim_feedforward:   int = 1536,
        max_layout_len:    int = 384,
        max_phrase_slots:  int = 96,
        num_difficulties:  int = 5,
        num_genres:        int = 11,
        dropout:           float = 0.1,
        learning_rate:     float = 1e-4,
        weight_decay:      float = 0.01,
        warmup_steps:      int = 1000,
        label_smoothing:   float = 0.1,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.model = LayoutPhraseModel(
            vocab_size=vocab_size, d_model=d_model, n_heads=n_heads,
            n_enc_layers=n_enc_layers, n_dec_layers=n_dec_layers,
            dim_feedforward=dim_feedforward, max_layout_len=max_layout_len,
            max_phrase_slots=max_phrase_slots, num_difficulties=num_difficulties,
            num_genres=num_genres, dropout=dropout,
        )
        self.loss_fn = nn.CrossEntropyLoss(
            ignore_index=IGNORE_INDEX, label_smoothing=label_smoothing,
        )

    # ------------------------------------------------------------------
    def _forward_batch(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self.model(
            layout_tokens=batch["layout_tokens"],
            token_slot   =batch["token_slot"],
            token_hand   =batch["token_hand"],
            token_role   =batch["token_role"],
            phrase_mert  =batch["phrase_mert"],
            phrase_mask  =batch["phrase_mask"],
            difficulty   =batch["difficulty"],
            genre        =batch["genre"],
        )   # [B, S, vocab]
        target = batch["target"]   # [B, S], IGNORE_INDEX on PAD
        loss = self.loss_fn(logits.reshape(-1, logits.size(-1)), target.reshape(-1))
        return logits, loss

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        _, loss = self._forward_batch(batch)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch: dict, batch_idx: int) -> None:
        logits, loss = self._forward_batch(batch)
        self.log("val_loss", loss, prog_bar=True, sync_dist=True)

        target = batch["target"]
        preds  = logits.argmax(dim=-1)
        valid  = target != IGNORE_INDEX

        # Overall token accuracy
        if valid.any():
            acc = (preds == target)[valid].float().mean()
            self.log("val_token_acc", acc, prog_bar=True, sync_dist=True)

        # Per-role accuracy — diagnoses where the model is weak. We look up
        # each target's role via token_role shifted-by-one (since target =
        # input shifted-by-one, the role of target[t] is the role at the
        # output position, i.e. the input position t+1).
        role_in = batch["token_role"]   # [B, S]
        # Roles align with the same shift: target[t] = input[t+1] → role at output t.
        role_out = torch.cat([role_in[:, 1:], torch.full_like(role_in[:, :1], -1)], dim=1)

        correct = (preds == target)
        for role_id, role_name in [
            (ROLE_KIND,    "kind"),
            (ROLE_X,       "x"),
            (ROLE_Y,       "y"),
            (ROLE_DIR,     "dir"),
            (ROLE_FIELD_D, "field_d"),
        ]:
            mask = valid & (role_out == role_id)
            if mask.any():
                acc_r = correct[mask].float().mean()
                self.log(f"val_acc_{role_name}", acc_r, sync_dist=True)

    # ------------------------------------------------------------------
    def configure_optimizers(self) -> dict:
        opt = torch.optim.AdamW(
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
            "optimizer": opt,
            "lr_scheduler": {"scheduler": torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda),
                             "interval": "step"},
        }
