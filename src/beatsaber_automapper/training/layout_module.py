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
        x_role_weight:     float = 2.0,
        ctx_len:           int   = 0,
        max_song_phrases:  int   = 0,
        sched_sampling_start: float = 0.0,
        sched_sampling_end:   float = 0.0,
        sched_sampling_epochs: int  = 20,
        use_contour:       bool  = False,
    ) -> None:
        """
        sched_sampling_start/end/epochs: linear ramp of scheduled-sampling
        probability from `start` to `end` over `epochs` epochs. At epoch 0
        the model is fully teacher-forced (standard). At epoch N it mixes in
        its own predicted tokens with probability `end`, which reduces the
        teacher-forcing exposure-bias gap at autoregressive inference time.
        Default (all zeros) = pure teacher forcing = no change from Run 3.
        """
        super().__init__()
        self.save_hyperparameters()

        self.model = LayoutPhraseModel(
            vocab_size=vocab_size, d_model=d_model, n_heads=n_heads,
            n_enc_layers=n_enc_layers, n_dec_layers=n_dec_layers,
            dim_feedforward=dim_feedforward, max_layout_len=max_layout_len,
            max_phrase_slots=max_phrase_slots, max_song_phrases=max_song_phrases,
            num_difficulties=num_difficulties, num_genres=num_genres, dropout=dropout,
            use_contour=use_contour,
        )
        # We compute per-position CE manually so we can apply a role weight to
        # the weakest role (X). Runs 1+2 showed kind=98% / field_d=100% / y=83%
        # / dir=82% / **x=67%** — the model is plenty capable, X is just hard
        # because the same musical hit can map to several legal columns. A
        # small weight bias toward ROLE_X tells the optimizer where to spend
        # the remaining capacity.
        self.loss_fn = nn.CrossEntropyLoss(
            ignore_index=IGNORE_INDEX, label_smoothing=label_smoothing,
            reduction="none",
        )
        self.x_role_weight = x_role_weight
        self.model.ctx_len = ctx_len   # expose for inference in generate.py
        self._sched_start  = sched_sampling_start
        self._sched_end    = sched_sampling_end
        self._sched_epochs = max(1, sched_sampling_epochs)

    # ------------------------------------------------------------------
    def _sched_p(self) -> float:
        """Current scheduled-sampling probability (linear ramp over epochs)."""
        if self._sched_end <= 0:
            return 0.0
        epoch = self.current_epoch if self.trainer is not None else 0
        t = min(epoch / self._sched_epochs, 1.0)
        return self._sched_start + t * (self._sched_end - self._sched_start)

    def _forward_batch(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
        layout_tokens = batch["layout_tokens"]

        p_sched = self._sched_p()
        if p_sched > 0 and self.training:
            # Scheduled sampling: on each non-context, non-BOS position replace
            # the input token with the model's argmax with probability p_sched.
            # Step 1 — teacher-forced pass to get predicted tokens (no grad).
            with torch.no_grad():
                logits_tf = self.model(
                    layout_tokens=layout_tokens,
                    token_slot   =batch["token_slot"],
                    token_hand   =batch["token_hand"],
                    token_role   =batch["token_role"],
                    phrase_mert  =batch["phrase_mert"],
                    phrase_mask  =batch["phrase_mask"],
                    difficulty   =batch["difficulty"],
                    genre        =batch["genre"],
                    song_fps     =batch.get("song_fps"),
                    song_fp_mask =batch.get("song_fp_mask"),
                    phrase_contour=batch.get("phrase_contour"),
                )
                # Predicted token at position t is used as input at position t+1
                pred_toks = logits_tf.argmax(dim=-1)   # [B, S]
                # Shift: pred_toks[t] is the replacement for layout_tokens[t+1]
                # (position 0 = BOS, never replaced; context prefix also preserved)
                ctx_n = int(batch.get("ctx_len", torch.zeros(1))[0].item())
                keep_mask = torch.zeros_like(layout_tokens, dtype=torch.bool)
                keep_mask[:, :ctx_n + 1] = True  # keep context prefix + BOS
                # PAD positions must also keep ground truth (preserve padding)
                keep_mask |= (layout_tokens == LAYOUT_PAD)
                replace_mask = (~keep_mask) & (torch.rand_like(layout_tokens.float()) < p_sched)
                # Replace: use prediction from previous position
                shifted_pred = torch.cat([layout_tokens[:, :1], pred_toks[:, :-1]], dim=1)
                mixed_tokens = torch.where(replace_mask, shifted_pred, layout_tokens)
            layout_tokens = mixed_tokens

        logits = self.model(
            layout_tokens=layout_tokens,
            token_slot   =batch["token_slot"],
            token_hand   =batch["token_hand"],
            token_role   =batch["token_role"],
            phrase_mert  =batch["phrase_mert"],
            phrase_mask  =batch["phrase_mask"],
            difficulty   =batch["difficulty"],
            genre        =batch["genre"],
            song_fps     =batch.get("song_fps"),
            song_fp_mask =batch.get("song_fp_mask"),
            phrase_contour=batch.get("phrase_contour"),
        )   # [B, S, vocab]
        target  = batch["target"]                              # [B, S]
        per_tok = self.loss_fn(
            logits.reshape(-1, logits.size(-1)), target.reshape(-1),
        ).view_as(target)                                       # [B, S]

        # Weight the X-role positions higher. role_out aligns to target the
        # same way the validation metric does: target[t] = input[t+1], so the
        # role of target[t] is token_role[t+1].
        role_in  = batch["token_role"]
        role_out = torch.cat(
            [role_in[:, 1:], torch.full_like(role_in[:, :1], -1)], dim=1,
        )                                                       # [B, S]
        weights  = torch.where(
            role_out == ROLE_X,
            torch.full_like(per_tok, self.x_role_weight),
            torch.ones_like(per_tok),
        )
        valid   = target != IGNORE_INDEX                         # [B, S]
        loss    = (per_tok * weights * valid).sum() / weights[valid].sum().clamp_min(1.0)
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
