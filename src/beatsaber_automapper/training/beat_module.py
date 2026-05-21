"""V7-3: Lightning module for Stage 1 BeatClassifier training.

Uses weighted binary cross-entropy to handle the strong class imbalance
(notes occupy ~15% of beat slots on Expert maps; 85% of slots are empty).

Logs per-hand F1, precision, and recall to TensorBoard at each validation step.
"""

from __future__ import annotations

import logging

import lightning
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.classification import BinaryF1Score, BinaryPrecision, BinaryRecall

from beatsaber_automapper.evaluation.onset_metrics import OnsetToleranceF1
from beatsaber_automapper.models.beat_classifier import BeatClassifier

logger = logging.getLogger(__name__)


class BeatLitModule(lightning.LightningModule):
    """Lightning training module for Stage 1 beat classifier.

    Args:
        mert_dim:        MERT feature dimension (768).
        d_model:         Internal model dimension.
        n_heads:         Attention heads.
        n_layers:        Transformer encoder layers.
        max_len:         Max window length for positional embeddings.
        dropout:         Dropout rate.
        learning_rate:   Peak learning rate.
        weight_decay:    AdamW weight decay.
        warmup_steps:    LR warmup steps.
        pos_weight:      Positive class weight for BCE. None = auto from data.
                         Set manually if auto-estimate is unavailable.
        threshold:       Decision threshold for F1/P/R metrics (default 0.5).
        tolerance_slots: ±slot match window for the tolerance F1 metric
                         (val_f1_avg_tol). 0 = exact-slot; 1 ≈ ±125 ms at
                         BPM=120, subdiv=4.
    """

    def __init__(
        self,
        mert_dim: int = 768,
        mix_dim:  int = 768,
        d_model: int = 256,
        n_heads: int = 4,
        n_layers: int = 2,
        max_len: int = 512,
        dropout: float = 0.1,
        learning_rate: float = 3e-4,
        weight_decay: float = 0.01,
        warmup_steps: int = 500,
        # Run 1 used pos_weight=6.0 for an assumed 15% positive rate.
        # Measured positive rate on Expert+ data is 21.8% → 78.2/21.8 ≈ 3.6.
        pos_weight: float = 3.6,
        threshold: float = 0.5,
        tolerance_slots: int = 1,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.model = BeatClassifier(
            mert_dim=mert_dim,
            mix_dim=mix_dim,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            max_len=max_len,
            dropout=dropout,
        )

        # BCE with positive class weight (applied equally to both hands)
        pw = torch.tensor([pos_weight])
        self.register_buffer("pos_weight", pw)

        # Torchmetrics (reset each epoch)
        self._val_metrics = nn.ModuleDict({
            "f1_left":        BinaryF1Score(threshold=threshold),
            "f1_right":       BinaryF1Score(threshold=threshold),
            "prec_left":      BinaryPrecision(threshold=threshold),
            "prec_right":     BinaryPrecision(threshold=threshold),
            "recall_left":    BinaryRecall(threshold=threshold),
            "recall_right":   BinaryRecall(threshold=threshold),
            "f1_tol_left":    OnsetToleranceF1(threshold=threshold, tolerance=tolerance_slots),
            "f1_tol_right":   OnsetToleranceF1(threshold=threshold, tolerance=tolerance_slots),
        })

    def forward(
        self,
        drum_features: torch.Tensor,
        mix_features:  torch.Tensor | None = None,
        difficulty:    torch.Tensor | None = None,
        slot_offset:   int = 0,
    ) -> torch.Tensor:
        return self.model(drum_features, mix_features, difficulty, slot_offset)

    def _loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Weighted BCE summed across left + right hands."""
        # logits: [B, W, 2], labels: [B, W] each
        left_logits  = logits[..., 0]   # [B, W]
        right_logits = logits[..., 1]   # [B, W]
        left_labels  = labels[..., 0].float()
        right_labels = labels[..., 1].float()
        loss_l = F.binary_cross_entropy_with_logits(
            left_logits, left_labels, pos_weight=self.pos_weight,
        )
        loss_r = F.binary_cross_entropy_with_logits(
            right_logits, right_labels, pos_weight=self.pos_weight,
        )
        return (loss_l + loss_r) / 2.0

    @staticmethod
    def _slot_offset(batch: dict) -> int:
        """Use the first sample's slot offset for the whole batch.

        Phase is a per-slot integer modulo 16; using the first sample's offset
        keeps the embedding aligned for that sample. Mixed-offset batches
        will see slight phase noise for non-first samples, which acts as a
        light regulariser (the model still sees consistent within-window phase).
        """
        so = batch.get("slot_offset")
        if so is None:
            return 0
        if isinstance(so, torch.Tensor):
            return int(so[0].item()) if so.ndim > 0 else int(so.item())
        return int(so)

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        drum = batch["drum_features"]                        # [B, W, 768]
        mix  = batch.get("mix_features")                     # [B, W, 768] or None
        diff = batch.get("difficulty")                       # [B] long or None
        labels = torch.stack(
            [batch["left_labels"], batch["right_labels"]], dim=-1
        ).long()                                             # [B, W, 2]

        logits = self(drum, mix, diff, self._slot_offset(batch))
        loss   = self._loss(logits, labels)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch: dict, batch_idx: int) -> None:
        drum = batch["drum_features"]
        mix  = batch.get("mix_features")
        diff = batch.get("difficulty")
        labels = torch.stack(
            [batch["left_labels"], batch["right_labels"]], dim=-1
        ).long()

        logits = self(drum, mix, diff, self._slot_offset(batch))
        loss   = self._loss(logits, labels)
        self.log("val_loss", loss, prog_bar=True, sync_dist=True)

        probs = torch.sigmoid(logits)   # [B, W, 2]

        # 2D views per hand for the tolerance metric (which is per-sample).
        probs_l_2d  = probs[..., 0]
        probs_r_2d  = probs[..., 1]
        labels_l_2d = labels[..., 0]
        labels_r_2d = labels[..., 1]

        flat_pred_l  = probs_l_2d.reshape(-1)
        flat_pred_r  = probs_r_2d.reshape(-1)
        flat_label_l = labels_l_2d.reshape(-1)
        flat_label_r = labels_r_2d.reshape(-1)

        self._val_metrics["f1_left"](flat_pred_l,     flat_label_l)
        self._val_metrics["f1_right"](flat_pred_r,    flat_label_r)
        self._val_metrics["prec_left"](flat_pred_l,   flat_label_l)
        self._val_metrics["prec_right"](flat_pred_r,  flat_label_r)
        self._val_metrics["recall_left"](flat_pred_l, flat_label_l)
        self._val_metrics["recall_right"](flat_pred_r,flat_label_r)
        self._val_metrics["f1_tol_left"](probs_l_2d,   labels_l_2d)
        self._val_metrics["f1_tol_right"](probs_r_2d,  labels_r_2d)

    def on_validation_epoch_end(self) -> None:
        f1_l = self._val_metrics["f1_left"].compute()
        f1_r = self._val_metrics["f1_right"].compute()
        avg_f1 = (f1_l + f1_r) / 2.0
        self.log("val_f1_left",  f1_l,    prog_bar=True)
        self.log("val_f1_right", f1_r,    prog_bar=True)
        self.log("val_f1_avg",   avg_f1,  prog_bar=True)
        f1_tol_l = self._val_metrics["f1_tol_left"].compute()
        f1_tol_r = self._val_metrics["f1_tol_right"].compute()
        avg_f1_tol = (f1_tol_l + f1_tol_r) / 2.0
        self.log("val_f1_tol_left",  f1_tol_l,   prog_bar=True)
        self.log("val_f1_tol_right", f1_tol_r,   prog_bar=True)
        self.log("val_f1_avg_tol",   avg_f1_tol, prog_bar=True)
        self.log("val_precision_left",  self._val_metrics["prec_left"].compute())
        self.log("val_precision_right", self._val_metrics["prec_right"].compute())
        self.log("val_recall_left",     self._val_metrics["recall_left"].compute())
        self.log("val_recall_right",    self._val_metrics["recall_right"].compute())
        for m in self._val_metrics.values():
            m.reset()

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
            return max(0.01, 0.5 * (1.0 + torch.cos(torch.tensor(3.14159 * progress)).item()))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }
