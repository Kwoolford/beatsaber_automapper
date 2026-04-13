"""Lightning module for Stage 2 (v2): Structured note prediction training.

Wraps AudioEncoder + NotePredictor for multi-head classification training.
Each onset's token sequence is converted to structured targets at load time,
then the model predicts all note attributes simultaneously.

Loss components:
  - n_notes CE: how many notes at this onset
  - Per-slot CE: color, column, row, direction, angle, event_type
  - Parity penalty: differentiable penalty for same-parity consecutive directions
  - Ergonomic penalty: penalizes red in right cols, blue in left cols
  - Collision penalty: penalizes multiple slots predicting same (col, row)
"""

from __future__ import annotations

import logging
import math

import lightning
import torch
import torch.nn as nn

from beatsaber_automapper.data.tokenizer import tokens_to_structured
from beatsaber_automapper.models.audio_encoder import AudioEncoder
from beatsaber_automapper.models.note_predictor import (
    ANGLE_CLASSES,
    COL_CLASSES,
    COLOR_CLASSES,
    DIR_CLASSES,
    EVENT_TYPE_CLASSES,
    MAX_SLOTS,
    ROW_CLASSES,
)

logger = logging.getLogger(__name__)

# Parity groups for flow loss
_FOREHAND_DIRS = frozenset({1, 6, 7})  # down, down-left, down-right
_BACKHAND_DIRS = frozenset({0, 4, 5})  # up, up-left, up-right


class NotePredictionLitModule(lightning.LightningModule):
    """Lightning training module for structured note prediction.

    Args:
        n_mels: Number of mel bands for audio encoder.
        encoder_d_model: Audio encoder model dimension.
        encoder_nhead: Audio encoder attention heads.
        encoder_num_layers: Audio encoder transformer layers.
        encoder_dim_feedforward: Audio encoder FFN dimension.
        encoder_dropout: Audio encoder dropout.
        pred_nhead: NotePredictor attention heads.
        pred_num_pool_layers: NotePredictor cross-attention pooling layers.
        pred_dim_feedforward: NotePredictor FFN dimension.
        pred_num_difficulties: Number of difficulty levels.
        pred_num_genres: Number of genre classes.
        pred_dropout: NotePredictor dropout.
        conditioning_dropout: Dropout for difficulty/genre embeddings (CFG).
        prev_context_k: Number of previous onset token sequences for context.
        vocab_size: Token vocabulary size.
        label_smoothing: Label smoothing for CE losses.
        learning_rate: Peak learning rate.
        weight_decay: AdamW weight decay.
        warmup_steps: Linear warmup steps.
        lr_min_ratio: Cosine decay floor ratio.
        freeze_encoder: Whether to freeze audio encoder weights.
        w_n_notes: Weight for n_notes loss.
        w_color: Weight for color loss.
        w_pos: Weight for position (col + row) loss.
        w_dir: Weight for direction loss.
        w_angle: Weight for angle loss.
        w_type: Weight for event type loss.
        lambda_parity: Weight for parity penalty.
        lambda_ergo: Weight for ergonomic penalty.
        lambda_collision: Weight for collision penalty.
    """

    def __init__(
        self,
        # Audio encoder
        n_mels: int = 80,
        encoder_d_model: int = 512,
        encoder_nhead: int = 8,
        encoder_num_layers: int = 6,
        encoder_dim_feedforward: int = 2048,
        encoder_dropout: float = 0.1,
        # NotePredictor
        pred_nhead: int = 8,
        pred_num_pool_layers: int = 2,
        pred_dim_feedforward: int = 2048,
        pred_num_difficulties: int = 5,
        pred_num_genres: int = 1,
        pred_dropout: float = 0.1,
        conditioning_dropout: float = 0.0,
        prev_context_k: int = 0,
        vocab_size: int = 183,
        # Training
        label_smoothing: float = 0.1,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        warmup_steps: int = 2000,
        lr_min_ratio: float = 0.01,
        freeze_encoder: bool = False,
        # Loss weights
        w_n_notes: float = 2.0,
        w_color: float = 1.0,
        w_pos: float = 1.0,
        w_dir: float = 2.0,
        w_angle: float = 0.5,
        w_type: float = 1.0,
        lambda_parity: float = 0.1,
        lambda_ergo: float = 0.1,
        lambda_collision: float = 0.2,
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

        from beatsaber_automapper.models.note_predictor import NotePredictor

        self.note_predictor = NotePredictor(
            d_model=encoder_d_model,
            nhead=pred_nhead,
            num_pool_layers=pred_num_pool_layers,
            dim_feedforward=pred_dim_feedforward,
            num_difficulties=pred_num_difficulties,
            num_genres=pred_num_genres,
            dropout=pred_dropout,
            conditioning_dropout=conditioning_dropout,
            prev_context_k=prev_context_k,
            vocab_size=vocab_size,
        )

        if freeze_encoder:
            for param in self.audio_encoder.parameters():
                param.requires_grad = False

        # CE losses (ignore_index=-1 for inactive slots)
        self.n_notes_ce = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.color_ce = nn.CrossEntropyLoss(
            ignore_index=-1, label_smoothing=label_smoothing
        )
        self.col_ce = nn.CrossEntropyLoss(
            ignore_index=-1, label_smoothing=label_smoothing
        )
        self.row_ce = nn.CrossEntropyLoss(
            ignore_index=-1, label_smoothing=label_smoothing
        )
        self.dir_ce = nn.CrossEntropyLoss(
            ignore_index=-1, label_smoothing=label_smoothing
        )
        self.angle_ce = nn.CrossEntropyLoss(
            ignore_index=-1, label_smoothing=label_smoothing
        )
        self.event_type_ce = nn.CrossEntropyLoss(
            ignore_index=-1, label_smoothing=label_smoothing
        )

    def _tokens_to_targets(
        self, tokens: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Convert token sequences to structured targets for the batch.

        Moves tokens to CPU once to avoid per-sample CUDA sync overhead,
        then builds all target tensors and moves them to GPU in one batch.

        Args:
            tokens: Token sequences [B, S].

        Returns:
            Dict of target tensors on the same device as input.
        """
        b = tokens.shape[0]
        device = tokens.device

        # Single GPU→CPU transfer for the whole batch
        tokens_cpu = tokens.cpu().tolist()

        n_notes_list = []
        color_list = []
        col_list = []
        row_list = []
        dir_list = []
        angle_list = []
        etype_list = []

        for i in range(b):
            structured = tokens_to_structured(tokens_cpu[i])
            n_notes_list.append(structured["n_notes"])

            for slot in structured["slots"]:
                is_active = slot["color"] != 2
                color_list.append(slot["color"] if is_active else -1)
                col_list.append(slot["col"] if is_active else -1)
                row_list.append(slot["row"] if is_active else -1)
                dir_list.append(slot["direction"] if is_active else -1)
                angle_list.append(slot["angle"] if is_active else -1)
                etype_list.append(slot["event_type"] if is_active else -1)

        # Single CPU→GPU transfer for all targets
        return {
            "n_notes": torch.tensor(n_notes_list, dtype=torch.long, device=device),
            "color": torch.tensor(
                color_list, dtype=torch.long, device=device
            ).reshape(b, MAX_SLOTS),
            "col": torch.tensor(
                col_list, dtype=torch.long, device=device
            ).reshape(b, MAX_SLOTS),
            "row": torch.tensor(
                row_list, dtype=torch.long, device=device
            ).reshape(b, MAX_SLOTS),
            "direction": torch.tensor(
                dir_list, dtype=torch.long, device=device
            ).reshape(b, MAX_SLOTS),
            "angle": torch.tensor(
                angle_list, dtype=torch.long, device=device
            ).reshape(b, MAX_SLOTS),
            "event_type": torch.tensor(
                etype_list, dtype=torch.long, device=device
            ).reshape(b, MAX_SLOTS),
        }

    def _compute_losses(
        self,
        preds: dict[str, torch.Tensor],
        targets: dict[str, torch.Tensor],
        prev_tokens: torch.Tensor | None = None,
        time_gap: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Compute all loss components.

        Returns dict with individual losses and total.
        """
        hp = self.hparams

        # n_notes CE
        n_notes_loss = self.n_notes_ce(preds["n_notes"], targets["n_notes"])

        # Per-slot attribute CE (flatten slots into batch dim)
        color_loss = self.color_ce(
            preds["color"].reshape(-1, COLOR_CLASSES), targets["color"].reshape(-1)
        )
        col_loss = self.col_ce(
            preds["col"].reshape(-1, COL_CLASSES), targets["col"].reshape(-1)
        )
        row_loss = self.row_ce(
            preds["row"].reshape(-1, ROW_CLASSES), targets["row"].reshape(-1)
        )
        dir_loss = self.dir_ce(
            preds["direction"].reshape(-1, DIR_CLASSES),
            targets["direction"].reshape(-1),
        )
        angle_loss = self.angle_ce(
            preds["angle"].reshape(-1, ANGLE_CLASSES), targets["angle"].reshape(-1)
        )
        etype_loss = self.event_type_ce(
            preds["event_type"].reshape(-1, EVENT_TYPE_CLASSES),
            targets["event_type"].reshape(-1),
        )

        # Main loss
        total = (
            hp.w_n_notes * n_notes_loss
            + hp.w_color * color_loss
            + hp.w_pos * (col_loss + row_loss)
            + hp.w_dir * dir_loss
            + hp.w_angle * angle_loss
            + hp.w_type * etype_loss
        )

        losses = {
            "n_notes_loss": n_notes_loss,
            "color_loss": color_loss,
            "col_loss": col_loss,
            "row_loss": row_loss,
            "dir_loss": dir_loss,
            "angle_loss": angle_loss,
            "etype_loss": etype_loss,
        }

        # Parity penalty (differentiable)
        if hp.lambda_parity > 0 and prev_tokens is not None:
            parity_loss = self._compute_parity_penalty(
                preds["direction"], prev_tokens, time_gap
            )
            total = total + hp.lambda_parity * parity_loss
            losses["parity_loss"] = parity_loss

        # Ergonomic penalty
        if hp.lambda_ergo > 0:
            ergo_loss = self._compute_ergo_penalty(
                preds["col"], preds["color"]
            )
            total = total + hp.lambda_ergo * ergo_loss
            losses["ergo_loss"] = ergo_loss

        # Collision penalty
        if hp.lambda_collision > 0:
            collision_loss = self._compute_collision_penalty(
                preds["col"], preds["row"]
            )
            total = total + hp.lambda_collision * collision_loss
            losses["collision_loss"] = collision_loss

        losses["total"] = total
        return losses

    def _compute_parity_penalty(
        self,
        dir_logits: torch.Tensor,
        prev_tokens: torch.Tensor,
        time_gap: torch.Tensor | None,
    ) -> torch.Tensor:
        """Differentiable parity penalty using soft direction probabilities.

        Pre-computes all CPU-side logic (prev direction extraction) in bulk,
        then uses a single vectorized softmax + gather on GPU.

        Args:
            dir_logits: Direction logits [B, S, 9].
            prev_tokens: Previous onset tokens [B, K, SeqLen].
            time_gap: Time gap from previous onset [B] or None.

        Returns:
            Scalar penalty (mean parity violation probability).
        """
        from beatsaber_automapper.data.tokenizer import DIR_OFFSET, NOTE

        b = dir_logits.shape[0]
        device = dir_logits.device

        forehand_ids = [1, 6, 7]
        backhand_ids = [0, 4, 5]

        # Move to CPU once for bulk processing
        prev_cpu = prev_tokens[:, -1].cpu().tolist()  # [B, SeqLen]
        time_cpu = time_gap.cpu().tolist() if time_gap is not None else None

        # Build list of (sample_idx, same_parity_ids) for valid samples
        valid_indices = []
        same_parity_masks = []

        for i in range(b):
            if time_cpu is not None and time_cpu[i] > 3.0:
                continue

            last_prev = prev_cpu[i]
            prev_dir = None
            for j, t in enumerate(last_prev):
                if t == NOTE and j + 4 < len(last_prev):
                    d = last_prev[j + 4] - DIR_OFFSET
                    if 0 <= d <= 8:
                        prev_dir = d
            if prev_dir is None or prev_dir in (2, 3, 8):
                continue

            prev_is_forehand = prev_dir in _FOREHAND_DIRS
            same_ids = forehand_ids if prev_is_forehand else backhand_ids

            valid_indices.append(i)
            # Create binary mask [9] where same-parity directions are 1
            mask = [0] * 9
            for d in same_ids:
                mask[d] = 1
            same_parity_masks.append(mask)

        if not valid_indices:
            return torch.tensor(0.0, device=device, requires_grad=False)

        # Vectorized GPU operations: single softmax + masked sum
        idx = torch.tensor(valid_indices, device=device)
        masks = torch.tensor(same_parity_masks, dtype=torch.float32, device=device)

        # Get direction probs for slot 0 of valid samples
        dir_probs = torch.softmax(dir_logits[idx, 0, :], dim=-1)  # [N, 9]
        # Sum probability on same-parity directions
        penalty = (dir_probs * masks).sum(dim=-1).mean()

        return penalty

    def _compute_ergo_penalty(
        self,
        col_logits: torch.Tensor,
        color_logits: torch.Tensor,
    ) -> torch.Tensor:
        """Penalize red notes in right columns, blue in left columns.

        Uses soft probabilities for differentiability.

        Args:
            col_logits: Column logits [B, S, 4].
            color_logits: Color logits [B, S, 3].

        Returns:
            Scalar ergonomic penalty.
        """
        col_probs = torch.softmax(col_logits, dim=-1)    # [B, S, 4]
        color_probs = torch.softmax(color_logits, dim=-1)  # [B, S, 3]

        # P(red) * P(col 2 or 3)
        red_wrong = color_probs[:, :, 0] * (col_probs[:, :, 2] + col_probs[:, :, 3])
        # P(blue) * P(col 0 or 1)
        blue_wrong = color_probs[:, :, 1] * (col_probs[:, :, 0] + col_probs[:, :, 1])

        return (red_wrong.mean() + blue_wrong.mean())

    def _compute_collision_penalty(
        self,
        col_logits: torch.Tensor,
        row_logits: torch.Tensor,
    ) -> torch.Tensor:
        """Penalize multiple slots predicting the same grid position.

        Uses soft probabilities: computes probability of collision between
        each pair of slots.

        Args:
            col_logits: Column logits [B, S, 4].
            row_logits: Row logits [B, S, 3].

        Returns:
            Scalar collision penalty.
        """
        col_probs = torch.softmax(col_logits, dim=-1)  # [B, S, 4]
        row_probs = torch.softmax(row_logits, dim=-1)  # [B, S, 3]

        # For each grid cell, compute P(slot_i at cell) * P(slot_j at cell)
        # Position probability for each slot = outer product of col and row probs
        # pos_probs[b, s, c, r] = P(slot s at col c, row r)
        pos_probs = col_probs.unsqueeze(-1) * row_probs.unsqueeze(-2)  # [B, S, 4, 3]

        # Sum collision probability across all slot pairs
        s = col_logits.shape[1]
        collision = torch.tensor(0.0, device=col_logits.device)
        for i in range(s):
            for j in range(i + 1, s):
                # P(collision between slot i and j) = sum over cells of P(i at cell) * P(j at cell)
                collision = collision + (pos_probs[:, i] * pos_probs[:, j]).sum()

        return collision / col_logits.shape[0]

    def forward(
        self,
        mel: torch.Tensor,
        difficulty: torch.Tensor,
        genre: torch.Tensor,
        structure: torch.Tensor | None = None,
        prev_tokens: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Forward pass: mel -> audio features -> note predictions."""
        audio_features = self.audio_encoder(mel, structure_features=structure)
        return self.note_predictor(
            audio_features, difficulty, genre,
            structure=structure, prev_tokens=prev_tokens,
        )

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        targets = self._tokens_to_targets(batch["tokens"])
        structure = batch.get("structure", None)
        prev_tokens = batch.get("prev_tokens", None)
        time_gap = batch.get("time_gap", None)

        preds = self(
            batch["mel"], batch["difficulty"], batch["genre"],
            structure=structure, prev_tokens=prev_tokens,
        )

        losses = self._compute_losses(preds, targets, prev_tokens, time_gap)

        self.log("train_loss", losses["total"], prog_bar=True)
        for key, val in losses.items():
            if key != "total":
                self.log(f"train_{key}", val, prog_bar=False)

        return losses["total"]

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> None:
        targets = self._tokens_to_targets(batch["tokens"])
        structure = batch.get("structure", None)
        prev_tokens = batch.get("prev_tokens", None)
        time_gap = batch.get("time_gap", None)

        preds = self(
            batch["mel"], batch["difficulty"], batch["genre"],
            structure=structure, prev_tokens=prev_tokens,
        )

        losses = self._compute_losses(preds, targets, prev_tokens, time_gap)
        self.log("val_loss", losses["total"], prog_bar=True, sync_dist=True)

        # Per-attribute accuracy
        self._log_accuracies(preds, targets)

    def _log_accuracies(
        self, preds: dict[str, torch.Tensor], targets: dict[str, torch.Tensor],
    ) -> None:
        """Log per-attribute prediction accuracy."""
        # n_notes accuracy
        n_pred = preds["n_notes"].argmax(dim=-1)
        n_acc = (n_pred == targets["n_notes"]).float().mean()
        self.log("val_n_notes_acc", n_acc, prog_bar=True, sync_dist=True)

        # Per-slot accuracies (only on active slots)
        for attr in ("color", "col", "row", "direction", "angle", "event_type"):
            pred_cls = preds[attr].reshape(-1, preds[attr].shape[-1]).argmax(dim=-1)
            target_cls = targets[attr].reshape(-1)
            mask = target_cls != -1
            if mask.sum() > 0:
                acc = (pred_cls[mask] == target_cls[mask]).float().mean()
                self.log(f"val_{attr}_acc", acc, sync_dist=True)

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

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }
