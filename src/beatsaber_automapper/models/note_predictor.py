"""Stage 2 (v2): Structured note prediction model.

Replaces autoregressive token generation with multi-head direct prediction.
For each onset, predicts a fixed-size output in ONE forward pass:
  - How many notes (0-3)
  - Per slot: color, column, row, direction, angle offset, event type

Key advantages over autoregressive:
  - No cascading token errors (each attribute predicted independently)
  - Constraint masks can be applied before softmax (hard physical constraints)
  - Single forward pass per onset (much faster inference)
  - Natural multi-task loss with domain-specific penalties

Architecture:
    Audio features [B, T, d_model] (from shared AudioEncoder)
    + difficulty embedding + genre embedding
    + optional prev-onset context
    -> Cross-attention pooling (learnable query attends to audio)
    -> MLP heads per attribute
"""

from __future__ import annotations

import logging
import math

import torch
import torch.nn as nn

from beatsaber_automapper.data.tokenizer import PAD

logger = logging.getLogger(__name__)

# Output sizes per attribute
N_NOTES_CLASSES = 4    # 0, 1, 2, 3 notes
COLOR_CLASSES = 3      # red, blue, none (inactive slot)
COL_CLASSES = 4        # columns 0-3
ROW_CLASSES = 3        # rows 0-2
DIR_CLASSES = 9        # directions 0-8
ANGLE_CLASSES = 7      # angle offsets (-45 to +45 in 15° steps)
EVENT_TYPE_CLASSES = 5  # note, bomb, arc_start, arc_end, chain

MAX_SLOTS = 3  # Maximum notes per onset


class NotePredictor(nn.Module):
    """Structured note prediction for a single onset.

    Instead of generating tokens autoregressively, predicts all note
    attributes simultaneously via independent classification heads.

    Args:
        d_model: Model dimension (must match AudioEncoder output).
        nhead: Number of attention heads for cross-attention pooling.
        num_pool_layers: Number of cross-attention pooling layers.
        dim_feedforward: FFN dimension in pooling layers.
        num_difficulties: Number of difficulty levels.
        num_genres: Number of genre classes.
        dropout: Dropout rate.
        conditioning_dropout: Dropout for difficulty/genre during training (CFG).
        prev_context_k: Number of previous onset token sequences for context.
        vocab_size: Token vocab size (for encoding prev_tokens).
        max_slots: Maximum note slots per onset.
    """

    def __init__(
        self,
        d_model: int = 512,
        nhead: int = 8,
        num_pool_layers: int = 2,
        dim_feedforward: int = 2048,
        num_difficulties: int = 5,
        num_genres: int = 1,
        dropout: float = 0.1,
        conditioning_dropout: float = 0.0,
        prev_context_k: int = 0,
        vocab_size: int = 183,
        max_slots: int = MAX_SLOTS,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.max_slots = max_slots
        self.conditioning_dropout = conditioning_dropout
        self.prev_context_k = prev_context_k
        self.vocab_size = vocab_size

        # Difficulty and genre conditioning
        self.difficulty_emb = nn.Embedding(num_difficulties, d_model)
        self.genre_emb = nn.Embedding(num_genres, d_model)

        # Previous onset context encoding (reuses token embedding approach)
        if prev_context_k > 0:
            self.prev_token_emb = nn.Embedding(vocab_size, d_model, padding_idx=PAD)
            self.prev_context_proj = nn.Linear(d_model, d_model)

        # Learnable slot queries: each slot gets its own query vector
        # These attend to audio features via cross-attention
        self.slot_queries = nn.Parameter(torch.randn(max_slots, d_model) * 0.02)

        # Also a "count query" for predicting number of notes
        self.count_query = nn.Parameter(torch.randn(1, d_model) * 0.02)

        # Cross-attention pooling: slot queries attend to audio memory
        pool_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.cross_attn_pool = nn.TransformerDecoder(pool_layer, num_layers=num_pool_layers)

        # Output heads
        self.n_notes_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, N_NOTES_CLASSES),
        )

        # Per-slot heads (shared across slots — slot identity comes from query)
        self.color_head = self._make_head(d_model, COLOR_CLASSES, dropout)
        self.col_head = self._make_head(d_model, COL_CLASSES, dropout)
        self.row_head = self._make_head(d_model, ROW_CLASSES, dropout)
        self.dir_head = self._make_head(d_model, DIR_CLASSES, dropout)
        self.angle_head = self._make_head(d_model, ANGLE_CLASSES, dropout)
        self.event_type_head = self._make_head(d_model, EVENT_TYPE_CLASSES, dropout)

    @staticmethod
    def _make_head(d_model: int, n_classes: int, dropout: float) -> nn.Sequential:
        """Create a classification head for one attribute."""
        return nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, n_classes),
        )

    def _encode_prev_context(self, prev_tokens: torch.Tensor) -> torch.Tensor:
        """Encode previous onset tokens into context vectors.

        Args:
            prev_tokens: [B, K, S] previous onset token sequences.

        Returns:
            Context vectors [B, K, d_model].
        """
        b, k, s = prev_tokens.shape
        flat = prev_tokens.reshape(b * k, s)
        emb = self.prev_token_emb(flat) * math.sqrt(self.d_model)
        mask = (flat != PAD).unsqueeze(-1).float()
        pooled = (emb * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        pooled = pooled.reshape(b, k, self.d_model)
        return self.prev_context_proj(pooled)

    def forward(
        self,
        audio_features: torch.Tensor,
        difficulty: torch.Tensor,
        genre: torch.Tensor,
        structure: torch.Tensor | None = None,
        prev_tokens: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Forward pass: audio features -> structured note prediction.

        Args:
            audio_features: Audio encoder output [B, T, d_model].
            difficulty: Difficulty indices [B].
            genre: Genre indices [B].
            structure: Optional structure features (already folded into audio_features
                by AudioEncoder, so ignored here — kept for API compat).
            prev_tokens: Optional previous onset tokens [B, K, S].

        Returns:
            Dict of logits:
                n_notes: [B, 4] — number of notes (0-3)
                color: [B, max_slots, 3] — per-slot color
                col: [B, max_slots, 4] — per-slot column
                row: [B, max_slots, 3] — per-slot row
                direction: [B, max_slots, 9] — per-slot direction
                angle: [B, max_slots, 7] — per-slot angle offset
                event_type: [B, max_slots, 5] — per-slot event type
        """
        b = audio_features.shape[0]

        # Build memory: audio features + optional prev onset context
        memory = audio_features
        if prev_tokens is not None and self.prev_context_k > 0:
            prev_ctx = self._encode_prev_context(prev_tokens)
            memory = torch.cat([memory, prev_ctx], dim=1)

        # Add conditioning to memory
        diff_emb = self.difficulty_emb(difficulty)  # [B, d_model]
        genre_emb = self.genre_emb(genre)           # [B, d_model]

        if self.training and self.conditioning_dropout > 0:
            drop_mask = (
                torch.rand(b, 1, device=diff_emb.device) < self.conditioning_dropout
            ).float()
            diff_emb = diff_emb * (1 - drop_mask)
            genre_emb = genre_emb * (1 - drop_mask)

        # Add conditioning as bias to all memory positions
        memory = memory + diff_emb.unsqueeze(1) + genre_emb.unsqueeze(1)

        # Prepare queries: [count_query, slot_0, slot_1, slot_2]
        queries = torch.cat([self.count_query, self.slot_queries], dim=0)  # [1+S, d_model]
        queries = queries.unsqueeze(0).expand(b, -1, -1)  # [B, 1+S, d_model]

        # Cross-attention: queries attend to audio memory
        pooled = self.cross_attn_pool(queries, memory)  # [B, 1+S, d_model]

        count_repr = pooled[:, 0, :]              # [B, d_model]
        slot_reprs = pooled[:, 1:, :]             # [B, S, d_model]

        # Predict number of notes
        n_notes_logits = self.n_notes_head(count_repr)  # [B, 4]

        # Predict per-slot attributes
        color_logits = self.color_head(slot_reprs)       # [B, S, 3]
        col_logits = self.col_head(slot_reprs)           # [B, S, 4]
        row_logits = self.row_head(slot_reprs)           # [B, S, 3]
        dir_logits = self.dir_head(slot_reprs)           # [B, S, 9]
        angle_logits = self.angle_head(slot_reprs)       # [B, S, 7]
        event_type_logits = self.event_type_head(slot_reprs)  # [B, S, 5]

        return {
            "n_notes": n_notes_logits,
            "color": color_logits,
            "col": col_logits,
            "row": row_logits,
            "direction": dir_logits,
            "angle": angle_logits,
            "event_type": event_type_logits,
        }
