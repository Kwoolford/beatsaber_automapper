"""Stage 1: Onset / beat prediction model.

Binary classification per audio frame — predicts whether a note should
appear at each time step. Uses a small 2-layer transformer on top of
audio encoder embeddings, conditioned on difficulty level.

Training: Binary cross-entropy with logits (sigmoid applied at inference).
Inference: Peak picking with configurable threshold.
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class OnsetModel(nn.Module):
    """Onset prediction head for Stage 1.

    Args:
        d_model: Input embedding dimension (from audio encoder).
        nhead: Number of attention heads.
        num_layers: Number of transformer encoder layers.
        num_difficulties: Number of difficulty levels (5: Easy-ExpertPlus).
        dropout: Dropout rate.
    """

    def __init__(
        self,
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 2,
        num_difficulties: int = 5,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.d_model = d_model

        # Difficulty conditioning: learned embedding added to every frame
        self.difficulty_emb = nn.Embedding(num_difficulties, d_model)

        # Small transformer for onset-specific processing
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output head: LayerNorm -> Linear -> squeeze to [B, T]
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 1),
        )

    def forward(self, audio_features: torch.Tensor, difficulty: torch.Tensor) -> torch.Tensor:
        """Predict onset logits per frame.

        Args:
            audio_features: Audio encoder output [B, T, d_model].
            difficulty: Difficulty index per sample [B].

        Returns:
            Raw logits [B, T] (apply sigmoid for probabilities).
        """
        # Add difficulty embedding to every frame
        diff_emb = self.difficulty_emb(difficulty)  # [B, d_model]
        x = audio_features + diff_emb.unsqueeze(1)  # [B, T, d_model]

        # Transformer
        x = self.transformer(x)

        # Output head
        logits = self.head(x).squeeze(-1)  # [B, T]
        return logits
