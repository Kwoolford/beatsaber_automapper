"""Onset Planner: bidirectional transformer for song-level note planning.

Runs ONCE per song over all onset audio embeddings, producing a plan vector
for each onset. This gives each onset awareness of global song structure:
density curves, chorus repetition, build-ups, and section transitions.

The plan vectors are concatenated to the SequenceModel's cross-attention
memory, providing each onset with rich song-level context beyond the
local K=8 previous onset mean-pooling.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from beatsaber_automapper.models.components import SinusoidalPositionalEncoding


class OnsetPlanner(nn.Module):
    """Bidirectional transformer that produces per-onset plan vectors.

    Args:
        d_model: Model dimension (should match audio encoder output).
        nhead: Number of attention heads.
        num_layers: Number of transformer encoder layers.
        dim_feedforward: Feed-forward dimension.
        dropout: Dropout rate.
        n_section_types: Number of section type categories (intro/verse/chorus/etc).
    """

    def __init__(
        self,
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        n_section_types: int = 6,
    ) -> None:
        super().__init__()
        self.d_model = d_model

        self.input_proj = nn.Linear(d_model, d_model)
        self.pos_enc = SinusoidalPositionalEncoding(d_model=d_model, dropout=dropout)

        # Section conditioning: learned embedding per section type + progress projection
        self.section_emb = nn.Embedding(n_section_types, d_model)
        self.progress_proj = nn.Linear(1, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_proj = nn.Linear(d_model, d_model)

    def forward(
        self,
        onset_embeddings: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
        section_ids: torch.Tensor | None = None,
        section_progress: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Produce plan vectors for all onsets in a song.

        Args:
            onset_embeddings: Audio embeddings at onset frames [B, N_onsets, d_model].
            padding_mask: Boolean mask [B, N_onsets] where True = padding position.
            section_ids: Section type indices per onset [B, N_onsets] (0-5).
            section_progress: Progress within section per onset [B, N_onsets] (0.0-1.0).

        Returns:
            Plan vectors [B, N_onsets, d_model].
        """
        x = self.input_proj(onset_embeddings)

        # Add section conditioning if provided
        if section_ids is not None:
            x = x + self.section_emb(section_ids)
        if section_progress is not None:
            x = x + self.progress_proj(section_progress.unsqueeze(-1))

        x = self.pos_enc(x)
        x = self.encoder(x, src_key_padding_mask=padding_mask)
        return self.output_proj(x)
