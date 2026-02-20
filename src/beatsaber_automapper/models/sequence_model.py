"""Stage 2: Note sequence generation model.

Transformer decoder that generates note token sequences at each onset
timestamp, conditioned on audio features and difficulty level. Uses
causal self-attention (autoregressive) and cross-attention to audio.

Generates all v3 note types: colorNotes, bombNotes, obstacles,
sliders (arcs), burstSliders (chains).

Architecture:
    Token embedding (scaled by sqrt(d_model)) + SinusoidalPositionalEncoding
    + difficulty embedding (additive) -> nn.TransformerDecoder (causal self-attn
    + cross-attn to audio) -> LayerNorm -> Linear(d_model, vocab_size)
"""

from __future__ import annotations

import logging
import math

import torch
import torch.nn as nn

from beatsaber_automapper.data.tokenizer import PAD
from beatsaber_automapper.models.components import SinusoidalPositionalEncoding

logger = logging.getLogger(__name__)


class SequenceModel(nn.Module):
    """Autoregressive note sequence generator for Stage 2.

    Args:
        vocab_size: Size of the token vocabulary.
        d_model: Model dimension.
        nhead: Number of attention heads.
        num_layers: Number of transformer decoder layers.
        dim_feedforward: Feed-forward network dimension.
        num_difficulties: Number of difficulty levels.
        num_genres: Number of genre classes.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        vocab_size: int = 167,
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 8,
        dim_feedforward: int = 2048,
        num_difficulties: int = 5,
        num_genres: int = 11,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model

        # Token embedding with PAD zeroed out
        self.token_emb = nn.Embedding(vocab_size, d_model, padding_idx=PAD)
        self.scale = math.sqrt(d_model)

        # Positional encoding for token sequence
        self.pos_enc = SinusoidalPositionalEncoding(d_model=d_model, dropout=dropout)

        # Difficulty conditioning: learned embedding added to every position
        self.difficulty_emb = nn.Embedding(num_difficulties, d_model)

        # Genre conditioning: learned embedding added to every position
        self.genre_emb = nn.Embedding(num_genres, d_model)

        # Transformer decoder (causal self-attention + cross-attention to audio)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # Output projection
        self.out_norm = nn.LayerNorm(d_model)
        self.out_proj = nn.Linear(d_model, vocab_size)

    def forward(
        self,
        tokens: torch.Tensor,
        audio_features: torch.Tensor,
        difficulty: torch.Tensor,
        genre: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass for teacher forcing.

        Args:
            tokens: Input token indices [B, S] (decoder input, typically BOS-prepended).
            audio_features: Audio encoder output [B, T, d_model].
            difficulty: Difficulty index per sample [B].
            genre: Genre index per sample [B].

        Returns:
            Logits over vocabulary [B, S, vocab_size].
        """
        b, s = tokens.shape

        # Token embedding scaled by sqrt(d_model) + positional encoding
        x = self.token_emb(tokens) * self.scale
        x = self.pos_enc(x)

        # Add difficulty and genre embeddings
        diff_emb = self.difficulty_emb(difficulty)  # [B, d_model]
        genre_emb = self.genre_emb(genre)           # [B, d_model]
        x = x + diff_emb.unsqueeze(1) + genre_emb.unsqueeze(1)

        # Causal mask: prevent attending to future tokens
        causal_mask = nn.Transformer.generate_square_subsequent_mask(
            s, device=tokens.device, dtype=x.dtype
        )

        # Padding mask: prevent attending to PAD tokens
        tgt_key_padding_mask = tokens == PAD  # [B, S], True = ignore

        # Decode with cross-attention to audio
        x = self.transformer_decoder(
            tgt=x,
            memory=audio_features,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
        )

        # Project to vocabulary
        x = self.out_norm(x)
        logits = self.out_proj(x)  # [B, S, vocab_size]
        return logits

    @torch.no_grad()
    def decode_step(
        self,
        tokens: torch.Tensor,
        audio_features: torch.Tensor,
        difficulty: torch.Tensor,
        genre: torch.Tensor,
    ) -> torch.Tensor:
        """Single-step decode for autoregressive inference.

        Returns logits only at the last token position for efficiency.

        Args:
            tokens: Token indices generated so far [B, S].
            audio_features: Audio encoder output [B, T, d_model].
            difficulty: Difficulty index per sample [B].
            genre: Genre index per sample [B].

        Returns:
            Logits at last position [B, vocab_size].
        """
        logits = self.forward(tokens, audio_features, difficulty, genre)
        return logits[:, -1, :]  # [B, vocab_size]
