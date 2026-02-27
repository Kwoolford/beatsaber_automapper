"""Stage 3: Lighting event generation model.

Transformer decoder conditioned on audio features and note context to produce
synchronized lighting events (basicBeatmapEvents, colorBoostBeatmapEvents).

Architecture:
    Light token embedding (scaled by sqrt(d_model)) + SinusoidalPositionalEncoding
    + note context (mean-pooled note embeddings, additive per-position)
    -> nn.TransformerDecoder (causal self-attn + cross-attn to audio)
    -> LayerNorm -> Linear(d_model, light_vocab_size)
"""

from __future__ import annotations

import logging
import math

import torch
import torch.nn as nn

from beatsaber_automapper.data.tokenizer import LIGHT_PAD, LIGHT_VOCAB_SIZE, VOCAB_SIZE
from beatsaber_automapper.models.components import SinusoidalPositionalEncoding

logger = logging.getLogger(__name__)


class LightingModel(nn.Module):
    """Autoregressive lighting event generator for Stage 3.

    Conditions on:
    - Audio features via cross-attention (same as Stage 2)
    - Note token context via mean-pooled embedding added to each position
    - Genre via learned embedding added to each position

    Args:
        light_vocab_size: Size of the lighting token vocabulary.
        note_vocab_size: Size of the note token vocabulary (for note embedding).
        d_model: Model dimension.
        nhead: Number of attention heads.
        num_layers: Number of transformer decoder layers.
        dim_feedforward: Feed-forward network dimension.
        num_genres: Number of genre classes.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        light_vocab_size: int = LIGHT_VOCAB_SIZE,
        note_vocab_size: int = VOCAB_SIZE,
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 2048,
        num_genres: int = 11,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.light_vocab_size = light_vocab_size
        self.d_model = d_model

        # Lighting token embedding with LIGHT_PAD zeroed out
        self.light_emb = nn.Embedding(light_vocab_size, d_model, padding_idx=LIGHT_PAD)
        self.scale = math.sqrt(d_model)

        # Positional encoding for lighting token sequence
        self.pos_enc = SinusoidalPositionalEncoding(d_model=d_model, dropout=dropout)

        # Note context: embed note tokens -> mean pool -> project to d_model
        # Added additively to every position of the lighting token sequence
        self.note_emb = nn.Embedding(note_vocab_size, d_model, padding_idx=0)
        self.note_proj = nn.Linear(d_model, d_model)

        # Genre conditioning: learned embedding added to every position
        self.genre_emb = nn.Embedding(num_genres, d_model)

        # Structural slot embedding: tells the model what TYPE of token to produce
        # Slot 0: event type marker (LIGHT_BASIC/LIGHT_BOOST)
        # Slot 1: ET (event type 0-14)
        # Slot 2: VAL (value 0-7)
        # Slot 3: BRIGHT (brightness bin)
        # Cycles every 4 tokens within each event, with SEP/EOS resetting
        self.slot_emb = nn.Embedding(4, d_model)

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
        self.out_proj = nn.Linear(d_model, light_vocab_size)

    def _encode_note_context(self, note_tokens: torch.Tensor) -> torch.Tensor:
        """Compute note context vector by mean-pooling non-PAD note embeddings.

        Args:
            note_tokens: Note token indices [B, N] (may contain PAD=0).

        Returns:
            Note context [B, d_model].
        """
        # Embed and mask out PAD positions
        emb = self.note_emb(note_tokens)  # [B, N, d_model]
        mask = (note_tokens != 0).float().unsqueeze(-1)  # [B, N, 1]
        masked = emb * mask
        counts = mask.sum(dim=1).clamp(min=1.0)  # [B, 1]
        pooled = masked.sum(dim=1) / counts  # [B, d_model]
        return self.note_proj(pooled)  # [B, d_model]

    def forward(
        self,
        light_tokens: torch.Tensor,
        audio_features: torch.Tensor,
        note_tokens: torch.Tensor,
        genre: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass for teacher forcing.

        Args:
            light_tokens: Lighting token indices [B, S] (decoder input, BOS-prepended).
            audio_features: Audio encoder output [B, T, d_model].
            note_tokens: Note token context for this beat [B, N].
            genre: Genre index per sample [B].

        Returns:
            Logits over lighting vocabulary [B, S, light_vocab_size].
        """
        b, s = light_tokens.shape

        # Lighting token embedding scaled by sqrt(d_model) + positional encoding
        x = self.light_emb(light_tokens) * self.scale
        x = self.pos_enc(x)

        # Add structural slot embedding (cycles every 4 positions)
        slot_ids = torch.arange(s, device=light_tokens.device) % 4
        x = x + self.slot_emb(slot_ids).unsqueeze(0)  # broadcast over batch

        # Add note context and genre embedding (broadcast over sequence)
        note_ctx = self._encode_note_context(note_tokens)  # [B, d_model]
        genre_emb = self.genre_emb(genre)                  # [B, d_model]
        x = x + note_ctx.unsqueeze(1) + genre_emb.unsqueeze(1)

        # Causal mask
        causal_mask = nn.Transformer.generate_square_subsequent_mask(
            s, device=light_tokens.device, dtype=x.dtype
        )

        # Padding mask for lighting tokens
        tgt_key_padding_mask = light_tokens == LIGHT_PAD  # [B, S]

        # Decode with cross-attention to audio
        x = self.transformer_decoder(
            tgt=x,
            memory=audio_features,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
        )

        x = self.out_norm(x)
        return self.out_proj(x)  # [B, S, light_vocab_size]

    @torch.no_grad()
    def decode_step(
        self,
        light_tokens: torch.Tensor,
        audio_features: torch.Tensor,
        note_tokens: torch.Tensor,
        genre: torch.Tensor,
    ) -> torch.Tensor:
        """Single-step decode for autoregressive inference.

        Args:
            light_tokens: Tokens generated so far [B, S].
            audio_features: Audio encoder output [B, T, d_model].
            note_tokens: Note token context [B, N].
            genre: Genre index per sample [B].

        Returns:
            Logits at last position [B, light_vocab_size].
        """
        logits = self.forward(light_tokens, audio_features, note_tokens, genre)
        return logits[:, -1, :]  # [B, light_vocab_size]
