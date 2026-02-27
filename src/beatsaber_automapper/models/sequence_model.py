"""Stage 2: Note sequence generation model.

Transformer decoder that generates note token sequences at each onset
timestamp, conditioned on audio features and difficulty level. Uses
causal self-attention (autoregressive) and cross-attention to audio.

Generates all v3 note types: colorNotes, bombNotes, obstacles,
sliders (arcs), burstSliders (chains).

Architecture:
    Token embedding (scaled by sqrt(d_model)) + SinusoidalPositionalEncoding
    + difficulty embedding (additive) -> CachedTransformerDecoder (causal self-attn
    + cross-attn to audio, with KV cache support) -> LayerNorm -> Linear(d_model, vocab_size)

KV Caching:
    The decode_step_cached() method uses incremental KV caching for 10x faster
    autoregressive inference. Only the new token is processed at each step;
    self-attention K/V from previous positions are cached and reused.
"""

from __future__ import annotations

import logging
import math

import torch
import torch.nn as nn

from beatsaber_automapper.data.tokenizer import PAD
from beatsaber_automapper.models.components import (
    CachedTransformerDecoder,
    LayerCaches,
    SinusoidalPositionalEncoding,
)

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
        conditioning_dropout: Dropout probability for difficulty/genre embeddings
            during training (enables Classifier-Free Guidance at inference).
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
        conditioning_dropout: float = 0.0,
        prev_context_k: int = 0,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.conditioning_dropout = conditioning_dropout
        self.prev_context_k = prev_context_k

        # Token embedding with PAD zeroed out
        self.token_emb = nn.Embedding(vocab_size, d_model, padding_idx=PAD)
        self.scale = math.sqrt(d_model)

        # Positional encoding for token sequence
        self.pos_enc = SinusoidalPositionalEncoding(d_model=d_model, dropout=dropout)

        # Difficulty conditioning: learned embedding added to every position
        self.difficulty_emb = nn.Embedding(num_difficulties, d_model)

        # Genre conditioning: learned embedding added to every position
        self.genre_emb = nn.Embedding(num_genres, d_model)

        # Inter-onset context: encode previous K onset token sequences
        # and add them to the cross-attention memory alongside audio features
        if prev_context_k > 0:
            self.prev_context_proj = nn.Linear(d_model, d_model)

        # Cached Transformer decoder (causal self-attention + cross-attention to audio)
        self.transformer_decoder = CachedTransformerDecoder(
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )

        # Output projection
        self.out_norm = nn.LayerNorm(d_model)
        self.out_proj = nn.Linear(d_model, vocab_size)

    def _encode_prev_context(self, prev_tokens: torch.Tensor) -> torch.Tensor:
        """Encode previous onset token sequences into context vectors.

        Args:
            prev_tokens: Previous onset tokens [B, K, S] where K is the number
                of previous onsets and S is the max token length.

        Returns:
            Context vectors [B, K, d_model] — one per previous onset.
        """
        b, k, s = prev_tokens.shape
        # Flatten to [B*K, S] for embedding
        flat = prev_tokens.reshape(b * k, s)
        emb = self.token_emb(flat) * self.scale  # [B*K, S, d_model]
        # Mean-pool non-PAD tokens per onset
        mask = (flat != PAD).unsqueeze(-1).float()  # [B*K, S, 1]
        pooled = (emb * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)  # [B*K, d_model]
        pooled = pooled.reshape(b, k, self.d_model)  # [B, K, d_model]
        return self.prev_context_proj(pooled)  # [B, K, d_model]

    def _build_memory(
        self,
        audio_features: torch.Tensor,
        prev_tokens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Combine audio features with optional previous onset context.

        Args:
            audio_features: Audio encoder output [B, T, d_model].
            prev_tokens: Optional previous onset tokens [B, K, S].

        Returns:
            Memory tensor [B, T+K, d_model] for cross-attention.
        """
        if prev_tokens is not None and self.prev_context_k > 0:
            prev_context = self._encode_prev_context(prev_tokens)  # [B, K, d_model]
            return torch.cat([audio_features, prev_context], dim=1)
        return audio_features

    def forward(
        self,
        tokens: torch.Tensor,
        audio_features: torch.Tensor,
        difficulty: torch.Tensor,
        genre: torch.Tensor,
        prev_tokens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass for teacher forcing.

        Args:
            tokens: Input token indices [B, S] (decoder input, typically BOS-prepended).
            audio_features: Audio encoder output [B, T, d_model].
            difficulty: Difficulty index per sample [B].
            genre: Genre index per sample [B].
            prev_tokens: Optional previous onset tokens [B, K, S] for inter-onset context.

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

        # Conditioning dropout for CFG
        if self.training and self.conditioning_dropout > 0:
            mask = torch.rand(b, 1, device=diff_emb.device)
            drop_mask = (mask < self.conditioning_dropout).float()
            diff_emb = diff_emb * (1 - drop_mask)
            genre_emb = genre_emb * (1 - drop_mask)

        x = x + diff_emb.unsqueeze(1) + genre_emb.unsqueeze(1)

        # Build cross-attention memory: audio features + optional prev onset context
        memory = self._build_memory(audio_features, prev_tokens)

        # Causal mask: prevent attending to future tokens
        causal_mask = nn.Transformer.generate_square_subsequent_mask(
            s, device=tokens.device, dtype=x.dtype
        )

        # Padding mask: prevent attending to PAD tokens
        tgt_key_padding_mask = tokens == PAD  # [B, S], True = ignore

        # Decode with cross-attention to audio + prev context (no cache during training)
        x = self.transformer_decoder(
            tgt=x,
            memory=memory,
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
        prev_tokens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Single-step decode for autoregressive inference (no cache).

        Returns logits only at the last token position for efficiency.
        This is the original interface, kept for backward compatibility.

        Args:
            tokens: Token indices generated so far [B, S].
            audio_features: Audio encoder output [B, T, d_model].
            difficulty: Difficulty index per sample [B].
            genre: Genre index per sample [B].
            prev_tokens: Optional previous onset tokens [B, K, S].

        Returns:
            Logits at last position [B, vocab_size].
        """
        logits = self.forward(tokens, audio_features, difficulty, genre, prev_tokens=prev_tokens)
        return logits[:, -1, :]  # [B, vocab_size]

    def new_caches(self) -> list[LayerCaches]:
        """Create fresh empty KV caches for all decoder layers."""
        return self.transformer_decoder.new_caches()

    @torch.no_grad()
    def decode_step_cached(
        self,
        token: torch.Tensor,
        audio_features: torch.Tensor,
        difficulty: torch.Tensor,
        genre: torch.Tensor,
        layer_caches: list[LayerCaches],
        step: int,
        prev_tokens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Single-step decode with KV cache for fast inference.

        Only processes the new token, reusing cached K/V from previous steps.
        This is ~10x faster than decode_step for long sequences.

        Args:
            token: Single new token [B, 1].
            audio_features: Audio encoder output [B, T, d_model].
            difficulty: Difficulty index per sample [B].
            genre: Genre index per sample [B].
            layer_caches: Per-layer KV caches (modified in-place).
            step: Current step index (0-based) for positional encoding.
            prev_tokens: Optional previous onset tokens [B, K, S].

        Returns:
            Logits at the new position [B, vocab_size].
        """
        # Embed just the new token
        x = self.token_emb(token) * self.scale  # [B, 1, d_model]

        # Add positional encoding for this step only
        if step >= self.pos_enc.pe.size(1):
            self.pos_enc._extend_pe(step + 1)
        x = x + self.pos_enc.pe[:, step : step + 1, :]

        # Add difficulty and genre embeddings (no dropout at inference)
        diff_emb = self.difficulty_emb(difficulty)  # [B, d_model]
        genre_emb = self.genre_emb(genre)           # [B, d_model]
        x = x + diff_emb.unsqueeze(1) + genre_emb.unsqueeze(1)

        # Build cross-attention memory: audio + prev context
        memory = self._build_memory(audio_features, prev_tokens)

        # Run through decoder with cache (no causal mask needed — cache handles it)
        x = self.transformer_decoder(
            tgt=x,
            memory=memory,
            layer_caches=layer_caches,
        )

        # Project to vocabulary
        x = self.out_norm(x)
        logits = self.out_proj(x)  # [B, 1, vocab_size]
        return logits.squeeze(1)  # [B, vocab_size]
