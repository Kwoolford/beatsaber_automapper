"""Stage 2: Note sequence generation model (V6 swing-event stream).

Autoregressive Transformer decoder that generates the per-hand swing-event
token stream for a full song. Conditioned on:

    - Audio encoder output (cross-attention, per-step)
    - Difficulty + genre embeddings (additive)
    - Saber state: 12-dim physical hand state (Linear → additive, per step)
    - Phrase embedding: 16-bar pooled audio context (Linear → additive, per step)
    - Mapper-id embedding (additive, cohort training only)

Architecture:
    Token embedding (scaled by sqrt(d_model)) + SinusoidalPositionalEncoding
    + difficulty/genre/mapper/saber_state/phrase conditioning (all additive)
    → CachedTransformerDecoder (causal self-attn + cross-attn to audio)
    → LayerNorm → Linear(d_model, vocab_size)

KV Caching:
    The decode_step_cached() method uses incremental KV caching for fast
    autoregressive inference. Only the new token is processed at each step.

V6 change log:
    - vocab_size: 183 → 118 (swing-event grammar)
    - Added saber_state_proj: Linear(12 → d_model), additive per step
    - Added phrase_proj: Linear(d_model → d_model), additive per step
    - Added num_mappers / mapper_id embedding
    - prev_context_k kept for backward compat but defaults to 0 (disabled)
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
    """Autoregressive swing-event sequence generator for Stage 2 (V6).

    Args:
        vocab_size: Token vocabulary size (118 for V6 swing grammar).
        d_model: Model dimension.
        nhead: Number of attention heads.
        num_layers: Number of transformer decoder layers.
        dim_feedforward: Feed-forward network dimension.
        num_difficulties: Number of difficulty levels (5).
        num_genres: Number of genre classes (11).
        num_mappers: Number of mapper/cohort IDs. 0 = no mapper embedding.
        saber_state_dim: Dimension of the saber-state input vector (12).
        dropout: Dropout rate.
        conditioning_dropout: Dropout probability for discrete conditioning
            embeddings (enables Classifier-Free Guidance at inference).
        prev_context_k: Legacy V5 inter-onset context (disabled by default).
    """

    def __init__(
        self,
        vocab_size: int = 118,
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 8,
        dim_feedforward: int = 2048,
        num_difficulties: int = 5,
        num_genres: int = 11,
        num_mappers: int = 0,
        saber_state_dim: int = 12,
        dropout: float = 0.1,
        conditioning_dropout: float = 0.0,
        prev_context_k: int = 0,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.conditioning_dropout = conditioning_dropout
        self.prev_context_k = prev_context_k
        self.num_mappers = num_mappers

        # Token embedding with PAD zeroed out
        self.token_emb = nn.Embedding(vocab_size, d_model, padding_idx=PAD)
        self.scale = math.sqrt(d_model)

        # Positional encoding for token sequence
        self.pos_enc = SinusoidalPositionalEncoding(d_model=d_model, dropout=dropout)

        # Discrete conditioning embeddings (all additive)
        self.difficulty_emb = nn.Embedding(num_difficulties, d_model)
        self.genre_emb = nn.Embedding(num_genres, d_model)
        if num_mappers > 0:
            self.mapper_emb = nn.Embedding(num_mappers, d_model)
        else:
            self.mapper_emb = None

        # V6 proprioception: saber physical state → d_model (additive per step)
        self.saber_state_proj = nn.Linear(saber_state_dim, d_model)

        # V6 phrase conditioning: pooled wide audio context → d_model (additive)
        self.phrase_proj = nn.Linear(d_model, d_model)

        # Legacy V5 inter-onset context (kept for backward compat; disabled by default)
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
        plan_vector: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Combine audio features with optional plan vector and previous onset context.

        Args:
            audio_features: Audio encoder output [B, T, d_model].
            prev_tokens: Optional previous onset tokens [B, K, S].
            plan_vector: Optional plan vector from OnsetPlanner [B, 1, d_model].

        Returns:
            Memory tensor [B, T(+1)(+K), d_model] for cross-attention.
        """
        parts = [audio_features]
        if plan_vector is not None:
            parts.append(plan_vector)
        if prev_tokens is not None and self.prev_context_k > 0:
            prev_context = self._encode_prev_context(prev_tokens)  # [B, K, d_model]
            parts.append(prev_context)
        if len(parts) == 1:
            return parts[0]
        return torch.cat(parts, dim=1)

    def forward(
        self,
        tokens: torch.Tensor,
        audio_features: torch.Tensor,
        difficulty: torch.Tensor,
        genre: torch.Tensor,
        saber_state: torch.Tensor | None = None,
        phrase_emb: torch.Tensor | None = None,
        mapper_id: torch.Tensor | None = None,
        prev_tokens: torch.Tensor | None = None,
        plan_vector: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass for teacher forcing.

        Args:
            tokens: Input token indices [B, S] (decoder input, BOS-prepended).
            audio_features: Audio encoder output [B, T, d_model].
            difficulty: Difficulty index per sample [B].
            genre: Genre index per sample [B].
            saber_state: Optional physical saber state per token step [B, S, 12].
                         Each row is the state BEFORE the token at that position.
            phrase_emb: Optional pooled phrase-level audio embedding [B, d_model].
                         Projected and added uniformly across all positions.
            mapper_id: Optional cohort mapper index per sample [B].
            prev_tokens: Legacy V5 inter-onset context [B, K, S] (unused by default).
            plan_vector: Legacy V5 planner vector [B, 1, d_model] (unused by default).

        Returns:
            Logits over vocabulary [B, S, vocab_size].
        """
        b, s = tokens.shape

        # Token embedding scaled by sqrt(d_model) + positional encoding
        x = self.token_emb(tokens) * self.scale
        x = self.pos_enc(x)

        # --- Discrete conditioning (all additive) ---
        diff_emb = self.difficulty_emb(difficulty)    # [B, d_model]
        genre_emb = self.genre_emb(genre)             # [B, d_model]

        # Conditioning dropout for CFG (drops all discrete signals together)
        if self.training and self.conditioning_dropout > 0:
            mask = torch.rand(b, 1, device=diff_emb.device)
            drop_mask = (mask < self.conditioning_dropout).float()
            diff_emb = diff_emb * (1 - drop_mask)
            genre_emb = genre_emb * (1 - drop_mask)

        x = x + diff_emb.unsqueeze(1) + genre_emb.unsqueeze(1)

        if mapper_id is not None and self.mapper_emb is not None:
            x = x + self.mapper_emb(mapper_id).unsqueeze(1)

        # --- V6 saber-state proprioception: per-position additive signal ---
        if saber_state is not None:
            # saber_state: [B, S, 12] → projected to [B, S, d_model]
            x = x + self.saber_state_proj(saber_state)

        # --- V6 phrase embedding: uniform additive signal ---
        if phrase_emb is not None:
            # phrase_emb: [B, d_model] → [B, 1, d_model] broadcast
            x = x + self.phrase_proj(phrase_emb).unsqueeze(1)

        # Build cross-attention memory
        memory = self._build_memory(audio_features, prev_tokens, plan_vector)

        # Causal mask: prevent attending to future tokens
        causal_mask = nn.Transformer.generate_square_subsequent_mask(
            s, device=tokens.device, dtype=x.dtype
        )

        # Padding mask: prevent attending to PAD tokens
        tgt_key_padding_mask = tokens == PAD  # [B, S], True = ignore

        x = self.transformer_decoder(
            tgt=x,
            memory=memory,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
        )

        x = self.out_norm(x)
        return self.out_proj(x)  # [B, S, vocab_size]

    @torch.no_grad()
    def decode_step(
        self,
        tokens: torch.Tensor,
        audio_features: torch.Tensor,
        difficulty: torch.Tensor,
        genre: torch.Tensor,
        saber_state: torch.Tensor | None = None,
        phrase_emb: torch.Tensor | None = None,
        mapper_id: torch.Tensor | None = None,
        prev_tokens: torch.Tensor | None = None,
        plan_vector: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Single-step decode for autoregressive inference (no cache).

        Returns logits only at the last token position for efficiency.

        Args:
            tokens: Token indices generated so far [B, S].
            audio_features: Audio encoder output [B, T, d_model].
            difficulty: Difficulty index per sample [B].
            genre: Genre index per sample [B].
            saber_state: Optional saber state for all positions [B, S, 12].
            phrase_emb: Optional phrase embedding [B, d_model].
            mapper_id: Optional mapper index [B].
            prev_tokens: Legacy V5 inter-onset context [B, K, S].
            plan_vector: Legacy V5 planner vector [B, 1, d_model].

        Returns:
            Logits at last position [B, vocab_size].
        """
        logits = self.forward(
            tokens, audio_features, difficulty, genre,
            saber_state=saber_state, phrase_emb=phrase_emb, mapper_id=mapper_id,
            prev_tokens=prev_tokens, plan_vector=plan_vector,
        )
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
        saber_state_step: torch.Tensor | None = None,
        phrase_emb: torch.Tensor | None = None,
        mapper_id: torch.Tensor | None = None,
        prev_tokens: torch.Tensor | None = None,
        plan_vector: torch.Tensor | None = None,
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
            saber_state_step: Saber state at this step [B, 1, 12].
            phrase_emb: Optional phrase embedding [B, d_model].
            mapper_id: Optional mapper index [B].
            prev_tokens: Legacy V5 inter-onset context [B, K, S].
            plan_vector: Legacy V5 planner vector [B, 1, d_model].

        Returns:
            Logits at the new position [B, vocab_size].
        """
        x = self.token_emb(token) * self.scale  # [B, 1, d_model]

        if step >= self.pos_enc.pe.size(1):
            self.pos_enc._extend_pe(step + 1)
        x = x + self.pos_enc.pe[:, step : step + 1, :]

        diff_emb = self.difficulty_emb(difficulty)
        genre_emb = self.genre_emb(genre)
        x = x + diff_emb.unsqueeze(1) + genre_emb.unsqueeze(1)

        if mapper_id is not None and self.mapper_emb is not None:
            x = x + self.mapper_emb(mapper_id).unsqueeze(1)

        if saber_state_step is not None:
            x = x + self.saber_state_proj(saber_state_step)  # [B, 1, d_model]

        if phrase_emb is not None:
            x = x + self.phrase_proj(phrase_emb).unsqueeze(1)

        memory = self._build_memory(audio_features, prev_tokens, plan_vector)

        x = self.transformer_decoder(
            tgt=x,
            memory=memory,
            layer_caches=layer_caches,
        )

        x = self.out_norm(x)
        return self.out_proj(x).squeeze(1)  # [B, vocab_size]
