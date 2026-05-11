"""V6-5: Style discriminator for cohort-style auxiliary loss.

A small transformer classifier over (audio_emb, swing_tokens) → mapper_id.
Pretrained on multi-cohort data, then frozen and used to score generated
swing token streams during sequence training. The sequence model is then
pushed to produce streams that the discriminator classifies as the target
cohort's mapper — providing a learned, audio-aware "style-closeness"
gradient signal that token-CE alone cannot give.

Accepts either:
    - Integer tokens [B, S]        — for pretraining on real cohort data.
    - Soft probabilities [B, S, V] — for sequence training, where the seq
      model's softmax logits flow gradients through the discriminator.

Both paths share the same token embedding, computed as ``emb_table[tokens]``
for the integer case and ``probs @ emb_table`` for the soft case.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from beatsaber_automapper.data.swing_tokenizer import PAD, VOCAB_SIZE
from beatsaber_automapper.models.components import SinusoidalPositionalEncoding


class StyleDiscriminator(nn.Module):
    """Audio-conditioned mapper classifier over swing-event token windows.

    Args:
        vocab_size: Size of the swing-event vocabulary (118 in V6).
        num_mappers: Number of cohort mappers to classify.
        audio_d_model: Dimension of the pooled audio embedding fed in
            alongside the swing window (typically 512 — matches the
            sequence model's audio encoder output).
        d_model: Discriminator hidden dimension.
        nhead: Number of attention heads.
        num_layers: Number of transformer encoder layers.
        dim_feedforward: Encoder FFN dim.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        vocab_size: int = VOCAB_SIZE,
        num_mappers: int = 18,
        audio_d_model: int = 512,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.num_mappers = num_mappers
        self.d_model = d_model
        self.audio_d_model = audio_d_model

        self.token_emb = nn.Embedding(vocab_size, d_model, padding_idx=PAD)
        self.audio_proj = nn.Linear(audio_d_model, d_model)
        self.pos_enc = SinusoidalPositionalEncoding(d_model=d_model, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Combine pooled swing emb with audio emb → mapper logits
        self.classifier = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Linear(d_model, num_mappers),
        )

    def forward(
        self,
        audio_emb: torch.Tensor,
        swing_input: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Classify the swing window's mapper.

        Args:
            audio_emb: Pooled audio context [B, audio_d_model].
            swing_input: Either integer tokens [B, S] (dtype long) or
                soft probabilities [B, S, V] (dtype float). The soft form
                supports gradient flow back to a generating model.
            padding_mask: Optional [B, S] boolean mask where True = ignore.
                For integer tokens auto-derived as (tokens == PAD) when None.

        Returns:
            Mapper logits [B, num_mappers].
        """
        if swing_input.dim() == 2:
            # Integer token path
            x = self.token_emb(swing_input)  # [B, S, d_model]
            if padding_mask is None:
                padding_mask = swing_input == PAD
        elif swing_input.dim() == 3:
            # Soft-probability path — multiply by embedding table for differentiable embed
            if swing_input.shape[-1] != self.vocab_size:
                raise ValueError(
                    f"swing_input vocab dim {swing_input.shape[-1]} != "
                    f"discriminator vocab_size {self.vocab_size}",
                )
            x = swing_input @ self.token_emb.weight  # [B, S, d_model]
            # No automatic padding inference for soft probs; caller must supply.
        else:
            raise ValueError(
                f"swing_input must be [B, S] or [B, S, V], got shape {tuple(swing_input.shape)}",
            )

        x = self.pos_enc(x)
        x = self.encoder(x, src_key_padding_mask=padding_mask)

        # Mean-pool over non-pad positions
        if padding_mask is not None:
            keep = (~padding_mask).float().unsqueeze(-1)  # [B, S, 1]
            denom = keep.sum(dim=1).clamp(min=1e-6)
            swing_pooled = (x * keep).sum(dim=1) / denom
        else:
            swing_pooled = x.mean(dim=1)

        audio_pooled = self.audio_proj(audio_emb)  # [B, d_model]
        combined = torch.cat([audio_pooled, swing_pooled], dim=-1)
        return self.classifier(combined)

    @torch.no_grad()
    def predict_mapper(
        self,
        audio_emb: torch.Tensor,
        swing_tokens: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Convenience inference: return argmax mapper [B]."""
        logits = self.forward(audio_emb, swing_tokens, padding_mask)
        return logits.argmax(dim=-1)

    def match_log_prob(
        self,
        audio_emb: torch.Tensor,
        swing_input: torch.Tensor,
        target_mapper: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Log-probability that the discriminator assigns to target_mapper.

        Used in sequence training: pass the generating model's soft
        probabilities as swing_input; the returned scalar (mean over batch)
        of log p_D(target_mapper) is gradient-friendly with respect to
        the soft input. Maximising this is the V6-5 style-closeness aux loss.

        Args:
            audio_emb: [B, audio_d_model]
            swing_input: [B, S] tokens or [B, S, V] soft probs.
            target_mapper: [B] mapper ids (long).
            padding_mask: optional [B, S] bool.

        Returns:
            Scalar = mean log p_D(target | audio, swings) over batch.
        """
        logits = self.forward(audio_emb, swing_input, padding_mask)
        log_probs = torch.log_softmax(logits, dim=-1)
        # Gather target log-probs per sample
        idx = target_mapper.unsqueeze(-1)  # [B, 1]
        target_lp = log_probs.gather(1, idx).squeeze(-1)  # [B]
        return target_lp.mean()
