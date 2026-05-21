"""V7-3: Stage 1 BeatClassifier.

Small transformer that maps beat-grid MERT features (drum + mix stems) to
binary note-presence logits for left and right hands independently.

Architecture:
  Inputs:
    drum_features [B, W, 768]  beat-aligned drum MERT
    mix_features  [B, W, 768]  beat-aligned mix (melody) MERT  (optional)
  Proj:
    Linear(drum, d_model) + Linear(mix, d_model)               (sum-fused)
    + LayerNorm
    + position embedding (window-relative)
    + phase embedding (slot-within-bar, modulo 16 at subdiv=4)
  Attn:    n_layers × full-window self-attention (32 beats)
  Head:    Linear(d_model, 2) → [left_logit, right_logit] per slot

The mix path lets the model learn which drum hits a human mapper "chooses" —
different genres/instruments yield different mapping styles. The phase
embedding gives the model an explicit downbeat signal.
"""

from __future__ import annotations

import torch
import torch.nn as nn


# Beat-grid phase: 16 slots per bar (4 beats × 4 subdiv). The phase embedding
# captures the cyclic 1-and-2-and-3-and-4-and structure that mappers respect.
PHASE_MOD = 16


class BeatClassifier(nn.Module):
    """Drum+mix MERT → beat-slot note presence classifier.

    Args:
        mert_dim:     Input MERT feature dimension (768 for MERT-v1-95M).
        mix_dim:      Mix-stem feature dimension (768). Set to 0 to disable.
        d_model:      Internal transformer dimension.
        n_heads:      Attention heads.
        n_layers:     Transformer encoder layers.
        max_len:      Maximum sequence length (window_size for positional emb).
        dropout:      Dropout rate.
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
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.use_mix = mix_dim > 0

        self.drum_proj = nn.Linear(mert_dim, d_model)
        self.mix_proj  = nn.Linear(mix_dim, d_model) if self.use_mix else None

        self.input_norm = nn.LayerNorm(d_model)

        self.pos_emb   = nn.Embedding(max_len, d_model)
        self.phase_emb = nn.Embedding(PHASE_MOD, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 2)  # [left_logit, right_logit]

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.xavier_uniform_(self.drum_proj.weight)
        nn.init.zeros_(self.drum_proj.bias)
        if self.mix_proj is not None:
            nn.init.xavier_uniform_(self.mix_proj.weight)
            nn.init.zeros_(self.mix_proj.bias)
        nn.init.xavier_uniform_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(
        self,
        drum_features: torch.Tensor,
        mix_features:  torch.Tensor | None = None,
        slot_offset:   int = 0,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            drum_features: [B, W, mert_dim] beat-aligned drum MERT.
            mix_features:  [B, W, mix_dim]  beat-aligned mix MERT (optional).
            slot_offset:   Absolute slot index of the first slot in the window.
                           Used to compute the within-bar phase embedding so
                           the phase signal is consistent across windows.

        Returns:
            logits: [B, W, 2] — [left_logit, right_logit] per beat slot.
        """
        B, W, _ = drum_features.shape
        device  = drum_features.device

        x = self.drum_proj(drum_features)
        if self.use_mix and mix_features is not None:
            x = x + self.mix_proj(mix_features)
        x = self.input_norm(x)

        positions = torch.arange(W, device=device)
        x = x + self.pos_emb(positions).unsqueeze(0)

        phase = (positions + slot_offset) % PHASE_MOD
        x = x + self.phase_emb(phase).unsqueeze(0)

        x = self.transformer(x)
        x = self.norm(x)
        return self.head(x)

    def predict_probs(
        self,
        drum_features: torch.Tensor,
        mix_features:  torch.Tensor | None = None,
        slot_offset:   int = 0,
    ) -> torch.Tensor:
        """Convenience wrapper: return sigmoid probabilities [B, W, 2]."""
        return torch.sigmoid(self(drum_features, mix_features, slot_offset))
