"""V7-3: Stage 1 BeatClassifier.

Small transformer that maps beat-grid MERT features (drum + mix stems) to
binary note-presence logits for left and right hands independently.

Architecture:
  Inputs:
    drum_features  [B, W, 768]  beat-aligned drum MERT
    mix_features   [B, W, 768]  beat-aligned mix (melody) MERT  (optional)
    struct_features [B, W, 8]   beat-aligned structure features  (optional)
                                rows: rms, onset_strength, bass, mid, high,
                                      spectral_centroid, section_id, section_progress
    difficulty     [B]          per-sample difficulty ID         (optional)
  Proj:
    Linear(drum, d_model) + Linear(mix, d_model) + Linear(8, d_model)  (sum-fused)
    + LayerNorm
    + position embedding (window-relative)
    + phase embedding (slot-within-bar, modulo 16 at subdiv=4)
    + difficulty embedding (broadcast across the window)
  Attn:    n_layers × full-window self-attention (32 beats)
  Head:    Linear(d_model, 2) → [left_logit, right_logit] per slot

The mix path lets the model learn which drum hits a human mapper "chooses" —
different genres/instruments yield different mapping styles. The phase
embedding gives the model an explicit downbeat signal. The difficulty
embedding lets the model distinguish Expert (~3 notes/bar) from ExpertPlus
(~6 notes/bar) when both are pooled in the training set — without it, the
same drum hit carries contradictory labels across difficulties. The struct
path directly encodes energy dynamics (RMS, onset strength) and section
position so the model can learn that loud/energetic sections warrant more
notes without inferring this purely from MERT activations.
"""

from __future__ import annotations

import torch
import torch.nn as nn


# Beat-grid phase: 16 slots per bar (4 beats × 4 subdiv). The phase embedding
# captures the cyclic 1-and-2-and-3-and-4-and structure that mappers respect.
PHASE_MOD = 16

# Difficulty embedding rows. Matches DIFFICULTY_MAP in data/dataset.py:
# Easy=0, Normal=1, Hard=2, Expert=3, ExpertPlus=4.
N_DIFFICULTIES = 5


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
        mert_dim:   int = 768,
        mix_dim:    int = 768,
        struct_dim: int = 8,
        instr_dim:  int = 0,
        d_model:    int = 256,
        n_heads:    int = 4,
        n_layers:   int = 2,
        max_len:    int = 512,
        dropout:    float = 0.1,
        n_difficulties: int = N_DIFFICULTIES,
    ) -> None:
        super().__init__()
        self.d_model    = d_model
        self.use_mix    = mix_dim > 0
        self.use_struct = struct_dim > 0
        self.use_instr  = instr_dim > 0
        self.use_diff   = n_difficulties > 0

        self.drum_proj   = nn.Linear(mert_dim, d_model)
        self.mix_proj    = nn.Linear(mix_dim,    d_model) if self.use_mix    else None
        # struct_proj uses a larger fan-in ratio so small 8-dim features get appropriate
        # gradient scale relative to the 768-dim MERT paths.
        self.struct_proj = nn.Linear(struct_dim, d_model) if self.use_struct else None
        # instr_proj: per-instrument layering features (drum/bass/synth/vocal density
        # + lead/bass pitch contour). The scoped-V8 density/structure signal that the
        # blurred mean-pooled MERT can't carry — gives Stage 1 explicit onset density
        # so it can learn where humans map notes without the hand-tuned section gate.
        self.instr_proj  = nn.Linear(instr_dim,  d_model) if self.use_instr  else None

        self.input_norm = nn.LayerNorm(d_model)

        self.pos_emb   = nn.Embedding(max_len, d_model)
        self.phase_emb = nn.Embedding(PHASE_MOD, d_model)
        self.diff_emb  = nn.Embedding(n_difficulties, d_model) if self.use_diff else None

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
        if self.struct_proj is not None:
            nn.init.xavier_uniform_(self.struct_proj.weight)
            nn.init.zeros_(self.struct_proj.bias)
        if self.instr_proj is not None:
            nn.init.xavier_uniform_(self.instr_proj.weight)
            nn.init.zeros_(self.instr_proj.bias)
        if self.diff_emb is not None:
            # Start at zero so the first epoch behaves identically to a no-diff baseline
            # and the embedding has to *earn* signal during training.
            nn.init.zeros_(self.diff_emb.weight)
        nn.init.xavier_uniform_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(
        self,
        drum_features:   torch.Tensor,
        mix_features:    torch.Tensor | None = None,
        difficulty:      torch.Tensor | int | None = None,
        slot_offset:     int = 0,
        struct_features: torch.Tensor | None = None,
        instr_features:  torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            drum_features:   [B, W, mert_dim] beat-aligned drum MERT.
            mix_features:    [B, W, mix_dim]  beat-aligned mix MERT (optional).
            difficulty:      [B] long tensor of difficulty IDs, or a scalar int
                             (broadcast to the whole batch). Per-sample so a
                             mixed-difficulty batch is handled correctly.
            slot_offset:     Absolute slot index of the first slot in the window.
                             Used to compute the within-bar phase embedding so
                             the phase signal is consistent across windows.
            struct_features: [B, W, struct_dim] beat-aligned structure features
                             (rms, onset_strength, bass, mid, high, centroid,
                              section_id, section_progress). Optional.
            instr_features:  [B, W, instr_dim] per-instrument layering features
                             (per-stem density + lead/bass pitch contour). Optional.

        Returns:
            logits: [B, W, 2] — [left_logit, right_logit] per beat slot.
        """
        B, W, _ = drum_features.shape
        device  = drum_features.device

        x = self.drum_proj(drum_features)
        if self.use_mix and mix_features is not None:
            x = x + self.mix_proj(mix_features)
        if self.use_struct and struct_features is not None:
            x = x + self.struct_proj(struct_features)
        if self.use_instr and instr_features is not None:
            x = x + self.instr_proj(instr_features)
        x = self.input_norm(x)

        positions = torch.arange(W, device=device)
        x = x + self.pos_emb(positions).unsqueeze(0)

        phase = (positions + slot_offset) % PHASE_MOD
        x = x + self.phase_emb(phase).unsqueeze(0)

        if self.use_diff and difficulty is not None:
            if isinstance(difficulty, int):
                difficulty = torch.full((B,), difficulty, dtype=torch.long, device=device)
            x = x + self.diff_emb(difficulty).unsqueeze(1)  # [B, 1, d_model] → broadcast over W

        x = self.transformer(x)
        x = self.norm(x)
        return self.head(x)

    def predict_probs(
        self,
        drum_features:   torch.Tensor,
        mix_features:    torch.Tensor | None = None,
        difficulty:      torch.Tensor | int | None = None,
        slot_offset:     int = 0,
        struct_features: torch.Tensor | None = None,
        instr_features:  torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Convenience wrapper: return sigmoid probabilities [B, W, 2]."""
        return torch.sigmoid(
            self(drum_features, mix_features, difficulty, slot_offset, struct_features, instr_features)
        )
