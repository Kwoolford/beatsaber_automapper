"""V7-5: Stage 2 LayoutModel — spatial token generator.

Predicts the spatial token sequence [KIND, X, Y, DIR, FIELD_D] for a single note
given MERT conditioning from three levels (local beat, section, full song),
saber-state conditioning, and optional retrieval context.

No HAND tokens (given by Stage 1 beat schedule).
No Δt tokens (given by Stage 1 beat schedule).

The token vocabulary is the full swing-event vocab (118 tokens), but the grammar
only produces spatial tokens (KIND → X → Y → [DIR → FIELD_D | SQUISH | done]).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

# Re-use the swing vocab constants for the spatial subset
from beatsaber_automapper.data.swing_tokenizer import VOCAB_SIZE

LAYOUT_PAD = 0
LAYOUT_BOS = 1
LAYOUT_EOS = 2

MERT_DIM = 768    # MERT-v1-95M output dimension
SABER_DIM = 12    # saber state dimension


class LayoutModel(nn.Module):
    """Autoregressive spatial token generator conditioned on MERT features.

    Architecture:
      - Conditioning: concat([local_mert, song_emb, section_emb]) → Linear → d_model
                      + Linear(saber_state, d_model)
                      + Linear(phrase_feat, d_model)
                      → summed as a single conditioning vector
      - Embedding: token_emb(vocab_size, d_model)
      - Cross-attention to conditioning (single-vector memory)
      - Causal transformer decoder (n_layers layers)
      - Output: Linear(d_model, vocab_size)

    The conditioning vector is treated as a "memory" of length 1 for cross-attention.

    Args:
        vocab_size:    Full vocab size (118 — spatial subset is a grammar constraint).
        d_model:       Transformer hidden dimension.
        n_heads:       Attention heads (must divide d_model).
        n_layers:      Decoder layers.
        dim_feedforward: FFN inner dimension.
        max_len:       Max sequence length (for positional encoding).
        dropout:       Dropout rate.
    """

    def __init__(
        self,
        vocab_size: int = VOCAB_SIZE,
        d_model: int = 512,
        n_heads: int = 8,
        n_layers: int = 4,
        dim_feedforward: int = 2048,
        max_len: int = 64,
        dropout: float = 0.1,
        num_difficulties: int = 5,
        num_genres: int = 11,
    ) -> None:
        super().__init__()

        # MERT conditioning: 3 levels → d_model
        self.mert_proj = nn.Linear(MERT_DIM * 3, d_model)

        # Saber state → d_model
        self.saber_proj = nn.Linear(SABER_DIM, d_model)

        # Phrase fingerprint → d_model
        self.phrase_proj = nn.Linear(MERT_DIM, d_model)

        # Difficulty + genre embeddings
        self.diff_emb  = nn.Embedding(num_difficulties, d_model // 4)
        self.genre_emb = nn.Embedding(num_genres,       d_model // 4)
        self.cond_fuse = nn.Linear(d_model + d_model // 2, d_model)

        # Token embedding + positional
        self.token_emb = nn.Embedding(vocab_size, d_model, padding_idx=LAYOUT_PAD)
        self.pos_emb   = nn.Embedding(max_len, d_model)

        # Causal transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)

        self.norm   = nn.LayerNorm(d_model)
        self.output = nn.Linear(d_model, vocab_size)

        self._max_len = max_len
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _build_conditioning(
        self,
        local_mert:  torch.Tensor,   # [B, 768]
        song_emb:    torch.Tensor,   # [B, 768]
        section_emb: torch.Tensor,   # [B, 768]
        saber_state: torch.Tensor,   # [B, 12]
        phrase_feat: torch.Tensor,   # [B, 768]
        difficulty:  torch.Tensor,   # [B]
        genre:       torch.Tensor,   # [B]
    ) -> torch.Tensor:               # [B, 1, d_model]
        """Build a single conditioning vector per sample."""
        mert_cond  = self.mert_proj(torch.cat([local_mert, song_emb, section_emb], dim=-1))
        saber_cond = self.saber_proj(saber_state)
        phrase_cond = self.phrase_proj(phrase_feat)

        diff_e  = self.diff_emb(difficulty)
        genre_e = self.genre_emb(genre)

        combined = mert_cond + saber_cond + phrase_cond
        combined = self.cond_fuse(
            torch.cat([combined, diff_e, genre_e], dim=-1)
        )
        return combined.unsqueeze(1)   # [B, 1, d_model] — memory for cross-attn

    def forward(
        self,
        decoder_input:  torch.Tensor,   # [B, S] token IDs (teacher-forced)
        local_mert:     torch.Tensor,   # [B, 768]
        song_emb:       torch.Tensor,   # [B, 768]
        section_emb:    torch.Tensor,   # [B, 768]
        saber_state:    torch.Tensor,   # [B, 12]
        phrase_feat:    torch.Tensor,   # [B, 768]
        difficulty:     torch.Tensor,   # [B]
        genre:          torch.Tensor,   # [B]
    ) -> torch.Tensor:                  # [B, S, vocab_size]
        """Teacher-forced forward pass for training."""
        B, S = decoder_input.shape
        memory = self._build_conditioning(
            local_mert, song_emb, section_emb, saber_state, phrase_feat,
            difficulty, genre,
        )  # [B, 1, d_model]

        positions = torch.arange(S, device=decoder_input.device)
        x = self.token_emb(decoder_input)               # [B, S, d_model]
        x = x + self.pos_emb(positions).unsqueeze(0)

        # Causal mask
        causal_mask = nn.Transformer.generate_square_subsequent_mask(S, device=x.device)

        x = self.decoder(x, memory, tgt_mask=causal_mask, tgt_is_causal=True)
        x = self.norm(x)
        return self.output(x)   # [B, S, vocab_size]

    @torch.no_grad()
    def generate(
        self,
        local_mert:  torch.Tensor,  # [1, 768]
        song_emb:    torch.Tensor,  # [1, 768]
        section_emb: torch.Tensor,  # [1, 768]
        saber_state: torch.Tensor,  # [1, 12]
        phrase_feat: torch.Tensor,  # [1, 768]
        difficulty:  torch.Tensor,  # [1]
        genre:       torch.Tensor,  # [1]
        temperature: float = 0.9,
        top_p:       float = 0.9,
        max_new_tokens: int = 8,
    ) -> list[int]:
        """Autoregressive generation for a single note at inference time.

        Returns the generated token IDs (spatial tokens only, no BOS/EOS/PAD).
        """
        memory = self._build_conditioning(
            local_mert, song_emb, section_emb, saber_state, phrase_feat,
            difficulty, genre,
        )

        tokens = [LAYOUT_BOS]
        for _ in range(max_new_tokens):
            inp = torch.tensor([tokens], device=local_mert.device)
            S   = inp.shape[1]
            pos = torch.arange(S, device=inp.device)
            x   = self.token_emb(inp) + self.pos_emb(pos).unsqueeze(0)
            mask = nn.Transformer.generate_square_subsequent_mask(S, device=x.device)
            x   = self.decoder(x, memory, tgt_mask=mask, tgt_is_causal=True)
            x   = self.norm(x)
            logits = self.output(x)[:, -1, :].squeeze(0)   # [vocab]

            tok = _nucleus_sample(logits, temperature, top_p)
            if tok in (LAYOUT_EOS, LAYOUT_PAD):
                break
            tokens.append(tok)

        return tokens[1:]   # strip BOS


def _nucleus_sample(logits: torch.Tensor, temperature: float, top_p: float) -> int:
    logits = logits / max(temperature, 1e-6)
    probs  = torch.softmax(logits, dim=-1)
    sorted_probs, sorted_idx = torch.sort(probs, descending=True)
    cumulative = torch.cumsum(sorted_probs, dim=0)
    nucleus = sorted_idx[cumulative - sorted_probs <= top_p]
    if len(nucleus) == 0:
        nucleus = sorted_idx[:1]
    return int(nucleus[torch.randint(len(nucleus), (1,))].item())
