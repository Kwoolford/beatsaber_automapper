"""V7-5b: Phrase-level Layout model.

Encoder-decoder transformer that generates the spatial token sequence for ALL
notes in a phrase as one causal pass. Saber state is gone — position, direction,
and parity are emergent properties of the decoder's prior-token self-attention.

Per-token metadata embeddings (slot-in-phrase, hand, role) tell the decoder
"this token is the X coordinate of the LEFT note at slot 12" — replacing the
hand-engineered saber-state summary that the previous design needed.

Encoder:
    phrase_mert [B, P, 768] + slot pos emb → encoder_out [B, P, d_model]

Decoder:
    token emb + slot emb + hand emb + role emb + (diff + genre once)
    → causal self-attn + cross-attn to encoder_out
    → output_proj [B, S, vocab]
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from beatsaber_automapper.data.layout_dataset import (
    HAND_SPECIAL_IDX,
    LAYOUT_PAD,
    LAYOUT_VOCAB_SIZE,
    MAX_PHRASE_SLOTS,
    N_HANDS_EMB,
    N_ROLES,
)

# Re-export so generation paths can import from one place.
__all__ = ["LayoutPhraseModel", "LAYOUT_PAD"]

MERT_DIM = 768


class LayoutPhraseModel(nn.Module):
    """Phrase-level autoregressive layout generator.

    Args:
        vocab_size:        Layout token vocab size (118 — same IDs as swing vocab).
        d_model:           Transformer hidden dim.
        n_heads:           Attention heads.
        n_enc_layers:      Encoder layers.
        n_dec_layers:      Decoder layers.
        dim_feedforward:   FFN inner dim.
        max_layout_len:    Max decoder seq length.
        max_phrase_slots:  Max encoder seq length (phrase length cap).
        num_difficulties:  Difficulty embedding size.
        num_genres:        Genre embedding size.
        dropout:           Dropout rate.
    """

    def __init__(
        self,
        vocab_size:       int = LAYOUT_VOCAB_SIZE,
        d_model:          int = 384,
        n_heads:          int = 6,
        n_enc_layers:     int = 3,
        n_dec_layers:     int = 4,
        dim_feedforward:  int = 1536,
        max_layout_len:   int = 384,
        max_phrase_slots: int = MAX_PHRASE_SLOTS,
        num_difficulties: int = 5,
        num_genres:       int = 11,
        dropout:          float = 0.1,
    ) -> None:
        super().__init__()
        self.d_model         = d_model
        self.max_layout_len  = max_layout_len
        self.max_phrase_slots = max_phrase_slots
        # Special-slot sentinel index matches the dataset's convention: the last
        # row of slot_emb. Tied to max_phrase_slots so dataset and model agree.
        self.special_slot_idx = max_phrase_slots

        # ---- Encoder side ----
        self.enc_proj    = nn.Linear(MERT_DIM, d_model)
        self.enc_pos_emb = nn.Embedding(max_phrase_slots, d_model)
        self.enc_norm    = nn.LayerNorm(d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_enc_layers)

        # ---- Decoder side ----
        # Token embedding (padding_idx=0 → LAYOUT_PAD never contributes a gradient)
        self.tok_emb  = nn.Embedding(vocab_size, d_model, padding_idx=LAYOUT_PAD)
        # +1 row on slot_emb for the SPECIAL_SLOT_IDX sentinel (BOS / EOS / PAD)
        self.slot_emb = nn.Embedding(max_phrase_slots + 1, d_model)
        self.hand_emb = nn.Embedding(N_HANDS_EMB, d_model)
        self.role_emb = nn.Embedding(N_ROLES,     d_model)
        self.dec_pos_emb = nn.Embedding(max_layout_len, d_model)
        self.dec_in_norm = nn.LayerNorm(d_model)

        # Global conditioning (difficulty + genre) → added once to every decoder pos
        self.diff_emb  = nn.Embedding(num_difficulties, d_model)
        self.genre_emb = nn.Embedding(num_genres,       d_model)

        dec_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=n_dec_layers)

        self.out_norm = nn.LayerNorm(d_model)
        self.out_proj = nn.Linear(d_model, vocab_size)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # ------------------------------------------------------------------
    def encode(
        self,
        phrase_mert: torch.Tensor,    # [B, P, 768]
        phrase_mask: torch.Tensor,    # [B, P]   bool: True = real
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode phrase MERT to a context memory.

        Returns:
            memory          [B, P, d_model]
            memory_kp_mask  [B, P]   True where the encoder position is PADDING
                                     (PyTorch convention for `memory_key_padding_mask`).
        """
        B, P, _ = phrase_mert.shape
        device  = phrase_mert.device
        pos = torch.arange(P, device=device)
        x = self.enc_proj(phrase_mert) + self.enc_pos_emb(pos).unsqueeze(0)
        x = self.enc_norm(x)
        # PyTorch wants the mask to mark PADDING with True.
        kp_mask = ~phrase_mask
        memory = self.encoder(x, src_key_padding_mask=kp_mask)
        return memory, kp_mask

    def forward(
        self,
        layout_tokens: torch.Tensor,   # [B, S]
        token_slot:    torch.Tensor,   # [B, S]
        token_hand:    torch.Tensor,   # [B, S]
        token_role:    torch.Tensor,   # [B, S]
        phrase_mert:   torch.Tensor,   # [B, P, 768]
        phrase_mask:   torch.Tensor,   # [B, P]
        difficulty:    torch.Tensor,   # [B]
        genre:         torch.Tensor,   # [B]
    ) -> torch.Tensor:                  # [B, S, vocab]
        """Teacher-forced training forward.

        `layout_tokens` is the input sequence (BOS, ...) — the target sequence
        is the shift-by-one of this (handled in the LightningModule).
        """
        memory, mem_kp = self.encode(phrase_mert, phrase_mask)

        B, S = layout_tokens.shape
        device = layout_tokens.device
        pos = torch.arange(S, device=device)

        x = (self.tok_emb(layout_tokens)
             + self.slot_emb(token_slot)
             + self.hand_emb(token_hand)
             + self.role_emb(token_role)
             + self.dec_pos_emb(pos).unsqueeze(0))

        # Global conditioning added once (broadcasts over the sequence dim)
        cond = (self.diff_emb(difficulty) + self.genre_emb(genre)).unsqueeze(1)  # [B,1,d]
        x = x + cond
        x = self.dec_in_norm(x)

        # Causal mask. The decoder shouldn't peek at future tokens.
        causal = nn.Transformer.generate_square_subsequent_mask(S, device=device)

        # Decoder-side padding mask: don't waste attention on PAD positions.
        # (Loss is masked separately by IGNORE_INDEX in the target tensor.)
        tgt_kp = layout_tokens == LAYOUT_PAD

        y = self.decoder(
            tgt=x,
            memory=memory,
            tgt_mask=causal,
            tgt_is_causal=True,
            tgt_key_padding_mask=tgt_kp,
            memory_key_padding_mask=mem_kp,
        )
        y = self.out_norm(y)
        return self.out_proj(y)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def generate_phrase(
        self,
        phrase_mert: torch.Tensor,         # [1, P, 768]
        phrase_mask: torch.Tensor,         # [1, P]
        onset_schedule: list[tuple[int, int]],   # [(slot_in_phrase, hand_idx), ...]
        difficulty: torch.Tensor,          # [1]
        genre: torch.Tensor,               # [1]
        temperature: float = 0.9,
        top_p: float = 0.9,
    ) -> list[int]:
        """Greedy/nucleus-sampled phrase generation.

        Walks the onset schedule emitted by Stage 1; for each onset it samples
        a variable-length spatial token sequence (KIND → X → Y → [DIR] → [FIELD_D])
        with the per-token role/slot/hand metadata set externally per step.

        Returns the flat token sequence (no BOS/EOS/PAD).
        """
        from beatsaber_automapper.data.layout_dataset import (
            LAYOUT_BOS, LAYOUT_EOS,
            ROLE_KIND, ROLE_X, ROLE_Y, ROLE_DIR, ROLE_FIELD_D, ROLE_SPECIAL,
        )
        from beatsaber_automapper.data.swing_tokenizer import (
            BOMB, CHAIN_TAIL, KIND_BASE, KIND_COUNT,
        )

        device = phrase_mert.device
        memory, mem_kp = self.encode(phrase_mert, phrase_mask)

        # Safety: each onset produces ≤5 tokens; BOS already occupies 1 slot.
        # Truncate the schedule so we never exceed max_layout_len.
        max_onsets = (self.max_layout_len - 1) // 5
        onset_schedule = onset_schedule[:max_onsets]

        # Running token / metadata buffers — appended per step.
        toks   = [LAYOUT_BOS]
        slots  = [self.special_slot_idx]
        hands  = [HAND_SPECIAL_IDX]
        roles  = [ROLE_SPECIAL]

        def _step(role: int, slot: int, hand: int) -> int:
            slots.append(slot)
            hands.append(hand)
            roles.append(role)
            # Forward-once with current buffers and sample the new token.
            S = len(toks) + 1
            x = (self.tok_emb(torch.tensor([toks + [LAYOUT_PAD]], device=device))
                 + self.slot_emb(torch.tensor([slots], device=device))
                 + self.hand_emb(torch.tensor([hands], device=device))
                 + self.role_emb(torch.tensor([roles], device=device))
                 + self.dec_pos_emb(torch.arange(S, device=device)).unsqueeze(0))
            cond = (self.diff_emb(difficulty) + self.genre_emb(genre)).unsqueeze(1)
            x = x + cond
            x = self.dec_in_norm(x)
            causal = nn.Transformer.generate_square_subsequent_mask(S, device=device)
            y = self.decoder(
                tgt=x, memory=memory, tgt_mask=causal, tgt_is_causal=True,
                memory_key_padding_mask=mem_kp,
            )
            y = self.out_norm(y)
            logits = self.out_proj(y)[:, -1, :].squeeze(0)
            tok = _nucleus_sample(logits, temperature, top_p)
            toks.append(int(tok))
            return int(tok)

        for slot_in_phrase, hand_idx in onset_schedule:
            kind_tok = _step(ROLE_KIND, slot_in_phrase, hand_idx)
            # Clamp into the KIND range if the sample wandered out (rare with low temp)
            if not (KIND_BASE <= kind_tok < KIND_BASE + KIND_COUNT):
                kind_tok = KIND_BASE  # fall back to NOTE
                toks[-1] = kind_tok

            _step(ROLE_X, slot_in_phrase, hand_idx)
            _step(ROLE_Y, slot_in_phrase, hand_idx)

            if kind_tok == BOMB:
                continue
            if kind_tok == CHAIN_TAIL:
                _step(ROLE_FIELD_D, slot_in_phrase, hand_idx)
                continue
            _step(ROLE_DIR,     slot_in_phrase, hand_idx)
            _step(ROLE_FIELD_D, slot_in_phrase, hand_idx)

        # Strip BOS, no explicit EOS appended (caller knows the schedule).
        return toks[1:]


def _nucleus_sample(logits: torch.Tensor, temperature: float, top_p: float) -> int:
    logits = logits / max(temperature, 1e-6)
    probs = torch.softmax(logits, dim=-1)
    sorted_p, sorted_i = torch.sort(probs, descending=True)
    cumulative = torch.cumsum(sorted_p, dim=0)
    keep = sorted_i[cumulative - sorted_p <= top_p]
    if len(keep) == 0:
        keep = sorted_i[:1]
    return int(keep[torch.randint(len(keep), (1,))].item())
