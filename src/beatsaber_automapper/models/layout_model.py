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

import logging
import math
import os

import torch
import torch.nn as nn

_log = logging.getLogger(__name__)

from beatsaber_automapper.data.layout_dataset import (
    CONTOUR_DIM,
    HAND_SPECIAL_IDX,
    LAYOUT_PAD,
    LAYOUT_VOCAB_SIZE,
    MAX_PHRASE_SLOTS,
    N_HANDS_EMB,
    N_ROLES,
    ROLE_CONTEXT,
    ROLE_SPECIAL,
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
        max_song_phrases: int = 150,
        num_difficulties: int = 5,
        num_genres:       int = 11,
        dropout:          float = 0.1,
        use_contour:      bool = False,
    ) -> None:
        super().__init__()
        self.d_model          = d_model
        self.max_layout_len   = max_layout_len
        self.max_phrase_slots = max_phrase_slots
        self.max_song_phrases = max_song_phrases
        self.use_contour      = use_contour
        self.ctx_len          = 0   # set by LayoutPhraseLitModule when ctx_len > 0
        # Special-slot sentinel index matches the dataset's convention.
        self.special_slot_idx = max_phrase_slots

        # ---- Encoder side (local: current phrase slots) ----
        self.enc_proj    = nn.Linear(MERT_DIM, d_model)
        self.enc_pos_emb = nn.Embedding(max_phrase_slots, d_model)
        self.enc_norm    = nn.LayerNorm(d_model)
        # TASK 3: per-slot pitch contour → encoder. Only instantiated when
        # use_contour=True so checkpoints trained without it load with no
        # missing keys (mirrors the song_fp_proj guard).
        self.contour_proj = nn.Linear(CONTOUR_DIM, d_model) if use_contour else None
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_enc_layers)

        # ---- Song-memory side (global: all phrase fingerprints) ----
        # Only instantiated when max_song_phrases > 0 so old checkpoints
        # (trained without song-memory) load cleanly with no missing keys.
        if max_song_phrases > 0:
            self.song_fp_proj    = nn.Linear(MERT_DIM, d_model, bias=False)
            self.song_fp_pos_emb = nn.Embedding(max_song_phrases, d_model)
            self.song_fp_norm    = nn.LayerNorm(d_model)
        else:
            self.song_fp_proj    = None
            self.song_fp_pos_emb = None
            self.song_fp_norm    = None

        # ---- Decoder side ----
        self.tok_emb  = nn.Embedding(vocab_size, d_model, padding_idx=LAYOUT_PAD)
        self.slot_emb = nn.Embedding(max_phrase_slots + 1, d_model)
        self.hand_emb = nn.Embedding(N_HANDS_EMB, d_model)
        self.role_emb = nn.Embedding(N_ROLES,     d_model)
        self.dec_pos_emb = nn.Embedding(max_layout_len, d_model)
        self.dec_in_norm = nn.LayerNorm(d_model)

        # Global scalar conditioning (difficulty + genre only — song/section
        # level now lives in the song_memory cross-attention, not a scalar)
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
        phrase_mert:   torch.Tensor,            # [B, P, 768]
        phrase_mask:   torch.Tensor,            # [B, P]   bool: True = real
        song_fps:      torch.Tensor | None = None,   # [B, M, 768] phrase fingerprints
        song_fp_mask:  torch.Tensor | None = None,   # [B, M]      bool: True = real
        phrase_contour: torch.Tensor | None = None,  # [B, P, 3] per-slot pitch contour
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode phrase MERT + optional song-level phrase fingerprints.

        The local encoder processes the current phrase's per-slot MERT features
        (dynamic but local). If song_fps is provided, the resulting encodings
        are concatenated with projected phrase fingerprints for the whole song,
        giving the decoder a global, dynamic memory it can attend to selectively.

        Returns:
            memory         [B, P + M, d_model]   (M=0 if no song_fps)
            memory_kp_mask [B, P + M]             True = padding (PyTorch convention)
        """
        B, P, _ = phrase_mert.shape
        device  = phrase_mert.device

        # Local: encode current phrase slots
        pos = torch.arange(P, device=device)
        x = self.enc_proj(phrase_mert) + self.enc_pos_emb(pos).unsqueeze(0)
        # TASK 3: add per-slot pitch-contour anchor. Slot-aligned with phrase_mert,
        # so the decoder cross-attention sees lead/bass pitch at each onset's slot.
        if self.contour_proj is not None and phrase_contour is not None:
            x = x + self.contour_proj(phrase_contour)
        x = self.enc_norm(x)
        local_kp = ~phrase_mask                          # [B, P]
        memory_local = self.encoder(x, src_key_padding_mask=local_kp)

        if song_fps is None or self.song_fp_proj is None:
            return memory_local, local_kp

        # Global: project all phrase fingerprints as additional memory
        B2, M, _ = song_fps.shape
        fp_pos = torch.arange(M, device=device)
        g = self.song_fp_proj(song_fps) + self.song_fp_pos_emb(fp_pos).unsqueeze(0)
        g = self.song_fp_norm(g)                         # [B, M, d_model]
        global_kp = ~song_fp_mask                        # [B, M] — True = padding

        memory    = torch.cat([memory_local, g],         dim=1)  # [B, P+M, d_model]
        kp_mask   = torch.cat([local_kp,     global_kp], dim=1)  # [B, P+M]
        return memory, kp_mask

    def forward(
        self,
        layout_tokens: torch.Tensor,             # [B, S]
        token_slot:    torch.Tensor,             # [B, S]
        token_hand:    torch.Tensor,             # [B, S]
        token_role:    torch.Tensor,             # [B, S]
        phrase_mert:   torch.Tensor,             # [B, P, 768]
        phrase_mask:   torch.Tensor,             # [B, P]
        difficulty:    torch.Tensor,             # [B]
        genre:         torch.Tensor,             # [B]
        song_fps:      torch.Tensor | None = None,   # [B, M, 768]
        song_fp_mask:  torch.Tensor | None = None,   # [B, M]  bool True=real
        phrase_contour: torch.Tensor | None = None,  # [B, P, 3]
        song_emb:      torch.Tensor | None = None,   # kept for ckpt compat, unused
        section_emb:   torch.Tensor | None = None,   # kept for ckpt compat, unused
    ) -> torch.Tensor:                  # [B, S, vocab]
        """Teacher-forced training forward.

        `layout_tokens` is the input sequence (BOS, ...) — the target sequence
        is the shift-by-one of this (handled in the LightningModule).
        """
        memory, mem_kp = self.encode(
            phrase_mert, phrase_mask, song_fps, song_fp_mask, phrase_contour,
        )

        B, S = layout_tokens.shape
        device = layout_tokens.device
        pos = torch.arange(S, device=device)

        x = (self.tok_emb(layout_tokens)
             + self.slot_emb(token_slot)
             + self.hand_emb(token_hand)
             + self.role_emb(token_role)
             + self.dec_pos_emb(pos).unsqueeze(0))

        # Global scalar conditioning: difficulty + genre
        # Song/section level now lives in the song_fps memory cross-attention.
        cond = (self.diff_emb(difficulty) + self.genre_emb(genre)).unsqueeze(1)
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
        top_p: float = 0.95,
        context_tokens: list[int] | None = None,      # last K tokens from prior phrase
        context_slots:  list[int] | None = None,
        context_hands:  list[int] | None = None,
        song_fps:     torch.Tensor | None = None,      # [1, M, 768] all phrase fingerprints
        song_fp_mask: torch.Tensor | None = None,      # [1, M]      bool True=real
        phrase_contour: torch.Tensor | None = None,    # [1, P, 3] per-slot pitch contour
        song_emb:    torch.Tensor | None = None,       # kept for ckpt compat, unused
        section_emb: torch.Tensor | None = None,       # kept for ckpt compat, unused
    ) -> list[int]:
        """Greedy/nucleus-sampled phrase generation.

        Walks the onset schedule emitted by Stage 1; for each onset it samples
        a variable-length spatial token sequence (KIND → X → Y → [DIR] → [FIELD_D])
        with the per-token role/slot/hand metadata set externally per step.

        Convention: at decoder position i with input token T_i and metadata
        (slot_i, hand_i, role_i), the head predicts T_{i+1}. So to sample a
        token with role R, we forward the buffer as-is and read logits at the
        last real position — its head was trained to predict the next token,
        whose role we know is R (because we're driving the schedule). The
        sampled token's metadata is appended AFTER sampling.

        Returns the flat token sequence (no BOS/EOS/PAD).
        """
        from beatsaber_automapper.data.layout_dataset import (
            LAYOUT_BOS, LAYOUT_EOS,
            ROLE_KIND, ROLE_X, ROLE_Y, ROLE_DIR, ROLE_FIELD_D, ROLE_SPECIAL,
        )
        from beatsaber_automapper.data.swing_tokenizer import (
            ANGLE_BASE, ANGLE_COUNT,
            ARC_HEAD, ARC_TAIL, BOMB,
            CHAIN_HEAD, CHAIN_TAIL,
            DIR_BASE, DIR_COUNT,
            KIND_BASE, KIND_COUNT,
            MU_BASE, MU_COUNT,
            NOTE,
            SLICE_BASE, SLICE_COUNT,
            SQUISH_BASE, SQUISH_COUNT,
            X_BASE, X_COUNT,
            Y_BASE, Y_COUNT,
        )

        device = phrase_mert.device
        memory, mem_kp = self.encode(
            phrase_mert, phrase_mask, song_fps, song_fp_mask, phrase_contour,
        )

        # Safety: each onset produces ≤5 tokens; BOS already occupies 1 slot.
        # Truncate the schedule so we never exceed max_layout_len.
        ctx_n = len(context_tokens) if context_tokens else 0
        max_onsets = (self.max_layout_len - 1 - ctx_n) // 5
        onset_schedule = onset_schedule[:max_onsets]

        # Running token / metadata buffers — start with optional cross-phrase
        # context prefix (role=ROLE_CONTEXT) then BOS.
        if context_tokens and ctx_n > 0:
            toks  = list(context_tokens) + [LAYOUT_BOS]
            slots = list(context_slots)  + [self.special_slot_idx]
            hands = list(context_hands)  + [HAND_SPECIAL_IDX]
            roles = [ROLE_CONTEXT] * ctx_n + [ROLE_SPECIAL]
        else:
            toks   = [LAYOUT_BOS]
            slots  = [self.special_slot_idx]
            hands  = [HAND_SPECIAL_IDX]
            roles  = [ROLE_SPECIAL]

        # Decode-time layout-diversity penalty (2026-06-30): the model mode-
        # collapses to row0×{col0,col2} (row_conc 0.94 vs human 0.47) because
        # beam/nucleus decoding of its peaked per-position logits always picks
        # the conditional mode. Env-gated frequency penalty: subtract
        # strength*count(token) from X/Y logits before sampling so already-used
        # columns/rows get discouraged. Counts are per-phrase. Default OFF.
        _div_on = os.environ.get("LAYOUT_DIVERSITY") == "1"
        _diag_on = _div_on or os.environ.get("LAYOUT_DIAG") == "1"
        _div_x = float(os.environ.get("LAYOUT_DIV_X", "0.0"))
        _div_y = float(os.environ.get("LAYOUT_DIV_Y", "0.0"))
        _div_d = float(os.environ.get("LAYOUT_DIV_D", "0.0"))  # DIR (cut-direction) penalty; default OFF
        _x_counts = [0] * X_COUNT
        _y_counts = [0] * Y_COUNT
        _dir_counts = [0] * DIR_COUNT

        # Windowed ADJACENCY anti-repeat (targets the map_metrics `pattern_repeat`
        # sub-signal = adjacent-identical (x,y,dir) tuples). Unlike the cumulative
        # LAYOUT_DIV_* penalty above (which flattens the WHOLE-phrase distribution
        # and over-diversifies grid/rows past human), this only penalizes tokens
        # emitted in the last W steps *for the same role*, so it breaks back-to-back
        # loops without touching the global cell/dir spread.
        # PROMOTED TO PRODUCTION 2026-07-23: the antirepeat_2026-07-23 sweep winner
        # ar_w1_s2 (W=1/S=2.0) is now the default — h_dist 0.020 < prev prod 0.039
        # while holding density (4/6), monotony 0.43=human, col_conc 0.29~human,
        # 0 parity violations. Env still overrides (set LAYOUT_ANTIREPEAT=0 to disable
        # for a no-antirepeat control / ablation).
        _ar_w = int(os.environ.get("LAYOUT_ANTIREPEAT", "1"))       # recent-window size
        _ar_s = float(os.environ.get("LAYOUT_AR_STRENGTH", "2.0"))  # penalty per in-window hit
        _ar_on = _ar_w > 0 and _ar_s > 0.0
        # Which roles the adjacency anti-repeat applies to. Default "xyd" (all
        # three) preserves the 2026-07-23 promoted behaviour. eval-suite v2 axis
        # A7 (2026-07-27/28) found we invert the human direction idiom -- diagonal
        # share 0.513 vs human 0.358 -- and traced it to this penalty running on
        # ROLE_DIR: penalizing a repeated up/down cut pushes the model toward
        # diagonals as the "least recently used" escape, which is the opposite of
        # what a human mapper does (diagonals are the deviation, not the norm).
        # LAYOUT_ANTIREPEAT_ROLES=xy narrows the penalty to X/Y only, leaving DIR
        # to the model's own (human-trained) distribution.
        _ar_roles_str = os.environ.get("LAYOUT_ANTIREPEAT_ROLES", "xyd")
        _ar_hist: dict[int, list[int]] = {}
        if "x" in _ar_roles_str:
            _ar_hist[ROLE_X] = []
        if "y" in _ar_roles_str:
            _ar_hist[ROLE_Y] = []
        if "d" in _ar_roles_str:
            _ar_hist[ROLE_DIR] = []

        # Travel penalty (eval-suite v2 axis A1, 2026-07-27). The flow metrics show
        # our hands move ~50% further per second than human hands (travel shift
        # +2.48 human-MADs). Penalize placing a note far from the SAME HAND's
        # previous note, scaled by how little time there is to get there — which is
        # exactly the quantity `flow.travel` measures (grid distance / second).
        # Default 0.0 = OFF, prior behaviour unchanged.
        _tp_s = float(os.environ.get("LAYOUT_TRAVEL_PENALTY", "0.0"))
        _tp_on = _tp_s > 0.0
        _last_pos: dict[int, tuple[int, int, int]] = {}   # hand -> (x_idx, y_idx, slot)

        # Idiom bonus (eval-suite v2 axis A3, 2026-07-27). Human mapping draws on a
        # small vocabulary: 130k human transitions collapse to 2510 idioms, the top
        # 500 covering ~90% of everything human mappers do. Our maps use that
        # vocabulary measurably less (coverage shift -2.73 human-MADs). At the DIR
        # step we know this hand's previous note, so we know which cut directions
        # would COMPLETE a known human idiom — boost those. A bonus, never a hard
        # constraint: humans use out-of-vocabulary transitions ~8% of the time, and
        # forcing coverage to 1.0 would be its own non-human artifact.
        _ib_s = float(os.environ.get("LAYOUT_IDIOM_BONUS", "0.0"))
        _ib_on = _ib_s > 0.0
        _ib_vocab: set = set()
        _last_note: dict[int, tuple[int, int, int, int]] = {}  # hand->(x,y,dir,slot)
        if _ib_on:
            try:
                from beatsaber_automapper.evaluation import idiom as _idm
                _, _ranked, _ = _idm.load_vocab()
                _ib_vocab = set(_ranked[: _idm.TOP_K])
                _ib_dt_class = _idm.dt_class
            except Exception:  # noqa: BLE001
                _ib_on = False

        def _div_counts_for(role: int):
            """(per-phrase used-token counts, penalty strength) for a penalized role."""
            if role == ROLE_X:   return _x_counts, _div_x
            if role == ROLE_Y:   return _y_counts, _div_y
            return _dir_counts, _div_d  # ROLE_DIR
        _diag = {"x_pmax": 0.0, "y_pmax": 0.0, "n": 0,  # mean argmax-prob diagnostic
                 "y_samp": [0] * Y_COUNT, "x_samp": [0] * X_COUNT}  # sampled-token histogram

        def _legal_range(role: int, kind: int | None) -> tuple[int, int]:
            """Return (lo, hi) — half-open token-id range allowed at this role."""
            if role == ROLE_KIND:    return (KIND_BASE, KIND_BASE + KIND_COUNT)
            if role == ROLE_X:       return (X_BASE,    X_BASE    + X_COUNT)
            if role == ROLE_Y:       return (Y_BASE,    Y_BASE    + Y_COUNT)
            if role == ROLE_DIR:     return (DIR_BASE,  DIR_BASE  + DIR_COUNT)
            # FIELD_D vocab depends on the event's KIND.
            if kind == NOTE:                return (ANGLE_BASE,  ANGLE_BASE  + ANGLE_COUNT)
            if kind in (ARC_HEAD, ARC_TAIL): return (MU_BASE,     MU_BASE     + MU_COUNT)
            if kind == CHAIN_HEAD:           return (SLICE_BASE,  SLICE_BASE  + SLICE_COUNT)
            if kind == CHAIN_TAIL:           return (SQUISH_BASE, SQUISH_BASE + SQUISH_COUNT)
            # Defensive: fall back to angle range
            return (ANGLE_BASE, ANGLE_BASE + ANGLE_COUNT)

        def _step(role: int, slot: int, hand: int, kind: int | None) -> int:
            S = len(toks)
            x = (self.tok_emb(torch.tensor([toks], device=device))
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
            # Constrained sampling: zero-mass anything outside the legal range
            # for this role so even a high-temp sample stays in-grammar.
            lo, hi = _legal_range(role, kind)
            mask = torch.full_like(logits, float("-inf"))
            mask[lo:hi] = 0.0
            logits = logits + mask

            # Diagnostic (gated) + diversity penalty on the position roles.
            if _diag_on and role in (ROLE_X, ROLE_Y):
                with torch.no_grad():
                    p = torch.softmax(logits[lo:hi], dim=-1)
                    _diag["x_pmax" if role == ROLE_X else "y_pmax"] += float(p.max())
                    if role == ROLE_Y:
                        _diag["n"] += 1
            if _div_on and role in (ROLE_X, ROLE_Y, ROLE_DIR):
                counts, strength = _div_counts_for(role)
                if strength:
                    pen = torch.zeros_like(logits)
                    for j in range(hi - lo):
                        pen[lo + j] = -strength * counts[j]
                    logits = logits + pen
            # Travel penalty: discourage long hand jumps in short time windows.
            # Penalty is proportional to grid distance from this hand's previous
            # note and INVERSELY proportional to the slot gap, so a wide move is
            # cheap when the hand has time and expensive when it does not.
            if _tp_on and role in (ROLE_X, ROLE_Y) and kind == NOTE:
                prev = _last_pos.get(hand)
                if prev is not None:
                    pj = prev[0] if role == ROLE_X else prev[1]
                    gap = max(slot - prev[2], 1)
                    pen = torch.zeros_like(logits)
                    for j in range(hi - lo):
                        pen[lo + j] = -_tp_s * abs(j - pj) / gap
                    logits = logits + pen
            # Idiom bonus: at DIR we know (dx, dy, dir_from, dt_class), so each
            # candidate direction either completes a known human idiom or does not.
            if _ib_on and role == ROLE_DIR and kind == NOTE:
                prev = _last_note.get(hand)
                cur = _last_pos.get(hand)
                if prev is not None and cur is not None:
                    dx, dy = cur[0] - prev[0], cur[1] - prev[1]
                    # ⚠️Was hardcoded `/ 4.0` with the comment "BEAT_SUBDIV = 4". A
                    # hardcoded copy of a constant does not fail when the constant
                    # changes — it silently misclassifies (every idiom dt_class would
                    # be wrong at any other subdivision, with no error). Read the real
                    # one. Guarded by `0 < dt_beats <= 2.0` below, so the old bug was
                    # a wrong bonus rather than a crash — the quiet kind.
                    from beatsaber_automapper.data.mert_encoder import BEAT_SUBDIV
                    dt_beats = max(cur[2] - prev[3], 0) / float(BEAT_SUBDIV)
                    if 0.0 < dt_beats <= 2.0:
                        cls = _ib_dt_class(dt_beats)
                        bon = torch.zeros_like(logits)
                        for j in range(hi - lo):
                            if (dx, dy, prev[2], j, cls) in _ib_vocab:
                                bon[lo + j] = _ib_s
                        logits = logits + bon
            # Windowed adjacency anti-repeat: penalize token ids seen in the last
            # _ar_w emissions for this role (weighted by in-window multiplicity).
            if _ar_on and role in _ar_hist:
                hist = _ar_hist[role]
                if hist:
                    pen = torch.zeros_like(logits)
                    for prev in hist:
                        pen[prev] -= _ar_s
                    logits = logits + pen

            tok = _nucleus_sample(logits, temperature, top_p)
            if _diag_on and role in (ROLE_X, ROLE_Y):
                samp = _diag["x_samp"] if role == ROLE_X else _diag["y_samp"]
                j = int(tok) - lo
                if 0 <= j < len(samp):
                    samp[j] += 1
                if role == ROLE_Y and kind == NOTE:
                    _diag.setdefault("y_note", [0] * Y_COUNT)
                    if 0 <= j < Y_COUNT:
                        _diag["y_note"][j] += 1
            if _div_on and role in (ROLE_X, ROLE_Y, ROLE_DIR):
                counts, _ = _div_counts_for(role)
                counts[int(tok) - lo] += 1
            if _ar_on and role in _ar_hist:
                h = _ar_hist[role]
                h.append(int(tok))
                if len(h) > _ar_w:
                    del h[0]
            toks.append(int(tok))
            slots.append(slot)
            hands.append(hand)
            roles.append(role)
            return int(tok)

        for slot_in_phrase, hand_idx in onset_schedule:
            kind_tok = _step(ROLE_KIND, slot_in_phrase, hand_idx, kind=None)
            x_tok = _step(ROLE_X, slot_in_phrase, hand_idx, kind=kind_tok)
            y_tok = _step(ROLE_Y, slot_in_phrase, hand_idx, kind=kind_tok)
            if (_tp_on or _ib_on) and kind_tok == NOTE:
                _last_pos[hand_idx] = (x_tok - X_BASE, y_tok - Y_BASE, slot_in_phrase)

            if kind_tok == BOMB:
                continue
            if kind_tok == CHAIN_TAIL:
                _step(ROLE_FIELD_D, slot_in_phrase, hand_idx, kind=kind_tok)
                continue
            dir_tok = _step(ROLE_DIR, slot_in_phrase, hand_idx, kind=kind_tok)
            if _ib_on and kind_tok == NOTE:
                # this note becomes the idiom's "from" end for the next one
                _last_note[hand_idx] = (x_tok - X_BASE, y_tok - Y_BASE,
                                        dir_tok - DIR_BASE, slot_in_phrase)
            _step(ROLE_FIELD_D, slot_in_phrase, hand_idx, kind=kind_tok)

        if (_div_on or os.environ.get("LAYOUT_DIAG") == "1") and _diag["n"]:
            ys = _diag["y_samp"]; xs = _diag["x_samp"]
            ytot = max(sum(ys), 1); xtot = max(sum(xs), 1)
            yn = _diag.get("y_note", [0] * Y_COUNT); yntot = max(sum(yn), 1)
            _log.info("layout decode: argmax-prob X=%.3f Y=%.3f n=%d | SAMPLED rows=%s "
                      "NOTE-rows=%s cols=%s",
                      _diag["x_pmax"] / _diag["n"], _diag["y_pmax"] / _diag["n"], _diag["n"],
                      [round(c / ytot, 2) for c in ys],
                      [round(c / yntot, 2) for c in yn], [round(c / xtot, 2) for c in xs])

        # Strip the cross-phrase context prefix (ctx_n tokens) AND the BOS, so the
        # returned stream starts at the first generated event token. The decoder
        # `_decode_phrase_tokens` parses from index 0 expecting a KIND token; the
        # old `toks[1:]` left the ctx_n context tokens in front (off-by-ctx_n
        # misalignment that read X/Y/DIR from the wrong slots and collapsed every
        # note to row0 — the "for-sport" bug). ctx_n=0 ⇒ toks[1:] as before.
        return toks[ctx_n + 1:]


def _nucleus_sample(logits: torch.Tensor, temperature: float, top_p: float) -> int:
    """Probability-weighted nucleus sampling.

    Previous implementation used `torch.randint` over the kept indices, which
    samples uniformly inside the nucleus — collapsing model confidence and
    drastically over-weighting low-probability tokens. We renormalize the kept
    probabilities and draw from them with `torch.multinomial`.
    """
    logits = logits / max(temperature, 1e-6)
    probs = torch.softmax(logits, dim=-1)
    sorted_p, sorted_i = torch.sort(probs, descending=True)
    cumulative = torch.cumsum(sorted_p, dim=0)
    # Include every token whose EXCLUSIVE cumsum is ≤ top_p — i.e. the
    # smallest set that strictly crosses top_p. Always keeps the argmax.
    keep_mask = (cumulative - sorted_p) <= top_p
    kept_p = sorted_p[keep_mask]
    kept_i = sorted_i[keep_mask]
    if kept_p.numel() == 0:
        kept_p = sorted_p[:1]
        kept_i = sorted_i[:1]
    kept_p = kept_p / kept_p.sum()
    pick = int(torch.multinomial(kept_p, num_samples=1).item())
    return int(kept_i[pick].item())
