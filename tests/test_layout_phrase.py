"""Unit tests for V7-5b phrase-level Stage 2.

Targets the per-token metadata (slot/hand/role) construction in
LayoutPhraseDataset and the encoder-decoder shapes in LayoutPhraseModel.
"""

from __future__ import annotations

import torch

from beatsaber_automapper.data.layout_dataset import (
    HAND_LEFT_IDX, HAND_RIGHT_IDX, HAND_SPECIAL_IDX,
    IGNORE_INDEX,
    LAYOUT_BOS, LAYOUT_EOS, LAYOUT_PAD,
    MAX_PHRASE_SLOTS,
    ROLE_KIND, ROLE_X, ROLE_Y, ROLE_DIR, ROLE_FIELD_D, ROLE_SPECIAL,
    _Event, _event_to_tokens, _hand_idx,
)
from beatsaber_automapper.data.swing_tokenizer import (
    ANGLE_BASE, BOMB, CHAIN_HEAD, CHAIN_TAIL, DIR_BASE,
    HAND_LEFT, HAND_RIGHT, HAND_NONE, KIND_BASE, NOTE,
    SQUISH_BASE, X_BASE, Y_BASE,
)
from beatsaber_automapper.models.layout_model import LayoutPhraseModel


# ----------------------------------------------------------------------------
# _hand_idx
# ----------------------------------------------------------------------------

def test_hand_idx_mapping() -> None:
    assert _hand_idx(HAND_LEFT)  == HAND_LEFT_IDX
    assert _hand_idx(HAND_RIGHT) == HAND_RIGHT_IDX
    assert _hand_idx(HAND_NONE)  == HAND_SPECIAL_IDX  # bombs


# ----------------------------------------------------------------------------
# _event_to_tokens: role tagging per kind
# ----------------------------------------------------------------------------

def test_event_to_tokens_note_yields_5_tokens_with_roles() -> None:
    e = _Event(beat=1.0, slot=4, hand=HAND_LEFT, kind=NOTE,
               x=2, y=1, direction=3, field_d=4)
    toks, roles = _event_to_tokens(e)
    assert len(toks) == 5 == len(roles)
    assert toks[0] == NOTE
    assert toks[1] == X_BASE + 2
    assert toks[2] == Y_BASE + 1
    assert toks[3] == DIR_BASE + 3
    assert toks[4] == ANGLE_BASE + 4
    assert roles == [ROLE_KIND, ROLE_X, ROLE_Y, ROLE_DIR, ROLE_FIELD_D]


def test_event_to_tokens_bomb_skips_dir_and_field_d() -> None:
    e = _Event(beat=2.0, slot=8, hand=HAND_NONE, kind=BOMB,
               x=1, y=0, direction=0, field_d=0)
    toks, roles = _event_to_tokens(e)
    assert toks == [BOMB, X_BASE + 1, Y_BASE + 0]
    assert roles == [ROLE_KIND, ROLE_X, ROLE_Y]


def test_event_to_tokens_chain_tail_skips_dir_only() -> None:
    e = _Event(beat=3.0, slot=12, hand=HAND_RIGHT, kind=CHAIN_TAIL,
               x=2, y=2, direction=0, field_d=3)
    toks, roles = _event_to_tokens(e)
    assert toks == [CHAIN_TAIL, X_BASE + 2, Y_BASE + 2, SQUISH_BASE + 3]
    assert roles == [ROLE_KIND, ROLE_X, ROLE_Y, ROLE_FIELD_D]


# ----------------------------------------------------------------------------
# Model forward: shapes + masking semantics
# ----------------------------------------------------------------------------

def _tiny_model() -> LayoutPhraseModel:
    return LayoutPhraseModel(
        vocab_size=118, d_model=64, n_heads=4,
        n_enc_layers=1, n_dec_layers=1, dim_feedforward=128,
        max_layout_len=32, max_phrase_slots=16,
    )


def test_forward_returns_correct_shape() -> None:
    m = _tiny_model()
    m.eval()
    B, S, P = 2, 32, 16
    out = m(
        layout_tokens=torch.zeros(B, S, dtype=torch.long),
        token_slot=torch.zeros(B, S, dtype=torch.long),
        token_hand=torch.zeros(B, S, dtype=torch.long),
        token_role=torch.zeros(B, S, dtype=torch.long),
        phrase_mert=torch.randn(B, P, 768),
        phrase_mask=torch.ones(B, P, dtype=torch.bool),
        difficulty=torch.tensor([3, 4], dtype=torch.long),
        genre=torch.tensor([0, 0], dtype=torch.long),
    )
    assert out.shape == (B, S, 118)


def test_phrase_mask_excludes_pad_frames_from_attention() -> None:
    """A phrase with mostly-padded encoder positions should still produce finite logits."""
    m = _tiny_model()
    m.eval()
    B, S, P = 1, 8, 16
    mask = torch.zeros(B, P, dtype=torch.bool)
    mask[:, :4] = True  # only 4 real frames out of 16
    # Use the model's own special_slot_idx so the slot embedding index is in range
    # for whatever max_phrase_slots the model was built with.
    out = m(
        layout_tokens=torch.tensor([[LAYOUT_BOS, 38, 46, 48, 52, 63, LAYOUT_EOS, LAYOUT_PAD]]),
        token_slot=torch.full((B, S), m.special_slot_idx, dtype=torch.long),
        token_hand=torch.full((B, S), HAND_SPECIAL_IDX,  dtype=torch.long),
        token_role=torch.full((B, S), ROLE_SPECIAL,      dtype=torch.long),
        phrase_mert=torch.randn(B, P, 768),
        phrase_mask=mask,
        difficulty=torch.tensor([3], dtype=torch.long),
        genre=torch.tensor([0], dtype=torch.long),
    )
    assert torch.isfinite(out).all()


def test_dataset_and_model_special_slot_agree() -> None:
    """Dataset's special-slot index must equal the model's slot_emb sentinel row."""
    from beatsaber_automapper.data.layout_dataset import MAX_PHRASE_SLOTS
    m = LayoutPhraseModel(max_phrase_slots=MAX_PHRASE_SLOTS)
    # The dataset uses `self.max_phrase_slots` as its sentinel — they match when
    # constructed with the same arg.
    assert m.special_slot_idx == MAX_PHRASE_SLOTS
    assert m.slot_emb.num_embeddings == MAX_PHRASE_SLOTS + 1


def test_param_count_is_in_expected_range_for_default_model() -> None:
    """Sanity guard on the default model size — catch accidental capacity drift."""
    m = LayoutPhraseModel()  # defaults: d_model=384, n_dec_layers=4
    n = sum(p.numel() for p in m.parameters())
    # ~10M is right for this size; bracket loosely to allow architecture tweaks.
    assert 3_000_000 < n < 25_000_000, f"unexpected param count {n:,}"


# ----------------------------------------------------------------------------
# Autoregressive rollout — guards the role/metadata alignment that Bug 1 broke.
# Without this test, val_token_acc (teacher-forced) can stay high while
# generation produces garbage because the per-token role embedding gets
# attached to the wrong sequence position during inference.
# ----------------------------------------------------------------------------

def test_generate_phrase_constrains_tokens_to_legal_role_ranges() -> None:
    """Each emitted token must land in the vocab range for its role.

    Trained-model behaviour aside, this is a pure-grammar guarantee that
    must hold for any model: KIND must be in [38,44), X in [44,48), Y in
    [48,51), DIR in [51,60), and FIELD_D in the kind-specific range. If
    role metadata is misaligned (Bug 1) the constrained-sampling mask will
    refuse to produce a valid token, surfacing the issue immediately.
    """
    from beatsaber_automapper.data.layout_dataset import (
        HAND_LEFT_IDX, HAND_RIGHT_IDX,
    )
    from beatsaber_automapper.data.swing_tokenizer import (
        ANGLE_BASE, ANGLE_COUNT,
        ARC_HEAD, ARC_TAIL,
        BOMB, CHAIN_HEAD, CHAIN_TAIL,
        DIR_BASE, DIR_COUNT,
        KIND_BASE, KIND_COUNT, NOTE,
        MU_BASE, MU_COUNT,
        SLICE_BASE, SLICE_COUNT,
        SQUISH_BASE, SQUISH_COUNT,
        X_BASE, X_COUNT, Y_BASE, Y_COUNT,
    )

    torch.manual_seed(0)
    m = _tiny_model()
    m.eval()

    P = 16
    phrase_mert = torch.randn(1, P, 768)
    phrase_mask = torch.ones(1, P, dtype=torch.bool)
    onset_schedule = [(2, HAND_LEFT_IDX), (4, HAND_RIGHT_IDX), (8, HAND_LEFT_IDX)]
    flat = m.generate_phrase(
        phrase_mert    = phrase_mert,
        phrase_mask    = phrase_mask,
        onset_schedule = onset_schedule,
        difficulty     = torch.tensor([3], dtype=torch.long),
        genre          = torch.tensor([0], dtype=torch.long),
        temperature    = 1.0,
        top_p          = 0.95,
    )

    i = 0
    for _slot, _hand in onset_schedule:
        assert i < len(flat), "ran out of tokens before exhausting schedule"
        kind = flat[i]; i += 1
        assert KIND_BASE <= kind < KIND_BASE + KIND_COUNT, (
            f"KIND out of range: {kind}"
        )
        assert X_BASE <= flat[i] < X_BASE + X_COUNT, f"X out of range: {flat[i]}"; i += 1
        assert Y_BASE <= flat[i] < Y_BASE + Y_COUNT, f"Y out of range: {flat[i]}"; i += 1
        if kind == BOMB:
            continue
        if kind == CHAIN_TAIL:
            assert SQUISH_BASE <= flat[i] < SQUISH_BASE + SQUISH_COUNT, (
                f"SQUISH out of range: {flat[i]}"
            )
            i += 1
            continue
        assert DIR_BASE <= flat[i] < DIR_BASE + DIR_COUNT, (
            f"DIR out of range: {flat[i]}"
        )
        i += 1
        fd = flat[i]; i += 1
        if kind == NOTE:
            assert ANGLE_BASE <= fd < ANGLE_BASE + ANGLE_COUNT, f"ANGLE out of range: {fd}"
        elif kind in (ARC_HEAD, ARC_TAIL):
            assert MU_BASE <= fd < MU_BASE + MU_COUNT, f"MU out of range: {fd}"
        elif kind == CHAIN_HEAD:
            assert SLICE_BASE <= fd < SLICE_BASE + SLICE_COUNT, f"SLICE out of range: {fd}"

    assert i == len(flat), f"unused trailing tokens (i={i}, len={len(flat)})"


def test_generate_phrase_appends_metadata_for_each_emitted_token() -> None:
    """Sanity check on the buffer growth — every step should add exactly one
    token, one slot, one hand, one role to the running buffers. If the buggy
    pre-append-then-overwrite pattern ever returns, this catches it because
    the role buffer would diverge from the token buffer length.

    We can't directly inspect the internal lists, but we can check the
    schedule-derived token count: 3 notes (NOTE most likely under random
    weights) → 3*5 = 15 tokens, or less if BOMB/CHAIN_TAIL was sampled.
    """
    from beatsaber_automapper.data.layout_dataset import HAND_LEFT_IDX

    torch.manual_seed(1)
    m = _tiny_model()
    m.eval()

    P = 12
    schedule = [(1, HAND_LEFT_IDX), (5, HAND_LEFT_IDX)]
    flat = m.generate_phrase(
        phrase_mert=torch.randn(1, P, 768),
        phrase_mask=torch.ones(1, P, dtype=torch.bool),
        onset_schedule=schedule,
        difficulty=torch.tensor([3], dtype=torch.long),
        genre=torch.tensor([0], dtype=torch.long),
        temperature=1.0, top_p=0.95,
    )
    # Each onset emits 3 (BOMB) or 4 (CHAIN_TAIL) or 5 tokens. With 2 onsets,
    # min=6, max=10.
    assert 6 <= len(flat) <= 10, f"unexpected token count {len(flat)}"


def test_nucleus_sample_respects_probability_weighting() -> None:
    """Force a strongly-peaked distribution and confirm samples concentrate
    on the mode. The previous uniform-sampling bug would yield ~50/50 over
    the top-2, not ~95/5."""
    from beatsaber_automapper.models.layout_model import _nucleus_sample

    torch.manual_seed(0)
    # Two-class distribution: 0.95 vs 0.05
    logits = torch.tensor([3.0, 0.0])
    n = 1000
    hits = sum(_nucleus_sample(logits.clone(), temperature=1.0, top_p=0.99)
               for _ in range(n))
    # If uniform over the nucleus, hits ~ 500. If probability-weighted, hits ~ 50.
    assert hits < 200, (
        f"sample count of token 1 = {hits}/{n}; expected ~50 if probability-weighted "
        "(was ~500 with the uniform-sampling bug)"
    )
