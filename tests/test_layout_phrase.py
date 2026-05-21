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
