"""Tests for V6-5 StyleDiscriminator."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F  # noqa: N812

from beatsaber_automapper.data.swing_tokenizer import (
    BOS,
    EOS,
    HAND_LEFT,
    NOTE,
    PAD,
    VOCAB_SIZE,
    X_BASE,
    Y_BASE,
)
from beatsaber_automapper.training.style_discriminator import StyleDiscriminator


@pytest.fixture
def small_disc():
    return StyleDiscriminator(
        vocab_size=VOCAB_SIZE,
        num_mappers=4,
        audio_d_model=32,
        d_model=16,
        nhead=2,
        num_layers=1,
        dim_feedforward=32,
        dropout=0.0,
    )


# ---------------------------------------------------------------------------
# Shape / basic invariants
# ---------------------------------------------------------------------------


def test_init_attributes(small_disc):
    assert small_disc.vocab_size == VOCAB_SIZE
    assert small_disc.num_mappers == 4
    assert small_disc.d_model == 16
    assert small_disc.audio_d_model == 32


def test_forward_tokens_shape(small_disc):
    audio = torch.randn(2, 32)
    tokens = torch.randint(0, VOCAB_SIZE, (2, 20), dtype=torch.long)
    logits = small_disc(audio, tokens)
    assert logits.shape == (2, 4)


def test_forward_soft_probs_shape(small_disc):
    audio = torch.randn(2, 32)
    probs = F.softmax(torch.randn(2, 20, VOCAB_SIZE), dim=-1)
    logits = small_disc(audio, probs)
    assert logits.shape == (2, 4)


def test_forward_rejects_wrong_dim(small_disc):
    audio = torch.randn(1, 32)
    bad = torch.randn(1, 20, 20, 20)
    with pytest.raises(ValueError, match="swing_input must be"):
        small_disc(audio, bad)


def test_forward_rejects_wrong_vocab(small_disc):
    audio = torch.randn(1, 32)
    bad_probs = F.softmax(torch.randn(1, 20, 99), dim=-1)
    with pytest.raises(ValueError, match="vocab dim"):
        small_disc(audio, bad_probs)


# ---------------------------------------------------------------------------
# Equivalence: one-hot probs should match integer tokens
# ---------------------------------------------------------------------------


def test_one_hot_matches_integer_tokens(small_disc):
    small_disc.eval()
    audio = torch.randn(2, 32)
    tokens = torch.tensor([
        [BOS, HAND_LEFT, NOTE, X_BASE + 1, Y_BASE + 0, EOS, PAD, PAD],
        [BOS, HAND_LEFT, NOTE, X_BASE + 2, Y_BASE + 1, EOS, PAD, PAD],
    ], dtype=torch.long)
    one_hot = F.one_hot(tokens, num_classes=VOCAB_SIZE).float()
    padding_mask = tokens == PAD

    with torch.no_grad():
        logits_int = small_disc(audio, tokens)
        logits_oh = small_disc(audio, one_hot, padding_mask=padding_mask)

    torch.testing.assert_close(logits_int, logits_oh, atol=1e-5, rtol=1e-5)


# ---------------------------------------------------------------------------
# Gradient flow through soft probs (the V6-5 critical path)
# ---------------------------------------------------------------------------


def test_gradient_flows_through_soft_probs(small_disc):
    audio = torch.randn(2, 32)
    logits = torch.randn(2, 10, VOCAB_SIZE, requires_grad=True)
    probs = F.softmax(logits, dim=-1)
    out = small_disc(audio, probs)
    out.sum().backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
    assert logits.grad.abs().max() > 0


def test_match_log_prob_grad(small_disc):
    audio = torch.randn(2, 32)
    raw = torch.randn(2, 10, VOCAB_SIZE, requires_grad=True)
    probs = F.softmax(raw, dim=-1)
    target = torch.tensor([0, 2])
    lp = small_disc.match_log_prob(audio, probs, target)
    assert lp.ndim == 0  # scalar
    lp.backward()
    assert raw.grad is not None
    assert raw.grad.abs().max() > 0


def test_match_log_prob_value_consistent(small_disc):
    small_disc.eval()
    audio = torch.randn(1, 32)
    tokens = torch.randint(0, VOCAB_SIZE, (1, 8), dtype=torch.long)
    target = torch.tensor([1])
    with torch.no_grad():
        logits = small_disc(audio, tokens)
        manual = torch.log_softmax(logits, dim=-1)[0, 1]
        lp = small_disc.match_log_prob(audio, tokens, target)
    torch.testing.assert_close(lp, manual)


# ---------------------------------------------------------------------------
# Conditioning checks
# ---------------------------------------------------------------------------


def test_audio_conditioning_changes_output(small_disc):
    small_disc.eval()
    tokens = torch.randint(0, VOCAB_SIZE, (1, 8), dtype=torch.long)
    a1 = torch.randn(1, 32)
    a2 = torch.randn(1, 32)
    with torch.no_grad():
        l1 = small_disc(a1, tokens)
        l2 = small_disc(a2, tokens)
    assert not torch.allclose(l1, l2, atol=1e-5)


def test_swing_conditioning_changes_output(small_disc):
    small_disc.eval()
    audio = torch.randn(1, 32)
    t1 = torch.randint(0, VOCAB_SIZE, (1, 8), dtype=torch.long)
    t2 = torch.randint(0, VOCAB_SIZE, (1, 8), dtype=torch.long)
    while torch.equal(t1, t2):
        t2 = torch.randint(0, VOCAB_SIZE, (1, 8), dtype=torch.long)
    with torch.no_grad():
        l1 = small_disc(audio, t1)
        l2 = small_disc(audio, t2)
    assert not torch.allclose(l1, l2, atol=1e-5)


# ---------------------------------------------------------------------------
# Padding mask
# ---------------------------------------------------------------------------


def test_padding_mask_changes_pooling(small_disc):
    """Tokens after the padding boundary should not affect the output."""
    small_disc.eval()
    audio = torch.randn(1, 32)
    base = torch.tensor([[HAND_LEFT, NOTE, X_BASE, Y_BASE, EOS, PAD, PAD, PAD]], dtype=torch.long)
    # Corrupt the tail with different values
    corrupted = base.clone()
    corrupted[0, 5:] = torch.tensor([20, 30, 40], dtype=torch.long)
    padding_mask = base == PAD  # uses base's PAD layout for both

    with torch.no_grad():
        l_base = small_disc(audio, base, padding_mask=padding_mask)
        l_corr = small_disc(audio, corrupted, padding_mask=padding_mask)
    torch.testing.assert_close(l_base, l_corr, atol=1e-5, rtol=1e-5)


def test_padding_mask_auto_inferred_for_tokens(small_disc):
    small_disc.eval()
    audio = torch.randn(1, 32)
    base = torch.tensor([[HAND_LEFT, NOTE, X_BASE, Y_BASE, EOS, PAD, PAD]], dtype=torch.long)
    corrupted = base.clone()
    corrupted[0, 5:] = torch.tensor([20, 30], dtype=torch.long)
    # No padding_mask supplied — base auto-infers via PAD comparison
    with torch.no_grad():
        l_base = small_disc(audio, base)
        # But corrupted has different non-PAD values in the tail; auto-inferred mask differs
        l_corr = small_disc(audio, corrupted)
    assert not torch.allclose(l_base, l_corr)


# ---------------------------------------------------------------------------
# Convenience methods
# ---------------------------------------------------------------------------


def test_predict_mapper_shape(small_disc):
    audio = torch.randn(3, 32)
    tokens = torch.randint(0, VOCAB_SIZE, (3, 8), dtype=torch.long)
    preds = small_disc.predict_mapper(audio, tokens)
    assert preds.shape == (3,)
    assert preds.dtype == torch.long
    assert (preds >= 0).all() and (preds < 4).all()


def test_pad_token_embedding_is_zero(small_disc):
    """PAD should be the padding_idx (zero embedding) so it doesn't contribute."""
    assert torch.all(small_disc.token_emb.weight[PAD] == 0.0)
