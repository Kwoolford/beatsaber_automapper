"""Tests for Stage 2 SequenceModel."""

import pytest
import torch

from beatsaber_automapper.data.tokenizer import PAD, VOCAB_SIZE
from beatsaber_automapper.models.sequence_model import SequenceModel


@pytest.fixture
def model():
    return SequenceModel(
        vocab_size=VOCAB_SIZE,
        d_model=64,
        nhead=4,
        num_layers=1,
        dim_feedforward=128,
        num_difficulties=5,
        num_genres=11,
        dropout=0.0,
    )


class TestSequenceModelForward:
    """Test forward pass shapes and behavior."""

    def test_output_shape(self, model):
        tokens = torch.randint(0, VOCAB_SIZE, (2, 8))
        audio = torch.randn(2, 16, 64)
        difficulty = torch.tensor([0, 3])
        genre = torch.tensor([0, 1])
        logits = model(tokens, audio, difficulty, genre)
        assert logits.shape == (2, 8, VOCAB_SIZE)

    def test_single_token(self, model):
        tokens = torch.randint(0, VOCAB_SIZE, (1, 1))
        audio = torch.randn(1, 4, 64)
        difficulty = torch.tensor([2])
        genre = torch.tensor([0])
        logits = model(tokens, audio, difficulty, genre)
        assert logits.shape == (1, 1, VOCAB_SIZE)

    def test_decode_step_shape(self, model):
        tokens = torch.randint(0, VOCAB_SIZE, (1, 5))
        audio = torch.randn(1, 10, 64)
        difficulty = torch.tensor([1])
        genre = torch.tensor([0])
        logits = model.decode_step(tokens, audio, difficulty, genre)
        assert logits.shape == (1, VOCAB_SIZE)


class TestCausalMasking:
    """Test that causal masking prevents future token leakage."""

    def test_future_tokens_dont_affect_earlier_logits(self, model):
        model.eval()
        audio = torch.randn(1, 10, 64)
        difficulty = torch.tensor([0])
        genre = torch.tensor([0])

        tokens_a = torch.tensor([[3, 4, 10, 19, 12, 16, 28, 1]])  # some valid tokens
        tokens_b = tokens_a.clone()
        tokens_b[0, 4:] = torch.randint(0, VOCAB_SIZE, (4,))  # change future tokens

        with torch.no_grad():
            logits_a = model(tokens_a, audio, difficulty, genre)
            logits_b = model(tokens_b, audio, difficulty, genre)

        # Logits at positions 0-3 should be identical regardless of future tokens
        for pos in range(4):
            torch.testing.assert_close(
                logits_a[0, pos],
                logits_b[0, pos],
                msg=f"Position {pos} logits differ when only future tokens changed",
            )


class TestDifficultyConditioning:
    """Test that difficulty affects output."""

    def test_different_difficulties_different_output(self, model):
        model.eval()
        tokens = torch.randint(4, VOCAB_SIZE, (1, 6))
        audio = torch.randn(1, 10, 64)
        genre = torch.tensor([0])

        with torch.no_grad():
            logits_easy = model(tokens, audio, torch.tensor([0]), genre)
            logits_expert = model(tokens, audio, torch.tensor([4]), genre)

        assert not torch.allclose(logits_easy, logits_expert, atol=1e-5)


class TestGradientFlow:
    """Test that gradients flow through the model."""

    def test_gradient_flows(self, model):
        tokens = torch.randint(4, VOCAB_SIZE, (2, 8))
        audio = torch.randn(2, 16, 64)
        difficulty = torch.tensor([0, 3])
        genre = torch.tensor([0, 1])

        logits = model(tokens, audio, difficulty, genre)
        loss = logits.sum()
        loss.backward()

        # Check that token embedding has gradients
        assert model.token_emb.weight.grad is not None
        assert model.out_proj.weight.grad is not None


class TestPadEmbedding:
    """Test that PAD token embedding is zero."""

    def test_pad_embedding_is_zero(self, model):
        pad_emb = model.token_emb.weight[PAD]
        assert torch.all(pad_emb == 0)
