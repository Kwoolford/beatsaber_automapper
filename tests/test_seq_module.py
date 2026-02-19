"""Tests for Stage 2 SequenceLitModule."""

import pytest
import torch

from beatsaber_automapper.data.tokenizer import VOCAB_SIZE
from beatsaber_automapper.training.seq_module import SequenceLitModule


@pytest.fixture
def module():
    return SequenceLitModule(
        n_mels=80,
        encoder_d_model=64,
        encoder_nhead=4,
        encoder_num_layers=1,
        encoder_dim_feedforward=128,
        encoder_dropout=0.0,
        vocab_size=VOCAB_SIZE,
        seq_d_model=64,
        seq_nhead=4,
        seq_num_layers=1,
        seq_dim_feedforward=128,
        seq_num_difficulties=5,
        seq_dropout=0.0,
        label_smoothing=0.1,
        learning_rate=1e-4,
        weight_decay=0.01,
        warmup_steps=100,
        freeze_encoder=False,
    )


class TestSequenceLitModuleForward:
    """Test forward pass of the Lightning module."""

    def test_forward_shape(self, module):
        mel = torch.randn(2, 80, 32)
        tokens = torch.randint(0, VOCAB_SIZE, (2, 10))
        difficulty = torch.tensor([0, 4])
        logits = module(mel, tokens, difficulty)
        assert logits.shape == (2, 10, VOCAB_SIZE)

    def test_training_step_returns_scalar(self, module):
        batch = {
            "mel": torch.randn(2, 80, 32),
            "tokens": torch.randint(4, VOCAB_SIZE, (2, 10)),
            "difficulty": torch.tensor([0, 3]),
        }
        loss = module.training_step(batch, 0)
        assert loss.ndim == 0
        assert loss.requires_grad


class TestFreezeEncoder:
    """Test freeze_encoder functionality."""

    def test_encoder_frozen(self):
        module = SequenceLitModule(
            n_mels=80,
            encoder_d_model=64,
            encoder_nhead=4,
            encoder_num_layers=1,
            encoder_dim_feedforward=128,
            encoder_dropout=0.0,
            vocab_size=VOCAB_SIZE,
            seq_d_model=64,
            seq_nhead=4,
            seq_num_layers=1,
            seq_dim_feedforward=128,
            seq_num_difficulties=5,
            seq_dropout=0.0,
            freeze_encoder=True,
        )
        for param in module.audio_encoder.parameters():
            assert not param.requires_grad

    def test_sequence_model_trainable_when_encoder_frozen(self):
        module = SequenceLitModule(
            n_mels=80,
            encoder_d_model=64,
            encoder_nhead=4,
            encoder_num_layers=1,
            encoder_dim_feedforward=128,
            encoder_dropout=0.0,
            vocab_size=VOCAB_SIZE,
            seq_d_model=64,
            seq_nhead=4,
            seq_num_layers=1,
            seq_dim_feedforward=128,
            seq_num_difficulties=5,
            seq_dropout=0.0,
            freeze_encoder=True,
        )
        # Sequence model params should still be trainable
        trainable = [p for p in module.sequence_model.parameters() if p.requires_grad]
        assert len(trainable) > 0


class TestTeacherForcing:
    """Test BOS prepend logic."""

    def test_prepare_teacher_forcing(self, module):
        from beatsaber_automapper.data.tokenizer import BOS

        tokens = torch.tensor([[4, 10, 19, 1, 0, 0]])  # some tokens + PAD
        dec_input, target = module._prepare_teacher_forcing(tokens)

        assert dec_input.shape == tokens.shape
        assert target.shape == tokens.shape
        assert dec_input[0, 0].item() == BOS
        # decoder_input[1:] = tokens[:-1]
        assert torch.equal(dec_input[0, 1:], tokens[0, :-1])
        assert torch.equal(target, tokens)
