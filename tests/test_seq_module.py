"""Tests for Stage 2 SequenceLitModule (V6)."""

import pytest
import torch

from beatsaber_automapper.data.swing_tokenizer import (
    BOS,
    EOS,
    HAND_LEFT,
    HAND_RIGHT,
    NOTE,
    PAD,
    VOCAB_SIZE,
)
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
        genre = torch.tensor([0, 1])
        logits = module(mel, tokens, difficulty, genre)
        assert logits.shape == (2, 10, VOCAB_SIZE)

    def test_training_step_returns_scalar(self, module):
        batch = {
            "mel": torch.randn(2, 80, 32),
            "tokens": torch.randint(3, VOCAB_SIZE, (2, 10)),  # avoid PAD/BOS/EOS
            "difficulty": torch.tensor([0, 3]),
            "genre": torch.tensor([0, 2]),
        }
        loss = module.training_step(batch, 0)
        assert loss.ndim == 0
        assert loss.requires_grad

    def test_training_step_with_v6_batch(self, module):
        """Full V6 batch with saber_state and mapper_id."""
        batch = {
            "mel": torch.randn(2, 80, 32),
            "tokens": torch.randint(3, VOCAB_SIZE, (2, 10)),
            "difficulty": torch.tensor([0, 3]),
            "genre": torch.tensor([0, 2]),
            "saber_state": torch.randn(2, 10, 12),
            "mapper_id": torch.tensor([0, 1]),
        }
        loss = module.training_step(batch, 0)
        assert loss.ndim == 0
        assert loss.requires_grad

    def test_forward_with_saber_state(self, module):
        mel = torch.randn(2, 80, 32)
        tokens = torch.randint(3, VOCAB_SIZE, (2, 8))
        difficulty = torch.tensor([0, 4])
        genre = torch.tensor([0, 1])
        saber_state = torch.randn(2, 8, 12)
        logits = module(mel, tokens, difficulty, genre, saber_state=saber_state)
        assert logits.shape == (2, 8, VOCAB_SIZE)


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
        trainable = [p for p in module.sequence_model.parameters() if p.requires_grad]
        assert len(trainable) > 0


class TestTeacherForcing:
    """Test standard LM teacher-forcing split.

    Dataset tokens are [BOS, t0, t1, ..., tN, PAD...].
    decoder_input = tokens[:-1] = [BOS, t0, ..., tN]
    target        = tokens[1:]  = [t0,  t1, ..., PAD]
    """

    def test_prepare_teacher_forcing(self, module):
        # Real token format: starts with BOS, ends with EOS
        tokens = torch.tensor([[BOS, HAND_LEFT, 10, NOTE, 44, 48, 51, 60, EOS]])
        dec_input, target = module._prepare_teacher_forcing(tokens)

        assert dec_input.shape == (1, tokens.shape[1] - 1)
        assert target.shape == (1, tokens.shape[1] - 1)
        assert dec_input[0, 0].item() == BOS          # decoder input starts with BOS
        assert torch.equal(target[0], tokens[0, 1:])  # target is left-shifted by 1

    def test_target_is_left_shifted(self, module):
        tokens = torch.tensor([[BOS, HAND_LEFT, 10, NOTE, 44, 48, 51, 60, EOS]])
        dec_input, target = module._prepare_teacher_forcing(tokens)
        assert torch.equal(target[0], tokens[0, 1:])

    def test_decoder_input_is_tokens_without_last(self, module):
        tokens = torch.tensor([[BOS, HAND_LEFT, 10, NOTE, 44, 48, 51, 60, EOS]])
        dec_input, _ = module._prepare_teacher_forcing(tokens)
        assert torch.equal(dec_input[0], tokens[0, :-1])


class TestMapperConditioning:
    """Test mapper_id embedding conditioning."""

    def test_mapper_emb_changes_output(self):
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
            seq_num_mappers=18,
            seq_dropout=0.0,
        )
        module.eval()
        mel = torch.randn(1, 80, 32)
        tokens = torch.randint(3, VOCAB_SIZE, (1, 6))
        difficulty = torch.tensor([3])
        genre = torch.tensor([0])

        with torch.no_grad():
            logits_0 = module(mel, tokens, difficulty, genre, mapper_id=torch.tensor([0]))
            logits_5 = module(mel, tokens, difficulty, genre, mapper_id=torch.tensor([5]))

        assert not torch.allclose(logits_0, logits_5, atol=1e-5)
