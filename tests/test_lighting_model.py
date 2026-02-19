"""Tests for LightingModel and LightingLitModule."""

from __future__ import annotations

import pytest
import torch

from beatsaber_automapper.data.tokenizer import LIGHT_VOCAB_SIZE, VOCAB_SIZE
from beatsaber_automapper.models.lighting_model import LightingModel


@pytest.fixture
def model() -> LightingModel:
    return LightingModel(
        light_vocab_size=LIGHT_VOCAB_SIZE,
        note_vocab_size=VOCAB_SIZE,
        d_model=64,
        nhead=4,
        num_layers=1,
        dim_feedforward=128,
        dropout=0.0,
    )


class TestLightingModelForward:
    def test_output_shape(self, model):
        light_tokens = torch.randint(0, LIGHT_VOCAB_SIZE, (2, 8))
        audio = torch.randn(2, 16, 64)
        note_tokens = torch.randint(0, VOCAB_SIZE, (2, 10))
        logits = model(light_tokens, audio, note_tokens)
        assert logits.shape == (2, 8, LIGHT_VOCAB_SIZE)

    def test_single_token(self, model):
        light_tokens = torch.randint(0, LIGHT_VOCAB_SIZE, (1, 1))
        audio = torch.randn(1, 4, 64)
        note_tokens = torch.randint(0, VOCAB_SIZE, (1, 5))
        logits = model(light_tokens, audio, note_tokens)
        assert logits.shape == (1, 1, LIGHT_VOCAB_SIZE)

    def test_decode_step_shape(self, model):
        light_tokens = torch.randint(0, LIGHT_VOCAB_SIZE, (1, 3))
        audio = torch.randn(1, 8, 64)
        note_tokens = torch.randint(0, VOCAB_SIZE, (1, 6))
        logits = model.decode_step(light_tokens, audio, note_tokens)
        assert logits.shape == (1, LIGHT_VOCAB_SIZE)

    def test_all_pad_note_tokens(self, model):
        """Model should handle all-PAD note context without crashing."""
        light_tokens = torch.randint(0, LIGHT_VOCAB_SIZE, (2, 4))
        audio = torch.randn(2, 8, 64)
        note_tokens = torch.zeros(2, 8, dtype=torch.long)  # all PAD
        logits = model(light_tokens, audio, note_tokens)
        assert logits.shape == (2, 4, LIGHT_VOCAB_SIZE)

    def test_output_finite(self, model):
        """All output logits should be finite."""
        light_tokens = torch.randint(1, LIGHT_VOCAB_SIZE, (2, 6))
        audio = torch.randn(2, 12, 64)
        note_tokens = torch.randint(1, VOCAB_SIZE, (2, 8))
        logits = model(light_tokens, audio, note_tokens)
        assert torch.isfinite(logits).all()

    def test_batch_independence(self, model):
        """Each sample in a batch should be processed independently."""
        model.eval()
        light = torch.randint(1, LIGHT_VOCAB_SIZE, (1, 5))
        audio = torch.randn(1, 8, 64)
        note = torch.randint(1, VOCAB_SIZE, (1, 4))

        logits_single = model(light, audio, note)

        # Stack same sample twice
        light2 = light.repeat(2, 1)
        audio2 = audio.repeat(2, 1, 1)
        note2 = note.repeat(2, 1)
        logits_batch = model(light2, audio2, note2)

        assert torch.allclose(logits_single, logits_batch[:1], atol=1e-5)


class TestLightingLitModule:
    @pytest.fixture
    def lit_module(self):
        from beatsaber_automapper.training.light_module import LightingLitModule

        return LightingLitModule(
            n_mels=80,
            encoder_d_model=64,
            encoder_nhead=4,
            encoder_num_layers=1,
            encoder_dim_feedforward=128,
            encoder_dropout=0.0,
            light_vocab_size=LIGHT_VOCAB_SIZE,
            note_vocab_size=VOCAB_SIZE,
            light_d_model=64,
            light_nhead=4,
            light_num_layers=1,
            light_dim_feedforward=128,
            light_dropout=0.0,
            label_smoothing=0.0,
        )

    def test_forward_shape(self, lit_module):
        mel = torch.randn(2, 80, 32)
        light_tokens = torch.randint(1, LIGHT_VOCAB_SIZE, (2, 6))
        note_tokens = torch.randint(1, VOCAB_SIZE, (2, 8))
        logits = lit_module(mel, light_tokens, note_tokens)
        assert logits.shape == (2, 6, LIGHT_VOCAB_SIZE)

    def test_training_step_returns_scalar(self, lit_module):
        batch = {
            "mel": torch.randn(2, 80, 32),
            "light_tokens": torch.randint(1, LIGHT_VOCAB_SIZE, (2, 8)),
            "note_tokens": torch.randint(1, VOCAB_SIZE, (2, 10)),
            "difficulty": torch.tensor([3, 3]),
        }
        loss = lit_module.training_step(batch, 0)
        assert loss.ndim == 0
        assert loss.item() > 0

    def test_teacher_forcing_shift(self, lit_module):
        from beatsaber_automapper.data.tokenizer import LIGHT_BOS

        tokens = torch.tensor([[5, 10, 15, 20, 1]])  # ends with LIGHT_EOS=1
        dec_input, target = lit_module._prepare_teacher_forcing(tokens)
        assert dec_input[0, 0].item() == LIGHT_BOS
        assert torch.equal(target, tokens)

    def test_configure_optimizers_returns_dict(self, lit_module):
        # Attach a mock trainer so lr_lambda doesn't break
        class MockTrainer:
            estimated_stepping_batches = 1000

        lit_module.trainer = MockTrainer()
        result = lit_module.configure_optimizers()
        assert "optimizer" in result
        assert "lr_scheduler" in result

    def test_freeze_encoder(self):
        from beatsaber_automapper.training.light_module import LightingLitModule

        module = LightingLitModule(
            encoder_d_model=64,
            encoder_nhead=4,
            encoder_num_layers=1,
            encoder_dim_feedforward=128,
            light_d_model=64,
            light_nhead=4,
            light_num_layers=1,
            light_dim_feedforward=128,
            freeze_encoder=True,
        )
        for param in module.audio_encoder.parameters():
            assert not param.requires_grad
        for param in module.lighting_model.parameters():
            assert param.requires_grad
