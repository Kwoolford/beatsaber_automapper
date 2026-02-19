"""Tests for the AudioEncoder model."""

import pytest
import torch

from beatsaber_automapper.models.audio_encoder import AudioEncoder


class TestAudioEncoder:
    """Tests for AudioEncoder."""

    def test_output_shape(self):
        """Output should be [B, T, d_model] with time preserved."""
        enc = AudioEncoder(n_mels=80, d_model=256, num_layers=1, dim_feedforward=512)
        mel = torch.randn(2, 80, 64)
        out = enc(mel)
        assert out.shape == (2, 64, 256)

    def test_time_preserved(self):
        """Time dimension must be preserved through the CNN."""
        enc = AudioEncoder(n_mels=80, d_model=128, num_layers=1, dim_feedforward=256)
        for t in [32, 100, 256]:
            mel = torch.randn(1, 80, t)
            out = enc(mel)
            assert out.shape[1] == t, f"Time {t} not preserved: got {out.shape[1]}"

    def test_gradient_flow(self):
        """Gradients should flow back through the encoder."""
        enc = AudioEncoder(n_mels=80, d_model=128, num_layers=1, dim_feedforward=256)
        mel = torch.randn(1, 80, 32, requires_grad=True)
        out = enc(mel)
        loss = out.sum()
        loss.backward()
        assert mel.grad is not None
        assert mel.grad.abs().sum() > 0

    def test_various_n_mels(self):
        """Should work with different n_mels values divisible by 16."""
        for n_mels in [16, 48, 80, 128]:
            enc = AudioEncoder(n_mels=n_mels, d_model=64, num_layers=1, dim_feedforward=128)
            mel = torch.randn(1, n_mels, 16)
            out = enc(mel)
            assert out.shape == (1, 16, 64)

    def test_invalid_n_mels(self):
        """Should reject n_mels not divisible by 16."""
        with pytest.raises(ValueError, match="divisible by 16"):
            AudioEncoder(n_mels=50)

    def test_batch_independence(self):
        """Different batch items should produce different outputs."""
        enc = AudioEncoder(n_mels=80, d_model=64, num_layers=1, dim_feedforward=128)
        enc.eval()
        mel = torch.randn(2, 80, 32)
        out = enc(mel)
        assert not torch.allclose(out[0], out[1])
