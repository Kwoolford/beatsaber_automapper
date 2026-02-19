"""Tests for the OnsetModel."""

import torch

from beatsaber_automapper.models.onset_model import OnsetModel


class TestOnsetModel:
    """Tests for OnsetModel."""

    def test_output_shape(self):
        """Output should be [B, T] logits."""
        model = OnsetModel(d_model=128, nhead=4, num_layers=1)
        features = torch.randn(2, 64, 128)
        difficulty = torch.tensor([0, 3])
        out = model(features, difficulty)
        assert out.shape == (2, 64)

    def test_all_difficulties(self):
        """Should accept all 5 difficulty levels without error."""
        model = OnsetModel(d_model=64, nhead=4, num_layers=1)
        features = torch.randn(1, 32, 64)
        for d in range(5):
            out = model(features, torch.tensor([d]))
            assert out.shape == (1, 32)

    def test_different_difficulties_different_output(self):
        """Different difficulties should produce different logits."""
        model = OnsetModel(d_model=64, nhead=4, num_layers=1)
        model.eval()
        features = torch.randn(1, 32, 64)
        out_easy = model(features, torch.tensor([0]))
        out_expert = model(features, torch.tensor([3]))
        assert not torch.allclose(out_easy, out_expert)

    def test_gradient_flow(self):
        """Gradients should flow back through the model."""
        model = OnsetModel(d_model=64, nhead=4, num_layers=1)
        features = torch.randn(1, 32, 64, requires_grad=True)
        difficulty = torch.tensor([2])
        out = model(features, difficulty)
        loss = out.sum()
        loss.backward()
        assert features.grad is not None
        assert features.grad.abs().sum() > 0

    def test_variable_sequence_length(self):
        """Should handle different sequence lengths."""
        model = OnsetModel(d_model=64, nhead=4, num_layers=1)
        for t in [16, 64, 256]:
            features = torch.randn(1, t, 64)
            out = model(features, torch.tensor([0]))
            assert out.shape == (1, t)
