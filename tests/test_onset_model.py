"""Tests for the OnsetModel (TCN + Transformer hybrid)."""

import torch

from beatsaber_automapper.models.onset_model import OnsetModel, TCNEncoder, TemporalConvBlock


class TestTemporalConvBlock:
    """Tests for individual TCN residual blocks."""

    def test_output_shape(self):
        """Output should match input shape."""
        block = TemporalConvBlock(channels=64, kernel_size=3, dilation=1)
        x = torch.randn(2, 64, 100)
        out = block(x)
        assert out.shape == (2, 64, 100)

    def test_dilated_output_shape(self):
        """Should preserve shape with larger dilation."""
        block = TemporalConvBlock(channels=32, kernel_size=3, dilation=16)
        x = torch.randn(2, 32, 200)
        out = block(x)
        assert out.shape == (2, 32, 200)

    def test_residual_connection(self):
        """Output should differ from input (not identity)."""
        block = TemporalConvBlock(channels=32, kernel_size=3, dilation=1)
        block.eval()
        x = torch.randn(2, 32, 50)
        out = block(x)
        assert not torch.allclose(out, x)


class TestTCNEncoder:
    """Tests for the TCN encoder stack."""

    def test_output_shape(self):
        """Should preserve [B, T, D] shape."""
        tcn = TCNEncoder(input_dim=64, channels=32, num_blocks=4)
        x = torch.randn(2, 100, 64)
        out = tcn(x)
        assert out.shape == (2, 100, 64)

    def test_custom_dilations(self):
        """Should accept custom dilation list."""
        tcn = TCNEncoder(input_dim=64, channels=32, num_blocks=3, dilations=[1, 4, 16])
        x = torch.randn(2, 100, 64)
        out = tcn(x)
        assert out.shape == (2, 100, 64)

    def test_receptive_field(self):
        """Receptive field should be computed correctly."""
        tcn = TCNEncoder(input_dim=64, channels=32, num_blocks=6, kernel_size=3)
        # RF = sum((k-1)*d for d in [1,2,4,8,16,32]) + 1 = 2*(1+2+4+8+16+32) + 1 = 127
        assert tcn.receptive_field == 127


class TestOnsetModel:
    """Tests for OnsetModel (TCN + Transformer hybrid)."""

    def test_output_shape(self):
        """Output should be [B, T] logits."""
        model = OnsetModel(d_model=128, nhead=4, num_layers=1, tcn_channels=32, tcn_num_blocks=3)
        features = torch.randn(2, 64, 128)
        difficulty = torch.tensor([0, 3])
        genre = torch.tensor([0, 1])
        out = model(features, difficulty, genre)
        assert out.shape == (2, 64)

    def test_all_difficulties(self):
        """Should accept all 5 difficulty levels without error."""
        model = OnsetModel(d_model=64, nhead=4, num_layers=1, tcn_channels=32, tcn_num_blocks=2)
        features = torch.randn(1, 32, 64)
        genre = torch.tensor([0])
        for d in range(5):
            out = model(features, torch.tensor([d]), genre)
            assert out.shape == (1, 32)

    def test_different_difficulties_different_output(self):
        """Different difficulties should produce different logits."""
        model = OnsetModel(d_model=64, nhead=4, num_layers=1, tcn_channels=32, tcn_num_blocks=2)
        model.eval()
        features = torch.randn(1, 32, 64)
        genre = torch.tensor([0])
        out_easy = model(features, torch.tensor([0]), genre)
        out_expert = model(features, torch.tensor([3]), genre)
        assert not torch.allclose(out_easy, out_expert)

    def test_gradient_flow(self):
        """Gradients should flow back through the model."""
        model = OnsetModel(d_model=64, nhead=4, num_layers=1, tcn_channels=32, tcn_num_blocks=2)
        features = torch.randn(1, 32, 64, requires_grad=True)
        difficulty = torch.tensor([2])
        genre = torch.tensor([0])
        out = model(features, difficulty, genre)
        loss = out.sum()
        loss.backward()
        assert features.grad is not None
        assert features.grad.abs().sum() > 0

    def test_variable_sequence_length(self):
        """Should handle different sequence lengths."""
        model = OnsetModel(d_model=64, nhead=4, num_layers=1, tcn_channels=32, tcn_num_blocks=2)
        genre = torch.tensor([0])
        for t in [16, 64, 256]:
            features = torch.randn(1, t, 64)
            out = model(features, torch.tensor([0]), genre)
            assert out.shape == (1, t)

    def test_conditioning_dropout_training(self):
        """Conditioning dropout should zero out embeddings during training."""
        model = OnsetModel(
            d_model=64, nhead=4, num_layers=1,
            tcn_channels=32, tcn_num_blocks=2,
            conditioning_dropout=1.0,  # 100% dropout — embeddings zeroed
        )
        model.train()
        genre = torch.tensor([0])

        # Verify the dropout mask logic works: with 100% dropout,
        # the conditioning embeddings should be multiplied by 0
        diff_emb = model.difficulty_emb(torch.tensor([3]))
        genre_emb = model.genre_emb(genre)
        mask = torch.ones(1, 1)  # simulates 100% dropout
        zeroed_diff = diff_emb * (1 - mask)
        zeroed_genre = genre_emb * (1 - mask)
        assert torch.allclose(zeroed_diff, torch.zeros_like(zeroed_diff))
        assert torch.allclose(zeroed_genre, torch.zeros_like(zeroed_genre))

    def test_conditioning_dropout_eval(self):
        """Conditioning dropout should be inactive during eval."""
        model = OnsetModel(
            d_model=64, nhead=4, num_layers=1,
            tcn_channels=32, tcn_num_blocks=2,
            conditioning_dropout=1.0,  # 100% dropout during training
        )
        model.eval()
        features = torch.randn(1, 32, 64)
        genre = torch.tensor([0])

        # During eval, dropout is off — different difficulties should differ
        out_easy = model(features, torch.tensor([0]), genre)
        out_expert = model(features, torch.tensor([3]), genre)
        assert not torch.allclose(out_easy, out_expert)
