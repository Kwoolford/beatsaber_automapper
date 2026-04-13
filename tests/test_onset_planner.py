"""Tests for the OnsetPlanner (bidirectional song-level planning)."""

import torch

from beatsaber_automapper.models.onset_planner import OnsetPlanner


class TestOnsetPlanner:
    """Tests for OnsetPlanner module."""

    def test_output_shape(self):
        """Output shape should match input shape."""
        planner = OnsetPlanner(d_model=64, nhead=4, num_layers=2, dim_feedforward=128)
        x = torch.randn(2, 100, 64)
        out = planner(x)
        assert out.shape == (2, 100, 64)

    def test_single_onset(self):
        """Should work with a single onset."""
        planner = OnsetPlanner(d_model=64, nhead=4, num_layers=2, dim_feedforward=128)
        x = torch.randn(1, 1, 64)
        out = planner(x)
        assert out.shape == (1, 1, 64)

    def test_padding_mask(self):
        """Should accept a padding mask without error."""
        planner = OnsetPlanner(d_model=64, nhead=4, num_layers=2, dim_feedforward=128)
        x = torch.randn(2, 50, 64)
        mask = torch.zeros(2, 50, dtype=torch.bool)
        mask[0, 30:] = True  # first sample has 30 real onsets
        mask[1, 40:] = True  # second has 40
        out = planner(x, padding_mask=mask)
        assert out.shape == (2, 50, 64)

    def test_gradient_flow(self):
        """Gradients should flow through the planner."""
        planner = OnsetPlanner(d_model=64, nhead=4, num_layers=2, dim_feedforward=128)
        x = torch.randn(1, 20, 64, requires_grad=True)
        out = planner(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert x.grad.abs().sum() > 0

    def test_long_sequence(self):
        """Should handle songs with many onsets (e.g., 500+)."""
        planner = OnsetPlanner(d_model=64, nhead=4, num_layers=2, dim_feedforward=128)
        x = torch.randn(1, 500, 64)
        out = planner(x)
        assert out.shape == (1, 500, 64)

    def test_section_conditioning(self):
        """Should accept and use section_ids and section_progress."""
        planner = OnsetPlanner(d_model=64, nhead=4, num_layers=2, dim_feedforward=128)
        x = torch.randn(2, 50, 64)
        section_ids = torch.randint(0, 6, (2, 50))
        section_progress = torch.rand(2, 50)
        out = planner(x, section_ids=section_ids, section_progress=section_progress)
        assert out.shape == (2, 50, 64)

    def test_section_conditioning_gradient(self):
        """Section embeddings should receive gradients."""
        planner = OnsetPlanner(d_model=64, nhead=4, num_layers=2, dim_feedforward=128)
        x = torch.randn(1, 10, 64)
        section_ids = torch.randint(0, 6, (1, 10))
        section_progress = torch.rand(1, 10)
        out = planner(x, section_ids=section_ids, section_progress=section_progress)
        out.sum().backward()
        assert planner.section_emb.weight.grad is not None
        assert planner.progress_proj.weight.grad is not None
