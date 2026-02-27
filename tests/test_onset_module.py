"""Tests for OnsetLitModule, peak_picking, and onset_f1."""

import torch

from beatsaber_automapper.evaluation.metrics import onset_f1, onset_f1_framewise
from beatsaber_automapper.models.components import peak_picking
from beatsaber_automapper.training.onset_module import OnsetLitModule


class TestPeakPicking:
    """Tests for the peak_picking utility."""

    def test_basic_peaks(self):
        """Should find clear peaks above threshold."""
        probs = torch.tensor([0.0, 0.1, 0.8, 0.1, 0.0, 0.1, 0.9, 0.1, 0.0])
        peaks = peak_picking(probs, threshold=0.5, min_distance=1)
        assert peaks.tolist() == [2, 6]

    def test_min_distance_suppression(self):
        """Should suppress nearby peaks, keeping the higher one."""
        probs = torch.tensor([0.0, 0.7, 0.8, 0.6, 0.0])
        peaks = peak_picking(probs, threshold=0.5, min_distance=3)
        assert peaks.tolist() == [2]

    def test_no_peaks_below_threshold(self):
        """Should return empty when all values are below threshold."""
        probs = torch.tensor([0.1, 0.2, 0.3, 0.2, 0.1])
        peaks = peak_picking(probs, threshold=0.5, min_distance=1)
        assert len(peaks) == 0

    def test_empty_input(self):
        """Should handle all-zero input."""
        probs = torch.zeros(10)
        peaks = peak_picking(probs, threshold=0.5, min_distance=1)
        assert len(peaks) == 0

    def test_single_peak(self):
        """Should find a single peak."""
        probs = torch.tensor([0.0, 0.0, 0.9, 0.0, 0.0])
        peaks = peak_picking(probs, threshold=0.5, min_distance=1)
        assert peaks.tolist() == [2]


class TestOnsetF1:
    """Tests for onset_f1 metric."""

    def test_perfect_match(self):
        """Perfect predictions should give F1=1.0."""
        result = onset_f1([1.0, 2.0, 3.0], [1.0, 2.0, 3.0], tolerance=0.05)
        assert result["f1"] == 1.0
        assert result["precision"] == 1.0
        assert result["recall"] == 1.0

    def test_no_predictions(self):
        """No predictions with true onsets should give F1=0."""
        result = onset_f1([], [1.0, 2.0], tolerance=0.05)
        assert result["f1"] == 0.0
        assert result["recall"] == 0.0

    def test_no_true_onsets(self):
        """Predictions with no true onsets should give F1=0."""
        result = onset_f1([1.0, 2.0], [], tolerance=0.05)
        assert result["f1"] == 0.0
        assert result["precision"] == 0.0

    def test_within_tolerance(self):
        """Predictions within tolerance should match."""
        result = onset_f1([1.02, 2.03], [1.0, 2.0], tolerance=0.05)
        assert result["f1"] == 1.0

    def test_outside_tolerance(self):
        """Predictions outside tolerance should not match."""
        result = onset_f1([1.1, 2.1], [1.0, 2.0], tolerance=0.05)
        assert result["f1"] == 0.0

    def test_both_empty(self):
        """Both empty should give F1=1.0."""
        result = onset_f1([], [], tolerance=0.05)
        assert result["f1"] == 1.0


class TestOnsetF1Framewise:
    """Tests for onset_f1_framewise."""

    def test_framewise_basic(self):
        """Should work with frame index tensors."""
        pred = torch.tensor([10, 20, 30])
        true = torch.tensor([10, 20, 30])
        result = onset_f1_framewise(pred, true, tolerance_frames=3)
        assert result["f1"] == 1.0

    def test_framewise_empty(self):
        """Should handle empty tensors."""
        pred = torch.tensor([])
        true = torch.tensor([10, 20])
        result = onset_f1_framewise(pred, true, tolerance_frames=3)
        assert result["f1"] == 0.0


class TestOnsetLitModule:
    """Tests for OnsetLitModule."""

    def _make_small_module(self) -> OnsetLitModule:
        """Create a small module for testing."""
        return OnsetLitModule(
            n_mels=16,
            encoder_d_model=64,
            encoder_nhead=4,
            encoder_num_layers=1,
            encoder_dim_feedforward=128,
            onset_d_model=64,
            onset_nhead=4,
            onset_num_layers=1,
            tcn_channels=32,
            tcn_num_blocks=2,
            tcn_kernel_size=3,
        )

    def test_forward_shape(self):
        """Forward pass should produce [B, T] logits."""
        module = self._make_small_module()
        mel = torch.randn(2, 16, 32)
        diff = torch.tensor([0, 3])
        genre = torch.tensor([0, 1])
        out = module(mel, diff, genre)
        assert out.shape == (2, 32)

    def test_training_step_returns_scalar(self):
        """training_step should return a scalar loss."""
        module = self._make_small_module()
        batch = {
            "mel": torch.randn(2, 16, 32),
            "labels": torch.zeros(2, 32),
            "difficulty": torch.tensor([0, 1]),
            "genre": torch.tensor([0, 2]),
        }
        loss = module.training_step(batch, 0)
        assert loss.ndim == 0
        assert loss.requires_grad
