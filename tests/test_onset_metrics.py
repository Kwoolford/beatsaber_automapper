"""Unit tests for tolerance-window onset F1.

Covers the per-sample 1D greedy matcher and the OnsetToleranceF1 torchmetric
that accumulates counts across the validation epoch.
"""

from __future__ import annotations

import pytest
import torch

from beatsaber_automapper.evaluation.onset_metrics import OnsetToleranceF1, _match_1d


# ----------------------------------------------------------------------------
# _match_1d  (sorted-list two-pointer matcher)
# ----------------------------------------------------------------------------

def test_match_empty_both() -> None:
    assert _match_1d([], [], 1) == (0, 0, 0)


def test_match_empty_preds() -> None:
    assert _match_1d([], [1, 3, 5], 1) == (0, 0, 3)


def test_match_empty_labels() -> None:
    assert _match_1d([0, 2], [], 1) == (0, 2, 0)


def test_match_exact_alignment() -> None:
    # tol=0 reduces to exact-slot matching
    assert _match_1d([1, 3, 5], [1, 3, 5], 0) == (3, 0, 0)


def test_match_off_by_one_within_tol() -> None:
    assert _match_1d([2], [1], 1) == (1, 0, 0)
    assert _match_1d([0], [1], 1) == (1, 0, 0)


def test_match_off_by_two_outside_tol() -> None:
    assert _match_1d([3], [1], 1) == (0, 1, 1)


def test_match_one_pred_cannot_match_two_labels() -> None:
    # Pred at slot 5, labels at 4 and 6; tolerance 1 lets either match
    # but each prediction is used at most once. Greedy left-to-right picks
    # the label-4 → pred-5 pair, leaving label-6 unmatched.
    tp, fp, fn = _match_1d([5], [4, 6], 1)
    assert (tp, fp, fn) == (1, 0, 1)


def test_match_walks_to_next_unused_pred() -> None:
    # Preds at 1, 3; labels at 2, 4; tolerance 1.
    # Greedy: label 2 picks pred 1 (within ±1), label 4 picks pred 3.
    tp, fp, fn = _match_1d([1, 3], [2, 4], 1)
    assert (tp, fp, fn) == (2, 0, 0)


def test_match_skips_too_far_preds() -> None:
    # Preds at 0 and 10; labels at 9. Only pred 10 is in tolerance.
    tp, fp, fn = _match_1d([0, 10], [9], 1)
    assert (tp, fp, fn) == (1, 1, 0)


def test_match_dense_preds_sparse_labels() -> None:
    # Three preds clustered around one label
    tp, fp, fn = _match_1d([4, 5, 6], [5], 1)
    assert tp == 1 and fn == 0
    assert fp == 2


# ----------------------------------------------------------------------------
# OnsetToleranceF1  (torchmetric wrapper)
# ----------------------------------------------------------------------------

def _probs_from_positions(positions: list[int], W: int = 16) -> torch.Tensor:
    """Build a [1, W] probs tensor with 1.0 at the given positions, 0.0 elsewhere."""
    t = torch.zeros(1, W)
    for p in positions:
        t[0, p] = 1.0
    return t


def _labels_from_positions(positions: list[int], W: int = 16) -> torch.Tensor:
    t = torch.zeros(1, W, dtype=torch.long)
    for p in positions:
        t[0, p] = 1
    return t


def test_metric_perfect_match() -> None:
    m = OnsetToleranceF1(threshold=0.5, tolerance=1)
    probs = _probs_from_positions([2, 5, 9])
    labels = _labels_from_positions([2, 5, 9])
    m.update(probs, labels)
    assert m.compute().item() == pytest.approx(1.0)


def test_metric_off_by_one_is_tp_with_tolerance() -> None:
    m = OnsetToleranceF1(threshold=0.5, tolerance=1)
    m.update(_probs_from_positions([2, 5, 9]), _labels_from_positions([3, 4, 10]))
    # Each prediction is exactly one slot off the corresponding label → all TP.
    assert m.compute().item() == pytest.approx(1.0)


def test_metric_off_by_one_is_miss_at_zero_tolerance() -> None:
    m = OnsetToleranceF1(threshold=0.5, tolerance=0)
    m.update(_probs_from_positions([2]), _labels_from_positions([3]))
    # tp=0, fp=1, fn=1 → F1=0
    assert m.compute().item() == pytest.approx(0.0)


def test_metric_threshold_filters_low_prob() -> None:
    m = OnsetToleranceF1(threshold=0.5, tolerance=1)
    probs = torch.zeros(1, 8)
    probs[0, 3] = 0.4  # below threshold → ignored
    probs[0, 5] = 0.9  # above threshold → counted
    labels = _labels_from_positions([5], W=8)
    m.update(probs, labels)
    assert m.compute().item() == pytest.approx(1.0)


def test_metric_accumulates_across_updates() -> None:
    m = OnsetToleranceF1(threshold=0.5, tolerance=1)
    # First batch: 2 TP, 0 FP, 0 FN
    m.update(_probs_from_positions([1, 5]), _labels_from_positions([1, 5]))
    # Second batch: 0 TP, 2 FP, 2 FN (predictions far from labels)
    m.update(_probs_from_positions([0, 1]), _labels_from_positions([10, 12]))
    # Totals: tp=2, fp=2, fn=2 → F1 = 2*2 / (2*2 + 2 + 2) = 4/8 = 0.5
    assert m.compute().item() == pytest.approx(0.5)


def test_metric_empty_state_returns_zero() -> None:
    m = OnsetToleranceF1(threshold=0.5, tolerance=1)
    # No updates → tp=fp=fn=0 → denom=0 → guarded to 0.0
    assert m.compute().item() == pytest.approx(0.0)


def test_metric_batched_samples_independent() -> None:
    """A single batch with two samples should match per-sample, not pool positions."""
    m = OnsetToleranceF1(threshold=0.5, tolerance=1)
    probs = torch.zeros(2, 10)
    labels = torch.zeros(2, 10, dtype=torch.long)
    # Sample 0: pred at 3, label at 3 → TP
    probs[0, 3] = 1.0
    labels[0, 3] = 1
    # Sample 1: pred at 7, label at 0 → FP + FN (cross-sample shouldn't match)
    probs[1, 7] = 1.0
    labels[1, 0] = 1
    m.update(probs, labels)
    # tp=1, fp=1, fn=1 → F1 = 2/(2+1+1) = 0.5
    assert m.compute().item() == pytest.approx(0.5)
