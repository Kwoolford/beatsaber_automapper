"""Quantitative evaluation metrics for generated Beat Saber maps.

Metrics:
    - Onset F1: Precision/recall of predicted onsets vs ground truth.
    - Token accuracy: Accuracy of generated token sequences (PR 4).
    - Pattern diversity: Measures variety in generated note patterns (PR 7).
"""

from __future__ import annotations

import logging

import torch

logger = logging.getLogger(__name__)


def onset_f1(
    predicted_onsets: list[float],
    true_onsets: list[float],
    tolerance: float = 0.05,
) -> dict[str, float]:
    """Compute onset detection F1 score using time-based matching.

    Uses greedy matching: each predicted onset is matched to the nearest
    unmatched true onset within the tolerance window (standard mir_eval approach).

    Args:
        predicted_onsets: List of predicted onset times in seconds.
        true_onsets: List of ground truth onset times in seconds.
        tolerance: Matching tolerance in seconds.

    Returns:
        Dictionary with precision, recall, and F1 score.
    """
    if not predicted_onsets or not true_onsets:
        return {
            "precision": 0.0 if predicted_onsets else 1.0,
            "recall": 0.0 if true_onsets else 1.0,
            "f1": 0.0 if (predicted_onsets or true_onsets) else 1.0,
        }

    pred_sorted = sorted(predicted_onsets)
    true_sorted = sorted(true_onsets)
    matched_true: set[int] = set()
    tp = 0

    for p in pred_sorted:
        best_dist = float("inf")
        best_idx = -1
        for i, t in enumerate(true_sorted):
            if i in matched_true:
                continue
            dist = abs(p - t)
            if dist < best_dist:
                best_dist = dist
                best_idx = i
        if best_dist <= tolerance and best_idx >= 0:
            tp += 1
            matched_true.add(best_idx)

    precision = tp / len(pred_sorted) if pred_sorted else 0.0
    recall = tp / len(true_sorted) if true_sorted else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {"precision": precision, "recall": recall, "f1": f1}


def onset_f1_framewise(
    pred_frames: torch.Tensor,
    true_frames: torch.Tensor,
    tolerance_frames: int = 3,
) -> dict[str, float]:
    """Compute onset F1 score using frame indices.

    Convenience wrapper for use in validation loops where onsets are
    represented as frame index tensors rather than times in seconds.

    Args:
        pred_frames: 1-D tensor of predicted onset frame indices.
        true_frames: 1-D tensor of ground truth onset frame indices.
        tolerance_frames: Matching tolerance in frames.

    Returns:
        Dictionary with precision, recall, and F1 score.
    """
    pred_list = pred_frames.cpu().tolist() if len(pred_frames) > 0 else []
    true_list = true_frames.cpu().tolist() if len(true_frames) > 0 else []
    return onset_f1(pred_list, true_list, tolerance=float(tolerance_frames))


def token_accuracy(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    ignore_index: int = 0,
) -> float:
    """Compute per-token accuracy ignoring a padding index.

    Args:
        predictions: Predicted token indices [B, S] or [S].
        targets: Ground truth token indices [B, S] or [S].
        ignore_index: Token index to ignore (default: PAD=0).

    Returns:
        Accuracy as a float in [0, 1]. Returns 0.0 if no valid tokens.
    """
    mask = targets != ignore_index
    if mask.sum() == 0:
        return 0.0
    correct = (predictions == targets) & mask
    return (correct.sum().float() / mask.sum().float()).item()
