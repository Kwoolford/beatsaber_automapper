"""Tolerance-window onset F1.

MIR-standard evaluation for onset / beat detection: a prediction at slot t
matches a label at any slot in [t-K, t+K]. Exact-slot F1 (BinaryF1Score)
double-counts off-by-one errors and is much harsher than human onset
perception — at subdiv=4, BPM=120, one slot is ~125 ms.

Matching is greedy 1:1 in sorted order on each sample, which is optimal for
1D interval matching. State accumulates across the validation epoch.
"""

from __future__ import annotations

import torch
from torchmetrics import Metric


def _match_1d(pred_idx: list[int], label_idx: list[int], tol: int) -> tuple[int, int, int]:
    """Return (tp, fp, fn) for a single sample given sorted positive positions.

    Greedy two-pointer match: walk labels left-to-right, match each to the
    leftmost still-unused prediction inside [label-tol, label+tol]. Each
    prediction matches at most one label; each label matches at most one
    prediction. Optimal for 1D point-to-point matching under a symmetric
    tolerance, since matching the leftmost feasible prediction never blocks
    a later label that an earlier (already-passed) prediction could have
    served.
    """
    n_p = len(pred_idx)
    n_l = len(label_idx)
    if n_p == 0:
        return 0, 0, n_l
    if n_l == 0:
        return 0, n_p, 0

    tp = 0
    used = [False] * n_p
    j_lo = 0  # leftmost pred index that could still be in tolerance for any future label

    for lab in label_idx:
        while j_lo < n_p and pred_idx[j_lo] < lab - tol:
            j_lo += 1
        k = j_lo
        while k < n_p and pred_idx[k] <= lab + tol:
            if not used[k]:
                used[k] = True
                tp += 1
                break
            k += 1
    return tp, n_p - tp, n_l - tp


class OnsetToleranceF1(Metric):
    """F1 with ±tolerance-slot match window between predicted and label positives.

    Args:
        threshold:    Decision threshold applied to sigmoid probabilities.
        tolerance:    Slot radius for matching. 0 = exact-slot (equivalent to
                      BinaryF1Score); 1 = ±125 ms at BPM=120 subdiv=4.
    """

    is_differentiable: bool = False
    higher_is_better: bool = True
    full_state_update: bool = False

    def __init__(self, threshold: float = 0.5, tolerance: int = 1) -> None:
        super().__init__()
        self.threshold = threshold
        self.tolerance = tolerance
        self.add_state("tp", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("fp", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("fn", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, probs: torch.Tensor, labels: torch.Tensor) -> None:
        """Accumulate counts.

        Args:
            probs:  [B, W] sigmoid probabilities for one hand.
            labels: [B, W] binary {0,1} labels for the same hand.
        """
        if probs.ndim != 2 or labels.ndim != 2:
            raise ValueError(f"probs/labels must be [B, W], got {probs.shape} / {labels.shape}")

        preds_bin = (probs >= self.threshold).to(torch.bool)
        labels_bin = labels.to(torch.bool)

        # Move to CPU once for the per-sample matching loop.
        preds_cpu = preds_bin.detach().cpu()
        labels_cpu = labels_bin.detach().cpu()

        tp = fp = fn = 0
        B = preds_cpu.shape[0]
        for b in range(B):
            p_idx = torch.nonzero(preds_cpu[b], as_tuple=False).flatten().tolist()
            l_idx = torch.nonzero(labels_cpu[b], as_tuple=False).flatten().tolist()
            tp_b, fp_b, fn_b = _match_1d(p_idx, l_idx, self.tolerance)
            tp += tp_b
            fp += fp_b
            fn += fn_b

        self.tp += torch.tensor(tp, device=self.tp.device)
        self.fp += torch.tensor(fp, device=self.fp.device)
        self.fn += torch.tensor(fn, device=self.fn.device)

    def compute(self) -> torch.Tensor:
        tp = self.tp.float()
        fp = self.fp.float()
        fn = self.fn.float()
        denom = 2 * tp + fp + fn
        if denom.item() == 0:
            return torch.tensor(0.0, device=tp.device)
        return 2 * tp / denom
