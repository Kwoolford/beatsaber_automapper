#!/usr/bin/env python3
"""Oracle-ceiling PoC (2026-06-29) for the density_corr DoD.

Non-circular question: how much density structure is LATENT in Stage-1's own
per-slot onset probabilities, BEFORE thresholding/NMS/density-curve flatten it?

We dump raw `beat_probs` [N,2] from the v7 beat classifier (env BEAT_PROBS_DUMP in
generation/generate.py), bin the continuous prob-mass into the SAME 2s windows the
DoD uses, and Spearman-correlate against the SAME reference (librosa onsets on
drums∪other). This is the ceiling a smarter *selection* over the existing model
could reach — if it clears 0.41 the signal is in the probs (stop flattening in post,
maybe no retrain); if it's ~0 the probs themselves are flat → conditioning retrain.
"""
from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np

from eval_alignment import _separate_stems, _detect_onsets_librosa  # scripts/ dir
from eval_density_corr import _bin_counts, _spearman, _pearson


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--audio", type=pathlib.Path, required=True)
    p.add_argument("--probs", type=pathlib.Path, required=True, help="beat_probs.npz dump")
    p.add_argument("--window-sec", type=float, default=2.0)
    p.add_argument("--sr", type=int, default=44100)
    p.add_argument("--json", type=pathlib.Path, default=None)
    args = p.parse_args()

    d = np.load(args.probs)
    beat_probs = d["beat_probs"]          # [N, 2] (left, right) sigmoid probs
    bpm = float(d["bpm"])
    subdiv = int(d["beat_subdiv"])
    n_slots = beat_probs.shape[0]

    # slot i -> time = (i / subdiv) beats * (60/bpm) sec/beat
    sec_per_slot = (60.0 / bpm) / subdiv
    slot_times = np.arange(n_slots) * sec_per_slot

    # prob that SOME note (L or R) is present at the slot
    pL, pR = beat_probs[:, 0], beat_probs[:, 1]
    prob_any = 1.0 - (1.0 - pL) * (1.0 - pR)

    # reference: identical to eval_density_corr
    stems = _separate_stems(args.audio, args.sr)
    drum_on = _detect_onsets_librosa(stems.get("drums", np.zeros(1)), args.sr)
    other_on = _detect_onsets_librosa(stems.get("other", np.zeros(1)), args.sr)
    ref_times = np.union1d(drum_on, other_on)

    duration = float(max(slot_times.max() if n_slots else 0.0,
                         ref_times.max() if len(ref_times) else 0.0))
    win = args.window_sec
    n_bins = max(1, int(np.ceil(duration / win)))
    edges = np.arange(n_bins + 1) * win

    # continuous prob-mass per window (sum of prob_any), plus mean & max variants
    bin_idx = np.clip((slot_times / win).astype(int), 0, n_bins - 1)
    probmass = np.zeros(n_bins)
    probmean_sum = np.zeros(n_bins)
    probmean_cnt = np.zeros(n_bins)
    probmax = np.zeros(n_bins)
    for i, b in enumerate(bin_idx):
        probmass[b] += prob_any[i]
        probmean_sum[b] += prob_any[i]
        probmean_cnt[b] += 1
        probmax[b] = max(probmax[b], prob_any[i])
    probmean = probmean_sum / np.clip(probmean_cnt, 1, None)

    ref_d = _bin_counts(ref_times, duration, win)
    n = min(len(probmass), len(ref_d))
    ref_d = ref_d[:n]

    out = {}
    for name, vec in (("probmass", probmass), ("probmean", probmean), ("probmax", probmax)):
        v = vec[:n]
        out[name] = {
            "spearman": _spearman(v, ref_d),
            "pearson": _pearson(v, ref_d),
            "cv": float(v.std() / v.mean()) if v.mean() else 0.0,
        }

    result = {
        "audio": str(args.audio),
        "probs": str(args.probs),
        "bpm": bpm,
        "n_slots": n_slots,
        "n_windows": int(n),
        "n_reference_onsets": int(len(ref_times)),
        "prob_any_mean": float(prob_any.mean()),
        "prob_any_cv": float(prob_any.std() / prob_any.mean()) if prob_any.mean() else 0.0,
        "variants": out,
        "dod_pass_any": bool(max(out[k]["spearman"] for k in out) >= 0.41),
    }
    if args.json:
        args.json.write_text(json.dumps(result, indent=2))

    print(f"\n=== ORACLE density ceiling  ({n} windows, {len(ref_times)} ref onsets) ===")
    print(f"  raw prob_any: mean={result['prob_any_mean']:.3f}  CV={result['prob_any_cv']:.3f}")
    for k, v in out.items():
        flag = "PASS" if v["spearman"] >= 0.41 else "fail"
        print(f"  {k:9s}  Spearman={v['spearman']:+.4f}  Pearson={v['pearson']:+.4f}  CV={v['cv']:.3f}  [{flag}]")
    print(f"  DoD(>=0.41) reachable by best variant = {result['dod_pass_any']}")


if __name__ == "__main__":
    main()
