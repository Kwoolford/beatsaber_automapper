#!/usr/bin/env python
"""Does Stage-1's probability know where the music actually is?

The question this settles (2026-08-02). Fixing the tempo grid took alignment from
5.41 to 0.49 (bar 0.39) — it closed ~89% of the gap and stopped short. The whole
remainder is `onset_precision`: 0.902 against a human 0.930, with timing scatter
already better than human. So: are we placing notes on slots where nothing
happens, and if so, is that the SELECTOR's fault or the MODEL's?

The selector is not the suspect. `_density_aware_select` with `BEAT_IOI_PRIOR=0`
(the default) takes `np.argsort(-p)` — greedy top-k by Stage-1 probability. And
cutting the note budget hard (nps 5.30 → 4.32) left precision flat. If keeping
only the most confident slots does not raise the share of notes that land on real
audio events, then **the probability ordering does not track onset-ness**, and the
residual gap is a representation problem rather than a decode one.

That is exactly the Track B thesis from 2026-07-27 — Stage-1 `version_4` has only
`drum_proj` and `mix_proj` and literally cannot hear the guitar — so this script
either supports it with a direct measurement or rules it out.

METHOD. `BEAT_PROBS_DUMP` writes the per-slot probabilities. Each slot has a time
(slot / subdiv * 60 / bpm), so every slot can be labelled by its distance to the
nearest DETECTED onset (the same stem-union onsets axis A8 scores against). Then:

  * AUROC of probability as a detector of "this slot is on an onset". 0.5 means
    the probability carries no information about where the music is; ~0.9 would
    mean the model knows and the decode is throwing it away.
  * precision within each probability decile — if the top decile is not markedly
    cleaner than the median, greedy selection cannot help and no threshold will.

Usage:
  BEAT_PROBS_DUMP=/tmp/p.npz python scripts/generate.py <audio> --v7 ...
  python scripts/eval_prob_vs_onset.py --dump /tmp/p.npz --song 1f767
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

ONSETS = REPO / "outputs" / "onset_cache"
TOL_S = 0.05  # same window A8 calls "on the sound"


def slot_times(n_slots: int, bpm: float, subdiv: int) -> np.ndarray:
    return np.arange(n_slots, dtype=np.float64) * (60.0 / bpm / subdiv)


def on_onset(times: np.ndarray, onsets: np.ndarray, tol: float = TOL_S) -> np.ndarray:
    if len(onsets) == 0:
        return np.zeros(len(times), bool)
    ref = np.sort(onsets)
    i = np.searchsorted(ref, times).clip(1, len(ref) - 1)
    d = np.minimum(np.abs(times - ref[i - 1]), np.abs(times - ref[i]))
    return d <= tol


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Rank-based AUROC; 0.5 = the score knows nothing about the label."""
    pos, neg = scores[labels], scores[~labels]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(np.concatenate([pos, neg]))
    ranks = np.empty(len(order), float)
    ranks[order] = np.arange(1, len(order) + 1)
    return float((ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2)
                 / (len(pos) * len(neg)))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dump", required=True, help="npz from BEAT_PROBS_DUMP")
    ap.add_argument("--song", required=True, help="song id, for the onset cache")
    a = ap.parse_args()

    d = np.load(a.dump, allow_pickle=False)
    probs, bpm = d["beat_probs"], float(d["bpm"])
    subdiv, n_slots = int(d["beat_subdiv"]), int(d["n_slots"])
    f = ONSETS / f"{a.song}.npz"
    if not f.exists():
        print(f"no cached onsets for {a.song} — run scripts/build_onset_cache.py")
        raise SystemExit(2)
    onsets = np.load(f, allow_pickle=False)["onsets"]

    t = slot_times(n_slots, bpm, subdiv)
    lab = on_onset(t, onsets)
    print(f"song {a.song}: {n_slots} slots at {bpm:.2f} bpm (1/{subdiv} beat = "
          f"{60.0 / bpm / subdiv * 1000:.1f} ms), {len(onsets)} onsets")
    print(f"slots that sit on a real onset: {lab.mean():.3f}\n")

    print(f"{'hand':8s}{'AUROC':>8s}{'top-decile prec':>18s}{'median-decile':>15s}"
          f"{'lift':>8s}")
    print("-" * 58)
    for hand, col in (("left", 0), ("right", 1)):
        p = probs[:min(len(probs), n_slots), col]
        n = min(len(p), len(lab))
        p, l = p[:n], lab[:n]
        au = auroc(p, l)
        order = np.argsort(-p)
        dec = max(len(order) // 10, 1)
        top = l[order[:dec]].mean()
        mid = l[order[len(order) // 2 - dec // 2: len(order) // 2 + dec // 2]].mean()
        print(f"{hand:8s}{au:8.3f}{top:18.3f}{mid:15.3f}{top / max(mid, 1e-9):8.2f}x")

    print("\nREAD:")
    print("  AUROC ~0.5      -> the probability carries NO information about where")
    print("                     the music is. No threshold, budget or selector can")
    print("                     fix precision; the gap is the Stage-1 representation")
    print("                     (Track B: version_4 has no instrument projection).")
    print("  AUROC >=0.8     -> the model knows and the DECODE is discarding it;")
    print("                     the fix is cheap and lives in selection.")
    print("  top-decile lift -> if the most confident slots are not markedly cleaner")
    print("                     than typical ones, greedy top-k selection cannot")
    print("                     raise precision, which is what the density re-tune")
    print("                     observed (nps 5.30 -> 4.32, precision flat).")


if __name__ == "__main__":
    main()
