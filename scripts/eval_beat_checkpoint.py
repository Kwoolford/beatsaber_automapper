"""V7-3 Run 3 re-evaluation: calibration, audio-support, density correlation.

F1 against a single mapper's labels conflates "wrong" with "different but valid"
on an inherently subjective task. This script computes three orthogonal
diagnostics on the validation split using only data we already have — no
retraining, no extra preprocessing.

Outputs are written to logs/beat_eval/<run-name>/ as JSON + a printed summary.

Usage:
    python scripts/eval_beat_checkpoint.py \
        --checkpoint logs/beat_classifier/version_3/checkpoints/beat-epoch=21-val_f1_avg_tol=0.588.ckpt \
        --run-name run3
"""

from __future__ import annotations

import argparse
import json
import logging
import pathlib
import sys
from collections import defaultdict

import numpy as np
import torch

REPO_ROOT = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("eval_beat")


# ----------------------------------------------------------------------------
# Per-song full-song inference (window the song to match training context)
# ----------------------------------------------------------------------------

def _predict_song(
    model,
    drum: torch.Tensor,         # [N, 768]
    mix:  torch.Tensor,         # [N, 768]
    difficulty: int,
    window_size: int,
    device: torch.device,
) -> np.ndarray:
    """Return sigmoid probs [N, 2] for the whole song.

    Non-overlapping windows of `window_size`, tail-padded with zeros.
    `slot_offset` is fed per window so the phase embedding stays aligned
    with the absolute bar grid.
    """
    N = drum.shape[0]
    probs = np.zeros((N, 2), dtype=np.float32)
    for start in range(0, N, window_size):
        end = min(start + window_size, N)
        w = end - start
        d = torch.zeros(1, window_size, 768, dtype=torch.float32, device=device)
        m = torch.zeros(1, window_size, 768, dtype=torch.float32, device=device)
        d[0, :w] = drum[start:end].float().to(device)
        m[0, :w] = mix[start:end].float().to(device)
        diff_t = torch.tensor([difficulty], dtype=torch.long, device=device)
        with torch.no_grad():
            logits = model(d, m, diff_t, slot_offset=start)   # [1, W, 2]
        p = torch.sigmoid(logits).squeeze(0).cpu().numpy()
        probs[start:end] = p[:w]
    return probs


# ----------------------------------------------------------------------------
# Diagnostics
# ----------------------------------------------------------------------------

def _calibration_bins(
    probs: np.ndarray, labels: np.ndarray, n_bins: int = 10,
) -> dict:
    """Return per-bin (mean_pred, fraction_pos, count). Also ECE."""
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_id = np.clip(np.digitize(probs, edges) - 1, 0, n_bins - 1)
    n = len(probs)
    rows = []
    ece = 0.0
    for b in range(n_bins):
        mask = bin_id == b
        cnt = int(mask.sum())
        if cnt == 0:
            rows.append({"bin": b, "lo": float(edges[b]), "hi": float(edges[b+1]),
                         "count": 0, "mean_pred": None, "frac_pos": None})
            continue
        mp = float(probs[mask].mean())
        fp = float(labels[mask].mean())
        ece += (cnt / n) * abs(mp - fp)
        rows.append({"bin": b, "lo": float(edges[b]), "hi": float(edges[b+1]),
                     "count": cnt, "mean_pred": mp, "frac_pos": fp})
    return {"bins": rows, "ece": float(ece), "n_total": int(n)}


def _onset_strength_per_slot(mel: np.ndarray, n_slots: int) -> np.ndarray:
    """Spectral-flux onset strength pooled to the beat grid.

    Spectral flux (sum of positive frame-to-frame log-mel differences) is a
    standard MIR onset detector. We compute it on the mixture mel, then
    max-pool across the ~10 mel frames per beat slot so each slot reflects
    the *strongest* transient in its window — what a mapper would respond to.
    """
    log_mel = np.log1p(np.maximum(mel, 0))
    diff = np.diff(log_mel, axis=1, prepend=log_mel[:, :1])
    flux = np.maximum(diff, 0).sum(axis=0)  # [T_frames]
    T = mel.shape[1]
    out = np.zeros(n_slots, dtype=np.float32)
    for s in range(n_slots):
        a = int(s * T / n_slots)
        b = int((s + 1) * T / n_slots)
        if b > a:
            out[s] = flux[a:b].max()
    return out


def _percentile_rank(x: np.ndarray) -> np.ndarray:
    """Per-element percentile rank within `x`, range [0, 1]."""
    order = np.argsort(x)
    rank = np.zeros_like(x, dtype=np.float64)
    rank[order] = np.arange(len(x)) / max(1, len(x) - 1)
    return rank


def _audio_support(
    probs: np.ndarray,            # [N, 2]
    onset_strength: np.ndarray,   # [N]  proper spectral-flux proxy
    labels_l: np.ndarray,         # [N]
    labels_r: np.ndarray,         # [N]
    threshold: float = 0.5,
) -> dict:
    """For predicted positives AND label positives, report the median
    onset-strength percentile rank within the song. Random baseline = 0.5;
    label baseline tells us where mappers actually place notes (so we can
    judge the model relative to the mapper rather than a uniform prior).
    """
    if onset_strength.std() == 0:
        return {k: None for k in
                ("median_pct_pred_left", "median_pct_pred_right",
                 "median_pct_label_left", "median_pct_label_right",
                 "frac_top30_pred_left", "frac_top30_pred_right",
                 "frac_top30_label_left", "frac_top30_label_right",
                 "n_pred_left", "n_pred_right", "n_label_left", "n_label_right")}

    rank = _percentile_rank(onset_strength)
    out: dict = {}

    def _summarize(mask: np.ndarray, prefix: str) -> None:
        n = int(mask.sum())
        out[f"n_{prefix}"] = n
        if n == 0:
            out[f"median_pct_{prefix}"] = None
            out[f"frac_top30_{prefix}"] = None
            return
        r = rank[mask]
        out[f"median_pct_{prefix}"] = float(np.median(r))
        out[f"frac_top30_{prefix}"] = float((r >= 0.7).mean())

    _summarize(probs[:, 0] >= threshold, "pred_left")
    _summarize(probs[:, 1] >= threshold, "pred_right")
    _summarize(labels_l.astype(bool),    "label_left")
    _summarize(labels_r.astype(bool),    "label_right")
    return out


def _density_correlation(
    probs: np.ndarray,           # [N, 2]
    labels_l: np.ndarray,        # [N]
    labels_r: np.ndarray,        # [N]
    onset_strength: np.ndarray,  # [N]
    phrase_boundaries: list,
    threshold: float = 0.5,
) -> list[dict] | None:
    """Per-phrase: (predicted count, label count, mean onset strength)."""
    if not phrase_boundaries:
        return None
    rows = []
    for span in phrase_boundaries:
        a, b = int(span[0]), int(span[1])
        b = min(b, len(onset_strength))
        if b <= a:
            continue
        seg_pred = ((probs[a:b, 0] >= threshold).sum() +
                    (probs[a:b, 1] >= threshold).sum())
        seg_lbl  = int(labels_l[a:b].sum() + labels_r[a:b].sum())
        seg_eng  = float(onset_strength[a:b].mean())
        rows.append({"start": a, "end": b, "pred_count": int(seg_pred),
                     "label_count": seg_lbl, "mean_onset": seg_eng})
    return rows


def _spearman(x: np.ndarray, y: np.ndarray) -> float | None:
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return None
    rx = np.argsort(np.argsort(x))
    ry = np.argsort(np.argsort(y))
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    return float((rx * ry).sum() / (np.sqrt((rx**2).sum()) * np.sqrt((ry**2).sum())))


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Re-evaluate Run 3 checkpoint")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-dir",   default="data/processed")
    parser.add_argument("--run-name",   default="run3")
    parser.add_argument("--difficulties", nargs="+", default=["Expert", "ExpertPlus"])
    parser.add_argument("--threshold",  type=float, default=0.5)
    parser.add_argument("--window-size", type=int, default=128)
    parser.add_argument("--max-songs",  type=int, default=0,
                        help="Limit val songs for a quick smoke run (0 = all)")
    args = parser.parse_args()

    from beatsaber_automapper.data.beat_grid import extract_beat_labels, BEAT_SUBDIV
    from beatsaber_automapper.data.dataset import DIFFICULTY_MAP
    from beatsaber_automapper.training.beat_module import BeatLitModule

    data_dir = REPO_ROOT / args.data_dir
    out_dir  = REPO_ROOT / "logs" / "beat_eval" / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info("Loading checkpoint %s", args.checkpoint)
    module = BeatLitModule.load_from_checkpoint(args.checkpoint)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    module = module.to(device).eval()

    splits = json.load(open(data_dir / "splits.json"))
    val_ids = splits["val"]
    if args.max_songs:
        val_ids = val_ids[:args.max_songs]
    log.info("Evaluating on %d val songs, difficulties %s", len(val_ids), args.difficulties)

    # Accumulators
    all_probs_l, all_probs_r = [], []
    all_labels_l, all_labels_r = [], []
    per_song_support: list[dict] = []
    per_phrase_rows: list[dict] = []   # rows from all songs

    n_used = n_skipped = 0
    for sid in val_ids:
        pt = data_dir / f"{sid}.pt"
        if not pt.exists():
            n_skipped += 1
            continue
        try:
            d = torch.load(pt, weights_only=False, mmap=True)
        except Exception:
            n_skipped += 1
            continue
        if "drum_beat_features" not in d or "mix_beat_features" not in d:
            n_skipped += 1
            continue

        drum = d["drum_beat_features"]
        mix  = d["mix_beat_features"]
        N    = drum.shape[0]
        bpm  = float(d.get("bpm", 120.0))

        mel = d.get("mel_spectrogram")
        if mel is None:
            n_skipped += 1
            continue
        onset_strength = _onset_strength_per_slot(mel.numpy(), N)

        diffs = d.get("difficulties", {}) or {}
        phrase_b = list(d.get("phrase_boundaries", []) or [])

        for diff_name in args.difficulties:
            dd = diffs.get(diff_name)
            if not dd or not dd.get("swing_tokens"):
                continue
            diff_id = DIFFICULTY_MAP.get(diff_name, 3)

            probs = _predict_song(module, drum, mix, diff_id, args.window_size, device)
            left_lbl, right_lbl, _, _ = extract_beat_labels(dd["swing_tokens"], bpm, N)
            left_lbl  = left_lbl.astype(np.int64)
            right_lbl = right_lbl.astype(np.int64)

            all_probs_l.append(probs[:, 0]);  all_labels_l.append(left_lbl)
            all_probs_r.append(probs[:, 1]);  all_labels_r.append(right_lbl)

            sup = _audio_support(probs, onset_strength,
                                 left_lbl, right_lbl, args.threshold)
            sup["song_id"] = sid
            sup["difficulty"] = diff_name
            per_song_support.append(sup)

            rows = _density_correlation(probs, left_lbl, right_lbl,
                                        onset_strength, phrase_b, args.threshold)
            if rows:
                for r in rows:
                    r["song_id"] = sid
                    r["difficulty"] = diff_name
                    per_phrase_rows.append(r)

            n_used += 1

    log.info("Used %d (song, difficulty) pairs; skipped %d songs", n_used, n_skipped)

    # ---- Aggregate ----
    pl = np.concatenate(all_probs_l)
    pr = np.concatenate(all_probs_r)
    ll = np.concatenate(all_labels_l)
    lr = np.concatenate(all_labels_r)

    # Calibration (per-hand and pooled)
    calib_l = _calibration_bins(pl, ll)
    calib_r = _calibration_bins(pr, lr)
    calib_p = _calibration_bins(np.concatenate([pl, pr]),
                                np.concatenate([ll, lr]))

    # Audio support: median across songs (per-hand, for predicted AND labels)
    def _med(key):
        vals = [s[key] for s in per_song_support if s.get(key) is not None]
        return float(np.median(vals)) if vals else None
    audio_summary = {
        "median_pct_pred_left":   _med("median_pct_pred_left"),
        "median_pct_pred_right":  _med("median_pct_pred_right"),
        "median_pct_label_left":  _med("median_pct_label_left"),
        "median_pct_label_right": _med("median_pct_label_right"),
        "top30_pred_left":        _med("frac_top30_pred_left"),
        "top30_pred_right":       _med("frac_top30_pred_right"),
        "top30_label_left":       _med("frac_top30_label_left"),
        "top30_label_right":      _med("frac_top30_label_right"),
        "baseline_random_median": 0.5,
        "baseline_random_top30":  0.3,
    }

    # Density: Spearman across phrases (pooled across songs)
    pe = np.array([r["pred_count"]   for r in per_phrase_rows], dtype=np.float64)
    le = np.array([r["label_count"]  for r in per_phrase_rows], dtype=np.float64)
    on = np.array([r["mean_onset"]   for r in per_phrase_rows], dtype=np.float64)
    density_summary = {
        "n_phrases":                 len(per_phrase_rows),
        "spearman_pred_vs_onset":    _spearman(pe, on),
        "spearman_label_vs_onset":   _spearman(le, on),
        "spearman_pred_vs_label":    _spearman(pe, le),
    }

    summary = {
        "checkpoint":  args.checkpoint,
        "n_songs":     n_used,
        "threshold":   args.threshold,
        "calibration_left":   {"ece": calib_l["ece"], "n": calib_l["n_total"]},
        "calibration_right":  {"ece": calib_r["ece"], "n": calib_r["n_total"]},
        "calibration_pooled": {"ece": calib_p["ece"], "n": calib_p["n_total"]},
        "audio_support":      audio_summary,
        "density":            density_summary,
    }

    # ---- Write outputs ----
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    (out_dir / "calibration_left.json").write_text(json.dumps(calib_l, indent=2))
    (out_dir / "calibration_right.json").write_text(json.dumps(calib_r, indent=2))
    (out_dir / "calibration_pooled.json").write_text(json.dumps(calib_p, indent=2))
    (out_dir / "per_song_support.json").write_text(json.dumps(per_song_support, indent=2))
    (out_dir / "per_phrase.json").write_text(json.dumps(per_phrase_rows, indent=2))

    # ---- Console report ----
    log.info("=" * 60)
    log.info("CALIBRATION (pooled L+R)  ECE = %.3f  n = %d",
             calib_p["ece"], calib_p["n_total"])
    log.info("%-12s %-8s %-12s %-12s", "bin", "count", "mean_pred", "frac_pos")
    for row in calib_p["bins"]:
        if row["count"] == 0:
            continue
        log.info("%4.2f-%-4.2f %-8d %-12.3f %-12.3f",
                 row["lo"], row["hi"], row["count"],
                 row["mean_pred"] or 0.0, row["frac_pos"] or 0.0)
    log.info("=" * 60)
    log.info("AUDIO SUPPORT  (onset-strength percentile rank in song;")
    log.info("                random baseline median=0.50, top30=0.30)")
    log.info("  median percentile  PRED  L=%.3f  R=%.3f",
             audio_summary["median_pct_pred_left"]  or 0.0,
             audio_summary["median_pct_pred_right"] or 0.0)
    log.info("  median percentile  LABEL L=%.3f  R=%.3f   (mapper-placement reference)",
             audio_summary["median_pct_label_left"]  or 0.0,
             audio_summary["median_pct_label_right"] or 0.0)
    log.info("  frac in top-30%%   PRED  L=%.3f  R=%.3f",
             audio_summary["top30_pred_left"]  or 0.0,
             audio_summary["top30_pred_right"] or 0.0)
    log.info("  frac in top-30%%   LABEL L=%.3f  R=%.3f",
             audio_summary["top30_label_left"]  or 0.0,
             audio_summary["top30_label_right"] or 0.0)
    log.info("=" * 60)
    log.info("DENSITY CORRELATION  (n_phrases = %d)", density_summary["n_phrases"])
    log.info("  Spearman pred-count  vs  onset-strength = %s",
             density_summary["spearman_pred_vs_onset"])
    log.info("  Spearman label-count vs  onset-strength = %s",
             density_summary["spearman_label_vs_onset"])
    log.info("  Spearman pred-count  vs  label-count    = %s",
             density_summary["spearman_pred_vs_label"])
    log.info("=" * 60)
    log.info("Wrote: %s", out_dir)


if __name__ == "__main__":
    main()
