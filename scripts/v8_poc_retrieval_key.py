"""Scoped V8 TASK 1 — is a per-instrument LAYERING fingerprint a better
song-memory retrieval key than the mean-MERT fingerprint?

Gates TASK 4 (swapping the song-memory cross-attn key). The North-Star bug is
"same chorus, inconsistent note patterns": the model has long-range memory
(song-memory attends over all ~150 phrase fingerprints) but its KEY is mean-pooled
MERT — a timbre average too coarse to recognize "the drop at 14s == the drop at
4:00". Hypothesis: a per-instrument layering+contour key separates human-identical
repeated phrases more sharply.

Method (per song, GPU-free — reuses CACHED .pt features):
  * phrases       : phrase_boundaries
  * human pattern : parse swing_tokens -> per-phrase binary occupancy over
                    (slot-in-phrase, hand, x, y, dir); pair similarity = cosine.
  * mean-MERT key : phrase_fingerprints  [N,768]
  * layering key  : mean-pool the cached instr_beat_features over each phrase [N,10]
  * For every within-song phrase pair, label it "human-identical" if human cosine
    >= HI (default 0.6) and "different" if <= LO (default 0.2); drop the ambiguous
    middle. Score each key by cosine similarity. ROC-AUC of (key cosine) predicting
    "human-identical" measures how well each key recovers the moments a human
    actually mapped the same way.

DoD: layering-key AUC > mean-MERT AUC (esp. on genre=electronic) -> green-light TASK 4.

Usage:
    python scripts/v8_poc_retrieval_key.py --n 40 --out outputs/v8_poc/retrieval_key.json
    python scripts/v8_poc_retrieval_key.py --songs 1ccca 1ddd1 --difficulty Expert
"""

from __future__ import annotations

import argparse
import json
import logging
import pathlib
import sys
import warnings

import numpy as np
import torch

warnings.filterwarnings("ignore")
REPO_ROOT = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from beatsaber_automapper.data.layout_dataset import _parse_events_from_tokens  # noqa: E402
from beatsaber_automapper.data.swing_tokenizer import BOMB  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("retrieval_key")

DATA_DIR = REPO_ROOT / "data/processed"

# Occupancy grid dims for the human-pattern vector.
N_HANDS, N_X, N_Y, N_DIR = 3, 4, 3, 9


def _phrase_human_vec(events, s: int, e: int) -> np.ndarray:
    """Binary occupancy vector for one phrase's human note pattern."""
    span = max(e - s, 1)
    vec = np.zeros(span * N_HANDS * N_X * N_Y, dtype=np.float32)
    for ev in events:
        if not (s <= ev.slot < e) or ev.kind == BOMB:
            continue
        sp = ev.slot - s
        h = min(max(ev.hand_idx if hasattr(ev, "hand_idx") else 0, 0), N_HANDS - 1)
        x = min(max(ev.x, 0), N_X - 1)
        y = min(max(ev.y, 0), N_Y - 1)
        idx = ((sp * N_HANDS + h) * N_X + x) * N_Y + y
        vec[idx] = 1.0
    return vec


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """ROC-AUC via the Mann-Whitney U statistic (no sklearn dep)."""
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(scores)
    ranks = np.empty(len(scores), dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    # average ties
    _, inv, counts = np.unique(scores, return_inverse=True, return_counts=True)
    sums = np.zeros(len(counts)); np.add.at(sums, inv, ranks)
    ranks = (sums / counts)[inv]
    r_pos = ranks[labels == 1].sum()
    n_p, n_n = len(pos), len(neg)
    return float((r_pos - n_p * (n_p + 1) / 2) / (n_p * n_n))


def process_song(pt_path: pathlib.Path, difficulty: str, hi: float, lo: float):
    try:
        d = torch.load(pt_path, weights_only=False, mmap=True)
    except Exception:
        return None
    if "instr_beat_features" not in d or "phrase_fingerprints" not in d:
        return None
    pb = d.get("phrase_boundaries") or []
    fps = d.get("phrase_fingerprints")
    instr = d.get("instr_beat_features")
    if len(pb) < 3 or fps is None or instr is None:
        return None

    diffs = d.get("difficulties", {})
    dname = difficulty if difficulty in diffs else next(
        (x for x in ("ExpertPlus", "Expert", "Hard", "Normal", "Easy") if x in diffs), None)
    if dname is None or not diffs[dname].get("swing_tokens"):
        return None
    events = _parse_events_from_tokens(diffs[dname]["swing_tokens"])
    # _Event has .hand (token id), not hand_idx; map to 0/1/2.
    from beatsaber_automapper.data.layout_dataset import _hand_idx
    for ev in events:
        ev.hand_idx = _hand_idx(ev.hand)
    if not events:
        return None

    fps = fps.float().numpy()
    instr = instr.float().numpy()
    N = min(len(pb), fps.shape[0])

    human_vecs, mert_keys, lay_keys = [], [], []
    for i in range(N):
        s, e = pb[i]
        hv = _phrase_human_vec(events, s, e)
        if hv.sum() < 2:           # skip near-empty phrases
            continue
        seg = instr[s:e]
        lay = seg.mean(axis=0) if len(seg) else np.zeros(instr.shape[1])
        human_vecs.append(hv); mert_keys.append(fps[i]); lay_keys.append(lay)

    if len(human_vecs) < 3:
        return None

    scores_mert, scores_lay, labels = [], [], []
    M = len(human_vecs)
    for i in range(M):
        for j in range(i + 1, M):
            # pad human vecs to equal length for cosine
            a, b = human_vecs[i], human_vecs[j]
            n = max(len(a), len(b))
            av = np.zeros(n); av[:len(a)] = a
            bv = np.zeros(n); bv[:len(b)] = b
            hsim = _cos(av, bv)
            if hsim >= hi:
                lab = 1
            elif hsim <= lo:
                lab = 0
            else:
                continue
            labels.append(lab)
            scores_mert.append(_cos(mert_keys[i], mert_keys[j]))
            scores_lay.append(_cos(lay_keys[i], lay_keys[j]))

    if sum(labels) == 0 or sum(labels) == len(labels):
        return None
    genre = d.get("mod_requirements", {}).get("genre", "unknown")
    return {
        "song_id": pt_path.stem, "genre": genre,
        "n_pairs": len(labels), "n_identical": int(sum(labels)),
        "scores_mert": np.array(scores_mert), "scores_lay": np.array(scores_lay),
        "labels": np.array(labels),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--songs", nargs="+", default=None)
    ap.add_argument("--n", type=int, default=40, help="max songs to scan for ones with instr features")
    ap.add_argument("--difficulty", default="Expert")
    ap.add_argument("--hi", type=float, default=0.6, help="human cosine >= hi -> 'identical' pair")
    ap.add_argument("--lo", type=float, default=0.2, help="human cosine <= lo -> 'different' pair")
    ap.add_argument("--out", type=pathlib.Path, default=REPO_ROOT / "outputs/v8_poc/retrieval_key.json")
    args = ap.parse_args()

    if args.songs:
        paths = [DATA_DIR / f"{s}.pt" for s in args.songs]
    else:
        paths = sorted(DATA_DIR.glob("*.pt"))

    results = []
    for p in paths:
        r = process_song(p, args.difficulty, args.hi, args.lo)
        if r is not None:
            results.append(r)
            log.info("[%s] genre=%s pairs=%d identical=%d", r["song_id"], r["genre"],
                     r["n_pairs"], r["n_identical"])
        if len(results) >= args.n:
            break

    if not results:
        log.error("No songs with cached instr_beat_features + usable phrases yet. "
                  "Run preprocess_instruments.py first (or wait for the batch).")
        sys.exit(1)

    def pooled_auc(key, subset=None):
        rs = results if subset is None else [r for r in results if r["genre"] == subset]
        if not rs:
            return float("nan"), 0
        sc = np.concatenate([r[key] for r in rs])
        lb = np.concatenate([r["labels"] for r in rs])
        return _auc(sc, lb), len(lb)

    auc_mert, n = pooled_auc("scores_mert")
    auc_lay, _ = pooled_auc("scores_lay")
    genres = sorted({r["genre"] for r in results})
    per_genre = {}
    for g in genres:
        am, ng = pooled_auc("scores_mert", g)
        al, _ = pooled_auc("scores_lay", g)
        per_genre[g] = {"n_pairs": ng, "auc_mert": round(am, 4), "auc_layering": round(al, 4),
                        "n_songs": sum(1 for r in results if r["genre"] == g)}

    report = {
        "n_songs": len(results), "n_pairs_total": int(n),
        "difficulty": args.difficulty, "hi": args.hi, "lo": args.lo,
        "auc_mean_mert": round(auc_mert, 4),
        "auc_layering": round(auc_lay, 4),
        "delta_layering_minus_mert": round(auc_lay - auc_mert, 4),
        "per_genre": per_genre,
        "verdict_layering_better": bool(auc_lay > auc_mert),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))

    print("\n" + "=" * 64)
    print(f"TASK 1 retrieval-key validation  ({len(results)} songs, {n} labeled pairs)")
    print("=" * 64)
    print(f"  mean-MERT key   AUC = {auc_mert:.4f}")
    print(f"  layering key    AUC = {auc_lay:.4f}   (Δ {auc_lay - auc_mert:+.4f})")
    print(f"  verdict: layering {'BETTER' if auc_lay > auc_mert else 'NOT better'} -> "
          f"{'GREEN-LIGHT' if auc_lay > auc_mert else 'HOLD'} TASK 4")
    for g, v in per_genre.items():
        print(f"    [{g:<12}] songs={v['n_songs']:>2} pairs={v['n_pairs']:>5}  "
              f"mert={v['auc_mert']:.3f}  layering={v['auc_layering']:.3f}")
    log.info("Wrote %s", args.out)


if __name__ == "__main__":
    main()
