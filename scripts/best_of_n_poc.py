#!/usr/bin/env python3
"""P1-4 — best-of-N rerank PoC (Phase-2 kickoff, 2026-06-16).

Given N stochastic V7 candidate maps for ONE song, pick the best by combining the
three Phase-1 perception signals into one reranker:

  1. feel-disc (the EARLY-STOPPED ep1 ranker, outputs/feel_disc_ep1_2026-06-15.pt):
     higher margin logit = more human-like. AUC(human vs V7)=1.0, within-V7 spread
     10.8% of the human gap (a usable ordering — the saturated 60-ep ckpt is NOT).
  2. swing-sim HARD FILTER: any candidate with >0 wrist-break violations is dropped
     outright (a physical un-playability gate). NOTE the P1-3 finding: production V7
     is post-processed parity-CLEAN, so this gate is usually inactive among real
     samples — it's a safety net, not the discriminator.
  3. MONOTONY / structure penalty (NEW, from P1-3): parity alone won't separate
     post-process candidates; the real discriminator Claude used blind was monotony +
     missing structure. So penalize, from the note stream itself:
        - pattern_repeat   : adjacent (x,y,dir) tuples identical  ("red→ + blue▲" loop)
        - pattern_entropy⁻ : low Shannon entropy of (x,y,dir) tuples (few patterns)
        - density_flatness : low CV of onsets/2s window (flat density ignores song)
        - row_concentration: notes piled in one row (bottom-row for-sport streams)

final_score = z(feel_logit) - LAMBDA * z(monotony), ranked descending over the
swing-sim survivors. The "winner" is rank-1; the "control" is the no-rerank pick
(first candidate by filename, i.e. what you'd ship without selection).

CLI:
  python scripts/best_of_n_poc.py --maps 'outputs/bon_2026-06-16/cand_*.zip' \
      --ckpt outputs/feel_disc_ep1_2026-06-15.pt --difficulty Expert \
      --out-dir outputs/bon_2026-06-16 [--lambda 1.0] [--no-render]
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import pathlib
import subprocess
import sys

import numpy as np
import torch

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "src"))

from feel_disc_poc import MAXLEN, NoteSeqDisc, load_v7  # noqa: E402
from beatsaber_automapper.evaluation import swing_sim as ss  # noqa: E402


# ----------------------------------------------------------------------------- feel-disc
def load_disc(ckpt_path: pathlib.Path, device: str) -> NoteSeqDisc:
    model = NoteSeqDisc().to(device)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


@torch.no_grad()
def feel_logit(model: NoteSeqDisc, seq: np.ndarray, device: str) -> float:
    """Margin logit (human - gen); rank-stable even when softmax saturates."""
    X = torch.from_numpy(seq).unsqueeze(0).to(device)          # [1,L,12]
    M = torch.zeros(1, seq.shape[0], dtype=torch.bool, device=device)
    logits = model(X, M)[0]
    return float(logits[1] - logits[0])


# ----------------------------------------------------------------------------- monotony
def monotony_features(seq: np.ndarray) -> dict:
    """All four monotony proxies straight from the [L,12] feel-disc feature seq.

    col0 = dt(s, capped 2.0); col1 = x/3; col2 = y/2; col3:12 = dir one-hot.
    Every component is in [0,1] with HIGHER = MORE MONOTONOUS = worse.
    """
    L = seq.shape[0]
    xs = np.rint(seq[:, 1] * 3).astype(int)
    ys = np.rint(seq[:, 2] * 2).astype(int)
    ds = seq[:, 3:12].argmax(axis=1)
    tuples = list(zip(xs.tolist(), ys.tolist(), ds.tolist()))

    # 1. adjacent-identical fraction
    pattern_repeat = float(np.mean([t == s for t, s in zip(tuples[:-1], tuples[1:])])) if L > 1 else 0.0

    # 2. inverse normalised entropy of (x,y,dir) tuples
    _, counts = np.unique(np.array(tuples), axis=0, return_counts=True)
    p = counts / counts.sum()
    H = float(-(p * np.log2(p)).sum())
    Hmax = math.log2(L) if L > 1 else 1.0
    pattern_entropy_inv = float(1.0 - (H / Hmax)) if Hmax > 0 else 1.0

    # 3. density flatness: low CV of onsets per 2s window = flat density (ignores song)
    t = np.cumsum(seq[:, 0])
    total = float(t[-1]) if L else 0.0
    if total > 2.0:
        nb = max(2, int(math.ceil(total / 2.0)))
        counts_w, _ = np.histogram(t, bins=nb, range=(0.0, total))
        m = counts_w.mean()
        cv = float(counts_w.std() / m) if m > 0 else 0.0
        density_flatness = float(1.0 / (1.0 + cv))   # cv 0 -> 1.0 (flat); large cv -> 0
    else:
        density_flatness = 1.0

    # 4. single-row concentration (bottom-row streams etc.)
    row_concentration = float(max((ys == r).mean() for r in (0, 1, 2))) if L else 0.0

    combined = float(np.mean([pattern_repeat, pattern_entropy_inv,
                              density_flatness, row_concentration]))
    return {
        "pattern_repeat": round(pattern_repeat, 4),
        "pattern_entropy_inv": round(pattern_entropy_inv, 4),
        "density_flatness": round(density_flatness, 4),
        "row_concentration": round(row_concentration, 4),
        "monotony": round(combined, 4),
    }


# ----------------------------------------------------------------------------- swing-sim
def swing_violations(zip_path: pathlib.Path, difficulty: str) -> int | None:
    try:
        bm, bpm = ss._load_difficulty(zip_path, difficulty)
    except Exception:  # noqa: BLE001
        return None
    return int(ss.simulate(bm, bpm=bpm).violations)


# ----------------------------------------------------------------------------- per-map
def score_map(zip_path: pathlib.Path, difficulty: str, model: NoteSeqDisc,
              device: str) -> dict | None:
    seq = load_v7(str(zip_path), difficulty)
    if seq is None:
        return None
    rec = {"name": zip_path.name, "path": str(zip_path), "n_notes": int(seq.shape[0])}
    rec["feel_logit"] = round(feel_logit(model, seq, device), 4)
    rec["swing_violations"] = swing_violations(zip_path, difficulty)
    rec.update(monotony_features(seq))
    return rec


def _z(vals: list[float]) -> np.ndarray:
    a = np.array(vals, dtype=float)
    sd = a.std()
    return (a - a.mean()) / sd if sd > 1e-9 else np.zeros_like(a)


def rerank(records: list[dict], lam: float) -> list[dict]:
    """Attach z-scores + final_score and return swing-sim survivors, best first."""
    survivors = [r for r in records if (r.get("swing_violations") or 0) == 0]
    pool = survivors if survivors else records  # never return empty
    zf = _z([r["feel_logit"] for r in pool])
    zm = _z([r["monotony"] for r in pool])
    for r, a, b in zip(pool, zf, zm):
        r["z_feel"] = round(float(a), 4)
        r["z_monotony"] = round(float(b), 4)
        r["final_score"] = round(float(a - lam * b), 4)
    pool.sort(key=lambda r: r["final_score"], reverse=True)
    return pool


# ----------------------------------------------------------------------------- render
def render(zip_path: str, difficulty: str, out_png: pathlib.Path, title: str) -> bool:
    cmd = [sys.executable, str(REPO / "scripts/render_map.py"), zip_path,
           "--difficulty", difficulty, "--out", str(out_png), "--title", title]
    r = subprocess.run(cmd, cwd=str(REPO), capture_output=True, text=True)
    if r.returncode != 0:
        print(f"  !! render failed for {zip_path}:\n{r.stderr[-500:]}")
        return False
    return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--maps", required=True, help="glob of candidate .zip maps")
    ap.add_argument("--ckpt", type=pathlib.Path,
                    default=REPO / "outputs/feel_disc_ep1_2026-06-15.pt")
    ap.add_argument("--difficulty", default="Expert")
    ap.add_argument("--out-dir", type=pathlib.Path, required=True)
    ap.add_argument("--lambda", dest="lam", type=float, default=1.0,
                    help="monotony penalty weight in z(feel) - lambda*z(monotony)")
    ap.add_argument("--no-render", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    maps = sorted(glob.glob(args.maps))
    if len(maps) < 2:
        print(f"!! need >=2 candidates, got {len(maps)} from {args.maps}")
        sys.exit(2)

    model = load_disc(args.ckpt, args.device)
    print(f"[bon] scoring {len(maps)} candidates (ckpt={args.ckpt.name}, lambda={args.lam})")

    records = []
    for mp in maps:
        rec = score_map(pathlib.Path(mp), args.difficulty, model, args.device)
        if rec is None:
            print(f"  -- skip (no notes): {pathlib.Path(mp).name}")
            continue
        records.append(rec)
    if len(records) < 2:
        print("!! fewer than 2 scorable candidates; aborting"); sys.exit(2)

    control = min(records, key=lambda r: r["name"])    # no-rerank = first by filename
    ranked = rerank(records, args.lam)
    winner = ranked[0]

    n_filtered = len(records) - len([r for r in records if (r.get("swing_violations") or 0) == 0])
    print(f"\n[bon] swing-sim filtered {n_filtered}/{len(records)} (violations>0)")
    print(f"[bon] WINNER  {winner['name']}: final={winner['final_score']} "
          f"feel={winner['feel_logit']} monotony={winner['monotony']} "
          f"viol={winner['swing_violations']}")
    print(f"[bon] CONTROL {control['name']}: "
          f"feel={control['feel_logit']} monotony={control['monotony']} "
          f"viol={control['swing_violations']}")
    same = winner["name"] == control["name"]
    print(f"[bon] rerank {'== control (no improvement on this draw)' if same else 'MOVED off control'}")

    # spread summary — does the reranker actually have signal to act on?
    feel_spread = max(r["feel_logit"] for r in records) - min(r["feel_logit"] for r in records)
    mono_spread = max(r["monotony"] for r in records) - min(r["monotony"] for r in records)
    print(f"[bon] N-spread: feel_logit={feel_spread:.3f}  monotony={mono_spread:.3f}")

    summary = {
        "n_candidates": len(records), "lambda": args.lam,
        "winner": winner["name"], "control": control["name"],
        "rerank_moved": not same,
        "feel_spread": round(feel_spread, 4), "monotony_spread": round(mono_spread, 4),
        "winner_rec": winner, "control_rec": control,
        "ranking": ranked,
    }
    out_json = args.out_dir / "bon_summary.json"
    out_json.write_text(json.dumps(summary, indent=2))
    print(f"[bon] wrote {out_json}")

    if not args.no_render:
        render(winner["path"], args.difficulty, args.out_dir / "winner.png",
               f"WINNER {winner['name']} (final {winner['final_score']})")
        render(control["path"], args.difficulty, args.out_dir / "control.png",
               f"CONTROL {control['name']} (no-rerank)")
        print(f"[bon] rendered winner.png + control.png in {args.out_dir}")


if __name__ == "__main__":
    main()
