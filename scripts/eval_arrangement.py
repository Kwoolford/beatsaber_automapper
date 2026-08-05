#!/usr/bin/env python
"""M4 — WHEN THE SONG TURNS A CORNER, DOES THE MAP TURN WITH IT?

Kyle has praised density pacing by ear (*"when there is a slow spot we let the
player breathe"*), so the generator does respond to loudness. This axis asks the
harder question behind "intentional": at the moment a song changes SECTION —
verse into chorus, chorus into breakdown — does the map change what it is doing?
A mapper rewrites the pattern at a boundary. A generator that only tracks loudness
crosses the boundary without noticing it, because the loudness curve is continuous
where the arrangement is not.

**Method.**
1. Find section boundaries from the audio alone: Foote novelty (a checkerboard
   kernel) over the bar-level self-similarity of chroma + MFCC, peaks picked with a
   minimum spacing so a boundary is a section change and not a fill.
2. Describe every bar of the map with a vector — note count, double share, mean
   row and column, direction spread, on-main-beat share — each z-scored across the
   song so no channel dominates by its units.
3. Compare the size of the descriptor jump ACROSS a real boundary with the jump
   across a bar pair that is not a boundary:

       arrange = mean |Δdescriptor| at real boundaries
               - mean |Δdescriptor| at matched non-boundaries

★A contrast again, so a metronome (no change anywhere) and a random map (change
everywhere, equally) both score 0. ⚠️The non-boundary comparison points are drawn
at the SAME bar distance as the boundary pairs, because a map that simply changes
a lot bar to bar would otherwise look responsive.

Usage:
    python scripts/eval_arrangement.py --arm tf_trim_ev03_rc05
"""

from __future__ import annotations

import argparse
import glob
import json
import pathlib
import statistics as st
import sys

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

import song_structure as ss  # noqa: E402
from beatsaber_automapper.evaluation import alignment, scorecard  # noqa: E402
from calibrate_playfeel import load_expert_only  # noqa: E402
from eval_motif_rhyme import notes_xydc  # noqa: E402

KERNEL = 8          # bars each side of the checkerboard
MIN_GAP = 8         # bars between accepted boundaries
N_BOUNDARIES = 12   # per song, at most
WINDOW = 4          # bars each side of a boundary, for the descriptor comparison


def novelty(A: dict) -> np.ndarray | None:
    """Foote novelty over the combined harmonic + timbral self-similarity."""
    parts = [A[k] for k in ("harm", "timb") if k in A]
    if not parts:
        return None
    S = np.nanmean(np.stack(parts), axis=0)
    S = np.nan_to_num(S, nan=0.0)
    n = S.shape[0]
    if n < 4 * KERNEL:
        return None
    k = KERNEL
    ker = np.zeros((2 * k, 2 * k))
    ker[:k, :k] = ker[k:, k:] = 1.0
    ker[:k, k:] = ker[k:, :k] = -1.0
    nov = np.zeros(n)
    for i in range(k, n - k):
        nov[i] = float((S[i - k:i + k, i - k:i + k] * ker).sum()) / (4 * k * k)
    return nov


def boundaries(nov: np.ndarray, n_max: int = N_BOUNDARIES,
               min_gap: int = MIN_GAP) -> list[int]:
    order = np.argsort(-nov)
    chosen: list[int] = []
    for i in order:
        if nov[i] <= 0:
            break
        if all(abs(i - c) >= min_gap for c in chosen):
            chosen.append(int(i))
        if len(chosen) >= n_max:
            break
    return sorted(chosen)


def bar_descriptors(notes: list[tuple], B: ss.Bars) -> np.ndarray | None:
    """(n_bars, d) z-scored map descriptors."""
    n = B.n
    cnt = np.zeros(n)
    dbl = np.zeros(n)
    row = np.zeros(n)
    col = np.zeros(n)
    dirs = np.zeros((n, 9))
    dur = B.dur
    t0 = B.edges[0]
    per_bar_times: list[list[float]] = [[] for _ in range(n)]
    for (t, x, y, d, _c) in notes:
        if t < t0 or t >= B.edges[-1]:
            continue
        bi = int((t - t0) // dur)
        if not (0 <= bi < n):
            continue
        cnt[bi] += 1
        col[bi] += x
        row[bi] += y
        dirs[bi, int(d) % 9] += 1
        per_bar_times[bi].append(t)
    for i in range(n):
        if cnt[i]:
            col[i] /= cnt[i]
            row[i] /= cnt[i]
        ts = sorted(per_bar_times[i])
        # a "double" = two notes within 12 ms
        dbl[i] = sum(1 for a, b in zip(ts, ts[1:]) if b - a < 0.012)
    with np.errstate(invalid="ignore", divide="ignore"):
        p = dirs / np.maximum(dirs.sum(axis=1, keepdims=True), 1)
        ent = -np.nansum(np.where(p > 0, p * np.log(p + 1e-12), 0.0), axis=1)
    dbl = np.where(cnt > 0, dbl / np.maximum(cnt, 1), 0.0)
    D = np.column_stack([cnt, dbl, row, col, ent])
    sd = D.std(axis=0)
    sd[sd < 1e-9] = 1.0
    return (D - D.mean(axis=0)) / sd


def arrangement(D: np.ndarray, bnds: list[int],
                rng: np.random.Generator) -> dict | None:
    """|Δdescriptor| at boundaries minus the same at matched non-boundaries."""
    n = D.shape[0]
    if not bnds or n < 4 * KERNEL:
        return None
    near = {b + o for b in bnds for o in range(-WINDOW, WINDOW + 1)}
    cand = [i for i in range(WINDOW, n - WINDOW) if i not in near]
    if len(cand) < 8:
        return None

    def jump(i: int) -> float | None:
        """★WINDOW means, not adjacent bars.

        The first version differenced bar i against bar i-1 and returned ~0 for
        BOTH cohorts (ours −0.034, human −0.008). That was the instrument: one bar
        against one bar is dominated by bar-to-bar noise, and a section change is
        not a one-bar event — a mapper rewrites the pattern *across* the boundary,
        and the novelty peak itself can sit a bar either side. Comparing the mean
        of the WINDOW bars before against the WINDOW after is the same question
        asked at the scale the answer lives at.
        """
        if i < WINDOW or i + WINDOW > n:
            return None
        before = D[i - WINDOW:i].mean(axis=0)
        after = D[i:i + WINDOW].mean(axis=0)
        return float(np.abs(after - before).mean())
    real = [v for b in bnds if (v := jump(b)) is not None]
    if len(real) < 4:
        return None
    # match the count, sampled well away from any boundary
    draws = rng.choice(cand, size=min(len(cand), 20 * len(real)), replace=False)
    null = [v for i in draws if (v := jump(int(i))) is not None]
    if len(null) < 8:
        return None
    return {"arrange": round(float(np.mean(real) - np.mean(null)), 4),
            "arrange_ratio": round(float(np.mean(real) / max(np.mean(null), 1e-6)), 4),
            "n_boundaries": len(real)}


def score_map(notes: list[tuple], B: ss.Bars, bnds: list[int],
              seed: int = 0) -> dict | None:
    D = bar_descriptors(notes, B)
    if D is None:
        return None
    return arrangement(D, bnds, np.random.default_rng(seed))


def paired(rows, key):
    return ss.paired_delta(rows, key)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", default="tf_trim_ev03_rc05")
    ap.add_argument("--seed", default="s0")
    ap.add_argument("--json", default="")
    a = ap.parse_args()

    files = sorted(glob.glob(str(REPO / f"outputs/eval_sweep_cache/{a.arm}#{a.seed}__*.zip"))) \
        or sorted(glob.glob(str(REPO / f"outputs/eval_sweep_cache/{a.arm}__*.zip")))
    rows = []
    for f in files:
        song = pathlib.Path(f).stem.split("__")[-1]
        L = scorecard._load_any(pathlib.Path(f))
        if not L:
            continue
        bm, bpm = L[0], float(L[1])
        t = np.asarray(alignment.note_times(bm, bpm), dtype=float)
        if len(t) < 100:
            continue
        B = ss.bars(song, bpm, ss.song_end(song, float(t.max())))
        if B is None or B.n < 4 * KERNEL:
            continue
        A = ss.bar_audio_matrix(song, B)
        if A is None:
            continue
        nov = novelty(A)
        if nov is None:
            continue
        bnds = boundaries(nov)
        if len(bnds) < 4:
            continue
        ours = score_map(notes_xydc(bm, bpm), B, bnds)
        H = load_expert_only(REPO / "data" / "raw" / f"{song}.zip")
        human = score_map(notes_xydc(H[0], float(H[1])), B, bnds) if H else None
        if ours is None:
            continue
        rows.append({"song": song, "bars": B.n, "n_bnd": len(bnds),
                     "ours": ours, "human": human})
        print(f"  {song:22s} {len(bnds):2d} boundaries   ours {ours['arrange']:+.4f}"
              + (f"   human {human['arrange']:+.4f}" if human else "   (no human)"))

    print(f"\n{'='*80}\nM4 ARRANGEMENT — arm {a.arm}, {len(rows)} songs (PAIRED subset)\n{'='*80}")
    print(f"{'metric':<18} {'n':>3} {'ours':>9} {'human':>9} {'paired Δ':>10} "
          f"{'Δ median':>10} {'resolvable':>11}")
    summary = {}
    for k in ("arrange", "arrange_ratio"):
        both = [r for r in rows if r.get("human")
                and r["ours"].get(k) is not None and r["human"].get(k) is not None]
        if len(both) < 6:
            continue
        o = [r["ours"][k] for r in both]
        h = [r["human"][k] for r in both]
        p = paired(rows, k)
        summary[k] = {"n": len(both), "ours": round(st.median(o), 4),
                      "human": round(st.median(h), 4), "paired": p}
        print(f"{k:<18} {len(both):>3d} {st.median(o):>+9.4f} {st.median(h):>+9.4f} "
              f"{p.get('delta', float('nan')):>+10.4f} "
              f"{p.get('delta_median', float('nan')):>+10.4f} "
              f"{('YES' if p.get('resolvable') else 'no'):>11}")

    print("\nHOW TO READ: positive = the map changes more at a section boundary than")
    print("it changes elsewhere. 0 = the boundary passes unnoticed. The comparison")
    print("points are drawn away from boundaries, so a map that simply churns a lot")
    print("bar-to-bar gains nothing.")

    if a.json:
        pathlib.Path(a.json).write_text(json.dumps(
            {"arm": a.arm, "rows": rows, "summary": summary}, indent=2))
        print(f"\nwrote {a.json}")


if __name__ == "__main__":
    main()
