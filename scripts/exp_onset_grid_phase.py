#!/usr/bin/env python
"""Does the ONSET-GRID phase predict where a human put his grid? (2026-09-17, P1.0)

For each corpus map with a cached onset set: the phase (mod one 1/4-beat slot) at which the most
onsets fall within 25 ms of a slot line, against the phase of the HUMAN's notes (the slot phase
most of his note times sit on). Both on the game clock. If the two agree up to one constant (the
onset detector's bias) with a tight spread, the onset grid is a sound per-song phase estimator and
the P1.0 outlier gate rests on something general, not on 1f9a0.

Run: python scripts/exp_onset_grid_phase.py --n 400
"""
from __future__ import annotations

import argparse
import pathlib
import random
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from exp_phase_sweep import load  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parent.parent
ONS = REPO / "outputs" / "onset_cache"
TOL = 0.025


def best_phase(times, q, step=0.002):
    phs = np.arange(0, q, step)
    share = np.array([np.mean(np.abs(((times - ph + q / 2) % q) - q / 2) <= TOL) for ph in phs])
    top = phs[share >= share.max() - 1e-12]          # centre of the flat top, not its first edge
    ang = np.angle(np.mean(np.exp(2j * np.pi * top / q)))
    return (ang / (2 * np.pi) * q) % q, float(share.max())


def wrap(d, q):
    return ((d + q / 2) % q) - q / 2


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n", type=int, default=400)
    a = ap.parse_args()
    paths = sorted(ONS.glob("*.npz"))
    random.Random(0).shuffle(paths)
    rows = []
    for f in paths:
        if len(rows) >= a.n:
            break
        zp = REPO / "data" / "raw" / f"{f.stem}.zip"
        if not zp.exists():
            continue
        try:
            bpm, off, t = load(zp)
            on = np.load(f)["onsets"]
        except Exception:  # noqa: BLE001
            continue
        if bpm <= 0 or len(t) < 100 or len(on) < 100:
            continue
        q = 60.0 / bpm / 4
        po, so = best_phase(on, q)
        ph, sh = best_phase(t + off, q)
        if sh < 0.5:            # the human is not on a 1/4-beat grid at all; nothing to compare
            continue
        rows.append((f.stem, bpm, wrap(po - ph, q), so, sh, q))
    d = np.array([r[2] for r in rows])
    med = np.median(d)
    res = np.array([wrap(x - med, r[5]) for x, r in zip(d, rows)])
    print(f"{len(rows)} maps. onset-grid minus human-grid phase: median {med * 1000:+.1f} ms "
          f"(the detector bias); residual |.| median {np.median(abs(res)) * 1000:.1f} ms, "
          f"p90 {np.percentile(abs(res), 90) * 1000:.1f} ms")
    for c in (0.010, 0.020, 0.030):
        print(f"  |residual| <= {c * 1000:.0f} ms on {(abs(res) <= c).mean():.1%}")
    so = np.array([r[3] for r in rows])
    lo = so < 0.45
    print(f"songs whose onsets sit poorly on any grid (share < 0.45): {lo.sum()}; their residual "
          f"|.| median {np.median(abs(res[lo])) * 1000 if lo.any() else float('nan'):.1f} ms")
    for r, x in sorted(zip(rows, res), key=lambda z: -abs(z[1]))[:8]:
        print(f"  {r[0]} {r[1]:6.1f} bpm  residual {x * 1000:+6.1f} ms  onsets-on-grid {r[3]:.2f}  human-on-grid {r[4]:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
