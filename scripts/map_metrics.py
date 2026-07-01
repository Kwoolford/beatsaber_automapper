#!/usr/bin/env python3
"""Shared map-only quality metrics (2026-06-30).

One place that turns a generated/human beatmap into the scorecard metrics the
eval loop tracks, so eval_sweep, eval_layout_ckpt, and the human-baseline command
all compute them identically. All metrics here are MAP-ONLY (no audio/Demucs) so
they are cheap; audio-coupled metrics (density_corr, alignment) live in the sweep
harness which has the cached references.

A metric dict has a stable schema; `HUMAN_TARGET` and `BETTER` document, per key,
the human-reference value and whether higher or lower is better — used to render
"vs human" columns and pass/fail.
"""
from __future__ import annotations

import math
import pathlib
import sys

import numpy as np

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
from feel_disc_poc import load_v7  # noqa: E402
from best_of_n_poc import monotony_features  # noqa: E402

# Direction in human maps; "better" = closer to human (not strictly mono),
# but for the simple ones higher/lower is a useful nudge arrow.
BETTER = {
    "row_conc": "low", "col_conc": "low", "monotony": "low", "pattern_repeat": "low",
    "grid_coverage": "high", "dir_entropy": "high", "nps": "human", "n_notes": "human",
}
# Refreshed by eval_sweep.human-baseline (writes human_baseline.json); these
# defaults are the 2026-06-30 40-human-map sample so the constants are useful
# even before a fresh baseline run.
HUMAN_TARGET = {
    "row_conc": 0.494, "col_conc": 0.287, "monotony": 0.431, "pattern_repeat": 0.002,
    "grid_coverage": 0.958, "dir_entropy": 0.804, "nps": 5.18, "n_notes": None,
}


def map_metrics(zip_or_dir: str | pathlib.Path, difficulty: str = "Expert") -> dict:
    """Map-only quality metrics from a beatmap zip/dir (no audio needed)."""
    seq = load_v7(str(zip_or_dir), difficulty)
    n = len(seq)
    if n == 0:
        return {"n_notes": 0}
    xs = np.rint(seq[:, 1] * 3).astype(int).clip(0, 3)
    ys = np.rint(seq[:, 2] * 2).astype(int).clip(0, 2)
    dirs = seq[:, 3:12].argmax(axis=1)
    dt = seq[:, 0]
    dur = float(np.cumsum(dt)[-1]) if n else 0.0

    row_dist = np.bincount(ys, minlength=3) / n
    col_dist = np.bincount(xs, minlength=4) / n
    # grid coverage: fraction of the 12 (col,row) cells that hold >=1 note
    cells = set(zip(xs.tolist(), ys.tolist()))
    grid_coverage = len(cells) / 12.0
    # direction variety: normalised Shannon entropy over the 9 cut-direction bins
    dcounts = np.bincount(dirs, minlength=9) / n
    nz = dcounts[dcounts > 0]
    dir_entropy = float(-(nz * np.log(nz)).sum() / math.log(9)) if len(nz) > 1 else 0.0

    mf = monotony_features(seq)
    return {
        "n_notes": int(n),
        "nps": round(n / dur, 3) if dur > 0 else 0.0,
        "row_conc": round(float(row_dist.max()), 3),
        "col_conc": round(float(col_dist.max()), 3),
        "grid_coverage": round(grid_coverage, 3),
        "dir_entropy": round(dir_entropy, 3),
        "monotony": mf["monotony"],
        "pattern_repeat": mf["pattern_repeat"],
        "row_dist": [round(float(v), 2) for v in row_dist],
        "col_dist": [round(float(v), 2) for v in col_dist],
    }


if __name__ == "__main__":
    import argparse, json
    ap = argparse.ArgumentParser()
    ap.add_argument("map")
    ap.add_argument("--difficulty", default="Expert")
    a = ap.parse_args()
    print(json.dumps(map_metrics(a.map, a.difficulty), indent=2))
