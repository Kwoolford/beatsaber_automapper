"""The onset definition the JUDGE uses, made available to the BUILDER.

**The problem this closes** (PROGRESS.md 2026-08-20): the agent places notes from
`events.py`, which runs `htdemucs_6s` with its own per-stem detection, and the
alignment axis scores those notes against `outputs/onset_cache`, which is the 4-stem
union built by `scripts/build_onset_cache.py`. The two disagree by a median of
23-35 ms, and only 83-91 % of the events we place from sit within the axis' 50 ms
tolerance of an onset it recognises. A perfect map therefore scores ~0.83-0.91, and
we score 0.856 -- we are AT that ceiling, and the shortfall is mostly two components
disagreeing about what an onset is.

★**The fix moves the PLACING path onto the SCORED path, never the reverse.**
`build_onset_cache.py` says in its own docstring that changing the detection path
moves the human baseline, so the scored definition is the fixed point: every number
in TODO.md is against it. Re-scoring against the events we chose would be circular
and would void the axis.

⚠️**This is a SNAP, not a filter.** An event with no reference onset nearby keeps its
own time rather than being dropped -- dropping would silently change the note budget
and confound every density metric with an alignment change.
"""
from __future__ import annotations

import pathlib

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[1]
CACHE = REPO / "outputs" / "onset_cache"

# How far an event may be moved onto a reference onset. 60 ms is just outside the
# axis' own 50 ms tolerance, so an event that is a NEAR MISS gets corrected while one
# that is genuinely elsewhere is left alone. (88 % of our missed notes sat 50-120 ms
# out, so the near-miss band is where the mass is.)
SNAP_WINDOW_S = 0.060


def reference_onsets(song_id: str) -> np.ndarray | None:
    """The judge's onsets for this song, or None if they were never cached."""
    f = CACHE / f"{song_id}.npz"
    if not f.exists():
        return None
    z = np.load(f)
    keys = list(z.keys())
    if not keys:
        return None
    return np.sort(np.asarray(z[keys[0]], dtype=float))


def snap(times, song_id: str,
         window: float = SNAP_WINDOW_S) -> tuple[list[float], int, int]:
    """Move each event onto the nearest reference onset within `window`.

    Returns (times, n_moved, n_in). Times with no reference onset in range are
    unchanged, so only alignment can move.

    ⚠️`n_moved` counts INPUT times that were snapped, and the returned list is
    deduplicated -- two events landing on one onset collapse to one. Reporting
    `moved` against the OUTPUT length prints nonsense like "moved 818/745".
    """
    ref = reference_onsets(song_id)
    n_in = len(times)
    if ref is None or not len(ref) or not n_in:
        return list(times), 0, n_in
    t = np.asarray(sorted(times), dtype=float)
    i = np.clip(np.searchsorted(ref, t), 1, len(ref) - 1)
    lo, hi = ref[i - 1], ref[i]
    near = np.where(np.abs(t - lo) <= np.abs(t - hi), lo, hi)
    d = np.abs(t - near)
    out = np.where(d <= window, near, t)
    return sorted(set(out.tolist())), int((d <= window).sum()), n_in
