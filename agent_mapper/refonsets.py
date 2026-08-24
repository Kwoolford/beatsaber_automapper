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

**VALIDATED 2026-08-24 against an independent reference** (`scripts/diag_snap_independent.py`).
The 0.856 -> 0.890 gain was open to the charge of being circular -- `onset_precision`
IS the share of our notes near the cache we snap to. Scored instead against the HUMAN
MAPPER'S OWN NOTE TIMES, which the snap knows nothing about: 17/23 songs move closer
(sign test p=0.035), near-human@50ms 0.627 -> 0.665, sign stable at every tolerance
20-100 ms. ★**Negative controls kill the concentration explanation** -- snapping to the
same onsets shifted +200 ms scores -0.034 and to random times -0.061, though both
concentrate event times identically. The lift is musical, not an artifact of
discretisation.
"""
from __future__ import annotations

import hashlib
import pathlib
import sys

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[1]
CACHE = REPO / "outputs" / "onset_cache"

# How far an event may be moved onto a reference onset. 60 ms is just outside the
# axis' own 50 ms tolerance, so an event that is a NEAR MISS gets corrected while one
# that is genuinely elsewhere is left alone. (88 % of our missed notes sat 50-120 ms
# out, so the near-miss band is where the mass is.)
SNAP_WINDOW_S = 0.060


def _load(f: pathlib.Path) -> np.ndarray | None:
    if not f.exists():
        return None
    z = np.load(f)
    keys = list(z.keys())
    if not keys:
        return None
    return np.sort(np.asarray(z[keys[0]], dtype=float))


# ⚠️`_reconcile` in mapctl runs once per candidate inside the accent-percentile
# search, so an uncached lookup would re-hash the audio and re-read the npz dozens of
# times per build. Memoised per process, keyed by (path, mtime, size) so an audio file
# edited in place is not served a stale answer.
_ONSET_MEMO: dict = {}
_KEY_MEMO: dict = {}


def audio_key(audio: pathlib.Path) -> str:
    """Cache key for a song that is not in the corpus: `audio_<content hash>`.

    ⚠️**Keyed by CONTENT, and deliberately prefixed.** A non-corpus song must never be
    able to write `outputs/onset_cache/<song_id>.npz` -- that file is the fixed point
    every alignment number in TODO.md is measured against, and a collision would move
    the human baseline silently. The prefix makes collision impossible; the hash makes
    the same audio reuse its onsets instead of paying Demucs twice.
    """
    st = audio.stat()
    memo = (str(audio.resolve()), st.st_mtime_ns, st.st_size)
    if memo in _KEY_MEMO:
        return _KEY_MEMO[memo]
    h = hashlib.sha1()
    with open(audio, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    _KEY_MEMO[memo] = f"audio_{h.hexdigest()[:16]}"
    return _KEY_MEMO[memo]


def compute_reference_onsets(audio: pathlib.Path,
                             verbose: bool = True) -> np.ndarray | None:
    """Run the JUDGE'S detection path on arbitrary audio and cache the result.

    ★**This is what makes `--snap-onsets` work on a song we have never seen** -- the
    reason it was corpus-only was never that the detector needs a corpus, only that
    the CACHE is keyed by corpus song id. `build_onset_cache.compute_onsets` takes a
    plain audio path.

    ⚠️**Reuses that function rather than reimplementing the detection.** The onset
    definition is the fixed point; a second implementation that drifted from it would
    reintroduce the exact detector disagreement this module exists to close.

    ⚠️Costs a **second Demucs pass** (4-stem `htdemucs`, on top of the `htdemucs_6s`
    the agent already runs for events). It is not free, which is why it stays opt-in.

    Returns None if Demucs is unavailable -- `compute_onsets` refuses the mix-only
    fallback, and degrading to a different detection path silently would be worse
    than not snapping at all.
    """
    key = audio_key(audio)
    dest = CACHE / f"{key}.npz"
    cached = _load(dest)
    if cached is not None:
        if verbose:
            print(f"snap-onsets: reusing computed onsets for {audio.name} ({key})")
        return cached

    sp = str(REPO / "scripts")
    if sp not in sys.path:
        sys.path.insert(0, sp)
    try:
        from build_onset_cache import compute_onsets
        if verbose:
            print(f"snap-onsets: computing the judge's onsets for {audio.name} "
                  f"(a second Demucs pass — one-off, cached as {key})")
        union, _per_stem = compute_onsets(audio)
    except Exception as exc:  # noqa: BLE001
        if verbose:
            print(f"⚠️snap-onsets: could not compute onsets for {audio.name} ({exc}); "
                  f"times UNCHANGED")
        return None

    import librosa
    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez(dest, onsets=union,
             duration=np.float64(librosa.get_duration(path=str(audio))),
             song_id=key, audio=str(audio.name), method="demucs_stem_union")
    return np.sort(np.asarray(union, dtype=float))


def reference_onsets(song_id: str,
                     audio: pathlib.Path | str | None = None,
                     compute: bool = False) -> np.ndarray | None:
    """The judge's onsets for this song, or None if they cannot be had.

    Prefers the corpus entry keyed by `song_id` -- for a songset song that file IS the
    scored reference, and recomputing it would be both wasteful and a chance to drift.
    Falls back to the content-keyed entry, and only then, with `compute=True`, to
    running the detector.
    """
    memo = (song_id, str(audio) if audio else "", bool(compute))
    if memo in _ONSET_MEMO:
        return _ONSET_MEMO[memo]

    hit = _load(CACHE / f"{song_id}.npz") if song_id else None
    if hit is None and audio is not None:
        audio = pathlib.Path(audio)
        if audio.exists():
            hit = _load(CACHE / f"{audio_key(audio)}.npz")
            if hit is None and compute:
                hit = compute_reference_onsets(audio)
    _ONSET_MEMO[memo] = hit
    return hit


def snap(times, song_id: str,
         window: float = SNAP_WINDOW_S,
         audio: pathlib.Path | str | None = None,
         compute: bool = False) -> tuple[list[float], int, int]:
    """Move each event onto the nearest reference onset within `window`.

    Returns (times, n_moved, n_in). Times with no reference onset in range are
    unchanged, so only alignment can move.

    Pass `audio` (and `compute=True`) to snap a song that is not in the corpus; see
    `reference_onsets`.

    ⚠️`n_moved` counts INPUT times that were snapped, and the returned list is
    deduplicated -- two events landing on one onset collapse to one. Reporting
    `moved` against the OUTPUT length prints nonsense like "moved 818/745".
    """
    ref = reference_onsets(song_id, audio=audio, compute=compute)
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
