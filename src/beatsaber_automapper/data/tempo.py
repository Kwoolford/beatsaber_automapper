"""Tempo AND PHASE estimation — the grid our notes are placed on.

Why this module exists (2026-08-02). Axis A8 measured our maps against the audio
for the first time and traced Kyle's "the notes are off beat" to the note grid:
`detect_bpm` was exact on **1 of 21** eval songs (median error 0.74%, four songs at
2/3 tempo). Stage-1 places every note on a 1/4-beat slot grid built from that
number, so on nearly every song the grid slides against the music as it plays.
Human maps sit on the *same* 1/4-beat grid we do — the grid is not too coarse, it
is in the wrong place.

`detect_bpm` calls `librosa.beat.beat_track` and keeps only the tempo scalar:

    tempo, _ = librosa.beat.beat_track(...)   # the beat POSITIONS are discarded

The discarded half is the useful half. This module keeps both, and adds the thing
the pipeline never estimated at all — the **phase**. The grid was anchored at t=0
regardless of where the music's first downbeat is.

    estimator     exact (<=0.1% of the human-declared bpm)   median |err|
    librosa            1/23                                       0.94%
    beat_lsq           3/23                                       0.93%
    comb              16/23                                       0.00%
    comb_multi        21/23                                       0.00%

Measured by `scripts/fit_tempo_grid.py` against 23 eval songs whose human mapper
synced the map by hand, which is the only ground truth available.

TWO THINGS THAT LOOK RIGHT AND ARE NOT, both found by doing them first:

1. **Maximising the fit over all tempo ratios picks half tempo.** The onset
   concentration R rises on COARSER grids whenever the music emphasises every
   other slot; unrestricted, it chose half tempo on 12 of 23 songs, including ones
   the local refinement already had exact. Ratios are restricted to >= 1 because
   every librosa error on this set is an UNDER-estimate and the directions are not
   symmetric: a grid finer than the music can express everything the music does; a
   coarser one cannot represent fast notes at all.
2. **A tempo with no confidence attached is how this went unnoticed for months.**
   R is reported alongside the estimate. Every song the local search got right
   scored R >= 0.177; every song it missed by a 2/3 ratio scored R <= 0.050.

WHAT R DOES NOT CATCH: a metrical-level TIE, where two levels both fit the music
well. The two remaining eval-set misses are exactly that (1fbda fits 116 better
than the human's 232; 1fbfb goes 3/2 too fine at R 0.175), and both report as
trusted because they *are* good fits — to the wrong level. `trusted` means "this
grid matches the music", not "this is the level a human would have written down".
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)

# Metrical levels tried before local refinement. A 2/3 error is not reachable by a
# few-percent refinement — a wrong metrical level is a different mistake and has to
# be enumerated. See the module docstring for why these are all >= 1.
RATIOS = (1.0, 4.0 / 3.0, 3.0 / 2.0, 2.0, 3.0)
BPM_MIN, BPM_MAX = 60.0, 250.0
# Within this fraction of the best fit the levels are metrically equivalent, and
# the FINEST is preferred (it can express everything the coarser one can).
R_NEAR = 0.9
# Below this, the grid has no real relationship to the music. Callers should treat
# an estimate this weak as untrusted rather than silently building a map on it.
R_TRUST = 0.10
SUBDIV = 4  # Stage-1's slot grid: 1/4 beat (beat_grid.BEAT_SUBDIV)


# Human mappers declare clean tempos — 160.0, 188.0, 138.0 — and the fitter lands
# within ~0.006 of them, emitting 159.99710481775244 where a human wrote 160.0.
# Snapping to the nearest half-integer inside this tolerance costs nothing
# measurable (0.006 bpm over a 3-minute song at 160 is ~7ms of accumulated drift,
# against a 50ms window) and buys the exact human value.
#
# It also appears to matter downstream: ArcViewer froze on every map carrying an
# unsnapped fitted tempo (2026-08-02) while the same maps with the old detector's
# tempo loaded, and every failing value sat a hair off an integer. That is
# suspicion, not proof — `outputs/arcviewer_probe_2026-08-02/` bisects it — but
# snapping is the right thing to do on the human-convention grounds alone.
SNAP_TO = 0.5
SNAP_TOL = 0.05


def snap_bpm(bpm: float, grid: float = SNAP_TO, tol: float = SNAP_TOL) -> float:
    """Round to the nearest multiple of `grid` when already within `tol` of it.

    Deliberately conservative: a genuinely non-integer tempo (145.3) is left
    alone, because snapping it would introduce the very drift this module exists
    to remove.
    """
    if not (bpm == bpm) or bpm <= 0:
        return bpm
    near = round(bpm / grid) * grid
    return float(near) if abs(bpm - near) <= tol else float(bpm)


@dataclass(slots=True)
class TempoFit:
    bpm: float
    phase_s: float
    r: float          # onset concentration on the fitted grid; doubles as confidence
    source: str       # which estimator produced it

    @property
    def trusted(self) -> bool:
        return self.r == self.r and self.r >= R_TRUST


def beat_lsq(beat_times: np.ndarray) -> tuple[float, float]:
    """Fit time = period * index + phase over detected beats -> (bpm, phase_s).

    The slope averages out per-beat jitter that a single tempogram scalar cannot,
    and the intercept is the phase. Tolerates a dropped or spurious beat by
    snapping each beat onto the index implied by the median inter-beat interval.
    """
    beat_times = np.asarray(beat_times, dtype=np.float64)
    if len(beat_times) < 8:
        return float("nan"), 0.0
    ibi = float(np.median(np.diff(beat_times)))
    if ibi <= 0:
        return float("nan"), 0.0
    idx = np.round((beat_times - beat_times[0]) / ibi)
    _, keep = np.unique(idx, return_index=True)
    idx, t = idx[keep], beat_times[keep]
    if len(idx) < 4:
        return float("nan"), 0.0
    period, phase = np.polyfit(idx, t, 1)
    if period <= 0:
        return float("nan"), 0.0
    return 60.0 / float(period), float(phase)


def comb_refine(onsets: np.ndarray, bpm0: float, subdiv: int = SUBDIV,
                span: float = 0.03, steps: int = 601) -> tuple[float, float, float]:
    """Maximise onset concentration on the 1/subdiv-beat grid near `bpm0`.

    R = |mean(exp(i*2*pi*t/slot))| over onsets: 1.0 means every onset sits exactly
    on a slot, ~0 means the grid is unrelated to the music. Returns (bpm, phase, R).
    """
    onsets = np.asarray(onsets, dtype=np.float64)
    if len(onsets) < 20 or not np.isfinite(bpm0) or bpm0 <= 0:
        return bpm0, 0.0, float("nan")
    best = (bpm0, 0.0, -1.0)
    for bpm in np.linspace(bpm0 * (1 - span), bpm0 * (1 + span), steps):
        slot = 60.0 / bpm / subdiv
        z = np.exp(1j * (2.0 * np.pi * (onsets / slot))).mean()
        r = float(abs(z))
        if r > best[2]:
            best = (float(bpm), float((np.angle(z) / (2.0 * np.pi)) * slot), r)
    return best


def fit_tempo(onsets: np.ndarray, bpm0: float, subdiv: int = SUBDIV) -> TempoFit:
    """Best (bpm, phase) over the metrical levels, refined against the onsets."""
    cands = []
    for ratio in RATIOS:
        cand = bpm0 * ratio
        if not (BPM_MIN <= cand <= BPM_MAX):
            continue
        bpm, phase, r = comb_refine(onsets, cand, subdiv=subdiv)
        if r == r:
            cands.append((bpm, phase, r))
    if not cands:
        return TempoFit(bpm0, 0.0, float("nan"), "fallback")
    best_r = max(c[2] for c in cands)
    near = [c for c in cands if c[2] >= R_NEAR * best_r]
    bpm, phase, r = max(near, key=lambda c: c[0])
    return TempoFit(snap_bpm(bpm), phase, r, "comb_multi")


def estimate_tempo(y: np.ndarray, sr: int, onsets: np.ndarray | None = None,
                   subdiv: int = SUBDIV) -> TempoFit:
    """Full estimate from audio. `onsets` (seconds) are used if supplied.

    Generation already has Demucs stems in hand, so the caller can pass the same
    stem-union onsets A8 scores against and this costs nothing extra.
    """
    import librosa

    y = np.asarray(y, dtype=np.float32)
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr, units="time")
    lib_bpm = float(np.atleast_1d(tempo)[0])
    bpm0, _phase0 = beat_lsq(np.asarray(beats, dtype=np.float64))
    if not np.isfinite(bpm0) or bpm0 <= 0:
        bpm0 = lib_bpm
    if onsets is None:
        onsets = librosa.onset.onset_detect(y=y, sr=sr, units="time", backtrack=True)
    fit = fit_tempo(np.asarray(onsets, dtype=np.float64), bpm0, subdiv=subdiv)
    if not fit.trusted:
        logger.warning(
            "tempo fit is WEAK (bpm %.2f, R %.3f < %.2f) — the grid may be unrelated "
            "to the music. librosa said %.2f.", fit.bpm, fit.r, R_TRUST, lib_bpm)
    return fit
