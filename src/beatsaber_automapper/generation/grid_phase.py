"""BEAT_GRID_PHASE — put the beat grid where the music's downbeat actually is.

**The defect.** `generate.py` runs `data.tempo.estimate_tempo`, takes `_fit.bpm`,
and then merely **logs** `_fit.phase_s`. The grid is anchored at **t = 0**, so a
song whose first beat does not land on the very first audio sample gets a grid
displaced by up to half a slot everywhere in the song. `estimate_tempo` fits
``time = period * index + phase``, so beat *k* belongs at ``phase + k*period``
while we place it at ``k*period``: the map is EARLY by ``phase``.

**The evidence** (2026-08-13, n=144 wide-cohort maps, no generation needed —
sweeping a global shift over maps we already had):

* 39 songs sit >0.10 below their human map on onset precision. A global shift
  recovers **+0.0428** on them against a **+0.0174** selection floor measured on
  the songs that are already fine.
* ★**20 of those 39 gain materially from a shift their HUMAN map does not want**
  ⇒ our grid is genuinely misplaced, not the onset detector. Only 1 song is a
  detector offset. Individual rescues are large: `2c352` 0.456 → 0.900,
  `2e593` 0.545 → 0.877, `29a01` 0.700 → 0.956 (above its own human).
* Phase and tempo are **independent** defects — 11 of the 20 have a correct BPM.
* Cohort median barely moves (−0.0327 → −0.0296) but **songs >0.10 below human go
  39 → 26**. Read the subset, not the mean.
* The phase we already estimate **predicts** the wanted shift: median |error|
  15.2 ms against a 39.1 ms chance level, and on the 12 songs a shift rescues most,
  **corr +0.757, median |error| 17.6 ms** — well inside the 50 ms tolerance.

⚠️**PARTLY CONFIRMED, and the limit is on the record**: 15 of the 39 failing songs
recover from no shift at all, and even the phase-fixable 20 keep a −0.076 median
residual. This fixes about half of one defect; it is not the alignment story.

★**Applied AFTER `postprocess_beatmap`, deliberately.** The diagnostic measured a
rigid translation of the finished note times — that is the thing with evidence
behind it. Re-gridding the MERT pooling would change what Stage-1 sees and is a
different (unmeasured) intervention. Postprocess also keeps operating on the grid
its parity and reachability rules were tuned against.

Default OFF (`BEAT_GRID_PHASE=1` to enable), per the project's one-lever-at-a-time
rule; nothing here runs unless it is switched on.
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

# NOTE: there is deliberately no "implausible phase" sanity bound here. The first
# draft had one, and a unit test showed it was dead code: `wrap_to_slot` already
# constrains the result to +-half a slot, so nothing can ever exceed a bound
# expressed in beats. A guard that cannot fire is worse than no guard — it reads as
# protection that is not there. The real degenerate inputs (bpm <= 0, subdiv < 1)
# are refused explicitly in `maybe_apply` instead.


def wrap_to_slot(phase_s: float, bpm: float, subdiv: int) -> float:
    """Wrap a phase into +-half a slot. A whole slot of offset is the same grid."""
    if bpm <= 0 or subdiv <= 0:
        return 0.0
    slot = 60.0 / bpm / subdiv
    if slot <= 0:
        return 0.0
    return (phase_s + slot / 2.0) % slot - slot / 2.0


def shift_beatmap(beatmap, *, bpm: float, phase_s: float) -> int:
    """Translate every note/bomb later by `phase_s` seconds. Returns notes dropped.

    A negative phase can push the earliest notes before the start of the song;
    those are dropped rather than clamped, because clamping would stack them all
    on beat 0 and manufacture a chord that was never generated.
    """
    if bpm <= 0 or phase_s == 0.0:
        return 0
    d_beats = phase_s * bpm / 60.0
    dropped = 0
    for attr in ("color_notes", "bomb_notes"):
        items = getattr(beatmap, attr, None)
        if not items:
            continue
        kept = []
        for n in items:
            nb = n.beat + d_beats
            if nb < 0.0:
                dropped += 1
                continue
            n.beat = nb
            kept.append(n)
        if len(kept) != len(items):
            items[:] = kept
    return dropped


# --------------------------------------------------------------------------- #
# MODE `search` — find the shift instead of predicting it
# --------------------------------------------------------------------------- #
# 🔴WHY MODE `1` (the fitted phase) IS DEAD. At n=149 it moved the failing subset
# only 39 -> 37 against an oracle's ~26 and DOUBLED the alignment gap (0.62 ->
# 1.32). The implementation was provably clean (note counts unchanged, every
# positional axis identical); the SHIFT was wrong: corr(applied, wanted) fell from
# the +0.367 validated offline to +0.065 in production, and to -0.318 on the songs
# that needed it most. The offline test fitted tempo from CACHED onsets while
# generate.py fits from freshly separated Demucs stems — a pre-build test run on a
# different input than production is not a pre-build test.
#
# ★WHAT SURVIVES: the diagnostic's "oracle" shift was never oracular. It maximised
# match rate against the cached STEM ONSETS — not against the human map — and the
# generator already computes stem onsets for the tempo fit. So the shift that
# recovered +0.0428 on the failing songs is FINDABLE at generation time. Search for
# it rather than predict it.
#
# ⚠️TWO GUARDS, both taught by the failure:
#   * `MIN_GAIN` — apply nothing unless the search finds a real improvement. Mode
#     `1` shifted the 105 already-fine songs by a median 22.1 ms for no reason, and
#     that is where the damage came from. **Do no harm to a song that is fine.**
#   * this optimises against OUR OWN onset detector, so it can in principle fit that
#     detector's systematic offset (the C2 / `h_dist` failure). The human control
#     said only 1 of 39 failing songs is a detector-offset case, so the risk is
#     bounded — but it must be re-checked against the human maps after any run, and
#     never assumed away.
SEARCH_RANGE_MS = 120.0
SEARCH_STEP_MS = 2.5
MIN_GAIN = 0.02


def _score(times, onsets, tol_s: float) -> tuple[float, float]:
    """(match rate, mean |offset| of matched notes) — the search objective.

    ⚠️**Match rate alone is a STEP function**: it only moves when a note crosses the
    tolerance boundary, so on a map whose notes are all within 50 ms it is flat and
    the search would see nothing to do. That is not the same as being aligned —
    on `1fccd` a −25 ms shift left precision identical to 4 dp while scatter went
    9.10 → 7.10 ms and lag +7.80 → −2.80. So rate is the primary objective (it is
    what the suite scores) and mean |offset| breaks ties, which recovers exactly
    that sub-tolerance centring for free.

    Uses the evaluation module's matcher so the generator optimises the quantity
    the suite measures — including its one-note-per-onset rule, without which note
    spam could manufacture a better "shift".
    """
    from beatsaber_automapper.evaluation.alignment import match_offsets

    if len(times) == 0 or len(onsets) == 0:
        return float("nan"), float("inf")
    matched, offsets = match_offsets(list(times), onsets, tol=tol_s)
    if not offsets:
        return 0.0, float("inf")
    return matched / len(times), sum(abs(o) for o in offsets) / len(offsets)


def search_shift(beatmap, *, bpm: float, onsets, tol_s: float = 0.050):
    """(best_shift_s, gain) maximising onset match rate. (0.0, 0.0) if none helps."""
    import numpy as np

    from beatsaber_automapper.evaluation.alignment import note_times

    if onsets is None or len(onsets) == 0 or bpm <= 0:
        return 0.0, 0.0
    times = np.asarray(note_times(beatmap, bpm), dtype=np.float64)
    if len(times) == 0:
        return 0.0, 0.0
    ref = np.sort(np.asarray(onsets, dtype=np.float64))

    base_rate, base_off = _score(times, ref, tol_s)
    if base_rate != base_rate:
        return 0.0, 0.0
    best_shift, best_rate, best_off = 0.0, base_rate, base_off
    n = int(SEARCH_RANGE_MS / SEARCH_STEP_MS)
    for k in range(-n, n + 1):
        d = k * SEARCH_STEP_MS / 1000.0
        if d == 0.0:
            continue
        r, off = _score(times + d, ref, tol_s)
        # Primary: match rate. Tie-break: tighter scatter. Final tie-break: the
        # smaller shift — if two shifts are equally good, the one that disturbs the
        # map less is the honest choice.
        if (r, -off, -abs(d)) > (best_rate, -best_off, -abs(best_shift)):
            best_shift, best_rate, best_off = d, r, off
    return best_shift, best_rate - base_rate


def maybe_apply(beatmap, *, bpm: float, phase_s: float, subdiv: int,
                onsets=None) -> bool:
    """Apply a grid-phase correction. Returns whether the map was shifted.

    `BEAT_GRID_PHASE=search` searches for the shift (recommended).
    `BEAT_GRID_PHASE=1` applies the FITTED phase — a measured negative at n=149,
    kept only so the refuted arm stays reproducible. Do not use it.
    """
    mode = os.environ.get("BEAT_GRID_PHASE", "0").lower()
    if mode not in ("1", "search"):
        return False
    if bpm <= 0 or int(subdiv) < 1:
        logger.info("BEAT_GRID_PHASE: unusable bpm=%r / subdiv=%r — no shift",
                    bpm, subdiv)
        return False

    if mode == "search":
        if onsets is None or len(onsets) == 0:
            logger.warning("BEAT_GRID_PHASE=search: no onsets available — no shift. "
                           "A silent skip here would look like a clean run on a song "
                           "that never got the treatment.")
            return False
        shift, gain = search_shift(beatmap, bpm=bpm, onsets=onsets)
        if shift == 0.0 or gain < MIN_GAIN:
            logger.info("BEAT_GRID_PHASE=search: best shift %.1f ms gains only "
                        "%+.4f (< %.2f) — LEAVING THE MAP ALONE",
                        shift * 1000.0, gain, MIN_GAIN)
            return False
        dropped = shift_beatmap(beatmap, bpm=bpm, phase_s=shift)
        logger.info("BEAT_GRID_PHASE=search: shifted %.1f ms, onset match %+.4f%s",
                    shift * 1000.0, gain,
                    f", dropped {dropped} note(s) before t=0" if dropped else "")
        return True

    # mode "1" — the refuted fitted-phase path.
    if not phase_s:
        logger.info("BEAT_GRID_PHASE: no usable phase (phase=%r) — no shift", phase_s)
        return False
    wrapped = wrap_to_slot(float(phase_s), float(bpm), int(subdiv))
    d_beats = wrapped * bpm / 60.0
    if wrapped == 0.0:
        logger.info("BEAT_GRID_PHASE: phase %.1f ms is a whole number of slots "
                    "— same grid, no shift", float(phase_s) * 1000.0)
        return False

    dropped = shift_beatmap(beatmap, bpm=bpm, phase_s=wrapped)
    logger.info("BEAT_GRID_PHASE: raw phase %.1f ms -> wrapped %.1f ms "
                "(%.3f beats); shifted map%s",
                float(phase_s) * 1000.0, wrapped * 1000.0, d_beats,
                f", dropped {dropped} note(s) before t=0" if dropped else "")
    return True
