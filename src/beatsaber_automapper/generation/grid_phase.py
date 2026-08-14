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


def maybe_apply(beatmap, *, bpm: float, phase_s: float, subdiv: int) -> bool:
    """Apply the fitted grid phase if `BEAT_GRID_PHASE=1`. Returns whether it ran."""
    if os.environ.get("BEAT_GRID_PHASE", "0") != "1":
        return False
    if not phase_s or bpm <= 0 or int(subdiv) < 1:
        logger.info("BEAT_GRID_PHASE: no usable phase (phase=%r, bpm=%r, subdiv=%r)"
                    " — no shift", phase_s, bpm, subdiv)
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
