# Stage-1 rebuild: make the model hear the instruments

**Written:** 2026-07-27, after Kyle's manual review of the hand-offset maps.
**Status:** planned, not started. Blocked behind the cheap architectural fixes (see TODO.md).

## The finding this is built on

Kyle, on a guitar-driven rock track: *"the main instrument in this song is a heavy loud guitar
that starts and lays out the main beat that the rest of the instruments beat to. This is just not
attached to any sort of flow that a normal mapper would create… it's clear that our beat onset is
not distinguishing between instruments."*

He is right, and the checkpoint proves it. The production beat model `version_4` has exactly two
input projections:

| projection | input |
|---|---|
| `drum_proj` (768→512) | MERT embedding of the Demucs **drum stem** |
| `mix_proj` (768→512) | MERT embedding of the **full mix** |

That is the whole sensory world of Stage-1: *drums*, and *everything else averaged together*. A
lead guitar, a bass line and a vocal are indistinguishable inside `mix`. The model can follow a
drum kit and can follow overall energy, and it cannot follow a melodic line — which is precisely
what a human mapper maps on a rock track.

**This is a representation gap, not a tuning gap.** No decode lever can recover information the
encoder never received, which is why the eight inference levers tried on 2026-07-27 moved
statistics around without making the maps musical.

## Why the earlier attempt was shelved, and why that was wrong

`instr_proj` and the per-stem features already exist. `data/instrument_features.py` produces a
`[n_slots, 10]` layering vector — kick / snare / hat / bass / vocals / lead density, active-stem
count, lead pitch, lead Δpitch, bass pitch — and `BeatClassifier` has a gated `instr_proj` path
for it. TASK 2 trained it (`version_7`, d512/4L) and it was **abandoned because `val_f1_avg_tol`
did not improve** (0.600 vs the 0.603 baseline).

That decision was made on the wrong metric. `val_f1_avg_tol` is per-slot binary accuracy, and we
have since established three separate times that it anti-correlates with map quality — it is on
the "do not select checkpoints by this" list in the session conventions. The v2 suite plus Kyle's
ear are the yardsticks now, and neither existed when TASK 2 was killed.

**So this is not a new bet. It is re-running a shelved experiment against a yardstick that works.**

## Plan

### Phase 0 — cheap re-evaluation of what already exists (no training)
The `version_7` checkpoint may still be on disk. If so, generate with it and score on the v2
suite. This is hours, not a GPU night, and it either resurrects the work or kills it honestly.
- **DoD:** `version_7` scored on all four axes plus NPS and direction-idiom, against `version_4`.
- If it is already better on the axes that matter, promote and skip to Phase 2.

### Phase 1 — retrain Stage-1 with per-instrument input
Train `BeatClassifier` with `instr_dim=10` on the `require_instr` cohort. The preprocessing is
already cached (`instr_beat_features` on all 5320 `.pt` files), so this is a training run, not a
data build.
- **Select the checkpoint by the v2 suite, never by `val_f1`.** Generate from several epochs and
  score each; that is the whole lesson of this project.
- **DoD:** on the 24-song set, a per-instrument model beats `version_4` on the drop-dynamics
  metric (density must rise into high-energy sections) with flow, idiom and parity held.

### Phase 2 — make the model follow a *line*, not just layer densities
The 10-dim feature vector is per-slot **densities**. A human mapper follows one instrument's
*melodic and rhythmic line* through a section and switches at boundaries. Two candidate
representations, in order of cost:
1. **Per-stem onset channels as separate inputs** rather than a concatenated density vector, so
   the attention can attend to "the guitar" as its own stream.
2. **Per-stem MERT** — run MERT on each Demucs stem and give the model one projection per stem,
   mirroring how `drum_proj` already works for drums. This is the honest generalisation of the
   current architecture: the drum stem gets its own encoder because drums matter, and the same
   argument applies to the lead.
   Cost: MERT over 4 stems instead of 2 for the whole corpus. Storage and preprocess time need
   estimating before committing.

### Phase 3 — the axis that should have caught this
Build **A4 musical-role** (planned in `eval_suite_v2.md`, never built): which instrument is the
map following, and does it switch at section boundaries? With per-stem onsets available this is
mostly wiring. Without it we have no way to *measure* "the map follows the guitar", which is the
thing Kyle actually asked for — so this axis is what turns Phase 1/2 from a hunch into a DoD.

## Sequencing note

Phases 0 and 3 are cheap and should overlap with the near-term architectural fixes. Phase 1 is the
first GPU night worth spending after those land. Do **not** start Phase 2 before Phase 1 has
proven that instrument input helps at all on the v2 suite.
