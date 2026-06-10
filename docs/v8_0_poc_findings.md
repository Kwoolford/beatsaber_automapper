# V8-0 PoC — Go/No-Go Findings (2026-06-02 overnight)

**Verdict in one line:** The *full* V8 rebuild (rip out the BPM-grid WHEN backbone and
replace it with a continuous-time transcription event-selector) is **NO-GO** — its core
premise is not supported by the data. A **scoped V8 is GO**: (1) kill the section-threshold
note-gate that actually causes the silent-drop, and (2) add per-stem transcription as
**Stage-2 conditioning** for directional cohesion. Both are cheap relative to the rebuild.

Gate scripts: `scripts/v8_poc.py` (single song, plots) and `scripts/v8_poc_alignment.py`
(N in-dataset songs, the decisive metric). Raw data: `outputs/v8_poc/`.

---

## What was tested

basic-pitch installed on Python 3.12 via the **ONNX** backend (TensorFlow has no cp312
wheel; installed `--no-deps` + `onnxruntime` + `mir_eval` + `resampy` + `pretty_midi`).
Runs at ~0.5 s/stem, ~6 s/song including Demucs. Per-stem transcription:
bass/vocals/other → basic-pitch; drums → multi-band librosa onset (kick/snare/hat).

Two evaluations:
1. **SO TIRED ROCK** (held-out, no human map): piano-roll + lead-contour plots.
2. **12 random in-dataset songs** (which DO have human maps): does the transcribed
   NoteEvent pool predict the human mapper's note times better than the signal V7 has?

---

## Results

### (c) Per-instrument structure — **STRONG PASS**
`outputs/v8_poc/so_tired_rock/pianoroll.png` + `lead_contour.png`. The transcription cleanly
recovers a **periodic bass riff** (~MIDI 28–33), a coherent **lead melodic contour**, and
**drum bands** (kick pattern, snare/hat). Structural breakdowns (kick dropouts at ~62–66 s
and ~107–110 s) show up as gaps. This is real per-instrument signal that V7 (which feeds Stage 2
only the blended `other` stem, mean-pooled to a grid) has **no access to**. This is the
half of the V8 thesis that holds up.

### (b) Alignment to human maps (12 songs, candidate-pool coverage)

| metric | transcribed pool | librosa union (≈V7 onset signal) |
|---|---|---|
| cover_recall @±50 ms | **0.895** | 0.743 |
| cover_recall @±25 ms | **0.788** | 0.541 |
| naive all-events F1 @±50 ms | 0.334 (prec 0.21) | 0.541 |

- Transcription **covers more human notes** than librosa onsets at both tolerances — a
  richer, more musical candidate pool. ✅ for "better signal than spectral-flux onsets."
- BUT the naive **F1 is 0.33 < the 0.41 literal gate** — because we keep *all* ~3 k events
  (precision 0.21). Stage 1's job would be to thin them; the PoC can't run an untrained
  selector, so the literal F1 bar is **not met**. The pool *ceiling* (cover_recall) is the
  honest number, and it's good — but see the next point for why that doesn't justify the rebuild.

### ⚑ The premise-killer: BPM-grid off-grid residual is ~0–6%

| tolerance | human notes NOT representable on V7's 1/16 BPM grid |
|---|---|
| ±50 ms | **0.7 %** |
| ±25 ms | **6.0 %** |

The V8 plan's **Layer 2** claims V7 is "trapped in BPM-quantized space and can never place a
note off-grid even when the music demands it." **The data refutes this.** Human mappers place
**94–99 %** of their notes on V7's existing 1/16-note grid. The grid already represents the
music's timing. (Grid anchored at t=0 with no offset reproduces this — so the global-BPM
anchoring the plan worried about is also fine in practice.)

Consequently V7's candidate pool — *every* grid slot — already covers ~100 % of human notes,
which is **more** than transcription's 89 %. **Transcription does not improve WHEN coverage.**
Replacing the grid Stage 1 with a continuous event-selector is a large rebuild that targets a
~1–6 % problem.

### (a) "Dense drop cluster V7 misses" — NOT demonstrated on this song
SO TIRED ROCK is constant-energy rock; 12–16 s is not denser than the rest, and **V7's own
`v7_section_aware.zip` placed 17 notes there** — its drop is not silent. The silent-drop
complaint was on the newer *song-memory* maps, and its cause is Layer 1 (below), not the input
representation.

---

## Root cause re-attribution (what actually makes maps bad)

| V8-plan layer | Claim | PoC verdict |
|---|---|---|
| **Layer 1** — section-threshold note-gate | drop lands in a mislabeled "intro" → gated at 0.68 → silenced | **CONFIRMED & it's the real silent-drop cause.** `generate.py:1595` `_SECTION_THRESHOLDS` (intro 0.68 / outro 0.72) can silence any region the energy detector mislabels. Pure inference hack — removable with **no rebuild**. |
| **Layer 2** — BPM-grid blurs/quantizes timing | grid can't place off-grid notes | **NOT SUPPORTED.** 94–99 % of human notes are on-grid. The grid is fine; the real WHEN issue is *selection* (flat Stage-1 probs) + the Layer-1 gate. |
| **Layer 3** — no per-instrument structure for WHAT | blended `other` stem → incoherent diagonal swings | **CONFIRMED & fixable with transcription as Stage-2 conditioning** (bass + lead contour), without touching WHEN. |

---

## Addendum (2026-06-03) — per-instrument events as a STRUCTURE signal

User pushback: the core case for V8 isn't off-grid timing, it's that **distinguishing
instruments lets the model read the song's rise/drop structure** (which should drive
WHEN/density, not just WHAT). Tested directly (`scripts/v8_poc_structure.py`, 12 songs,
2 s windows): per-song Spearman of each signal vs human note density.

| signal | mean r vs human density |
|---|---|
| **drum event density** | **+0.408** |
| total event density | +0.379 |
| kick density | +0.344 |
| **section-detector rank (V7's current signal)** | **+0.271** |
| bass activity | +0.139 |
| lead activity | +0.130 |
| #active stems | +0.069 |

**Confirmed: per-instrument event activity predicts where humans map notes ~50% better
than the energy-percentile section detector (0.41 vs 0.27).** The win is concentrated in
**drum/kick** activity — which V7 currently sees only as mean-pooled MERT, never as explicit
onset density. Bass/lead are weak *density* predictors (they belong to WHAT/direction, not
WHEN). Magnitudes are moderate (mapper-subjectivity ceiling), but the ordering is consistent.

**Implication:** this is a stronger, broader case than my original "Stage-2 contour cohesion."
Per-instrument event features should feed **Stage 1 (WHEN/density)** too, and can **retire the
section-detector + `_SECTION_THRESHOLDS` stack** in favour of a learned density signal — the
same stack whose mislabeling caused the silent-drop. This does NOT revive the continuous-time
backbone (still no-go); the BPM grid stays as the output timing lattice.

**Do NOT over-index on drums (user correction, accepted).** The drum dominance here is a
rock-leaning sample; for EDM the bass/synth *layering* carries the structure and the kick is
often a steady four-on-the-floor. The generalizable input is the **full per-instrument layering
vector** (all stems) with the model weighting it per song/genre — not a single hand-picked stem.
Stratify future tests by `genre` and verify EDM explicitly.

### Consistency via a retrieval KEY swap (the highest-leverage idea)
The model already has long-range memory: **song-memory cross-attention attends over all ~150
phrase fingerprints**. The weakness is the *key* — those fingerprints are **mean-pooled MERT**
(timbre average), too coarse to recognize "the 14 s drop == the 4:00 drop". Swap the key for a
**per-instrument layering + pitch-contour fingerprint** so analogous moments match and get
consistent notes — the original North-Star failure ("same chorus, inconsistent patterns").

`ctx_len=16` is the *local* mechanism and should stay small: the ablation showed ctx16 > ctx0 >
ctx32 with **ctx32 collapsing on the final chorus (drift)**. So raw long context is the wrong
tool; long-range consistency belongs to **sparse, content-addressed retrieval on a good key**
(DeepSeek MLA/NSA-style). Validate the key swap before retraining — see TASK 1 in `TODO.md`.

## Recommendation — Scoped V8 (updated)

**Do NOT** build V8-1..V8-3 as written (continuous-event WHEN backbone + event-selector +
deleting the grid label path). The evidence says the grid is not the problem.

**DO**, in priority order:
1. **Remove the section-threshold note-gate (Layer 1).** Stop letting a mislabeled section
   silence real onsets. Cheap inference change; directly fixes the headline complaint.
   (Implemented this session — see `generate_v7_level(section_gate=...)`.)
2. **Orthogonal cohort filter** (Expert / NPS 4–8) — independent, addresses "NPS too high /
   for-sport swings." (Implemented this session.)
3. **Scoped V8 = per-instrument events as INPUT to BOTH stages, BPM grid kept for output.**
   This is the real V8, centered on the user's "distinguish instruments" insight:
   - **Stage 1 (WHEN/density):** add the **full per-instrument event-density vector** (all
     stems; drum/kick is the strongest predictor *on this rock-leaning sample* at r=0.41 vs the
     detector's 0.27, but EDM will weight bass/synth — let the model decide). Goal: learn
     rise/drop/breakdown density and **retire `_SECTION_THRESHOLDS` + the energy-percentile
     section detector** (the silent-drop's root mechanism) entirely.
   - **Stage 2 (WHAT/cohesion):** add `bass` line + `other` lead **pitch contour** as a
     conditioning channel so swing directions follow the melody. Measure a contour-follow
     metric vs V7.
   - **Stage 3 (consistency):** swap the song-memory retrieval **key** from mean-MERT to the
     per-instrument layering+contour fingerprint, so repeated sections match and replay
     consistent notes. Gated on the TASK-1 validation. Keep `ctx_len=16` for local flow.
   - Keep the 1/16 BPM grid as the output lattice (the off-grid rebuild stays no-go).
   - Sequence (see `TODO.md` tasks): TASK 0 eval version_12 → TASK 1 validate the retrieval key
     → TASK 2 Stage-1 density (kills the section detector) → TASK 3 Stage-2 contour →
     TASK 4 retrieval-key swap → TASK 5 (stretch) sparse long-range attention.

**Net:** the user's instinct "we digest the song wrong" is **right about the part that
matters** — the missing per-instrument structure is a real flaw, and it's a better signal
than V7's hand-tuned section detector for BOTH density (drums, +0.41 vs +0.27) and direction
(lead/bass contour). What's *wrong* is only the specific "BPM grid can't represent the timing"
premise (Layer 2): the grid already holds 94–99% of human note timing, so per-instrument events
should be **input/conditioning around the existing grid**, not a continuous-time replacement of it.

---

## Addendum — 2026-06-05: scoped-V8 Stage-1 + retrieval-key results (BOTH NEGATIVE)

Overnight chain (`scripts/overnight_chain_2026-06-04.sh`, run after a 06-04 power cut; preprocess
resumed cleanly to 5319/5320, 0 corrupt .pt). Summary: `logs/overnight/chain_2026-06-04_summary.log`.

### TASK 1 (layering vs mean-MERT retrieval key) — DEAD, well-powered
`scripts/v8_poc_retrieval_key.py --n 60 --difficulty Expert` → `outputs/v8_poc/retrieval_key_2026-06-04.json`.
60 songs, 25,950 labeled pairs (vs the prelim's 9).

| key | overall AUC | electronic (15 songs, 10k pairs) |
|---|---|---|
| mean-MERT | **0.8484** | **0.864** |
| layering | 0.8237 (Δ −0.025) | 0.800 (Δ −0.064) |

Layering is worse overall and worst on electronic — the genre it was predicted to win. The
preliminary "layering 0.902 > 0.893 / EDM win" was an underpowered (9-pair) artifact. **→ TASK 4
(swap the song-memory key to a layering fingerprint) is KILLED.** Chorus-consistency does not come
from a layering retrieval key; the weakness of mean-MERT is real but layering is not the fix.

### TASK 2 (Stage-1 `--use-instr`) — null on val_f1; real DoD still untested
`logs/beat_classifier/version_7` (d512/4L, Expert+ExpertPlus, require_instr = 166,998 windows).
Best `val_f1_avg_tol = 0.600 @ epoch 0`, never improved (early-stopped ep8) vs version_4 baseline
0.603. Third independent confirmation the ~0.60 per-slot metric is a mapper-subjectivity ceiling
(Run 6 struct = 0.598, instr = 0.600 — input features don't move it).

**Caveat:** `val_f1_avg_tol` is the WRONG yardstick for TASK 2 — it's per-slot binary accuracy,
which has repeatedly anti-correlated with alignment/density quality. TASK 2's actual DoD is
inference-side (does generated per-section NPS track human density with the section gate OFF;
structure-corr ≥ 0.41). That was **never tested** — instr is still not wired into `generate_v7_level`.
So TASK 2 is **inconclusive, not refuted.** Next decision: run the inference-side density test
(the only unfalsified piece of the per-instrument thesis), pivot to TASK 3 (pitch contour for
WHAT-cohesion, untouched), or accept the ceiling and keep the `section_gate="loud_only"` fix that
demonstrably cured the silent-drop. Do NOT delete `_SECTION_THRESHOLDS` until the inference test runs.
