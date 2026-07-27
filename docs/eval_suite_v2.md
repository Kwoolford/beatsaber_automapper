# Evaluation Suite v2 — design doc

**Written:** 2026-07-26
**Goal (Kyle's, verbatim):** *"Continue to update evaluation suite so I do not have to be the
judge anymore on whether our training is working. You have significantly more collective
knowledge but are handicapped by evaluation suite. I want to get to a point where our
evaluation suite is so good I could give an agent a set of instructions to build it by itself
without machine learning, which has the benefit of you being able to audit the architecture."*

Three requirements fall out of that:

1. **Replace the human judge.** The suite must decide "is this map good?" without Kyle playing it.
2. **Be prescriptive, not just descriptive.** A metric that says *"row_conc is 0.94"* tells you
   nothing about what to do. A spec an agent can build against must say *what a correct map does*.
3. **Be auditable.** One coherent scoring system with stated assumptions — not a pile of scripts.

---

## 1. Audit of the current suite (evidence, not opinion)

Built `scripts/audit_eval_suite.py`: score real human maps, our production maps, and four
**degenerate control maps** with the current scorecard. This is how you test any classifier —
on cases it *must* fail. Controls keep the human note *times* and vary what is placed:

| control | what it isolates |
|---|---|
| `random` | uniform-random (x, y, dir). Maximal variety, zero intent. |
| `shuffled` | human map with its (x, y, dir) triples **permuted** — byte-identical marginal distributions, all sequencing destroyed. |
| `metronome` | one cell, one direction, constant interval. |
| `zigzag` | the old V7 failure mode (bottom row, 2 columns, alternating) — a regression control. |

Result (12 human maps, `outputs/eval_audit_2026-07-26.json`):

| cohort | row_conc | col_conc | grid_cov | dir_ent | monotony | pat_rep | nps | viol |
|---|---|---|---|---|---|---|---|---|
| human | 0.543 | 0.308 | 0.986 | 0.759 | 0.450 | 0.002 | 4.15 | 0.0 |
| prod (ours) | 0.432 | 0.286 | 0.965 | 0.799 | 0.422 | 0.000 | 6.23 | 0.0 |
| random | 0.358 | 0.268 | **1.000** | **0.997** | 0.350 | 0.009 | 4.15 | 9.3 |
| shuffled | **0.543** | **0.308** | **0.986** | **0.759** | 0.464 | 0.060 | 4.15 | 51.8 |
| metronome | 1.000 | 1.000 | 0.083 | 0.000 | 0.994 | 1.000 | 5.88 | 739.2 |
| zigzag | 1.000 | 0.501 | 0.167 | 0.315 | 0.657 | 0.000 | 4.15 | 180.9 |

### Finding 1 — the ranking metric has run out of resolution

`h_dist` (the scalar `eval_sweep` picks winning arms by) is
`mean |arm − human| / human` over exactly five keys: `row_conc, col_conc, grid_coverage,
dir_entropy, monotony`. Measured:

```
prod(ours)     h_dist 0.033     <-- ranks BETTER than real human maps
human          h_dist 0.060
shuffled       h_dist 0.067     <-- a destroyed map, ~= a real one
random         h_dist 0.162
zigzag         h_dist 0.746
metronome      h_dist 1.346
```

**Our generated maps are "more human than human" on our own ranking metric.** That is the
signature of a saturated, Goodharted metric: we tuned the generator until it matched the target
statistics, and now the metric cannot order anything we care about. It is why the last several
sessions produced ever-smaller wins that Kyle could not perceive — the numbers moved, the maps
did not.

### Finding 2 — five of the seven map metrics are blind to sequencing

All five `h_dist` keys are **permutation-invariant**: they are histograms over notes, so
shuffling every note's position and direction *cannot change them*. That is why `shuffled`
reproduces the human row/col/grid/dir/nps numbers exactly. Of the whole map-only scorecard,
only `pattern_repeat` (0.060 vs 0.002) and the swing simulator (51.8 violations vs 0) notice
that the map has been destroyed.

**The swing simulator is doing nearly all the real work.** Everything else is a
marginal-distribution matcher.

### Finding 3 — "more diversity = more human" is false, and the suite encodes it

`random` beats the human maps on `grid_coverage` (1.000 vs 0.986) and `dir_entropy`
(0.997 vs 0.759), and has "better" `row_conc` and `monotony` under the suite's own
`BETTER = {"grid_coverage": "high", "dir_entropy": "high", "row_conc": "low", ...}` arrows.
Human mapping is not high-entropy; it is *structured* — a small vocabulary of idioms, deployed
deliberately. The anti-repeat lever promoted on 2026-07-23 pushes toward entropy, which is the
right direction only because we started far *below* human. There is no headroom left there, and
continuing to push is actively wrong.

### Finding 4 — three parallel scoring systems (the auditability problem)

"How good is this map?" has three different answers in this repo:

| system | status |
|---|---|
| `scripts/map_metrics.py` + `swing_sim` + `eval_sweep.py` | the live loop |
| `src/beatsaber_automapper/research/metrics.py` (`composite_score`, 60% playability / 40% style, 8 weighted terms) | used by `runner_v7`/`leaderboard`, diverged from the live loop |
| `src/beatsaber_automapper/evaluation/{map_quality,playability}.py` (`evaluate_map`, `check_playability`, its own `_check_parity`) | dead relative to the live loop, still exported from `evaluation/__init__` |

There are two independent parity implementations (`playability._check_parity` and `swing_sim`),
and the superseded one is still the package's public API.

---

## 2. Design principles for v2

1. **Every metric must pass the control battery.** A metric earns its place only if human maps
   beat `random`, `shuffled`, `metronome`, and `zigzag`. `audit_eval_suite.py` is the gate; run
   it whenever a metric is added or changed.
2. **Sequence-aware by default.** If a metric is invariant to shuffling the note order, it can
   only ever be a sanity check, never a quality judgement. Marginals are necessary, not
   sufficient — keep them as *guards* (is anything wildly out of range?), not as the score.
3. **Prescriptive form.** State each axis as a rule a mapper must satisfy, with the human
   distribution as evidence. "Notes fall on the beat grid at 1/4, 1/8, 1/12 or 1/16" is
   buildable; "subdivision_entropy = 0.61" is not.
4. **Compare against distributions, not point targets.** A point target invites Goodharting
   (Finding 1). Score against the human *distribution* (is this value inside the human range?)
   so "more extreme than human" stops being rewarded.
5. **One scoring system.** Consolidate the three above into one module with one entry point.

---

## 3. Proposed axes for v2

Ordered by expected value. Axes 1–3 are the ones that would actually let the suite replace the
human judge; they are all sequence-aware, and all four controls fail them by construction.

### A1 — Flow / ergonomics (beyond legality) — ✅ **BUILT 2026-07-27**
`src/beatsaber_automapper/evaluation/flow.py`, calibrated by `scripts/calibrate_flow.py`
(200 human maps, median/MAD reference at `outputs/flow_human_reference.json`), tested in
`tests/test_flow.py` (10 tests).

`swing_sim` answers *"is this parity-legal?"*. Flow answers *"is this comfortable?"*. Metrics,
all computed from *consecutive* swings so none can be satisfied by matching marginals:
`angle_change` (wrist rotation between swings, in the parity-aware frame — a clean down/up
stream is 0°), `angle_harsh_frac` (fraction of transitions above 90°), `travel` (grid distance
per second between swings), `ebpm_burst` (95th-pct burst rate, converted to **wall-clock**
swings/minute — `swing_sim` reports it per beat, which is tempo-blind), plus `crossover` and
`handedness` as order-invariant guards.

**Human reference (200 maps):** `angle_change` 19.1° (MAD 4.5), `angle_harsh_frac` 0.004,
`travel` 4.0/s, `crossover` 0.218, `handedness` 0.012, `ebpm_burst` 250/min.

**DoD — MET.** Control battery (`--n 12`, reference held disjoint from the audit cohort via
`calibrate_flow.py --skip 32`), ranked by `flow_gap`:

| cohort | flow_gap | min_spread |
|---|---|---|
| human | **0.21** | 0.52 |
| prod (ours) | **0.89** | 0.44 |
| shuffled | 1.54 | 0.51 |
| zigzag | 2.57 | 0.00 |
| metronome | 3.21 | 0.00 |
| random | 11.68 | 0.19 |

`flow_dist`/`flow_gap` is the **first metric in the suite to catch all four controls**, and the
first to rank our maps *below* human rather than above.

**Two design lessons, both learned the hard way here:**

1. *A per-map distance to the human median re-creates the h_dist failure.* The first version
   scored `flow_dist` per map; our maps came out at 1.37 vs human 1.54 — "more human than
   human" again — because a mode-collapsed cohort sits nearer the median than typical human
   maps do. The fix is `flow.cohort_comparison()`: compare **distributions**, reporting per
   metric a `shift` (median offset in human MADs) *and* a `spread` (cohort MAD / human MAD).
   Mode collapse is invisible to shift and obvious in spread. **Rank generators by `flow_gap`,
   not by per-map `flow_dist`.**
2. *Order-invariant terms dilute a sequence-aware composite.* `crossover` and `handedness` are
   unchanged by the `shuffled` control by construction, so including them in the composite
   weakened exactly the detection this axis exists for. Only `SEQUENCE_KEYS` enter `flow_gap`.

**Real quality gaps this exposed in our production maps** (invisible to the old scorecard):
- **`travel` shift +2.48 human-MADs** — our hands move ~50% further per second than human
  hands (6.0 vs 4.0). The single most actionable flow defect.
- **`crossover` 0.000 vs human 0.218** — `enforce_color_separation` in the postprocess forces
  red-left/blue-right, so our maps *never* cross over; human mappers do on ~22% of notes.
- **`angle_harsh_frac` spread 0.44** — under-dispersed; our maps are uniformly smooth where
  human maps vary.

*Prescriptive form:* successive swings of a hand continue the current swing plane (median
rotation ~19°, harsh >90° transitions on <1% of transitions); a hand travels ~4 grid-units/sec
between swings; hands stay on their own side except for deliberate crossovers on ~20% of notes;
burst rate stays near ~250 swings/min.

### A2 — Rhythm / beat-grid — ✅ **BUILT 2026-07-27** — ★ our largest measured gap ★
`src/beatsaber_automapper/evaluation/rhythm.py`, calibrated by `scripts/calibrate_rhythm.py`
(200 human maps), tested in `tests/test_rhythm.py` (7 tests).

Nothing in the suite measured note **times** — every existing metric is computed over note
*attributes*. That turned out to be the biggest blind spot in the whole suite.

**Result: `rhythm_gap` 2.30 for our maps vs 0.31 for human** — far worse than the flow axis
(0.89). Our maps are metronomic: `pulse_stability` +2.17 human-MADs, `ioi_cond_entropy` −2.92,
`ioi_switch_rate` −1.81, `min_spread` 0.31 (collapsed). 75% of our inter-onset intervals land on
exactly 1/8, against 41% for humans.

**Diagnosis — it is hand LOCKSTEP, not the note grid.** Our *per-hand* intervals are already
human-like. The defect is that **our two hands fire simultaneously on 85.6% of beats, against a
human rate of 17.5%**: the two probability channels are driven by the same audio and select the
same slots, so the union rhythm collapses onto one repeated spacing. (The first hypothesis —
that the NMS min-distance in `_density_aware_select` imposed a 1/8 floor — was wrong; per-hand
1/16 intervals are ~0.6% in *both* human and our maps.)

Metrics: `pulse_stability`, `ioi_cond_entropy`, `ioi_switch_rate` (sequence-aware, in the
composite); `dominant_share`, `ioi_entropy`, `offgrid_frac` (guards). Human reference:
pulse 0.551, cond-entropy 0.536, switch-rate 13.7/100 notes, dominant share 0.509.

Deliberately *not* measured: on-grid purity. V7 emits on a 1/16 grid by construction and human
maps are 94–99% on that same grid, so it cannot discriminate.

**The control battery needed extending.** Every pre-existing control preserves note times, so
none of them can test a rhythm metric — `random`, `shuffled` and `zigzag` score *identically to
human* on A2, which is correct rather than blind. Added `timing_random` (times randomised on the
grid) and `timing_jitter` (times nudged off-grid); A2 catches both hard (6.30 / 8.90) plus
`metronome` (5.21). Blind-spot reporting in `audit_eval_suite.py` is now **axis-aware**: each
metric is only judged against the controls that attack what it measures.

*Prescriptive form:* hold a pulse about half the time and break it the rest
(`pulse_stability` ≈ 0.55); change rhythmic gear ~14 times per 100 notes; play both hands
together on ~18% of beats, not 86%.

### A3 — Pattern vocabulary / idiom — ✅ **BUILT 2026-07-27**
`src/beatsaber_automapper/evaluation/idiom.py`, mined by `scripts/calibrate_idiom.py`,
tested in `tests/test_idiom.py` (7 tests).

**The premise held.** Over 181 human maps, **130,395 per-hand transitions collapse to 2,510
distinct idioms**; the top 200 cover **74.8%** of everything human mappers do, the top 500
cover 89.5%, the top 1000 cover 96.4%. Human mapping is a small vocabulary deployed
deliberately — the direct rebuttal to the "more diversity = more human" assumption in §1
Finding 3. **This vocabulary is the artifact a rule-based mapper can sample from**, which is
the project goal; it is checked in at `outputs/idiom_vocab_human.json`.

An idiom is one hand's transition `(dx, dy, dir_from, dir_to, dt_class)`, with dt bucketed to
{stack, 1/16, 1/8, 1/4, slow} — the same geometric move is a different pattern at a different
speed. Metrics: `idiom_coverage` (share drawn from the top-500), `idiom_top50` (share from the
50-idiom core), `idiom_jsd` (is the *mix* human?), `idiom_entropy` (guard).

Control battery: human **0.50**, prod **1.84**, shuffled 5.45, zigzag 5.63, random **9.69**,
metronome 10.73. A3 is the only axis that correctly ranks a uniform-random map near the bottom.
Vocabulary and reference are mined from **disjoint** corpus slices so a map's own idioms cannot
inflate its coverage.

### (original plan for A3, kept for context) Pattern vocabulary / idiom
Human maps are built from a small vocabulary of recognizable idioms (streams, stacks, towers,
sliders, crossovers, doubles) — not from maximum entropy. Build the idiom inventory from the
human corpus (n-gram over (Δposition, Δdirection, Δtime) triples), then measure what fraction of
a map's transitions are drawn from the human idiom set, and whether the idiom *mix* matches.
This is the metric that would correctly rank `random` below both human and our maps, which no
current metric does.
*Prescriptive form:* the idiom inventory **is** the mapper's building blocks — this axis makes
the non-ML mapper Kyle described directly buildable.

### A6 — Hand-role division — ✅ **BUILT 2026-07-27** — ★ our worst axis ★
`src/beatsaber_automapper/evaluation/handrole.py`, calibrated by `scripts/calibrate_handrole.py`
(200 human maps).

**Found by reading, not by statistics.** Putting a generated map next to its human counterpart in
`scripts/map_view.py` showed that human mappers give **one hand the lead within a passage** — a
sustained run — while the other punctuates, then they swap. Ours run both hands at identical
density throughout.

| metric | human | ours |
|---|---|---|
| `role_asymmetry` (per 2 bars) | 0.115 | 0.031 |
| `role_swap_rate` | 0.461 | 0.269 |
| `role_run_len` (guard) | 1.364 | 1.05 |
| **`handrole_gap`** | **0.34** | **3.50** |

**"Globally balanced, locally lopsided."** Both cohorts are near-perfectly balanced over a whole
song — `flow.handedness` is 0.012 for both, so the existing hand metric sees nothing. Human maps
get that balance by rotating the lead; ours by splitting every bar evenly. Balance at *every*
scale is the unnatural thing.

Control battery: human **0.34**, prod **3.50**, random 2.64, shuffled 2.43, zigzag 2.33,
metronome 21.50. **Our maps are further from human hand-role behaviour than a uniformly random
map is** — the largest single-axis defect measured anywhere in this project.

`role_run_len` is a **guard, not a composite driver**: notes on the same beat are ordered
L-then-R, so a map whose hands fire simultaneously has run length ~1.0 by construction, making it
largely a restatement of the A2 simultaneity finding rather than independent evidence.

### A4 — Musical-role correctness
Which instrument is the map following — kick, snare, vocal, lead? Human mappers follow one
layer and switch at section boundaries. We already compute per-stem onsets (Demucs) and per-slot
instrument features, so this is mostly wiring. This is the root of the original "for-sport"
complaint: mapping *nothing in particular*.

### A5 — Structural self-consistency — ❌ **NEGATIVE RESULT 2026-07-27, not shipped**
`src/beatsaber_automapper/evaluation/structure.py` exists but is **dormant** — it does not
discriminate, so by the suite's own rule it does not earn a place on the scorecard.

Premise tested: human maps echo themselves at bar-aligned lags (8/16/32 bars) more than at
arbitrary lags. **False as measured.** Across three similarity tokens (rhythm-only, rhythm+hand,
full note tuple), `struct_lift` was ≈ 0 for *every* cohort including human (+0.001 / +0.007 /
+0.003; metronome −0.002).

Why the shortcut fails: you cannot assume *where* repeated sections are. Song structure does not
sit at fixed bar multiples across genres, so a fixed-lag probe cannot find it. **A5 needs
audio-derived section boundaries** — identify which sections actually repeat, then ask whether
the map echoes them. The machinery already exists (`detect_sections`, phrase boundaries in
generation); re-spec on top of it rather than on fixed lags.

Worth keeping from the attempt: with a rhythm+hand token our maps' `struct_recall` is 0.587
against human 0.329 — we repeat ourselves far more than humans do. But with lift ≈ 0 that
repetition is *uniform*, not structural, making it a restatement of the A2 rhythm finding rather
than a new axis.

### A6 — Difficulty calibration & pacing
Does the map match its claimed difficulty by community norms, and does it have a shape
(intro → build → drop → rest)? Partially covered by `density_corr`; the stamina/rest-period
dimension is not.

### Guards (keep, demote from "score" to "sanity check")
`row_conc`, `col_conc`, `grid_coverage`, `dir_entropy`, `nps`, `n_notes` — retain as range
checks against the human distribution. They catch gross breakage (metronome, zigzag) cheaply.
Stop using them as the objective; retire `h_dist` as the ranking scalar once A1–A3 exist.

---

## 3b. The no-ML mapper — does the suite actually specify one? (2026-07-27)

`scripts/rule_mapper.py` is the direct test of the project goal. It contains **no model, no
checkpoint, no learned weights** — only the measured rules above, plus the mined idiom
vocabulary. Given onset times and a BPM it produces a map by: alternating hands with ~17.5%
doubles (A2), sampling the human idiom vocabulary conditioned on the hand's previous note (A3),
preferring travel near 4 grid-units/sec (A1), alternating parity with `swing_sim`'s 0.30 s
wrist-break floor, and keeping hands on their own side except ~20% crossovers.

**Scored on the same 24-map cohort protocol:**

| onsets from | flow | rhythm | idiom | parity |
|---|---|---|---|---|
| our prod maps | 1.00 | 2.41 | **1.22** | 0 viol |
| held-out human maps | 1.45 | **0.25 PASS** | **0.99** (bar 1.00) | 0 viol |
| *(our ML model, for reference)* | *0.81* | *2.41* | *1.84* | *0 viol* |

Three findings:

1. **The suite is prescriptive enough to build a competitive non-ML mapper.** With correct
   onsets it reaches human-range rhythm and near-bar idiom usage with zero parity violations,
   and it beats our trained model on the idiom axis (1.22 vs 1.84) — from rules alone.
2. **Rhythm is inherited entirely from the onset layer.** The same rule mapper scores 2.41 on
   prod onsets and 0.25 on human onsets, having changed nothing about how it places notes.
   This independently confirms the A2 diagnosis: the rhythm gap lives in Stage-1 selection and
   hand assignment, *not* in the pattern layer.
3. **The suite does NOT yet prescribe variety.** Every rule-based cohort is mode-collapsed
   (spread 0.13–0.30 against a 0.35 bar). Widening the per-note sampling makes it *worse*, not
   better — width 6→60 degrades flow 1.45→1.77 and idiom 0.99→1.37, drops spread 0.30→0.17, and
   introduces parity violations. Per-note randomness is not cohort diversity: sampling further
   down a ranked candidate list picks systematically worse options rather than different ones.
   Real human variety is **map-level style** (different mappers, different songs, different
   characteristic vocabularies), and nothing in the suite currently expresses that. **This is
   the most important open gap in the suite.**

## 4. Build order

1. `audit_eval_suite.py` — **DONE** (2026-07-26). The gate every new metric must pass.
2. **A1 flow/ergonomics** — **DONE** (2026-07-27). DoD met; see above.
3. **A2 rhythm/grid** — cheap, map-only, no audio.
4. **A3 idiom vocabulary** — needs a corpus pass over `data/raw`; unlocks the non-ML mapper.
5. **Consolidate** the three scoring systems into one module + one report; delete or clearly
   archive the dead ones.
6. **A4–A6** once 1–5 hold.

**DoD for the suite as a whole (the thing Kyle asked for):** the suite ranks
`human > prod > random/shuffled > zigzag/metronome` on its composite, with the human cohort
separated from every control by more than the human cohort's own spread — *and* every
individual axis states a rule a mapper could be built against.
