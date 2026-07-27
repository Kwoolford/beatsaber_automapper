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

### A3 — Pattern vocabulary / idiom
Human maps are built from a small vocabulary of recognizable idioms (streams, stacks, towers,
sliders, crossovers, doubles) — not from maximum entropy. Build the idiom inventory from the
human corpus (n-gram over (Δposition, Δdirection, Δtime) triples), then measure what fraction of
a map's transitions are drawn from the human idiom set, and whether the idiom *mix* matches.
This is the metric that would correctly rank `random` below both human and our maps, which no
current metric does.
*Prescriptive form:* the idiom inventory **is** the mapper's building blocks — this axis makes
the non-ML mapper Kyle described directly buildable.

### A4 — Musical-role correctness
Which instrument is the map following — kick, snare, vocal, lead? Human mappers follow one
layer and switch at section boundaries. We already compute per-stem onsets (Demucs) and per-slot
instrument features, so this is mostly wiring. This is the root of the original "for-sport"
complaint: mapping *nothing in particular*.

### A5 — Structural self-consistency
Repeated musical sections (chorus 1 vs chorus 2) should get *similar but not identical*
patterns. We have song-memory in the model and no metric for it.

### A6 — Difficulty calibration & pacing
Does the map match its claimed difficulty by community norms, and does it have a shape
(intro → build → drop → rest)? Partially covered by `density_corr`; the stamina/rest-period
dimension is not.

### Guards (keep, demote from "score" to "sanity check")
`row_conc`, `col_conc`, `grid_coverage`, `dir_entropy`, `nps`, `n_notes` — retain as range
checks against the human distribution. They catch gross breakage (metronome, zigzag) cheaply.
Stop using them as the objective; retire `h_dist` as the ranking scalar once A1–A3 exist.

---

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
