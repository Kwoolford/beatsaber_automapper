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

### A1 — Flow / ergonomics (beyond legality) — **highest value**
`swing_sim` answers *"is this parity-legal?"*. It does not answer *"is this comfortable?"* —
the difference between a map that passes and a map that is fun. Measure per hand: angle
continuity between consecutive swings, wrist travel distance, hand crossovers, inward-facing
awkward pairs, and swing-speed (EBPM) stability. Human maps have a tight, characteristic
distribution here; `shuffled` will be wildly outside it.
*Prescriptive form:* "consecutive swings of one hand differ by ≤ X°, except at deliberate
resets; hands do not cross the centre line more than Y times per phrase."

### A2 — Rhythmic placement / beat-grid sanity
Do notes sit on musically meaningful subdivisions (1/4, 1/8, 1/12, 1/16) of the beat, and does
the map keep a *consistent* subdivision within a phrase? Human maps are overwhelmingly on clean
subdivisions and change subdivision at section boundaries. Currently unmeasured — `onset_hit`
only asks "within 50 ms of any onset", which a random-but-dense map passes.
*Prescriptive form:* "quantize to the phrase's dominant subdivision; change it only at
section boundaries."

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
2. **A1 flow/ergonomics** — biggest single gain; extends the already-validated `swing_sim`.
3. **A2 rhythm/grid** — cheap, map-only, no audio.
4. **A3 idiom vocabulary** — needs a corpus pass over `data/raw`; unlocks the non-ML mapper.
5. **Consolidate** the three scoring systems into one module + one report; delete or clearly
   archive the dead ones.
6. **A4–A6** once 1–5 hold.

**DoD for the suite as a whole (the thing Kyle asked for):** the suite ranks
`human > prod > random/shuffled > zigzag/metronome` on its composite, with the human cohort
separated from every control by more than the human cohort's own spread — *and* every
individual axis states a rule a mapper could be built against.
