# Beat Saber Automapper — V7 Plan (MERT + Demucs + Retrieval Architecture)

**Last updated:** 2026-07-27 — ★ HAND-ROLE (A6) IS THE HEADLINE: our worst axis, found by READING ★

## ⏭️ NEXT SESSION — pick up here (written 2026-07-27, autonomous loop)

# ★★ THE ORGANISING DISCOVERY: HANDS HAVE ROLES ★★

Found by **reading a map next to its human counterpart** in the new `scripts/map_view.py` — not
by any statistic, and not by any metric we had. In a human map, **within a passage one hand
carries a sustained run while the other punctuates**, and the two swap that job between passages.
Our maps run both hands at identical density throughout, with no role division at all.

| metric | human | ours |
|---|---|---|
| local asymmetry (per 2 bars) | 0.115 | **0.031** |
| dominant-hand swap rate | 0.461 | 0.269 |
| same-hand run length | 1.364 | 1.05 |
| **`handrole_gap` (A6)** | **0.34 PASS** | **3.50 FAIL** |

**The key insight is "globally balanced, locally lopsided."** Both cohorts are near-perfectly
balanced over a whole song, so the existing `flow.handedness` metric (0.012 for both) sees
nothing. Human maps earn that balance by giving one hand the lead for a stretch then swapping;
ours earn it by splitting every single bar down the middle. **Balance at every scale is the
unnatural thing.**

**A6 is now our worst axis — 3.50 against a human 0.34, and worse than a uniformly random map
(2.64).** On hand-role division our maps are less human-like than noise. Built as
`evaluation/handrole.py`, calibrated on 200 human maps, passes the control battery, wired into
`evaluation/scorecard.py` (which now has four axes; a held-out human cohort passes all four).

**Why this discovery matters beyond the metric:** it is the first thing found by the *direct
reading* channel rather than by aggregate statistics, and it validates the whole
`docs/map_authoring_plan.md` direction. The metrics had been averaging this away for months.

**The lever (`BEAT_HAND_ROLE`, new, default OFF):** reassigns *which hand* plays each
already-selected onset per 2-bar window, leaving onset TIMES untouched, targeting the measured
human reference (asymmetry 0.115, swap 0.461, doubles 0.175). Two bugs already caught in
smoke-testing and fixed: (a) taking the union of the two hands' selections collapsed every
simultaneous double onto one hand and silently deleted ~38% of the notes; (b) giving the lead
hand a *contiguous* block overshot run length to 6.7 (human 1.36) and read as one hand idling —
"carrying a passage" means a majority **share** distributed through alternation, not a solo.
**RUNNING NOW** as part C (`scripts/overnight_2026-07-27c.sh`, arms hr05/hr075/hr10/best/best_hr).

---

# ★ THE UNIFYING PRINCIPLE: GLOBALLY RIGHT, LOCALLY WRONG ★

Three independent findings now share one shape. **Every metric in the original scorecard was a
whole-map histogram, and whole-map histograms are exactly where this generator looks good.**

| | global statistic (looks fine) | local structure (broken) |
|---|---|---|
| sequencing | `h_dist` histograms pass | a *shuffled* map scores like a human one |
| hand balance | `flow.handedness` **0.012 for both** | local asymmetry 0.115 human vs **0.031** ours |
| idiom vocabulary | 238 distinct idioms vs human **219** | 0.861 human vs **0.703** ours per 16-note window |

**Design rule going forward: when adding an axis, measure it inside a window before measuring it
over a map.** The whole-map version will usually look fine and tell you nothing. This is also why
the direct-reading channel keeps finding what the aggregates cannot — `map_view.py` shows local
structure by construction.

Latest instance: with inline idiom annotation, the right hand was visibly alternating between
exactly **two** idioms (`#51 → #50 → #51 → #50`) for bars at a time, while the whole-map idiom
count looked *better than human*. Added as `idiom_local`, which raised our idiom gap 1.84 → 2.34.

---

## Immediate stack, curated around A6

1. **★ NOTHING IS PROMOTABLE. No arm passes, and the levers TRADE AGAINST EACH OTHER. ★**
   All arms re-scored on one consistent metric set (bars: flow 0.50 / rhythm 0.70 / idiom 1.00 /
   handrole 2.00; `*` = passes):

   | arm | flow | rhythm | idiom | handrole | viol | notes |
   |---|---|---|---|---|---|---|
   | prod | 0.81 | 2.41 | 2.34 | 3.50 | 0 | 1375 |
   | tp1 | **0.30\*** | 2.54 | 2.29 | 3.24 | 0 | 1371 |
   | xsep_ext | 0.86 | 2.52 | **1.07** | 3.39 | 0 | 1375 |
   | tp2_xsep | 0.68 | 2.43 | 2.98 | 3.06 | 0 | 1378 |
   | best (tp1+xsep) | **0.46\*** | 2.44 | 2.06 | 3.52 | 0 | 1381 |
   | ib1 | 0.66 | 2.32 | 1.98 | 3.15 | 0 | 1373 |
   | hr05 | 0.61 | **4.05** | **0.59\*** | **2.27** | 0 | 1041 |
   | hr10 | 0.66 | 4.04 | 0.80 | 2.94 | 0 | 1038 |

   **Two corrections to earlier reporting in this file:**
   - `xsep_ext` idiom is **1.07, not 0.30**. The 0.30 was measured *before* `idiom_local` was
     added to A3. Any number recorded earlier in this session against the pre-`idiom_local`
     suite is not comparable — always re-score all arms after changing a metric.
   - `tp1` and `xsep_ext` are **NOT orthogonal**, contrary to what was claimed when they were
     first promoted as a pair. Alone they give idiom 2.29 and 1.07; together **2.06** — the
     travel penalty undoes most of the crossover fix. Levers must be validated *in combination*,
     not assumed to compose.

   `BEAT_HAND_ROLE` **trades rhythm for idiom**: best idiom (0.59) and best hand-role (2.27) of
   any arm, but rhythm 2.41 → 4.05, spread collapses to 0.16–0.25 everywhere, and note count
   drops 24%. Before abandoning it: (a) `_assign_hand_roles` uses a fixed seed for every song,
   which likely explains the spread collapse — vary it; (b) de-doubling changes the union IOI
   distribution, so the budget inflation must add slots that *preserve* the interval mix.

   **RHYTHM IS THE WALL.** It sits at ~2.4 for every arm and no lever improves it.
   `rule_mapper.py` already proved rhythm is inherited *entirely* from the onset layer (2.41 on
   our onsets, 0.25 on human onsets, identical placement code), so this is Stage-1 selection,
   not layout.

2. **★ PART D VERDICT: TEMPO IS NOT THE CAUSE. The rhythm gap is Stage-1. ★**
   Regenerated prod + best with `--true-bpm` (the human map's declared BPM) and compared:

   | cohort | | rhythm | flow | idiom | handrole |
   |---|---|---|---|---|---|
   | all 24 (prod) | detected | 2.41 | 0.81 | 2.34 | 3.50 |
   | | true BPM | **2.37** | 0.71 | 1.85 | 3.23 |
   | **mis-tempo only (n=6)** | detected | 1.96 | 0.73 | 1.94 | 2.84 |
   | | true BPM | **2.13** | 0.69 | 1.85 | 2.95 |
   | correct-tempo (n=17) | detected | 2.54 | 0.93 | 2.41 | 3.42 |
   | | true BPM | 2.38 | 0.96 | 1.76 | 3.25 |

   Rhythm moves 2.41 → 2.37 overall, and **on the songs that actually had wrong tempo it gets
   slightly WORSE** (1.96 → 2.13) — correcting the tempo removes the artificial inflation that
   beat-domain metrics get from tempo error, which gives the honest (worse) number. Fixing tempo
   would make our *measurements* more truthful; it would not fix the maps.
   **⇒ The next GPU night is a Stage-1 onset-selection change, not a tempo model.** The tempo
   defect stays on the backlog as a correctness issue, not as the rhythm fix.

3. **✅ NOISE FLOOR MEASURED — most verdicts survive.** `prod_rep` is a byte-identical config to
   `prod`; decode is stochastic (temp 0.9 / top_p 0.97) and `generate.py` has **no seed flag**,
   so two runs of it bound the noise on every comparison in this file:

   | axis | prod | prod_rep | **noise** |
   |---|---|---|---|
   | flow | 0.71 | 0.74 | **0.03** |
   | rhythm | 2.37 | 2.45 | **0.08** |
   | idiom | 1.85 | 1.76 | **0.09** |
   | handrole | 3.23 | 2.94 | **0.29** |

   **Read arm differences against these, not against a guess.** A difference above ~0.1 on
   flow/rhythm/idiom, or above ~0.3 on handrole, is signal; below is noise.
   This *corrects* the earlier caution in this file that differences under ~0.5 were unresolved —
   the 2.41 → 1.76 idiom swing that prompted it came from the **BPM change** (a different beat
   grid), not from stochastic decode.

   Consequences for today's table: `tp1` flow (Δ0.51) ✅ real; `best` flow (Δ0.35) ✅ real;
   `xsep_ext` idiom (Δ1.27) ✅ real; hand-role's rhythm damage (Δ1.64) and its hand-role gain
   (Δ1.23) ✅ both real. But `prod` vs `best` on handrole (3.50 vs 3.52, Δ0.02) is **noise** —
   the travel/crossover levers do nothing for hand role, as expected.

   **Standing rule: any future arm claim must clear the noise floor for that axis.** Re-run
   `prod_rep` whenever decode defaults change, since the floor is a property of the config.
2. **Work `docs/map_authoring_plan.md` Phase 1→2** (this is now the priority channel, having
   produced both A6 and the tempo bug):
   - annotate each transition inline with its **idiom id + human corpus frequency**
   - mark swing violations and flow outliers inline
   - `--find` queries (every occurrence of an idiom / a violation, with context)
   - **`--vs` time-aligned comparison** against the human map for the same song — note bar
     numbers do NOT align, because 30% of our maps are at the wrong tempo
   - cache per-song stem features so the audio lanes are instant
3. **Phase 3 authoring** — parse the score text back through the existing `export.py` write
   path; compose at the **idiom/phrase level** (a 3-min map is 1300+ notes). Then the `/map`
   skill. DoD: a hand-authored map scores human-range AND plays well to Kyle; any disagreement
   between those two is the next blind spot.
4. **A2 wall-clock guard** — beat-domain rhythm is provably gameable by tempo error.
5. **Map-level style/variety** — still the top *unmeasured* gap (every rule-based cohort
   mode-collapses; per-note randomness makes it worse).
6. **A4 musical-role** (per-stem onsets: which instrument is the map following?) — last unbuilt
   planned axis.

---

## (previous framing, kept for the lever/negative-result record)

**★ THE SUITE NOW JUDGES WITHOUT KYLE ★** `evaluation/scorecard.py` — one command, one verdict.
Validated both ways on disjoint data: a **held-out human cohort PASSES** every axis
(flow 0.13 / rhythm 0.25 / idiom 0.31 vs bars 0.50/0.70/1.00); **current production FAILS all
three** (0.81 / 2.41 / 1.84), parity clean. Axes: A1 flow (`evaluation/flow.py`), A2 rhythm
(`rhythm.py`), A3 idiom (`idiom.py`), all scored by cohort **shift + spread** via `_dist.py`.

**LEVER RESULTS (24-song sweeps, `logs/overnight/flow_levers_2026-07-27.log` + `rhythm_idiom_*`):**
| lever | result |
|---|---|
| `LAYOUT_TRAVEL_PENALTY=1` (`tp1`) | ✅ **flow 0.81 → 0.30 PASS** |
| `COLOR_SEP_MODE=extreme` (`xsep_ext`) | ✅ **idiom 1.84 → 0.30 PASS** |
| `LAYOUT_TRAVEL_PENALTY=4` (`tp4`) | ❌ over-corrects: flow 1.77, **spread 0.00** (all maps identical) |
| `COLOR_SEP_MODE=off` | ❌ overshoots: flow 1.04 |
| `BEAT_HAND_INTERLEAVE` (`il5`/`il7`) | ❌ **rhythm WORSE** (2.99 / 2.81 vs prod 2.41), spread collapses, il7 breaks parity |
| `LAYOUT_IDIOM_BONUS` (`ib*`), `combo` | ⏳ still running |

**⚠️ WHY THE INTERLEAVE LEVER LOOKED GOOD AND ISN'T — READ THIS BEFORE TRUSTING ANY PROBE.**
I designed it from a **single-song probe on 1f333**, which turns out to be one of the two
**half-tempo** songs. A2 measures intervals in the BEAT domain, so on a half-tempo song the
beat-domain intervals are stretched and *manufacture* apparent rhythmic variety. The probe was
measured in a distorted frame. **Rule: validate every lever on the full 24-song set before
believing it. Single-song probes are for smoke-testing the code path, not for evidence.**

**★ NEW DEFECT: 30% OF SONGS GENERATE AT THE WRONG TEMPO ★** (`scripts/bpm_octave_probe.py`)
Found by *reading a map next to its human counterpart* in the new `scripts/map_view.py` — ours
said 94 BPM, the human map said 188. Against human-declared BPM as ground truth, raw librosa
detection is correct on only **16/23**; 2 songs at exactly half tempo, 3 at a 2:3 misread. At
half tempo the finest grid slot is **twice as coarse in real time**, so the fast notes cannot be
represented at all. **And the metrics REWARD it** — mis-tempo maps score better on all three
axes (flow 0.73 vs 0.93, rhythm 1.96 vs 2.54, idiom 1.36 vs 1.91).
- Both fix attempts FAILED (octave rescoring 10/23; conservative doubling 14/23) — the
  hypothesis that the true metrical level has balanced odd/even beat energy is false.
  `detect_bpm` left alone; needs a real tempo model, not a heuristic.
- Added `eval_sweep --true-bpm` (uses the human map's BPM) to remove the confound from
  evaluation. **Not a production fix** — production has no human map.

**Next tasks (highest-value first):**
1. **Harvest the `ib*` / `combo` arms** when part B finishes, then promote. Expected winner is
   **`tp1` + `xsep_ext` + an idiom bonus** — the two proven levers are complementary (one fixes
   flow, one fixes idiom) and orthogonal by construction. **Do NOT include the interleave lever.**
2. **Re-run the sweep with `--true-bpm`** and compare. This is the cleanest available estimate of
   how much of our remaining gap is tempo detection vs map quality. Invalidates the cache, so
   budget a full regeneration.
3. **A2 needs a wall-clock guard.** It is gameable by tempo error (proven above). Add
   seconds-domain interval metrics alongside the beat-domain ones, and a tempo-sanity check.
4. **Hand-role axis (new, from direct reading).** Human maps give the two hands different
   musical jobs within a passage — one carries a sustained run, the other punctuates, alternating
   at 1/16 offsets. Ours run both hands at identical density with no role division. No axis
   measures this. See `docs/map_authoring_plan.md`.
5. **Map-level style/variety** — the top open gap. Every rule-based cohort is mode-collapsed and
   per-note randomness does NOT fix it (widening sampling made everything worse). Human variety
   is map-level style; nothing in the suite expresses it.
6. **A4 musical-role** (per-stem onsets: is the map following kick/snare/vocal/lead?) — the last
   unbuilt planned axis. A5 structural self-consistency is a **documented negative** (see below).

**NEGATIVE RESULTS — do not re-attempt as written:**
- **A5 structural self-consistency**: human maps are NOT more self-similar at bar-aligned lags
  than at arbitrary ones (`struct_lift` ≈ 0 for every cohort incl. human, 3 similarity tokens).
  Needs audio-derived section boundaries, not fixed lags. `evaluation/structure.py` is dormant.
- **BPM octave correction** via onset-energy balance (see above).
- **`BEAT_HAND_INTERLEAVE`** (see above).

**Landmines (in addition to those below):**
- **Validate levers on the full 24-song set**, never a single song (the 1f333 half-tempo trap).
- `flow_dist`/per-map distance is a sanity check ONLY; rank by cohort `*_gap` + spread.
- Keep calibration references DISJOINT from the cohorts they judge (`--skip 32`).
- `scripts/map_view.py` reads a map as a text score (hands side by side, stem lanes) — the
  independent channel for auditing when metrics lie. It already found the tempo bug.
- `scripts/rule_mapper.py` is a no-ML mapper built from the suite's rules; on human onsets it
  passes rhythm (0.25) and nearly passes idiom (0.99). Useful as a baseline and as a test of
  whether the suite is prescriptive.

---

## (superseded) NEXT SESSION — written 2026-07-27 (A1 only)

**State:** no jobs running, GPU idle (this axis is CPU-only). Nothing to resume. Committed + pushed.

## 2026-07-27 — ★ A1 FLOW/ERGONOMICS BUILT, DoD MET ★ (first metric to catch all four controls)

Built axis A1 of the v2 eval suite (`docs/eval_suite_v2.md` has the full write-up):
`src/beatsaber_automapper/evaluation/flow.py` + `scripts/calibrate_flow.py` +
`tests/test_flow.py` (10 tests). `swing_sim` says "parity-legal"; flow says "comfortable".

**Control-battery result (the DoD), ranked by `flow_gap` — human < prod < every degenerate:**

| cohort | flow_gap | min_spread |
|---|---|---|
| human | **0.21** | 0.52 |
| prod (ours) | **0.89** | 0.44 |
| shuffled | 1.54 | 0.51 |
| zigzag | 2.57 | 0.00 |
| metronome | 3.21 | 0.00 |
| random | 11.68 | 0.19 |

**First metric in the suite to catch all four controls, and the first to rank our maps BELOW
human** (h_dist still ranks prod 0.038 *ahead of* human 0.060 — that saturation is unchanged
and is why A2/A3 still matter).

**Two design lessons — read these before building A2/A3:**
1. **Rank generators by cohort `flow_gap`, NOT per-map `flow_dist`.** The first version scored
   per-map distance-to-human-median and our maps came out at 1.37 vs human 1.54 — "more human
   than human", the exact h_dist failure reproduced in a brand-new metric. Cause: a
   mode-collapsed cohort sits *nearer the median* than typical human maps do. Fix =
   `flow.cohort_comparison()`, which reports per metric a `shift` (median offset in human MADs)
   AND a `spread` (cohort MAD / human MAD). Mode collapse is invisible to shift, obvious in
   spread. **Any future axis must be scored this way.**
2. **Order-invariant terms dilute a sequence-aware composite.** `crossover`/`handedness` are
   unchanged by the `shuffled` control by construction; including them in the composite weakened
   the very detection the axis exists for. Only `flow.SEQUENCE_KEYS` enter `flow_gap`.

**Real quality gaps in our production maps that the old scorecard could not see:**
- **`travel` +2.48 human-MADs** — our hands move ~50% further per second than human hands
  (6.0 vs 4.0/s). Most actionable flow defect. Do NOT act on it yet (see "don't tune blind").
- **`crossover` 0.000 vs human 0.218** — `enforce_color_separation` in the postprocess forces
  red-left/blue-right, so our maps *never* cross over; humans do on ~22% of notes.
- **`angle_harsh_frac` spread 0.44** — under-dispersed; uniformly smooth where humans vary.

**Next tasks (highest-value first) — `docs/eval_suite_v2.md` §4:**
1. **A2 — rhythm / beat-grid sanity.** Are notes on clean subdivisions (1/4, 1/8, 1/12, 1/16),
   consistent within a phrase? Cheap, map-only, no audio. Must pass the control battery and be
   scored via `cohort_comparison`-style shift/spread.
2. **A3 — pattern-idiom vocabulary.** Mine the human corpus for idiom n-grams over
   (Δposition, Δdirection, Δtime); score what fraction of a map's transitions are human idioms.
   **This is the axis that makes Kyle's non-ML mapper buildable** — the idiom inventory is that
   mapper's building blocks.
3. **Consolidate the three parallel scoring systems** (doc §1 Finding 4).
4. **Only after A2+A3:** act on the `travel`/`crossover` gaps above. Fixing them now would be
   tuning against a single axis — the same mistake that saturated h_dist.

**Landmines (in addition to the ones below):**
- `flow_dist` (per-map) is a sanity/outlier check ONLY. `flow_gap` (cohort) is the ranking stat.
- The flow reference must stay DISJOINT from the cohort it judges: `calibrate_flow.py --skip 32`
  skips the head of the same seed-0 shuffle `audit_eval_suite.py` draws its human cohort from.
  Re-running calibration without `--skip` silently makes the human score in-sample flattery.
- `swing_sim.Swing` gained `x/y/end_x/end_y` (additive, defaults) because flow needs positions;
  `_swing_ebpm_p95` is swings-per-BEAT (tempo-blind) — flow multiplies by bpm.

**★ KYLE'S STRATEGIC REDIRECT THIS SESSION (supersedes the old ship-it/step-back fork) ★**
Asked whether to ship / try the diversity-reg fine-tune / step back, Kyle chose none of them:

> *"Continue to update evaluation suite so I do not have to be the judge anymore on whether our
> training is working. You have significantly more collective knowledge but are handicapped by
> evaluation suite. I want to get to a point where our evaluation suite is so good I could give an
> agent a set of instructions to build it by itself without machine learning, which has the benefit
> of you being able to audit the architecture as well."*

**The work is now the EVALUATION SUITE, not the generator.** Full design doc:
**`docs/eval_suite_v2.md`** (read this first next session). Two things landed this session:

1. **Late-song collapse CLOSED.** Scaled the eval songset 6 → **24 songs** and added a *true
   human-map* comparison to `scripts/eval_late_window.py` (`human_gap`, loaded from the human
   map in `data/raw`, not just the audio-onset reference). Result: **0/24 songs collapse** at
   both final-20% and final-10% tails; mean `late_gap` −0.027/−0.015, mean `human_gap`
   −0.014/−0.013 (gen puts *slightly more* in the tail than the human map), `late_corr`
   +0.38/+0.52. Log `logs/overnight/late_window_scale_2026-07-26.log`. All four original
   complaints are now addressed. Kyle has NOT played recent maps, so this is confirmed by
   metric, not by ear — which is exactly the gap the pivot is about.
2. **Audited the eval suite itself → it is saturated.** New `scripts/audit_eval_suite.py` scores
   human maps, our maps, and four degenerate controls (`random`, `shuffled`, `metronome`,
   `zigzag`). Headline numbers (`outputs/eval_audit_2026-07-26.json`):
   - **`h_dist` — the scalar the sweep ranks arms by — puts our maps (0.033) AHEAD of real human
     maps (0.060).** Textbook Goodhart: we tuned until we matched the target statistics and the
     metric lost all resolution. This explains why recent wins were real on paper and invisible
     to Kyle.
   - **A `shuffled` human map (all sequencing destroyed, 51.8 parity violations) scores h_dist
     0.067 ≈ human 0.060.** All five `h_dist` keys are permutation-invariant histograms, so they
     *cannot* see sequencing.
   - **`random` beats human on `grid_coverage` (1.000 vs 0.986) and `dir_entropy` (0.997 vs
     0.759)** — the suite's "more diversity = more human" assumption is false and we have no
     headroom left there. Do NOT push anti-repeat/diversity further.
   - Only `swing_sim` and `pattern_repeat` catch the shuffled control. **The swing simulator is
     doing nearly all the real work.**

**Next tasks (highest-value first) — from `docs/eval_suite_v2.md` §4:**
1. **A1 — flow/ergonomics metric.** `swing_sim` says "parity-legal"; nothing says "comfortable".
   Per hand: angle continuity between swings, wrist travel, hand crossovers, awkward inward
   pairs, EBPM stability. Extends the already-validated swing_sim. *DoD: passes the control
   battery (human beats all four controls by > the human cohort's own spread).*
2. **A2 — rhythm / beat-grid sanity.** Are notes on clean subdivisions (1/4, 1/8, 1/12, 1/16),
   consistent within a phrase? Cheap, map-only, currently unmeasured (`onset_hit` only asks
   "within 50 ms of *any* onset", which a dense random map passes).
3. **A3 — pattern-idiom vocabulary.** Mine the human corpus for the idiom n-grams, score what
   fraction of a map's transitions are human idioms. **This is the axis that makes the non-ML
   mapper Kyle described buildable** — the idiom inventory is that mapper's building blocks.
4. **Consolidate the three parallel scoring systems** (see doc §1 Finding 4): the live loop
   (`map_metrics`+`swing_sim`+`eval_sweep`), `research/metrics.py::composite_score`, and the
   dead-but-still-exported `evaluation/{map_quality,playability}.py` (which has a second, older
   parity implementation as the package's public API). One module, one entry point.

**Open follow-up questions for Kyle:**
- None blocking. He has not played recent maps; if he does, a specific "this felt bad here"
  report is still the best calibration data for the v2 suite.

**Landmines:**
- `scripts/generate.py` needs `--v7` or it silently uses **untrained** models (0-note garbage).
- Prod decode defaults **temp 0.9 / top_p 0.97**; prod layout has **anti-repeat W=1/S=2.0** baked
  in (`LAYOUT_ANTIREPEAT=0` disables). **Do not tune these further** — see Finding 3 above.
- `map_metrics.map_metrics()` now delegates to `map_metrics_from_seq()` so synthetic control maps
  can be scored without writing zips. Behaviour for zip inputs is unchanged.
- Any new metric must be added to `audit_eval_suite.py`'s battery and pass it *before* being used
  to steer the generator.

---

## (superseded 2026-07-26 by the eval-suite pivot) NEXT SESSION — written 2026-07-23 session 2

**State:** no jobs running, GPU idle. This session PROMOTED the anti-repeat winner (`ar_w1_s2`,
W=1/S=2.0) to the production layout default and confirmed it live; then built the late-song-collapse
metric and found that complaint does NOT reproduce on the eval set (likely already fixed). Nothing to
resume. All code committed + pushed (see bottom of this block).

**What Kyle decided this session (via AskUserQuestion):** promote **W=1/S=2.0** (done); next research
target = **late-song collapse** (built the metric + diagnosed — see below).

**Next tasks (highest-value first):**
1. **Validate the late-song-collapse verdict with Kyle on a real song.** New diagnostic
   `scripts/eval_late_window.py` (per-song: gen vs human note-share in the final tail + tail-only
   density corr) says current prod does NOT collapse late — mean `late_gap` **−0.024** (final 20%) /
   **−0.018** (final 10%), i.e. gen puts *slightly more* notes in the tail than the human map, and
   tail density still tracks the song (late_corr +0.32/+0.46). Strong hypothesis: `section_gate=
   loud_only` + density-select-γ2.5 already fixed it (both post-date the original ~160-164s complaint).
   BUT the 6-song eval set may not include a song Kyle actually perceived collapse on. **Ask Kyle for
   a specific song/timestamp he remembers dying at the end**, run `eval_late_window.py --map <gen.zip>
   --ref <song.ref.npz>` on it; if late_gap stays ≤0.03 there too, mark late-collapse CLOSED. If it
   reproduces, THEN diagnose Stage-1 probs vs Stage-2 context drift in the tail.
   *DoD for a fix (only if it reproduces):* mean late_gap ≤ 0.03 AND late_corr ≥ 0.30, holding
   whole-song density_corr + monotony + viol.
2. **If late-collapse is confirmed closed, the three original complaints are ALL addressed**
   (drop-@-13s via loud_only; flat-density via density-select; monotony/grid-coverage via
   anti-repeat) → this is a "ship it / step back" fork for Kyle. Optional remaining lever = a
   targeted diversity-reg fine-tune (the no-retrain levers are now exhausted), but the renders +
   metrics say we're at ~human on the map-only axes. Get Kyle's judgment on shipped feel.

**Open follow-up questions for Kyle:**
- Give a specific song + timestamp where a map "died at the end" so I can confirm the late-collapse
  metric on it — or is late-collapse subjectively gone for you now?
- With all three original complaints addressed, is the layout good enough to ship, or do you want the
  diversity-reg fine-tune tried first?

**Landmines:**
- `scripts/generate.py` needs `--v7` or it silently uses **untrained** models (0-note garbage).
  eval_sweep passes it; manual runs must too.
- Prod decode defaults are **temp 0.9 / top_p 0.97**; prod layout now also has **anti-repeat W=1/S=2.0
  baked in** as the default in `layout_model.py` (env `LAYOUT_ANTIREPEAT=0` disables it for ablation).
- `eval_sweep.py` `prod` arm now = new production (inherits the baked anti-repeat default); the new
  **`noar`** arm is the pre-promotion baseline for regression.
- The WALL/CHAIN vocab-118 crash fix only touches the **non-v7** `beam_search` path; v7 was never affected.
- `pattern_repeat` is already ~human (~0.0) — don't chase it; the real residual was grid/dir coverage.
- h_dist wanders ~[0.02,0.05] across fresh temp-0.9 draws — read the ON-vs-OFF *gap*, not absolutes.

---

## 2026-07-23 (session 2) — ★ ANTI-REPEAT PROMOTED TO PROD ★ + late-song-collapse metric built → complaint doesn't reproduce

Kyle greenlit (via AskUserQuestion): promote **W=1/S=2.0**, then target **late-song collapse** next.

**DONE — anti-repeat W=1/S=2.0 baked into the production layout default.** In
`src/beatsaber_automapper/models/layout_model.py` the `LAYOUT_ANTIREPEAT`/`LAYOUT_AR_STRENGTH` env
reads now default to **"1"/"2.0"** (were "0"/"0.0"), so the plain v7 generate path gets the sweep
winner without any env flag. Env still overrides (`LAYOUT_ANTIREPEAT=0` = ablation/off). `eval_sweep.py`
ARMS updated: `prod` = new production (inherits the baked default), added **`noar`** (anti-repeat OFF)
as the pre-promotion regression baseline; `ar_w1_s2` kept as the explicit-equals-default sanity arm.

**Rendered W1 vs prod for Kyle** (`outputs/antirepeat_promote_2026-07-23/`, 2 songs, SO TIRED ROCK +
1f333, sent). The beats-114–122 panel is the clearest win: old prod locks into a rigid 2-row
blue-right/red-left loop; W1 uses all 3 rows + varied cut directions, density curve + parity unchanged.

**Regression check PASS** (`scripts/eval_sweep.py sweep --arms prod,noar --force`,
`logs/overnight/promote_regcheck_2026-07-23.log`, `outputs/eval_sweep_cache/leaderboard.json`):

| config | h_dist↓ | grid_cov↑ | dir_ent↑ | col_conc | row_conc | monotony | density (#pass) | viol |
|---|---|---|---|---|---|---|---|---|
| HUMAN | — | 0.96 | 0.80 | 0.29 | 0.49 | 0.43 | — | — |
| **prod (NEW, anti-repeat ON)** | **0.036** | 0.972 | 0.792 | 0.297 | 0.45 | 0.42 | 0.513 (5/6) | 0 |
| noar (OFF baseline) | 0.048 | 0.889 | 0.711 | 0.290 | 0.49 | 0.45 | 0.538 (4/6) | 0 |

Plain prod path (no env) now produces the anti-repeat gain: ON is more human than OFF (grid_cov
0.97 vs 0.89, dir_ent 0.79 vs 0.71), density + parity hold. (Absolute h_dist 0.036 vs the sweep's
0.020 is temp-0.9 draw noise — the whole scale shifted up this draw; noar OFF = 0.048, so ON<OFF
holds. Read the gap, not the absolute.)

**BUILT the late-song-collapse metric — the last untouched original complaint.** New
`scripts/eval_late_window.py`: per song, gen vs HUMAN-reference note-share in the final tail
(`late_gap = ref_late_frac − gen_late_frac`; positive = gen under-produces the tail = collapse) plus
a tail-only density Spearman (`late_corr`). Reuses the eval_songset refs — no regeneration needed.

**FINDING — late collapse does NOT reproduce in current production.** On all 6 eval songs, at both
final-20% and final-10% tails, mean `late_gap` is **negative** (−0.024 / −0.018): gen actually puts a
*slightly higher* note-share in the tail than the human map, and tail density still tracks the song
(late_corr +0.32 / +0.46, above the 0.30 bar). No song shows a meaningful positive gap. Strong
hypothesis: the original ~160-164s collapse was already fixed as a side-effect of `section_gate=
loud_only` (final chorus is loud → kept dense) + density-select-γ2.5. **Caveat:** the 6-song set may
not include a song Kyle actually saw collapse on → next session, get a specific song from Kyle and
confirm before declaring it CLOSED (see handoff task 1).

**Net:** all three original complaints now addressed — drop-@-13s (loud_only), flat-density
(density-select), monotony/grid-coverage (anti-repeat promoted this session) — and the late-collapse
complaint appears already resolved (metric built to prove/catch it). Next = Kyle validates late-collapse
on a real song, then a ship-it/step-back fork.

**Code committed + pushed** (layout_model default flip, eval_sweep noar arm, eval_late_window.py, this
retro, memory). `git push origin main` works (gh auth resolved 2026-07-23).

## 2026-07-23 — ★ TEMP NUDGE PROMOTED TO PROD ★ + fixed a latent 0-note crash + ANTI-REPEAT sweep WON (ar_w1_s2)

Kyle's two calls this session: (1) **promote the decode nudge** and (2) target **monotony / pattern_repeat** next.

**DONE — decode nudge shipped to production.** `scripts/generate.py` defaults bumped
temp 0.8→**0.9**, top_p 0.85→**0.97** (the g2.5_temp arm that won the 06-30 sweep: grid_cov
0.85→0.93, dir_ent 0.69→0.74, h_dist 0.19→0.05, density/viol unchanged). Rendered prod-vs-temp
for Kyle first (`outputs/temp_nudge_2026-07-23/`). `eval_sweep._gen` hardcoded decode also moved
to 0.9/0.97 so the sweep control = new prod.

**BUG FOUND + FIXED (latent, pre-existing) — stochastic 0-note maps / IndexError.** The NON-v7
path (`generate_level` → `nucleus_sampling_decode` → `beam_search.apply_constraints`) crashes
whenever the sequence model samples a **WALL or CHAIN** event: those events' grammar attribute
ranges reach token idx 162–182 but the model vocab is only **118**, so `mask[offset+i]` indexes
past the tensor → fatal IndexError (or, when it EOS'd first, a near-empty map). Stochastic, so
06-30 got lucky. FIX in `beam_search.py`: added `_selectable_events(vocab_size)` — only offer
event types whose grammar fits the model's logit width (NOTE/BOMB/ARC fit; WALL/CHAIN don't at
vocab 118) + a defensive `min(count, mask.width-offset)` clamp on the grammar write. NOTE: the
**v7 production path is unaffected** (uses `generate_v7_level`, not this), so this didn't block
the sweep — but it's a real robustness fix for the v6/untrained path. (Also: `--v7` is REQUIRED
on scripts/generate.py or it silently falls to untrained models — eval_sweep passes it; manual
runs must too.)

**NEW LEVER built (gated, default OFF) — windowed adjacency anti-repeat.** In
`models/layout_model.py`: `LAYOUT_ANTIREPEAT`=W (recent-window size) + `LAYOUT_AR_STRENGTH`=S
penalize only tokens emitted in the last-W steps PER ROLE (X/Y/DIR) — breaks back-to-back loops
WITHOUT flattening the whole-phrase distribution (unlike the cumulative LAYOUT_DIV_* penalty,
which over-flattens: div10 → col_conc 0.26, rows 0.35). Smoke (1f333, W1/S2, v7): 512 notes,
grid_cov 0.67→**1.0**, dir_ent 0.72→**0.80 (=human)**, monotony **0.43 (=human)**, col_conc
**0.29 (~human)**, viol 0. ⚠️ **Smoke surfaced that `pattern_repeat` is ALREADY ~human (~0.0)** in
shipped maps — so the real residual is grid/dir coverage + composite monotony, not literal repeats.
Also surfaced `pattern_repeat` as its own scorecard column (was hidden inside the monotony composite).

**SWEEP COMPLETE — WINNER `ar_w1_s2` (`scripts/overnight_2026-07-23_antirepeat.sh`,
`logs/overnight/antirepeat_2026-07-23.log`).** The windowed adjacency anti-repeat at **W=1 /
S=2.0** is the most human-like layout config measured — **h_dist 0.020 < prod 0.039** while holding
every guard (density_corr 0.521 4/6, monotony 0.43=human, pattern_repeat 0.00, col_conc 0.29~human,
row_conc 0.47, viol 0). Full leaderboard:

| arm | h_dist↓ | grid_cov | dir_ent | monot | col_conc | row_conc | dens (#pass) | viol | verdict |
|---|---|---|---|---|---|---|---|---|---|
| HUMAN | — | 0.96 | 0.80 | 0.43 | 0.29 | 0.49 | — | — | — |
| prod (0.9/0.97) | 0.039 | 0.92 | 0.80 | 0.44 | 0.31 | 0.46 | 0.511 (4/6) | 0 | control |
| **ar_w1_s2** | **0.020** | 0.93 | 0.80 | 0.43 | 0.29 | 0.47 | 0.521 (4/6) | 0 | ★ **DoD MET** |
| ar_w2_s2 | 0.038 | 0.93 | 0.81 | 0.43 | 0.27 | 0.45 | 0.524 (5/6) | 0 | DoD MET (marginal) |
| ar_w3_s3 | 0.086 | 1.00 | 0.88 | 0.41 | 0.27 | 0.40 | 0.539 (5/6) | 0 | over-diversifies, no h_dist gain |
| g2.5_div10 | 0.145 | 1.00 | 0.94 | 0.38 | 0.26 | 0.35 | 0.531 (5/6) | 0 | over-flatten ref (as expected) |

Takeaway: gentle **W=1** (forbid only the immediate per-role repeat) nudges toward human without
the over-diversification that W≥3 and the cumulative div penalty cause (grid→1.0, dir→0.88-0.94 ≫
human 0.80, rows collapse to 0.35-0.40). **NOT yet promoted** — promotion + render-for-Kyle is the
top next-session task (see handoff). The lever stays default-OFF until then.

**Code UNCOMMITTED** (generate.py defaults, beam_search event-selection fix, layout_model
anti-repeat knob, eval_sweep arms+pattern_repeat column+prod decode, overnight script, this retro).

## 2026-06-30 (PM-3) — ★ THE grid_cov/dir_entropy "GAPS" WERE A GREEDY-DECODE HARNESS ARTIFACT ★ (+ eval_sweep now decodes at prod temp)

**RESULT (sweep ran, `logs/overnight/layoutdiv_2026-06-30.log`):** the PM-2 scorecard measured layout
diversity while `eval_sweep` forced `--temperature 0.0` (greedy → nucleus collapses to argmax), which
UNDERSTATED the shipped maps. Production `generate.py` defaults to **temp 0.8 / top_p 0.85**, not greedy.
Measured at those exact prod defaults: **grid_cov 0.85 (not 0.64), dir_ent 0.69 (not 0.62)**, col_conc
0.31 ≈ human 0.29, row_conc 0.48 ≈ human 0.49, density +0.54 (5/6), viol 0. So shipped maps are already
near-human on cell/direction coverage; the residual is a modest dir_entropy 0.69→0.80.

| arm (all dsel_g2.5) | grid_cov | dir_ent | col_conc | row_conc | dens | pass | viol |
|---|---|---|---|---|---|---|---|
| HUMAN | 0.96 | 0.80 | 0.29 | 0.49 | — | — | — |
| greedy (old harness) | 0.64 | 0.62 | 0.36 | 0.48 | 0.53 | 4/6 | 0 |
| **PROD (0.8/0.85)** | **0.85** | **0.69** | 0.31 | 0.48 | 0.54 | 5/6 | 0 |
| **temp (0.9/0.97)** | **0.93** | **0.74** | 0.30 | 0.47 | 0.53 | 5/6 | 0 |
| div05 penalty | 0.94 | 0.91 | 0.27 | 0.39 | 0.54 | 5/6 | 0 |
| div10 penalty | 1.00 | 0.94 | 0.26 | 0.35 | 0.53 | 4/6 | 0 |

**Two takeaways:** (1) **HARNESS FIX (shipped):** `eval_sweep._gen` now decodes at prod defaults
(temp 0.8/top_p 0.85) instead of temp 0.0 — the layout-quality axes were systematically wrong before.
Density conclusions (Stage-1 note counts) are unaffected by layout temp, so the gamma sweep still holds.
(2) **OPTIONAL prod nudge (Kyle's call):** temp 0.8→0.9 + top_p 0.85→0.97 (`g2.5_temp`) pushes grid
0.85→0.93 / dir 0.69→0.74 while KEEPING human-like col/row conc — a clean, structure-preserving gain.
The DIR-penalty (`LAYOUT_DIV_D`, new gated knob) works but OVER-diversifies (dir_ent 0.91-0.94 ≫ human
0.80, rows flatten to 0.35-0.39) → keep dormant, don't ship. **NEXT SESSION:** render `g2.5_temp` vs
prod for Kyle; if he likes it, bump generate.py decode defaults to 0.9/0.97. Code UNCOMMITTED
(layout_model `LAYOUT_DIV_D`, eval_sweep temp fix + arms, overnight script).

### (superseded plan — kept for context) originally QUEUED: LAYOUT-DIVERSITY sweep

Old Scoped-V8 TASK stack (TASK 0-5) is fully DONE/DEAD/no-premise (unchanged since 06-09) — no
live architecture items there. The **live research items are the two gaps the PM-2 hardened
scorecard exposed** and that survived the decode-bug fix:
- **grid_coverage** ~0.61-0.68 vs human 0.96 (model under-uses the 12 grid cells)
- **dir_entropy** ~0.58-0.63 vs human 0.80 (model under-uses the 9 cut directions)

**Key realization:** the sweep decodes layout GREEDILY — `eval_sweep.py` passes `--temperature 0.0`,
which makes `_nucleus_sample` collapse to argmax (top_p irrelevant). So those numbers are the model's
*argmax* diversity, and raising top_p alone does nothing. Two no-retrain levers, both on the
production density config (dsel_g2.5 = control):
- **(a) stochastic decode** `g2.5_temp` (temp 0.9, top_p 0.97) — let the tail through.
- **(b) frequency penalty** `g2.5_div05/10` — deterministic anti-repeat. Was X/Y only (grid_cov);
  **extended to the DIR role** via new env `LAYOUT_DIV_D` (default 0.0) so it can move dir_entropy.
  Smoke test (div10, temp 0.0, 1f333): despite peaked argmax (Y~0.85) it spread SAMPLED rows to
  [0.33,0.33,0.33] and cols to ~[0.25×4] — the anti-repeat rotates cells/dirs deterministically, rc=0.

Code (UNCOMMITTED): `LAYOUT_DIV_D` + `_div_counts_for` helper generalizing the X/Y penalty to
ROLE_DIR in `models/layout_model.py`; 4-arm ARMS refresh in `scripts/eval_sweep.py`;
`scripts/overnight_2026-06-30_layoutdiv.sh`. Launched detached → `logs/overnight/layoutdiv_2026-06-30.log`.
**DoD (per the script's verdict block):** an arm reaching **grid_coverage ≥ 0.80 AND dir_entropy ≥ 0.72**
while HOLDING density_corr ≥ 0.41, row_conc ≤ 0.60, col_conc ≥ 0.20 (not over-flattened), viol == 0
⇒ promote to production layout config + render vs control for Kyle. If every lever over-flattens
(col_conc < 0.20 / monotony spikes) without closing the gap ⇒ logits are the ceiling ⇒ next step is
a *targeted* diversity-reg fine-tune (distinct from the superseded entropy-reg, which over-diversified).

## 2026-06-30 (PM-2) — EVAL LOOP HARDENED + visibility upgrades (autonomous research cycle)

Spent the back half hardening the eval loop so theories can be tested without hand-holding. All in
`scripts/eval_sweep.py` + new `scripts/map_metrics.py`; documented in `docs/eval_harness.md`
(linked from README). Changes:
- **Shared map-metrics** (`map_metrics.py`): row_conc, col_conc, grid_coverage, dir_entropy,
  monotony, pattern_repeat, nps — one source of truth, also surfaces NEW gaps (grid coverage,
  direction variety) the old scorecard hid.
- **Human baselines baked in**: `eval_sweep.py human-baseline` (40 maps → human_baseline.json,
  auto-loaded) so every metric prints vs its human target. Baselines: row_conc 0.49, col_conc 0.29,
  grid_coverage 0.96, dir_entropy 0.80, monotony 0.43.
- **Composite human-distance** (`h_dist`) auto-ranks arms by overall layout closeness to human.
- **report.md** per sweep: density_corr + quality-vs-human tables + embedded before/after renders.
- **Onset-alignment** metric (onset_hit) + **live progress** (line-buffered, per-song lines under nohup).
- Pruned dead arms (rejected temperature theory); arms now `baseline` + density-select gammas.
- **17-vs-16 crash FIXED** (same context-prefix root cause as the decode bug; NO-REPRO ×25).

**Post-fix refresh sweep (`logs/overnight/refresh_sweep_2026-06-30.log`, report.md):** row_conc
0.49-0.50 (=human), col_conc 0.30-0.33 (~human 0.29), monotony 0.49 (human 0.43), viol 0. **density
DoD: dsel_g2.5 +0.550 (5/6).** Tradeoff now VISIBLE: g2.5 best density; g4.0 best layout-human-dist
(0.13) via higher grid_cov/dir_ent.

**NEW GAPS the upgraded loop exposes (next research):** grid_coverage ~0.6 vs human 0.96 and
dir_entropy ~0.6 vs 0.80 — even post-fix the model uses fewer of the 12 cells and less direction
variety than humans. These are the next layout-quality levers (were invisible before this scorecard).

## 2026-06-30 (PM) — ★ ROW COLLAPSE WAS A ONE-LINE DECODE BUG ★ (off-by-ctx_n; fix → row_conc 0.94→0.48 ≈ human, plain v10, no retrain)

## 2026-06-30 (PM) — ★ ROW COLLAPSE WAS A ONE-LINE DECODE BUG ★ (off-by-ctx_n; fix → row_conc 0.94→0.48 ≈ human, plain v10, no retrain)

**THE "for-sport" bottom-row collapse was a token-misalignment BUG in inference, not the model.**
`LayoutPhraseModel.generate_phrase` builds `toks = context_tokens(ctx_n=16) + [BOS] + events` but
returned `toks[1:]` — stripping only ONE token, leaving 15 context tokens + BOS in front of the
event stream. `_decode_phrase_tokens` parses from index 0 expecting KIND, so EVERY field (KIND/X/
**Y**/DIR) was read off-by-ctx_n; the garbage Y tokens (mostly < Y_BASE) clamped to row0 → 94% row0
in every v10 map. **Fix (1 line): `return toks[ctx_n + 1:]`** (ctx_n=0 ⇒ unchanged).

**Localized by instrumentation:** the model SAMPLES diverse rows (NOTE-rows ~[0.30,0.38,0.32]) but
`all_events` came out [0.78,0.04,0.19] → collapse is between decode and assembly = the misaligned
parse. Confirmed in raw .dat.

**Result of the fix (1f3d7, plain v10, DEFAULT decode temp0.9/top_p0.85):** row_conc **0.94→0.484**
(human 0.47), rows [0.48,0.27,0.25] vs human [0.47,0.31,0.21], cols [0.45,.04,.50,.01]→
[0.31,0.19,0.21,0.29] vs human even, viol 0. **Human-level layout diversity from a 1-line fix, no
retrain, no top_p change.** **VALIDATED across the 6-song set (plain v10, default decode):
MEAN row_conc 0.476 (human 0.47!), per-song 0.44-0.51, density_corr 0.528 (5/6 pass, held),
total_viol 0.** Cols now spread across all 4 (e.g. [.19,.30,.30,.21]) vs old [.45,.04,.50,.01].
Render `outputs/density_select_2026-06-30/v10_bugfix.png` (sent to Kyle): lattice panels use all 3
rows + varied directions vs the old bottom-row zigzag. This also un-scrambles KIND/X/DIR → broadly
better layout quality (directions were also misaligned, previously masked by the parity-fixing
postprocess). **Both of Kyle's complaints (flat density + for-sport bottom-row) now addressed:
density-select γ2.5 + the 1-line decode fix.**

**The entropy-reg fine-tune (below) is now SUPERSEDED / unnecessary** — it was treating a symptom;
with the bug fixed, plain v10 is human-level. (ft model + high top_p over-diversifies: row_conc
0.34, ~uniform.) Keep `LAYOUT_ENT_REG` as a dormant gated knob; default decode is fine.

### (superseded) earlier PM path: entropy-reg fine-tune + raised top_p

Chased the bottom-row/2-col collapse to its actual mechanism (a chain of negatives that each
ruled out a layer):
1. **Tokenizer round-trip is faithful** — encode→decode a human map preserves row_conc 0.46
   exactly. Representation is innocent.
2. **Postprocess only touches COLUMNS** — `enforce_color_separation` pushes red→left/blue→right
   (explains col0/col2); it never changes Y. PRE vs POST `BS_PREPOST_OUT` dump: rows identical.
3. **The model logits were peaked** — decode diagnostic (`LAYOUT_DIAG=1`, logs mean argmax-prob at
   X/Y steps in `generate_phrase`): v10 **Y argmax-prob 0.92, X 0.78** → nucleus always picks the
   mode = row0/col0. Decode-time frequency penalty (`LAYOUT_DIVERSITY`) and temperature both fail
   against logits this peaked (rejected).
4. **TWO compounding causes:** peaked logits AND the tight default nucleus `--top-p 0.85` that
   discards the tail. Fixing either alone isn't enough.

**FIX (working): entropy-reg fine-tune + raised top_p.** Added `LAYOUT_ENT_REG` (env-gated) to
`layout_module._forward_batch`: an entropy BONUS on the X/Y position softmaxes (over their legal
ranges) that flattens the over-confident logits. `scripts/finetune_layout_diversity.py` loads v10
weights and fine-tunes a few epochs (~15 min/epoch, 187k phrases). β=3.0/lr=1e-4 epoch-0 dropped
decode argmax-prob **Y 0.92→0.36, X 0.78→0.30**. Then at generation `--top-p 0.999` lets the
flattened tail through. **ft-ep0 + top_p0.999 (1f3d7): row_conc 0.94→0.78, cols
[.45,.04,.50,.01]→[.37,.13,.43,.07], viol 0.** Both axes moving toward human (row 0.47,
cols even), playability intact. Epochs 1-3 still training (flatter logits → expect further drop);
`scripts/eval_layout_ckpt.py` evals each epoch on the song set (row_conc + cols + viol +
density_corr), log `logs/overnight/ft_epoch_eval_2026-06-30.log`. DoD: row_conc → <0.65 (toward
0.47) holding density_corr ≥0.41 + viol 0. β=0.5/lr=3e-5 was too weak (row_conc barely moved) —
needed the stronger β + higher LR.

New code (all UNCOMMITTED): `LAYOUT_ENT_REG` in layout_module.py, `LAYOUT_DIVERSITY`/`LAYOUT_DIAG`
in layout_model.py (penalty rejected, diag kept), `scripts/{finetune_layout_diversity,
eval_layout_ckpt}.py`. Recommend raising the generation `--top-p` default (0.85→~0.97+) ONLY paired
with the entropy-reg model (high top_p on the peaked v10 just adds noise).

## 2026-06-30 — DENSITY-AWARE SELECTION WORKS (DoD GREEN, no retrain) + EVAL LOOP EXPANDED → residual = Stage-2 LAYOUT monotony

**The oracle prediction held: a post-process selection change solves the density DoD — no retrain.**
Implemented `DENSITY_SELECT` (env-gated, default OFF) in `generation/generate.py`: keeps the SAME
total note count as the threshold method but RE-ALLOCATES it across 2s windows ∝ (window-mean
prob)^γ, with NMS spacing (`_density_aware_select`, ~L1773). Knobs: `DENSITY_SELECT_GAMMA`,
`DENSITY_SELECT_WIN`.

**Built the multi-song/multi-arm eval harness** `scripts/eval_sweep.py` (the "test more theories per
night" ask): a cached 6-song full-length set (`data/eval_songset/`, refs precomputed once via
Demucs) × named arms (env+flags) → leaderboard of density_corr + monotony + gen_cv + notes +
swing-viol. Add a theory = one line in `ARMS`. Results (`outputs/eval_sweep_cache/leaderboard.json`):

| arm | mean Spearman | #pass | monotony↓ | gen_cv↑ | notes | viol |
|---|---|---|---|---|---|---|
| control    | +0.260 | 1/6 | 0.622 | 0.290 | 1988 | 0 |
| dsel γ1.0  | +0.533 | 4/6 | 0.618 | 0.244 | 1908 | 0 |
| dsel γ1.5  | +0.515 | 5/6 | 0.615 | 0.299 | 1847 | 0 |
| **dsel γ2.5** | **+0.531** | **5/6** | 0.606 | **0.384** | 1719 | 0 |
| dsel γ4.0  | +0.495 | 3/5 | 0.596 | 0.454 | 1611 | 0 |

**Selection ~doubles density_corr (0.26→0.53), DoD 1/6→5/6, every song improves, 0 viol.** Sweet
spot **γ≈2.5** (best cv, 5/6 pass, ~14% fewer notes = quiet windows correctly thinned). ArcViewer
renders `outputs/density_select_2026-06-30/{control,dsel_g2.5}.png`: control = flat ~8 NPS plateau;
g2.5 density BREATHES (intro 2 notes vs 10, breakdown dips, outro thins). **Kyle is final judge.**

**RESIDUAL (next lever, now QUANTIFIED): Stage-2 bottom-row collapse.** The monotony complaint =
**row_concentration**, not pattern repeat (pat_repeat=0.000 — notes aren't literally identical; the
zigzag alternates). Human-calibrated baseline (12 human maps): **row_conc mean 0.474** (range
0.41-0.59, notes spread across rows); V7 = **0.94 — ~2× worse, ~94% of notes in ONE row.** Combined
monotony human 0.424 vs V7 0.606.

**Stage-2 TEMPERATURE sweep RAN (NEGATIVE) — `logs/overnight/stage2_temp_sweep_2026-06-30.log`.**
density-select γ2.5 held on, temperature ∈ {0, 0.7, 1.0, 1.2}: density_corr holds (~0.52-0.54, all
5/6 pass) but **row_conc stays pinned 0.941-0.948 at EVERY temperature** — sampling temperature does
NOTHING to the row collapse. ⇒ the bottom-row stream is baked into Stage-2's learned distribution,
not a decoding-diversity issue. Temperature is NOT the lever.

**ROOT-CAUSE DIAGNOSED — Stage-2 mode-collapse to a 2-of-12-cell lattice, SYSTEMIC (not checkpoint).**
Row/col distribution (load_v7): V7 rows **[0.95, 0.04, 0.01]**, cols **[0.45, 0.04, 0.50, 0.01]** →
notes live almost only in `row0 × {col0, col2}` (red col0 / blue col2, bottom row = the "for-sport"
zigzag). Human: rows [0.47, 0.31, 0.21], cols [0.26, 0.24, 0.24, 0.26] (all 12 cells even).
- **Checkpoint-swap RULED OUT:** the EARLIEST available layout ckpt (version_7 epoch-3, acc 0.865)
  collapses IDENTICALLY (row_conc 0.943, rows [0.94,0.06,0], cols [0.46,0.04,0.48,0.02]). All layout
  ckpts across versions 0-14 sit in a narrow band (token_acc 0.856-0.870, epoch≥3) and all collapse.
  So it's NOT late-epoch token-acc saturation — the model collapses by epoch 3. Systemic to the
  Stage-2 layout objective/representation.

**NEXT THEORY = break the Stage-2 layout collapse via OBJECTIVE/REPRESENTATION, not checkpoint/temp.**
DoD = row_concentration 0.94 → human ~0.47 (target <0.65) + col spread, holding density_corr ≥0.41 and
viol 0. Candidate levers (scope next session, needs Kyle's call on a GPU night): (a) Stage-2 retrain
with an anti-collapse / position-diversity term (current CE/token-acc objective lets the model win by
emitting the dominant swing token — diversity is unpenalized); (b) inspect the swing-tokenizer
vocabulary for a dominant `row0×{col0,col2}` token + class-imbalance reweighting; (c) post-hoc layout
redistribution (riskier — must preserve parity/swing-sim). Harness ready: row_conc + pat_repeat +
col in scorecard, human baseline 0.47, layout-ckpt swappable per arm via --layout-ckpt.

> **BUGS FOUND:** (1) the `RuntimeError: size of tensor a (17) must match b (16)` crash (17 = ctx_len
> 16 + BOS) was the SAME context-prefix bug — the misaligned `flat_tokens` made the cross-phrase
> context slot/hand rebuild mismatch its token count. **FIXED by the 1-line decode fix** (verified:
> NO-REPRO across ~25 post-fix attempts on the crash songs). (2) harness prints were buffered under
> nohup — FIXED: `sys.stdout.reconfigure(line_buffering=True)` + per-song progress lines.
> **GIT:** generate.py (DENSITY_SELECT + earlier BEAT_PROBS_DUMP), `scripts/{eval_sweep,
> oracle_density_ceiling}.py` all UNCOMMITTED; push still pending Kyle's GitHub auth.



## 2026-06-29 — DoD density_corr BASELINED + INFERENCE LEVERS EXHAUSTED → Phase-2 must change Stage-1 (training-time), not inference flags

**The TASK-2 DoD metric (`eval_density_corr.py`, Spearman ≥0.41) had NEVER been numbered on a
real V7 generation** — bon's "monotony" was an internal feature, not this DoD. Now measured.
Baseline (bon winner cand_16, production loud_only): **Spearman = −0.005, FAIL** (Pearson 0.45,
gen CV 0.199). The Pearson/Spearman split is the tell: a weak *linear* energy effect exists but
**zero monotonic rank-tracking** — exactly what the DoD's Spearman choice exposes.

**Decisive in-session lever sweep** (temp=0 deterministic, song=SO TIRED ROCK, all 4 arms,
`outputs/density_sweep_2026-06-29/`): every exposed inference lever lands at Spearman ≈ 0 →

| arm | Spearman | Pearson | gen notes | CV |
|---|---|---|---|---|
| section-gate=loud_only | 0.0005 | 0.449 | 1384 | 0.191 |
| section-gate=off       | 0.0596 | 0.474 | 1385 | 0.191 |
| --use-instr (gate off) | 0.0033 | 0.416 | 1386 | 0.189 |
| --no-use-instr         | −0.0213| 0.438 | 1380 | 0.205 |

**FINDING:** section-gate and the per-instrument layering feature (whose *entire stated purpose*
is densifying drops) move density_corr by **noise**. Note count pins at ~1380 regardless. ⇒
**Inference-time structure conditioning is exhausted** — the flat density is learned into Stage-1.
Reaching ≥0.41 requires a **training-time** change to Stage-1 (structure/density-conditioned onset
generation), confirming the memory's "STRUCTURE-FIRST GENERATION, NOT selection." Supporting prior:
`v8_poc_structure.py` already showed per-instrument event density correlates r=0.41 with human note
density — the signal is IN the features; Stage-1 just isn't learning to use it.

**ORACLE-CEILING PoC RAN (2026-06-29) — QUALIFIED GREEN → the flat density is a POST-PROCESS
artifact, NOT a model limit → NEXT = density-aware SELECTION (cheaper than a retrain).**
Built `scripts/oracle_density_ceiling.py` + a flag-gated `BEAT_PROBS_DUMP` in
`generation/generate.py` (dumps raw Stage-1 `beat_probs[N,2]` BEFORE threshold/NMS/density-curve;
default behavior unchanged). Non-circular test: bin the continuous per-window prob-mass into the
same 2s windows and Spearman vs the same reference (librosa drums∪other). Full-length songs
(short clips were Spearman noise, excluded):

| song | dur | windows | probMEAN Spearman | shipped-map Spearman |
|---|---|---|---|---|
| SO TIRED ROCK | 176s | 88 | **+0.437 PASS** | −0.005 |
| 1f1e1 | 148s | 75 | **+0.468 PASS** | — |
| 1f333 | 275s | 138 | +0.298 (close) | — |

Mean ≈ **0.40**, 2/3 ≥0.41, all positive — vs the **shipped maps at ≈0**. `prob_any` CV ≈ 0.63
(NOT flat). ⇒ Stage-1 ALREADY encodes density structure; the per-slot threshold + NMS +
`_apply_density_curve` EQUALIZE per-window counts and destroy the window-mean signal. The best
ceiling metric is per-window **mean** prob (probmean > probmass > probmax). Artifacts
`outputs/density_sweep_2026-06-29/{oracle_*.json,probs_*.npz,beat_probs.npz}`.

**QUEUED NEXT — density-aware selection (recover the ~0.40 ceiling in the actual map):** replace
the count-equalizing post-process with a per-window **note budget ∝ window-mean prob** (keep the
existing within-window NMS for placement, but let loud/dense windows KEEP more notes and thin quiet
ones), gated behind a flag, prior behavior default. DoD: `eval_density_corr.py` Spearman ≥0.41 on
the 3 full-length songs (currently ≈0). Read the verdict: PASS on ≥2/3 ⇒ Phase-2 density solved by
selection, no retrain. If selection caps well under the oracle (~0.40) ⇒ fall back to the Stage-1
density-conditioned retrain (inject per-window target-density + retrain). Cheap; not a GPU night.

## 2026-06-16 — P1-4 BEST-OF-N PoC BUILT + RAN (mechanism GREEN, but finding = best-of-N ALONE can't fix V7 monotony) → NEXT = STRUCTURE-FIRST GENERATION (Phase-2 proper)

**P1-4 best-of-N=16 rerank PoC DONE.** Built `scripts/best_of_n_poc.py` (the Phase-2
reranker: wraps the ep1 feel-disc to score arbitrary maps + a NEW monotony/structure penalty
+ the swing-sim hard filter) and `scripts/overnight_2026-06-16.sh` (16 stochastic V7 draws of
ONE song → filter → rank → render winner vs no-rerank control). Ran clean: **16/16 generated,
0 swing-sim violations (post-process parity-clean, as P1-3 predicted), rerank GREEN by its own
logic** — winner `cand_16` dominates control `cand_01` on BOTH axes (feel −1.707 > −1.790,
monotony 0.635 < 0.647). Artifacts in `outputs/bon_2026-06-16/` (`bon_summary.json`,
`winner.png`, `control.png`).

**THE REAL FINDING (looked at the renders — this is what matters):** the winner and control are
**visually near-indistinguishable and BOTH deeply monotonous.** Density pins flat at ~8 NPS the
whole song (the "ignores structure" complaint — present in both); every lattice panel
(beats 114-122 / 228-236 / 342-350) is the SAME metronomic bottom-row stream (blue-down + red-up
alternating at row0, perfect zigzag swing trace). The numbers confirm the eye: **N-spread is
tiny — feel 0.144, monotony only 0.016; all 16 draws sit at monotony 0.63–0.65.** ⇒ Best-of-N
over plain stochastic resampling of the SAME model **cannot escape V7's systemic monotony floor**
— every draw shares the same structure, so selection only nudges within a bad basin. The rerank
*mechanism* is validated (ranker orders correctly, swing-sim/feel/monotony all wired + working);
the *strategy* of "select over a monotonous generator" is insufficient for Kyle's complaint.

**Kyle is final judge** — ArcViewer `outputs/bon_2026-06-16/{winner,control}.png`; expectation is
he'll find them ~equally monotonous (matches the metrics). 

### TOP OF STACK — Phase-2 proper: STRUCTURE-FIRST GENERATION (not selection)
The P1-4 result re-points Phase 2: the lever is the GENERATOR, not the reranker. Options to scope
next session, in rough priority:
1. **Phrase-level resampling / diversity** — best-of-N at phrase granularity with an explicit
   anti-repetition objective (the monotony penalty becomes a *generation* constraint, not just a
   post-hoc score), so candidates actually differ structurally instead of all collapsing to the
   bottom-row stream. The cheapest test of "can selection work if candidates have real variance."
2. **Structure-conditioned generation** — make density TRACK the song (the flat ~8 NPS is the
   single most legible defect in both renders); condition Stage-1 on section/RMS structure so the
   density line stops pinning flat. Reuse `eval_density_corr.py` (Spearman ≥0.41) as the DoD.
3. The monotony penalty (`monotony_features` in best_of_n_poc.py: pattern_repeat,
   pattern_entropy_inv, density_flatness, row_concentration) is reusable as a reward/constraint in
   any of the above.

> **GIT:** P1-4 code (best_of_n_poc.py, overnight_2026-06-16.sh) is NOT yet committed; prior
> phase-1 work + this still need `git push origin main` (push pending since 2026-06-15, needs
> Kyle's GitHub auth). `outputs/bon_2026-06-16/` are artifacts, not commits.

## 2026-06-15 — P1-2 RENDERER + P1-3 CALIBRATION GATE DONE (GATE PASSED) → TOP OF STACK = P1-4 BEST-OF-N PoC

**PHASE-1 PERCEPTION CHANNEL COMPLETE (P1-1, P1-2, P1-3 all DoD-MET).** The agent-side
ArcViewer works: Claude-vision can blind-separate human from V7 output and its reasons match
the known complaints. P1-4 (Phase-2 kickoff) is now UNGATED.

> **GIT (2026-06-15):** all work is COMMITTED on `main` (phase-1 perception channel; `main` is
> ~23 commits ahead of origin). **PUSH STILL PENDING** — `git push origin main` failed in-session
> (HTTPS remote, no gh/SSH/token auth in the agent env). Run it yourself after auth (`gh auth
> login`, a PAT in the URL, or switch remote to SSH). Commits are safe locally across restarts.

- **P1-2 renderer DONE** — `scripts/render_map.py` (matplotlib, CPU). Three views per map:
  (a) whole-song density-vs-RMS strip with violation marks; (b) mapper's-eye lattice panels
  (time x, 4×3 grid unrolled on y, cut-direction arrows, hand colors, beat lines, dots=hollow
  circles); (c) per-hand swing-path/parity trace (resets ○, violations ✗) from swing_sim.
  CLI: `render_map.py <zip> --difficulty Expert --out x.png [--panels N --no-audio]`.
- **P1-3 calibration gate PASSED** — `scripts/calibration_gate.py` (render→blind→score).
  Rendered **5 human (data/raw) + 5 real V7 cohort (outputs/v7_cohort_2026-06-10/, post-process)**
  blind-shuffled; Claude ranked. **DoD MET: 5/5 clean separation** (blind top-5 = all human,
  bottom-5 = all V7) AND reasons cite all three complaints (diagonals/for-sport, monotony,
  dead drops). Artifacts `outputs/calib/{sample_*.png,key.json,ranking.json}`.
  **KEY FINDING:** the V7 cohort maps are **parity-CLEAN (0 swing-sim violations — postprocess
  rewrites directions)**, so the discriminator was NOT parity but **monotony + missing structure**:
  near-identical per-beat patterns (red→ at row0 + blue triangle), bottom-row for-sport streams,
  flat/step-function density ignoring the song. This is exactly Kyle's complaint set, now
  machine-legible. ⇒ Phase-2 selection must optimize structure/variety, not just parity.

### TOP OF STACK — P1-4 (Phase-2 kickoff PoC), now ungated
Best-of-N (N=16) rerank on ONE song using the **early-stopped feel-disc** (rule: max within-
generator ranking spread s.t. AUC ≥ 0.9 — the ep1 ckpt, NOT the saturated 60-ep one) + the
**swing-sim hard filter** (now available) + (NEW from P1-3) a **monotony/structure penalty**
since parity alone won't separate post-process candidates. Deliverable: render winner vs a
no-rerank control for Kyle to ArcViewer (he stays final judge — milestone re-anchor).
**Open q (1) RESOLVED 2026-06-15:** minted the early-stopped ranker
`outputs/feel_disc_ep1_2026-06-15.pt` (`feel_disc_poc.py --epochs 1 --save-ckpt`). Held-out
AUC(human vs V7) = 1.000 (≥0.9 ✓) AND **within-V7 logit spread = 10.8% of the human-V7 gap**
(saturated 60-ep was 0.3%; usable ordering, p10/p50/p90 = -1.84/-1.75/-1.65, max 0.57) → a
usable best-of-N ranker per the Phase-2 reward rule. Scores: `outputs/feel_disc_ep1_scores_2026-06-15.json`.
**Open q (2) still open:** generate.py has --temperature/--top_p but NO seed/N flag → best-of-N =
N stochastic invocations of generate.py (start whole-map N=16, scripted; phrase-level later).
**Build remaining for P1-4:** (a) wrap feel_disc model to SCORE an arbitrary generated map (reuse
load_v7 featurizer from feel_disc_poc.py); (b) a monotony/structure penalty (P1-3 finding: parity
is clean post-process, so penalize flat density + repeated per-beat patterns); (c) best-of-N harness
= gen 16 → swing-sim hard filter → feel-disc+monotony rank → render winner vs no-rerank control for
Kyle to ArcViewer.

## 2026-06-14 — P1-1 SWING SIMULATOR DONE (DoD MET)

**TASK P1-1 (swing simulator) COMPLETE.** `src/beatsaber_automapper/evaluation/swing_sim.py`
+ `scripts/eval_swing_sim.py` (DoD harness) + `tests/test_swing_sim.py` (9 tests, pass).
JoshaParity-style per-hand parity state machine: swing extraction → forehand/backhand
assignment → reset classification (bomb / intentional / fast_single / **violation**) →
per-map scorecard + `seam_hand_states()` for Phase-2 seam stitching + swing-EBPM.

**DoD MET (artifact `outputs/swing_sim/dod_2026-06-14.log`):**
- **600 human Standard-Expert maps → 0 violations** (median reset-rate 0.003, p99 0.08).
- **Raw V7 PRE-postprocess → ~1208–1245 violations/map** (reset-rate ~0.91).
- Sanity: V7 POST → 0 violations (postprocess rewrites directions to fix parity → the
  metric tracks real quality; clean pre/post contrast).

**The model that made human=0 / V7≫0 work (all physically motivated, no threshold-fudging
— each was found by inspecting a specific false-positive against real maps):**
1. Reset timing is **wall-clock seconds, not beats** (needs BPM): wrist-break floor
   `HARD_RESET_SEC=0.30` (human fastest reset ~0.34s; V7 crammed at 0.244s).
2. **Dots (all-dot swings) are parity-FREE** — never assign them a geometric direction for
   parity; they absorb a flip. (This was the single biggest false-positive source.)
3. A **neutral (L/R/dot) swing absorbs one parity flip** for the next directional note.
4. **Angle-flow gate** (`ANGLE_FLOW_DEG=90`): same-parity but ≥90° apart (dnL↔dnR) = a
   playable wrist *roll*, not a reset. Only near-identical-direction repeats reset.
5. **Run requirement**: a LONE fast reset = playable "double"; only the 2nd+ consecutive
   fast reset is a violation (V7's signature = sustained runs).
6. **Symmetric bomb window**: a bomb just before OR after a same-dir stream = deliberate
   bomb-reset, not a wrist-break.
7. **Standard-characteristic scoping** in the loader: load Standard/<difficulty> via
   Info.dat; SKIP maps lacking it (OneSaber/90-360/**Lawless** have different/no parity —
   they were the only two residual "human" false-positives at scale; both resolved).

**NEXT (live, in order): P1-2 renderer → P1-3 calibration gate → P1-4 best-of-N PoC.**
P1-1 unblocks the swing-path trace panel in P1-2 and the simulator hard-filter in P1-4.

## 2026-06-12 — PHASE 0 CLOSED → TOP OF STACK = PHASE 1 "MAP PERCEPTION" (READ FIRST)

**Strategy reset 2026-06-12 (user-requested fresh-eyes review). Master plan =
`docs/research_2026-06-12_fresh_eyes_plan.md` — read it before building anything.** Diagnosis:
8 architectures optimized per-slot proxies that anti-correlate with quality; the missing piece is
the JUDGE (perception), not the generator. Pipeline: judges first (Phase 1), then structure-first
generation + best-of-N selection (Phase 2), DPO only if needed (Phase 3), lighting decorator
(Phase 4).

### Phase 0 — DONE 2026-06-12 (reward gate at scale)
- V7 cohort grown to **400 maps** (`outputs/v7_cohort_2026-06-10/`, 5 fails, 24s avg).
- Feel-discriminator (`scripts/feel_disc_poc.py`, now has `--save-ckpt`/`--dump-scores`):
  **held-out AUC(human vs V7) = 1.0000 on ALL arms** (none/dt/spatial/dir) → gate PASSED, not a
  one-feature fingerprint (V7 is distinguishable in every feature group).
- **Saturation finding:** the 60-epoch model is a perfect detector but a USELESS ranker (all V7
  logits ≈ −10.23, within-V7 sd = 0.3% of human gap). **Fix VALIDATED = early stopping:** @1 epoch
  AUC 0.994 with within-V7 sd = 14% of gap (smooth ordering). **Reward-ckpt rule for Phase 2:
  maximize within-generator ranking spread subject to AUC ≥ 0.9.**
- Artifacts: `outputs/feel_disc_{none,dt,spatial,dir}_2026-06-11.json`,
  `outputs/feel_disc_2026-06-12.pt` (60-ep, saturated — do NOT use for ranking),
  `outputs/feel_disc_scores{,_ep1}_2026-06-12.json`.

### TOP OF STACK — Phase 1 tasks (plan doc §4, in order)
1. **TASK P1-1 — swing simulator** `src/beatsaber_automapper/evaluation/swing_sim.py` (extend
   `evaluation/playability.py`): per-hand parity state machine, swing-angle sequence, reset /
   wrist-break detection, swing-EBPM; per-map scorecard + per-seam entry/exit hand state. Port
   JoshaParity concepts (github.com/Joshabi/JoshaParity). Author tiny known-violation fixtures as
   unit tests. **DoD: 0 violations on human Expert maps; >0 on raw PRE-postprocess V7 output
   (use the `BS_PREPOST_OUT` env dump in `generation/generate.py`).**
2. **TASK P1-2 — renderer** `scripts/render_map.py` (matplotlib, no GPU): (a) mapper's-eye
   lattice panels, 8–16 beats each — time on x, 4×3 grid unrolled on y, cut-direction arrows,
   hand colors, beat lines, RMS strip; (b) whole-song density-vs-RMS strip w/ section overlay;
   (c) per-hand swing-path trace from P1-1. Output PNGs for Claude vision eval.
3. **TASK P1-3 — calibration gate**: render 5 human + 5 V7 maps blind-shuffled; Claude ranks +
   states reasons. **DoD: ranking separates human/V7 AND reasons match the known complaints
   (diagonals, monotony, dead drops).** If FAIL → fix perception before any generation work.
4. **TASK P1-4 (gated on 1–3) — Phase-2 kickoff PoC**: best-of-N (N=16) phrase rerank on ONE song
   using early-stopped feel-disc (rule above) + swing-sim hard filter; ArcViewer the winner vs a
   no-rerank control.

### Standing decisions (do not relitigate; rationale in plan doc)
- **NO arcs/chains at generation** — mask kinds 39–42 (ARC/CHAIN_HEAD/TAIL,
  `swing_tokenizer.py`) in constrained sampling; arc decorator = optional postprocess later.
- Eval protocol = 3 tiers: sim+reward over 100% of timeline; 1 whole-song macro strip; ~12–20
  stratified vision panels (unique section types + seams + drop + judge-flagged worst windows).
  Vision scoring is COMPARATIVE vs same-section human references, never absolute.
- NOT doing: V9 rebuild, whole-song attention, per-slot-F1 retrains, new per-slot features.

### Housekeeping
- ⚠️ **`git push origin main` still pending (needs user auth)** — 22+ commits + ALL of
  06-10→06-12 uncommitted (feel_disc/gen_v7_cohort/overnight scripts, plan doc, leak fix).
  Suggested: one "phase-0: reward gate at scale + fresh-eyes plan" commit, then push.
- FIXED 2026-06-12: `eval_contour_follow._load_notes_with_direction` leaked 15MB tempdir per zip
  load (filled root partition w/ 1,610 dirs ≈ 24GB). Cleanup now in `finally`. If disk fills
  again, check `/tmp/contour_eval_*` first. `CLAUDE_CODE_TMPDIR=/mnt/giga_speed/claude_tmp` is in
  user Claude settings (active from next session).

## ✅ 2026-06-09 — MACHINE-SWAP HANDOFF (RESOLVED — dual-boot done, repo/data intact; push to origin STILL pending)

**You are migrating machines. NOTHING since 2026-05-25 is committed** — `git log` shows the last
commit is `a51022c` (Run-6 prep), which predates ALL the V7-harness / scoped-V8 / reward-gate work.
A plain `git clone` on the new box loses everything. The big data + checkpoints are **gitignored**
(`/data/`, `logs/`, `outputs/`, `*.pt`) so they will NOT travel with the repo either.

### Before you wipe the old machine — copy/commit these
1. **COMMIT THE CODE (most important).** ✅ **DONE 2026-06-09 — committed as `39c877f` on `main`**
   (63 files, 4.6M: code/docs/specs + tiny TensorBoard events/hparams; NO ckpts/data). Tree CLEAN.
   NOTE: `logs/` is NOT gitignored (only `*.ckpt`/`*.log` inside are), so the small
   `events.*`/`hparams.yaml` for version_5..14 got committed — fine/intended.
   **STILL TODO before wipe — get the commit OFF this box (it protects nothing until it leaves):**
   ```bash
   git bundle create /path/to/usb/beatsaber.bundle --all   # or push to a remote
   ```
2. **COPY the gitignored artifacts you can't cheaply rebuild** (rsync to USB/NAS/new box):
   | path | size | rebuildable? |
   |---|---|---|
   | `data/raw/`           | 36G   | source maps — the seed for everything; HARD to re-fetch (couldn't fetch over wire). **COPY.** |
   | `data/processed/`     | 59G   | the 5320 `.pt` feature cache. Rebuildable from `data/raw` via preprocess (~4–7h GPU) — copy if you value the 4–7h. |
   | `data/test_songs/`    | 6.8M  | `SO TIRED ROCK - NUEKI.mp3` — the only test song, couldn't re-fetch. **COPY.** |
   | `logs/beat_classifier/version_4/`  | 619M | **PRODUCTION beat ckpt** (val_f1=0.603). **COPY.** |
   | `logs/layout_phrase/version_10/`   | 723M | **PRODUCTION layout ckpt** (ctx16+song-mem, align-F1 0.410). **COPY.** |
   | `logs/layout_phrase/version_13,14/`| 723M ea | TASK-3 contour A/B ckpts — **TASK 3 is DEAD, safe to DROP.** |
   | `outputs/`            | 11G   | generated maps + evals; only `outputs/2026-06-07/` (reward-gate probe inputs) + `outputs/reward_gate_smoke.json` matter. Rest droppable. |
   | `.venv/`              | 7.8G  | **DO NOT copy — rebuild** (see below). |
3. **REBUILD THE ENV on the new box** (Python **3.12.3**, RTX 5090 sm_120 needs PyTorch nightly cu128):
   ```bash
   uv sync                       # restores from uv.lock + pyproject.toml
   # basic-pitch on py3.12 has NO TF cp312 wheel → ONNX backend special-case:
   uv pip install basic-pitch --no-deps onnxruntime mir_eval resampy pretty_midi
   ```
   Verify: `pytest -q` (**415 passed, 4 xfailed, 5 xpassed, ~9s** as of 2026-06-09), then
   `nvidia-smi` shows the GPU, then run the reward-gate smoke (below) to confirm end-to-end.

   **⚠️ THIS IS A DUAL-BOOT OS SWITCH (same machine), NOT new hardware.** Booting Linux→Windows
   2026-06-09/10. The "copy 95G to a USB/new box" framing in the table above is overkill for the
   *code* — same disks. What actually matters:
   - **Code travels via `origin` (GitHub), not the disk.** The repo here lives on the Linux ext4
     partition; Windows can't read ext4 natively. So push to origin and `git clone`/`pull` on the
     Windows side (or work from WSL, which CAN see the Linux files). **`git push origin main` is the
     real safety action before booting away.**
   - **Data (`data/raw` 36G, `data/processed` 59G) is gitignored + on ext4** → not reachable from
     native Windows. If you intend to do project work on the Windows side, either run under WSL2
     (mounts the ext4) or stage the data on a shared NTFS partition. If Windows is just for
     gaming/other, ignore this — the Linux partition keeps everything intact for next Linux boot.
   - If running natively on Windows: `.venv\Scripts\activate` (not `source`); `uv sync` +
     basic-pitch ONNX line work as-is; `nohup ... &` → `Start-Process`/scheduled task; the bash
     `overnight_*.sh` runners need Git-Bash/WSL.
   - **Claude Code memory does NOT travel with git** — `~/.claude/projects/.../memory/` (`MEMORY.md`
     + the two project memories). On a fresh Windows Claude Code it starts blind; under WSL it reads
     the same Linux home, so prefer WSL to keep continuity.

### Uncommitted-file inventory (what `git add -A` will capture — all this session's lineage)
**Modified (13)** — core pipeline changes since `a51022c`:
`TODO.md`, `scripts/generate.py`, `scripts/train_beats.py`, `scripts/train_layout.py`,
`src/.../data/audio.py` (energy-percentile section detector), `src/.../data/beat_dataset.py`
(`require_instr` + instr features), `src/.../data/layout_dataset.py` (`use_contour` + NPS cohort),
`src/.../generation/generate.py` (V7 inference, `section_gate`, `use_instr`/`use_contour`,
**`BS_PREPOST_OUT`** dump added today), `src/.../models/beat_classifier.py` (`instr_proj`/`struct_proj`),
`src/.../models/layout_model.py` (`contour_proj`, song-memory), `src/.../training/beat_module.py`,
`src/.../training/layout_module.py`, `tests/test_audio.py`.
**Untracked, KEEP (code/docs/specs):** `src/.../data/instrument_features.py`,
`src/.../research/{spec_v7.py,runner_v7.py}`, `scripts/auto_research_v7.py`,
`scripts/eval_{alignment,contour_follow,density_corr}.py`, `scripts/preprocess_instruments.py`,
`scripts/v8_poc{,_alignment,_structure,_retrieval_key}.py`, **`scripts/reward_gate_poc.py`** (today),
`scripts/confound_prepost_2026-06-08.sh` (today), the `scripts/overnight_*.sh` + `run_scoped_v8_stage1.sh`
+ `task0_eval_v12.sh` runners, `tests/test_{cohort_filter,instrument_features,section_gate}.py`,
`docs/architecture_v8_plan.md`, `docs/v8_0_poc_findings.md`,
`experiments/leaderboard_v7.jsonl` + `experiments/queue/*.yaml`.
**Untracked, gitignored (won't be added — copy separately, see table above):** all `logs/**/version_*`.
> Suggested commit hygiene: `logs/` should be in `.gitignore` (it is) — don't force-add it. Consider
> one squashed "wip" commit now for safety, then split into logical commits later if you care.

### Where the project stands (one paragraph)
V7 (MERT+Demucs two-stage) is the live pipeline. The scoped-V8 stack is **exhausted** — every
per-slot-F1 lever (T1/T2/T3) came back null, T4 killed, T5 has no live premise (full post-mortem
below). User pivoted to a **whole-map "feel" objective** (learned reward / preference, not slot-F1).
The de-risk gate for that pivot **PASSED GREEN today** (see next section) — so the next real build
is the preference/reward model. Production inference ckpts remain version_10 (layout) + version_4
(beat), `section_gate="loud_only"`.

---

## 2026-06-10 — GATE HARDENED @ n=1500 → DoD-B COLLAPSES (GREEN→AMBER): handcrafted reward CAN'T rank our maps

Ran build-step 1 (`reward_gate_poc.py --n 1500`, full Expert cohort, CPU; out `outputs/reward_gate_n1500.json`,
log `logs/overnight/reward_gate_n1500_2026-06-10.log`). **The 06-09 GREEN does NOT survive scaling:**
- **DoD-A HOLDS/STRENGTHENS:** AUC(human vs corrupt) = **0.9199** (was 0.905 @ n=80). The cheap feel
  signal vs RANDOM corruptions is real & robust. Top features stable (`ini_cv +1.91`, `horiz_dot_frac
  −1.29`, `parity_viol_proxy −0.84`, `contour_follow +0.84`, `density_corr_drum +0.75`).
- **DoD-B COLLAPSES → FAIL:** the SAME 4 V7 maps that scored ~0.33 @ n=80 now score **0.79–0.87**
  (human mean 0.77) → Δ = **−0.055** (needs ≥+0.25). Verdict flipped **GREEN → AMBER**. The
  handcrafted featurizer rates our V7 maps as ~human (slightly MORE human than avg) — it CANNOT
  distinguish human from our generator, even though we KNOW (ArcViewer) the maps are bad.
- **Root cause of the flip:** n=80 used the alphabetically-first ~80 .pt (biased, non-representative
  human set); the logistic boundary overfit it. n=1500 is representative → V7 maps land INSIDE the
  human cloud. **The smoke GREEN was a small-sample artifact — exactly the original caveat ("corrupt
  negatives are EASY; AUC vs easy negatives ≠ reward can rank two plausible maps").** Note the gen
  maps are even handicapped (featurized with `drum_density=None` → density_corr=0, an anti-human
  value) and STILL score human — so the can't-separate conclusion is if anything understated.
- **IMPLICATION:** build option 2a (calibrated handcrafted-feature reward) is **DEAD as a ranking
  reward** — it would score our bad maps as human → useless for best-of-N / RL. Per the build plan's
  own gate ("if it collapses, the handcrafted features can't tell bad-but-plausible from human →
  escalate to a learned map encoder"), the path forward is **2b: a learned map encoder** (reuse
  `src/.../training/style_discriminator.py`, swap AudioEncoder→pooled MERT, head→human-vs-generated).

**NEW UNLOCK (kills a long-standing false blocker):** the "only one test song" limit was ILLUSORY for
this. `data/raw/*.zip` (5374 maps) each bundle `Song.egg` (audio) + `Info.dat` (BPM). `generate.py`
takes a positional audio (accepts .ogg) + `--bpm` → V7 cohort over MANY real songs is buildable.
New harness `scripts/gen_v7_cohort.py` (extracts egg→ogg + `_beatsPerMinute`; production config: beat
v4 + layout v10, `loud_only`). Generated **60 V7 Expert maps from 60 distinct real songs in ~24min,
0 failures** (~24s/map — fast, NOT an overnight job) → `outputs/v7_cohort_2026-06-10/`.

**RIGOROUS CONFIRMATION DONE (user chose "confirm first") — handcrafted reward 2a is DEAD.** Extended
`reward_gate_poc.py` with `--v7-glob` (reads each map's real BPM → correct `nps`) to compute the
build-plan's true gate, **AUC(human vs V7)** (out `outputs/reward_gate_auc_v7_2026-06-10.json`):
- **AUC(human vs V7) = 0.3135** (n=60). Needs ≥0.75. Not just a miss — it's **below 0.5**, i.e. the
  reward ranks our V7 maps as MORE human than real humans (V7 cohort mean P(human)=**0.918** vs human
  0.771). Using this as a reward would push generation toward MORE of its current failure mode.
- **Why:** the classifier was trained to separate human from RANDOM corruptions (shuffle/rand-dir/
  flatten); those destroy `ini_cv`-type regularity. V7 maps are over-regular → they ace the "is this
  non-random?" test while still being bad in ways the 11 features never measure (incohesive diagonals,
  for-sport swings, late-song collapse). The featurizer asks "structured?", not "good?".
- **VERDICT: option 2a (calibrated handcrafted-feature reward) KILLED.** Per the build plan's own gate,
  escalate to **2b: a LEARNED map encoder** whose NEGATIVE class is our-own-generated maps (not random
  corruptions). Repurpose `src/.../training/style_discriminator.py` (already takes soft `[B,S,V]` probs
  so gradients flow): AudioEncoder→pooled MERT, head→human-vs-V7, train on (human ≻ V7) pairs. We now
  HAVE the negatives (60 maps; +more at 24s each — likely want ~300–500 for a real train set). **Next
  experiment = build + train that discriminator; DoD = held-out AUC(human vs V7) ≥0.75 from the LEARNED
  encoder** (if even a learned encoder can't separate, the human/V7 gap is perceptual, not in
  measurable map space → deeper rethink). **AWAITING USER GREENLIGHT on the 2b build.**

⚠️ **SAFETY: local `main` is 22 commits AHEAD of `origin/main` (0 behind) — STILL UNPUSHED.** Host is
still `AI-Mainframe` (Linux; nothing swapped yet, all data/ckpts intact). `git push origin main` is
the outstanding #1 handoff action.

---

## 2026-06-09 — OBJECTIVE/EVAL PIVOT → REWARD-SIGNAL GATE = **GREEN** → BUILD THE REWARD MODEL (Top of Stack)

User chose the **objective/eval pivot** (over "accept pipeline" and "attack flat density"): per-slot
F1 keeps hitting a subjectivity ceiling (Stage-1 val_f1 ~0.60 ×6 runs; Stage-2 x-acc ~70% ×7 runs;
contour ~chance) because human mappers disagree per-slot but agree on FEEL. New thesis: optimize a
WHOLE-MAP "feel" objective (human-preference / learned reward / ranking), not slot-wise agreement.

### De-risk GATE result — GREEN, decisive (`scripts/reward_gate_poc.py`, smoke n=80)
Cheap handcrafted-feature classifier, human Expert vs feel-destroyed (corrupt) maps, then probe V7:
- **DoD-A: CV AUC(human vs corrupt) = 0.905** (≥0.80 PASS) → a map-level feel signal IS learnable
  from cheap features, no deep encoder needed for the *signal* to exist.
- **DoD-B: mean human P(human)=0.751 vs V7 mean=0.33 → Δ=+0.405** (≥0.25 PASS) → the signal scores
  our generator as clearly sub-human → **usable as a reward.** (V7 probe: A_contour_ep 0.44,
  A_contour_ex 0.33, B_control_ep 0.31, B_control_ex 0.31.)
- **Feature weights corroborate the user's own complaints** (signed, + = human-like):
  `ini_cv +1.54` (humans VARY note spacing; ours is metronomic), `horiz_dot_frac −0.99` &
  `diagonal_frac −0.83` (too many horizontals/diagonals = NON-human → exactly the "for-sport
  diagonals / random horizontals" complaint), `contour_follow +0.75`, `parity_viol_proxy −0.75`,
  `density_corr_drum +0.70` (tracking the drums = human; our flat density = not). 
- Artifacts: `outputs/reward_gate_smoke.json`. **VERDICT logged: GREEN — build the reward model.**
- ⚠️ Caveat to validate at scale before trusting as a training reward: the corrupt negatives are
  EASY (random/shuffled). High AUC vs easy negatives ≠ the reward can rank two *plausible* maps. The
  honest next test is human-vs-OUR-GENERATED as the negative (harder), and ideally pairwise human
  preference. Treat 0.905 as "signal exists," not "reward is solved."

### NEXT BUILD — the preference/reward model (detailed)
Build order (each step gated, cheapest first; keep V7 generation frozen until a reward is trusted):
1. **[ ] Harden the gate (1 run, CPU).** Re-run `reward_gate_poc.py --n 1500` (full Expert cohort)
   to confirm AUC holds at scale. Then add a 4th negative class = **our V7-generated maps** (not just
   corruptions) and report AUC(human vs V7) separately — that's the discrimination the reward must
   actually make. DoD: AUC(human vs V7) ≥ 0.75. If it collapses, the handcrafted features can't tell
   "bad-but-plausible" from human → escalate to a learned map encoder.
2. **[ ] Reward model proper.** Two options, prefer (a) first:
   (a) **Calibrated feel-score** = the logistic-reg P(human) from a frozen, full-cohort featurizer.
       Cheap, interpretable, immediately usable as a scalar reward. Persist `mu/sd/coef` to
       `models/reward_v0.json`. 
   (b) **Learned pairwise ranker** (only if (a)'s features cap out): small MLP/transformer on the
       map token stream + MERT, trained on (human ≻ corrupt) and (human ≻ V7) pairs with a
       Bradley-Terry / margin loss. This is the "real" preference model.
       **⭐ BIG HEAD START — reuse `src/.../training/style_discriminator.py` (`StyleDiscriminator`).**
       It's a V6-era audio-conditioned transformer over (audio_emb, swing_tokens)→mapper_id, and it
       **already accepts soft probabilities `[B,S,V]` so gradients flow from the seq model through
       the discriminator** — i.e. it was purpose-built as a learned "style-closeness" reward. It's
       NOT wired into V7 (uses the old AudioEncoder, not MERT; vocab is V6's 118). To repurpose:
       swap AudioEncoder→pooled MERT, retarget the head from mapper_id to human-vs-generated (or
       keep mapper_id and reward "classified as a real cohort mapper"), retrain on the V7 token
       grammar. Tested (`tests/test_style_discriminator.py`, 15 cases pass).
3. **[ ] Use the reward to improve Stage-2 — cheapest usage first:**
   (a) **Best-of-N rerank at inference** (no training): generate N layouts per phrase/song, keep the
       max-reward one. Measure reward lift + ArcViewer feel. If best-of-N already feels better, that
       alone is a shippable win and validates the reward.
   (b) **Reward-weighted fine-tune / RL** (expensive, only if best-of-N helps): fine-tune Stage-2 to
       maximize reward (REINFORCE / DPO-style on sampled pairs). Guard against reward-hacking by
       keeping per-slot F1 + density-corr as regression tripwires.
4. **[ ] DoD for the whole direction:** a best-of-N or fine-tuned map (i) raises mean reward vs greedy
   V7, (ii) does NOT regress align-F1/density-corr, and — the North Star — (iii) the user ArcViewers
   it and it feels more human ("who mapped this?", not "is this AI?").

Smoke command (re-verify after machine swap):
```bash
python scripts/reward_gate_poc.py --n 80 --json outputs/reward_gate_smoke.json   # expect AUC~0.90, GREEN
```

## 2026-06-08 Session — TASK-3 EVAL'D → NULL → CONFOUND RULED OUT → TASK 3 DEAD; SCOPED-V8 STACK EXHAUSTED

Evaluated the 06-07 overnight A/B and ran the prescribed confound test. **TASK 3 is dead.**

**End-to-end A/B (06-07 run, from each arm's `last.ckpt`, beat version_4, gate=loud_only):**

| arm | contour-follow | density-spear | gen_cv | align note-count |
|---|---|---|---|---|
| A_contour Expert     | 0.5214 | -0.057 | 0.205 | 1379 |
| B_control Expert     | 0.5014 | -0.010 | 0.197 | 1383 |
| A_contour ExpertPlus | 0.4567 |  0.124 | 0.191 | 1362 |
| B_control ExpertPlus | 0.5015 |  0.162 | 0.205 | 1354 |

End-to-end delta: **Expert +0.0199, ExpertPlus −0.0448** (contour HURT on Ex+). Both << +0.05 DoD ⇒ NULL.
Density-corr still flat (~0.12–0.16, all FAIL ≥0.41) — no regression, no gain. align note-counts ~equal.

**CONFOUND TEST (the gate before killing TASK 3) — ruled out.** Added env-gated `BS_PREPOST_OUT`
to `generate_v7_level` (deep-copies the beatmap and exports it BEFORE `postprocess_beatmap`, so the
parity-fix can't rewrite swing directions; production behavior unchanged when unset). Re-scored
contour-follow on the **pre-postprocess** token stream (`scripts/confound_prepost_2026-06-08.sh`,
log `logs/overnight/confound_prepost_2026-06-08.log`, out `outputs/2026-06-07/prepost/`):

| arm | PRE contour-follow |
|---|---|
| A_contour Expert     | 0.4076 |
| B_control Expert     | 0.4110 |
| A_contour ExpertPlus | 0.4273 |
| B_control ExpertPlus | 0.4651 |

PRE delta: **Expert −0.0033, ExpertPlus −0.0378** — contour arm NO BETTER (worse on Ex+) even before
postprocess. (Note pre-postprocess rates sit BELOW chance ~0.41 and postprocess RAISES them to ~0.50 —
the parity-fix wasn't erasing contour signal, it was *adding* the loose up/down alternation that
loosely tracks melody. The model never learned contour-following.) **→ TASK 3 KILLED (well-tested,
both end-to-end and pre-postprocess). Stage-2 swing DIRECTION is a mapper-subjectivity ceiling, same
as the Stage-1 ~0.60 val_f1 ceiling.** `--use-contour` stays OFF (default); version_10 layout +
version_4 beat remain production. Uncommitted: the `BS_PREPOST_OUT` dump in generate.py + the
confound script + version_13/14 layout dirs + outputs/2026-06-07.

**SCOPED-V8 STACK IS EXHAUSTED — every build bet came back null:** TASK 0 done (cohort eval neutral);
TASK 1 null (layering retrieval key WORSE than mean-MERT → TASK 4 KILLED); TASK 2 null (instr density
doesn't propagate to OUTPUT density, <0.41); TASK 3 null (contour not learned). **TASK 5 (sparse
long-range "DeepSeek" retrieval) is gated on preconditions that BOTH failed:** "only if S3/contour
helps" (it didn't) AND a better-than-MERT layering key for the sparse top-k (TASK 1 proved MERT wins).
So TASK 5 as written has no live premise either. **The two-stage MERT pipeline sits at a confirmed
quality plateau: align-F1 ~0.40, density-corr ~0.15 (flat ~8 NPS, ignores structure), contour ~chance,
late-song/final-chorus collapse persists.** This is a strategic fork for the user (see below) — NOT
auto-queuing another overnight.

**DECISION FORK (awaiting user):** (1) accept the pipeline as-is and ship the gate-fix wins; (2)
re-spec TASK 5 with a NEW key (not layering) — but its "if S3 helps" premise is gone; (3) step back
to a different lever entirely (the per-slot subjectivity ceiling keeps capping per-note metrics — the
honest North-Star question may need a different objective/eval than per-slot F1, e.g. learned reward /
human-preference, or a fundamentally different WHAT representation). Key notes status unchanged this
session: silent-drop FIXED (gate=loud_only), but flat-density / late-song-collapse / for-sport
diagonals all PERSIST and are NOT addressed by anything in the scoped-V8 stack.

## 2026-06-07 Session — TASK-3 BUILT + LAUNCHED (Stage-2 pitch-contour) (Top of Stack)

Implemented TASK 3 (the last live build item) and launched the overnight A/B.
**What shipped this session (code + smoke tests):**
- **Per-slot pitch contour → Stage-2 encoder.** `LayoutPhraseDataset(use_contour=True)`
  slices cols 7:10 of the already-cached `instr_beat_features` (lead_pitch/lead_dpitch/
  bass_pitch) into a `phrase_contour [P,3]` tensor, slot-aligned 1:1 with `phrase_mert`
  (**no new preprocess pass** — those columns already ship in 5319/5320 .pt). `LayoutPhraseModel`
  gains a guarded `contour_proj = Linear(3,d_model)` (None unless `use_contour`, so old ckpts
  load clean) added to the encoder input. Threaded through `encode`/`forward`/`generate_phrase`
  + `layout_module` (both fwd calls) + `train_layout.py --use-contour`.
- **Inference wiring.** `generate_v7_level(use_contour=…)` auto-detects from the layout ckpt
  (`model.use_contour`), reuses the same Demucs→transcription pass as `--use-instr`, builds a
  per-phrase contour padded like `phrase_mert`. `scripts/generate.py --use-contour/--no-use-contour`.
- **DoD eval `scripts/eval_contour_follow.py`** — fraction of vertical-swing notes whose swing
  sign (up=0,4,5 → +1; down=1,6,7 → −1; left/right/dot skipped) matches the lead Δpitch sign at
  that slot (deadband 0.05 on |dpitch| to skip flat/jitter). 0.5 = chance.
- Smoke: tests pass; forward+generate with contour changes logits Δ3.3 (not a no-op); tiny
  `--use-contour` train completes; control generate path unbroken; **eval on the existing
  no-contour version_10 map = 0.4257 (below chance)** — the baseline to beat.

**RUNNING NOW:** `scripts/overnight_2026-06-07.sh` (launched ~23:19, log
`logs/overnight/task3_contour_dod_2026-06-07.log`, out `outputs/2026-06-07/`). Two training
arms, single variable = contour: **A** = version_10 config (`--ctx-len 16`, d384/3enc/4dec,
song-mem 150) **+ `--use-contour`**; **B** = same recipe, no contour (control). ~3 h each (early-
stop ~ep18). Generate from each arm's **`last.ckpt`** (NOT best-val_token_acc — anti-correlates),
production beat version_4, `section_gate=loud_only`, Expert + ExpertPlus.

**DoD / how to read the verdict next session:** contour-follow(A) − contour-follow(B) **≥ +0.05
at BOTH difficulties** AND alignment-F1 / density-corr not regressed vs B ⇒ **TASK 3 MET** →
make `--use-contour` the Stage-2 default + ArcViewer check. If delta < 0.05 ⇒ **before killing
TASK 3, rule out the CONFOUND**: postprocess parity-fix rewrites **~48% of swing directions**
(observed "corrected 661/1380 violations") and can erase the model's contour choices. Re-run
the eval on the **pre-postprocess token stream** to disambiguate "model didn't learn it" vs
"parity-fix erased it." Only a pre-postprocess null kills TASK 3 → then TASK 5 / accept pipeline.
Summary block in the runner prints the table + verdict automatically.

## 2026-06-06 Session — TASK-2 INFERENCE DoD RAN → NULL → pivot to TASK 3 (Top of Stack)

Built the missing TASK-2 inference test (the only unfalsified piece of the per-instrument
thesis) and ran it. **Wired `instr_beat_features` into `generate_v7_level` Stage-1 inference**
(new `use_instr` arg on `generate_v7_level` + `--use-instr/--no-use-instr` on `scripts/generate.py`;
computes `compute_instrument_features` once per song at gen time, feeds per-128-window). New eval
`scripts/eval_density_corr.py` = Spearman(generated note density, ref onset density) over uniform
2s windows — decoupled from the energy section detector (unlike `eval_alignment`'s per-section).
DoD: **≥0.41** (the structure-PoC bar). Runner `scripts/overnight_2026-06-05.sh`; outputs
`outputs/2026-06-05/`, log `logs/overnight/task2_infer_dod_2026-06-05.log`. SO TIRED ROCK, 5 arms.

| arm | spearman | gen_cv | DoD |
|---|---|---|---|
| A instr, gate off, Expert | 0.153 | 0.259 | fail |
| A instr, gate off, ExpertPlus | 0.133 | 0.258 | fail |
| B baseline, gate off, Expert (control) | 0.060 | 0.204 | fail |
| B baseline, gate off, ExpertPlus (control) | 0.151 | 0.173 | fail |
| C instr, **loud_only**, Expert | **0.191** | 0.285 | fail |

**Verdict — TASK-2 NULL ON INFERENCE TOO.** Instr features *do* raise density variation (gen_cv
0.26 vs control 0.20) and on Expert beat the control (0.153 vs 0.060, Δ+0.093) — but ExpertPlus is
a wash and **nothing clears 0.41**. The r=0.41 the drum/instr density had *as an INPUT feature*
(structure PoC) does **not** propagate to r≥0.41 in *generated OUTPUT* density. So per-instrument
conditioning of Stage-1 is confirmed null on the metric that matters. **→ TASK 2 closed (null).
The live build item is now TASK 3 (Stage-2 pitch-contour for WHAT-cohesion).** Note `instr_proj`
inference path is shipped + smoke-tested (instr logit Δ0.68) but **not made default** — version_4
remains the production beat ckpt.

**TASK 3 is cheaper than written:** the per-slot contour (`lead_pitch`/`lead_dpitch`/`bass_pitch`,
cols 7–9 of the already-cached `instr_beat_features`) needs **no new preprocess pass** — just wire
those columns into `LayoutPhraseDataset`/`LayoutPhraseModel` as a per-note conditioning channel and
retrain Stage-2 (version_10 config). That retrain is the real overnight job.

**Status:** Overnight chain (06-04→05, post power-cut resume) ran scoped-V8 Stage-1 retrain +
TASK-1 retrieval-key eval. **Both came back negative.** (1) **TASK 1 DEAD (well-powered):** layering
fingerprint is a WORSE song-memory key than mean-MERT — AUC 0.824 < 0.848, and loses worst on
electronic (0.800 vs 0.864). The prelim "layering wins" was a 9-pair artifact. → **TASK 4 KILLED.**
(2) **TASK 2 null on val_f1:** Stage-1 `--use-instr` (`version_7`, d512/4L) best val_f1_avg_tol=0.600
@ep0 vs 0.603 baseline — 3rd confirmation the per-slot metric is a subjectivity ceiling. BUT val_f1
is the wrong yardstick; TASK 2's real DoD (inference-side density tracking w/ section gate OFF) was
NOT tested — instr never wired into `generate_v7_level`. **TASK 2 = inconclusive, not dead.**
Open decision: run the TASK-2 inference DoD test, pivot to TASK 3 (contour for WHAT-cohesion,
untouched), or accept ceiling + keep the gate-fix that fixed silent-drop. Full writeup in memory +
`docs/v8_0_poc_findings.md` addendum. Prior: ctx16+song-mem ON align F1 0.410 (`version_10`,
production); val_token_acc anti-correlates with alignment F1 — don't select checkpoints on it.
**North star:** A player plays a generated map and says *"who mapped this?"* — not *"is this AI?"*

**Full implementation plan:** [`docs/architecture_v7_plan.md`](docs/architecture_v7_plan.md)
**V6 post-mortem:** [`PROGRESS.md`](PROGRESS.md) — "V6 Post-Mortem" section

---

## 2026-06-02 (late) → 06-03 Overnight Session — V8-0 GATE RUN (Top of Stack)

**TL;DR:** Ran the V8-0 de-risk PoC (the hard gate). Outcome: **the full V8 WHEN-rebuild
is NO-GO; a scoped V8 is GO.** Shipped the two supported cheap wins and launched a
cohort-quality retrain. Full writeup: [`docs/v8_0_poc_findings.md`](docs/v8_0_poc_findings.md).

### What the gate found (data, not hunch)
basic-pitch installed on py3.12 via the **ONNX** backend (TF has no cp312 wheel). Per-stem
transcription (bass/vocals/other → basic-pitch, drums → multi-band librosa onset) on the test
song + **12 in-dataset songs with human maps**:

| finding | number | implication |
|---|---|---|
| transcribed pool covers human notes | 0.79 (±25ms) / 0.90 (±50ms) | richer pool than librosa (0.54/0.74) ✅ |
| **BPM-grid off-grid residual** | **0.7% (±50ms) / 6% (±25ms)** | **V7's 1/16 grid already represents 94–99% of human note timing** — refutes V8 Layer-2 ("trapped in BPM space") ❌ |
| per-instrument structure (bass riff, lead contour, breakdowns) | see `outputs/v8_poc/*/pianoroll.png` | real signal V7 lacks — supports V8 Layer-3 (WHAT) ✅ |

**Root-cause re-attribution:** the silent-drop is **Layer 1** (the section-threshold *gate*,
not the representation) — confirmed live: the energy detector labels SO TIRED ROCK `0–16s` as
"intro", so the ~13–15s drop was gated at 0.68. **Layer 2 (BPM grid) is NOT the timing flaw.**
**Layer 3 (no melodic anchor) IS real** and is the right target for a *scoped* V8.

### Shipped this session (code + tests + run)
1. **Section-gate fix** (Layer 1) — `generate_v7_level(section_gate="loud_only")` (new default).
   A section can only *lower* the onset threshold (densify a drop), never *raise* it (silence a
   region). New module helper `_build_section_threshold_vector` + 6 tests (`test_section_gate.py`).
   **Demonstrated run** on SO TIRED ROCK (production ckpts, ExpertPlus):
   | region | legacy gate | loud_only |
   |---|---|---|
   | intro 0–16s | 75 | **107** |
   | drop 12–16s | 19 | **32** |
   | outro 168–176s | **0 (silent)** | **5** |
   Maps: `outputs/v8_gatefix_{legacy,loudonly}.zip`.
2. **Cohort NPS filter** (orthogonal data fix) — `LayoutPhraseDataset(min_nps, max_nps)` +
   `train_layout.py --min-nps/--max-nps` + 3 tests (`test_cohort_filter.py`). Drops for-sport
   ExpertPlus density (>8 NPS = ~7% of Expert+) and near-empty maps.
3. **Cohort-filtered Stage-2 retrain — ✅ COMPLETE** (`logs/layout_phrase/version_12`,
   version_10 config + `--min-nps 4 --max-nps 8`). Best `val_token_acc=0.863` @epoch10 (vs
   version_10's 0.865 — filtering did NOT cost teacher-forced accuracy). Log
   `logs/overnight/v8_cohort_layout_2026-06-02.log`. **NOT yet evaluated** — see TASK 1 below.

### A second + third test refined the direction (2026-06-03, after user pushback)

**Test 2 — structure signal** (`scripts/v8_poc_structure.py`, 12 songs, 2s windows, Spearman
vs human note density). Per-instrument event activity predicts *where humans map notes* better
than V7's section detector: **drum density r=0.41, total 0.38, kick 0.34 > section_detector_rank
0.27**; bass/lead weak (~0.13 — they're WHAT not WHEN). → per-instrument events are a better
**structure/density signal** than the hand-tuned detector, not just a direction signal.

**User correction (important, accepted):** do NOT lean on drums — that was a rock-leaning sample;
for EDM the bass/synth *layering* carries structure. **Pass the full per-instrument layering
vector and let the model weight it per genre.** The generalizable input is the whole layering
picture, not any one stem.

**User insight — consistency via layering as a retrieval KEY (the big one):** the model already
has a long-range memory mechanism — **song-memory cross-attention attends over ALL ~150 phrase
fingerprints**. Its weakness is the *key*: those fingerprints are **mean-pooled MERT** (a timbre
average), too coarse to recognize "the drop at 14s == the drop at 4:00". Replace the key with a
**per-instrument layering + pitch-contour fingerprint** → the model can match analogous moments
and replay consistent notes (the original North-Star "same chorus, inconsistent patterns" bug).

**On `ctx_len=16`:** it's NOT arbitrary — ablation showed ctx16 > ctx0 > ctx32, and **ctx32
collapsed on the final chorus (drift)**. So *raw* long context is the wrong tool here; the
long-range job belongs to **sparse, content-addressed retrieval** on a good key (DeepSeek
MLA/NSA-style "attend to a good latent key, not everything"). Keep ctx16 for local flow; move
long-range consistency onto the better-keyed song-memory retrieval. This is the user's "DeepSeek
context optimization" north star, and the model's own drift behavior argues FOR it.

---

## ⇒ NEXT-SESSION IMPLEMENTATION PLAN — "Scoped V8" (per-instrument INPUT around the kept grid)

**Architecture in one paragraph:** Keep the 1/16 BPM grid as the output timing lattice (off-grid
rebuild stayed no-go). Add **per-instrument note events** (Demucs stems → basic-pitch for
bass/vocals/other, multi-band librosa onset for drums; code already in `scripts/v8_poc.py`) as
**INPUT/conditioning** in three places: (S1) Stage-1 density, (S2) Stage-2 direction, (S3)
song-memory retrieval key. Then retrain. Each is independently shippable.

### TASK 0 — Evaluate the version_12 cohort retrain (cheap, do first) ✅ DONE 2026-06-03
- [x] Generated v12 + v10 @ Expert/ExpertPlus (`--section-gate loud_only`), eval_alignment in
      `outputs/task0/`, 2 leaderboard rows added.
- **Verdict: NPS-4–8 cohort filter did NOT lower generated density** — all maps stayed ~7.8–7.9 NPS
      (pinned at the cap); F1 a wash (EP v12 0.4151 vs v10 0.4106; Ex v12 0.395 vs v10 0.399).
      Density is set by Stage-1 thresholds/section gate, NOT the Stage-2 layout cohort → reinforces
      TASK 2. **Keep version_10 as inference default.**

### TASK 1 — ❌ DEAD (2026-06-05, well-powered): layering key is WORSE than mean-MERT → TASK 4 KILLED
**Do this BEFORE the S3 retrain — it gates whether the key-swap is worth a training cycle.**
**RESULT (definitive):** `--n 60 --difficulty Expert`, 60 songs / 25,950 pairs →
`outputs/v8_poc/retrieval_key_2026-06-04.json`. mean-MERT AUC **0.848** > layering **0.824**
(Δ−0.025), and layering loses worst on **electronic** (mert 0.864 vs layering 0.800, 15 songs/10k
pairs) — the genre it was predicted to win. The preliminary "layering 0.902>0.893" was a 9-pair
artifact. DoD FAILED → **do NOT build TASK 4.** The North-Star consistency fix is not a layering key.
- [x] `scripts/v8_poc_retrieval_key.py` built (GPU-free: layering key = pooled cached
      `instr_beat_features` per phrase vs mean-MERT `phrase_fingerprints`; ROC-AUC of key-cosine
      predicting human-identical phrase pairs over (slot,hand,x,y) occupancy).
- [~] **Preliminary (5 songs preprocessed so far): layering AUC 0.902 > mean-MERT 0.893; on the lone
      electronic song layering 0.873 > 0.828 (Δ+0.045 — the predicted EDM win).** UNDERPOWERED
      (9 identical pairs). `outputs/v8_poc/retrieval_key.json`.
- [x] **RE-RAN `--n 60 --difficulty Expert` (2026-06-05)** → definitive NO (see header above).
- DoD: layering-key AUC > mean-MERT AUC (esp electronic) → ❌ FAILED. TASK 4 killed.

### TASK 2 (S1) — ❌ NULL (2026-06-06): instr density doesn't propagate to OUTPUT density (<0.41). CLOSED.
> Inference DoD ran (`scripts/overnight_2026-06-05.sh`): 5 arms, best Spearman 0.191 « 0.41 bar.
> instr_proj path shipped+smoke-tested but NOT default; version_4 stays production. Detail below.
- [x] `data/instrument_features.py` — Demucs→per-stem transcription→`events_to_slot_features`
      `[N_slots, 10]` (kick/snare/hat/bass/vocals/lead density + n_active_stems + lead_pitch +
      lead_dpitch + bass_pitch), mass-preserving onset interp, same 1/4-note grid. 11 tests pass.
- [x] `scripts/preprocess_instruments.py` — caches `instr_beat_features` (fp16, non-destructive).
      Smoke: ~3s/song, ~98–100% slots active; **Spearman(drum_density, human density)=0.52 on 1ccca.**
- [x] `models/beat_classifier.py` + `beat_module.py`: dedicated `instr_proj` sum-fused path +
      `instr_features` threaded through. `beat_dataset.py`: reads key + `require_instr`.
      `train_beats.py --use-instr`.
- [x] **RAN (2026-06-05, overnight chain post power-cut).** Preprocess finished 5319/5320; Stage-1
      `--use-instr --d-model 512 --n-layers 4 --n-heads 8` → `logs/beat_classifier/version_7`,
      log `logs/overnight/instr_stage1_train_2026-06-04.log`. **Best val_f1_avg_tol=0.600 @ EPOCH 0,
      never improved (early-stopped ep8)** vs version_4 baseline 0.603. Instr features did NOT move
      val_f1 — 3rd confirmation (struct=0.598, instr=0.600) the per-slot metric is a subjectivity
      ceiling. Best ckpt: `version_7/checkpoints/beat-epoch=00-val_f1_avg_tol=0.600.ckpt`.
- [ ] ⚠️ **val_f1 gate (≥0.603) NOT met, BUT val_f1 is the WRONG metric** (per-slot binary acc,
      known to anti-correlate w/ alignment F1). The real DoD below was NEVER tested — instr is still
      not wired into `generate_v7_level`. **DECISION PENDING:** run the inference-side test (wire
      `compute_instrument_features` per song at gen time, gate OFF, measure generated-vs-human
      per-section NPS corr) to truly adjudicate TASK 2 — OR pivot. Do NOT delete `_SECTION_THRESHOLDS`
      yet (the gate-fix `section_gate="loud_only"` is the only thing that demonstrably fixed silent-drop).
- DoD: generated per-section NPS tracks human density with NO section gate; structure-corr ≥ 0.41.
      **(UNTESTED — this is the only unfalsified piece of the per-instrument thesis.)**

### TASK 3 (S2) — ❌ DEAD (2026-06-09): contour not learned, confirmed end-to-end AND pre-postprocess.
> Built + ran A/B (06-07, version_13 contour vs version_14 control). End-to-end contour-follow delta
> Expert +0.020 / Ex+ −0.045 (« +0.05). Confound (parity-fix) RULED OUT: pre-postprocess delta
> −0.003 / −0.038 (`scripts/confound_prepost_2026-06-08.sh`). Stage-2 swing DIRECTION is a
> subjectivity ceiling. `--use-contour` stays OFF. version_13/14 ckpts droppable. Detail at TODO top.
> _Original spec (for reference only — DONE/disproven):_
- [x] contour wired WITHOUT new preprocess (cols 7:10 of cached `instr_beat_features`).
- [x] `layout_dataset.py`/`layout_model.py` `use_contour` + `contour_proj`; `eval_contour_follow.py`.
- [x] Retrained + measured contour-follow → NULL (above).
- DoD: contour-follow rate up vs V7; ArcViewer: fewer "diagonal swings for sport".

### TASK 4 (S3) — ❌ KILLED (2026-06-05): gated on TASK 1, which FAILED well-powered.
**Gated on TASK 1 passing — IT DID NOT.** Layering key AUC 0.824 < mean-MERT 0.848 (worse on EDM).
Do NOT build this. The North-Star chorus-consistency fix does not come from a layering retrieval key.
Steps below retained for the record only.
- [ ] Compute a per-phrase **layering+contour fingerprint** (concat: per-stem activity profile +
      lead contour summary) → store as new `.pt` key alongside `phrase_fingerprints`.
- [ ] `models/layout_model.py` song-memory cross-attn: key on the layering fingerprint instead of
      mean-MERT. `generation/phrase_index.py`: same key for hard-retrieval fallback.
- [ ] Keep `ctx_len=16` (local). Retrain.
- DoD: **consistency metric** — note-pattern similarity between human-identical repeated sections
      (e.g. chorus1 vs chorus2) is higher than version_10 baseline. This is the North-Star test.

### TASK 5 — ⛔ NO LIVE PREMISE (2026-06-09): both preconditions failed.
> Was gated on "only if S3/contour helps" (TASK 3 DEAD) AND a better-than-MERT layering key for the
> sparse top-k (TASK 1 proved mean-MERT WINS, AUC 0.848>0.824). Neither holds → not actionable as
> written. Superseded by the reward/preference direction at TODO top.
- [ ] ~~sparse top-k song-memory by layering-key similarity (NSA-style)~~ — shelved, premise gone.

### Genre note (carry forward)
User listens to a lot of EDM and wants generalization. Several tests so far skewed rock. For
TASK 1/2/3 **stratify by `mod_requirements.genre`** and verify EDM specifically (bass/synth
layering should matter more there than drums). The cohort/leaderboard should not be rock-only.

### Status of the speculative V8 rebuild plan (below) — SHELVED by the gate
The `docs/architecture_v8_plan.md` full rebuild and the V8-1..V8-5 work breakdown below are
**superseded** — their premise (BPM grid can't represent the music) was tested and rejected.
The continuous-time event-selector / deleting the grid label path are SHELVED. What survives from
that doc: per-stem transcription (now used as INPUT/conditioning, not a WHEN backbone) and the
cohort filter (done). Full reasoning: `docs/v8_0_poc_findings.md`.

---

## 2026-06-01 → 2026-06-02 Overnight Results

### Song-memory ablation (✅ ran via V7 harness)
Queue `experiments/queue/v7_layout_songmem_ablation.yaml`, 2 arms, both capped at 200min.
Held ctx_len=16 fixed and flipped song-memory ON/OFF, both eval'd in-harness (unlike the
confounded v3-vs-v8 comparison). Completes the ctx_len × song-mem grid:

| ctx_len | song-mem | val_acc | align F1 | version |
|---------|----------|---------|----------|---------|
| **16**  | **ON**   | 0.865   | **0.4099** | version_10 (NEW, best overall) |
| 0       | ON       | 0.856   | 0.4059   | version_8 |
| 16      | OFF      | 0.868   | 0.4027   | version_11 (NEW, in-harness version_3 repro) |
| 32      | ON       | 0.868   | 0.3978   | version_9 |

- **Song-memory helps at ctx16** (+0.007 F1). Reverses last night's tentative "song-mem may
  hurt" — that compared v3 (legacy eval) vs v8 (two knobs different).
- **ctx16 is the alignment sweet spot** (0.410 > 0.406 > 0.398 across ctx{0,16,32}). Extends
  "don't go beyond 16" with "16 > 0 too".
- **val_token_acc anti-correlates with align F1**: song-mem ON had lower val (0.865 vs 0.868)
  but higher F1. Stop selecting inference checkpoints by val_token_acc.
- **In-harness version_3 repro = 0.403, not the legacy 0.415** quoted before → legacy and
  harness eval paths differ; old 0.415 wasn't comparable. This re-run was the right call.
- **NEW failure mode — final-chorus collapse.** Per-section ON−OFF mostly +0.02..0.04 EXCEPT
  final chorus 160-164s: ON 0.327 vs OFF 0.537 (−0.21). Same spot ctx32 collapsed last night.
  Collapse tracks song-memory aggressiveness, not ctx_len. Suspect song-memory retrieval
  over-commits to an earlier chorus that doesn't match the final chorus's onsets.

**Production config:** ctx16 + song-memory ON (max_song_phrases=150). Inference ckpt
`logs/layout_phrase/version_10/checkpoints/layout-epoch=09-val_token_acc=0.865.ckpt`.

**Caveats:** ~0.01 F1 spread, single test song — needs a 2nd song to confirm (still blocked:
only SO TIRED ROCK present). Outro 168-176s still 0/0 in both (section detector, unchanged).

### ⚑ MAJOR: V7 input representation is the fundamental flaw → V8 designed
User review 2026-06-02: every V7 map is incohesive ("diagonal swings for sport"), NPS too high,
and **zero notes at the ~13-15s drop on every song**. Root-caused in code to 3 audio-blind layers:
(1) note timing overridden by 6 hand-tuned section thresholds (drop@13s lands in "intro"→gated 0.68
→silenced), (2) MERT mean-pooled onto a BPM grid (onsets blurred, no phase lock), (3) only Demucs
"other" stem → no per-instrument line for directions to follow. **Full V8 blueprint written:**
[`docs/architecture_v8_plan.md`](docs/architecture_v8_plan.md) — symbolic per-stem transcription
(basic-pitch) NoteEvent backbone, event-driven WHEN (no section gate), pitch-contour-conditioned
WHAT, phased + gated on a V8-0 PoC. Also: filter training to Expert / NPS 4-8 (Expert+ teaches
for-sport swings). **Next concrete step: V8-0 PoC** — install basic-pitch, transcribe SO TIRED ROCK,
prove (a) drop yields a dense onset cluster V7 misses, (b) transcribed-onset alignment beats 0.41.

### V8 Work Breakdown (from [`docs/architecture_v8_plan.md`](docs/architecture_v8_plan.md))

Phases are sequential unless marked ∥ (parallelizable). **V8-0 is a hard gate** — do not start
V8-1+ until the PoC goes green.

#### V8-0 — De-risk PoC ⚑ GATE
- [ ] Install transcription deps: `uv pip install basic-pitch pretty_midi` (+ add to `pyproject.toml`).
- [ ] PoC script: Demucs-separate SO TIRED ROCK, run basic-pitch per stem (drums via multi-band
      `librosa.onset`), dump a `NoteEvent` list + piano-roll plot.
- [ ] **Validate (a):** the ~13–15s drop produces a dense onset cluster that V7's generated map misses.
- [ ] **Validate (b):** transcribed-onset → human-map alignment F1 **beats current 0.41** (reuse
      `scripts/eval_alignment.py` with transcribed onsets as the "generated" set).
- [ ] **Validate (c):** lead-stem (`other`) pitch contour visibly tracks the melody (eyeball).
- [ ] **Go/no-go writeup** in leaderboard/PROGRESS; only proceed to V8-1 if (a)+(b) pass.

#### V8-1 — Transcription preprocessing (depends V8-0)
- [ ] `data/note_events.py` — `NoteEvent` dataclass (onset_sec, dur_sec, pitch, stem, salience),
      .pt (de)serialization, piano-roll render helper.
- [ ] `data/transcribe.py` — per-stem basic-pitch + drum-band onset → merged `NoteEvent` stream.
- [ ] Tune `other`-stem `salience > τ` gate + within-window chord-merge (kills distorted-guitar smear).
- [ ] Batch-transcribe all 5320 songs → new `.pt` keys `note_events`, `lead_contour` (non-destructive).
- [ ] Sanity report: median events/sec (expect ~2–6), per-song coverage, failures.

#### V8-2 — Label representation (depends V8-1)
- [ ] `data/event_dataset.py` — match each GT Beat Saber note to nearest `NoteEvent` within ±ε ms;
      emit selected/hand/spatial labels per event.
- [ ] Report **unmatched-GT residual** (GT notes with no nearby event); decide ε, or add a
      fallback grid-candidate channel if residual is large.

#### V8-3 — Stage 1 Event Selector (depends V8-2)
- [ ] `models/event_selector.py` — sequence model over the event stream → P(note), P(hand=L/R).
- [ ] Training module + train run; **delete** dependence on `extract_beat_labels` BPM-grid path.
- [ ] Sanity: drops dense / breakdowns sparse without any section-threshold gate; report selection F1.

#### V8-4 — Stage 2 contour conditioning (depends V8-3)
- [ ] `data/layout_dataset.py` — add lead pitch-contour conditioning channel (relative Δpitch).
- [ ] `models/layout_model.py` — accept contour conditioning.
- [ ] `generation/phrase_index.py` — key retrieval on contour-segment similarity (not mean-MERT).
- [ ] Train; report a **directional-cohesion metric** (contour-follow rate) vs V7 baseline.

#### V8-5 — Inference + harness + ArcViewer (depends V8-4)
- [ ] `generation/generate.py::generate_v8_level`; **delete** `_SECTION_THRESHOLDS`, the per-slot
      threshold vector, `_apply_density_curve`, `_compute_adaptive_threshold`, and section-as-note-gate
      (sections may stay for lighting only).
- [ ] `research/spec_v8.py` + `research/runner_v8.py` mirroring the V7 harness; `auto_research_v8.py`.
- [ ] End-to-end generate on SO TIRED ROCK → **ArcViewer human play (the real DoD):** drop has notes,
      swings cohere, NPS in band.

#### ∥ Orthogonal data-quality fix (independent of V8 phases)
- [ ] Filter training cohort to **Expert-only, or all difficulties capped at NPS 4–8** (Expert+ teaches
      ergonomically hard "for-sport" swings). Cheap cohort filter; fold into the V8 retrain.

### Follow-ups (next session)
- [ ] **Investigate final-chorus collapse** at 160-164s: dump song-memory cross-attn weights
  there, or try gating/attenuating song-memory on the last phrase, then re-eval. (V7-era; may be
  moot if V8 proceeds.)
- [ ] **Add a 2nd test song** to `data/test_songs/` and re-run the 4-point grid to confirm the
  ordering generalises (still can't fetch over the wire — manual drop).
- [ ] Carry forward: ctx16 + song-mem ON is the inference default going forward.

---

## 2026-05-26 → 2026-05-27 Overnight Results (Top of Stack)

### Section detector replacement (✅ shipped)
- `data/audio.py::detect_sections_energy_percentile()` — new RMS-percentile detector. Top-25 % windows → `drop`, bottom-25 % at song edges → `intro` / `outro`. Replaces chroma+MFCC agglomerative clustering as the primary path in `generate_v7_level`; clustering kept as fallback.
- **Why**: the clustering detector collapsed everything after ~40 s into a single "outro" cluster on EDM and stable-timbre rock, which mapped to threshold 0.72 and produced a *pause at the drop* in ArcViewer review.
- Alignment-eval results, same test song, ExpertPlus, ±50 ms vs. drum+melody onset union:

  | Map | Notes | Overall F1 | drops 16-32s | drops 144-160s | last 8s outro |
  |-----|-------|------------|--------------|-----------------|----------------|
  | `outputs/v7_section_aware.zip` (clustering) | 1036 | **0.375** | 0.561 | 0.225 | 0.049 |
  | `outputs/v7_energy_sections.zip` (energy)  | 1270 | **0.415** | 0.583 | 0.366 | 0.000 |

  Late-song drops (the parts the user said "got worse over the song") improved most.
- Tests: 3 new cases in `tests/test_audio.py`; full suite 395 + 4 xfailed + 5 xpassed.

### Beat Classifier Run 6 — struct features (negative result)
- Config: d=512 / 4-layer / mix + difficulty + **struct (rms, onset_strength, bass/mid/high, centroid, section_id, section_progress)**
- Best `val_f1_avg_tol` = **0.598** at epoch 18 → `logs/beat_classifier/version_6/checkpoints/beat-epoch=18-val_f1_avg_tol=0.598.ckpt`
- Previous best (version_4, no struct) = 0.603. **Struct features did not lift the metric**; they slightly underperformed within run-to-run noise.
- Interpretation: MERT already encodes RMS/timbre/onset content. The hand-engineered 8-dim path is redundant. The 0.60 ceiling continues to look like mapper-choice subjectivity (different mappers select different subsets of the same drum hits).
- **Implication**: don't bother wiring `compute_structure_features()` into `generate_v7_level` Stage 1 — it isn't worth the inference complexity if the model can't use it. Use version_4's checkpoint for inference going forward.

### New eval tool — `scripts/eval_alignment.py` (✅ shipped)
Compares a generated map to librosa-detected onsets on the Demucs drum + melody stems, with ±tolerance windowing and per-section breakdown.
```
python scripts/eval_alignment.py \
  --audio data/test_songs/<song>.mp3 \
  --map outputs/<map>.zip \
  --difficulty ExpertPlus --tolerance-ms 50 \
  --json outputs/<date>/alignment.json
```
Use this to answer "do generated notes line up to real musical events" without ArcViewer. Per-section P/R/F1 surfaces *which* sections drift (precision low → "random notes on top of nothing"; recall low → "missing the obvious beats").

### Bugs found and fixed this session
- **`a51022c` broke checkpoint loading.** Adding `struct_proj` to `BeatClassifier` made strict `load_from_checkpoint` fail on any pre-struct checkpoint (`Missing key(s): model.struct_proj.weight`). This is why a `python scripts/train_beats.py …` started at 17:18 today stalled — but more importantly **any inference call also failed silently for users on the old checkpoint**. Fix: `generate.py` now loads with `strict=False`. The struct_proj weights are uninitialised in that path, but they're a no-op because we never pass `struct_features=` at inference.
  - Follow-up: consider adding a defensive `state_dict` check in `BeatLitModule.load_from_checkpoint` so the silent-mismatch failure mode doesn't recur on the next field addition.
- **`scripts/generate.py` takes `audio` as a positional argument**, not `--audio`. The first overnight launch silently exited because the script wrapper used `--audio`. Documented in `scripts/overnight_2026-05-26.sh`.
- **Demucs returns `[channels, samples]` stereo**. `librosa.onset.onset_detect` errors with `sparse=True does not support 2-dimensional inputs`. `scripts/eval_alignment.py::_separate_stems` now collapses to mono before onset detection.
- **`@dataclass(slots=True)` removes `__dict__`**. `eval_alignment.py` initially serialized with `.__dict__` → `AttributeError`. Use `dataclasses.asdict()`.

### Follow-ups (next session, prioritised)

#### High — user-flagged complaints not yet fully resolved
- [ ] **Add a clear-drop test song to `data/test_songs/`**. User wants a known EDM-style track with an unambiguous build → drop to validate that the energy-percentile detector and Stage 1 thresholds actually fire at the drop. (Couldn't add over the wire — drop a file in manually.)
- [ ] **Verify in ArcViewer** that `outputs/v7_energy_sections.zip` no longer has the *pause at the drop*. Numbers say it should be fixed; visual confirmation needed.
- [ ] **"Random horizontal notes"** — this is the X-column 70 % ceiling. Use `eval_alignment.py` per-section to see which sections have the worst precision (those are where "random" notes live). The bridge 76-112 s and chorus 124-136 s sections both have precision < 0.45. Probably needs Stage 2 work — see "Architectural lessons" in PROGRESS-equivalent below.

#### Medium — known gaps
- [ ] **Generate phrase context-buffer bug suspect**. Per-section F1 degrades over song time even with the new detector (drop @ 16-32 s F1=0.58 → drop @ 144-160 s F1=0.37). Suspect either (a) the cross-phrase context buffer `_prev_ctx_*` builds up drift across many phrases, or (b) position-encoding extrapolation in the layout decoder. Worth an ablation: regenerate with `--ctx-len 0` (no cross-phrase prefix) and see if late-song F1 holds up better.
- [ ] **Outro detection bias**. The energy detector still flags 168-176 s as outro and generates 0 notes there, but librosa finds 42 onsets in that range (the song fades but still has events). Either the detector's "tail low → outro" rule is too aggressive (last 8 s should arguably be `verse` if energy isn't actually low) or 0-NPS is the desired behaviour. Decide via ArcViewer.
- [ ] **Stage 1 inference does not use difficulty**. `generate_v7_level` passes `diff_t` only to the layout model, not to the beat classifier. version_4/version_6 both train with a difficulty embedding — the inference path is missing that signal.

#### Low — research / nice-to-have
- [ ] Drop the struct-features code path entirely if a second run also fails to help. Currently `BeatClassifier` carries it as dead weight at inference.
- [ ] MERT-vs-transcription experiment (user question): MERT is opaque about *which musical events* it sees. For ground-truth alignment we use librosa onsets. A heavier-weight alternative is a pitch-tracker like `basic-pitch` to enumerate guitar/vocal note onsets explicitly — would give the alignment eval finer signal than spectral-flux onsets.

---

## Why V6 Failed (Short Version)

V6 collapsed two separate problems into one autoregressive token stream:

1. **WHEN** should a note appear? (beat/onset timing)
2. **WHAT** should the note look like? (spatial layout, hand, direction)

The Δt token was doing all the work for Problem 1, but cross-entropy loss on Δt tokens
has no audio-aligned gradient for timing — the 3-second audio context covers only 1/6 of
the 18-second event window. The model learned the statistical Δt distribution, not
audio-to-beat mapping. Every hyperparameter sweep, aux loss, and epoch budget increase
hit the same ceiling: ~1 NPS on a 4–10 NPS target.

Additionally: even if timing were fixed, the 3-second context window causes cross-song
drift. The same guitar riff appearing at bar 8 and bar 40 produces inconsistent note
patterns because the model has no memory of what it did at bar 8.

---

## V7 Architecture — Three Coordinated Changes

### Change 1 — Pretrained Audio Understanding (replaces scratch AudioEncoder)

**Demucs** (`htdemucs`) separates audio into stems before encoding:
- `drums` stem → cleaner beat signal (drums are nearly 1:1 with Beat Saber notes)
- `other` (melody) stem → instrument-specific features for layout

**MERT-v1-95M** (frozen, HuggingFace `m-a-p/MERT-v1-95M`) encodes each stem:
- Trained on massive music corpora via masked acoustic modeling
- Produces frame-level embeddings at 75 Hz (dense enough for 1/16-note resolution)
- Benchmarked at ~0.94 AUC on beat tracking tasks out of the box
- Replaces the scratch-trained `models/audio_encoder.py` entirely

### Change 2 — Explicit Two-Stage Separation (solves the timing problem)

**Stage 1: Beat Classifier** — small MLP on drum MERT features
- Input: `drum_mert[beat_slot]` — MERT features pooled to 1/4-note grid
- Output: `P(left_note)`, `P(right_note)` per beat slot
- Loss: weighted binary cross-entropy (ground truth from existing swing_tokens)
- This gives Stage 2 an explicit onset schedule — it never has to predict WHEN

**Stage 2: Layout Generator** — autoregressive, conditioned on known positions
- Input: confirmed beat position (from Stage 1) + MERT features + retrieval context
- Output: `[KIND, X, Y, DIR, FIELD_D]` per note — **no HAND, no Δt tokens**
- HAND is given by the beat slot (left or right). Δt is gone — timing is external.
- Saber-state conditioning (12-dim) preserved from V6.

### Change 3 — Cross-Song Phrase Memory (solves the consistency problem)

**PhraseIndex** — cosine similarity lookup over MERT phrase fingerprints:
- Before generation: segment full song into 4-bar windows, fingerprint each with mean MERT
- At generation: for each window, look up the k nearest prior windows in the same song
- If `max_similarity > 0.85`: **hard retrieval** — replay the stored note pattern as conditioning
- If no match: generate freely, then record the pattern for future windows
- Result: the second chorus produces nearly identical note patterns to the first chorus

Start with hard retrieval; switch to soft (cross-attention over retrieved tokens) only
if the output is perceptibly too repetitive.

---

## What Survives From V6

| Component | Status |
|-----------|--------|
| Swing-event grammar (`data/swing_tokenizer.py`) | Keep — just remove HAND + Δt from Stage 2 token stream |
| Saber-state extractor (`data/saber_state.py`) | Keep |
| Grammar-constrained decoder (`generation/beam_search_v6.py`) | Keep — simplify (shorter grammar) |
| Postprocessor (`generation/postprocess.py`) | Keep |
| Lighting rules (`generation/lighting_rules.py`) | Keep |
| Training infrastructure (Lightning, Hydra configs) | Keep |
| Cohort data + splits | Keep |
| Leaderboard / auto-researcher harness | Keep |

## What Gets Replaced

| Component | Replacement |
|-----------|-------------|
| `models/audio_encoder.py` (scratch mel transformer) | MERT-v1-95M wrapper (frozen) |
| `training/seq_module.py` V6 sequence module | `training/beat_module.py` (Stage 1) + `training/layout_module.py` (Stage 2) |
| `data/dataset.py::SwingSequenceDataset` | `data/beat_dataset.py` + `data/layout_dataset.py` |
| Windowed full-song Δt inference | Beat-slot iteration (Stage 1 schedule → Stage 2 per onset) |
| `dt_density_alpha`, `bomb_hand_weight` aux losses | Not needed — timing is now explicit |

---

## Phase Plan

### V7-0 — Dependencies + Proof of Concept ✅ DONE (2026-05-15)
- [x] `uv pip install demucs transformers` in venv; added to `pyproject.toml`
- [x] Demucs `htdemucs` separates test song into 4 stems in ~2s on RTX 5090
- [x] MERT-v1-95M produces `[13210, 768]` at 75 Hz for 176s test song (correct)
- [x] Beat grid: 1444 slots at 1/4-note resolution (9.1 MERT frames/slot at 123 BPM)
- [x] sklearn logistic regression (same-song, frozen MERT): **F1_avg = 0.59** → PASS

**DoD met.** Script: `scripts/v7_poc.py`

### V7-1 — Preprocessing Pipeline ✅ DONE (2026-05-17)
- [x] `scripts/preprocess_v7.py` written and tested on single song
- [x] Demucs → MERT pipeline: drum stem + melody stem encoded to beat grid
- [x] Phrase fingerprints (4-bar windows) computed and stored
- [x] All keys written to `.pt` files in fp16 (non-destructive)
- [x] **Full dataset run complete:** 5319/5320 songs have V7 features (99.98%)
  - 1 unrecoverable: song `3aa51` (corrupted zip, no audio)
  - OOM fix shipped: `mert_encoder.py::extract_features` now chunks long audio at 30s
    (`_CHUNK_SECS = 30`) — songs up to 39 min now process without OOM
- [ ] `frame_index.json` update deferred — not blocking training

**DoD met.**

### V7-2 — Beat Grid Labels ✅ DONE (2026-05-15)
- [x] `data/beat_grid.py::extract_beat_labels()` — parses swing_tokens → binary left/right per slot
- [x] `beat_labels_from_pt()` — convenience loader from a .pt dict
- [x] Validated on `1ccca.pt`: 66L + 66R notes detected, 14.1% positive rate (confirms pos_weight=6.0)
- [x] Labels computed on-the-fly at dataset load time (no separate precompute step needed)

**DoD met.**

---

### V7-3 — Stage 1: Beat Classifier 🔧 RUN 3 PLAN (2026-05-20)

#### Run 2 Result (2026-05-19 → 2026-05-20)
- Best `val_f1_avg = 0.442` at **epoch 0**, then 10 epochs of no improvement → early stop at epoch 10.
- Run 1 was 0.422. Run 2's fixes (pos_weight 6.0→3.6, mix-stem fusion, phase embedding) moved the needle ~2 points.
- "Peaks at epoch 0 then decays" is the signature of a frozen-encoder head saturating against an irreducible label-noise floor — the head extracts everything the features can explain in one pass, then overfits.

#### Audit Findings (2026-05-20)

Re-derived diagnosis on Run 2 results. Two structural issues remain on top of any subjectivity ceiling:

1. **No in-model difficulty conditioning.** `BeatDataset.__getitem__` returns `difficulty` but `BeatClassifier.forward(drum, mix, slot_offset)` never consumes it. With Expert (~3 notes/bar) and ExpertPlus (~6 notes/bar) pooled, the same drum hit gets label `0` in one and `1` in the other; the model can only predict the marginal.
2. **Exact-slot F1 is too brutal.** A prediction one slot off (≈125 ms at 120 BPM, subdiv=4) is currently double-counted (FP + FN). MIR-standard onset evaluation uses a ±tolerance window (typically ±50 ms or ±1 slot). Our reported F1 is systematically below the inter-mapper agreement floor.

Looked-for and confirmed absent (not regressing for tonight; documented as follow-up):
- Mapper-cohort conditioning: cohort scripts (`scripts/cohort_eda.py`, `compute_cohort_reference.py`) exist but the V7 preprocessing didn't write `mapper` into `mod_requirements` — value is `None` for every `.pt` file. Blocked on a preprocessing backfill pass.
- Density-regression target instead of binary BCE per slot: bigger redesign, not 1-session-safe.

#### Run 3 Plan (overnight, 2026-05-20)

Code changes for this run:

1. **`models/beat_classifier.py`** — add `nn.Embedding(N_DIFF, d_model)` summed into the input post-`input_norm`. `forward(drum, mix, difficulty, slot_offset)`.
2. **`training/beat_module.py`** — read `difficulty` from batch and plumb through to the model. Add a tolerance-window onset F1 metric (`val_f1_avg_tol`) alongside the exact-slot metric.
3. **`data/beat_dataset.py`** — already returns `difficulty`; no change.
4. **`scripts/train_beats.py`** — no signature change; tolerance value (`--tolerance-slots`, default 1) exposed for ablation.

Tolerance metric semantics (implementation note for the audit step):
- A predicted positive at slot `t` matches a label positive at any slot in `[t - K, t + K]` (default K=1, ≈125 ms at 120 BPM).
- Greedy nearest-match: walk predicted positives in order, each can match at most one label, each label matches at most one prediction.
- Reported per-hand and averaged. Logged as `val_f1_avg_tol` (don't replace `val_f1_avg` — keep both so we can see the gap).

**Run 3 command:**
```bash
python scripts/train_beats.py \
  --max-epochs 30 \
  --batch-size 64 \
  --pos-weight 3.6 \
  --patience 8 \
  --difficulties Expert ExpertPlus \
  --tolerance-slots 1
```

**Success criteria:**
- `val_f1_avg_tol` ≥ 0.65 → tolerance metric alone explains the gap, model was always fine
- `val_f1_avg` ≥ 0.55 with diff-embedding → conditioning unlocks the pooling-noise headroom
- Both: ready to move to Stage 2 training
- Neither: confirms subjectivity ceiling, escalate to density-regression or per-mapper plan

#### Earlier Run History (for reference)

#### Audit + Fix Pass (2026-05-19) — produced Run 2

#### Audit + Fix Pass (2026-05-19)

Code changes applied this session (`git diff` shows the full set):

- `models/beat_classifier.py`
  - Added `mix_dim` parameter; `mix_proj` Linear(768→d_model) added in parallel with `drum_proj`
  - Drum + mix projections sum-fused → input `LayerNorm` for training stability
  - Learned **phase embedding** indexed by `(slot + slot_offset) % 16` — gives the model
    explicit downbeat/within-bar phase, independent of pos_emb (which is window-relative)
  - `forward(drum_features, mix_features, slot_offset)` — backward-compat: mix may be None
- `data/beat_dataset.py`
  - Requires both `drum_beat_features` and `mix_beat_features` keys
  - Returns `mix_features` and `slot_offset` per sample
  - Beat labels cached per (song, difficulty) — was recomputing per-window (O(W) wasted work)
- `training/beat_module.py`
  - Default `pos_weight = 3.6` (was 6.0 — measured positive rate is 21.8%, not 15%)
  - `forward(drum, mix, slot_offset)` plumbed through training_step/validation_step
- `scripts/train_beats.py`
  - `--pos-weight` default 3.6, added `--mix-dim` (set 0 to disable), added `--patience`
  - Patience wired to `EarlyStopping` (was hardcoded to 5)

Param count went from ~1.0M → ~2.0M (mix_proj 200K + phase_emb 4K + slightly larger
input path). Still trivially small for our dataset; no overfitting risk added.

#### Run 1 Results (2026-05-17) — kept for reference

#### Run 1 Results (2026-05-17)
- Dataset: 187,855 train windows / 11,251 val windows from 4,457 songs
- Best checkpoint: `logs/beat_classifier/version_0/checkpoints/beat-epoch=03-f1=val_f1_avg=0.422.ckpt`
- **val_f1_avg = 0.422** at threshold 0.5 (target: 0.80) — early stopping at epoch 8
- Best achievable with threshold tuning: **~0.46 at threshold 0.65** — still far short

#### Post-Mortem: Why It Failed

**Root cause: low precision, not low recall.**

At the optimal threshold (0.65):
```
prec=0.33  recall=0.65  f1=0.46
```
The model predicts 3-4× more positives than ground truth. It detects drum hits well
but Beat Saber notes only cover a *subset* of drum hits — different mappers choose
different subsets. The model has no signal to make that distinction.

**Two specific bugs:**

1. **`pos_weight` miscalibrated**: Set to 6.0 (designed for 15% positive rate).
   Actual dataset positive rate is **21.8%** (measured across val set).
   Correct value: `neg_rate / pos_rate = 78.2 / 21.8 ≈ 3.6`
   Too-high pos_weight forces the model to over-predict positives, crushing precision.

2. **Missing melody features**: `mix_beat_features` (melody stem MERT) is stored in
   every `.pt` file but is **not used** as input to the classifier. The melody is the
   primary signal for *which* drum hits a human mapper chooses to include — different
   genres/instruments create different mapping styles. Without melody context, the
   model can only guess the statistical average onset rate, not song-specific choices.

#### Fix Plan for Run 2

**Code changes needed before retraining:**

1. **`training/beat_module.py`**: Change default `pos_weight=6.0` → `pos_weight=3.6`

2. **`models/beat_classifier.py`**: Modify `__init__` to accept `mix_dim=768` as a
   second input. Concatenate drum + mix features before the input projection:
   `input_proj = Linear(768 + 768, d_model)` (or project separately and add).
   Forward signature: `forward(drum_features, mix_features) → [B, W, 2]`

3. **`data/beat_dataset.py`**: Add `mix_features` to `__getitem__` return dict —
   load `data["mix_beat_features"][start:end].float()` alongside drum features.

4. **`scripts/train_beats.py`**: Pass `pos_weight=3.6` and update BeatLitModule init.

**Run 2 command (after code changes):**
```bash
python scripts/train_beats.py \
  --max-epochs 30 \
  --batch-size 64 \
  --pos-weight 3.6 \
  --patience 8
```
*(add `--patience` arg to train_beats.py — currently hardcoded to 5)*

**Expected improvement:** Correcting pos_weight alone should lift precision from 0.33
to ~0.50. Adding melody features should further lift by teaching the model which drum
hits a mapper would "choose" given the song's melodic content. Target: F1 ≥ 0.65 as
a realistic intermediate; F1 ≥ 0.80 remains the DoD.

#### Existing Code (unchanged)
- [x] `models/beat_classifier.py` — 2-layer local self-attention, drum MERT only
- [x] `data/beat_dataset.py` — sliding-window dataset, 128-slot windows, hop 64
- [x] `training/beat_module.py` — weighted BCE, F1/P/R via torchmetrics
- [x] `scripts/train_beats.py` — standalone training script
- [x] **Run 2 code changes** — mix-stem fusion, phase embedding, pos_weight=3.6
- [x] **Run 2 trained** — val_f1_avg=0.442 (peaked at epoch 0)
- [ ] **Run 3 code changes** — diff embedding + tolerance F1 metric
- [ ] **Run 3 trained** — overnight 2026-05-20
- [ ] **Threshold sweep** after Run 3 converges
- [ ] Follow-up: backfill `mapper` field into V7 `.pt` files to enable cohort conditioning
- [ ] Follow-up: ablation of density-regression target if Run 3 still saturates
- [ ] Follow-up: inference call site in `generation/generate.py::generate_v7_level`
      currently calls `beat_module(drum_t)` only — needs `mix_t` and `diff_t` passed
      so inference matches Run 3 training conditioning. Deferred from the Run 3
      commit to keep scope tight; file had unrelated uncommitted edits.

**DoD:** `val_f1_avg_tol` ≥ 0.80 (with ±1-slot tolerance). Exact-slot F1 is a secondary diagnostic.

### V7-4/5 — Stage 2: Layout Generator 🔧 REDESIGN IN PROGRESS (2026-05-21)

#### Reevaluation (2026-05-21)

With Run 3 Stage 1 producing trustworthy onset schedules (and diagnostics confirming
the model places notes in audio-coherent positions), Stage 2 is now the bottleneck.
Re-audited the design:

**The current per-note design is structurally limited.** Each onset generates its
own 5-token sequence in isolation. The only cross-note information is a 12-dim
hand-engineered saber-state vector (`saber_state.py`) summarising the LAST event
per hand. Concretely this means the model:

- Cannot see the actual prior-note tokens (only their hand-designed summary)
- Cannot plan ahead (set up a position for a future note)
- Cannot learn multi-note motifs (zig-zag setups, 4-note runs, build-and-release)
- Has parity (red/blue alternation) baked in as a scalar field, not learned

The 12-dim saber state IS the "borderline force red/blue alternation" bandaid we
flagged. The V6 inference path adds explicit constrained-decoding parity tracking
on top (`generate.py:938`); the V7 path doesn't, but still relies on the conditioning.

#### V7-5b redesign: phrase-level autoregression

Replace per-note generation with per-phrase generation. Each phrase (16 beats =
~64 slots) becomes one training sample. The decoder emits the spatial tokens for
ALL notes in the phrase as a single sequence, autoregressive within the phrase.

```
Encoder: phrase MERT  [T_phrase, 768] + slot position embedding → encoder_out
Decoder: layout tokens [BOS, n0_KIND, n0_X, n0_Y, n0_DIR, n0_FIELD_D,
                              n1_KIND, n1_X, n1_Y, n1_DIR, n1_FIELD_D, ...,
                              EOS]
         + per-token slot embedding (which onset)
         + per-token hand embedding (left/right)
         + per-token phase embedding (KIND/X/Y/DIR/FIELD_D position in note)
         + global difficulty + genre conditioning
         → causal self-attention + cross-attention to encoder_out
         → output_proj over vocab
```

Saber state is dropped entirely. Position, direction, and parity become emergent
properties the decoder learns from its own prior-token attention within the phrase.

#### Files affected (V7-5b)

- `data/layout_dataset.py`           — REPLACE: per-phrase samples
- `models/layout_model.py`           — REPLACE: encoder-decoder transformer
- `training/layout_module.py`        — REPLACE: CE+mask over phrase token sequence
- `scripts/train_layout.py`          — UPDATE: new sample shape, longer max_len
- `generation/generate.py::generate_v7_level` — UPDATE inference path (deferred to
  follow-up commit; training is the gating step for tonight)
- `tests/test_layout_phrase.py`      — NEW: dataset + model unit tests

#### Trade-offs taken

- **Cross-phrase continuity is dropped** (user-confirmed). The first note of each
  new phrase sees no token history from the previous phrase. Bet: 16-beat phrase
  boundaries are far enough apart that local discontinuity is acceptable.
  Mitigation if it shows in eval: condition first decoder step on last K tokens
  of the previous phrase.
- **Sample count drops from ~50× per song to ~6× per song** (phrases instead of
  onsets). Each sample is much richer (~100-160 tokens vs 5-7), so total token
  volume is similar.
- **Inference is one decode per phrase instead of per-note state-passing.** Simpler.
  PhraseIndex retrieval still bypasses the decoder for high-similarity phrases.

#### Status

- [x] Re-audit + plan (2026-05-21)
- [x] Fix v3 decorative bomb leak (`fix(beatmap): filter decorative (fake)` — commit d7017d0)
- [x] Implement `LayoutPhraseDataset` (per-phrase samples)
- [x] Implement `LayoutPhraseModel` (encoder-decoder w/ token-history attention)
- [x] Implement `LayoutPhraseLitModule` (CE loss + per-role token-acc metrics)
- [x] Update `train_layout.py`
- [x] Smoke test (389 tests pass; GPU bf16 fwd+bwd ok at 15.4M params, 1.8 GB peak)
- [x] **Run 1 complete** (2026-05-21): 18 epochs, best val_token_acc=0.859 at epoch 11
      (d_model=384, batch=32, 200K train / 22K val phrases). DoD 0.85 MET.
      Per-role breakdown: kind=98% field_d=100% y=83% dir=82% **x=67%** (weakest)
      Logs: `logs/layout_phrase/version_0/`
- [x] **Run 2 LAUNCHED** (2026-05-21 23:28): overnight, PID 5208
      d_model=512, n_heads=8, n_enc_layers=4, n_dec_layers=6, dim_ff=2048 (38.7M params)
      batch=64, lr=2e-4, max_epochs=60, patience=12
      Goal: push x-column accuracy above 67%, overall acc above 0.86
      Logs: `logs/train_layout_v1.log` → `logs/layout_phrase/version_1/`
- [ ] Follow-up: update `generate_v7_level` to use new model architecture
      (currently imports `LayoutLitModule` — will fail at inference until rewritten)

**DoD pending:** val_token_acc ≥ 0.85. Run after Stage 1 converges:
```bash
python scripts/train_layout.py --max-epochs 30
```

### V7-6 — PhraseIndex ✅ DONE (2026-05-15)
- [x] `generation/phrase_index.py::PhraseIndex` — cosine similarity lookup over 4-bar fingerprints
- [x] `NotePattern` dataclass — stores (relative_slot, hand) → spatial_token_list
- [x] Hard retrieval: `query()` returns stored pattern if sim > threshold (0.85), else None
- [x] `record()` fills the nearest pre-indexed slot (or appends if not pre-indexed)
- [x] `build()` pre-computes fingerprints from mix MERT; `clear()` resets between songs
- [x] Smoke-tested: query returns None before record, returns pattern after record ✓

**DoD met** (manual phrase-match test deferred until trained models available).

### V7-7 — End-to-End Inference ✅ DONE (2026-05-22)
- [x] `generation/generate.py::generate_v7_level()` — updated for LayoutPhraseModel
  - Stage 1: windowed (128-slot) BeatClassifier inference with mix+difficulty conditioning
  - Stage 2: per-phrase generation via `model.generate_phrase()`
  - Added `_decode_phrase_tokens()` helper to decode phrase token list into _SwingEvent objects
  - Added `max_layout_len` guard in `generate_phrase()` to prevent pos_emb overflow
- [x] **End-to-end test run** (2026-05-22): SO TIRED ROCK - NUEKI.mp3, Expert
  - Stage 1: 888L + 891R onsets across 1444 slots
  - Stage 2: 1508 notes generated (~17s)
  - Post-process: **8.6 → 6.0 NPS** (target 4-10 ✓)
  - V6 best: 1.08 NPS — **V7 is 5.5× denser**
  - Output: `outputs/v7_first_test.zip`

**DoD MET:** NPS 6.0 ≥ 3.0

**Follow-up for V7-8 threshold tuning:**
- Stage 1 threshold=0.4 gives 61% slot density (888+891 notes). Consider 0.5 for fewer false positives.
- Postprocessor trimmed 8.6→6.0 NPS; threshold=0.5 would reduce trimming waste.
- 0 arcs/chains/bombs generated — Stage 1 only predicts note presence; arc/chain types
  would need Stage 1 to predict multi-class note type (future enhancement).
- Color separation moved 35% of notes — X-position accuracy (67%) is the remaining gap.

### V7-8 — Evaluation + Tuning ✅ BUGS FIXED, ARCHITECTURE ITERATION IN PROGRESS

#### Status (2026-05-25)
- [x] Generate on test song — 6.0 NPS at Expert ✓ (V7-7 done)
- [x] ArcViewer review (2026-05-22) — three bugs found
- [x] **All three bugs fixed** (2026-05-23) — see below
- [x] EDA confirmed fix: Y=top-row 89.7%→28%, D=dot 99.5%→0%, X spread collapsed→even
- [x] Section-aware thresholds replace flat energy scaling (2026-05-25)
- [x] `fix_parity` + `convert_dot_notes` re-enabled in postprocessor
- [x] `top_p` default raised 0.90→0.95 (unblocks D=2/3 horizontal swipes)
- [ ] **Run ArcViewer on `outputs/v7_section_aware.zip`** — section-aware map with Run 4 checkpoint
- [ ] Wire `_compute_adaptive_threshold()` into V7 for per-section NPS targeting (see backlog)
- [ ] Generate ExpertPlus variant to check density scaling
- [ ] Tune PhraseIndex similarity threshold (currently unused now that song-memory replaces it)

#### Bug 1 (CRITICAL — FIXED 2026-05-23): Off-by-one role alignment in `generate_phrase._step`
**Symptom:** ~100% of notes appear in the top row (Y=2), ~100% use dot/any-direction.  
**Root cause:** In `layout_model.py::generate_phrase._step`, the new role/slot/hand metadata
is appended to the sequence buffers **before** the forward pass runs — placing `role=KIND` at
the LAYOUT_PAD placeholder position rather than at the sampled token's position. The model
was trained so that position i with `role=R_i` predicts `T_{i+1}` (the next token). So at the
placeholder with `role=KIND`, the model outputs X-range tokens. At the placeholder with
`role=X`, it outputs Y-range tokens. And so on — a systematic one-step circular shift:
- `role=KIND` → 91% X-range tokens (IDs 44–47)
- `role=X`    → 90.5% Y-range tokens (IDs 48–50)
- `role=Y`    → 91.4% DIR-range tokens (IDs 51–59)
- `role=DIR`  → 90.2% ANGLE-range tokens (IDs 60–66)
- `role=FIELD_D` → 87.5% KIND-range tokens (IDs 38–43)

The hard clamp in `_decode_phrase_tokens` then converts out-of-range tokens to boundary values:
- DIR-range (51–59) decoded as Y: `max(0, min(tok - 48, 2))` → **always 2 (top row)**
- ANGLE-range (60–66) decoded as DIR: `max(0, min(tok - 51, 8))` → **always 8 (dot)**

**Why training didn't catch it:** `val_token_acc` is teacher-forced — the model sees ground-truth
previous tokens at every step and correctly predicts the next one. The role misalignment
only surfaces during autoregressive rollout, which the metric never tests.

**Fix:** Restructure `_step` in `layout_model.py::generate_phrase` to read logits from the last
real token's output position *before* appending the new metadata. Append role/slot/hand *after*
sampling, together with the newly sampled token. No retraining needed.

```python
def _step(role: int, slot: int, hand: int) -> int:
    S = len(toks)   # use real sequence length, no placeholder
    x = (tok_emb([toks]) + slot_emb([slots]) + hand_emb([hands])
         + role_emb([roles]) + dec_pos_emb(arange(S)))
    ...
    logits = out_proj(y)[:, -1, :]   # last real token predicts next
    tok = nucleus_sample(logits)
    toks.append(tok)    # append token THEN metadata
    slots.append(slot)
    hands.append(hand)
    roles.append(role)
    return tok
```

---

#### Bug 2: Stage 1 threshold too low → fixed-interval appearance
**Symptom:** Beat pattern looks like a metronome — notes on nearly every 16th-note slot,
no rhythmic variation, can't "speed up" for fast passages because the grid is already saturated.  
**Root cause:** `beat_threshold=0.4` produces 888+891 onsets across 1444 slots (62% density).
At 123 BPM with 4 subdiv, 62% density = a note every ~1.6 16th notes on average = 8+ NPS
before postprocessing. A real Expert rock map runs 3–5 NPS with large rhythmic gaps.
Threshold=0.4 is far below the operating point that produces musical density variation.  
**Fix:** Raise threshold to 0.5–0.6 and regenerate. Profile the onset probability histogram
to find the natural gap between "clear beat" and "marginal prediction" and use that as the
threshold. Additionally, within a window, high-probability slots should suppress adjacent
low-probability ones (non-maximum suppression within ±1 slot).

---

#### Bug 3: No energy or section adaptation — monotone throughout
**Symptom:** The generated map is identical in density and intensity from intro to breakdown
to chorus to outro. There is no distinction between quiet and loud sections, guitar vs. bass
vs. drum passages, or beat drops.  
**Root cause:** `generate_v7_level` applies a fixed threshold for the entire song and does not
use `structure_features` (RMS energy, spectral flux, etc.) which are already computed and
stored in every `.pt` file. The V6 pipeline had `_compute_adaptive_threshold()` (still present
in `generate.py:272`) that raised thresholds in quiet sections and lowered them in loud sections,
but it is never called from `generate_v7_level`. Neither Stage 1 nor Stage 2 has any
section/energy conditioning during inference.  
**Fix:** For Stage 1, extract per-phrase energy from `mix_beat` (mean L2 norm per 64-slot
window is a cheap proxy) and scale the beat threshold inversely — low-energy phrases get a
higher threshold (fewer notes), high-energy phrases get a lower threshold (more notes).
Alternatively, route the existing `_compute_adaptive_threshold()` function from the V6 path
into the V7 windowed inference loop. No retraining needed; this is a pure inference change.

---

- [x] **Fix Bug 1** — restructure `_step` in `layout_model.py` (2026-05-23)
- [x] **Fix Bug 2** — threshold raised to 0.55, ±1-slot NMS added (2026-05-23)
- [x] **Fix Bug 3** — section-aware per-slot thresholds replace flat energy scaling (2026-05-25)
- [x] Nucleus sampling fixed: uniform→probability-weighted (2026-05-23)
- [x] Constrained sampling added: logits masked to legal role vocab range (2026-05-23)
- [x] `fix_parity` + `convert_dot_notes` re-enabled in postprocessor (2026-05-25)
- [ ] Wire `_compute_adaptive_threshold()` for target-NPS-per-section (see backlog)
- [ ] Compare V6 vs V7 NPS on same test songs (V6 best: 1.08 NPS, V7: 6.0 NPS)
- [ ] Generate ExpertPlus variant to check density scaling
- [ ] ArcViewer pass on `outputs/v7_section_aware.zip` (section-aware map, Run 4 ckpt)

---

## V7 Architecture Iteration Log (2026-05-23 → 2026-05-25)

### Training Run History

| Run | Version | Config | Best val_token_acc | X-acc | Notes |
|-----|---------|--------|--------------------|-------|-------|
| Run 1 | layout/version_0 | d=384, 3enc+4dec, 15.4M | 0.859 | 67% | Baseline, DoD met |
| Run 2 | layout/version_1 | d=512, 4enc+6dec, 38.7M | 0.861 | 68% | Bigger model, no gain |
| Run 3 | layout/version_2 | same + x_role_weight=2.0 | 0.861 | 68% | X-weight didn't help → ceiling is subjectivity |
| Run 4 | layout/version_3 | same + ctx_len=16 | **0.870** | **70%** | Cross-phrase prefix broke ceiling — +0.009 overall |
| Run 5 | layout/version_4 | ctx_16 + scheduled_sampling | 0.869 | 70% | No benefit from scheduled sampling |
| Run 6 | layout/version_5 | ctx_16 + song_emb/section_emb scalar | 0.870 | 70% | Scalar conditioning confirmed useless |
| **Run 7** | layout/version_6 | ctx_16 + **song-memory cross-attn** | 🔄 IN PROGRESS | — | Dynamic: decoder attends to all phrase fingerprints |

Beat Classifier:
| Run | Version | Config | Best val_f1_avg_tol |
|-----|---------|--------|---------------------|
| Run 3 | bc/version_3 | d=256, 2-layer | 0.588 |
| Run 5 | bc/version_4 | **d=512, 4-layer** | **0.603** |

### Architectural Lessons

1. **Fixed window encoder ≠ dynamic context.** The phrase encoder processes a fixed 64-slot window. Every run (1–6) with the same local encoder hit the same 0.861 ceiling. Scalar song/section embeddings added zero lift — the model needs *attentional* access to song history, not a summary vector.

2. **Cross-phrase token prefix is the cheapest win.** ctx_len=16 (last 16 tokens from prior phrase) pushed the ceiling to 0.870 and improved every spatial role. The decoder uses its causal self-attention to leverage this — no architecture change needed.

3. **X-column accuracy is ~70%, structural.** Role weighting (2×), bigger model, and everything else left X at 68–70%. This is mapper subjectivity: same melody legitimately maps to multiple columns. The ceiling is not capacity or optimization.

4. **Song-memory cross-attention (Run 7) is the right fix** for the original V6 failure mode ("same chorus at bar 8 and bar 40 → inconsistent patterns"). Phrase fingerprints are precomputed in every .pt file. The decoder now attends to [local 64-slot MERT | all N_phrases fingerprints] jointly — soft retrieval instead of the hard-threshold PhraseIndex.

5. **Stage 1 probability distribution is flat.** The beat classifier outputs near-uniform probabilities (18–31% density from threshold 0.30–0.80). There is no bimodal gap. Section-aware thresholds (drop=0.38, outro=0.72) create 5–8 NPS variation but not the 0–9 NPS of real maps. Target: wire `_compute_adaptive_threshold()` to find per-section threshold that hits desired NPS.

6. **Section detector needs calibration for EDM.** `detect_sections()` (agglomerative clustering on chroma+MFCC) labels most EDM tracks as "outro" after ~40s because EDM has consistent RMS post-intro. Genre-aware weighting or a simpler energy-percentile threshold would serve better.

---

## Backlog (Prioritised)

### High — expected to move needle

- **Wire `_compute_adaptive_threshold()` into V7** `generate_v7_level`.
  V6 had this: binary-search the threshold that produces a target NPS per section.
  Target: drop=7 NPS, verse=4 NPS, intro/outro=2 NPS.
  File: `generation/generate.py` — function already exists at line ~272.
  No training needed; pure inference change.

- **Evaluate Run 7 checkpoint** (song-memory cross-attention) once training completes.
  Expected metric: val_token_acc > 0.870. More importantly: does second chorus match
  first chorus pattern in ArcViewer? That's the true test.

- **Improve section detector for EDM**.
  Options: (a) use raw RMS percentiles instead of clustering — any window in the top-25%
  energy percentile is a "drop", bottom-25% is "intro/outro"; (b) pass genre to
  `detect_sections` and tune aggressiveness by genre.

### Medium — good experiments, lower certainty

- **Scheduled sampling (revisit after Run 7).**
  Run 5 showed no benefit vs Run 4 teacher-forcing. Hypothesis: exposure bias isn't the
  bottleneck when constrained sampling already prevents grammar errors. May be worth
  revisiting after song-memory is proven.

- **Larger Beat Classifier (Run 6).**
  version_4 at d=512/4-layer gives 0.603. Try d=768 or adding structure_features as
  auxiliary input (RMS, onset strength pooled to beat grid). Stage 1 is still the
  limiting factor for density shaping.

- **ExpertPlus + Hard difficulty training.**
  Currently Expert+ExpertPlus only. Adding Hard (lower density, different patterns)
  might help generalization. Or try a density-conditioned model that scales NPS
  to a requested target rather than learning a fixed Expert density.

### Low — future / research

- **KV-cache for autoregressive decoding.**
  `generate_phrase` recomputes the full forward pass at every step → O(N²) cost.
  With ~150 tokens/phrase × 45 phrases/song, this is ~300K FLOPs/step. A KV-cache
  would halve generation time. Not blocking quality.

- **Arc/chain generation.**
  Stage 1 currently only predicts note presence (binary). Multi-class prediction
  (note vs arc vs chain vs bomb per slot) would unlock the full beatmap vocabulary.
  Would require retraining Stage 1 with multi-class labels from swing_tokens.

- **Per-mapper style conditioning.**
  `mapper` field was never backfilled into V7 .pt files (preprocessing gap). A
  preprocessing pass + mapper embedding would let the model mimic specific mapping
  styles. Low priority until the base quality is strong.

- **Replace PhraseIndex entirely.**
  With song-memory cross-attention (Run 7), the PhraseIndex hard-retrieval is
  superseded — the model learns soft retrieval. Keep PhraseIndex as a fallback but
  disable by default once Run 7 is validated.

---

## Explicitly Deprecated (Do Not Revisit)

| Thing | Why |
|-------|-----|
| Scratch `AudioEncoder` mel transformer | MERT knows more music than we can teach it |
| Δt tokens in Stage 2 | Timing is now explicit from Stage 1 — conflating WHEN and WHAT was the root failure |
| `phrase_energy_alpha` KL loss | MERT makes audio-density alignment unnecessary; retrieval handles consistency |
| `dt_density_alpha` hinge loss | Symptom treatment; root cause was missing explicit timing |
| `bomb_hand_weight` tuning | Bomb attractor was a symptom of bad timing loss; with explicit timing it won't recur |
| Per-window Δt autoregressive inference | Replaced by beat-slot iteration from Stage 1 schedule |

---

## Success Criteria

V7 is working when:

1. **Stage 1 F1 ≥ 0.80** on held-out songs (onset detection, both hands)
2. **NPS ≥ 3.0** on the test song at Expert difficulty (was 1.08 best V6)
3. **NPS ≥ 5.0** after tuning (Expert target range: 4–10)
4. **Cross-song consistency:** second chorus note patterns are visually similar to first chorus (manual ArcViewer review)
5. **No bombs / no parity violations** pre-postprocess (structural grammar handles this)
6. **Iteration speed:** full preprocessing + Stage 1 train + Stage 2 train ≤ 8 hours total on RTX 5090
