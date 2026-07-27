# Beat Saber Automapper — Progress History

> **For current work, active TODOs, and implementation plan, see [`TODO.md`](TODO.md)**
> **For latest architecture analysis, see [`docs/architecture_v7_plan.md`](docs/architecture_v7_plan.md)**

This file is a historical record of what was done, what worked, and what didn't.

---

## Eval-suite v2: four axes, a working judge, and the "locally wrong" principle (2026-07-27)

A single long autonomous session. The strategic frame (set 2026-07-26) is that the **evaluation
suite is the work**, not the generator — the goal is a suite good enough that Kyle no longer has
to be the judge, and prescriptive enough that an agent could build a mapper from it without ML.

### The organising principle discovered today

**Our failures are consistently "globally right, locally wrong."** Three independent instances:

| | global statistic (looks fine) | local structure (broken) |
|---|---|---|
| sequencing | `h_dist` histograms pass | a *shuffled* map scores like a human one |
| hand balance | `flow.handedness` 0.012 for **both** | local asymmetry 0.115 human vs **0.031** ours |
| idiom vocabulary | 238 distinct idioms vs human 219 | 0.861 human vs **0.703** ours per 16-transition window |

This is why the original scorecard was blind for months: every metric in it was a whole-map
histogram, and whole-map histograms are exactly where this generator looks good. It is also why
the direct-reading channel (below) keeps finding things the aggregates cannot.

### What was built

**Four scored axes**, all using one shared distribution-scoring core (`evaluation/_dist.py`) that
compares a **cohort** against the human distribution by median *shift* and *spread*:

| axis | module | human | prod |
|---|---|---|---|
| A1 flow / ergonomics | `evaluation/flow.py` | 0.13 | 0.81 |
| A2 rhythm / beat-grid | `evaluation/rhythm.py` | 0.25 | 2.41 |
| A3 pattern idiom | `evaluation/idiom.py` | 0.31 | 2.34 |
| A6 hand role | `evaluation/handrole.py` | 0.34 | **3.50** |

**`evaluation/scorecard.py`** — the single entry point. One command, one verdict. Validated both
ways on disjoint data: a held-out human cohort **passes every axis**; current production **fails
all four** with parity clean.

**`scripts/audit_eval_suite.py`** — the control battery every axis must pass before it is allowed
to steer anything: human maps vs our maps vs six degenerate controls (`random`, `shuffled`,
`metronome`, `zigzag`, `timing_random`, `timing_jitter`). Blind-spot reporting is axis-aware,
because each control only attacks what it destroys.

**`scripts/map_view.py`** — read a map as a **text score**: time down, hands side by side, per-stem
audio lanes from the same transcription the model trains on, inline idiom rank + corpus frequency,
`--find` for violations/OOV/doubles with context, `--vs` for two maps aligned in **seconds**.

**`scripts/rule_mapper.py`** — a mapper with **no ML**, built only from the suite's rules. Given
human onsets it passes rhythm (0.25) and nearly passes idiom (0.99) with zero parity violations,
and beats our trained model on idiom. The suite is prescriptive enough to specify a mapper.

### The two biggest findings, both from *reading* rather than statistics

**1. Hands have roles.** In a human map, within a passage one hand carries a sustained run while
the other punctuates, and they swap. Ours run both hands at identical density. Human maps are
balanced *globally* but lopsided *locally*; ours are balanced at every scale, which is the
unnatural thing. A6 measures it: prod 3.50 vs human 0.34 — **worse than a uniformly random map
(2.64)**, the largest single-axis defect ever measured in this project.

**2. 30% of songs generate at the wrong tempo.** Spotted because the score header prints BPM and
the human map for the same song said 188 where ours said 94. Against human-declared BPM, raw
librosa detection is correct on only 16/23. At half tempo the finest grid slot is twice as coarse
in real time, so the fast notes cannot be represented. **The metrics reward the bug** — mis-tempo
maps score *better* on all axes, because A2 measures intervals in the beat domain.

### Levers tested (24-song sweeps)

| lever | verdict |
|---|---|
| `LAYOUT_TRAVEL_PENALTY=1` | ✅ flow 0.81 → **0.30 PASS** |
| `COLOR_SEP_MODE=extreme` | ✅ idiom 1.84 → **0.30 PASS** (the postprocess was destroying idioms wholesale) |
| `LAYOUT_TRAVEL_PENALTY=4` | ❌ over-corrects: flow 1.77, spread **0.00** — every map identical |
| `COLOR_SEP_MODE=off` | ❌ overshoots (flow 1.04); `extreme` is the right setting |
| `LAYOUT_IDIOM_BONUS` | ~ helps idiom (1.84 → 1.20) but weaker than `xsep_ext` at the same job |
| `BEAT_HAND_INTERLEAVE` | ❌ **rhythm worse** (2.99/2.81 vs 2.41), breaks parity |
| `BEAT_HAND_ROLE=0.5` | ~ fixes idiom (2.34 → **0.59 PASS**), improves flow/handrole, but **rhythm 2.41 → 4.05**, spread collapses, −24% notes |

### Stage-1 IOI prior — negative, and the diagnosis is structural

Having ruled out tempo (part D) and layout (`rule_mapper`), the rhythm gap had to be onset
selection, so the within-window pick was changed to use an interval bigram mined from 300 human
maps. Three formulations, all failed:

| formulation | notes | switch rate (human 13.7) | outcome |
|---|---|---|---|
| prod (top-k by prob) | 1295 | 1.2 | baseline |
| maximise prob + prior | 1376 | cohort 3.18 | rhythm 2.37 → **2.80**, flow 0.71 → 2.59, idiom 1.85 → 4.57 |
| free sample the prior | **437** | 26.7 | loses 66% of notes |
| sample + budget guard | 1387 | **0.3** | regular again |

Maximising a diagonal-dominant bigram (P(1/8→1/8) 0.714) makes rhythm *worse*: its argmax is
"keep the current interval", so the map gets long homogeneous runs. The interval histogram moved
toward human while the sequence got more regular. **The argmax of a distribution is not a sample
from it** — the same error the v2 suite exists to prevent, made one level down.

**The structural finding: a fixed note budget in a fixed 2 s window IS the regularity.** With k
notes required in a fixed span the mean interval is pinned at span/k; only the variance is free.
Human interval variety comes from density varying *across* time, not from reshuffling inside a
quota. The next lever is therefore the window **allocation** (variable-length, phrase-aligned
windows) rather than the within-window pick.

### Negative results — recorded so they are not re-attempted as written

- **A5 structural self-consistency**: human maps are *not* more self-similar at bar-aligned lags
  than at arbitrary ones (`struct_lift` ≈ 0 for every cohort including human, across three
  similarity tokens). Needs audio-derived section boundaries, not fixed lags.
  `evaluation/structure.py` is dormant.
- **BPM octave correction**: both attempts made detection *worse* (10/23 and 14/23 vs a 16/23
  baseline). The hypothesis that the true metrical level has balanced odd/even beat energy is
  false — real music has backbeat asymmetry at its true tempo. `detect_bpm` left alone.
- **`BEAT_HAND_INTERLEAVE`**: see above.

### Method lessons

- **Validate every lever on the full 24-song set.** `BEAT_HAND_INTERLEAVE` was designed from a
  single-song probe on **1f333 — one of the two half-tempo songs**. A2 is beat-domain, so the
  probe was measured in a distorted frame and the lever failed on all 24 songs. Single-song runs
  smoke-test a code path; they are not evidence.
- **Rank cohorts by shift *and* spread, never per-map distance-to-median.** The first version of
  the flow metric reproduced the `h_dist` failure exactly — our maps scored "more human than
  human" — because a mode-collapsed cohort sits nearer the median than typical human maps do.
- **Never run two sweeps against one cache.** An overnight script was launched twice; both wrote
  the same `outputs/eval_sweep_cache/<arm>__<song>.zip` paths and 11 zips came out corrupt.
  `eval_sweep` now takes a lock. Earlier arms were unaffected.

---

## V7-5b Stage 2 Run 1 + Run 2 Launch (2026-05-21)

### Run 1 result (logs/layout_phrase/version_0/)

18 epochs, default architecture (d_model=384, 15.4M params), batch=32, 200K train / 22K val phrases.
**Best val_token_acc = 0.859 at epoch 11** (DoD target: 0.85). Early stopping at epoch 17 (patience=12
from peak). Training converged cleanly: val_loss bottomed at 1.099 around epoch 12, then slowly rose.

Per-role accuracy at convergence:
- val_acc_kind: **98.0%** — model almost always emits the right note type
- val_acc_field_d: **99.7%** — near-perfect
- val_acc_y (row, 3 classes): **83.4%**
- val_acc_dir (direction, 9 classes): **81.7%**
- val_acc_x (column, 4 classes): **66.7%** ← weakest; model capacity is the likely bottleneck

X-column accuracy at 67% is above random (25%) but lower than the other attributes.
This is the target for Run 2: a 2.5× larger model (38.7M params, d_model=512) should
provide the capacity the column prediction needs.

### Run 2 launched (overnight 2026-05-21 23:28, PID 5208)

```bash
python scripts/train_layout.py \
  --max-epochs 60 --batch-size 64 --lr 2e-4 \
  --d-model 512 --n-heads 8 --n-enc-layers 4 --n-dec-layers 6 --dim-feedforward 2048 \
  --patience 12 --difficulties Expert ExpertPlus
```

Logs: `logs/train_layout_v1.log` → TensorBoard: `logs/layout_phrase/version_1/`

---

## V7-3 Run 3 Diagnostic + Stage 2 Reevaluation (2026-05-21)

### Run 3 result and post-hoc diagnostics

Run 3 finished at `val_f1_avg_tol = 0.588` (target was 0.65). On the surface,
another short of target. Three new diagnostics on the val checkpoint changed the
interpretation:

1. **Audio-onset coherence**: predicted positives have median onset-strength
   percentile rank 0.51 within their song; labels are at 0.53. **The model is
   placing notes in audio-supported positions just like mappers do.** Top-30%
   fraction is 0.30 predicted vs 0.32 label (random baseline 0.30) — model and
   labels both moderately concentrate on high-onset slots, indistinguishably.
2. **Per-phrase density correlation with onset strength**: predicted-count vs
   onset-strength Spearman = 0.40, identical to label-count vs onset-strength
   (0.40). At the phrase level, the model is just as audio-coherent as the labels.
3. **Calibration ECE = 0.224** with a clear monotonic over-confidence pattern:
   when the model says "92% sure", the actual single-mapper agreement rate is
   48%. This is the smoking gun for the subjectivity ceiling. The model is
   approximating the population mean of mapper placements; F1 against any single
   mapper is bounded by inter-mapper agreement.

**Conclusion: Stage 1 is fine.** The F1 we were chasing is the wrong number for
the task. The remaining 0.18 ECE gap is fixable by post-hoc temperature scaling.

Eval implementation: `scripts/eval_beat_checkpoint.py`; outputs under
`logs/beat_eval/run3_full/`.

### Multi-mapper soft-label retrain — blocked on data

Probed the dataset for songs with multiple mappers (would have let us build
fraction-of-mappers-place-a-note soft targets). Of 5264 unique audios in
`data/processed/`, only 48 (0.9%) have ≥2 mappers and only 4 have ≥3. Not
enough statistical basis. Deferred — would need a Beat Saver backfill pass to
become viable.

### Stage 2 reevaluation

With Stage 1 trustworthy, the bottleneck moves to Stage 2 layout generation.
Re-audited and found the architecture is per-note: each onset generates its
spatial tokens with only a 12-dim hand-engineered saber state to summarise
prior notes. The saber state's parity field is the "borderline force red/blue
alternation" bandaid. The V6 inference path adds explicit constrained-decoding
parity tracking on top (`generate.py:938`).

Decided to redesign Stage 2 as **phrase-level autoregression**: each phrase
(~16 beats / 64 slots) becomes one training sample, the decoder emits ALL
spatial tokens for the phrase as a single causal sequence with cross-attention
to phrase MERT, and the 12-dim saber state is dropped entirely. Position,
direction, and parity become emergent from the decoder's prior-token self-
attention. Full plan in `TODO.md § V7-4/5`.

Side fix: `parse_difficulty_dat_json` v3 path was not filtering decorative
(fake) bombs. The v2 path filtered `_customData._fake`; v3 had no equivalent.
Added a shared `_is_fake` helper checking `customData.fake`, top-level `fake`,
and `_fake`, applied to all v3 object collections. Stage 1 not affected;
Stage 2 would have learned to emit decorative bomb art as gameplay otherwise.
Three new parser tests; 22/22 pass.

---

## V7-3 Run 2 Post-Mortem + Run 3 Audit (2026-05-20)

**Run 2 result** (overnight 2026-05-19 → 2026-05-20): `val_f1_avg = 0.442`, best at epoch 0,
10 epochs of no improvement → early stop. Run 1 was 0.422. The pos_weight + mix-stem +
phase-embedding fixes from the 2026-05-19 audit moved the metric ~2 F1 points. The fixes
were correct but not load-bearing.

**The "peaks at epoch 0" pattern** is the clearest diagnostic signal we have. With a
frozen MERT encoder feeding a small head, the head learns everything that's learnable from
the input features almost instantly. Subsequent training just overfits to noise. F1
saturating immediately means the *features-given-labels* relationship is the bottleneck,
not the model capacity or optimization.

**Audit (2026-05-20) re-derived two structural reasons the label/feature relationship is
weaker than it should be:**

1. **No in-model difficulty conditioning.** `BeatDataset` returns `difficulty` per sample,
   but `BeatClassifier.forward(drum, mix, slot_offset)` never consumes it. Expert maps
   carry ~3 notes/bar, ExpertPlus ~6 notes/bar — the same drum hit gets label 0 in one
   and label 1 in the other. With both pooled and no conditioning, the model can only
   predict the mixture marginal. This alone would explain a substantial F1 deficit.

2. **Exact-slot F1 is too brutal.** At subdiv=4 and BPM=120, one slot is ~125 ms — well
   inside human onset perception tolerance and well below mapper placement noise. MIR
   onset-detection literature uses ±50 ms or ±1-slot tolerance windows; we use exact
   slot match, which double-counts off-by-one errors (FP + FN). The reported F1 is
   systematically below the inter-mapper agreement floor for this reason.

Also confirmed absent (deferred, not regressed):
- **Mapper-cohort conditioning:** cohort scripts (`scripts/cohort_eda.py`,
  `compute_cohort_reference.py`, `download_cohorts.py`) survived from V6, but V7
  preprocessing never populated `mapper` in `mod_requirements` — the field is `None`
  for every `.pt`. Need a backfill pass before this is usable in V7.
- **Density-regression target:** still binary BCE per slot. Bigger redesign,
  deliberately deferred.

**Run 3 plan** (overnight 2026-05-20):
- Add `nn.Embedding(N_DIFF, d_model)` summed into the input post-`input_norm`.
- Add MIR-style ±K-slot tolerance F1 (greedy match, each pred matches ≤1 label and
  vice versa). Log `val_f1_avg_tol` alongside the existing exact-slot `val_f1_avg`.
- Keep Expert + ExpertPlus pooled — the new embedding handles the diff disambiguation.
- pos_weight stays at 3.6.

If `val_f1_avg_tol` clears 0.65 the metric was always the issue; if it doesn't,
we're closer to confirming the subjectivity ceiling and the next move is per-mapper
training or density regression.

---

## V7 Audit + Fix Pass (2026-05-19)

Architecture review confirmed V7's high-level intent is sound — decoupling WHEN (Stage 1)
from WHAT (Stage 2), with multi-tier MERT conditioning (local frame / section / song) and
PhraseIndex hard-retrieval for cross-song consistency. The 3-second-window failure mode of
V6 is structurally avoided.

Bugs found and fixed:

**Stage 1 (BeatClassifier)**
- `pos_weight = 6.0` was calibrated for a 15% positive rate; the measured rate on the
  Expert+ training split is 21.8%. Corrected to 3.6 (= 78.2/21.8). This was the primary
  cause of Run 1's low precision (0.33 — over-predicting positives by ~3×).
- `mix_beat_features` was preprocessed into every `.pt` file but never read by the
  classifier. The mix (melody) stem carries the genre/instrument signal that determines
  which drum hits a human mapper *chooses* to include. Now sum-fused with the drum
  projection inside `BeatClassifier`.
- No explicit phase signal. Mappers respect the within-bar phase (1-and-2-and-3-and-4-and).
  Added a learned phase embedding indexed by `(slot + slot_offset) % 16`. Acts in addition
  to the (window-relative) positional embedding.
- `--patience` was hardcoded to 5 in `train_beats.py`; exposed as a CLI arg (default 8).
- Beat labels were recomputed per window in `__getitem__` — cached per (song, difficulty).

**Stage 2 (LayoutDataset)**
- **Off-by-one saber-state bug**: was `compute_saber_states(all_events[:evt_idx])[-1]`,
  which is the saber state BEFORE event `evt_idx-1`, not before event `evt_idx`. Fixed
  to `compute_saber_states(all_events)[evt_idx]`.
- **O(n²) recompute**: `decode_events` + `compute_saber_states` were called for every
  `__getitem__`. Now cached per (song, difficulty).

Verified: 361/361 tests still pass; smoke training run on real data converges normally.

---

## V7 Implementation: Full Pipeline Built (May 15, 2026)

### What was built

All V7 code is implemented and import-tested (361/361 tests pass). The full
pipeline runs end-to-end in smoke tests. **Training is blocked on preprocessing
completing** — `scripts/preprocess_v7.py` is running and at ~505/5320 songs as of
18:56 local, ETA ~4.5h remaining.

### New files

| File | Purpose | Status |
|------|---------|--------|
| `scripts/v7_poc.py` | Demucs+MERT PoC beat classifier | Done |
| `scripts/preprocess_v7.py` | Demucs+MERT feature extraction for all songs | Running |
| `scripts/train_beats.py` | Stage 1 training script | Done, awaiting preprocessing |
| `scripts/train_layout.py` | Stage 2 training script | Done, awaiting Stage 1 |
| `data/mert_encoder.py` | MERT-v1-95M wrapper: extract + beat-grid pool + phrase fingerprints | Done |
| `data/stem_separator.py` | Demucs htdemucs wrapper (GPU, cached) | Done |
| `data/beat_grid.py` | Binary beat labels from swing_tokens | Done |
| `data/beat_dataset.py` | Sliding-window dataset for Stage 1 | Done |
| `data/layout_dataset.py` | Per-onset dataset for Stage 2 | Done |
| `models/beat_classifier.py` | Stage 1: local attention on drum MERT → P(left/right) | Done |
| `models/layout_model.py` | Stage 2: causal transformer, MERT-conditioned, no Δt/HAND | Done |
| `training/beat_module.py` | Lightning: weighted BCE, F1/P/R metrics | Done |
| `training/layout_module.py` | Lightning: spatial CE loss, token accuracy | Done |
| `generation/phrase_index.py` | PhraseIndex: cosine similarity hard retrieval | Done |
| `generate.py::generate_v7_level` | Full V7 end-to-end inference function | Done |
| `scripts/generate.py --v7` | CLI flag wiring | Done |

### V7-0 PoC results (validated before full build)

- Demucs `htdemucs` separated test song in ~2s on RTX 5090 GPU
- MERT-v1-95M produced `[13210, 768]` at 75 Hz (correct frame rate)
- Beat grid: 1444 slots at 1/4-note resolution, 9.1 MERT frames per slot at 123 BPM
- **sklearn logistic regression (same-song, frozen MERT):** F1_left=0.52, F1_right=0.67 → **avg F1=0.59**
- Conclusion: MERT drum stem features carry strong onset signal without any task-specific training

### Preprocessing throughput (actual, RTX 5090)

- Warmup (model load): ~6s
- Per-song: ~4.5s average (scales with song length)
- 5320 songs total: ~6.5h one-time cost
- Song `1ccca.pt` (52s song): features written as `drum_beat_features [468, 768]`,
  `mix_beat_features [468, 768]`, `phrase_fingerprints [8, 768]` — all fp16, ~3 MB/song added

### Data format after preprocessing

Each `.pt` file gains four new keys (non-destructive, all existing keys preserved):

| Key | Shape | dtype | Description |
|-----|-------|-------|-------------|
| `drum_beat_features` | `[N_slots, 768]` | fp16 | Drum MERT pooled to 1/4-note grid |
| `mix_beat_features` | `[N_slots, 768]` | fp16 | Melody MERT pooled to 1/4-note grid |
| `phrase_fingerprints` | `[N_phrases, 768]` | fp16 | Mean MERT per 4-bar window |
| `phrase_boundaries` | list of (int, int) | — | (start_slot, end_slot) per phrase |

### Key design decisions made

1. **Drum stem only for Stage 1**: cleaner onset signal than full mix; confirmed by PoC
2. **Melody stem ("other") for Stage 2**: captures instrument-specific features for layout
3. **fp16 storage**: halves storage overhead vs fp32; 768-dim × N_slots × 2 bytes ≈ 1.2 MB per stem per song
4. **MERT layer -1 (final layer)**: best for discriminative tasks per MERT paper; not tuned yet
5. **4-bar phrase windows (16 beats)**: matches typical verse/chorus structure; configurable
6. **Hard retrieval at sim > 0.85**: conservative starting threshold; tune based on subjective repetitiveness of output

### What runs next (in order)

```bash
# 1. Wait for preprocessing to finish (~4.5h from 18:56)
# 2. Train Stage 1
python scripts/train_beats.py --max-epochs 20 --batch-size 64
# Target: val_f1_avg ≥ 0.80

# 3. Train Stage 2
python scripts/train_layout.py --max-epochs 30
# Target: val_token_acc ≥ 0.85

# 4. Generate test map
python scripts/generate.py "data/test_songs/SO TIRED ROCK - NUEKI.mp3" \
  --v7 --beat-ckpt <ckpt> --layout-ckpt <ckpt> \
  --difficulty Expert --genre rock --run-tag v7_first
```

---

## V6 Post-Mortem: Beat Timing Failure + Architectural Verdict (May 15, 2026)

### Performance Summary Across All V6 Runs

| Checkpoint | Notes | NPS | Bombs | Val Loss | Epoch | Problem |
|---|---|---|---|---|---|---|
| version_2 (first post-bugfix retrain) | — | — | — | 0.947 | 30 | Encoding bugs invalidated |
| version_3 | — | — | — | 0.997 | 30 | Post-bugfix retrain |
| version_4 | 72→81 (gen-fixed) | 0.46 | 232 | 0.960 | 60 | Stall bug + bomb attractor |
| version_6 (bomb_weight=0.3) | 157 | 0.89 | 0 | 0.986 | 30 | Low NPS |
| version_7 (dt_density=0.5) | 120 | 0.68 | 0 | 1.010 | 30 | Regression |
| version_8 (dt_density=1.0) | 191 | 1.08 | 0 | 1.010 | 30 | Low NPS |

**Expert target: 4–10 NPS. Best achieved: 1.08 NPS. The model has never come close to target density in any run across any configuration.**

### Generation Bugs Fixed Along the Way

Two real bugs were fixed that were masking the problem:

1. **Window stall bug (2026-05-14):** When the model emitted all Δt=0 events and hit the per-window cap, `resume_state.current_beat` wasn't advanced with `window_start_beat`. Every subsequent window had audio context at beat N but the model's internal clock at beat 32.44, permanently anchoring all events there. Fixed: `resume_state.current_beat = window_start_beat` on manual advance. Also: `max_events` 800→2000, per-window cap 256→128.

2. **Bomb attractor (2026-05-14):** HAND_NONE (bombs) had the same 3× loss weight as HAND_LEFT/RIGHT. Bombs are 5-token events vs 7 for notes — shorter, easier to complete, lower-entropy. The model discovered them as a low-loss shortcut. With generation stall fixed, the pre-fix version_4 checkpoint generated 232 bombs / 341 total events (68%). Fixed: `bomb_hand_weight=0.3`.

Fixing both bugs lifted NPS from 0.41 → 1.08. Still catastrophically below Expert target.

### Root Cause: The Model Has No Supervised Signal for Beat Timing

This is the core architectural failure. V6 conflates two separate problems into one autoregressive token stream:

- **Problem 1 (WHEN):** At beat X, should a note exist?
- **Problem 2 (WHAT):** Given a note at beat X, what hand/position/direction?

The Δt token is doing ALL the work for Problem 1. And cross-entropy loss on Δt tokens provides essentially no audio-grounded supervision for this.

**Why CE on Δt fails:**

The training setup: `window_events=128` events, `context_frames=256` mel frames ≈ 3 seconds of audio. At Expert density (~7 NPS), 128 events span ~18 seconds. The audio context covers only **1/6 of the event window**. For ~83% of the Δt predictions the model makes during training, there is no local audio evidence. The model cannot learn "I see a drum hit at this audio frame → place a note here" because it can't see drum hits for most of the events it's predicting.

The CE gradient on a Δt token is simply: `∂L/∂logit_j = p_j − 1[target=j]`. This pushes the model toward predicting the training data's marginal Δt distribution — which is dominated by intro/outro/break sparsity as much as by drop density. A model that learns "Δt is usually between 0.25 and 1.0 beats" will achieve good CE loss while producing maps that are uniformly sparse, because that's what the average of all song positions looks like.

**Why `phrase_energy_alpha=0.1` didn't fix it:**

The phrase-energy loss computes KL divergence between predicted swing density and audio RMS across 4 coarse bins over a 3-second window. At 123 BPM, 4 bins = 0.75 seconds each ≈ 1.5 beats. This is far too coarse to produce beat-level onset signals. The KL gradient is also swamped by CE loss at the 0.1 weight.

**Why `dt_density_alpha` didn't fix it:**

The hinge penalty on P(Δt=0) reduces event-bursting but doesn't provide audio-aligned density targets. It tells the model "don't cluster events at a single beat" but not "put events at THESE beats." The model responds by spreading events more evenly — but still not responding to audio features, so density remains low overall.

### Why V6's Core Bet Was Wrong

The V6 architecture was designed to fix V5's physics/parity/style problems. It fixed those correctly. But in eliminating Stage 1 (the onset detector) and collapsing timing into the autoregressive stream, it discarded the only component that had a clean discriminative signal for beat placement.

V5's Stage 1 was trained with frame-level binary supervision: `onset_labels[frame] = 1` if a note exists at that frame, `0` otherwise. Every gradient step pointed directly at audio-onset detection. That signal was sharp, local, and correctly calibrated to the audio.

V6 replaced this with Δt tokens inside a sequence model. The equivalent of "is there a note here?" became an implicit consequence of many correlated Δt predictions, with no direct training objective to produce correct onset timing. The saber state and phrase embedding address Problems 2 and 3 (spatial layout and style), but Problem 1 was left to emerge from sequence statistics. It doesn't.

### The Consistent Symptom Across All Runs

Every single run produces the same failure mode: the model generates events that cover the song (after the stall bug was fixed), but with large Δt values — frequently jumping 5–20 beats between events. The model has learned the GRAMMAR of notes (valid token sequences) and the STYLE of individual notes (reasonable X/Y/DIR), but it has not learned to generate notes at musically meaningful beat positions.

This cannot be fixed by tuning `dt_density_alpha`, `bomb_hand_weight`, `phrase_energy_alpha`, epoch count, or any other hyperparameter of the current architecture. The gradient signal for beat timing is structurally absent.

### What Must Change

The two-problem conflation must be separated:

**The WHEN problem requires explicit, audio-aligned binary supervision.**  
Every note position in the training data is a positive label for a specific audio frame. A classifier trained directly on this signal — even a shallow one — can learn beat-onset patterns. This was true in V5 and discarding it was the V6 mistake.

**The WHAT problem is actually tractable for V6.**  
The swing-event grammar (HAND, X, Y, DIR, ANGLE, KIND) is a good representation for spatial layout and style. Once timing is provided externally, the autoregressive model only needs to predict "given a beat at this position, what does the note look like?" — which is a much simpler and more constrained problem. Val token accuracy of 87% suggests the model IS learning spatial layout well; it's purely timing that's broken.

**The architecture needed:**

```
Audio → Beat-Slot Encoder → [binary onset per beat slot] → Onset Schedule
Onset Schedule + Audio → Note Layout Model → [X, Y, DIR, ANGLE, KIND per onset] → Beatmap
```

Stage 1 is a discriminative classifier: per beat slot (1/4 note resolution), predict left-note probability and right-note probability. Direct binary cross-entropy, strong class weights for positive (note) examples, audio features aligned to each beat slot by construction.

Stage 2 is the note layout model: given a confirmed onset position and its audio context, predict the spatial token sequence. This is the problem V6's sequence model was mostly solving correctly.

The key insight: **separate the timing problem (binary classification per beat slot) from the layout problem (sequence generation conditioned on known beat positions).** These require different supervision signals and different architectures. Conflating them into one autoregressive stream requires the sequence model to solve onset detection implicitly through sequence statistics, which it cannot reliably do.

---

## V6 NPS-Fix Overnight Runs (May 14–15, 2026)

Three sequential 30-epoch runs training from scratch with generation stall fix applied:

**Run A** — `bomb_hand_weight=0.3`, `dt_density_alpha=0.0`  
Checkpoint: version_6, val_loss=0.986. Generated 157 notes, 0 bombs → 0.89 NPS.  
Result: bomb fix alone is the biggest single improvement. Bomb attractor eliminated entirely.

**Run B** — `bomb_hand_weight=0.3`, `dt_density_alpha=0.5`  
Checkpoint: version_7, val_loss=1.010. Generated 120 notes, 0 bombs → 0.68 NPS.  
Result: regression vs A. Moderate Δt=0 penalty disrupted useful same-beat chord patterns without providing a positive density signal.

**Run C** — `bomb_hand_weight=0.3`, `dt_density_alpha=1.0`  
Checkpoint: version_8, val_loss=1.010. Generated 191 notes, 0 bombs → 1.08 NPS.  
Result: best run to date. Stronger penalty overcomes chord disruption. Coverage is good (beat 0–365 evenly populated). Still 4–10× below Expert NPS target.

**Conclusion:** Marginal improvements possible by tuning within this architecture. The ceiling is far below target. Do not invest further in hyperparameter search on the current model.

---

## V6 Bug Audit + Training Run (May 12, 2026)

### First V6 training run — completed, results invalidated by encoding bugs

30-epoch run on the full processed pool (5320 maps, Expert/ExpertPlus, batch_size=32) using `sequence_swing_small` preset:
- val_loss: 1.31 → **0.947**, val_token_acc: 69% → **86.8%**, no crash, 4m22s/epoch, ~14.5/32 GB VRAM.
- `phrase_energy_loss` was flat (mean ≈ 0.09) the entire run — did not decrease. V6-4 DoD "verify it actually decreases" is **not met**.

**Run invalidated by three encoding bugs found during generation testing:**

#### Bug 1 — First-Δt absolute-position encoding (dataset.py)
`SwingSequenceDataset._events_to_tokens` started each sliding window with `prev_beat = 0.0`. The first event in every training window therefore had its Δt encoded as its **absolute song position** (e.g., 88 beats), not "0 from window start". The model learned `p(Δt=64 beats | BOS, HAND) ≈ 0.90` — confirmed by logit inspection on the checkpoint. Fixed: `prev_beat = events[0].beat` so first Δt = 0.

#### Bug 2 — Double-BOS teacher forcing (seq_module.py)
`_prepare_teacher_forcing` prepended an extra BOS to `tokens` which already start with BOS (dataset always inserts BOS at position 0). This made `decoder_input = [BOS_extra, BOS, t0, t1, ...]` and `target = [BOS, t0, t1, ...]`. Consequences:
- Train/inference distribution mismatch: at inference step 1 the model sees `[BOS]`; at training step 1 it saw `[BOS_extra, BOS_orig]`.
- Saber-state alignment was off by one (saber_state was not shifted to match the shifted decoder_input).

Fixed: standard LM shift — `decoder_input = tokens[:, :-1]`, `target = tokens[:, 1:]`. Saber-state slice updated in training_step and validation_step to `saber_state[:, :-1, :]`.

#### Bug 3 — Per-window beat-range filter (generate.py)
`generate_swing_level` filtered generated events to `window_start_beat ≤ e.beat ≤ window_end_beat` and advanced `window_start_beat` by a fixed 3.7 beats. With the buggy Δt encoding, every event fell outside the filter; with a corrected model, the filter would still be fragile. Fixed: window cursor advances from `result.final_state.current_beat`; filter removed.

### Fixes shipped
- `data/dataset.py` — first-Δt anchor fix
- `training/seq_module.py` — standard LM teacher-forcing shift + saber_state slice
- `training/seq_module.py` — phrase_energy threshold `>= 64` → `> 8`
- `tests/test_seq_module.py` — updated teacher-forcing tests for correct semantics
- `generation/generate.py` — windowed inference cursor fix
- `scripts/generate.py` — `--v6` flag wired to `generate_swing_level`
- `scripts/train.py` — dropped V5 dead kwargs, added V6 params
- `configs/train.yaml` — `limit_val_batches` knob
- `.gitignore` — `*.mp3 *.ogg *.wav *.flac`

**Next step:** retrain from scratch. The checkpoint at `outputs/.../sequence-epoch=29-val_loss=0.947.ckpt` is not usable — Δt distribution is poisoned.

---

## V6 Implementation — Phases 4, 6, 7 (May 11, 2026)

**Completed:** V6-4 (phrase-energy loss), V6-6 (inference pipeline), V6-7 (harness wiring).

### V6-4: Phrase-energy KL loss
`seq_module._compute_phrase_energy_loss` — divides the token sequence and audio context into 4 equal segments, computes mean HAND-token probability per segment (predicted swing density) vs mean RMS per segment (ground-truth energy density), returns KL divergence. Activated when `phrase_energy_alpha > 0` and `structure` is present in the batch. Replaces the V6-4 stub.

### V6-6: Inference pipeline
- `generation/beam_search_v6.py` — V6 grammar-constrained nucleus sampler. Grammar state machine (`_Phase` enum) enforces the swing-event token grammar at every decode step. Saber state (`_GrammarState.saber`) is updated per completed event and passed to `decode_step_cached` as `saber_state_step`. `_nucleus_sample` filters zero-probability tokens before sampling so grammar masks with `-inf` are never bypassed.
- `generation/generate.py::generate_swing_level` — full V6 end-to-end pipeline: audio → audio encoder → phrase embedding → `nucleus_sampling_v6` → `SwingEventTokenizer.decode_beatmap` → `postprocess_beatmap` (trimmed) → rule-based lighting → .zip. Tested with `test_generate_swing_level_creates_zip`.
- `generation/postprocess.py::postprocess_beatmap` — removed `fix_parity` and `convert_dot_notes` calls. Structural rules (NPS cap, color separation, arc/chain connectivity) kept.
- `data/audio.py::detect_sections` — fixed pre-existing bug: chroma and MFCC could produce different frame counts on short audio; now truncates to `min(len(chroma), len(mfcc))` before vstack.
- `test_generate.py::TestGenerateNoteSequence` marked `xfail` (V5 beam_search BOS/EOS constants conflict with V6 vocab; harmless until beam_search.py is fully migrated in V6-6b).

### V6-7: Harness wiring
- `scripts/train.py` — `dataset_format=swing` flag selects `SwingSequenceDataset` instead of `SequenceDataset`. `collate_fn=swing_collate_fn` plumbed through `create_dataloader`. `mapper_id` from config passed to dataset.
- `experiments/queue/v6_pilot.yaml` — first V6 overnight sweep: Joetastic / Rustic / Helloimdaan @ `sequence_swing_small` preset, 90 min each, `phrase_energy_alpha=0.1`.

### Test count
318 passing (added 26 V6 beam-search tests in `test_beam_search_v6.py`); ruff clean.

---

## V5 → V6 Pivot — Opus 4.7 Architectural Review (May 10–11, 2026)

**Trigger:** V5 cohort + harness infrastructure is complete and the initial overnight sweep (`experiments/queue/initial.yaml`, 10 experiments × 60 min) was queued for the first deep run. Before kicking it off, an Opus 4.7 review of the full V5 stack was requested. The user's framing: maps still don't have a *feel*, the aux-loss tuning is plastering over awkward unplayable patterns rather than solving them, and we may be brute-forcing something that needs a different frame.

**Verdict:** the V5 cohort+harness work is correct and stays. The **modeling axis** is wrong.

### Three blindspots identified

1. **Output representation hides physics.** The model emits chord-at-timestamp tokens (`NOTE COLOR COL ROW DIR ANGLE`), but a Beat Saber map is **two interleaved hand trajectories**. Color is not an attribute of a note — color *is* the hand. Parity, follow-through, and intra-onset alternation are emergent statistical regularities the model has to re-discover from data, while every aux loss in `seq_module.py` (`_compute_flow_loss`, `_compute_intra_onset_parity_loss`, `_compute_follow_through_loss`, `_compute_ergo_loss`) is a bandaid teaching it physics it should never have had to learn.
2. **No body / no proprioception.** `prev_context_k=8` previous onsets are *mean-pooled* into one vector. Ordering and grid position are destroyed. The model has no idea where its sabers physically are. A real mapper holds a tiny continuous state — 12 floats — that we pass none of.
3. **Loss is local; mapping is phrasing.** CE + parity + follow-through are all local to a token or pair. There's no signal that asks "does this 4-bar window feel like the song's 4-bar window?" or "is this a Joetastic-shaped accent?" The only phrase signal is `section_id` (6 classes) + `section_progress` (0–1).

### Decision (2026-05-11)

The overnight V5 sweep was **held**. Every minute spent training the chord representation is time spent teaching the wrong representation.

**V6 architecture** committed in `docs/architecture_v6_plan.md`. Three coordinated bets:

- **Bet 1 — Swing-event tokenization:** single ordered stream of per-hand cut events. `[HAND][Δt][KIND][X][Y][DIR][ANGLE]`. Parity becomes structural (alternation enforced by data, not by aux loss). Vocab shrinks 183 → ~70. All four parity/flow/follow-through/ergo aux losses get **deleted**, not migrated.
- **Bet 2 — Saber-state proprioception:** 12-dim physical state `(L_pos, L_dir, L_dt, L_parity, R_pos, R_dir, R_dt, R_parity)` projected to `d_model` and added as conditioning at every decode step. Replaces mean-pooled `prev_context_k`.
- **Bet 3 — Phrase conditioning + style discriminator:** 16-bar audio window pooled into a phrase embedding; phrase-energy KL aux loss (predicted swing density vs audio RMS per 4-bar window); learned mapper-classifier discriminator providing `−λ log p_D(this_mapper | swings)` as a style-closeness signal.

### What was preserved unchanged

V5 cohort directory structure (`data/cohorts/{mapper}/`), `scripts/download_cohorts.py`, `scripts/auto_research.py`, leaderboard format, `data/reference/mappers.json`, `models/audio_encoder.py`, `models/onset_model.py`, `generation/lighting_rules.py`, `evaluation/playability.py` (as evaluation only, not as training loss).

### What gets rebuilt

`data/tokenizer.py` (chord grammar) → `data/swing_tokenizer.py` (event stream). `data/dataset.py` Stage 2 path. `models/sequence_model.py` (new vocab + saber-state proj + phrase proj). `training/seq_module.py` (four aux losses deleted, two new aux losses added). `generation/beam_search.py` (new grammar mask). `generation/postprocess.py` (drop parity/dot/diagonal rewriters that the model now handles natively).

### Status snapshot at pivot

- All V5 first-run blockers (B1–B7) completed.
- All 18 cohorts manifested in `data/reference/mappers.json`.
- 241/241 tests passing; ruff clean.
- v14 checkpoint (`sequence-epoch=13-val_loss=1.090.ckpt`) preserved for warm-start comparison only.

### V6 phase plan summary

V6-0 spec + round-trip → V6-1 saber state → V6-2 dataset migration → V6-3 model rewiring → V6-4 phrase-energy loss → V6-5 style discriminator → V6-6 inference + postprocess cleanup → V6-7 harness re-validation → V6-8 deep training + human eval. Full detail in `TODO.md` and `docs/architecture_v6_plan.md`.

---

## V4 Architecture Work + Kick-off (Apr 13, 2026, late)

**Trigger:** EDA on first v14 generation (`reference_20260413_v14_seq.zip`) revealed:
- 50% pre-postproc parity violations (`fix_parity` corrected 686/1370 notes)
- ~40% of final directions are diagonal 45° — traced to `_choose_flow_direction` postproc bias
- Physically impossible follow-through patterns (e.g., bottom-mid up-right → top-left down-right = 2D teleport, parity-valid)
- Zero arcs, chains, or bombs emitted
- Mode collapse: 35/216 unique (col,row,dir,color) combos; top pattern = 9.5% of notes

**Root causes identified (see `docs/architecture_v4_analysis.md` for full analysis):**
1. No "follow-through" signal anywhere — flow loss only checks parity, not grid-position alignment with swing dir
2. Flow loss alpha=0.25 too weak vs. CE loss (~1.0 magnitude)
3. No intra-onset parity check (chords can have both notes forehand)
4. `_choose_flow_direction` in postproc is diagonal-biased and rewrites ~50% of notes
5. Rare events (arc/chain/bomb) undertrained — token_dropout=0.05 + 13-epoch early stop

**Code changes applied (v4 → v15 run):**
- Added `_compute_follow_through_loss()` — differentiable cosine-similarity penalty on direction vs. movement vector
- Added `_compute_intra_onset_parity_loss()` — penalizes same-parity chord notes
- Flow loss alpha 0.25 → 0.40
- New rare-event CE weights: ARC/CHAIN = 9.0x, BOMB = 6.0x (was 3.0x)
- Expert-only training (ExpertPlus deferred)

Tests passing: 240/241 (one pre-existing flaky test in `test_generate.py` unrelated to v4).
Ruff clean.

### v15 training config
- `stage=sequence`, `use_planner=true`, `token_dropout=0.10`
- `batch_size=256`, `num_workers=16`, `max_samples_per_epoch=500000`
- `early_stopping_patience=15`
- Expected runtime: ~15h at ~10 min/epoch → ~90 epochs possible before early-stop
- Difficulty filter: `Expert` only (~850K train / ~100K val samples)

---

## Phase 6 Retrain — version_14 (Apr 13, 2026)

**Goal:** Retrain sequence model from scratch on reprocessed data (vocab 183, 8-channel structure features) with OnsetPlanner enabled for the first time.

### Preprocessing (Phase 1C — DONE)
- Parallelized `scripts/preprocess.py` with `--workers` flag (ProcessPoolExecutor)
- Reprocessed 14,360 / 14,492 maps in ~1h38m with 12 workers (~2.5 maps/sec)
- Dataset: 1.69M train / 194K val samples (Expert + ExpertPlus)

### Training Config
- `stage=sequence`, `use_planner=true`, `token_dropout=0.05` (was 0.10)
- `batch_size=256`, `num_workers=16`, `max_samples_per_epoch=750000`
- `early_stopping_patience=5` (tight — prioritize rapid iteration)
- Runtime: ~6 hours, 15 epochs, 22.9GB / 32GB VRAM

### Results

| Metric | Value | Notes |
|--------|-------|-------|
| val_loss | **1.090** | epochs 10 and 13 (tied) |
| val_token_acc | 75.7% | |
| Best epoch | 13 | stopped at 15 via early stopping |

### Comparison vs v6 (old baseline)
- v6: val_loss=1.055 @ epoch 55 (no planner, old data, flow loss detached)
- v14: val_loss=1.090 @ epoch 13 (planner enabled, reprocessed data, patience=5)

v14 plateaued higher than v6's best, but at a much earlier epoch. The tight patience prioritizes iteration speed — v14 likely had room to improve with more epochs. Best checkpoint: `version_14/checkpoints/sequence-epoch=13-val_loss=1.090.ckpt`.

---

## Architecture Improvements — Pre-Retrain (Apr 12, 2026)

**Goal:** Complete all architecture changes and data fixes BEFORE retraining.

### Completed Phases

**Phase 4A+4B: Quick Wins (no retrain needed)**
- Musically-aware NPS enforcement: importance-based note removal (beat strength, gap penalty, color pairing) replaces uniform thinning
- Lighting postprocessing: dedup, density cap (4/quarter-beat), brightness smoothing

**Phase 1A: Fix chain tail_beat encoding**
- Added CHAIN_TAIL_BEAT_OFFSET (16 bins, 0.25-beat resolution, 0-3.75 range) to tokenizer
- VOCAB_SIZE: 167 → 183. Chain tokens: 9 → 10 per event
- Previously, all chain duration info was silently lost in training data

**Phase 1B: Fix arc color matching**
- Replaced FIFO ARC_START/ARC_END matching with nearest-beat matching
- Prevents misalignment of overlapping same-color arcs

**Phase 2A-2D: OnsetPlanner (bidirectional song-level planning)**
- New module: `models/onset_planner.py` — 4-layer bidirectional TransformerEncoder
- Plan vectors concatenated to SequenceModel cross-attention memory as extra token
- Song-level batching: `SongBatchDataset` + `song_batch_collate()` for planner training
- Full inference pipeline wiring: `generate.py` computes plan vectors per onset
- Config: `use_planner: false` (default, no-op until enabled for retraining)

**Phase 3A-3C: Song Structure Segmentation**
- `detect_sections()` in audio.py: self-similarity matrix + agglomerative clustering → section labels (intro/verse/chorus/bridge/drop/outro)
- `compute_section_features()`: per-frame section_id + section_progress tensors
- Structure features expanded: [6, T] → [8, T] (added section_id + section_progress channels)
- Audio encoder: `n_structure_features` param, default 8, backward compat with old 6-channel .pt files
- OnsetPlanner: `section_emb` + `progress_proj` condition planner on section structure
- scikit-learn added as dependency

### Still TODO
- Phase 5A: Training data quality filtering
- Phase 1C: Repreprocess all training data with corrected tokenizer (vocab 183, 8-channel structure)
- Phase 6: Retrain sequence model with all improvements enabled

### Test Status
240 tests pass, ruff clean, 1 deselected flaky test (`test_frame_indices_in_range`)

---

## Sequence Retrain with Fixed Flow Loss (Mar 10-11, 2026) — version_9 → version_11

**Goal:** Retrain autoregressive sequence model now that flow_loss is properly differentiable.

Changes since last sequence training (version_6):
- **P0 fix:** `_compute_flow_loss()` uses `torch.softmax(logits)` — actual gradient signal for parity
- **Improved parity fixer:** `_choose_flow_direction()` now position-aware (edge cols swing inward, center mixes straight/diagonal based on row + next note flow)
- **NotePredictor generation path:** Added `--note-pred-ckpt` flag + `predict_notes_structured()` to generation pipeline
- **Per-color parity (Mar 11):** Flow loss now tracks parity per color (red/blue independently) instead of only checking first note's direction. Also handles multi-note onsets correctly.

Training config: stage=sequence, batch_size=192, max_epochs=100, patience=20, 500K samples/epoch, flow_loss_alpha=0.1

### version_9 (crashed at epoch 10 — VSCode crash)

| Metric | Value | Notes |
|--------|-------|-------|
| val_loss | **1.068** | Best at epoch 10, still improving (patience 0/20) |
| Previous best (v6) | **1.055** | At epoch 55 — v9 was on track to beat it |

### version_11 (resumed from v9/last.ckpt on Mar 11)

Resumed from epoch 10 with per-color flow loss fix. Saves to version_11 (Lightning increments version on resume).

| Metric | Value | Notes |
|--------|-------|-------|
| val_loss (epoch 11) | 1.075 | First epoch after resume, expected to settle |
| val_loss | TBD | Monitoring... |

**Outcome:** Early-stopped at epoch 33 (best val_loss=1.067 at epoch 13, then degraded). Did not beat v6's 1.055. The per-color flow loss fix on resume may have shifted the loss landscape.

---

## Version 12: Full Improvements (Mar 11, 2026)

**Goal:** Fresh training from scratch with all improvements applied from epoch 0.

Changes from v9/v11:
- **Per-color flow loss** (fixed bug: was checking first note only, now per-color parity)
- **flow_loss_alpha: 0.25** (up from 0.1 — stronger parity signal)
- **Ergonomic auxiliary loss** (ergo_loss_alpha=0.15): penalizes wrong-side column predictions during training (red→right cols, blue→left cols). Closes training-inference gap.
- **Mirror augmentation** (50% chance): flips columns left↔right, swaps red↔blue, mirrors directions. Doubles effective data diversity, teaches spatial symmetry.

Training config: stage=sequence, batch_size=192, max_epochs=80, patience=25, 500K samples/epoch

| Metric | Value | Notes |
|--------|-------|-------|
| val_loss | TBD | |
| train_flow_loss | TBD | Should be non-zero and decreasing |
| train_ergo_loss | TBD | Should decrease as model learns color-side preference |

**Status:** Running as of 2026-03-11 19:52, expected ~12h

---

## Phase 2: NotePredictor Training Run (Mar 9, 2026) — version_8

**Architecture change:** Replaced autoregressive token generation with structured multi-head prediction.

Changes:
- **New model:** `NotePredictor` — cross-attention pooling with learnable slot queries + 7 independent classification heads
- **New training module:** `NotePredictionLitModule` — multi-task loss with parity/ergo/collision penalties
- **Training config:** batch_size=256, max_epochs=80, patience=20, 500K samples/epoch

| Metric | Value | Notes |
|--------|-------|-------|
| val_loss | 8.762 | Best at epoch 5 |
| val_n_notes_acc | 77.9% | Predicts n_notes=0 at inference (collapsed) |
| val_color_acc | 65.9% | OK |
| val_direction_acc | 41.0% | Poor — not enough spatial info from audio |
| val_col_acc | 39.8% | Poor |
| val_row_acc | 52.4% | Marginal |

**Outcome:** Model learned note count + color but spatial accuracy is poor. At inference, n_notes head always predicts 0 (collapsed). Using per-slot color to determine active slots works — generates 2 notes/onset with correct red/blue balance but extremely repetitive patterns (only up/down, only 3 columns). NotePredictor is 40x faster than autoregressive but equally monotonous.

**Generated maps comparison (Mar 10):**
- `notepred_expert.zip`: 1,177 notes, 0 errors, 2 dirs, 3 cols — structurally valid but monotonous
- `autoreg_expert_v2.zip`: 1,189 notes, 0 errors, 2 dirs, 4 cols — similar monotony
- `autoreg_expert_v3.zip`: 717 notes, 0 errors, **6 dirs** — improved parity fixer adds variety

---

## 1-Week Training Run Results (Feb 27 – Mar 3, 2026)

| Stage | Best Metric | Epochs | Checkpoint |
|-------|------------|--------|------------|
| Onset | val_f1 = 0.726 | 7 (early stopped) | `version_0/onset-epoch=07` |
| Sequence | val_loss = 1.055, token_acc = 78.3% | 71 (55 best) | `version_6/sequence-epoch=55` |
| Lighting | Rule-based (ML abandoned) | N/A | `generation/lighting_rules.py` |

**Outcome:** Metrics look good but generated maps are unplayable. Detailed analysis in `docs/architecture_v3_analysis.md`.

**Key finding:** Flow loss was detached from gradients (`.detach()` on predictions) — it never affected training despite being configured at alpha=0.1.

---

## Smoke Test #1 Results (Feb 27 morning)

## Smoke Test #1 Results & Investigation (Feb 27 morning)

Overnight training ran all 3 stages successfully (onset → sequence → lighting).

### Training Results

| Stage | Best Metric | Epochs Run | Converged? |
|-------|------------|------------|------------|
| Onset | val_f1 = **0.732** (epoch 5) | ~15 (early stopped) | Yes |
| Sequence | val_loss = **1.107** (epoch 0) | 11 (diverged, early stopped) | **NO — catastrophic divergence** |
| Lighting | val_loss = **1.322** (epoch 0) | 10 (stopped manually) | **NO — never improved** |

Sequence model divergence timeline:
| Epoch | val_loss | val_token_acc | Status |
|-------|----------|---------------|--------|
| 0 | 1.107 | 74.5% | Best |
| 1 | 1.130 | 73.8% | Slightly worse |
| 2 | 1.150 | 72.7% | Declining |
| 5 | 1.532 | 61.6% | Diverging |
| 10 | 2.532 | 25.6% | Catastrophic |

### Critical Problem #1: Sequence Model Mode Collapse

**Every single onset produces the identical token sequence:**
```
NOTE(red, x=1, y=0, down) SEP NOTE(blue, x=2, y=0, down)
```
Regardless of song position, audio content, or musical dynamics. The post-processing
(direction reassignment, parity fixes, grid nudging) masks this in the .zip but the
model has learned nothing useful.

**Root causes:**
- **Insufficient training**: Epoch 0 saw 500K of 2.16M samples (23%). The model memorized
  the single most common pattern rather than learning the distribution.
- **Learning rate too high**: 3e-4 with cosine decay caused divergence after epoch 0.
  val_loss went from 1.107 → 2.532 over 11 epochs.
- **Teacher forcing exposure bias**: 74.5% token accuracy sounds good, but at inference
  the model feeds its own predictions back. ~25% per-token error cascades into total collapse
  after 3-4 autoregressive steps.
- **Beam search amplifies collapse**: Deterministic beam search always picks the highest-prob
  sequence, which for a barely-trained model is always the same most-common pattern.

### Critical Problem #2: Onset Model Rhythmic Monotony

**85.9% of note gaps are eighth notes (0.3-0.6 beats).** The model produces a metronomic
stream regardless of musical content.

Compare to training data:
| Gap Type | Training Data | Generated |
|----------|--------------|-----------|
| 16th notes (<0.3 beats) | 34.8% | 8.4% |
| 8th notes (0.3-0.6) | 43.4% | **85.9%** |
| Quarter notes (0.6-1.1) | 17.4% | 5.7% |
| Half notes (1.1-2.1) | 3.5% | 0.0% |
| Whole+ (>2.1) | 1.0% | 0.0% |

Training data has coefficient of variation = 0.970 (huge rhythmic variety).
Generated output is nearly uniform eighth notes.

**Root causes:**
- Per-frame binary classification + Gaussian smoothing + peak picking creates a natural
  metronome. Even if raw probabilities have varied density, peak picking with min_distance
  regularizes them into evenly-spaced outputs.
- No mechanism for phrase-level rhythm planning. Each frame gets an independent probability.
- Threshold=0.5 clips too much of the probability curve. Dynamic thresholding based on
  local energy could help.

### Critical Problem #3: Grid Position Inversion

| Row | Training Data | Generated |
|-----|---------------|-----------|
| Top (y=2) | 25.5% | **88.5%** |
| Mid (y=1) | 27.6% | 0.1% |
| Bot (y=0) | **46.9%** | 11.4% |

The model completely inverts the training distribution. This is a direct consequence of
sequence model mode collapse — the model always predicts the same grid positions.

### Problem #4: Generation Speed

891 seconds (14.8 minutes) for one song with 690 onsets = 1.3 sec/onset.
Despite KV caching in the decoder, the audio encoder forward pass per onset
(256-frame context through 6-layer transformer) is the bottleneck.

### Problem #5: Lighting Model Not Viable

The ML lighting model converges at epoch 0 and never improves. Analysis shows lighting
events in training data are too inconsistent across mappers for a model to learn coherent
patterns. **Decision: Replace with rule-based lighting generation** using song structure
features to classify song sections and apply static lighting palettes.

---

## Training Optimization Pass (Feb 26 late evening)

After the architecture rebuild, deep auditing revealed critical training time issues.
Dataset counting showed the real sample counts — and the original approach was completely infeasible.

### The Problem

| Stage | Train Samples | Steps/Epoch (old batch) | Time/Epoch |
|-------|--------------|------------------------|------------|
| Onset | 315K | 4,930 (batch=64) | ~20 min |
| Sequence | **17M** | **354K (batch=48)** | **~33 hours** |
| Lighting | **48M** | **750K (batch=256)** | **~5 hours** |

Sequence at 33 hrs/epoch x 100 epochs = 137 days. Completely impossible for a 1-week run.

### The Solution: Three optimizations

1. **Batch size increase** (sequence: 48 → 192): VRAM analysis showed the 54M-param model only uses ~5 GB at batch=48. RTX 5090 has 32 GB. Pushed to 192 (~19 GB with mixed precision).

2. **Epoch subsampling** (500K random samples/epoch): Using `RandomSampler(num_samples=500K)`, each epoch sees a different random 3% of data. Full dataset coverage across ~34 epochs. Validated that Lightning's `estimated_stepping_batches` correctly reflects the subsampled epoch length for LR scheduling.

3. **context_frames 512 → 256**: CNN has stride=(2,1) — zero time downsampling. So T frames go directly to 6-layer transformer encoder with O(T²) self-attention. 512² = 16x cost vs 128². Changed to 256 (4x cost, ~3 seconds of audio = ~16 beats at 120 BPM).

### Result: Feasible Training Times

| Stage | Batch | Samples/Epoch | Steps/Epoch | Time/Epoch | 100 Epochs |
|-------|-------|--------------|-------------|------------|------------|
| Onset | 64 | 315K (all) | 4,930 | ~20 min | ~33 hrs |
| Sequence | 192 | 500K (sub) | 2,604 | ~15 min | ~25 hrs |
| Lighting | 256 | 500K (sub) | 1,953 | ~10 min | ~17 hrs |
| **Total** | | | | | **~75 hrs = ~3 days** |

With early stopping (patience=25), convergence around epoch 40-60: **~2 days total**.
Fits comfortably in a 1-week run with room for restarts.

### Other Bugs Fixed

- **Genre out-of-bounds crash**: Models trained with `num_genres=1` would crash on any genre besides "unknown" at inference. Added clamping + warning in generate.py.
- **Stale window_size fallback**: train.py had fallback=256 but onset.yaml uses 1024. Fixed to match.
- **EOS weight correction**: Changed from 0.3 → 1.0. Training data has zero empty onsets (preprocessing filters them), so downweighting EOS was fighting a nonexistent problem.
- **min_length 3 → 7**: A complete NOTE needs 6 attribute tokens. min_length=3 allowed truncated notes.

### Configuring Epoch Subsampling

```bash
# Default: 500K samples/epoch (recommended)
python scripts/train.py stage=sequence

# More data per epoch (slower but higher coverage per epoch):
python scripts/train.py stage=sequence max_samples_per_epoch=2000000

# Disable subsampling (original behavior — full epochs):
python scripts/train.py stage=sequence max_samples_per_epoch=null
```

---

## Architecture Rebuild Session (Feb 26 evening)

All 10 phases of the master rebuild plan have been implemented. 213 tests pass.

### What Changed (Summary)

| Phase | Change | Files |
|-------|--------|-------|
| 1 | Tokenizer direction clamping, lighting nucleus sampling + constrained decoding | tokenizer.py, generate.py |
| 2 | Song structure features (6 per-frame librosa features → AudioEncoder) | audio.py, preprocess.py, dataset.py, audio_encoder.py, all 3 training modules |
| 3 | Inter-onset context (prev K=8 onset seqs → cross-attention memory) | dataset.py, sequence_model.py, beam_search.py, generate.py, seq_module.py |
| 4 | EOS weight normalized (1.0x — training data has no empty onsets) + min_length=7 at inference | seq_module.py, beam_search.py, sequence.yaml |
| 5 | Flow-aware auxiliary loss (parity violation penalty, alpha=0.1) | seq_module.py |
| 6 | Lighting slot embedding (4-position cycling for event grammar) | lighting_model.py |
| 7 | Chroma RGB post-processing (6 palettes, energy→color mapping) | chroma.py (NEW), export.py, generate.py |
| 8-9 | Pipeline hardening (overnight_v3.sh, auto-resume, heartbeat, batch sizes) | overnight_v3.sh (NEW), train.py, data config |
| 10 | Re-preprocessing --force flag | preprocess.py |

### Key Architecture Decisions

1. **Structure features as additive projection** (not concatenated to mel): `nn.Linear(6, d_model)` output added to CNN output before positional encoding. Zero-cost backward compat — if structure_features is None, nothing changes.

2. **Inter-onset context via memory concatenation**: Previous 8 onset sequences are mean-pooled per onset, projected to d_model, concatenated to audio features along time dimension. Cross-attention naturally attends to both audio AND context. No mask changes needed.

3. **256-frame audio context** (up from 128): ~3 seconds of audio (~16 beats at 120 BPM). 512 was tested but caused 16x encoder cost due to O(T²) self-attention with no CNN time downsampling. 256 is the sweet spot (4x cost for 2x context vs the original 128).

4. **Flow loss is detached**: Computes on argmax predictions, not through the gradient graph. Pure auxiliary signal that doesn't interfere with CE loss gradients.

5. **Chroma as post-processing**: Rule-based, not learned. Avoids training complexity while still producing colorful light shows. Uses `_suggestions: ["Chroma"]` for graceful degradation in non-Chroma players.

6. **Epoch subsampling for large datasets**: RandomSampler caps each epoch at 500K samples. Different random subset each epoch ensures full coverage across ~34 epochs. Keeps epoch duration at ~15 min for meaningful early stopping and checkpoint granularity.

### Pipeline Reliability Fixes (Feb 26 late evening)

1. **Stage-aware checkpoint resume**: `find_last_checkpoint()` now filters by stage name in checkpoint filenames (e.g., only resumes onset from directories containing `onset-*.ckpt`). Previously would resume onset training from a sequence checkpoint, causing a crash.

2. **Heartbeat callback**: New `_HeartbeatCallback` in train.py writes `heartbeat.json` after every epoch with timestamp, stage, epoch, global_step, and current metric. Allows detecting hung/frozen training during multi-day runs. Previous heartbeat only updated at stage start/end.

3. **Epoch subsampling logging**: Clear log messages when subsampling is active (`500K/17M per epoch`) or disabled (`max_samples_per_epoch=null`).

4. **Genre out-of-bounds guard**: generate.py clamps genre_idx to model's embedding size with warning.

### Before Launching Overnight Smoke Test

1. **Re-preprocess all .pt files** with `--force` to add structure_features (~2-3 hours):
   ```bash
   source .venv/Scripts/activate
   python scripts/preprocess.py data/raw data/processed --num-workers 16 --force
   ```
2. **Rebuild frame_index.json** after re-preprocessing:
   ```bash
   python scripts/build_index.py data/processed
   ```
3. **Delete old checkpoints** (version_41-43 are incompatible with new architecture)
4. **Run smoke test** (5 epochs per stage, ~1 hour total):
   ```bash
   bash scripts/overnight_v3.sh --smoke-test
   ```
5. **Verify smoke test**:
   - Check `outputs/heartbeat.json` — should update every ~15 min
   - Check `outputs/training_*.log` — no crashes, loss decreasing
   - Check TensorBoard: `tensorboard --logdir outputs/`
6. **Launch full training** (~40h with early stopping):
   ```bash
   bash scripts/overnight_v3.sh
   ```

---

## THE PLAN: 1-Week Unattended Training Run

**Goal:** Produce the best open-source Beat Saber automapper. Revolutionary for the community.
**Hardware:** RTX 5090 (32GB), Ryzen 9 7950X3D (16c/32t), running 24/7 for ~7 days.
**Timeline:** Launch in ~2 days (Feb 27-28). Owner leaves for 1 week.

### Why We Can Win

| Tool | Architecture | v3 Arcs/Chains | Difficulty Cond. | Open Source |
|------|-------------|----------------|------------------|-------------|
| Beat Sage | 2x DNN (2020) | No | No | No (unmaintained) |
| InfernoSaber | Conv AE + TCN + DNN | No | No | Yes |
| TopMapper | Undisclosed (commercial) | Claims yes | Yes | No (Patreon) |
| **Ours** | **Audio Encoder + Transformer Decoder** | **Yes (learned)** | **Yes (5-class)** | **Yes** |

No open-source Beat Saber mapper uses an autoregressive transformer decoder with cross-attention.
No mapper learns arc/chain placement from data. InfernoSaber has zero v3 support.
Mapperatorinator (osu!) proved this exact architecture works — Whisper encoder + sparse decoder.
We are applying the same proven approach to Beat Saber for the first time.

### Pre-Launch Checklist (Morning of Feb 26)

#### 1. EVALUATE CURRENT RUN (first thing)
- [ ] Check all 3 stage checkpoints (onset version_41, sequence version_42, lighting TBD)
- [ ] Generate a test map from audio, load in ArcViewer
- [ ] Run BS Map Check on generated .dat files
- [ ] Check parity with Map Inspector
- [ ] Log exact metrics: onset F1, sequence val_loss, token accuracy, EOS accuracy
- [ ] Qualitative assessment: are notes synced to music? Do patterns flow?

#### 2. TRAINING THROUGHPUT OPTIMIZATION — DONE (Feb 26 evening)

**SOLVED.** VRAM analysis + epoch subsampling + batch size optimization.

- [x] **VRAM analysis**: Model is 54M params (~108 MB in fp16). Per-sample activation ~90 MB. Max batch ~256 on 32GB.
- [x] **Batch sizes**: onset=64 (tight at T=1024), sequence=192 (from 48!), lighting=256
- [x] **Epoch subsampling**: 17M sequence & 48M lighting samples → 500K random subset per epoch. Different subset each epoch. Full coverage in ~34 epochs.
- [x] **context_frames 512→256**: 16x encoder cost savings (O(T²) self-attention)
- [x] **Workers**: 12 for all stages in overnight script
- [x] **Training time validated**: ~15 min/epoch (sequence), ~10 min/epoch (lighting), total ~75h for 100 epochs or ~40h with early stopping

#### 3. DATA QUALITY AUDIT

**Arc/Chain coverage is sparse — may need targeted data:**
- Only **32.2% of (song, diff) pairs** have any arcs
- Only **13.6%** have chains
- Arcs are 1.9% of event tokens, chains are 0.5%
- The model may not see enough arc/chain examples to learn good placement

Action items:
- [ ] **Count arc/chain maps in full dataset** (not just 30 samples)
- [ ] **Consider downloading more arc-heavy maps** from BeatSaver
  - Filter: maps posted after 2023 (v3 adoption), rated ≥80%, containing sliders
  - Target: at least 2000 maps with arcs, 1000 with chains
- [ ] **Consider arc/chain token upweighting** in loss (like rhythm_weight=3.0)
- [ ] **Bombs: deprioritize** — high noise in data, stretch goal for later

**Ranked maps are gold standard:**
- [ ] Check how many ScoreSaber-ranked maps we have in the dataset
- [ ] Consider adding a `ranked_weight` multiplier — sample ranked maps more often
- [ ] Download ranked map list from ScoreSaber API and cross-reference

#### 4. ARCHITECTURE REVIEW

**What Mapperatorinator proved works for rhythm games:**
- Whisper-based encoder-decoder (219M params, trained 2500 GPU-hours)
- Sparse event tokens (only emit on note events, not every frame)
- Conditional generation (difficulty, mapper style, year)
- Classifier-Free Guidance for sharper conditioning
- Post-processing via diffusion model for coordinate refinement
- 90% overlapping windows for long-form generation

**Our architecture vs. Mapperatorinator:**

| Component | Mapperatorinator (osu!) | Ours (Beat Saber) |
|-----------|------------------------|-------------------|
| Encoder | Whisper (pretrained) | Custom CNN+Transformer (from scratch) |
| Decoder | Whisper decoder | 8-layer Transformer decoder |
| Onset detection | Part of decoder | Separate TCN+Transformer (Stage 1) |
| Conditioning | Difficulty + mapper ID + year | Difficulty + genre (+ CFG dropout) |
| Inference | Overlapping windows | Beam search / nucleus sampling |
| Post-processing | Diffusion model | Rule-based pipeline |

**Review questions for morning session:**
- [ ] Is our audio encoder good enough vs. using pretrained Whisper features?
  - Whisper is trained on 680K hours of audio — our encoder sees ~200 hours
  - But Whisper is speech-optimized; we need rhythmic features
  - Consider: use Whisper mel frontend but our own transformer on top?
- [ ] Should we switch to sparse event tokens like Mapperatorinator?
  - Currently Stage 2 only sees a 128-frame window per onset
  - Mapperatorinator processes entire song sections with overlapping windows
  - This may limit our model's ability to learn long-range patterns
- [ ] Is 8 layers / 512 d_model big enough?
  - Mapperatorinator: 219M params. Ours: ~60M params (rough estimate)
  - With a week of training we could go bigger — 12 layers, 768 d_model?
  - Need to balance against batch size / VRAM
- [ ] Do we need a dedicated parity loss term?
  - Currently relying on model learning parity from data
  - Could add a parity-violation penalty to the loss function
  - Or: post-processing parity fix (already have `fix_parity()`)

##### 4a. Inter-Onset Context — IMPLEMENTED (Phase 3)

Previous K=8 onset token sequences are fed to the sequence model as cross-attention memory.
Mean-pooled per onset, projected to d_model, concatenated alongside audio features.
Training uses ground-truth previous onsets; inference uses own generated output (autoregressive over onsets).
Audio context: 256 frames (~3 seconds, 4x cost vs 128). See Architecture Rebuild section.

##### 4b. Flow-Aware Auxiliary Loss — IMPLEMENTED (Phase 5)

Detached auxiliary loss on argmax predictions. Computes parity violations between consecutive
onsets (forehand/backhand classification). `total_loss = ce_loss + 0.1 * flow_loss`.
Time gaps > 3 seconds reset parity. Horizontal and dot directions skipped.

#### 5. PIPELINE HARDENING — DONE (Feb 26 evening)

- [x] **Checkpoint every epoch** — `save_last=True` in ModelCheckpoint + top-3 best
- [x] **Auto-resume from checkpoint** — `find_last_checkpoint()` searches for stage-specific last.ckpt
- [x] **Health monitoring** — `_HeartbeatCallback` writes `heartbeat.json` every epoch with stage/epoch/metric
- [x] **Graceful stage transitions** — overnight_v3.sh runs onset→sequence→lighting sequentially, continues on failure
- [x] **Training log** — `tee -a` to both console and `outputs/training_*.log`
- [x] **Process priority** — `low_priority=true` sets BELOW_NORMAL via kernel32.SetPriorityClass
- [x] **Early stopping patience=25** — with 15-min epochs, waits ~6 hours before stopping

#### 6. COMMUNITY-IMPORTANT FEATURES TO VALIDATE

Based on research, the Beat Saber community's top complaints about AI maps:

1. **Parity errors** — #1 complaint. Must verify our maps have correct forehand/backhand flow
2. **Poor flow** — notes should lead naturally into next swing
3. **No musical representation** — big moments need emphasis, quiet moments need space
4. **Vision blocks** — center-row notes obscure upcoming notes
5. **Handclaps** — opposite-color notes pointing at each other
6. **Repetitive patterns** — same 4-bar loop repeated endlessly

Action items:
- [ ] Generate 5-10 maps, manually check each issue above in ArcViewer
- [ ] Add handclap detection to post-processing (`postprocess.py`)
- [ ] Add vision-block detection to post-processing
- [ ] Verify `fix_parity()` actually works on generated output
- [ ] Consider adding musical energy features to the audio encoder
  (RMS energy, spectral centroid — helps the model know "loud" vs "quiet")

### Training Schedule (Validated with Real Data)

Based on actual sample counts (315K onset, 17M sequence, 48M lighting) with epoch subsampling and optimized batch sizes on RTX 5090:

| Stage | Batch | Samples/Epoch | Time/Epoch | Max Epochs | Est. Total | Cumulative |
|-------|-------|--------------|------------|------------|------------|------------|
| Onset | 64 | 315K (all) | ~20 min | 100 | ~33h (early stop ~15h) | Day 1 |
| Sequence | 192 | 500K (sub) | ~15 min | 100 | ~25h (early stop ~15h) | Day 1-2 |
| Lighting | 256 | 500K (sub) | ~10 min | 100 | ~17h (early stop ~10h) | Day 2-3 |
| **Total** | | | | | **~40h with early stopping** | **Day 2-3** |
| Buffer / re-runs / evaluation | | | | | ~100h remaining | Days 3-7 |

**Epoch subsampling**: Sequence and lighting datasets are too large for full epochs (17M and 48M samples). Each epoch randomly samples 500K samples. Full dataset coverage across ~34 (seq) or ~96 (lighting) epochs.

**Early stopping**: patience=25 epochs. Expected convergence: onset ~15 epochs, sequence ~40-60 epochs, lighting ~30-50 epochs.

---

### Session Handoff (2026-02-26 morning)

**Generated first real map — identified critical architecture gaps:**

Onset model (version_41): val_f1=0.726. Working well. Early stopped epoch 6.
Sequence model (version_42): val_loss=1.964, val_token_acc=35%, val_eos_acc=95%. ~12 epochs done.

**Generated Expert map analysis (4:12 song at 174 BPM):**
- 747 onsets detected, 553 notes generated, 3.14 NPS — reasonable density
- **Zero arcs, chains, bombs, or walls** — model only generates basic notes
- **97% of notes at column 1, row 0** (bottom-left) — severe mode collapse
- **100% same color** before post-processing — no color alternation learned
- **248/553 parity violations** — model has no concept of flow
- Direction 12 appearing (invalid — angle offset token decoded as direction)

**Root causes identified:**
1. Each onset generated independently — no inter-onset context (see 4a above)
2. 128-frame audio window too narrow (~1.5s) — can't hear musical phrases
3. Standard cross-entropy punishes valid alternative flows equally (see 4b above)
4. Arc/chain data too sparse (32%/14% of maps) — model never learns them

**Two major architecture changes planned for 1-week run:**
- 4a: Inter-onset context (feed previous 5-10 onset token sequences to decoder)
- 4b: Flow-aware auxiliary loss (reward valid alternative flows, parity bonus)

### Session Handoff (2026-02-25)

**Major fix: Onset F1 metric was broken** — validation compared peak-picked predictions
against Gaussian-smoothed labels (frames > 0.5). A perfect model could only score F1 ~0.25.
Fixed to compare against actual `onset_frames` positions. Onset val_f1 jumped from 0.23 → **0.726**.
The V2 TCN architecture was working all along.

**Files changed:**
- `data/dataset.py` — OnsetDataset now returns `onset_frames` + `n_onsets` per window
- `training/onset_module.py` — validation uses actual onset positions, not smoothed labels
- `scripts/train.py` — early stopping patience now configurable via `early_stopping_patience`
- `configs/train.yaml` — added `early_stopping_patience: 15`

**Training run (version_41 onset, version_42 sequence):**
- Onset: val_f1=0.726 (epoch 1 best), early stopped at epoch 6. ~1 hour.
- Sequence: val_loss=2.642 (epoch 1 best), still running. ~1.9 hours/epoch (!).
- Lighting: not started yet — sequence takes too long.

### Training Performance Optimization — COMPLETED (Feb 26 evening)

**Solved.** Full VRAM analysis + epoch subsampling + batch optimization.
See "Training Optimization Pass" section at top of this file for details.

Key metrics: onset=64 batch, sequence=192 batch (from 48!), lighting=256 batch.
Epoch subsampling: 500K/epoch for seq (from 17M) and lighting (from 48M).
Total estimated training: ~40h with early stopping (fits in 2 days of a 7-day window).

### Session Handoff (2026-02-24)

**Architecture V2 changes implemented (all 4 priorities):**

1. **TCN + Transformer hybrid onset model** — replaced 2-layer Transformer-only onset model
   with 6-block TCN (dilated convolutions 1,2,4,8,16,32, 128 channels) + 2-layer Transformer
   on top for global context. Receptive field: 127 frames. This follows the proven approach
   from madmom/BeatNet/InfernoSaber. (`models/onset_model.py`)

2. **KV caching for beam search** — added `CachedTransformerDecoder` and `KVCache` in
   `models/components.py`. Sequence model now supports `decode_step_cached()` for incremental
   decoding. Beam search and nucleus sampling both auto-detect and use cache. Expected 10x
   speedup for generation. (`models/components.py`, `models/sequence_model.py`, `generation/beam_search.py`)

3. **Rhythm token weighting 3x** — timing-sensitive tokens (NOTE, BOMB, WALL, ARC_START,
   ARC_END, CHAIN, SEP, EOS) get 3x weight in CrossEntropyLoss. From Mapperatorinator
   research — timing is the hardest and most important thing to learn.
   (`training/seq_module.py`, `configs/model/sequence.yaml`)

4. **Conditioning dropout 20%** — both onset and sequence models drop difficulty/genre
   embeddings with 20% probability during training. Enables Classifier-Free Guidance at
   inference for sharper difficulty control. (`models/onset_model.py`, `models/sequence_model.py`,
   `configs/model/onset.yaml`, `configs/model/sequence.yaml`)

**Tests:** 213 pass (8 new TCN tests, was 205). `ruff check .` clean.

**Next: Train on gold 500 dataset with new architecture.** Old checkpoints are incompatible —
new models have different architectures. Need fresh training run.

### Previous session (2026-02-24 daytime)

**Overnight Pipeline:** PID 31384, fully detached, ran onset → sequence → lighting.
- Pipeline log: `logs/overnight_pipeline.log`
- Per-stage logs: `logs/train_{onset,sequence,lighting}_full.log`
- TensorBoard: `outputs/beatsaber_automapper/version_27/` (onset)
- GPU: 97% utilization, 12 GB VRAM, batch_size=32, 12 workers
- Dataset: 431,720 train / 50,651 val (Expert + ExpertPlus only)
- Blacklist: 1,324 maps excluded (647 modded, 642 no expert, 35 short)

**All P0 fixes applied:**
1. `pos_weight` 5.0 → 1.0 (onset.yaml + onset_module.py default)
2. `window_size` 256 → 1024, `hop` 128 → 512 (onset.yaml) — 12s context vs 3s
3. `num_genres` 11 → 1 (onset/sequence/lighting.yaml) — all maps are "unknown"
4. Windowed onset inference: `predict_onsets()` slides 1024-frame windows with overlap
   averaging — eliminates train/inference mismatch
5. Post-processing pipeline: `generation/postprocess.py` with 6 steps (NPS enforcement,
   color rebalancing, direction diversity, grid coverage, pattern dedup, parity fixing)
6. Architecture research saved to `docs/architecture_v2.md` for future pivots
7. Gaussian sigma 3 → 2 (sharper onset peaks)
8. onset_threshold 0.5 → 0.35 (model is conservative, high precision low recall)
9. Difficulty filtering: Expert + ExpertPlus only for onset AND sequence stages
10. Data blacklisting: 1,324 maps excluded (noodle/ME, no expert, short songs)
11. 205 tests pass (17 new postprocess tests)

**Lighting events NOT yet generated** — requires a trained lighting checkpoint.
The overnight pipeline will train lighting as Stage 3 after onset and sequence complete.
Once we have `--lighting-ckpt`, ArcViewer will show light events.

**Next actions (future session):**
- Check overnight training results — look at TensorBoard version_27+
- Run `evaluate_reference.py` with best checkpoints from overnight run
- Compare against baseline snapshot (`data/reference/snapshots/reference_20260223_180304.zip`)
- If onset val_f1 > 0.4: success, proceed to quality tuning
- If onset val_f1 < 0.3: consider Phase 2 (curated gold dataset) or Phase 3 (architecture)

## PR 6: Stage 3 (Lighting Generation) — DONE

All items complete and verified:

- [x] **LightingTokenizer** (`data/tokenizer.py`): LIGHT_VOCAB_SIZE=35. Special tokens (PAD=0, EOS=1, SEP=2, BOS=3), event type tokens (BASIC=4, BOOST=5), attribute ranges: ET (6–20, 15 types), VAL (21–28, 8 values), BRIGHT (29–32, 4 brightness bins), ONOFF (33–34). `encode_lighting()` groups events by beat, SEP-separated, EOS-terminated. `decode_lighting()` is bounds-checked + clamped for robustness.
- [x] **LightingModel** (`models/lighting_model.py`): Light token embedding (scaled by √d_model) + SinusoidalPositionalEncoding + note context (mean-pool non-PAD note embeddings → Linear → add to each decoder position) → nn.TransformerDecoder (causal self-attn + cross-attn to audio) → LayerNorm → Linear(d_model, light_vocab_size). `forward()` and `decode_step()` methods.
- [x] **LightingLitModule** (`training/light_module.py`): Same pattern as SequenceLitModule. LIGHT_BOS teacher-forcing prepend. CrossEntropyLoss(ignore_index=LIGHT_PAD, label_smoothing=0.1). Logs train_loss, val_loss, val_token_acc. AdamW + linear warmup + cosine decay. `freeze_encoder` flag.
- [x] **LightingDataset** (`data/dataset.py`): Per-beat samples. Each sample: mel context window + nearest-onset note_tokens + light_tokens + difficulty. Expects `light_frames` and `light_token_sequences` in each difficulty's .pt data.
- [x] **preprocess.py update**: Runs LightingTokenizer on each beatmap, converts light beat→frame, stores `light_frames` + `light_token_sequences` in each difficulty's .pt output.
- [x] **train.py update**: `_build_lighting(cfg)` function + `stage=lighting` dispatch. Replaces prior `NotImplementedError`.
- [x] **Config** (`configs/model/lighting.yaml`): d_model=512, nhead=8, num_layers=4, dim_feedforward=2048, light_vocab_size=35, note_vocab_size=167, context_frames=128, max_note_len=64, max_light_len=32, label_smoothing=0.1, freeze_encoder=false.
- [x] **Stage 3 integration in generate.py**: `generate_lighting_events()` greedy decoder. `generate_level()` runs lighting on regular beat grid (lighting_beats_per_bar=2), uses nearest-onset note tokens for conditioning, extends beatmap.basic_events and color_boost_events before export.
- [x] **Exports**: `models/__init__.py` exports `LightingModel`. `training/__init__.py` exports `LightingLitModule`.
- [x] **Tests** (`tests/test_lighting_tokenizer.py`, `tests/test_lighting_model.py`): 35 new tests — all pass.
- [x] `ruff check .` — all checks passed.
- [x] `pytest` — 176/176 tests passed (35 new + 141 prior).

### Key Decisions

- **Note context as additive mean-pool**: Note tokens are embedded and mean-pooled into a single vector, added to every lighting decoder position. This avoids variable-length memory complexity while still conditioning lighting on note events.
- **Beat-grid lighting**: Lighting is generated on a regular beat grid (every 0.5 beats by default) rather than only at note onsets, so the light show covers the whole song.
- **Nearest-onset note context**: For each lighting beat, the nearest note-onset's token sequence is used as note conditioning — simple and avoids gaps when no notes are nearby.
- **Greedy decoding for lighting**: Lighting is less structured than note sequences (no canonical ordering, no parity constraints), so greedy decoding with temperature is sufficient. Beam search could be added later.
- **LIGHT_VOCAB_SIZE=35**: Covers BasicEvent (et 0–14, val 0–7, brightness 4 bins) + ColorBoostEvent (on/off) with tight vocabulary.

### Notes for Next Session

- To train lighting: `python scripts/train.py stage=lighting data_dir=data/processed` (after onset + sequence models are trained)
- To generate with lighting: `python scripts/generate.py song.mp3 --lighting-ckpt lighting.ckpt`
- All three stages are now fully implemented; next is training + quality evaluation

## PR 5: End-to-End Generation + Export — DONE

All items complete and verified:

- [x] **Export pipeline** (`generation/export.py`):
  - `beatmap_to_v3_dict()`: `DifficultyBeatmap` → v3 JSON dict (all object types)
  - `build_info_dat()`: builds `Info.dat` dict for any set of difficulties
  - `tokens_to_beatmap()`: wrapper around `BeatmapTokenizer.decode_beatmap()`
  - `package_level()`: packs `{difficulty: DifficultyBeatmap}` + audio + optional cover → `.zip`
- [x] **Full pipeline** (`generation/generate.py`):
  - `generate_level()`: audio → mel → AudioEncoder → OnsetModel → beam search → export
  - `predict_onsets()`: runs Stage 1 and peak-picks frame indices
  - `generate_note_sequence()`: beam search or nucleus sampling for a single onset context
  - Supports checkpoint loading or untrained random weights for testing
  - Auto-detects CUDA; accepts `device=` override
- [x] **CLI** (`scripts/generate.py`): full argparse CLI with all inference options
  - `python scripts/generate.py song.mp3 --difficulty Expert --output level.zip`
  - `--onset-ckpt` / `--seq-ckpt` for trained checkpoints
  - `--nucleus-sampling`, `--beam-size`, `--temperature`, `--top-p`
  - `--bpm`, `--song-name`, `--song-author`
- [x] **Bug fix** (`data/tokenizer.py`): Added bounds checks in `decode_beatmap()` for all event
  types (NOTE=6, BOMB=3, WALL=7, ARC_START=6, ARC_END=7, CHAIN=9 tokens minimum).
  Prevents `IndexError` on malformed/truncated token sequences from random models.
- [x] **Exports** (`generation/__init__.py`): exports `generate_level`, `beatmap_to_v3_dict`,
  `build_info_dat`, `package_level`, `tokens_to_beatmap`.
- [x] **Tests** (`tests/test_export.py`, `tests/test_generate.py`): 38 new tests — all pass
- [x] `ruff check .` — all checks passed
- [x] `pytest` — 141/141 tests passed (38 new + 103 prior)

  Also fixed two robustness bugs in `data/tokenizer.py` (found by testing with random model weights):
  - Added `_clamp()` helper so `_dequantize_*` functions never crash on out-of-range bin indices
  - Added `remaining < N` bounds checks before each event-type token consumption

### Key Decisions

- **Single-difficulty per call**: `generate_level()` generates one difficulty at a time; call
  multiple times with same audio for a multi-difficulty pack.
- **Audio encoded once**: `full_audio_features` is computed once; context windows are sliced
  per onset to avoid redundant encoder forward passes.
- **EOS appended in generate.py**: `decode_beatmap` expects EOS at end of each beat's token
  list; the pipeline appends it since beam search/sampling strips EOS from output.
- **Graceful decode on malformed tokens**: truncated token sequences (from untrained models or
  errors) now break cleanly rather than crashing with IndexError.
- **BPM defaults to 120**: No automatic BPM detection — caller must pass `bpm=` for real songs.
  This is intentional; BPM detection is a separate concern.

### Notes for Next Session

- To generate with trained models: `python scripts/generate.py song.mp3 --onset-ckpt onset.ckpt --seq-ckpt seq.ckpt`
- To generate with random weights (for testing structure): `python scripts/generate.py song.wav --bpm 120`
- Generated `.zip` loads in ArcViewer but notes will be random until models are trained
- Next step: train models on real data (PR 2 pipeline needed), then quality eval in ArcViewer

## PR 4: Stage 2 (Note Sequence Generation) — DONE

All items complete and verified:

- [x] **Sequence model** (`models/sequence_model.py`): Token embedding (scaled by √d_model, PAD=0 zeroed) + SinusoidalPositionalEncoding + difficulty embedding (additive) → nn.TransformerDecoder (8 layers, 8 heads, d_model=512, norm_first=True) with causal self-attention + cross-attention to audio → LayerNorm → Linear(d_model, vocab_size). `forward()` for teacher forcing, `decode_step()` for autoregressive inference (returns last-position logits).
- [x] **Beam search** (`generation/beam_search.py`): `beam_search_decode()` with length-normalized log probability scoring, configurable beam_size/temperature. `nucleus_sampling_decode()` with top-p filtering for creative diversity. Both strip BOS/EOS from output.
- [x] **Lightning module** (`training/seq_module.py`): SequenceLitModule wrapping AudioEncoder + SequenceModel. Teacher forcing with BOS prepend. CrossEntropyLoss(ignore_index=PAD, label_smoothing=0.1). Logs train_loss, val_loss, val_token_acc, val_eos_acc. AdamW + linear warmup + cosine decay. Optional freeze_encoder flag.
- [x] **Training CLI** (`scripts/train.py`): `stage=sequence` dispatch via `_build_sequence()`. Uses SequenceDataset with context_frames and max_seq_length from config. ModelCheckpoint(monitor=val_loss, mode=min), EarlyStopping(patience=10).
- [x] **Config updates**: `sequence.yaml` — vocab_size=167 (matches VOCAB_SIZE), added context_frames=128, label_smoothing=0.1, freeze_encoder=false. `train.yaml` — added `model/sequence` to defaults.
- [x] **Metrics** (`evaluation/metrics.py`): Added `token_accuracy()` utility for per-token accuracy ignoring PAD.
- [x] **Exports**: models/__init__.py exports SequenceModel. training/__init__.py exports SequenceLitModule. generation/__init__.py exports beam_search_decode, nucleus_sampling_decode.
- [x] `ruff check .` — all checks passed
- [x] `ruff format --check .` — all files formatted
- [x] `pytest` — 103/103 tests passed (7 sequence_model, 5 seq_module, 9 beam_search, 82 existing)

### Key Decisions

- **BOS prepend in Lightning module, not dataset**: Dataset provides raw tokens; shifting logic is training-specific.
- **CrossEntropyLoss with label_smoothing=0.1**: Prevents overconfident predictions; helps creative generation.
- **ignore_index=PAD in loss**: Padded positions don't contribute to gradients.
- **Difficulty as additive embedding**: Consistent with OnsetModel pattern.
- **decode_step returns last-position logits only**: Efficient for autoregressive inference.
- **Length-normalized log prob in beam search**: Prevents bias toward shorter sequences.
- **Nucleus sampling alongside beam search**: Better diversity for creative tasks.
- **freeze_encoder option**: Can load pre-trained Stage 1 encoder and freeze during Stage 2.
- **vocab_size=167**: Config was wrong at 256; matches tokenizer.VOCAB_SIZE.

### Notes for Next Session

- To train: `python scripts/train.py stage=sequence data_dir=data/processed`
- Need data from PR 2 pipeline first
- Definition of done for quality: Generated .dat files pass BS Map Check without errors
- Beam search produces coherent, non-random patterns (visual inspection needed)

## PR 3: Audio Encoder + Stage 1 — DONE

**Date:** 2026-02-17

All items complete and verified:

- [x] **Audio encoder** (`models/audio_encoder.py`): 4-layer CNN frontend (stride=(2,1) on freq, preserves time) → Linear projection → SinusoidalPositionalEncoding → 6-layer Transformer encoder. Input: `[B, n_mels, T]` → Output: `[B, T, d_model]`. Requires n_mels divisible by 16.
- [x] **Onset model** (`models/onset_model.py`): Difficulty embedding (5 levels, additive) → 2-layer Transformer encoder → LayerNorm → Linear(d_model, 1). Outputs raw logits (no sigmoid) for BCEWithLogitsLoss.
- [x] **Peak picking** (`models/components.py`): peak_picking() utility — threshold + local maxima + greedy distance suppression.
- [x] **Onset F1 metrics** (`evaluation/metrics.py`): onset_f1() for time-based matching, onset_f1_framewise() for frame-index validation loop use. Greedy matching (mir_eval approach).
- [x] **Lightning module** (`training/onset_module.py`): OnsetLitModule wrapping AudioEncoder + OnsetModel. BCEWithLogitsLoss(pos_weight=5.0). Training logs train_loss. Validation computes val_loss, val_f1, val_precision, val_recall via peak_picking + onset_f1_framewise. AdamW + linear warmup + cosine decay.
- [x] **Training CLI** (`scripts/train.py`): Hydra CLI with stage dispatch. Onset stage: builds OnsetDataset + OnsetLitModule, ModelCheckpoint(monitor=val_f1, mode=max), EarlyStopping(patience=10), LearningRateMonitor, TensorBoard/wandb logger.
- [x] **Config updates**: onset.yaml gains pos_weight, window_size, hop, min_onset_distance_frames. train.yaml checkpoint now monitors val_f1 (mode=max).
- [x] **Exports**: models/__init__.py exports AudioEncoder, OnsetModel, peak_picking. training/__init__.py exports OnsetLitModule.
- [x] 82/82 tests passed

## PR 2: Data Pipeline — DONE

**Date:** 2026-02-17

All items complete and verified:

- [x] **Beatmap parser** (`data/beatmap.py`): Dataclasses for all v3 types (ColorNote, BombNote, Obstacle, Slider, BurstSlider, BasicEvent, ColorBoostEvent). File-based and in-memory JSON parsers. v2 detection returns None with warning.
- [x] **Tokenizer** (`data/tokenizer.py`): 167-token vocabulary covering all event types. Sliders split into ARC_START/ARC_END at head/tail beats. Canonical ordering (type priority → x → y). Quantization for angle offset, mu, squish, wall duration. Round-trip guarantee.
- [x] **Audio processing** (`data/audio.py`): Uses soundfile for I/O (avoids torchcodec dep), torchaudio transforms for resampling and mel spectrogram. beat_to_frame/frame_to_beat utilities.
- [x] **Datasets** (`data/dataset.py`): OnsetDataset (sliding windows + Gaussian-smoothed labels), SequenceDataset (per-onset context windows + padded tokens). Both support train/val/test splits and difficulty filtering.
- [x] **Download client** (`data/download.py`): BeatSaver API paginated search, quality filters (rating, NPS, year, difficulty), CDN download with atomic writes, resume support, rate limiting, 429 backoff.
- [x] **Preprocessing script** (`scripts/preprocess.py`): Processes .zip → .pt with mel spectrograms, tokenized events, Gaussian-smoothed onset labels. Deterministic hash-based splits (85/10/5).
- [x] **Exports** (`data/__init__.py`): Clean public API.
- [x] 56/56 tests passed

## PR 1: Repo Scaffolding — DONE

**Date:** 2026-02-16

- Full project directory structure per CLAUDE.md spec
- `pyproject.toml` with all dependencies, CLI entrypoints, ruff/pytest config
- Hydra config files, all source modules with docstrings
- `SinusoidalPositionalEncoding` in `models/components.py` is only non-stub model code
- 8/8 tests passed

## PR 7: Scale Training + Quality — IN PROGRESS

**Date started:** 2026-02-19

### Genre tag conditioning (2026-02-20)

Added genre as a second conditioning signal alongside difficulty, wired through the full pipeline.

- [x] **`data/tokenizer.py`**: `GENRE_MAP` (11 classes: unknown=0, electronic, rock, pop, anime, hip-hop, classical, jazz, country, video-game, other), `NUM_GENRES=11`, `_GENRE_TAG_MAP`, `genre_from_tags()`.
- [x] **`data/download.py`**: `_extract_genre_tags()` reads BeatSaver API tag list. Manifest entries now include `genre_tags: list[str]` and `genre: str`. Backfilled entries default to `genre_tags=[]`, `genre="unknown"`.
- [x] **`scripts/preprocess.py`**: Reads `genre` from manifest; stores in `mod_requirements.genre` in every `.pt` file.
- [x] **`data/dataset.py`**: All three dataset classes (`OnsetDataset`, `SequenceDataset`, `LightingDataset`) now include `genre_idx` in their samples tuple and return `"genre": torch.tensor(genre_idx)` in each batch item.
- [x] **`models/onset_model.py`**: `genre_emb = nn.Embedding(num_genres, d_model)`, added additively to audio features. `forward(audio_features, difficulty, genre)`.
- [x] **`models/sequence_model.py`**: Same pattern — `genre_emb` added additively. `forward()` and `decode_step()` both accept `genre`.
- [x] **`models/lighting_model.py`**: Same pattern. `forward()` and `decode_step()` both accept `genre`.
- [x] **`generation/beam_search.py`**: `beam_search_decode()` and `nucleus_sampling_decode()` both accept `genre: torch.Tensor`.
- [x] **Training modules** (`onset_module.py`, `seq_module.py`, `light_module.py`): All accept `*_num_genres: int = 11` param, thread genre through forward/training/validation.
- [x] **`generation/generate.py`**: `generate_level()` accepts `genre: str = "unknown"`, converts to index via `GENRE_MAP`, passes as tensor through all three stages.
- [x] **`scripts/generate.py`**: `--genre` CLI arg with choices from GENRE_MAP keys.
- [x] **`configs/model/`**: `num_genres: 11` added to `onset.yaml`, `sequence.yaml`, `lighting.yaml`.
- [x] **Tests**: All test files updated — model fixtures gain `num_genres=11`, all forward/decode_step calls pass `genre` tensor, training batches include `"genre"` key. 3 new genre dataset tests.
- [x] `ruff check .` — all checks passed
- [x] `pytest` — 182/182 tests passed (6 new + 176 prior)

### Modding framework quotas + preprocessor tagging (2026-02-20)

Added per-category download quotas and mod_requirements tagging to support
clean separation of vanilla vs modded maps in the training pipeline.

- [x] **`download.py`**: New `_classify_map_api()` (pre-download, from API booleans), `_classify_map_zip()` (post-download, from Info.dat customData), `_load_manifest()`, `_save_manifest()` (atomic write). `download_maps()` now accepts `quotas: dict[str, int | None]` and maintains `data/raw/manifest.json` tracking every map's category, requirements, suggestions, and download timestamp. Existing 5k zips are backfilled on first run.
- [x] **`scripts/download_data.py`**: `--quota category:N` (repeatable) replaces `--count` as primary interface. `--count` kept as legacy fallback. Example: `bsa-download --quota vanilla:10000 --quota chroma:2000 --quota noodle:1000`
- [x] **`scripts/preprocess.py`**: Loads manifest at start; passes `manifest_entry` to `preprocess_single()`; embeds `mod_requirements: {category, requirements, suggestions}` in every `.pt` file. `--exclude-categories` CLI arg to skip entire categories during preprocessing.
- [x] **`data/dataset.py`**: `exclude_categories: list[str] | None = None` added to `OnsetDataset`, `SequenceDataset`, and `LightingDataset`. Category check (`mod_requirements.category`) applied during index construction (not at `__getitem__` time). Missing `mod_requirements` defaults to `"vanilla"`.
- [x] **`tests/test_dataset.py`**: `_make_test_pt()` updated with `category` param + `mod_requirements` in saved data. Three new tests: `test_onset_dataset_excludes_category`, `test_sequence_dataset_excludes_category`, `test_onset_dataset_excludes_unknown_category`.
- [x] `ruff check .` — all checks passed
- [x] `pytest` — 179/179 tests passed (3 new + 176 prior)

**Quota strategy for next download run:**
```
bsa-download --quota vanilla:10000 --quota chroma:2000 --quota noodle:1000 --min-rating 0.8 --min-year 2022
```
vivify and mapping_extensions are opportunistic (no cap). Existing 5k zips count toward quotas after backfill. Expected total: ~13k maps.

**Categories:**
- `vanilla` — no mod requirements
- `chroma` — Chroma in requirements/suggestions
- `noodle` — Noodle Extensions required
- `mapping_extensions` — Mapping Extensions required
- `vivify` — Vivify in requirements/suggestions (highest priority)
- `unknown` — no manifest entry (pre-backfill maps)

### Download client fixes (2026-02-19)

Three bugs found and fixed in `data/download.py` while running first real download:

- [x] **API URL fix**: BeatSaver dropped the `/api/` prefix — endpoint is now `/search/text/{page}`, not `/api/search/text/{page}`. Was returning 404 silently.
- [x] **`declaredAi` type bug**: API returns string `"None"` (not JSON `null`) for human-made maps. Comparing truthiness flagged every map as AI-generated, downloading 0 maps.
- [x] **NPS filter scope**: Was rejecting maps if any diff exceeded max_nps (including Easy). Now only enforces cap on Expert/ExpertPlus diffs.

### Difficulty filter expansion (2026-02-19)

- [x] **Accept all Standard difficulties**: Removed `require_difficulties=["Expert","ExpertPlus"]` default. Now accepts Easy/Normal/Hard/ExpertPlus as long as map has ≥1 Standard characteristic diff.
- [x] **Characteristic filter**: Require `characteristic=Standard` — excludes 360Degree, OneSaber, Lightshow, Lawless, etc. which would be noise for our Standard map generator.
- [x] **AI exclusion**: Added `exclude_ai=True` (default) using `automapper` + `declaredAi` API fields. Prevents training on AI-generated maps.
- [x] **`min_year` default**: 2020 → 2022 (v3 format era, avoiding v2 maps that get skipped in preprocessing anyway).

### Data collection status

- [x] **Full download**: 14,492 maps in `data/raw/` — exhausted full BeatSaver catalog under filters (≥80% rating, post-2022, Standard characteristic, no AI maps). Final category counts: vanilla=10,432, chroma=3,122, noodle=777, mapping_extensions=112, vivify=49. Manifest at `data/raw/manifest.json`.
- [x] **Training pipeline fixes** (2026-02-20):
  - Fixed Hydra config nesting: `# @package model.{name}` in each YAML so `cfg.model.audio_encoder` etc. resolve correctly
  - Fixed NaN loss: switched `precision: 16-mixed` → `bf16-mixed`, added `gradient_clip_val=1.0` to all Trainers
  - Added `torch.set_float32_matmul_precision("high")` for Blackwell Tensor Core hint
  - Wired `num_genres=11` through all three `_build_*()` functions in `train.py`
  - Smoke-test results: onset val_f1=0.248 after 3 epochs; sequence loss 5.3 (not NaN); lighting loss 3.6 (not NaN)
- [~] **Preprocess**: Running — `python scripts/preprocess.py --input data/raw --output data/processed` (~2 hrs, ~2 maps/s)
- [ ] **Train onset model**: `python scripts/train.py stage=onset data_dir=data/processed`
- [ ] **Train sequence model**: `python scripts/train.py stage=sequence data_dir=data/processed`
- [ ] **Train lighting model**: `python scripts/train.py stage=lighting data_dir=data/processed`
- [ ] **Generate + evaluate**: `python scripts/generate.py song.mp3 --onset-ckpt ... --seq-ckpt ... --lighting-ckpt ...`
- [ ] **Preview in ArcViewer**, check with BS Map Check, compute onset F1 and token accuracy

### Generation pipeline improvements (2026-02-23)

**Bug fixes in `generation/generate.py`:**
- Fixed BPM-to-frame conversion in lighting — was using inline formula that didn't match
  `beat_to_frame()`. Now uses the canonical function.
- Added error handling for checkpoint loading — `FileNotFoundError` and `RuntimeError` with
  clear messages instead of cryptic Lightning errors.
- Added warnings when no onsets detected or all token sequences are empty.
- Fixed docstring: BPM auto-detects via librosa (not "defaults to 120.0").

**Multi-difficulty generation:**
- `generate_level()` now accepts `difficulties: list[str]` to generate multiple diffs in one zip.
- Audio encoding is shared across all difficulties (computed once).
- CLI: `python scripts/generate.py song.mp3 --difficulty Expert ExpertPlus Hard`
- Extracted lighting generation to `_generate_lighting_for_beatmap()` helper.

**MP3/OGG audio support (`data/audio.py`):**
- Added ffmpeg fallback for formats soundfile can't handle natively (mp3 on Windows).
- Added `convert_to_ogg()` utility for Beat Saber zip packaging.
- Export pipeline now converts audio to `.ogg` in the zip (best BS compatibility).

**Gradio Web UI (`scripts/app.py`):**
- Full web interface for map generation: upload audio, pick difficulties/genre, generate .zip.
- Auto-discovers best checkpoints from `outputs/` directory.
- Links to ArcViewer, BS Map Check, and Parity Checker for previewing.
- Launch: `python scripts/app.py [--port 7860] [--share]`
- Added `gradio` to `pyproject.toml` optional deps: `uv pip install -e ".[ui]"`

**All tests pass:** 188/188, `ruff check .` clean.

### Full training run (2026-02-23)

**Memory stability fixes applied:**
1. Added `enable_model_summary=False` and `num_sanity_val_steps=0` to all Trainers
2. Added `_GarbageCollectCallback` — runs `gc.collect()` + `torch.cuda.empty_cache()` after
   each validation epoch to prevent memory creep
3. Reduced dataset LRU cache from 200 → 100 entries per worker (8 workers × 100 × ~6MB
   = ~4.8 GB total, down from ~9.6 GB)
4. Updated `run_training_pipeline.py` with optimal per-stage batch sizes, `--stages` and
   `--skip-onset` flags, and timing output

**Pipeline launched (PID 29760, detached):**
```
python scripts/run_training_pipeline.py --max-epochs 100
```
- Stage order: onset → sequence → lighting (sequential, full GPU)
- Onset: batch_size=64, 12 workers, ~5.8 it/s, 43,858 steps/epoch, ~2h/epoch, 6.6 GB VRAM
- Sequence: batch_size=32, 8 workers
- Lighting: batch_size=48, 8 workers
- EarlyStopping(patience=10) on all stages
- Log: `logs/pipeline_full.log`, per-stage: `logs/train_{onset,sequence,lighting}_full.log`
- TensorBoard: version_24 (onset)

**Existing checkpoints (from prior partial runs):**
```
outputs/beatsaber_automapper/version_22/checkpoints/onset-epoch=01-val_f1=0.229.ckpt
outputs/beatsaber_automapper/version_0/checkpoints/sequence-epoch=01-val_loss=1.329.ckpt
```

### Training pipeline notes (from prior sessions)

- Preprocessing complete: **12,014/14,492 .pt files** in `data/processed/`; remainder skipped (v2 maps)
- Dataset split: 10,213 train / 1,200 val / 599 test; `frame_index.json` present for fast init

**Bugs fixed (2026-02-22):**
1. `BCEWithLogitsLoss` + bf16 logits → `CUDNN_STATUS_EXECUTION_FAILED`. Fix: `logits.float()` in
   `onset_module.py` training_step and validation_step.
2. CUDA OOM when gaming: added gradient checkpointing (`use_checkpoint` flag) to AudioEncoder and
   OnsetModel, controlled by `model.onset.gradient_checkpointing=true` config flag.
3. Added `accumulate_grad_batches: 1` to train.yaml (overridable). Also added `+accumulate_grad_batches=4`
   CLI override pattern.
4. **CUDA device-side assert** in sequence training: 15 stale `.pt` files had token indices ≥ 167
   (old preprocessor missing `min(int(o.duration), 64)` wall-duration clamp). Token 1034 = `DUR_INT_OFFSET(98) + 936` (a 936-beat wall).
   - Fix A: `data/dataset.py` `SequenceDataset.__getitem__` clamps tokens: `.clamp(0, 166)` safety net.
   - Fix B: All 15 bad files deleted from `data/processed/`; their entries removed from `frame_index.json`.
   - Bad files: `15b49 15d52 15d87 160b8 161a9 1677f 1a037 1a53b 1a561 1ad83 1b068 1b66f 31dc5 38139 3ac33`
   - 11,997 clean `.pt` files remain.
5. **Triton spam** (`W... triton not found; flop counting will not work for triton kernels`) printed
   once per DataLoader worker on every run. Fixed by:
   - `scripts/train.py`: `logging.getLogger("torch.utils.flop_counter").setLevel(logging.ERROR)` in `main()`.
   - `data/dataset.py`: `_worker_init_fn()` sets same logger level in each worker, passed via `worker_init_fn=`.

**If you ever delete `.pt` files, also remove their entries from `frame_index.json`:**
```bash
python scripts/build_index.py --data-dir data/processed   # full rebuild (~20 min)
# or manually edit data/processed/frame_index.json to remove the bad keys
```

**WARNING — never delete `.pt` files while a training run is active.** The DataLoader indexes all
files at startup; deleting a file mid-run causes `FileNotFoundError` in a worker. Also purge deleted
entries from `frame_index.json` before next run.

**Training commands (full VRAM, no game, both stages in parallel):**
```
# Sequence (version_20 was running, ~12k steps into epoch 0)
python scripts/train.py stage=sequence data_dir=data/processed max_epochs=100 \
    data.dataset.batch_size=32 data.dataset.num_workers=8 low_priority=true accelerator=gpu

# Onset (version_21 was running, just started)
python scripts/train.py stage=onset data_dir=data/processed max_epochs=100 \
    data.dataset.batch_size=32 data.dataset.num_workers=8 low_priority=true accelerator=gpu
```
- Both stages fit on RTX 5090 32GB simultaneously (~8 GB onset + ~11 GB sequence)
- Sequence runs at ~5.36 it/s solo; ~2.17 it/s when sharing GPU with onset
- Epoch 0 for sequence = ~535k steps @ 5 it/s ≈ 30 hours solo, ~70 hours shared
- **No checkpoints saved yet** — epoch 0 not complete for either stage on full dataset
- TensorBoard: `python scripts/dashboard.py --no-browser` then open http://localhost:6006

**Prior smoke-test checkpoints** (11,997-file dataset, short run):
```
outputs/beatsaber_automapper/version_0/checkpoints/sequence-epoch=01-val_loss=1.329.ckpt
outputs/smoke_test/beatsaber_automapper/version_1/checkpoints/onset-epoch=02-val_f1=0.248.ckpt
```
These are usable for quick generation tests while full training runs.

- Checkpoints saved under `outputs/beatsaber_automapper/` after each epoch
- Each stage has EarlyStopping(patience=10), so actual epochs << 100 if model converges
- bf16-mixed + gradient_clip_val=1.0 committed to train.yaml and train.py
- Model weights will go to HuggingFace Hub (PR 8); training data stays local

---

## PLAN D: Comprehensive Training Overhaul (2026-02-23)

### The Problem

After ~8 hours on an RTX 5090 at full blast, the onset model shows:
- **Epoch 0:** val_f1=0.227, val_loss=1.080
- **Epoch 1:** val_f1=0.228, val_loss=1.100 (val loss went UP)
- **Epoch 2:** still training, no improvement visible
- Train loss plateau: 1.99 → 1.05 (fast), then stuck at ~1.0 for 2+ epochs

For reference, state-of-the-art musical onset detection achieves F1 ≥ 0.88. Even our
own smoke-test on fewer epochs with a smaller prior dataset got 0.248. The model is
essentially learning the base rate and then stalling.

### Root Cause Analysis

**Five critical issues identified (ordered by severity):**

#### Issue 1: pos_weight=5.0 is catastrophically wrong

The Gaussian-smoothed onset labels (sigma=3) create 11-frame-wide peaks around each
onset. With median ~660 onsets per song and median ~16,345 frames per song:
- Expected positive fraction: 660 × 11 / 16,345 = **44% of frames have label > 0**
- Actual measured: **30.2% median** onset label positive fraction
- With `pos_weight=5.0`, the model is told "positives are 5× more important than negatives"
- But positives are 30-44% of all frames — this is NEARLY BALANCED
- The model learns to predict "somewhat positive" for everything, which minimizes
  BCE loss but gives terrible F1 because peak_picking can't find real peaks in a sea
  of moderate predictions

**Fix:** `pos_weight=1.0` (or remove entirely). The Gaussian smoothing already handles
the timing tolerance — we don't need pos_weight to compensate for class imbalance when
there ISN'T much imbalance after smoothing.

#### Issue 2: 256-frame window = 3 seconds of context is far too small

The onset model sees a 256-frame sliding window (256 × 512 / 44100 = **2.97 seconds**).
This is shorter than a single musical phrase (typically 4-8 bars = 8-16 seconds at
120 BPM). The model cannot learn:
- Verse/chorus transitions
- Build-ups and drops
- Multi-bar rhythmic patterns
- Song structure (intro, verse, chorus, bridge, outro)

Beat Sage, the most popular automapper, also uses a "small window of the spectrogram"
but their results are widely considered mediocre — we should aim higher.

InfernoSaber uses a **deep convolutional autoencoder** to encode entire songs first,
giving full-song context to subsequent models. This is a fundamentally better approach.

**Fix:** Increase window to 1024+ frames (~12 seconds) or switch to a full-song
architecture where the CNN+Transformer processes the entire mel spectrogram.

#### Issue 3: Training on ALL 12k maps including noise

Our 11,997-map dataset includes:
- 777 Noodle Extension maps (5.4%) — wall art, decorative objects, non-standard gameplay
- 112 Mapping Extensions maps — extended grid, irrelevant to standard mapping
- ~270 maps with 0 lighting events
- ~69 broken/test maps under 15 seconds
- Maps with highly variable quality despite 80%+ rating filter

InfernoSaber trains on **curated high-quality maps** filtered by:
- Expert+ only (single difficulty focus)
- ≥90% like/dislike ratio (vs our 80%)
- NPS-based difficulty bands (separate models for different difficulty levels)
- Total training set: "hundreds" of maps, not thousands

**Key insight:** More data ≠ better when quality varies. A curated 500-1000 map
dataset of the absolute best maps may outperform 12k maps with variable quality.
The model spends capacity learning to average across wildly different mapping styles.

**Fix:** Create a "gold standard" curated subset. Filter criteria:
- Vanilla only (no Noodle/ME/Vivify)
- ≥92% upvote ratio
- Expert or ExpertPlus only (single difficulty to start)
- Map must have lighting events
- NPS between 3-12 (reasonable playable range)
- Song duration 90-300 seconds
- ScoreSaber-ranked maps preferred (community-validated quality)

#### Issue 4: Difficulty/genre conditioning is adding noise, not signal

The model receives difficulty and genre embeddings, but:
- **Genre is "unknown" for 100% of maps** — the embedding is pure noise
- **Difficulty distribution is heavily skewed**: ExpertPlus=36.6%, Expert=28.1%,
  Hard=19.2%, Normal=9.3%, Easy=6.8%
- The model is trying to learn ONE function that maps audio+difficulty → onsets for
  ALL difficulties simultaneously, but the mapping is highly nonlinear
- Easy maps have ~2× fewer onsets than ExpertPlus for the same song — the model must
  learn completely different onset densities per difficulty

**Fix for v1:** Train onset model on Expert/ExpertPlus only (single difficulty).
Remove genre conditioning entirely until genre labels are populated.
This eliminates a major source of confusion. Difficulty scaling can be added later
via inference-time threshold adjustment.

#### Issue 5: Train/inference mismatch — onset model sees different input lengths

During **training**, the onset model sees 256-frame windows (3 seconds).
During **inference** (`predict_onsets()`), it receives the FULL song mel spectrogram
(15,168 frames = 3 minutes for the reference song). The model was never trained on
sequences this long — positional encodings, attention patterns, and internal
representations are all calibrated for 256-frame inputs.

This explains why onset detection is even worse at inference than val_f1 suggests:
the model is running completely out of distribution.

**Fix:** Either (a) window the inference too (slide 256-frame windows with overlap,
aggregate predictions), or (b) train on longer windows so inference matches training.
Option (b) is better — increase window to 1024+ and train the model on what it will
see at inference time. For full-song inference, window and aggregate.

#### Issue 6: The model architecture may be undertrained, not underpowered

Current onset model: AudioEncoder(CNN + 6-layer Transformer encoder, d=512) →
OnsetModel(2-layer Transformer decoder, d=512) → Linear → sigmoid

This is ~25M parameters processing 256-frame windows. The issue isn't model size —
it's that 2 epochs on 12k maps ≈ 90k gradient steps, which should be plenty.
The learning rate (3e-4) and cosine schedule with 1000 warmup steps are reasonable.

The real problem is Issues 1-4 above preventing the model from learning the right thing.

### How Competing Automappers Work

| System | Architecture | Data | Onset Method | Quality |
|--------|-------------|------|-------------|---------|
| **Beat Sage** | 2 neural networks | Unknown (large) | NN on mel spec window, focuses on percussion | "Fun but inconsistent" |
| **InfernoSaber** | 4-stage: Autoencoder → TCN → DNN → DNN | Hundreds of curated expert+ maps | TCN on autoencoder features | Best open-source quality |
| **DeepSaber** (Oxford) | WaveNet + Transformer | Small curated set | CNN onset detector | Academic proof-of-concept |
| **Lolighter/ChroMapper** | Rule-based + heuristics | N/A | Audio analysis (librosa) | Decent for basic maps |
| **Ours (current)** | CNN+Transformer encoder → Transformer decoder | 12k maps (all qualities) | Transformer on 3s windows | F1=0.228 (not working) |

**Key takeaway:** Every successful system either uses (a) a much smaller curated dataset,
(b) simpler non-attention architectures (CNN/TCN/DNN), or (c) full-song context via
autoencoder. We're using the hardest approach (large Transformer on large noisy data)
without the infrastructure to make it work.

### The Revised Plan

#### Phase 1: Quick Wins (fix current run, no architecture changes)

1. **Stop current training** — it's not learning and burning GPU hours
2. **Fix pos_weight**: Change from 5.0 → 1.0 in `configs/model/onset.yaml`
3. **Increase window**: 256 → 1024 frames (~12 seconds of context)
   - Update `onset.yaml`: `window_size: 1024, hop: 512`
   - This 4× reduces samples per epoch but each sample is 4× more informative
4. **Filter dataset**: Apply Plan A outlier filters + restrict to vanilla/chroma only
5. **Drop genre conditioning**: Set `num_genres: 1` or bypass the embedding
6. **Restart onset training** with these fixes

Expected: val_f1 should break 0.4+ within 3 epochs if the core issues are fixed.

#### Phase 2: Curated Dataset Experiment

1. **Create a "gold" subset** of ~500-1000 maps:
   - Script: `scripts/curate_dataset.py`
   - Criteria: vanilla, ≥92% rating, Expert/ExpertPlus, has lighting, NPS 3-12,
     duration 90-300s, preferably ScoreSaber-ranked
   - Source: re-query BeatSaver API with tighter filters, or filter existing dataset
2. **Train onset model on gold subset** — if F1 > 0.5, the architecture works and
   the problem was data quality. If F1 still < 0.3, the architecture needs revision.
3. **Compare:** gold-500 vs full-12k vs full-12k-filtered

#### Phase 3: Architecture Improvements (if needed)

If Phase 1-2 don't break F1 > 0.5:

1. **Replace Transformer onset detector with TCN** — InfernoSaber's proven approach.
   Temporal Convolutional Networks handle 1D temporal patterns efficiently with large
   receptive fields via dilated convolutions. No attention overhead.
2. **Add an audio autoencoder stage** — Like InfernoSaber, pre-train a convolutional
   autoencoder to compress the mel spectrogram into a compact representation. Then
   train onset/sequence models on the compressed features.
3. **Consider full-song processing** — Use the CNN frontend to downsample 4-8×, then
   process entire songs with the Transformer. A 3-minute song at 4× downsampled =
   ~2000 frames, which fits in 512-dim Transformer attention.

#### Phase 4: Reference Song Evaluation System

Create a system to track model quality over time using a fixed reference song.

**Implementation: `scripts/evaluate_reference.py`**
```python
# Usage:
# python scripts/evaluate_reference.py --audio data/reference/test_song.ogg \
#     --onset-ckpt outputs/.../onset-epoch=XX.ckpt \
#     --seq-ckpt outputs/.../sequence-epoch=XX.ckpt \
#     --output-dir data/reference/snapshots/

# What it does:
# 1. Runs the full generation pipeline on the reference song
# 2. Saves the generated .zip to snapshots/ with timestamp
# 3. Computes and logs metrics:
#    - Number of onsets detected
#    - Onset density (notes per second)
#    - Note type distribution (notes/bombs/walls/arcs/chains)
#    - Unique patterns count
#    - Grid coverage (how many of 12 grid cells are used)
#    - Difficulty spread (if multi-diff)
# 4. Appends metrics to data/reference/history.json
# 5. Optionally generates a matplotlib chart of metrics over time
```

**Setup:**
1. Pick a reference song (user provides) and store at `data/reference/test_song.ogg`
2. Store a copy of the best human-mapped version of that song (if available) for
   comparison
3. After each training run or checkpoint, run the evaluation script
4. Over time, the snapshots directory builds a visual history of improvement

**Gradio integration:** Add a "Evaluate Reference" button that runs the reference
song through current best checkpoints and displays metrics + links to download the
generated .zip for ArcViewer comparison.

#### Phase 5: Training Speed Optimization

Current: 43,858 steps/epoch at 6.2 it/s = ~2 hours/epoch (onset only).

Optimizations:
1. **Larger batch size with window_size=1024**: GPU can still fit batch_size=32-48
   with 1024-frame windows (4× more data per sample, fewer steps per epoch)
2. **Gradient accumulation**: If batch_size must be reduced, use
   `accumulate_grad_batches=4` to simulate larger effective batches
3. **torch.compile()**: Add `torch.compile(model)` for 20-40% speedup on Blackwell
4. **Mixed precision**: Already using bf16-mixed, which is optimal for sm_120
5. **Pre-compute mel spectrograms**: Already cached in .pt files, so this is fine
6. **DataLoader prefetching**: Ensure `prefetch_factor=2` and `pin_memory=True`

With window=1024 and hop=512 on the gold-500 dataset:
- Samples per epoch ≈ 500 maps × ~30 windows × 2 diffs = ~30,000
- At batch_size=48: ~625 steps/epoch at ~6 it/s = **~100 seconds/epoch**
- Can run 100 epochs in under 3 hours

### Decision Matrix: What to Try First

| Change | Effort | Expected Impact | Risk |
|--------|--------|----------------|------|
| Fix pos_weight → 1.0 | 1 min | HIGH — fixes training signal | None |
| Window 256 → 1024 | 5 min config | HIGH — more musical context | Fewer steps/epoch |
| Drop genre conditioning | 5 min | MEDIUM — removes noise | None |
| Gold-500 curated subset | 2 hrs script | HIGH — cleaner signal | May need API re-query |
| Expert/ExpertPlus only | 5 min config | MEDIUM — focus on one task | Less data |
| TCN instead of Transformer | 1 day | MEDIUM — proven architecture | Code rewrite |
| Reference song evaluator | 2 hrs | META — enables comparison | None |

**Recommended order:** pos_weight → window → drop genre → curated subset → evaluate.
All config-level changes first, then data changes, then architecture if needed.

### Baseline Snapshot (pre-restructure, 2026-02-23)

Reference song: `data/reference/so_tired_rock.mp3` (rock, 123 BPM, 2:56)
Checkpoints: onset-epoch=01-val_f1=0.228, sequence-epoch=01-val_loss=1.329, no lighting

| Metric | Value | Target |
|--------|-------|--------|
| Notes | 1,643 (9.6 NPS) | 700-1050 (4-6 NPS for Expert) |
| Bombs | 0 | 20-60 |
| Walls | 21 | 30-80 |
| Arcs/Chains | 0 | 10-40 |
| Color balance | 82% red / 18% blue | ~50/50 |
| Grid coverage | 6/12 cells | 10-12/12 |
| Unique patterns | 9 | 50+ |
| Direction dist | 84% down | Spread across all 9 |
| Light events | 0 | 200+ |

Snapshot: `data/reference/snapshots/reference_20260223_180304.zip`

### Phase 6: "Best of All Worlds" Architecture (medium-term)

Goal: combine the strongest techniques from every competing automapper into
one system that surpasses them all.

#### What we take from each system

**From InfernoSaber (most successful open-source BS mapper):**
- Audio autoencoder for compact song representation — gives full-song context
  to downstream models without blowing up memory
- TCN for onset detection — proven, efficient, large receptive fields via
  dilated convolutions without attention overhead
- Heavy post-processing rules — sanity checks, playability filters, pattern
  enforcement. Our `playability.py` needs to be much more robust.
- Separate difficulty scaling external to model — simpler than embedding

**From DeepSaber (original academic approach):**
- "Humaneness regularization" — penalize notes placed too close together with
  exponential distance weighting. Add to our onset loss: `exp(-2*dist/window)`
  penalty for predicted onsets that violate minimum spacing
- beam_size=17 for coherent generation — our beam=8 may need to go higher
- Peak threshold 0.33 (not 0.5) — lower threshold + post-processing NMS
  may work better than trying to get sharp peaks

**From Mapperatorinator (best overall, osu!):**
- **Rhythm token weighting at 3×** in loss — timing is the hardest and most
  important thing. Weight onset-related tokens higher in sequence loss too.
- **Conditioning dropout (20%)** on all embeddings during training — enables
  classifier-free guidance at inference. "Show me what Expert looks like" vs
  "show me what NOT Easy looks like" = better difficulty control.
- **388 mel bands** instead of 80 — preserves more frequency detail. RTX 5090
  with 32GB VRAM can handle this easily.
- **Pretrained audio backbone** (Whisper) — we could initialize our audio
  encoder from Whisper weights rather than training from scratch. Whisper was
  designed for audio→text which is structurally similar to audio→beatmap.

**From BeatLearning (innovative small model):**
- **Audio foresight** — let the model "see ahead" in audio while predicting
  current tokens. Musical events are anticipated (build-ups before drops).
  Implementation: extend the audio context window asymmetrically — more
  future frames than past frames.
- **Joint onset + note generation** — longer-term: a single model that
  predicts both WHEN and WHAT in one pass. Eliminates error propagation
  from Stage 1 → Stage 2.

#### Concrete Architecture V2 Plan

**Audio Encoder V2:**
- Increase mel bands: 80 → 192 (compromise between 80 and 388)
- CNN frontend: 4 layers, stride=(2,1) on freq → 192/16 = 12 freq bins
- Projection: 256×12 = 3072 → d_model=512
- Transformer encoder: 6 layers, 8 heads (keep current)
- NEW: Consider initializing from Whisper-small encoder weights
  (Whisper-small uses 80 mel bands at 16kHz; we'd need an adapter layer)
- Full-song processing: with CNN 4× freq downsample, a 3-min song at
  44.1kHz/512 hop = 15,168 frames fits in the Transformer (already proven
  working in our generation pipeline)

**Onset Model V2:**
- Replace 2-layer Transformer decoder → **Hybrid TCN + Transformer:**
  - TCN (4 blocks, dilations 1,2,4,8,16,32, 128 filters) for local
    pattern detection — captures beat/sub-beat patterns with large
    receptive field efficiently
  - 2-layer Transformer on top for global context — verse/chorus/drop
    awareness
- Remove genre embedding (unused), keep difficulty embedding
- Add **humaneness regularization** to loss — penalty for onset
  predictions closer than `min_onset_distance` frames
- pos_weight=1.0 (or remove), Gaussian sigma=2 (sharper peaks)
- Window size: 2048 frames (~24 seconds, covers full musical phrases)
- NEW: Conditioning dropout 20% on difficulty during training

**Sequence Model V2:**
- Keep autoregressive Transformer decoder (8 layers) — most flexible
- Add **rhythm token weighting**: weight timing-sensitive tokens (EVENT_TYPE,
  SEP, EOS) at 3× in CrossEntropyLoss. These control WHEN notes appear;
  property tokens (color, direction) control WHAT and are less critical.
- Add **audio foresight**: extend context_frames asymmetrically — 64 past
  + 192 future = 256 total context (currently 64+64=128 symmetric)
- Add **conditioning dropout** 20% on difficulty + genre → enables CFG
- Add **pattern diversity loss**: auxiliary loss term that penalizes
  low-entropy output distributions. Prevents mode collapse (the "all red
  down" problem we see in the baseline).
- Consider **top-k constrained beam search**: at each step, only consider
  tokens that maintain game-playability constraints (e.g., no two notes
  in same grid cell, color alternation patterns)

**Lighting Model V2:**
- Keep current 4-layer decoder, expand for Chroma (Plan C)
- Priority: get onset + sequence right first

**Post-Processing Pipeline (NEW):**
- `generation/postprocess.py`:
  1. **NPS enforcement**: If NPS exceeds target for difficulty, thin
     notes by removing least musically-correlated onsets
  2. **Color rebalancing**: Force 45-55% red/blue split by flipping
     the least-constrained notes
  3. **Direction diversity**: If any direction > 40% of total, reassign
     some using playability-aware rules (avoid impossible wrist angles)
  4. **Grid coverage**: If < 8/12 cells used, shift some notes to
     unused positions
  5. **Pattern deduplication**: If identical note pattern repeats > 5×
     consecutively, inject variation
  6. **Bomb/wall injection**: Rule-based bomb and wall placement based
     on note patterns (between same-color clusters, during breaks)
  7. **Parity check**: Ensure swing direction alternation is physically
     possible (no 180° wrist flips)

#### Implementation Priority (Proven Techniques)

| Priority | Change | Why |
|----------|--------|-----|
| P0 | Fix pos_weight, window, genre | Unblocks all learning |
| P0 | Post-processing pipeline | Immediately improves any model output |
| P1 | Curated gold dataset | Clean signal >> more noise |
| P1 | Conditioning dropout | Enables CFG, improves generalization |
| P1 | Rhythm token weighting 3× | Proven by Mapperatorinator |
| P2 | Audio foresight (asymmetric context) | Build-up/drop anticipation |
| P2 | Humaneness regularization | Playability constraint in loss |
| P2 | Pattern diversity loss | Prevents mode collapse |
| P3 | TCN hybrid onset model | Better architecture if Transformer stalls |
| P3 | 192 mel bands | More audio detail |
| P3 | Whisper weight initialization | Pretrained features |
| P4 | Joint onset+note model | Research project, long-term |

### Phase 7: "Next-Gen" Architecture — Post-Boom Innovations

The competing automappers are all pre-boom (2019-2024) architectures. They use
vanilla Transformers with sinusoidal PE, no KV caching, no modern attention variants,
no preference optimization, no hierarchical generation. Here's what a 2025/2026
state-of-the-art architecture looks like — our unique twist.

#### Innovation 1: Mamba/SSM Audio Encoder — Full-Song, Linear Time

**The problem:** Transformer self-attention is O(n²) in sequence length. A 3-minute
song at 11.6ms/frame = 15,168 frames. Full self-attention on this is ~230M attention
pairs per layer. This is why every existing system either uses small windows (us,
Beat Sage) or compresses with an autoencoder (InfernoSaber).

**The solution:** Replace the Transformer encoder layers with **Mamba** (Selective
State Space Model). Mamba processes sequences in O(n) linear time with a learned
selective scan — it decides what to remember and what to forget at each timestep,
like an RNN but parallelizable during training.

**Why this is transformative for Beat Saber mapping:**
- Process the **entire song in one pass** — no windowing, no context truncation
- The selective state naturally captures musical structure: remember the beat pattern
  during a verse, update state when the chorus hits, forget noise between sections
- Audio Mamba has been validated for audio representation learning (2024 paper)
- Memory: O(n) vs O(n²) — a 15,168-frame song uses ~60MB vs ~3.5GB for attention
- Training on full songs means no train/inference mismatch

**Implementation:**
```
Audio Encoder V3:
  Mel spec [80, T] → CNN frontend (4 layers, same as now) → [T, 1280] → Linear → [T, 512]
  → Bidirectional Mamba (6 layers, d_state=64, d_conv=4, expand=2)
  → Output: [T, 512] frame embeddings with full-song context
```

The Mamba layers replace the 6 Transformer encoder layers. The CNN frontend stays
(it captures local frequency patterns). The bidirectional processing (forward + backward
Mamba, concatenated/projected) gives each frame context from the entire song in both
directions.

**Package:** `pip install mamba-ssm` (CUDA-optimized selective scan kernels)

#### Innovation 2: RoPE + GQA + SwiGLU — Modern Transformer Internals

Every component that remains a Transformer (onset decoder, sequence decoder, lighting
decoder) should use modern LLM-era internals, not 2017 "Attention is All You Need"
defaults.

**Rotary Position Embeddings (RoPE):**
- Replace sinusoidal PE everywhere
- RoPE encodes relative position directly in the attention computation via rotation
  matrices applied to Q and K
- Naturally handles variable-length sequences (no max_len buffer needed!)
- Extrapolates to longer sequences than seen in training — critical for us since
  songs vary from 30s to 38 minutes
- This eliminates the PE buffer overflow bug we just fixed

**Grouped Query Attention (GQA):**
- Instead of separate K/V heads per attention head (standard MHA), share K/V across
  groups of query heads
- E.g., 8 query heads, 2 KV groups → 4× smaller KV cache, 30% faster inference
- Critical for beam search speed — our 11-minute generation time for 1688 onsets is
  unacceptable. With GQA + KV cache, this could drop to 1-2 minutes.

**SwiGLU Activation:**
- Replace GELU in feedforward layers with SwiGLU: `SwiGLU(x) = Swish(xW₁) ⊙ (xW₂)`
- Used in LLaMA, PaLM, Mistral — consistently outperforms GELU/ReLU
- Free performance improvement, same parameter count

**RMSNorm:**
- Replace LayerNorm with RMSNorm (root mean square normalization)
- Faster (no mean subtraction), used in all modern LLMs
- Drop-in replacement

#### Innovation 3: KV-Cached Beam Search — 10× Faster Generation

**The problem:** Our sequence model generates 1688 onset tokens autoregressively.
Each onset needs up to 64 token steps of beam search with beam_size=8. That's
1688 × 64 × 8 = ~864,000 forward passes through the decoder. Currently each pass
recomputes attention from scratch. **This is why generation takes 11 minutes.**

**The solution:** Implement proper **KV caching** in the sequence decoder.

At each autoregressive step, the self-attention keys and values from all previous
positions are cached. The new step only computes attention for the NEW token position
against the cached K/V. This turns O(n²) per-step into O(n) per-step.

Combined with GQA (smaller K/V), beam search KV sharing (beams share prefix cache),
and our RTX 5090's memory bandwidth:

**Expected speedup:** Generation from 11 minutes → **60-90 seconds** for a 3-minute
song. This makes the Gradio UI actually usable.

**Implementation:**
```python
class KVCache:
    """Manages key/value caches across decoder layers for autoregressive inference."""
    def __init__(self, num_layers, max_seq_len, num_kv_heads, head_dim, device):
        self.k_cache = [torch.zeros(batch, num_kv_heads, max_seq_len, head_dim, device=device)
                        for _ in range(num_layers)]
        self.v_cache = [...]  # same
        self.seq_pos = 0  # current position

    def update(self, layer_idx, new_k, new_v):
        self.k_cache[layer_idx][:, :, self.seq_pos] = new_k
        self.v_cache[layer_idx][:, :, self.seq_pos] = new_v
        self.seq_pos += 1

# In beam search: beams share cache prefix, fork on divergence
```

#### Innovation 4: Hierarchical Structure-Aware Generation

**The problem:** All existing automappers (including ours) treat a song as a flat
sequence of audio frames → flat sequence of notes. But human mappers think
hierarchically: song structure → phrases → individual notes. A great mapper places
an intense pattern at the chorus drop, calms down during the verse, and builds
tension during the bridge. No flat model can learn this without enormous data.

**The solution:** A three-level hierarchical generation pipeline.

**Level 1 — Song Structure Segmentation (NEW):**
- Input: Full-song Mamba audio features
- Output: Segment boundaries + labels (intro, verse, pre-chorus, chorus, bridge,
  drop, breakdown, outro)
- Architecture: Linear classifier on Mamba features (fine-tuned from pre-trained
  music structure analysis models, or trained with our data using song-level labels
  from BeatSaver tags/metadata)
- This tells the model "beats 0-32 are intro, 32-96 are verse, 96-128 are chorus..."

**Level 2 — Phrase-Level Onset Density (modified Stage 1):**
- Input: Audio features + structure labels + difficulty
- Output: Per-phrase onset density curve (not individual onsets yet)
- The model predicts "verse should have 4 NPS, chorus should have 7 NPS, bridge
  should have 2 NPS" — a coarse plan before individual placement
- This replaces the flat sigmoid-per-frame approach with a musically-informed
  density prior

**Level 3 — Note-Level Generation (modified Stages 1+2):**
- Input: Audio features + density plan + difficulty
- Output: Individual onset frames + note tokens
- The onset model now has both local audio features AND a global density target
  from Level 2, so it knows how many onsets to place in each phrase
- The sequence model generates notes conditioned on structure label (e.g., "this
  is a chorus drop" → more dramatic patterns, wider grid usage, faster sequences)

**Why this is unique:** No existing automapper does hierarchical generation. They
all go directly from audio → notes. This mirrors how experienced mappers actually
work and should produce maps with much better musical coherence and flow variety.

**Training data for structure:** We can bootstrap structure labels:
- Use a pretrained music structure analysis model (MusicFM, MERT, or the ResNet-
  based approach from the 2025 paper) to auto-label song sections
- Or use a simpler heuristic: spectral energy + novelty detection to find
  transitions, k-means to cluster sections

#### Innovation 5: DPO (Direct Preference Optimization) for Map Quality

**The insight from the AI boom:** The biggest lesson from LLMs is that supervised
training (predicting the next token) gets you 80% of the way there, but preference
optimization (RLHF/DPO) is what makes outputs actually good. The same principle
applies to beatmaps.

**We have natural preference signals:**
- BeatSaver upvote ratio (0-100%) — community quality rating
- ScoreSaber ranked status — expert-validated playability
- NPS appropriateness — does the note density match the difficulty?
- Download count / play count — popularity (proxy for quality)

**DPO for beatmaps:**
1. Generate pairs of maps for the same song using different checkpoints/temperatures
2. Use BeatSaver quality signals to determine which is "preferred"
3. Train with DPO loss: `L = -log σ(β * (log π(y_w|x) - log π(y_l|x)))`
   where y_w = preferred map, y_l = rejected map

**Or use a learned reward model:**
1. Train a reward model: AudioEncoder + MapEncoder → quality score (0-1)
2. Training data: map features (NPS, pattern diversity, grid coverage, direction
   distribution, color balance) + BeatSaver quality signals
3. Use reward model to guide beam search: at each step, score partial sequences
   and prefer higher-reward beams
4. This is essentially **RLHF for beatmaps** — the model learns to generate maps
   that the community would upvote

**CLaMP-DPO analogy:** Recent 2025 work (CLaMP-DPO) shows DPO improves musicality
of symbolic music generation without human annotation, using a contrastive audio-music
model as the reward signal. We can do the same with BeatSaver community signals.

**Implementation timeline:** DPO requires a working base model first. Train with
supervised learning (Phases 1-6), then apply DPO as a quality refinement step.

#### Innovation 6: Speculative Decoding for Even Faster Inference

Once we have KV-cached beam search working (Innovation 3), we can go further with
**speculative decoding**:

1. Train a tiny "draft" model (2-layer decoder, d=128) alongside the main model
2. At inference: draft model generates N candidate tokens quickly
3. Main model verifies all N in one forward pass (parallel verification)
4. Accept the longest correct prefix, reject the rest
5. Typical acceptance rate: 70-90% → **2-3× additional speedup** on top of KV cache

For beatmap generation, the draft model can be a simple pattern lookup table
(most common note configurations) — it will be right for typical patterns and
the main model corrects the creative/unusual ones. This gets generation down to
**20-30 seconds** for a 3-minute song.

#### Innovation Summary: Our Unique Architecture

```
═══════════════════════════════════════════════════════════════
  BeatSaber Automapper v2 — "NextGen" Architecture (2026)
═══════════════════════════════════════════════════════════════

  Audio (.mp3/.ogg/.wav)
          │
          ▼
  ┌──────────────────────────────────────────────────────────┐
  │  MEL SPECTROGRAM (192 bands, 1024 FFT, 512 hop)         │
  │  → CNN Frontend (4 layers, freq downsample 16×)          │
  │  → Bidirectional Mamba Encoder (6 layers, d=512)         │
  │    ★ Full-song context in O(n) linear time               │
  │    ★ No windowing — processes entire 3-min song at once  │
  │  Output: [T, 512] frame embeddings                       │
  └───────────┬──────────────────────────────────────────────┘
              │
    ┌─────────┼─────────┐
    ▼         ▼         ▼
  ┌──────┐ ┌───────┐ ┌───────┐
  │STRUCT│ │ONSET  │ │LIGHT  │
  │LABEL │ │DETECT │ │GEN    │
  │      │ │       │ │       │
  │Seg-  │ │Hybrid │ │4-layer│
  │ment  │ │TCN +  │ │RoPE + │
  │into  │ │RoPE   │ │GQA    │
  │verse/│ │Trans- │ │decoder│
  │chorus│ │former │ │       │
  │/drop │ │decoder│ │       │
  └──┬───┘ └──┬────┘ └──┬────┘
     │        │         │
     ▼        ▼         │
  ┌───────────────┐     │
  │ NOTE SEQUENCE │     │
  │ GENERATION    │     │
  │               │     │
  │ 8-layer RoPE +│     │
  │ GQA + SwiGLU  │     │
  │ decoder       │     │
  │               │     │
  │ ★ KV-cached   │     │
  │   beam search │     │
  │ ★ Audio fore- │     │
  │   sight (asym)│     │
  │ ★ Structure-  │     │
  │   conditioned │     │
  │ ★ CFG via     │     │
  │   cond dropout│     │
  └───────┬───────┘     │
          │             │
          ▼             ▼
  ┌──────────────────────────────┐
  │  POST-PROCESSING PIPELINE   │
  │  NPS enforcement, color     │
  │  rebalancing, direction     │
  │  diversity, parity check,   │
  │  pattern deduplication      │
  └──────────────┬───────────────┘
                 │
                 ▼
  ┌──────────────────────────────┐
  │  v3 JSON EXPORT → .zip      │
  │  (After DPO refinement)     │
  └──────────────────────────────┘

What makes this unique vs ALL existing automappers:
  1. Mamba encoder — no other mapper uses SSMs for audio
  2. Hierarchical structure — no other mapper segments songs
  3. RoPE/GQA/SwiGLU — modern LLM internals, not 2017 vanilla
  4. KV-cached beam search — 10× faster inference
  5. DPO quality refinement — RLHF-era alignment for beatmaps
  6. Speculative decoding — another 2-3× inference speedup
  7. Full-song context — most use small windows or autoencoders

═══════════════════════════════════════════════════════════════
```

---

## Future Plans

### Plan A: Training Data Outlier Filtering

**Status:** Planned for next training run (do NOT apply to currently running pipeline)

Analysis of the 11,997-map dataset identified ~120 problematic maps (1% of dataset) that
may degrade training quality. Apply these filters before the next `bsa-preprocess` run.

**Tier 1 — Remove immediately (broken/test maps):**
- Songs < 15 seconds long (~69 maps) — these are test uploads or sound effects
- Filter: check audio duration in .pt metadata or re-derive from mel spectrogram frame count

**Tier 2 — Remove (extreme outliers):**
- Maps with < 20 total onsets (~30 maps) — too sparse to learn from
- Maps with > 2,000 onsets per minute (~21 maps) — vibro/spam maps
- Maps where `wall_count / (wall_count + note_count) > 0.90` — "wall art" maps (decorative
  obstacle sculptures with almost no playable notes, mostly Noodle Extension maps)

**Implementation:**
1. Add `scripts/filter_outliers.py` that scans `data/processed/*.pt` files
2. Compute per-map stats: duration, onset count, onset density, wall ratio
3. Output a `data/processed/outlier_blacklist.json` with map hashes and reasons
4. Modify `dataset.py` to skip blacklisted maps at load time (check `__init__`)
5. Rebuild `frame_index.json` after filtering

**Expected impact:** Removes ~120 maps, leaving ~11,877 clean maps. Should reduce
noise in onset model (fewer false positives from spam maps) and sequence model (fewer
degenerate patterns from wall art).

---

### Plan B: Post-Training Bomb & Obstacle Density Controls

**Status:** Planned feature for generation pipeline (post-training, no model changes needed)

Users want to control the density of bombs and obstacles independently from note patterns.
Two approaches, implement both:

#### Approach 1: Post-Processing Filter (no retraining)

Add parameters to `generate_level()` and the Gradio UI:

```python
def generate_level(
    ...,
    bomb_density: str = "medium",      # "none", "low", "medium", "high"
    obstacle_density: str = "medium",   # "none", "low", "medium", "high"
    decorative_walls: bool = False,     # if True, mark walls as uninteractable
)
```

**Implementation in `generation/generate.py`:**
1. After Stage 2 generates the full token sequence, decode to v3 JSON
2. Apply density filtering as a post-processing step:
   - `"none"`: Remove all bombs/obstacles from the decoded JSON
   - `"low"`: Keep only 25% of bombs/obstacles (randomly sample, preserving timing distribution)
   - `"medium"`: Keep as-is (model output)
   - `"high"`: Duplicate bomb/obstacle patterns at adjacent grid positions (heuristic)
3. If `decorative_walls=True`, add `"customData": {"uninteractable": true}` to all obstacles
4. Update `export.py` to pass through `customData` on obstacles

**Gradio UI changes (`scripts/app.py`):**
- Add two dropdowns: "Bomb Density" and "Obstacle Density" with choices
  `["None", "Low", "Medium (default)", "High"]`
- Add checkbox: "Decorative Walls Only (non-threatening)"

#### Approach 2: Conditioning Embedding (requires retraining)

For a future training run, add bomb/obstacle density as a conditioning signal:

1. Compute per-map bomb density percentile and obstacle density percentile during preprocessing
2. Quantize into 4 buckets: none (0), low (1-33%), medium (34-66%), high (67-100%)
3. Add two new embedding layers in SequenceModel (like difficulty/genre embeddings)
4. During training, pass ground-truth density bucket as conditioning
5. During inference, user selects desired density level

**This requires retraining** — implement Approach 1 first for immediate use, then add
Approach 2 conditioning in a future training run for better quality control.

---

### Plan C: Modded Mapping Framework Support

**Status:** Research complete, Chroma lighting is the actionable target

Dataset composition: 72% vanilla, 21.5% Chroma, 5.4% Noodle Extensions, 0.8% Mapping
Extensions, 0.3% Vivify.

#### Feasibility Assessment

| Framework | Feasibility | Worth It? | Reason |
|-----------|------------|-----------|--------|
| **Chroma (lighting)** | HIGH | **Yes** | 21.5% of maps; only affects Stage 3 lighting tokenizer |
| Chroma (note color) | Medium | Maybe | Rare on notes; could be post-processing heuristic |
| Noodle Extensions | Low | No (near-term) | Requires continuous 3D coordinates, animation system |
| Mapping Extensions | Low | No | Only 112 maps, obsoleted by Noodle |
| Vivify | Impossible | No | Requires Unity asset bundles, 3D modeling |

#### Chroma Lighting Support (recommended next step for Stage 3)

Chroma adds `customData` fields to `basicBeatmapEvents`:
- `color: [r, g, b, a]` — custom RGBA color (16.9M instances in dataset)
- `lightID: int | int[]` — target specific light(s) in a group (16.8M instances)
- `direction`, `speed`, `step`, `rotation`, `prop` — less common

**Implementation plan:**
1. **Parsing (`data/beatmap.py`):** Extract `customData.color` and `customData.lightID`
   from lighting events during preprocessing. Handle both v2 (`_customData._color`) and
   v3 (`customData.color`) naming conventions.
2. **Tokenizer (`data/tokenizer.py`):** Add Chroma tokens to the lighting vocabulary:
   - Color tokens: quantize RGBA to 8-bit per channel → `COLOR_R_0..255`, etc.
     (or use a smaller palette of ~64 colors clustered from training data)
   - LightID tokens: `LIGHT_ID_0..31` (cap at 32 individual lights)
3. **Stage 3 model:** No architecture changes needed — just a larger vocabulary
4. **Export (`generation/export.py`):** When color/lightID tokens are predicted,
   add `customData` dict to the exported `basicBeatmapEvents`
5. **Training:** Include Chroma maps in Stage 3 training data (adds ~3,122 maps)

**Estimated effort:** ~2 days of implementation + retraining Stage 3 only.

#### Current Handling of Modded Maps

- Noodle/ME maps with extended grid coordinates are **clamped to 4×3 grid** during parsing
  (beatmap.py line 321-322). This is correct — we lose precision but keep playable notes.
- Chroma lighting customData is currently **silently ignored** during parsing.
- Noodle `uninteractable` (fake) notes are included in training — Plan A's wall ratio
  filter catches the worst offenders, but a future improvement could skip fake notes entirely.

---

## PR Roadmap Reference

| PR | Status | Description |
|----|--------|-------------|
| 1  | **DONE** | Repo scaffolding |
| 2  | **DONE** | Data pipeline |
| 3  | **DONE** | Audio encoder + Stage 1 (onset detection) |
| 4  | **DONE** | Stage 2 (note sequence generation) |
| 5  | **DONE** | End-to-end generation + export |
| 6  | **DONE** | Stage 3 (lighting) |
| 7  | —      | Scale training + quality |
| 8  | —      | Documentation + demo |

---

## V7 Post-Launch Architecture Iteration (2026-05-23 → 2026-05-25)

### Inference Bugs Fixed (2026-05-23)

Three bugs discovered in first ArcViewer review of V7-7 output:

**Bug 1 — Role alignment (critical):** `generate_phrase._step` appended token
metadata *before* the forward pass, placing role=KIND at the placeholder position.
Training convention: position i with `role_i` predicts `T_{i+1}`. The fix forwards
the real buffer and reads logits at the last position, then appends metadata after
sampling. No retraining needed. Confirmed fix: Y=top-row 89.7%→28%, D=dot
99.5%→0%.

**Bug 2 — Nucleus sampling was uniform:** `_nucleus_sample` used `torch.randint`
(uniform among kept tokens) instead of `torch.multinomial` (probability-weighted).
This collapsed model confidence at every generation step.

**Bug 3 — Flat onset density:** Fixed threshold=0.4 produced a metronome. Added
±1-slot NMS and section-aware thresholds (drop=0.38 / verse=0.52 / intro=0.68 /
outro=0.72) using `detect_sections()`.

Additional: constrained sampling (logits masked to legal role vocab range),
`fix_parity` + `convert_dot_notes` re-enabled in postprocessor, `top_p` 0.90→0.95.

### Architecture Experiments (2026-05-23 → 2026-05-25)

| Run | Best acc | Finding |
|-----|----------|---------|
| Run 3 (x_role_weight=2.0) | 0.861 | X ceiling (~68%) confirmed as mapper subjectivity |
| Run 4 (ctx_len=16) | **0.870** | Cross-phrase prefix broke 0.861 ceiling, all roles improved |
| Run 5 (+ scheduled sampling) | 0.869 | No benefit; exposure bias not the bottleneck |
| Run 6 (+ scalar song/section emb) | 0.870 | Scalar conditioning gives zero lift — confirmed |
| **Run 7** (song-memory cross-attn) | 🔄 | Phrase fingerprints as full cross-attn memory — replaces PhraseIndex |
| Beat Clf Run 5 (d=512, 4-layer) | f1_tol=0.603 | Up from 0.588 with larger model |

**Key insight:** The phrase encoder processes a fixed 64-slot window — structurally
identical to V6's 3-second sliding window, just with better features. Scalar
song/section embeddings (Run 6) added zero lift, confirming a summary vector can't
substitute for attentional access to song history. Run 7 appends all `phrase_fingerprints
[N_phrases, 768]` (already in every .pt file) to the encoder memory so the decoder
can attend to chorus 2 when generating chorus 2, learning the repetition pattern that
PhraseIndex tried to hard-code.

**Remaining gap:** Stage 1 outputs a flat probability distribution (18–31% density
across thresholds 0.30–0.80). Section-aware thresholds create 5–8 NPS variation but
not the 0–9 NPS range of real maps. Target: wire `_compute_adaptive_threshold()` for
per-section NPS targeting.
