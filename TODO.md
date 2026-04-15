# Beat Saber Automapper — V5 Plan (Style-First Architecture)

**Last updated:** 2026-04-14
**Status:** V4 (v15 run) DEPRECATED. Committing to V5 — style-cohort training + auto-researcher + trajectory output.
**North star:** A player plays a generated map and says *"who mapped this?"* — not *"is this AI?"*

---

## Why We Pivoted

V1–V4 all shared the same fatal premise: one averaged model, one token-CE loss, one big mashed-together dataset. Every iteration hit the same ceiling because:

1. **Averaging kills style.** Training on 14k maps from thousands of mappers produces output that satisfies no mapper. There's no "correct" forehand/backhand direction when averaged over conflicting conventions.
2. **Token CE is spatially local.** It optimizes per-token probability; it does not optimize the *motion* a player performs. Models find loopholes (diagonal spam, wall spam in v15) that lower CE while producing unplayable motion.
3. **8–15h iteration is too slow.** We cannot steer an architecture at that cadence. Every failed run is a week lost.

V5 fixes all three.

---

## The Three Bets

### Bet 1 — Trajectory/Flow output (physics-first objective)

Replace token-CE with a loss that *sees motion*. Represent a map as a continuous saber-trajectory field over time — or at minimum, optimize a direct playability objective (differentiable parity + follow-through + collision) alongside event emission. Diffusion or flow-matching head over a short temporal window.

Why this matters: the model will never generate flowing maps if the loss cannot distinguish flowing output from non-flowing output. It currently cannot.

### Bet 2 — Single-mapper cohorts (style-first data)

Train on one mapper's catalog at a time. Joetastic model. Rustic model. Emilia model. A player loading the "Joetastic" variant should feel Joetastic's choices: his phrasing, his accent placement, his comfort level with parity resets.

Why this matters: style is the product, not a nice-to-have. Averaged mappers will always lose to real ones. Committing to a style makes the model *something* rather than *nothing*.

### Bet 3 — Auto-researcher harness (fast iteration)

A loop that: (1) reads an experiment spec (cohort + hyperparams), (2) trains a small model for 30–60 min on a small cohort, (3) generates test maps, (4) runs automated playability EDA, (5) writes results to a leaderboard. Enables 10–20 experiments/day instead of 1/week.

Why this matters: Bets 1 and 2 are both bets — we don't know which mapper transfers best, which trajectory formulation converges, which bucket blend works. Without a harness we're guessing. With a harness we're iterating.

**Build order:** Bet 3 first. It makes 1 and 2 cheap.

---

## Cohort Structure (from `data/reference/mappers.json`)

**18 mappers, 9 style buckets.** See `data/reference/mappers.json` for the full schema.

| Bucket | Mappers | Use |
|--------|---------|-----|
| `anime_jpop_flow` | Joetastic, ETAN, Emilia, Alice, Nolanimations | Highest-volume style; try first |
| `acc_ranked_flowy` | cerret, Aquaflee, Skeelie, hexagonial, olaf, Alice | Ranked-clean reference for parity metrics |
| `fast_rock_metal` | rustic, BennyDaBeast, Emilia, fatbeanzoop | Physical-mapping test |
| `dance_pop_kpop` | Joetastic, ETAN, BennyDaBeast, ryger | Groove-style test |
| `tech_gimmick` | cerret, Skeelie, hexagonial, helloimdaan, oddloop, uninstaller, Aquaflee | Hard-mode — does model learn angle language? |
| `vibro_speed` | helloimdaan, oddloop, fatbeanzoop, uninstaller | Edge-case robustness |
| `cinematic_variety` | rustic, ETAN, Nolanimations, muffn | Musical-structure test |
| `meme_variety_experimental` | oddloop, ryger, muffn, uninstaller | Non-generic pattern exposure |
| `slow_lofi_chill` | olaf, ryger | Low-density restraint |

---

## First-Run Blockers (2026-04-14)

Everything required before a single spec from `experiments/queue/initial.yaml` can execute. ~6h of work, all non-blocking on the running download.

| # | Task | File(s) | Status |
|---|------|---------|--------|
| B1 | Cohort-aware preprocessing | `scripts/preprocess.py` — add `--cohort <slug>`, root swap | [x] |
| B2 | Dataset cohort filter | `src/beatsaber_automapper/data/dataset.py` — already scans `data_dir`; cohort/bucket routed via `data_dir` swap | [x] |
| B3 | Bucket manifest builder | `scripts/build_bucket_manifests.py` — hardlinked bucket dirs + combined splits/frame_index | [x] |
| B4 | Hydra wiring for cohort/bucket | `scripts/train.py` + `configs/train.yaml` — top-level `cohort=` / `bucket=` overrides `data_dir` | [x] |
| B5 | Cohort reference stats | `scripts/compute_cohort_reference.py` — mean NPS, direction histogram, parity baseline, color balance | [x] |
| B6 | `output_dir` override validation | Lightning uses `default_root_dir` = `output_dir`; runner `rglob`s for ckpt; **Hydra model-group override fixed** via `configs/model/sequence/` subdir | [x] |
| B7 | Shared onset ckpt wiring | `scripts/auto_research.py` default: `outputs/.../version_0/.../onset-epoch=05-val_f1=0.732.ckpt`. No per-cohort onset training. | [x] |

**All first-run blockers resolved.** Ready to execute initial queue once downloads complete + preprocessing runs.

## Polish Before Scaling (deferred, not blocking first runs)

| # | Task | Why |
|---|------|-----|
| P1 | Auto-detect test audio duration | Done in `runner.py::_audio_duration_sec` — `--test-duration-sec` is now optional | [x] |
| P2 | Seed reaches Lightning Trainer | `configs/train.yaml` exposes `seed`; `scripts/train.py` calls `seed_everything`; runner emits `seed={spec.seed}` | [x] |
| P3 | Per-cohort EDA dashboard | `scripts/cohort_eda.py` — renders all `reference.json` files as a sortable comparative table | [x] |
| P4 | Crash recovery — OOM vs other | Runner currently treats any nonzero rc as failure | [ ] |
| P5 | Composite-score weight calibration | Current weights (0.4/0.2/0.1/0.3) are guesses | [ ] |

## Architecture Changes Needed

| Phase | Change | Defer until |
|-------|--------|-------------|
| V5-1 MVP | **None.** Current `SequenceModel` works for cohort training at smaller size. | — |
| V5-2 | Optional `mapper_id` embedding (additive conditioning alongside difficulty/genre), ~30 lines in `sequence_model.py` | After harness shows which cohort is most learnable |
| V5-3 | Trajectory/flow-matching head (Bet 1) — new output + loss + decoder path | After V5-2 proves (or disproves) that token-CE can hit style transfer |

**Not changing:** `AudioEncoder`, `OnsetModel`, tokenizer, `beatmap.py`, postprocessing.

---

## Implementation Plan

### Phase V5-0: Cohort Data Infrastructure (days 1–3)

**Goal:** Every mapper's full catalog downloaded, preprocessed, and addressable as a cohort.

- [ ] **0.1** Validate all `beatsaver_id` values against BeatSaver API (`/users/id/{id}`). Mark mismatches in `mappers.json`.
- [ ] **0.2** Add `scripts/download_cohorts.py`: reads `mappers.json`, downloads each mapper's full catalog via `/maps/uploader/{id}/{page}` to `data/cohorts/{mapper_name}/raw/{map_id}.zip`. Rate-limit 5 req/s, exponential backoff on 429. Persistent manifest per cohort.
- [ ] **0.3** Extend `scripts/preprocess.py` with `--cohort <name>` flag. Output → `data/cohorts/{mapper_name}/processed/`.
- [ ] **0.4** Build bucket index: `data/cohorts/_buckets/{bucket_id}.json` listing member .pt files for all mappers in that bucket. Used by dataset filter.
- [ ] **0.5** Extend `BeatSaberDataset` with `cohort` / `bucket` filter. Reuse existing frame_index format.
- [ ] **0.6** Sanity EDA per cohort: notes/sec, parity violations, direction distribution, NPS histogram. Confirm each cohort is self-consistent before training.

**DoD:** `python scripts/download_cohorts.py` completes. `data/cohorts/joetastic/processed/` exists with >100 .pt files. A training batch can be built from a single cohort.

---

### Phase V5-1: Auto-Researcher Harness (days 3–7)

**Goal:** Run 10+ short experiments in an afternoon. Answer: which mappers are learnable? At what model size? With what loss?

- [ ] **1.1** `experiments/spec.yaml` schema: `{cohort, bucket, model_size, max_epochs, loss_weights, seed}`.
- [ ] **1.2** `scripts/auto_research.py`: reads a queue of specs, trains each for a capped wall-clock (30–60 min), generates a fixed test song, runs playability EDA, appends results to `experiments/leaderboard.jsonl`.
- [ ] **1.3** Reuse `evaluation/playability.py` (6 checks). Add: style-closeness (direction histogram KL vs cohort reference, NPS match, parity-violation-rate gap vs cohort baseline).
- [ ] **1.4** Small-model preset in `configs/model/sequence_small.yaml` (d_model=256, 4 layers). Target: <60 min training on single-mapper cohort.
- [ ] **1.5** Shared test song: `data/reference/so_tired_rock.mp3`. Every experiment generates against it so outputs are directly comparable.
- [ ] **1.6** `scripts/leaderboard.py`: renders `experiments/leaderboard.jsonl` as a table, sorted by composite score.

**DoD:** `python scripts/auto_research.py experiments/queue/initial.yaml` runs five experiments end-to-end and writes a ranked leaderboard.

---

### Phase V5-2: Single-Mapper Cohort Validation (days 7–10)

**Goal:** Prove style transfer. One generated map that a BS community member can identify as "{mapper}-style."

- [ ] **2.1** From V5-1 leaderboard, pick the top 3 mappers by style-closeness. Run full-size training on each (4–8 hours each).
- [ ] **2.2** Generate 3 test maps per mapper (different genres from test pool) → `data/generated/cohort_eval/`.
- [ ] **2.3** Blind human evaluation: share with community mapper if possible; at minimum, self-eval against the mapper's real catalog.
- [ ] **2.4** Document wins/losses per mapper in `docs/cohort_results.md`.

**DoD:** At least one mapper cohort produces output that is distinctly stylistic (not the averaged-mapper output v14 generated).

---

### Phase V5-3: Trajectory Output (weeks 2–4)

**Goal:** Loss function that sees motion, not just tokens.

Approaches to prototype in V5-1 harness before committing:

- [ ] **3.1** **Option A — Soft-trajectory auxiliary loss:** Keep token output. Add a differentiable saber-trajectory simulator (piecewise cubic) over the generated window. Loss = cosine-distance between predicted trajectory and ground-truth trajectory derived from the real map. Should reduce follow-through violations to near zero if trained long enough.
- [ ] **3.2** **Option B — Flow-matching head:** Replace token emission with a diffusion/flow-matching head predicting `(hand_pos, hand_vel, swing_intent)` at each onset frame, decoded back to v3 events post-hoc. Harder; higher ceiling.
- [ ] **3.3** **Option C — Hybrid:** Token output for event type + flow head for continuous params (direction, angle). Simpler than B; more signal than A.

Pick one based on harness results.

**DoD:** A training run produces maps with <5% parity violations *pre*-postprocessing (vs v14's 50%).

---

### Phase V5-4: Style Mixing / Bucket Conditioning (week 3+)

**Goal:** One model conditioned on `mapper_id` that can produce any of the cohorts on demand.

- [ ] **4.1** Mapper embedding (18-class) + bucket embedding (9-class) as conditioning inputs.
- [ ] **4.2** Train on all cohorts with conditioning dropout (CFG-style). Enables both single-mapper and blended output.
- [ ] **4.3** Test style-interpolation: "70% Joetastic, 30% rustic" via embedding blend.

---

## Explicitly Deprecated (Do Not Revisit)

| Thing | Why it's dead |
|-------|--------------|
| V4 v15 training run | Catastrophic output (1 note + 1572 walls) — rare-event CE reweighting + Expert-only shrinkage collapsed the model |
| "One model for all mappers" | Averaging produces nothing; committed to style-cohort approach |
| Token-CE as sole objective | Loss is not motion-aware; model finds loopholes. Replaced by trajectory loss in Bet 1 |
| Constrained decoding as a bandaid | Keep the code — still useful — but stop treating inference patches as fixes for a mis-trained model |

v14 checkpoint is preserved as potential warm-start for cohort fine-tuning. Not discarded.

---

## File Map (V5)

### To Create
| File | Purpose |
|------|---------|
| `data/reference/mappers.json` | Cohort source-of-truth (EXISTS) |
| `scripts/download_cohorts.py` | Per-mapper catalog downloader |
| `scripts/auto_research.py` | Experiment runner |
| `scripts/leaderboard.py` | Results viewer |
| `configs/model/sequence_small.yaml` | Fast-iteration model preset |
| `experiments/spec.yaml` | Experiment schema |
| `experiments/queue/initial.yaml` | First batch of specs |
| `experiments/leaderboard.jsonl` | Results log |
| `docs/cohort_results.md` | Per-mapper wins/losses |

### To Modify
| File | Change |
|------|--------|
| `src/beatsaber_automapper/data/dataset.py` | Add `cohort` / `bucket` filter |
| `scripts/preprocess.py` | Add `--cohort` flag |
| `src/beatsaber_automapper/evaluation/playability.py` | Add style-closeness metrics |

### To Reference (keep working)
| File | What it does |
|------|-------------|
| `models/audio_encoder.py` | Audio encoder — still valid |
| `models/onset_model.py` | Onset detection — still valid |
| `models/sequence_model.py` | Keep; train on cohorts |
| `generation/beam_search.py` | Constrained decoding — keep |
| `generation/postprocess.py` | Playability pass — keep |

---

## Commands (V5)

```bash
# Validate mappers.json against BeatSaver API
python scripts/download_cohorts.py --validate-only

# Download all cohorts (full catalogs)
python scripts/download_cohorts.py

# Preprocess a single cohort
python scripts/preprocess.py --cohort joetastic --workers 8

# Run a small experiment on one cohort
python scripts/train.py stage=sequence \
    data.cohort=joetastic \
    model.sequence.d_model=256 model.sequence.num_layers=4 \
    max_epochs=15 max_samples_per_epoch=50000

# Run the auto-researcher on a queue
python scripts/auto_research.py experiments/queue/initial.yaml

# View leaderboard
python scripts/leaderboard.py
```

---

## Success Criteria

V5 is working when:

1. **Infrastructure:** All 18 cohorts downloaded, preprocessed, individually trainable.
2. **Iteration speed:** 10+ experiments/day achievable.
3. **Style transfer:** At least one single-mapper model produces output a human can identify as that mapper's style.
4. **Motion quality:** Parity violation rate <5% pre-postprocess on generated maps (vs ~50% in v14).
5. **Portfolio-ready:** Demo shows side-by-side generation in 3 distinct mapper styles from the same input song.
