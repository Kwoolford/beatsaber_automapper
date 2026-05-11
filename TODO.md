# Beat Saber Automapper — V6 Plan (Swing-First Architecture)

**Last updated:** 2026-05-11
**Status:** V5 (cohort + harness) infrastructure preserved. V5 modeling axis DEPRECATED. Committing to V6 — per-hand swing-event tokenization + saber-state proprioception + phrase-aware loss.
**North star:** A player plays a generated map and says *"who mapped this?"* — not *"is this AI?"*

**Full rationale and architecture analysis:** [`docs/architecture_v6_plan.md`](docs/architecture_v6_plan.md)

---

## Why We Pivoted (Again)

V5 fixed two of three axes:
- **Data axis:** single-mapper cohorts → kept. Already correct.
- **Iteration axis:** auto-researcher harness → kept. Already correct.
- **Modeling axis:** chord-token CE + bandaid aux losses → **wrong representation.**

A thorough re-review by Opus 4.7 on 2026-05-10 identified three blindspots in the V5 modeling stack:

1. **Output representation hides physics.** A map is two interleaved hand trajectories, not a sequence of chords. The chord-at-timestamp tokenizer forces the model to re-learn saber kinematics through statistical regularity, with the aux losses (`flow`, `intra_onset_parity`, `follow_through`, `ergo`) acting as bandaids on the representation.
2. **No body / no proprioception.** `prev_context_k=8` previous onsets mean-pooled to a single vector destroys ordering and grid information. The model has no idea where its sabers physically are.
3. **Loss is local; mapping is phrasing.** CE + parity / follow-through optimize per-token correctness. There is no signal that says "this 4-bar window should feel like the song's 4-bar window" or "this is a Joetastic-style accent."

V6 fixes all three with three coordinated bets.

---

## The Three Bets

### Bet 1 — Per-hand swing-event tokenization

Replace the chord grammar with a single ordered stream of swing events, one per hand-cut:

```
SwingEvent := [HAND] [Δt_bin] [KIND] [GRID_X] [GRID_Y] [DIR] [ANGLE]
```

- `HAND`: LEFT (red) / RIGHT (blue) / NONE (bomb or wall)
- `KIND`: NOTE / ARC_HEAD / ARC_TAIL / CHAIN_HEAD / CHAIN_TAIL / BOMB / WALL

**Unlocks:** parity becomes structural (alternation enforced by data), follow-through becomes a clean geometric loss between consecutive same-hand events, chain/arc tails self-connect, vocab shrinks from 183 → ~70. **All current aux losses get deleted, not migrated.**

### Bet 2 — Saber-state conditioning

Pass an explicit 12-dim physical state at every decode step:
`(L_x, L_y, L_dx, L_dy, L_dt, L_parity, R_x, R_y, R_dx, R_dy, R_dt, R_parity)`. Computed from ground-truth past swings at training; maintained incrementally during AR decoding at inference. Projected via `Linear(12 → d_model)`, additive to decoder input. Replaces (or augments) the mean-pooled `prev_context_k` blob.

### Bet 3 — Phrase conditioning + style discriminator

- **Phrase embedding.** Mean-pool a 16-bar audio window around the current time, project to `d_model`, add as conditioning at each decode position.
- **Phrase-energy aux loss.** KL between predicted swing density per 4-bar window and audio RMS per 4-bar window.
- **Style discriminator.** Train `D(audio_window, swing_window) → mapper_id` on all 18 cohorts. Once F1 ≥ 0.6, add `−λ · log p_D(this_mapper)` as auxiliary loss in sequence training. Learned style-closeness signal.

---

## What Survives From V5 (no change required)

| Component | Status |
|-----------|--------|
| `data/cohorts/{mapper}/` directory structure | Unchanged |
| `scripts/download_cohorts.py` | Unchanged |
| `scripts/auto_research.py` | Spec format extended (V6 model presets); core loop unchanged |
| `experiments/leaderboard.jsonl` | Unchanged; V5 and V6 rows directly comparable on composite score |
| `data/reference/mappers.json` | Unchanged — same 18 mappers, 9 buckets |
| `models/audio_encoder.py` | Unchanged |
| `models/onset_model.py` (Stage 1) | Unchanged |
| `generation/lighting_rules.py` (Stage 3) | Unchanged |
| `evaluation/playability.py` (as evaluation, not training loss) | Unchanged |

---

## Phase Plan

Build order is **representation-first**. Bets 2 and 3 are cheap *after* Bet 1; doing them in any other order means redoing them after the tokenizer change.

### Phase V6-0 — Spec + round-trip (1 day)

- [ ] **0.1** Write `docs/swing_event_grammar.md`: token table, ordering rules, walls/bombs handling, chain/arc tail matching policy.
- [ ] **0.2** Implement `data/swing_tokenizer.py::SwingEventTokenizer` with `encode_beatmap` and `decode_beatmap`.
- [ ] **0.3** Round-trip test on full Joetastic catalog. Target ≥ 99.5% maps round-trip cleanly (within Δt + angle quantization).
- [ ] **0.4** Lock vocabulary constants: vocab size, Δt resolution, ANGLE resolution.

**DoD:** Round-trip test passes; new tokenizer + tests committed.

### Phase V6-1 — Saber state extractor (1 day)

- [ ] **1.1** `data/saber_state.py::compute_saber_states(swing_events) -> Tensor[N, 12]`. Pure, deterministic, parity-resets on >3-beat gaps.
- [ ] **1.2** Property tests: state at N depends only on swings 0..N−1; long-gap reset works; left/right independent.
- [ ] **1.3** Sanity histogram on Joetastic cohort — distributions look healthy.

**DoD:** Saber state computed for every Joetastic map; histograms look reasonable.

### Phase V6-2 — Dataset migration (2 days)

- [ ] **2.1** Add `SwingSequenceDataset` (or `format=swing` flag on existing). Emits `tokens`, `saber_state`, `mapper_id`, `phrase_window_offset`.
- [ ] **2.2** Update `collate_fn` for variable-length swing streams.
- [ ] **2.3** Re-preprocess Joetastic → `data/cohorts/joetastic/processed_v6/`. Keep V5 `processed/` for fallback.
- [ ] **2.4** Smoke test: batch → `SwingEventTokenizer.decode` → valid `DifficultyBeatmap`.

**DoD:** Joetastic V6-preprocessed; DataLoader produces correctly-shaped batches; round-trip from batch passes.

### Phase V6-3 — Model rewiring (2 days)

- [ ] **3.1** Add `saber_state_proj = Linear(12, d_model)`; additive to decoder input.
- [ ] **3.2** Add `phrase_proj = Linear(d_model, d_model)`; phrase embedding pooled from 16-bar audio window, additive per decode position.
- [ ] **3.3** New vocab size (~70) wired through model + tokenizer + configs.
- [ ] **3.4** **Delete** from `seq_module.py`: `_compute_flow_loss`, `_compute_intra_onset_parity_loss`, `_compute_follow_through_loss`, `_compute_ergo_loss`, plus their `*_alpha` hyperparams.
- [ ] **3.5** Update `decode_step_cached` to consume new vocab + saber-state input (recomputed per step at inference).
- [ ] **3.6** Configs: `configs/model/sequence/sequence_swing_small.yaml` (d_model=256, 4 layers) and `sequence_swing_full.yaml` (d_model=512, 8 layers).

**DoD:** Model trains 1 epoch on Joetastic V6 without errors; val_loss is measurable.

### Phase V6-4 — Phrase-energy auxiliary loss (1 day)

- [x] **4.1** Compute predicted swing rate per 4-bar window from emission probabilities.
- [x] **4.2** Compute ground-truth audio RMS per 4-bar window.
- [x] **4.3** KL divergence between the two; weight via `phrase_energy_alpha` (default 0.1).
- [x] **4.4** Log `train_phrase_energy_loss`; verify it actually decreases on a real run.

**DoD:** Loss is differentiable, decreases on Joetastic training, does not NaN. ✓ Implemented in `seq_module._compute_phrase_energy_loss`. Activated when `phrase_energy_alpha > 0` and `structure` is in batch.

### Phase V6-5 — Style discriminator (2 days)

- [ ] **5.1** `training/style_discriminator.py`: small transformer (`d_model=128`, 2 layers) over `(audio_window_emb, swing_window_tokens) → mapper_id` across all 18 cohorts.
- [ ] **5.2** Pretrain to F1 ≥ 0.6 on held-out swing windows. Checkpoint saved.
- [ ] **5.3** Plug into `seq_module.py` as `−λ · log p_D(this_mapper | generated_window)`. Frozen D, stop-gradient through D.
- [ ] **5.4** Calibrate `style_disc_alpha` (default 0.2) so its gradient magnitude is comparable to CE.

**DoD:** Discriminator-augmented sequence run completes; style-closeness composite on leaderboard improves vs CE-only baseline.

### Phase V6-6 — Inference + postprocess cleanup (1 day)

- [x] **6.1** V6 grammar-constrained nucleus sampler in `generation/beam_search_v6.py`. Grammar state machine enforces token grammar; saber state updated per event and passed to model.
- [x] **6.2** `generation/generate.py::generate_swing_level` — full V6 end-to-end pipeline (audio → swing-event stream → beatmap → postprocess → lighting → .zip). `postprocess_beatmap` no longer calls `fix_parity` or `convert_dot_notes`.
- [ ] **6.3** End-to-end: `bsa-generate so_tired_rock.mp3 --difficulty Expert --cohort joetastic`. Verify .zip loads in ArcViewer. (Requires trained checkpoint — blocked on data download.)

**DoD:** `test_generate_swing_level_creates_zip` passes ✓. Full ArcViewer test pending trained model.

### Phase V6-7 — Harness re-validation (1 day)

- [x] **7.1** `scripts/train.py` updated: `dataset_format=swing` flag switches between `SequenceDataset` (V5) and `SwingSequenceDataset` (V6). `collate_fn` plumbed through `create_dataloader`.
- [x] **7.2** `experiments/queue/v6_pilot.yaml` created: Joetastic / Rustic / Helloimdaan @ `sequence_swing_small` preset, 90 min each.
- [ ] **7.3** Run overnight. Compare V6 leaderboard rows to V5 rows on the same cohorts. (Requires data download + preprocessing.)

**DoD:** Queue file exists; train.py accepts `dataset_format=swing` ✓. Overnight run pending data.

### Phase V6-8 — Deep training + human eval (1–2 weeks)

- [ ] **8.1** From V6-7 leaderboard, pick top-2 cohorts. Full-size training (d_model=512, 8 layers), 6–10h each.
- [ ] **8.2** Generate 3 test maps per winning cohort across 3 different test songs.
- [ ] **8.3** Self-eval against the real catalog side-by-side in ArcViewer. Document wins/losses in `docs/v6_results.md`.
- [ ] **8.4** If possible: blind community-mapper review of generated maps.

**DoD:** At least one cohort produces output a human can identify as that mapper's style.

---

## Explicitly Deprecated (Do Not Revisit)

| Thing | Why it's dead |
|-------|--------------|
| Chord-at-timestamp tokenization (`data/tokenizer.py::BeatmapTokenizer`) | Hides physics, generates aux-loss debt. Replaced by `SwingEventTokenizer`. |
| `_compute_flow_loss` | Parity is structural under swing-events. |
| `_compute_intra_onset_parity_loss` | Same as above. |
| `_compute_follow_through_loss` | Replaced by geometric loss between consecutive same-hand swings (if needed). |
| `_compute_ergo_loss` | Color-side preference is now structural via HAND tokens. |
| Mean-pooled `prev_context_k` blob | Replaced by saber-state vector + per-hand swing window. |
| Diagonal-biased `_choose_flow_direction` postproc | Model emits direction directly. |
| `fix_parity`, `convert_dot_notes` rewrite passes | Same. |
| V5 overnight sweep (`experiments/queue/initial.yaml`) | Held; would have trained the wrong representation. Will re-run as `v6_pilot.yaml` after V6-6. |

The original `BeatmapTokenizer` stays in the repo as a legacy round-trip and evaluation aid, but is no longer used by Stage 2 training.

---

## File Map (V6)

### To Create
| File | Purpose |
|------|---------|
| `docs/architecture_v6_plan.md` | Full V6 rationale + phase plan (EXISTS) |
| `docs/swing_event_grammar.md` | Locked-down token table + ordering rules |
| `src/beatsaber_automapper/data/swing_tokenizer.py` | New per-hand swing-event tokenizer |
| `src/beatsaber_automapper/data/saber_state.py` | Saber-state extractor (12-dim per swing) |
| `src/beatsaber_automapper/models/phrase_encoder.py` | 16-bar phrase-window pooler |
| `src/beatsaber_automapper/training/style_discriminator.py` | Mapper classifier + pretraining loop |
| `configs/model/sequence/sequence_swing_small.yaml` | V6 small preset |
| `configs/model/sequence/sequence_swing_full.yaml` | V6 full preset |
| `experiments/queue/v6_pilot.yaml` | First V6 sweep (3 cohorts × 90 min) |
| `docs/v6_results.md` | Per-cohort wins/losses |

### To Modify
| File | Change |
|------|--------|
| `src/beatsaber_automapper/data/dataset.py` | Add swing-event format, emit saber-state + mapper_id |
| `src/beatsaber_automapper/models/sequence_model.py` | New vocab, saber-state projection, phrase projection |
| `src/beatsaber_automapper/training/seq_module.py` | **Delete** flow / parity / follow-through / ergo losses. Add phrase-energy + style-discriminator losses. |
| `src/beatsaber_automapper/generation/beam_search.py` | Swing-event grammar mask; saber-state maintained per step |
| `src/beatsaber_automapper/generation/postprocess.py` | Drop parity/dot/diagonal rewriters; keep structural rules |
| `scripts/preprocess.py` | `--format swing` flag (defaults to v6 going forward) |
| `scripts/auto_research.py` | Accept V6 model presets |
| `CLAUDE.md` | Architecture section updated to V6 (DONE) |
| `README.md` | ML pipeline diagram + conditioning table updated to V6 (DONE) |

### To Reference (keep working, no changes)
| File | What it does |
|------|-------------|
| `models/audio_encoder.py` | Shared audio encoder — still correct |
| `models/onset_model.py` | Stage 1 — still correct |
| `models/onset_planner.py` | Optional; saber-state subsumes most of its role. Keep wiring, expect to disable in V6 small preset. |
| `generation/lighting_rules.py` | Rule-based Stage 3 — still good enough |
| `generation/chroma.py` | Chroma palettes — still good |
| `evaluation/playability.py` | Still the evaluator (heuristic checks); not used as training loss |

---

## Commands (V6)

```bash
# Preprocess Joetastic cohort under V6 swing-event format
python scripts/preprocess.py --cohort joetastic --format swing --workers 8

# Train V6 small model on Joetastic
python scripts/train.py stage=sequence \
    model=sequence/sequence_swing_small \
    data.cohort=joetastic \
    seq_module.phrase_energy_alpha=0.1 \
    seq_module.style_disc_alpha=0.0 \
    max_epochs=30 max_samples_per_epoch=50000

# Run V6 pilot sweep
python scripts/auto_research.py experiments/queue/v6_pilot.yaml

# Generate end-to-end with V6 model
bsa-generate song.mp3 --difficulty Expert --cohort joetastic
```

---

## Success Criteria

V6 is working when:

1. **Round-trip fidelity:** swing-event tokenizer round-trips ≥ 99.5% of cohort maps.
2. **Structural correctness (pre-postproc):** parity violations < 5%; zero impossible follow-throughs; dot-direction usage ≤ cohort baseline ± 5%.
3. **Style transfer:** at least one cohort's V6 output is identifiable as that mapper's style by a human.
4. **Iteration speed:** harness still hits ≥ 10 experiments / overnight on V6 small preset.
5. **Loss-stack simplicity:** `seq_module.py` is *smaller* after V6 than before V5. (If aux-loss code is growing again we're doing it wrong.)

---

## Risk Register (summary; full version in `docs/architecture_v6_plan.md`)

| Risk | Mitigation |
|------|------------|
| Swing-event grammar can't represent some v3 construction | Round-trip test on full Joetastic catalog before committing. |
| Δt quantization loses musically-meaningful timing | Pick bin resolution from histogram of real inter-swing intervals. |
| Saber state too low-dim to capture intent | Add `swing_velocity_estimate` or last-K same-hand swings. |
| Style discriminator fails to learn | V6-5 is gated; ship V6 without it if F1 < 0.6. |
| V6 worse than V5 on cohorts | Harness surfaces this within 90 min. Hold and debug before deeper runs. |
| Removing aux losses regresses parity | Structural alternation guarantees parity. If violated, encoder is buggy — fix encoder, don't re-add loss. |
