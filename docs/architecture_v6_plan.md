# V6 Architecture Plan — Per-Hand Swing Events + Saber Proprioception + Phrasing

**Date:** 2026-05-11
**Status:** Active. Supersedes V5's modeling axis. Keeps V5's data axis (cohorts) and iteration axis (auto-researcher).
**North star (unchanged from V5):** a player plays a generated map and says *"who mapped this?"* — not *"is this AI?"*

---

## Why V5 Was Not Enough

V5 fixed two of three axes:

| Axis | V5 status | Why it wasn't enough |
|------|-----------|---------------------|
| **Data** — single-mapper cohorts | Fixed: 18 cohorts, 9 buckets, full catalogs downloadable per mapper | Necessary but not sufficient. CE on cohort data still optimizes token-match, not style. |
| **Iteration** — auto-researcher | Fixed: harness, leaderboard, fixed test song | Necessary; will be re-used as-is in V6. |
| **Modeling** — output representation, conditioning, loss | **Untouched.** Same chord-at-timestamp tokens, same mean-pooled context, same local aux losses. | **The bottleneck.** |

The V4 → V5 transition assumed token-CE on cohort data could clear the "feel" bar with enough style-specific training. After thorough re-review by Opus 4.7 on 2026-05-10, we believe that assumption is wrong. The output representation is wrong for the underlying object.

---

## The Three Blindspots V6 Fixes

### Blindspot 1 — Chord-at-timestamp tokenization hides physics

A Beat Saber map is **not** a sequence of chords. It is **two interleaved hand trajectories** — left saber alternates F/B, right saber alternates F/B, and they meet/separate to form phrasing. Color is not an attribute of a note; color **is** the hand.

The current tokenizer emits per-onset chord-tokens (`NOTE COLOR COL ROW DIR ANGLE`). Parity, follow-through, and intra-onset alternation are *emergent statistical regularities* the model must re-discover from data, signalled only via auxiliary losses that whisper at it through softmax probabilities on the direction position.

**Every aux loss in `seq_module.py` exists to teach the model physics it should never have had to learn.**

### Blindspot 2 — No body / no proprioception

`prev_context_k=8` previous onsets get **mean-pooled** into a single vector per onset. That destroys ordering, grid position, and direction state. The model has no idea where its sabers physically are.

A real mapper holds a tiny continuous state: `(L_pos, L_dir, L_t, R_pos, R_dir, R_t)`. That's 12 floats. We pass none of it.

### Blindspot 3 — Loss is local; mapping is phrasing

CE + parity + follow-through are **all local**. They optimize per-token and per-pair correctness. They never ask "does this 4-bar window feel like the song's 4-bar window?" or "does this match the way Joetastic accents a snare hit?"

The only phrase-level conditioning is `section_id` (6 classes) and `section_progress` (0–1 scalar). That is not nearly enough phrase structure for the model to build a phrasing prior.

---

## The Three Bets of V6

### Bet 1 — Swing-event tokenization (representation shift)

Replace the per-onset chord grammar with a **per-hand swing-event stream**. Each event is one of:

```
SwingEvent := [HAND] [Δt_bin] [KIND] [GRID_X] [GRID_Y] [DIR] [ANGLE]
```

- `HAND`: `LEFT` (red), `RIGHT` (blue), or `NONE` (bomb / wall — see below).
- `Δt_bin`: time since previous swing in song-relative beats, quantized.
- `KIND`: `NOTE`, `ARC_HEAD`, `ARC_TAIL`, `CHAIN_HEAD`, `CHAIN_TAIL`, `BOMB`, `WALL`.
- `GRID_X`, `GRID_Y`, `DIR`, `ANGLE`: per-event positional/directional fields.

Walls and bombs ride the same timeline with `HAND=NONE`, so the global event stream remains totally ordered.

**What this unlocks:**

1. **Parity becomes structural.** The model sees alternating-by-construction same-hand events. Per-hand parity is now a property of the data the model trains on — no aux loss needed.
2. **Follow-through becomes a clean geometric loss** between consecutive same-hand swings, not a noisy softmax penalty against a mean-pooled blob.
3. **Chain/arc tails self-connect.** A chain tail is just "the next LEFT swing's position must equal this CHAIN_HEAD's tail_position." The model emits both ends naturally.
4. **Vocabulary shrinks.** ~70 tokens vs. 183, with the saved capacity moved into Δt resolution and angle resolution where it matters.
5. **Most of `seq_module.py`'s aux-loss code goes away.** `_compute_flow_loss`, `_compute_intra_onset_parity_loss`, `_compute_follow_through_loss`, `_compute_ergo_loss` are all bandaids on the chord representation. They get **deleted**, not migrated.

### Bet 2 — Saber-state conditioning (proprioception)

At every decoder step, additionally condition on the current 12-dim saber state:

```
saber_state := (
    L_x, L_y,           # left hand last grid position
    L_dir_x, L_dir_y,   # left hand last swing unit vector
    L_dt_since,         # beats since left swing (clamped/log)
    L_parity,           # +1 forehand / -1 backhand / 0 neutral
    R_x, R_y,
    R_dir_x, R_dir_y,
    R_dt_since,
    R_parity,
)
```

Projected via `Linear(12 → d_model)`, added to the decoder input alongside difficulty/genre. Replaces (or augments) the mean-pooled `prev_context_k` blob.

At training: derived from ground-truth past swings, recomputed per step.
At inference: maintained incrementally during AR decoding.

This is the missing "body" the model has been pretending it could infer from token soup. It is the literal physical state the player inhabits.

### Bet 3 — Phrase conditioning + style discriminator (global signal)

**Phrase conditioning.** At each decode step, also pass a **phrase embedding** computed from a 16-bar audio window around the current time (mean-pooled audio encoder output, projected to `d_model`, additive). This gives the model a "what's happening over the next 8 seconds" signal that mappers actively use.

**Phrase-energy auxiliary loss.** Compute predicted swing density per 4-bar window and compare against audio RMS curve per 4-bar window (KL divergence). Cheap, end-to-end differentiable, attacks the "feel" problem directly.

**Style discriminator.** Train a small classifier `D(audio_window, swing_window) → mapper_id` on real cohort data. Once it reaches reasonable F1 (>0.6 on held-out swings), add `−λ · log p_D(this_mapper | generated_swing_window)` as an auxiliary loss during sequence training. This is a poor-man's GAN that optimizes *style-closeness* directly — exactly the metric we already track in the auto-researcher leaderboard.

---

## What Survives Untouched From V5

| Component | V6 status |
|-----------|-----------|
| `data/cohorts/{mapper}/` structure | Unchanged. Already correct. |
| `scripts/download_cohorts.py` | Unchanged. |
| `scripts/auto_research.py` harness | Unchanged interface; reads same spec YAML. |
| `experiments/leaderboard.jsonl` | Unchanged; V5 and V6 runs are directly comparable via composite score. |
| `data/reference/mappers.json` | Unchanged. |
| `models/audio_encoder.py` | Unchanged. |
| `models/onset_model.py` (Stage 1) | Unchanged. Onset detection is a separate, working subproblem. |
| `generation/lighting_rules.py` (Stage 3) | Unchanged. Rule-based lighting is good enough. |
| `generation/chroma.py` | Unchanged. |
| `evaluation/playability.py` heuristics | Unchanged as evaluation; the *training* aux losses are deleted. |

---

## What Changes

| Component | V6 change |
|-----------|----------|
| `data/tokenizer.py` (Stage 2 grammar) | **Deprecated** for sequence model. Kept for legacy evaluation only. Replaced by `data/swing_tokenizer.py`. |
| `data/dataset.py` (Stage 2 dataset) | Emits per-hand swing token streams, saber-state tensor per step, phrase embeddings, mapper_id. |
| `models/sequence_model.py` | New vocabulary, saber-state projection, phrase-embedding projection. AR decode now interleaved-by-hand. |
| `training/seq_module.py` | Removes `_compute_flow_loss`, `_compute_intra_onset_parity_loss`, `_compute_follow_through_loss`, `_compute_ergo_loss`. Adds phrase-energy KL loss and style-discriminator loss. |
| `generation/beam_search.py` | Adapts grammar mask to swing events; saber-state maintained during beam expansion. |
| `generation/postprocess.py` | Most fixers (`fix_parity`, `convert_dot_notes`, `_choose_flow_direction`) become **no-ops** under V6. Keep `enforce_max_notes_per_beat`, `enforce_nps`, `enforce_color_separation`, wall sanity — the structural rules that don't overlap with the new model's natural output. |
| `models/onset_planner.py` | Becomes optional; saber-state largely subsumes its role. Keep wiring but expect to disable. |

---

## Phase Plan

Build order is intentionally **representation-first**. Bets 2 and 3 are cheap once Bet 1 is in place; doing them in any other order means re-doing them after the tokenizer change.

### Phase V6-0 — Spec and Round-Trip (1 day)

**Goal:** Lock the swing-event grammar with proof of zero information loss.

- [ ] **0.1** Write `docs/swing_event_grammar.md`: full token table, ordering rules, walls/bombs handling, chain/arc tail matching policy.
- [ ] **0.2** Implement `data/swing_tokenizer.py::SwingEventTokenizer` with `encode_beatmap` and `decode_beatmap`.
- [ ] **0.3** Round-trip test: for every map in the Joetastic cohort, `decode(encode(beatmap)) == beatmap` (within Δt and angle quantization tolerance). Target ≥99.5% maps round-trip cleanly.
- [ ] **0.4** Vocabulary audit: confirm vocab size, Δt bin resolution, ANGLE bin resolution. Lock these constants.

**DoD:** 99.5%+ round-trip success on Joetastic. New tokenizer module + tests committed.

### Phase V6-1 — Saber State Extractor (1 day)

**Goal:** Compute the 12-dim saber state at every swing event in a ground-truth map.

- [ ] **1.1** `data/saber_state.py::compute_saber_states(swing_events) -> Tensor[N, 12]`. Pure-function; deterministic; handles parity reset on long gaps (>3 beats).
- [ ] **1.2** Property tests: state at swing N depends only on swings 0..N-1; long-gap reset works; left/right are independent.
- [ ] **1.3** Sanity histogram: dump saber-state distributions for the Joetastic cohort to confirm reasonable spread (no degenerate states).

**DoD:** Saber state computed for every Joetastic map; distributions look healthy.

### Phase V6-2 — Dataset Migration (2 days)

**Goal:** The Stage 2 dataset emits V6-shape batches.

- [ ] **2.1** Add `SwingSequenceDataset` (or a `format=swing` flag on existing dataset). Emits `tokens`, `saber_state`, `mapper_id` (for cohort training), `phrase_window_offset` per sample.
- [ ] **2.2** Update `collate_fn` for variable-length swing streams.
- [ ] **2.3** Re-preprocess Joetastic cohort under the new format → `data/cohorts/joetastic/processed_v6/`. Keep old `processed/` for V5 fallback.
- [ ] **2.4** Smoke test: a batch from `SwingSequenceDataset` round-trips through `SwingEventTokenizer.decode` back to a valid `DifficultyBeatmap`.

**DoD:** Joetastic V6-preprocessed; a DataLoader iter produces correctly-shaped batches; round-trip passes.

### Phase V6-3 — Model Rewiring (2 days)

**Goal:** `SequenceModel` accepts swing-event vocabulary + saber state + phrase embedding.

- [ ] **3.1** Add `saber_state_proj = Linear(12, d_model)`; add to decoder input alongside difficulty/genre.
- [ ] **3.2** Add `phrase_proj = Linear(d_model, d_model)`; phrase embedding pooled from 16-bar audio window around current time, additive at every decode position.
- [ ] **3.3** New vocab size (~70) wired through model + tokenizer config.
- [ ] **3.4** **Delete** `_compute_flow_loss`, `_compute_intra_onset_parity_loss`, `_compute_follow_through_loss`, `_compute_ergo_loss` from `seq_module.py`. Delete their alphas from configs. Delete the structured-prediction code paths that depended on the chord grammar.
- [ ] **3.5** Update `decode_step_cached` to consume the new vocabulary and the saber-state input (saber state is recomputed per step at inference, passed in by the beam-search loop).
- [ ] **3.6** Configs: `configs/model/sequence/sequence_swing_small.yaml` (d_model=256, 4 layers, V6 vocab); `sequence_swing_full.yaml` (d_model=512, 8 layers).

**DoD:** Model trains for one epoch on Joetastic V6 data without errors. CE val_loss is **measurable** (we don't have a baseline yet; this is just a smoke target).

### Phase V6-4 — Phrase-Energy Auxiliary Loss (1 day)

**Goal:** Loss term that scores phrase-level density agreement with the audio energy curve.

- [ ] **4.1** During training, for each batch, compute predicted swing rate per 4-bar window (sum of swing-emission probabilities). Compare to ground-truth audio RMS per 4-bar window via KL divergence.
- [ ] **4.2** Add `phrase_energy_alpha` hyperparam (default 0.1) to `seq_module.py`.
- [ ] **4.3** Log `train_phrase_energy_loss` to TB / wandb. Verify it actually decreases during training.

**DoD:** Loss is differentiable, decreases on Joetastic training run, doesn't NaN.

### Phase V6-5 — Style Discriminator (2 days)

**Goal:** A learned style-closeness signal to use as auxiliary loss during sequence training.

- [ ] **5.1** `training/style_discriminator.py`: small transformer encoder (`d_model=128`, 2 layers) over `(audio_window_emb, swing_window_tokens) → mapper_id`. Trained on all 18 cohorts simultaneously.
- [ ] **5.2** Pretrain to F1 ≥ 0.6 on held-out swing windows. Save checkpoint.
- [ ] **5.3** Plug as auxiliary loss into `seq_module.py`: `−λ · log p_D(this_mapper | generated_window)`. Stop-gradient through D; D is frozen during sequence training.
- [ ] **5.4** Add `style_disc_alpha` hyperparam (default 0.2). Calibrate so its gradient magnitude is comparable to CE.

**DoD:** Discriminator-augmented sequence training run completes; style-closeness composite score on leaderboard improves vs. CE-only baseline.

### Phase V6-6 — Inference + Postprocess Cleanup (1 day)

**Goal:** Generate a playable .zip end-to-end under V6.

- [ ] **6.1** Update `generation/beam_search.py` to swing-event grammar. Saber state is recomputed each step and passed in alongside cached decoder state.
- [ ] **6.2** Sweep `generation/postprocess.py`: remove fixers whose job the model now does natively (`fix_parity`, `convert_dot_notes`, `_choose_flow_direction`). Keep structural rules (NPS cap, color separation, wall sanity).
- [ ] **6.3** End-to-end: `bsa-generate so_tired_rock.mp3 --difficulty Expert --cohort joetastic`. Check it loads in ArcViewer.

**DoD:** Generated .zip plays without errors in ArcViewer; pre-postproc parity violation rate < 5% (vs ~50% in V4); zero impossible follow-throughs in inspection.

### Phase V6-7 — Harness Re-validation (1 day)

**Goal:** The V5 auto-researcher works on V6 models.

- [ ] **7.1** Update `scripts/auto_research.py` to accept V6 model presets in the spec YAML.
- [ ] **7.2** New queue: `experiments/queue/v6_pilot.yaml` — top 3 V5 cohorts (Joetastic, Rustic, Helloimdaan) at small-model preset, 90 min each.
- [ ] **7.3** Run overnight. Compare V6 leaderboard rows to V5 rows on the same cohorts.

**DoD:** Leaderboard shows V6 single-mapper composite scores strictly higher than V5's, on at least 2 of 3 cohorts.

### Phase V6-8 — Deep Training + Human Eval (1–2 weeks)

**Goal:** Pick winners, train them deep, get human verdicts.

- [ ] **8.1** From V6-7 leaderboard, pick top-2 cohorts. Run full-size (d_model=512, 8 layers) training for 6–10h each.
- [ ] **8.2** Generate 3 test maps per winning cohort across 3 different test songs.
- [ ] **8.3** Self-eval against the real catalog (side-by-side ArcViewer). Document where V6 succeeds vs. fails in `docs/v6_results.md`.
- [ ] **8.4** If possible: share blind samples with a community mapper for the "who mapped this?" test.

**DoD:** At least one cohort produces output a human can identify as that mapper's style. Documented wins/losses.

---

## Success Criteria

V6 is working when:

1. **Round-trip fidelity:** Swing-event tokenizer round-trips ≥99.5% of cohort maps without data loss.
2. **Structural correctness (pre-postproc):** parity violations < 5%, zero impossible follow-throughs, dot-direction usage ≤ cohort baseline ± 5%.
3. **Style transfer:** at least one cohort's V6 output is identifiable as that mapper's style by a human listener.
4. **Iteration speed:** harness still hits ≥ 10 experiments / overnight on V6 small-model preset.
5. **Loss-stack simplicity:** seq_module.py is smaller after V6 than before V5, not larger. (If we're adding bandaids again we're doing it wrong.)

---

## Explicitly Deprecated In V6

| Thing | Why it's dead |
|-------|--------------|
| Chord-at-timestamp tokenization | Hides physics, generates aux-loss debt |
| `_compute_flow_loss` | Replaced by structural property of swing-event stream |
| `_compute_intra_onset_parity_loss` | Same as above |
| `_compute_follow_through_loss` | Replaced by geometric loss between consecutive same-hand swings (if we even need that; structural alternation may suffice) |
| `_compute_ergo_loss` | Color-side preference is now structural: HAND tokens are first-class, not derived from a COLOR attribute |
| Mean-pooled `prev_context_k` blob | Replaced by saber-state vector + per-hand swing window |
| Diagonal-biased `_choose_flow_direction` postproc | Model emits direction directly; no postproc rewrite needed |
| `fix_parity` rewrite pass | Same |
| `convert_dot_notes` rewrite pass | Same |
| `flow_loss_alpha`, `follow_through_alpha`, `intra_onset_parity_alpha`, `ergo_loss_alpha` config knobs | Loss terms removed |

---

## Risk Register

| Risk | Mitigation |
|------|------------|
| Swing-event grammar can't represent some legal v3 construction | Round-trip test on full Joetastic catalog before committing. If <99.5%, identify the gap and extend the grammar before proceeding. |
| Δt quantization loses musically-meaningful timing | Pick bin resolution from histogram of real inter-swing intervals. Default 1/16-beat resolution + log-spaced for the long tail. |
| Saber state is too low-dimensional to capture intent | Backstop: add an extra `swing_velocity_estimate` field (3 floats). Or pass the last K=4 same-hand swings as an additional input. |
| Style discriminator collapses or fails to learn | Discriminator is gated behind a separate phase (V6-5). If it doesn't reach F1≥0.6, ship V6 without it. Phrase-energy aux loss remains. |
| V6 model is *worse* than V5 on cohorts | The harness will surface this within 90-min experiments. If true, V6 leaderboard rows will be lower than V5; we hold and debug before deeper runs. |
| Removing aux losses regresses parity | Structural alternation guarantees per-hand parity. If empirically violated, that means the swing-event encoder is buggy. Fix the encoder, not by re-adding the loss. |

---

## References

- V3 architecture analysis: `docs/archive/architecture_v3_analysis.md`
- V4 architecture analysis: `docs/archive/architecture_v4_analysis.md`
- V5 plan: `TODO.md` (until 2026-05-11) — preserved in git history pre-V6 commit.
- Triggering Opus 4.7 review: see PROGRESS.md entry dated 2026-05-10.
