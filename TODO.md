# Beat Saber Automapper — V7 Plan (MERT + Demucs + Retrieval Architecture)

**Last updated:** 2026-05-20
**Status:** V7-1 preprocessing complete. V7-3 Run 1 (F1=0.422) and Run 2 (F1=0.442) both failed — Run 2's pos_weight + mix-stem fixes barely moved the metric and peaked at epoch 0. Audit (2026-05-20) identified that the model has no difficulty conditioning and the eval is exact-slot F1 (no MIR-standard tolerance window). Run 3 plan: add difficulty embedding + tolerance-window F1, keep Expert+ExpertPlus dataset.
**North star:** A player plays a generated map and says *"who mapped this?"* — not *"is this AI?"*

**Full implementation plan:** [`docs/architecture_v7_plan.md`](docs/architecture_v7_plan.md)
**V6 post-mortem:** [`PROGRESS.md`](PROGRESS.md) — "V6 Post-Mortem" section

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

**DoD:** `val_f1_avg_tol` ≥ 0.80 (with ±1-slot tolerance). Exact-slot F1 is a secondary diagnostic.

### V7-4/5 — Stage 2: Layout Dataset + Generator ✅ CODE DONE / ⏳ AWAITING STAGE 1 (2026-05-15)
- [x] `data/layout_dataset.py` — per-onset samples: local/section/song MERT + saber state + spatial tokens
- [x] `models/layout_model.py` — causal transformer decoder, MERT conditioning at 3 levels, no Δt/HAND
- [x] `training/layout_module.py` — CE loss over spatial tokens, val token accuracy metric
- [x] `scripts/train_layout.py` — training script
- [x] **Bug fix (2026-05-19): saber-state off-by-one** in `LayoutDataset.__getitem__`. Was
      `compute_saber_states(all_events[:evt_idx])[-1]` — state BEFORE the *previous* event,
      not the current one. Now uses `compute_saber_states(all_events)[evt_idx]` directly.
- [x] **Perf fix (2026-05-19): O(n²) decode** — `decode_events` + `compute_saber_states` were
      called per-`__getitem__`. Now cached per (song, difficulty) in `_evt_cache`.
- [ ] **Run training** (blocked on Stage 1 checkpoint)

#### Known architectural limitation (audit, 2026-05-19)

Stage 2 currently generates each onset's 5-token spatial sequence **independently**, with
the only cross-note state being the 12-dim saber state (which only carries the LAST event
per hand). This caps musical "flow" between adjacent notes within a phrase. PhraseIndex
hard-retrieval mitigates this for repeated phrases, but novel phrases get note-level
independence. Candidate fixes (deferred — schedule after first end-to-end run):

1. Pass a short window of recent spatial events (last K=4-8) as additional cross-attn memory
2. Switch to phrase-level autoregression: the decoder unrolls all notes in a 4-bar window in
   one teacher-forced pass instead of one note at a time

For tonight: train Stage 2 as-designed; the multi-tier MERT conditioning already gives it
song/section/phrase context that V6 lacked. Re-evaluate flow after first ArcViewer review.

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

### V7-7 — End-to-End Inference ✅ CODE DONE / ⏳ AWAITING TRAINED MODELS (2026-05-15)
- [x] `generation/generate.py::generate_v7_level()` — full Demucs→MERT→Stage1→PhraseIndex→Stage2 pipeline
- [x] `_decode_spatial_tokens()` helper — spatial token list → `_SwingEvent`
- [x] `scripts/generate.py --v7` — CLI flag wired; requires `--beat-ckpt` and `--layout-ckpt`
- [ ] **End-to-end test run** (blocked on trained checkpoints)

**DoD pending:** NPS ≥ 3.0. Run after both checkpoints exist:
```bash
python scripts/generate.py "data/test_songs/SO TIRED ROCK - NUEKI.mp3" \
  --v7 --beat-ckpt <ckpt> --layout-ckpt <ckpt> \
  --difficulty Expert --genre rock --run-tag v7_first
```

### V7-8 — Evaluation + Tuning ⏳ NOT STARTED
- [ ] Generate on test song; check NPS and ArcViewer
- [ ] Tune Stage 1 threshold (start at 0.4, sweep 0.3–0.6)
- [ ] Tune PhraseIndex similarity threshold (start at 0.85)
- [ ] If repetitive: lower threshold to 0.80 or 0.75
- [ ] If drifting: raise threshold to 0.90
- [ ] Compare V6 vs V7 NPS on same test songs

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
