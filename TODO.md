# Beat Saber Automapper — V7 Plan (MERT + Demucs + Retrieval Architecture)

**Last updated:** 2026-05-22
**Status:** V7 end-to-end done (6.0 NPS). ArcViewer review found 3 inference bugs (see V7-8). Bug 1 (role alignment in generate_phrase._step) is critical and must be fixed before eval is meaningful.
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
