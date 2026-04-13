# Beat Saber Automapper — Active TODO

**Last updated:** 2026-04-13
**Current focus:** Evaluate v14 output quality; tune if needed
**Detailed analysis:** `docs/architecture_v3_analysis.md`
**Historical progress:** `PROGRESS.md`

---

## Where We Are

### Latest Training Results

| Stage | Best Metric | Checkpoint | Status |
|-------|------------|------------|--------|
| Onset | val_f1 = **0.732** | `version_0/onset-epoch=05-val_f1=0.732.ckpt` | GOOD — keep |
| Sequence | val_loss = **1.090**, token_acc = **75.7%** | `version_14/sequence-epoch=13-val_loss=1.090.ckpt` | Phase 6 retrain — needs eval |
| Lighting | N/A | Rule-based (`generation/lighting_rules.py`) | Working |

Previous baseline: v6 hit val_loss=1.055 at epoch 55 on old data with no planner. v14 stopped at epoch 13 (patience=5) with planner + reprocessed data.

### What's Wrong with Generated Maps

The model has high token accuracy but generates **unplayable maps**:

| Issue | Severity | Detail |
|-------|----------|--------|
| Note clumping (4-6 notes/beat) | CRITICAL | 40+ beats have 4-6 simultaneous notes |
| Same-color multi-notes | CRITICAL | 32.6% of beats have 2+ same-color notes — unswingable |
| No swing flow/parity | CRITICAL | 301 parity violations in 823 notes (36.6%) |
| Grid position chaos | HIGH | Red notes on both sides, hand-crossing |
| All v3 features stripped | HIGH | Post-processing removes ALL arcs/chains/bombs (1,927 objects) |
| Direction 8 overuse | MEDIUM | 14.9% "any" direction — too many dot notes |

### Root Causes

1. **Token-level CE ≠ sequence quality**: 78.3% per-token accuracy → only 2.6% chance of correct 14-token sequence (two notes)
2. **Flow loss is non-functional**: `_compute_flow_loss()` uses `.detach()` on predictions — **zero gradient signal** to the model. The parity loss has never affected training.
3. **No constraints in generation**: Nothing prevents 6 notes at one beat, same-color duplicates, or grid collisions during inference
4. **Post-processing is destructive**: Strips all learned v3 features, randomly reassigns directions/positions
5. **Mean-pooling prev context**: Destroys structural information from previous onsets

---

## Implementation Plan

### Phase 1: Constrained Inference — IMPLEMENTED (2026-03-08)

**Goal:** Make current model produce playable maps by fixing inference and post-processing.

#### 1A. Grammar-Constrained Decoding (`generation/beam_search.py`) — DONE

Added `ConstraintState` class and constraint system to both nucleus sampling and beam search:

- **Grammar enforcement**: State machine tracks position in event token grammar (EVENT_TYPE → COLOR → COL → ROW → DIR → ANGLE → SEP/EOS). Only valid token ranges allowed at each position.
- **Max notes per beat**: Expert=2 (1/color), ExpertPlus=3 (2/color). Forces EOS after limit.
- **No same-color duplicates**: Tracks color counts, masks maxed-out colors.
- **Parity enforcement**: Tracks last direction per color across onsets. After forehand, masks forehand directions (allows backhand + neutral). After backhand, masks backhand directions.
- **No grid collisions**: Tracks (col, row) positions used, masks occupied rows for the current column.
- **Color separation bias**: Soft logit boost (+2) for preferred side (red→cols 0-1, blue→cols 2-3), penalty (-1) for wrong side.
- **Cross-onset parity**: `last_dir` persists across onsets via `generate_level()` tracking.

Key classes: `ConstraintState`, `init_constraints()`, `apply_constraints()`, `update_constraints()`

#### 1B. Fix Post-Processing (`generation/postprocess.py`) — DONE

- **Replaced `strip_non_note_objects()` with `cap_non_note_objects()`**: Keeps bombs, arcs, chains at reasonable densities instead of stripping all.
- **Added `enforce_max_notes_per_beat()`**: Caps notes per beat with ergonomic selection (prefers notes on preferred color side).
- **Added `enforce_color_separation()`**: Moves red notes from right cols to left, blue from left to right (collision-aware).
- **Improved `fix_parity()`**: Picks ergonomic direction based on grid position (left cols → diagonal-left, right cols → diagonal-right) instead of random.
- **Removed destructive steps**: `diversify_directions()`, `expand_grid_coverage()`, `deduplicate_patterns()` no longer called (constrained decoding handles these).

#### 1C. Temperature & Inference Tuning — DONE

- Temperature: 0.95 → **0.8** (less random)
- Top-p: 0.92 → **0.85** (tighter nucleus)
- Repetition penalty: 1.2 → **1.5** (more variety)
- Updated in `generate.py` defaults, CLI defaults, and `generate_level()` defaults

#### 1D. Quick Validation

After Phase 1 changes:
1. Generate Expert + ExpertPlus maps for `data/reference/so_tired_rock.mp3`
2. Run analysis script: count notes/beat, parity violations, color balance, grid coverage
3. Load in ArcViewer — visual inspection
4. Compare before/after snapshots

---

### Phase 2: Structured Prediction Model — MODELS IMPLEMENTED (2026-03-09)

**Goal:** Replace autoregressive token generation with multi-head direct prediction.

> For detailed architecture diagrams and rationale, see `docs/architecture_v3_analysis.md` § "Change 1"

#### 2A. New Model: `NotePredictor` (`models/note_predictor.py`) — DONE

Per-onset, predict a fixed-size output in ONE forward pass (not autoregressive):

```python
class NotePredictor(nn.Module):
    """Predicts note placement for a single onset.

    Input: audio_features [B, T, d_model] + difficulty_emb + plan_vector
    Output (per slot, up to max_notes=3):
        n_notes:    [B, 4]  — softmax over {0, 1, 2, 3}
        color:      [B, 3, 3]  — per-slot softmax {red, blue, none}
        col:        [B, 3, 4]  — per-slot softmax {0, 1, 2, 3}
        row:        [B, 3, 3]  — per-slot softmax {0, 1, 2}
        direction:  [B, 3, 9]  — per-slot softmax {0..8}
        angle:      [B, 3, 7]  — per-slot softmax {-45..+45}
        event_type: [B, 3, 4]  — per-slot softmax {note, bomb, arc_start, chain}
    """
```

Key design decisions:
- **3 slots max** — covers 99%+ of training data (only 4% have 3+ notes/onset)
- **Independent heads** — no cascading token errors
- **Constraint masks applied before softmax** — hard physical constraints
- Slots sorted by canonical order (left-to-right, bottom-to-top) for consistency

#### 2B. Multi-Task Loss — DONE (in `training/note_module.py`)

```python
loss = (
    w_n * CE(n_notes_pred, n_notes_target)
    + w_color * CE(color_pred, color_target)
    + w_pos * (CE(col_pred, col_target) + CE(row_pred, row_target))
    + w_dir * CE(dir_pred, dir_target)
    + w_angle * CE(angle_pred, angle_target)
    + w_type * CE(type_pred, type_target)
    + lambda_parity * parity_penalty(dir_pred, prev_dir)
    + lambda_ergo * ergonomic_penalty(col_pred, color_pred)
    + lambda_collision * collision_penalty(col_pred, row_pred)
)
```

Where:
- `parity_penalty`: **differentiable** soft parity check using direction logits (NOT detached argmax)
- `ergonomic_penalty`: penalize red notes in cols 2-3, blue notes in cols 0-1
- `collision_penalty`: penalize multiple slots predicting same (col, row)

#### 2C. Training Data Adapter — DONE (`tokenizer.tokens_to_structured()`)

Convert existing `token_sequences` to structured format:

```python
def tokens_to_structured(token_seq: list[int]) -> dict:
    """Convert [NOTE, COLOR, COL, ROW, DIR, ANGLE, SEP, ...] to structured dict.
    Returns: {n_notes: int, slots: [{color, col, row, dir, angle, type}, ...]}
    """
```

This runs at dataset load time (no reprocessing of .pt files needed).

#### 2D. Lightning Module (`training/note_module.py`) — DONE

- Same pattern as `SequenceLitModule` but with multi-head outputs
- Reuse existing `AudioEncoder` (proven working)
- Log per-attribute accuracy alongside total loss
- Validate with playability metrics (parity violations, collision rate)

---

### Phase 3: Bidirectional Onset Planner (1-2 weeks)

**Goal:** Give the model song-level context for planning note density and patterns.

> See `docs/architecture_v3_analysis.md` § "Change 2" for full details

#### 3A. Onset Planner Model (`models/onset_planner.py`)

```
Pass 1 (runs ONCE per song):
  Input: per-onset audio embeddings [N, d_model] (from audio encoder)
  Model: Bidirectional Transformer encoder (4-6 layers)
  Output: per-onset plan vectors [N, d_model]

  Learns:
  - Note density planning (more in chorus, less in verse)
  - Pattern repetition (similar audio → similar plans)
  - Build-up/drop dynamics
```

#### 3B. Song Structure Segmentation (`data/audio.py`)

Add `segment_song()` using librosa's self-similarity matrix:

```python
def segment_song(waveform, sr, bpm) -> list[Section]:
    """Returns: [(start_beat, end_beat, section_type, section_id), ...]
    section_type: intro=0, verse=1, chorus=2, bridge=3, drop=4, outro=5
    """
    # 1. Compute beat-synchronous chromagram
    # 2. Build self-similarity matrix (recurrence_matrix)
    # 3. Compute novelty curve from SSM diagonal
    # 4. Peak-pick section boundaries
    # 5. Cluster sections by similarity
```

Each onset gets: `section_type_emb + section_position (0.0-1.0) + section_energy`

#### 3C. Copy Mechanism for Repeated Sections

When the planner detects a section similar to a previously generated one:
- Cross-attend to the note patterns from the first occurrence
- Apply small perturbations (keeps consistency, adds variety)
- This is how human mappers work: "chorus 2 = chorus 1 with slight variation"

---

### Phase 4: Playability Reward (1 week)

**Goal:** Add a training signal that directly optimizes for playability.

> See `docs/architecture_v3_analysis.md` § "Change 4" for full details

#### 4A. PlayabilityScorer Model

Train a small classifier to distinguish good maps from bad:
- Positive: High-rated maps from BeatSaver (>90% upvote, ScoreSaber-ranked)
- Negative: Same maps with random perturbations (swap colors, flip directions, move positions)
- Architecture: Small Transformer or MLP over note sequence features
- Output: scalar 0-1 (0 = unplayable, 1 = perfect flow)

#### 4B. Differentiable Rule-Based Scoring (alternative to learned scorer)

```python
def playability_score(note_sequence, prev_notes):
    score = 1.0
    score -= parity_violation_rate(note_sequence)      # 0-1
    score -= collision_rate(note_sequence)              # 0-1
    score -= hand_crossing_rate(note_sequence)          # 0-1
    score -= 0.5 * direction_monotony(note_sequence)   # 0-1
    return max(0, score)
```

Use as auxiliary loss: `total_loss = main_loss + beta * (1 - playability_score)`

---

### Phase 5: Data Quality (ongoing, parallel to other phases)

#### 5A. Curated Training Set

Build a **gold standard** of 500-1000 maps:
1. Start with ScoreSaber-ranked maps
2. Filter to mappers with 10+ ranked maps
3. Run BS Parity Checker — keep maps with <5% violations
4. Run BS Map Check — keep maps with zero errors
5. Store as `data/processed/whitelist.json`

#### 5B. Data Augmentation (future)

Currently NO augmentation on tokens. Consider:
- Random canonical reordering (breaks left-to-right bias)
- Mirror augmentation (flip left ↔ right, swap red ↔ blue)
- Time stretch (adjust note density proportionally)

---

## Bug Fixes Needed

| Bug | File | Priority |
|-----|------|----------|
| ~~Flow loss is detached (no gradients)~~ | `training/seq_module.py` | ~~P0~~ FIXED — uses soft probs now |
| ~~`_build_token_weights` default eos_weight=0.3 confusing~~ | `training/seq_module.py` | ~~P2~~ FIXED — default aligned to 1.0 |
| ~~`playability.py` has NotImplementedError~~ | `evaluation/playability.py` | ~~P2~~ FIXED — implemented 6 checks |
| ~~Genre always "unknown" (num_genres=1)~~ | `configs/model/*.yaml` | ~~P3~~ WONTFIX — num_genres=1 works fine, plumbing kept for future |

---

## File Map (key files for V3 work)

### To Modify
| File | Change |
|------|--------|
| `generation/beam_search.py` | Add constrained decoding (Phase 1A) |
| `generation/postprocess.py` | Fix destructive post-processing (Phase 1B) |
| `generation/generate.py` | Wire constraints, tune temperature (Phase 1C) |
| `training/seq_module.py` | Fix detached flow loss (Bug fix) |
| `data/dataset.py` | Add structured prediction adapter (Phase 2C) |
| `data/audio.py` | Add `segment_song()` (Phase 3B) |

### Created (Phase 2)
| File | Purpose |
|------|---------|
| `models/note_predictor.py` | Structured prediction model — DONE |
| `training/note_module.py` | Lightning module for NotePredictor — DONE |
| `configs/model/note_pred.yaml` | Hydra config for note predictor — DONE |
| `evaluation/playability.py` | Playability checker (6 checks) — DONE |

### To Create (Phase 3+)
| File | Purpose |
|------|---------|
| `models/onset_planner.py` | Bidirectional onset planner (Phase 3A) |

### Reference (don't modify unless needed)
| File | What it does |
|------|-------------|
| `models/audio_encoder.py` | CNN + Transformer encoder — working, keep |
| `models/onset_model.py` | TCN + Transformer onset detection — working, keep |
| `models/sequence_model.py` | Current autoregressive model — will be superseded by NotePredictor |
| `data/tokenizer.py` | Token vocabulary — sound design, keep |
| `generation/lighting_rules.py` | Rule-based lighting — working, keep |
| `generation/chroma.py` | Chroma RGB post-processing — working, keep |

---

## Commands Reference

```bash
# Generate with current model
python scripts/generate.py data/reference/so_tired_rock.mp3 \
    --onset-ckpt outputs/beatsaber_automapper/version_0/checkpoints/onset-epoch=07-val_f1=0.726.ckpt \
    --seq-ckpt outputs/beatsaber_automapper/version_6/checkpoints/sequence-epoch=55-val_loss=1.055.ckpt \
    --difficulty Expert --output data/generated/test.zip

# Train note predictor (Phase 2 — structured prediction)
python scripts/train.py stage=note_pred data_dir=data/processed output_dir=outputs \
    max_epochs=80 data.dataset.batch_size=512 data.dataset.num_workers=8 \
    early_stopping_patience=20

# Train sequence model (Phase 1 — autoregressive, legacy)
python scripts/train.py stage=sequence data_dir=data/processed output_dir=outputs \
    max_epochs=100 data.dataset.batch_size=192 data.dataset.num_workers=8

# Run tests
.venv/Scripts/pytest tests/

# Lint
.venv/Scripts/ruff check .
```
