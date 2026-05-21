# V7 Architecture Plan: MERT + Demucs + PhraseIndex

**Status:** Implementation complete 2026-05-15. Preprocessing running (~505/5320 songs). Training pending.
**Supersedes:** `docs/architecture_v6_plan.md`

### Implementation status at a glance

| Component | Status |
|-----------|--------|
| `data/mert_encoder.py` | ✅ Done |
| `data/stem_separator.py` | ✅ Done |
| `scripts/preprocess_v7.py` | ✅ Done / 🔄 Running (~4.5h remaining) |
| `data/beat_grid.py` | ✅ Done |
| `data/beat_dataset.py` | ✅ Done |
| `models/beat_classifier.py` | ✅ Done |
| `training/beat_module.py` | ✅ Done |
| `scripts/train_beats.py` | ✅ Done — awaiting preprocessing |
| `data/layout_dataset.py` | ✅ Done |
| `models/layout_model.py` | ✅ Done |
| `training/layout_module.py` | ✅ Done |
| `scripts/train_layout.py` | ✅ Done — awaiting Stage 1 |
| `generation/phrase_index.py` | ✅ Done |
| `generation/generate.py::generate_v7_level` | ✅ Done — awaiting checkpoints |
| `scripts/generate.py --v7` | ✅ Done |
| Stage 1 trained checkpoint | ⏳ Pending |
| Stage 2 trained checkpoint | ⏳ Pending |
| End-to-end generation test | ⏳ Pending |

---

## Problem Statement

V6 failed because it conflated two separate problems:

- **Problem 1 — WHEN:** At beat X, does a note exist?
- **Problem 2 — WHAT:** Given a note at beat X, what are its spatial attributes?

The Δt autoregressive token handled both, but cross-entropy loss on Δt provides no
audio-aligned gradient for timing. The model learned the marginal Δt distribution from
training data, not audio-to-beat mapping. Additionally, a 3-second context window
cannot maintain cross-song consistency — the same musical phrase at bar 8 and bar 40
produced inconsistent note patterns.

V7 separates Problem 1 and Problem 2, adds a pretrained audio model that already
understands music, and adds an explicit phrase memory to enforce consistency.

---

## New Dependencies

```bash
uv pip install demucs                  # source separation
uv pip install transformers            # MERT + HuggingFace ecosystem
```

**Demucs** (`htdemucs` variant): separates audio into 4 stems — drums, bass, other
(melody), vocals. Drum stem provides the cleanest onset signal; drums are nearly 1:1
with Beat Saber note placement by design.

**MERT-v1-95M** (`m-a-p/MERT-v1-95M` on HuggingFace): pretrained music encoder,
75 Hz frame rate, 768-dim embeddings per frame. Trained via masked acoustic modeling
on large music corpora; benchmarked at ~0.94 AUC on beat tracking. Used **frozen** —
fine-tuning risks forgetting on our small dataset (5320 songs).

---

## Data Flow

### Preprocessing (offline, run once on full dataset)

```
audio.mp3
  │
  ▼ Demucs htdemucs (GPU, ~30s/song)
  │
  ├── drums.wav   ─▶ MERT-95M ─▶ drum_mert [T_frames, 768] at 75 Hz
  │                              │
  │                              ▼ pool to 1/4-note beat grid
  │                              drum_beat [T_beats, 768]
  │
  └── other.wav   ─▶ MERT-95M ─▶ mix_mert [T_frames, 768]
  (melody stem)                  │
                                 ▼ pool to 1/4-note beat grid
                                 mix_beat [T_beats, 768]
                                 │
                                 ▼ mean over 4-bar windows
                                 phrase_fingerprints [N_phrases, 768]

All stored in existing .pt files as new keys (non-destructive).
```

**Beat-grid pooling:**
At BPM B, one 1/4-note slot = `(60 / B / 4)` seconds = `(60 / B / 4) × 75` MERT frames.
Pool (mean) MERT frames within each slot. Example at 123 BPM: 1/4-note = 0.122s × 75 =
~9 frames per beat slot. Clean, well-supported resolution.

**Phrase fingerprints:**
4 bars = 16 beats. Mean-pool `mix_beat[bar_start:bar_end]` → one 768-dim vector.
This is the key for the PhraseIndex cosine lookup at inference time.

---

### Training

#### Stage 1: Beat Classifier

```
drum_beat[t]  [768]
     │
     ▼ Linear(768 → 256)
     │
     ▼ ±4-beat local self-attention (2 layers)
       captures "there was a hit 1 beat ago → likely another hit now"
     │
     ▼ Linear(256 → 2)
     │
  [left_logit, right_logit]  per beat slot
     │
  Sigmoid → P(left note at t), P(right note at t)
     │
  Loss: weighted BCE
    pos_weight = (total_negative_slots / total_positive_slots)
    typically ~8:1 ratio for Expert maps
```

Ground truth labels derived from existing `swing_tokens` via `extract_beat_labels()`:
snap each note to its nearest 1/4-note slot, set the corresponding binary label.

#### Stage 2: Layout Generator

```
Input at each onset (beat t, hand h):

  local_mert   = mix_beat[t]                              [768]
  song_emb     = mean(mix_beat[0:T])                      [768]
  section_emb  = mean(mix_beat[section_start:section_end])[768]
  saber_state  = compute_saber_states(...)[h]              [12]
  retrieval    = prior_pattern from PhraseIndex           [K, 5 tokens]
                 (zeros + mask if no prior match)

  conditioning = Linear([local_mert; song_emb; section_emb]) → [d_model]
               + Linear(saber_state) → [d_model]

Output token sequence (causal autoregressive):
  [KIND] → [X] → [Y] → [DIR] → [FIELD_D]   (NOTE: 5 tokens)
  [KIND] → [X] → [Y] → [SQUISH]            (CHAIN_TAIL: 4 tokens)
  [KIND] → [X] → [Y]                        (BOMB: 3 tokens)

  No HAND token (given by Stage 1).
  No Δt token (given by Stage 1).

Loss: CE over spatial tokens, ignore_index=PAD.
```

Retrieval conditioning via cross-attention: the decoder cross-attends to
`retrieval_tokens` (embedded), gated by `retrieval_sim` as attention bias.
At hard-retrieval threshold (sim > 0.85), the retrieved tokens are prepended
as a "hint prefix" — the model sees what was done before and tends to copy it.

---

### Inference

```python
def generate_v7_level(audio_path, ...):
    # 1. Separate audio
    stems = demucs.separate(audio_path)          # drums, other
    
    # 2. Extract MERT features
    drum_beat = mert_encode(stems["drums"])       # [T_beats, 768]
    mix_beat  = mert_encode(stems["other"])       # [T_beats, 768]
    
    # 3. Stage 1: onset schedule
    left_probs, right_probs = beat_classifier(drum_beat)  # [T_beats, 2]
    left_onsets  = threshold(left_probs,  threshold_L)     # list[beat_slot]
    right_onsets = threshold(right_probs, threshold_R)     # list[beat_slot]
    
    # 4. Build phrase memory
    phrase_index = PhraseIndex()
    phrase_index.build(mix_beat, phrase_boundaries)
    
    # 5. Generate layout for each 4-bar window
    all_events = []
    for window_start in phrase_boundaries:
        window_end = window_start + 16  # beats
        
        # Query phrase memory
        fingerprint = mix_beat[window_start:window_end].mean(0)
        prior_pattern = phrase_index.query(fingerprint, threshold=0.85)
        
        # Generate notes for this window
        window_events = []
        for beat_slot in range(window_start, window_end):
            for hand, onsets in [(LEFT, left_onsets), (RIGHT, right_onsets)]:
                if beat_slot not in onsets:
                    continue
                
                if prior_pattern and beat_slot in prior_pattern:
                    # Hard retrieval: use stored pattern
                    event = prior_pattern[beat_slot][hand]
                else:
                    # Generate with soft conditioning
                    context = build_context(mix_beat, beat_slot, song_emb,
                                            section_emb, saber_state,
                                            retrieval=prior_pattern)
                    event = layout_model.generate(context)
                
                window_events.append(event)
        
        # Record pattern for future matching
        phrase_index.record(fingerprint, window_events)
        all_events.extend(window_events)
    
    # 6. Convert events to beatmap
    beatmap = events_to_beatmap(all_events)
    return postprocess_and_export(beatmap, ...)
```

---

## Model Architectures (as built)

### BeatClassifier (`models/beat_classifier.py`)

```
Input:   [B, W, 768]  drum MERT features, beat-grid aligned
          W = window_size (128 slots = 32 beats)

Proj:    Linear(768 → d_model=256)
Pos:     Learned positional embedding [max_len, 256]
Attn:    2 × TransformerEncoderLayer(d=256, heads=4, ffn=1024, norm_first=True)
         Full-window attention (not masked) — the full window is 32 beats,
         which is fine to attend over; local-window mask not needed at this scale
Norm:    LayerNorm(256)
Head:    Linear(256 → 2)    [left_logit, right_logit] per slot

Output:  [B, W, 2]  → sigmoid → P(left note), P(right note) per beat slot

Loss:    Weighted BCE  pos_weight=6.0  (~85% negative / 15% positive for Expert)
Params:  ~1M total
```

### LayoutModel (`models/layout_model.py`)

```
Input conditioning (per note onset):
  local_mert   [B, 768]  mix MERT at this beat slot
  song_emb     [B, 768]  mean mix MERT over full song
  section_emb  [B, 768]  mean mix MERT over current section
  saber_state  [B, 12]   physical saber state from prior events
  phrase_feat  [B, 768]  phrase fingerprint (which 4-bar window)
  difficulty   [B]       int index
  genre        [B]       int index

Conditioning pathway:
  mert_proj:   Linear(768*3 → d_model=512)   concat of 3 MERT levels
  saber_proj:  Linear(12 → 512)
  phrase_proj: Linear(768 → 512)
  diff_emb:    Embedding(5, 128)
  genre_emb:   Embedding(11, 128)
  cond_fuse:   Linear(512+256 → 512)         final conditioning vector [B, 1, 512]
               (used as memory for cross-attention)

Decoder:
  token_emb:   Embedding(vocab_size=118, 512)
  pos_emb:     Embedding(max_len=64, 512)
  4 × TransformerDecoderLayer(d=512, heads=8, ffn=2048, norm_first=True)
               cross-attends to conditioning memory [B, 1, 512]
  norm:        LayerNorm(512)
  output:      Linear(512 → 118)

Output:  [B, S, 118]  logits over full vocab (spatial grammar constrains valid tokens)
Loss:    CE, ignore_index=PAD, label_smoothing=0.1

Token grammar (no HAND, no Δt):
  NOTE:        [KIND=38] [X] [Y] [DIR] [ANGLE]   = 5 tokens
  ARC_HEAD:    [KIND=39] [X] [Y] [DIR] [MU]       = 5 tokens
  CHAIN_HEAD:  [KIND=41] [X] [Y] [DIR] [SLICE]    = 5 tokens
  CHAIN_TAIL:  [KIND=42] [X] [Y] [SQUISH]         = 4 tokens
  BOMB:        [KIND=43] [X] [Y]                   = 3 tokens
```

### PhraseIndex (`generation/phrase_index.py`)

```python
@dataclass
class NotePattern:
    window_start_slot: int
    tokens_by_position: dict[tuple[int, int], list[int]]
    # key: (relative_slot, hand_id) → spatial token list

class PhraseIndex:
    _fingerprints: list[Tensor]    # [768] per pre-indexed phrase
    _patterns:     list[NotePattern | None]  # None until record() is called
    _boundaries:   list[tuple[int, int]]     # (start_slot, end_slot) per phrase
    patterns:     list[NotePattern]
    
    def build(self, mix_beat, boundaries): ...   # compute fingerprints
    def query(self, emb, threshold) -> Optional[NotePattern]: ...  # cosine search
    def record(self, emb, pattern): ...          # add to memory
```

No external library needed — cosine similarity over a list of at most ~50 phrase
fingerprints (a 3-minute song at 4-bar windows = ~12 phrases). O(N) search is fine.

---

## File Map (as implemented 2026-05-15)

### New Files ✅ All written and import-tested

| File | Purpose |
|------|---------|
| `scripts/v7_poc.py` | PoC: validated MERT beat signal before build |
| `scripts/preprocess_v7.py` | Demucs + MERT preprocessing for all songs |
| `data/mert_encoder.py` | MERT-v1-95M wrapper: extract + beat-grid pool + fingerprints |
| `data/stem_separator.py` | Demucs htdemucs wrapper (GPU, LRU-cached) |
| `data/beat_grid.py` | `extract_beat_labels()` + `beat_labels_from_pt()` |
| `data/beat_dataset.py` | `BeatDataset` — drum features + binary labels |
| `data/layout_dataset.py` | `LayoutDataset` — per-onset MERT context + spatial tokens |
| `models/beat_classifier.py` | Stage 1: local attention on drum MERT → [left, right] logits |
| `models/layout_model.py` | Stage 2: causal transformer, MERT-conditioned, no Δt/HAND |
| `training/beat_module.py` | Lightning: weighted BCE, F1/P/R via torchmetrics |
| `training/layout_module.py` | Lightning: CE over spatial tokens, val token accuracy |
| `generation/phrase_index.py` | PhraseIndex + NotePattern — cosine similarity phrase memory |
| `scripts/train_beats.py` | Stage 1 training script |
| `scripts/train_layout.py` | Stage 2 training script |
| `docs/architecture_v7_plan.md` | This file |

### Modified Files ✅ Done

| File | Change |
|------|--------|
| `generation/generate.py` | Added `generate_v7_level()` and `_decode_spatial_tokens()` |
| `scripts/generate.py` | Added `--v7`, `--beat-ckpt`, `--layout-ckpt`, `--beat-threshold`, `--phrase-similarity` |
| `pyproject.toml` | Added `demucs>=4.0`, `transformers>=4.40` |

### Kept Unchanged

| File | Reason |
|------|--------|
| `data/swing_tokenizer.py` | Grammar survives in Stage 2 spatial token subset |
| `data/saber_state.py` | Saber state conditioning carried into Stage 2 |
| `generation/postprocess.py` | Unchanged |
| `generation/lighting_rules.py` | Unchanged |
| `generation/export.py` | Unchanged |
| `generation/chroma.py` | Unchanged |
| `evaluation/` | Unchanged |

### Archived (file exists, not used in V7 pipeline)

| File | Replacement |
|------|-------------|
| `models/audio_encoder.py` | `data/mert_encoder.py` + MERT-v1-95M |
| `training/seq_module.py` | `training/beat_module.py` + `training/layout_module.py` |
| `data/dataset.py::SwingSequenceDataset` | `data/beat_dataset.py` + `data/layout_dataset.py` |
| `generation/generate.py::generate_swing_level` | `generate_v7_level()` |
| `generation/beam_search_v6.py` | No beam search in V7; grammar handled by hard token masks |

---

## Preprocessing: Actual Performance (RTX 5090)

| Metric | Value |
|--------|-------|
| Model warm-up (Demucs + MERT load) | ~6s |
| Throughput per song | ~4.5s average (scales with song duration) |
| 5320 songs total | ~6.5h one-time cost |
| Run started | 2026-05-15 18:27 local |
| ETA | ~2026-05-15 23:00 local |
| Storage added per song | ~1.2 MB (fp16 beat features × 2 stems) |

---

## Staging / Iteration Plan (actual)

**2026-05-15 — V7-0 through V7-7:** All code implemented in one session.
PoC validated (F1=0.59 before any task-specific training). Preprocessing running.

**Next: Run Stage 1 training** as soon as preprocessing completes:
```bash
python scripts/train_beats.py --max-epochs 20
```
Expected: < 1h, F1 ≥ 0.80.

**Then: Run Stage 2 training:**
```bash
python scripts/train_layout.py --max-epochs 30
```
Expected: 1–2h, val_token_acc ≥ 0.85.

**Then: First end-to-end generation + V7-8 tuning.**
Target: NPS ≥ 3.0 (anything below Expert minimum) as a floor; push for ≥ 5.0.

**Day 7+ — V7-8:** Tune thresholds, evaluate on diverse songs, human review in ArcViewer.

---

## Risk Register

| Risk | Likelihood | Mitigation |
|------|------------|------------|
| MERT F1 < 0.70 on onset detection | Low — it's designed for MIR | Try later MERT layers; try finer grid (1/8-note) |
| Demucs drum separation poor on specific genres | Medium — drums bleed at low volume | Fall back to full-mix MERT for beat classifier if drum F1 < full-mix F1 |
| PhraseIndex threshold too tight: no matches | Low — can tune | Lower threshold to 0.70 |
| PhraseIndex threshold too loose: everything copies | Low — can tune | Raise to 0.90 or add minimum gap |
| Stage 2 ignores retrieval conditioning | Medium | Raise retrieval conditioning weight; try hard-copy prefix instead of cross-attention |
| Layout model drifts to few note positions | Low — explicit spatial CE | Add diversity regularization if needed |
| Preprocessing 5320 songs takes > 8h | Low — GPU demucs is fast | Process subset for initial runs; full set overnight |
