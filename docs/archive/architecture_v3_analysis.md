# Architecture V3 Analysis: Post-1-Week Training Deep Dive

**Date:** 2026-03-08
**Status:** Analysis complete, recommendations ready for implementation
**Context:** 1-week production training run completed (Feb 27 – Mar 3). Onset model val_f1=0.726 (epoch 7), sequence model val_loss=1.055, val_token_acc=78.3% (epoch 55/71). Generated maps have fundamental playability issues despite good training metrics.

---

## Table of Contents
1. [Executive Summary](#executive-summary)
2. [What Went Right](#what-went-right)
3. [Critical Issues Identified](#critical-issues-identified)
4. [Root Cause Analysis](#root-cause-analysis)
5. [Literature Review & State of the Art](#literature-review--state-of-the-art)
6. [Recommended Architecture V3](#recommended-architecture-v3)
7. [Implementation Roadmap](#implementation-roadmap)

---

## Executive Summary

The 1-week training run produced a model that has learned token-level patterns (78.3% accuracy) but generates **unplayable maps**. The core problem is a fundamental mismatch between what the model optimizes (next-token prediction) and what makes a good Beat Saber map (physical playability, musical flow, pattern coherence).

**Key symptoms:**
- **Note clumping**: 40+ beats with 4-6 simultaneous notes (including impossible 2x3 grids)
- **Same-color multi-notes**: 88 beats have 2+ notes of the same color — physically unswingable
- **No flow**: Notes jump randomly across the grid with no ergonomic hand movement
- **Excessive "any" direction (d=8)**: 14.9% of notes have no direction — lazy fallback
- **Post-processing as crutch**: 1,927 non-note objects stripped, 301 parity violations force-corrected. The model generates garbage; post-processing tries to make it playable.

**The core insight:** Token-level cross-entropy is a necessary but not sufficient training signal. The model can predict individual tokens well but cannot compose them into physically coherent sequences. We need **structural constraints built into the architecture** and **playability-aware training signals**.

---

## What Went Right

### 1. Onset Detection (Stage 1) — Solid Foundation
- **val_f1 = 0.726** is respectable for a custom model trained from scratch
- TCN + Transformer architecture converges reliably
- Adaptive thresholding and density curves produce reasonable onset density
- The gap distribution (Expert: mean 1.28 beats, stdev 1.54) shows rhythmic variety — NOT a metronomic stream anymore
- **Verdict: Keep as-is.** Minor tuning possible but not the bottleneck.

### 2. Training Infrastructure
- 71 epochs completed without crash over 5 days
- Epoch subsampling (1.5M/2.16M per epoch) worked perfectly
- KV caching for inference (3 min/song generation)
- Lightning + Hydra stack is solid
- Heartbeat monitoring, auto-resume all functioning

### 3. Audio Encoder
- CNN frontend + 6-layer Transformer encoder produces meaningful features
- Structure features (RMS, onset strength, spectral) are being injected
- Cross-attention from decoder to audio features is connecting (loss converges)
- **Verdict: Keep architecture. May need more CNN downsampling (see recommendations).**

### 4. Token Vocabulary Design
- Comprehensive coverage of v3 format (notes, bombs, walls, arcs, chains)
- Clean separation of event types and attributes
- SEP/EOS structure allows variable-length multi-note sequences
- **Verdict: Sound design. The vocabulary is not the problem.**

---

## Critical Issues Identified

### Issue 1: No Constraint on Notes-Per-Beat

**Symptom:** Beat 34.00 has 6 notes. Beat 257.62 has 6 notes in two adjacent columns. Beat 80.50 has 6 notes scattered across the grid.

**Why it happens:** The model generates an autoregressive sequence of tokens separated by SEP. Nothing prevents it from emitting 5 SEP tokens and creating 6 notes at one timestamp. In training data, 78% of onsets have only 1 note (len=7) and 18% have 2 notes (len=14), but during inference the model has no hard limit.

**Impact:** 2-note patterns are the standard in Beat Saber (one red, one blue, alternating sides). 4+ notes at once are reserved for rare emphasis moments in top-tier maps. The model treats every onset like a possible 6-note wall.

### Issue 2: Same-Color Multi-Notes at Same Beat

**Symptom:** 88 out of 270 beats (32.6%) have 2+ notes of the same color simultaneously. Example: beat 24.00 has 3 blue notes — physically impossible to swing a single blue saber at three positions.

**Why it happens:** The model generates tokens autoregressively and there's no constraint preventing it from emitting `NOTE blue ... SEP NOTE blue ...`. The training data does have same-color multi-notes (for stacks and sliders), but the model over-generates them because:
1. No penalty for same-color duplicates at the same beat
2. Beam search / nucleus sampling doesn't check physical constraints
3. The model has weak understanding of "one saber = one swing"

### Issue 3: No Swing Flow (Parity Awareness)

**Symptom:** Post-processing corrected 301 parity violations in a 823-note map (36.6% of notes). Red notes jump from down to down, up to up, with no alternating forehand/backhand.

**Why it happens:**
- The flow loss (`flow_loss_alpha=0.1`) is computed on **detached argmax predictions** — it's an auxiliary monitoring metric, NOT a gradient signal. It literally does not affect learning.
- Inter-onset context (prev_context_k=8) passes previous note embeddings, but the model hasn't learned to USE them for flow — there's no explicit incentive to alternate swing directions.
- At inference, each onset is generated independently by the sequence model. The previous 8 onsets' tokens are passed as context, but the model treats them as weak conditioning, not as hard constraints.

### Issue 4: Grid Position Chaos

**Symptom:** Red notes appear at (0,1) AND (3,2) on the same beat. Notes jump from column 0 to column 3 between consecutive beats. No concept of "left hand stays on left side, right hand stays on right side."

**Why it happens:**
- The model treats grid position (COL, ROW) as independent token predictions without understanding spatial ergonomics
- No loss signal for impossible hand-crossing patterns
- Training data has consistent left/right color separation (red = left columns, blue = right columns in most maps), but the model hasn't strongly learned this because it's an implicit pattern, not an explicit constraint

### Issue 5: Post-Processing is a Crutch, Not a Solution

**The current post-processing pipeline:**
1. `strip_non_note_objects` — removes ALL bombs, walls, arcs, chains (1,927 objects!)
2. `rebalance_colors` — randomly flips note colors
3. `diversify_directions` — randomly reassigns directions to match target distribution
4. `expand_grid_coverage` — randomly moves notes to unused cells
5. `remove_unplayable_patterns` — removes overlapping notes
6. `deduplicate_patterns` — injects random variation
7. `fix_parity` — randomly flips swing directions

**This is destructive.** It throws away all the model's learned patterns (arcs, chains, bombs — the v3 features that differentiate us from competitors) and replaces direction/position with random choices. The result looks statistically correct but plays like random noise.

### Issue 6: Audio Context Window Mismatch

**Current:** 256 frames (~3 seconds) centered on each onset.

**Problem:** The model sees a tiny window of audio for each note decision. It cannot learn:
- Phrase-level patterns (8-bar phrases are 16+ seconds)
- Song structure (verse vs. chorus should have different intensity)
- Build-ups and drops (require anticipating energy changes)
- Repetition (same riff = same mapping pattern)

The structure features (RMS, onset strength, spectral) help somewhat, but they're also windowed to 256 frames. The model has no concept of "this is the second time this riff plays" or "we're approaching the drop."

### Issue 7: Token-Level Loss vs. Sequence-Level Quality

**Fundamental problem:** Cross-entropy loss optimizes for predicting each token correctly in isolation. A model with 78.3% token accuracy still produces incoherent sequences because:
- 21.7% per-token error cascades across a 14-token sequence (2 notes)
- Probability of a perfect 14-token sequence: 0.783^14 = 2.6%
- The model is incentivized to predict the most common token at each position, not the most coherent sequence

This is the classic **exposure bias** problem in autoregressive models. The token dropout (10%) helps but doesn't solve it.

---

## Root Cause Analysis

### The Fundamental Architecture Gap

Our current approach treats Beat Saber map generation as **language modeling** — predict the next token given previous tokens and audio context. This works well for text because:
- Text has flexible word order (rearranging words usually preserves meaning)
- Grammar constraints are soft (grammatically imperfect text is still readable)
- There's no physical constraint on output

Beat Saber maps are **NOT** like text:
- **Spatial constraints are hard** — two notes in the same cell at the same time is invalid, period
- **Physical constraints are hard** — a human arm can only swing in alternating directions
- **Timing is tied to music** — notes must align with specific rhythmic features
- **Patterns must be ergonomic** — consecutive notes must form physically comfortable movements
- **Repetition is structural** — similar musical sections should have similar (but not identical) mappings

### What the Model Actually Learned (vs. What We Wanted)

| Aspect | What We Wanted | What the Model Learned |
|--------|---------------|----------------------|
| Token prediction | Correct note attributes | Most common attribute at each position |
| Multi-note patterns | Meaningful 2-note patterns (red left, blue right) | Random number of notes per beat |
| Flow | Alternating forehand/backhand | Random direction per note |
| Grid position | Ergonomic hand movement | Most common grid positions |
| Event types | Notes + bombs + arcs + chains | Mostly notes (with random arcs/chains stripped in post) |
| Musical structure | Different patterns for verse/chorus | Same approach regardless of section |

### Why 78.3% Token Accuracy is Misleading

Consider a typical 2-note sequence: `NOTE RED COL1 ROW0 DOWN ANG0 SEP NOTE BLUE COL2 ROW0 UP ANG0 EOS` (14 tokens)

If the model gets 11/14 tokens right (78.6%), the 3 errors could be:
- Wrong color on second note → both notes same color → unplayable
- Wrong direction → parity violation → uncomfortable swing
- Wrong grid position → notes in same cell → collision

One wrong token out of 14 can make the entire onset unplayable. **Token accuracy doesn't measure playability.**

---

## Literature Review & State of the Art

### Rhythm Game Map Generation

#### Mapperatorinator (osu!, 2024-2025)
The closest comparable work. Key architecture decisions:
- **Whisper encoder** for audio features (pretrained, finetuned)
- **Sparse decoder** — generates note objects directly, not token sequences
- **Constrained decoding** — hard constraints on valid note placement
- **Segment-based generation** — generates phrases, not individual notes
- Rhythm token weighting (3x on timing tokens) — we adopted this

#### BeatSage (2020)
- Two-stage: **timing network** (mel spectrogram windows → onset detection, no global BPM dependency) + **block type network** (assigns color + direction combos)
- Learns to recognize musical moments (drum fills, drops, vocal peaks) emergently from training
- **No autoregressive generation** — direct per-frame classification
- Very fast but low quality (no flow, no parity, limited pattern diversity)
- Demonstrates that direct prediction > autoregressive for note placement when model capacity is limited

#### InfernoSaber (2023-2024)
- **4-stage pipeline** on TensorFlow: Deep Convolutional Autoencoder → timing → note placement/direction → difficulty scaling
- Multiple pretrained variants on HuggingFace (e.g., `expert_15` trained on curated 8+ NPS maps with >90% upvote)
- **Per-frame prediction** of note attributes — no autoregressive loop
- Limited to v2 format (no arcs, chains, angle offsets)
- Quality is mediocre but consistent — never produces garbage patterns

#### DDC (Dance Dance Convolution, ICML 2017)
- Foundational rhythm game work
- CNN over mel spectrograms (80 bands, 23ms/46ms/92ms windows, 10ms stride) + conditional LSTM for step selection
- **Key insight:** Separating "when" from "what" is critical — confirmed by our good onset results

#### Beat-Aligned Spectrogram-to-Sequence (Yi et al., ISMIR 2023, arXiv 2311.13687)
- Formulates chart generation as **sequence generation with a Transformer**
- **Tempo-informed preprocessing** — aligns spectrogram frames to beats rather than fixed time intervals
- Benefits from pretraining on large dataset then finetuning on specific games
- **Applicable:** We could align our mel frames to beat positions rather than raw time

#### BeatLearning
- **BEaRT tokenization** — each 100ms time slice becomes a token encoding up to two note events
- Transformer with masked-language-modeling approach (BERT/GPT hybrid)
- **Game-agnostic** design supporting 1, 2, or 4 track games
- **Key insight:** Fixed time-slice tokenization avoids variable-length autoregressive generation

#### DeepSaber (oxai, 2020, historical)
- Multi-stream LSTM ingesting MFCCs, beat information, and partial chart context
- Introduced **action embeddings** (Word2Vec/FastText) for Beat Saber blocks — encoding similarity between placements as vector distances
- Framing: "beat maps are sentences; actions are words"

#### TaikoNation (FDG 2021, arXiv 2107.12506)
- **Patterning quality** is the primary identifier of high-quality rhythm game content in human rankings
- Produced charts with more congruent, human-like patterning than prior work
- **Key insight for us:** Focus on pattern quality metrics, not just token accuracy

#### Key Takeaway from Literature
The most successful approaches use **constrained direct prediction** rather than **unconstrained autoregressive generation** for note placement. Autoregressive models excel at generating variable-length sequences (text, music), but Beat Saber notes have a **fixed spatial structure** (4x3 grid, 9 directions, 2 colors) that's better served by direct prediction with hard constraints. Beat-aligned preprocessing and time-slice tokenization (BeatLearning, Yi et al.) are promising alternatives to our current per-onset autoregressive approach.

### Musical Structure Awareness

#### SING: Self-Similarity as Attention (arXiv 2406.15647, 2024)
- Uses a **user-supplied self-similarity matrix as an attention mask** over previous timesteps
- Architecture: LSTM + custom attention layer where weights come from the SSM
- Effective at replicating long-term structure over 700+ beats (~3 minutes)
- **Directly applicable:** Compute SSM of input song, use it to bias cross-attention so similar musical sections produce similar note patterns

#### Librosa Laplacian Segmentation (ready-to-use)
- `librosa.segment.recurrence_matrix()` builds sparse self-similarity matrix from beat-synchronous features
- Laplacian spectral decomposition + K-means clustering segments the song into structural sections
- **Low-effort, high-value:** Can be added to preprocessing pipeline immediately

#### Hierarchical Music Generation (ICLR 2024, Wang/Min/Xia)
- **Cascaded diffusion model** with 4 hierarchical levels: phrase structure → melody reduction → lead sheet → full accompaniment
- Each level conditions on its upper levels
- **Key insight:** Generate a structural plan first (onset density map, section labels), then condition note generation on that plan

#### Theme Transformer
- **Gated parallel attention** and **theme-aligned positional encoding** to reference thematic material
- Enables repetition of thematic materials with perceptible variations
- **Applicable:** When the song repeats a section, retrieve note patterns from first occurrence as a soft template via cross-attention

#### Music Transformer (ICLR 2019, Google Magenta)
- **Relative attention** explicitly modulates attention based on token distance
- Captures repetition at multiple timescales (motif, phrase, section)
- Generalizes beyond training sequence length
- **Applicable:** Replace sinusoidal positional encoding with relative positional encoding in our sequence model

#### Repetition-Aware Generation (REMI, Compound Word Transformer)
- Use self-attention with explicit repetition structure tokens
- **Applicable:** "This is beat 3 of pattern X in chorus 2" as conditioning

### Playability-Aware Training Signals

#### GAN/Discriminator Approaches
- **CESAGAN** (Conditional Embedding Self-Attention GAN): trains discriminator on real game levels, uses bootstrapping to add good generated levels back into training data
- **Rumi-GAN**: uses negative examples (bad levels) alongside positive to improve generator
- **ContraGAN**: 2C contrastive loss considers data-to-data and data-to-class relations
- **Applicable:** Train a discriminator on (audio_window, note_pattern) pairs from high-rated maps

#### RLHF-Style Reward Models
- Train on (prompt, chosen, rejected) triples to maximize score gap between good and bad
- BeatSaver's rating system provides implicit preferences (upvote ratio)
- **Applicable:** Collect paired map comparisons, train reward model on (audio_context, good_pattern, bad_pattern)

#### Rule-Based Reward Signals (Differentiable)
- Parity violation penalty (currently implemented but **detached from gradients**)
- Grid collision penalty
- Same-color same-beat penalty
- Hand-crossing penalty (red on right side, blue on left side)
- Swing distance penalty (consecutive notes too far apart)
- Pattern diversity loss: penalize repetitive patterns using self-similarity on generated sequences
- Onset density consistency loss: compare NPS curve against audio energy curve
- Structural repetition loss: when SSM indicates similar regions, penalize divergent patterns
- **These can be differentiable** using soft penalties instead of hard violations

---

## Recommended Architecture V3

### Core Philosophy Change

**FROM:** "Generate a token sequence for each onset independently"
**TO:** "Generate a placement grid for each onset with hard physical constraints"

### Change 1: Replace Autoregressive Token Generation with Structured Prediction

**Current:** Transformer decoder generates `NOTE COLOR COL ROW DIR ANGLE SEP NOTE...` autoregressively.

**Proposed:** **Multi-head structured prediction** — for each onset, predict a fixed-size output representing what happens at that beat:

```
Per-onset output (generated in one forward pass, NOT autoregressively):
┌─────────────────────────────────────────────────────┐
│ n_notes_head:     softmax over {0, 1, 2, 3}        │  ← How many notes
│                                                      │
│ For each slot (up to max 3 notes):                  │
│   color_head:     softmax over {red, blue, none}    │  ← Note color
│   col_head:       softmax over {0, 1, 2, 3}         │  ← Grid column
│   row_head:       softmax over {0, 1, 2}             │  ← Grid row
│   direction_head: softmax over {0..8}                │  ← Swing direction
│   angle_head:     softmax over {-45..+45}            │  ← Angle offset
│   type_head:      softmax over {note, bomb, arc, ..} │  ← Event type
│                                                      │
│ pattern_embedding: dense [d_model]                   │  ← Latent pattern
│                                                      │
│ Constraints applied as masks BEFORE softmax:         │
│   - slot[i].color must differ from slot[j].color     │
│   - slot[i].(col,row) must differ from slot[j]       │
│   - slot[0].direction must alternate from prev onset │
└─────────────────────────────────────────────────────┘
```

**Why this is better:**
1. **No cascading errors** — each attribute is predicted independently with its own head
2. **Hard constraints as masks** — physically impossible outputs are zeroed before softmax
3. **Fixed output size** — no variable-length generation, no EOS prediction issues
4. **Parallel prediction** — all slots predicted in one pass (fast inference)
5. **Loss on each attribute independently** — can weight spatial accuracy differently from direction accuracy

**Training:** Multi-task cross-entropy on each head, with:
- Hard constraint masking (collision avoidance, color separation)
- Parity-aware direction loss (alternating forehand/backhand as a hard constraint, not a soft penalty)
- Grid ergonomics loss (penalize notes far from expected hand position)

### Change 2: Sequence-Level Context via Bidirectional Onset Encoder

**Current:** Previous 8 onset tokens are mean-pooled and concatenated to cross-attention memory.

**Problem:** Mean-pooling destroys sequence information. The model can't distinguish "previous onset was a downswing red note at (1,0)" from "previous onset was an upswing blue note at (2,2)" because the pooling averages everything.

**Proposed:** Replace the onset-by-onset generation with a **two-pass architecture**:

```
Pass 1: Onset-level planning (bidirectional)
  Input:  [onset_1_audio, onset_2_audio, ..., onset_N_audio] (all onsets in the song)
  Model:  Bidirectional Transformer encoder over onset embeddings
  Output: [plan_1, plan_2, ..., plan_N] — per-onset planning vectors

  This sees the ENTIRE song structure and can plan:
  - Note density curves (more notes in chorus, fewer in verse)
  - Pattern repetition (same riff → similar pattern)
  - Build-up/drop dynamics

Pass 2: Per-onset note prediction (conditioned on plan)
  Input:  plan_i + audio_context_i + prev_note_embeddings
  Model:  Structured prediction heads (from Change 1)
  Output: Note placement for onset i
```

**Why this is better:**
1. **Global context** — the planner sees all onsets, not just the previous 8
2. **Repetition awareness** — bidirectional attention naturally links similar audio segments
3. **Density planning** — the planner can allocate "2 notes here, 1 note there" based on musical structure
4. **Fast inference** — Pass 1 runs once for the whole song, Pass 2 runs per onset (but is parallel-friendly)

### Change 3: Song Structure Segmentation as Preprocessing

**Current:** Structure features (RMS, onset strength, spectral) are computed per-frame and added to CNN output.

**Proposed:** Add explicit **song section detection** as a preprocessing step:

```python
def segment_song(audio, bpm):
    """Detect song structure: intro, verse, chorus, bridge, drop, outro.

    Uses:
    1. Self-similarity matrix (SSM) from chromagram
    2. Novelty curve from SSM diagonal
    3. Peak picking on novelty curve for section boundaries
    4. Clustering of sections by audio similarity

    Returns: List of (start_beat, end_beat, section_type, section_id)
    """
```

Each onset gets additional conditioning:
- `section_type` embedding (intro=0, verse=1, chorus=2, bridge=3, drop=4, outro=5)
- `section_id` (which instance of this section type: chorus_1, chorus_2)
- `section_position` (normalized position within the section: 0.0 = start, 1.0 = end)
- `section_energy` (average energy of this section relative to song mean)

**Copy mechanism:** When the planner (Change 2) encounters a section that's similar to a previous section, it can copy the planning vectors and apply small perturbations. This naturally produces "same riff = same pattern with variation."

### Change 4: Playability Reward Model

**Current:** Flow loss is detached from gradients (monitoring only). Post-processing randomly fixes violations.

**Proposed:** Train a **playability scoring model** and use it as an auxiliary training signal:

```python
class PlayabilityScorer(nn.Module):
    """Scores a sequence of notes for physical playability.

    Outputs: scalar 0-1 (0 = unplayable, 1 = perfect flow)

    Checks (differentiable):
    1. Parity: alternating forehand/backhand per color stream
    2. Grid distance: consecutive same-color notes within reachable distance
    3. Color separation: red tends left, blue tends right
    4. No collisions: notes don't overlap in space
    5. Swing comfort: direction change is physically natural
    6. Density appropriateness: note count matches difficulty
    """
```

**Training the scorer:**
1. Take 1000 high-rated maps from BeatSaver → label as "good" (score=1.0)
2. Apply random perturbations (swap colors, move positions, flip directions) → label as "bad" (score=0.0)
3. Train a small classifier to distinguish good from bad
4. Use as auxiliary loss during sequence model training: `total_loss = CE_loss + beta * (1 - playability_score)`

**Alternatively:** Hand-craft differentiable rules:

```python
def parity_penalty(note_sequence, prev_notes):
    """Differentiable parity violation penalty.

    Uses soft direction classification (forehand/backhand probability)
    instead of hard argmax, so gradients flow.
    """
    # Soft forehand probability from direction logits
    forehand_prob = dir_logits[:, [1, 6, 7]].sum(-1)  # down, down-left, down-right
    backhand_prob = dir_logits[:, [0, 4, 5]].sum(-1)  # up, up-left, up-right

    # Penalty: consecutive same-parity swings
    prev_forehand = ...  # from previous onset
    violation = forehand_prob * prev_forehand + backhand_prob * (1 - prev_forehand)
    return violation.mean()
```

### Change 5: Constrained Inference (Even Without Architecture Changes)

These can be implemented immediately as improvements to the current beam search:

#### 5a. Grammar-Constrained Decoding
```python
def constrained_decode(model, audio, difficulty, prev_notes):
    """Decode with hard constraints on valid note patterns."""
    # After generating each token, compute valid next tokens:
    if last_token_was(NOTE):
        # Must be followed by COLOR
        valid_next = COLOR_TOKENS
    elif last_token_was(COLOR):
        # Must be followed by COL
        valid_next = COL_TOKENS
    elif last_token_was(SEP):
        # Next note must have DIFFERENT color from previous note in this onset
        valid_next = EVENT_TOKENS  # but constrain color in next step
    # ... etc

    # Mask invalid tokens before softmax
    logits[~valid_next_mask] = -inf
```

#### 5b. Max Notes Per Beat
```python
MAX_NOTES_PER_BEAT = {
    "Easy": 1, "Normal": 1, "Hard": 2, "Expert": 2, "ExpertPlus": 3
}
# After generating MAX notes, force EOS
if n_notes_so_far >= MAX_NOTES_PER_BEAT[difficulty]:
    logits[SEP] = -inf  # prevent more notes
    logits[EOS] = logits.max()  # force EOS
```

#### 5c. Parity-Aware Direction Selection
```python
def constrain_direction(logits, prev_direction, color):
    """Force alternating forehand/backhand."""
    if prev_direction in FOREHAND_DIRS:
        # Must be backhand
        for d in FOREHAND_DIRS:
            logits[DIR_OFFSET + d] = -inf
    elif prev_direction in BACKHAND_DIRS:
        # Must be forehand
        for d in BACKHAND_DIRS:
            logits[DIR_OFFSET + d] = -inf
```

#### 5d. Spatial Constraint (No Collisions)
```python
def constrain_position(logits, occupied_positions):
    """Prevent notes in already-occupied grid cells."""
    for (col, row) in occupied_positions:
        logits[COL_OFFSET + col] = -inf  # simplified; real implementation is per-step
```

### Change 6: Data Quality — Curated Training Set

**Current:** 11,997 .pt files from BeatSaver with loose quality filters (>80% upvote, post-2020).

**Problem:** BeatSaver's rating system doesn't guarantee mapping quality. A map can have 90% upvotes but terrible flow, because most players don't notice parity violations — they just notice if a map is "fun enough."

**Proposed:** Create a **gold standard training set** of 500-1000 maps:
1. Start with ScoreSaber-ranked maps (verified competitive quality)
2. Filter to mappers with 10+ ranked maps (consistent quality)
3. Run BS Parity Checker on each map — only keep maps with <5% parity violations
4. Run BS Map Check — only keep maps with zero errors
5. Manual spot-check a random 10% sample

**Why smaller and better > larger and noisy:**
- 500 perfect maps teach better patterns than 12,000 mediocre maps
- The model currently learns bad habits from poorly-mapped training data
- Garbage in → garbage out, especially for spatial/flow patterns

### Change 7: Onset-to-Onset Recurrence (Keep Autoregressive But Fix It)

If we keep the autoregressive approach (easier migration path), fix the context mechanism:

**Current problems with prev_context_k=8:**
1. Mean-pooling previous onset tokens loses all structural information
2. The model has no explicit training signal to use the context
3. Context is added via concatenation to cross-attention memory — it competes with audio features for attention

**Proposed fixes:**
1. **Replace mean-pooling with per-attribute extraction:**
   ```python
   # Instead of mean-pooling the whole token sequence:
   prev_context = {
       'color': prev_notes.color,        # which colors were used
       'position': (prev_notes.col, prev_notes.row),  # grid positions
       'direction': prev_notes.direction,  # swing directions
       'n_notes': len(prev_notes),         # how many notes
   }
   # Encode as structured embedding, not mean-pooled tokens
   ```

2. **Explicit parity conditioning:**
   ```python
   # Pass the expected parity (forehand/backhand) as a conditioning input:
   expected_parity = "backhand" if prev_was_forehand else "forehand"
   parity_embedding = self.parity_emb(parity_idx)  # [d_model]
   ```

3. **Cross-attention to previous notes via separate attention heads:**
   Instead of concatenating prev context to audio memory, use dedicated cross-attention layers for note context vs. audio context.

---

## Comparison: Autoregressive Token vs. Structured Prediction

| Aspect | Current (Autoregressive Tokens) | Proposed (Structured Prediction) |
|--------|-------------------------------|--------------------------------|
| Output format | Variable-length token sequence | Fixed-size multi-head prediction |
| Cascading errors | Yes (21.7% per-token → 97.4% sequence failure) | No (each head independent) |
| Hard constraints | Difficult (must mask during generation) | Easy (mask before softmax per head) |
| Training signal | Single CE loss across all tokens | Per-attribute CE + playability reward |
| Inference speed | Sequential (one token at a time per onset) | Parallel (all attributes in one pass) |
| Variable notes/beat | Natural (SEP tokens) | Requires n_notes head + slot masking |
| Advanced events | Natural (arcs, chains as token subsequences) | Requires separate heads/modes |
| Migration effort | N/A (current) | High (new model architecture) |

### Recommendation: Hybrid Approach

1. **Short-term (1-2 days):** Implement constrained inference (Change 5) on the existing model. This immediately fixes the worst playability issues without retraining.

2. **Medium-term (1 week):** Implement structured prediction (Change 1) with onset-level planning (Change 2). Retrain on curated dataset (Change 6).

3. **Long-term (2-3 weeks):** Add playability reward model (Change 4) and song structure segmentation (Change 3).

---

## Implementation Roadmap

### Phase 1: Immediate Fixes (No Retraining) — 1-2 Days

**Goal:** Make the existing model produce playable maps by constraining inference.

1. **Grammar-constrained decoding** in `beam_search.py`:
   - Enforce valid token sequences (NOTE must be followed by COLOR, etc.)
   - Limit max notes per beat based on difficulty
   - Force alternating parity (forehand/backhand)
   - Prevent grid collisions within a beat

2. **Fix post-processing** in `postprocess.py`:
   - Stop stripping ALL non-note objects — keep reasonable bombs/arcs/chains
   - Replace random direction reassignment with parity-aware reassignment
   - Replace random grid movement with ergonomic grid movement (red=left, blue=right)

3. **Color separation constraint**: Red notes should prefer columns 0-1, blue should prefer columns 2-3. This single constraint eliminates hand-crossing.

**Expected impact:** Maps should go from "unplayable garbage" to "mediocre but correct" with zero retraining.

### Phase 2: Structured Prediction Model — 1 Week

**Goal:** Replace autoregressive token generation with a multi-head prediction model.

1. **New model: `NotePredictor`**
   ```
   Input: audio_features [B, T, d_model] + difficulty_emb + plan_vector
   Output:
     - n_notes: [B, 4] (softmax over 0-3 notes)
     - Per slot (×3):
       - color: [B, 3] (red, blue, none)
       - col: [B, 4]
       - row: [B, 3]
       - direction: [B, 9]
       - angle: [B, 7]
       - event_type: [B, 6] (note, bomb, arc_start, arc_end, chain, none)
   ```

2. **Onset-level planner** (bidirectional Transformer):
   - Input: sequence of per-onset audio embeddings (one embedding per onset, extracted from the audio encoder)
   - Output: per-onset planning vectors
   - Trained jointly with NotePredictor

3. **Loss function:**
   ```python
   loss = (
       CE(n_notes_pred, n_notes_target)
       + sum(CE(attr_pred, attr_target) for attr in [color, col, row, dir, angle, type])
       + lambda_parity * parity_penalty(dir_pred, prev_dir)
       + lambda_ergo * ergonomic_penalty(col_pred, row_pred, color_pred)
       + lambda_collision * collision_penalty(positions)
   )
   ```

4. **Training data adapter**: Convert existing token sequences to structured format
   ```python
   def tokens_to_structured(token_seq):
       """Convert [NOTE, COLOR, COL, ROW, DIR, ANGLE, SEP, ...] to:
       {n_notes: 2, slots: [{color: 0, col: 1, row: 0, dir: 1, ...}, ...]}
       """
   ```

### Phase 3: Song Structure & Repetition — 1-2 Weeks

1. **SSM-based section detection**: Implement `segment_song()` using librosa's self-similarity matrix
2. **Section conditioning**: Add section_type, section_position embeddings to the planner
3. **Copy mechanism**: When planner detects repeated sections, bias toward copying previous section's patterns
4. **Curated dataset**: Build whitelist of 500-1000 top-quality maps

### Phase 4: Playability Reward — 1 Week

1. Train PlayabilityScorer on real maps + perturbations
2. Integrate as auxiliary loss
3. Optional: RLHF-style fine-tuning (PPO on playability score)

---

## Quick Win: Minimum Viable Fix List

If time is very limited, these 5 changes give the most bang for the buck:

1. **Cap notes per beat**: `MAX_NOTES = {"Expert": 2, "ExpertPlus": 3}` — force EOS after max notes
2. **Force parity**: Track last direction per color, mask out same-parity directions
3. **Color separation**: Red → columns 0-1, Blue → columns 2-3 (soft bias, not hard constraint)
4. **Stop stripping arcs/chains**: Let the model's v3 features through (cap density, don't eliminate)
5. **Reduce temperature**: Current 0.95 is too high for a model that hasn't fully converged. Try 0.7-0.8.

These 5 changes can be implemented in `beam_search.py` and `postprocess.py` in under a day.

---

## Appendix A: Bug — EOS Weight Config/Code Mismatch

The data pipeline agent discovered a **config/code mismatch** in EOS weight handling:

- `sequence.yaml` line 21: `eos_weight: 1.0` (what we intended)
- `seq_module.py` line 47: `_build_token_weights(vocab_size, rhythm_weight, eos_weight=0.3)` (default parameter)
- BUT `seq_module.py` line 257: Actually passes `eos_weight=eos_weight` from `__init__` params

**Verdict:** The config IS correctly used (it flows through `__init__` → `_build_token_weights`). The default parameter of 0.3 is only a fallback. No bug, but confusing code — the function signature suggests 0.3 is the default while the config says 1.0.

## Appendix B: Additional Bug — Flow Loss is Detached

The flow loss computation at `seq_module.py:337`:
```python
preds = logits.argmax(dim=-1).detach()
flow_loss = _compute_flow_loss(preds, prev_tokens, time_gap)
loss = ce_loss + alpha * flow_loss
```

`preds` is **detached** (`.detach()`) and `flow_loss` is computed as a simple Python ratio (`violations / count`), returned as `torch.tensor(value)`. This means:
1. **No gradients flow** from the flow loss back to the model
2. The flow loss is purely a monitoring metric — it does NOT affect training
3. The `alpha * flow_loss` addition to the total loss is meaningless for optimization

This is a critical finding: **the flow/parity loss that was supposed to teach the model alternating swing directions has zero effect on training.**

## Appendix C: Generated Map Analysis

### Expert Map (so_tired_rock.mp3, 1-week checkpoint)

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| Total notes | 662 (raw 569 after post) | 500-800 | OK |
| Notes per second | ~3.2 | 4-8 | LOW |
| X distribution | {0:117, 1:188, 2:215, 3:142} | Roughly even | OK |
| Y distribution | {0:296, 1:156, 2:210} | {0:47%, 1:28%, 2:25%} | OK (was inverted before!) |
| Direction (d=8 "any") | 98/662 = 14.8% | <5% | HIGH |
| Beats with 4+ notes | 40+ | <10 in a 3-min song | VERY HIGH |
| Same-color multi-notes | 88/270 = 32.6% | <5% | CRITICAL |
| Parity violations (pre-fix) | 301/823 = 36.6% | 0% | CRITICAL |
| Bombs/walls/arcs/chains | 0 (all stripped) | Some | MISSING |
| Post-processing corrections | 1,927 + 301 + 88 + 17 = 2,333 | Should be <50 | CATASTROPHIC |

### Training Data Baseline (20 songs, Expert/ExpertPlus)

| Metric | Value |
|--------|-------|
| Sequence length distribution | 48% len=7 (1 note), 31% len=14 (2 notes), 12% len=21 (3 notes) |
| Most common onset type | Single note (78%) |
| 2-note onsets | ~18% |
| 3+ note onsets | ~4% |

---

## Key Files to Modify

| File | Change |
|------|--------|
| `generation/beam_search.py` | Add constrained decoding (Phase 1) |
| `generation/postprocess.py` | Fix post-processing (Phase 1) |
| `generation/generate.py` | Wire up constraints, reduce temperature |
| `models/note_predictor.py` | NEW: structured prediction model (Phase 2) |
| `models/onset_planner.py` | NEW: bidirectional onset planner (Phase 2) |
| `training/note_module.py` | NEW: Lightning module for structured prediction (Phase 2) |
| `data/audio.py` | Add `segment_song()` (Phase 3) |
| `evaluation/playability.py` | Expand with differentiable scoring (Phase 4) |

---

## References

- Mapperatorinator (osu! automapper): Whisper encoder + sparse decoder architecture
- BeatSage: CNN onset detection + DNN note placement (non-autoregressive)
- InfernoSaber: Conv AE + TCN + DNN (per-frame prediction, v1 only)
- DDC (Dance Dance Convolution): Foundational rhythm game generation paper
- REMI: Compound word representation for music (handles repetition)
- Beat Saber parity concepts: https://bsmg.wiki/mapping/basic-mapping.html#parity
- BS Map Check criteria: https://kivalevan.me/BeatSaber-MapCheck/
- Self-similarity matrices: Müller, "Fundamentals of Music Processing", Chapter 4
