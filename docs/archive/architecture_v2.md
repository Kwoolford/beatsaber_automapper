# Architecture V2 — Research & Design Document

**Created:** 2026-02-23
**Purpose:** Preserve all architecture research, competing system analysis, and next-gen
design decisions so we can reference them without re-researching if we need to pivot.

---

## Table of Contents

1. [Competing Automapper Analysis](#competing-automapper-analysis)
2. [Phase 6: Best of All Worlds (Proven Techniques)](#phase-6-best-of-all-worlds)
3. [Phase 7: Next-Gen Post-Boom Innovations](#phase-7-next-gen-post-boom-innovations)
4. [Full V2 Architecture Diagram](#full-v2-architecture-diagram)
5. [Implementation Priority Tables](#implementation-priority-tables)
6. [Alternative Architectures Considered](#alternative-architectures-considered)

---

## Competing Automapper Analysis

### InfernoSaber (Best Open-Source BS Mapper)
- **GitHub:** https://github.com/fred-brenner/InfernoSaber---BeatSaber-Automapper
- **Architecture:** 4-stage pipeline
  1. Convolutional Autoencoder — compresses mel spectrogram into compact latent representation
  2. TCN (Temporal Convolutional Network) — onset detection on autoencoder features
  3. DNN — note type classification (picks from a dictionary of note patterns)
  4. DNN — difficulty scaling
- **Data:** Hundreds of curated Expert+ maps (quality over quantity)
- **Key insight:** Uses classification into a PATTERN DICTIONARY instead of token-by-token
  generation. Picks from ~200 pre-extracted note patterns rather than generating from scratch.
- **Strengths:** Full-song context via autoencoder, proven TCN for temporal patterns,
  heavy post-processing rules for playability
- **Weaknesses:** Pattern dictionary limits creativity, no arcs/chains, Expert+ only

### DeepSaber (Oxford Academic, 2019)
- **GitHub:** https://github.com/oxai/deepsaber
- **Architecture:** WaveNet encoder + Transformer decoder
- **Key innovations:**
  - "Humaneness regularization" — exponential penalty for notes placed too close together:
    `loss += lambda * exp(-2 * dist / window)` for each pair of predicted onsets
  - beam_size=17 for coherent multi-note generation
  - Peak threshold 0.33 (not 0.5) — lower threshold + NMS works better than sharp peaks
- **Weaknesses:** v1 map format only, small dataset, WaveNet is slow

### Mapperatorinator (Best Overall, osu! Mapper)
- **Architecture:** Whisper-backbone audio encoder + Transformer decoder
- **Key innovations:**
  - **388 mel bands** (vs standard 80) — preserves much more frequency detail
  - **39,000 maps** training set with aggressive quality filtering
  - **2,500 GPU hours** of training
  - **Rhythm token weighting at 3x** in loss — timing tokens are 3x more important than
    property tokens. This is critical — WHEN notes appear matters more than their color.
  - **Conditioning dropout (20%)** on all embeddings during training — enables
    Classifier-Free Guidance (CFG) at inference. Generate with:
    `output = uncond + scale * (cond - uncond)` for sharper difficulty/style control.
  - **DPO (Direct Preference Optimization)** — used community quality signals to refine
    the model after supervised pretraining
- **Whisper backbone details:** Initialized from Whisper-small encoder (768-dim, 12 layers).
  Modified for longer sequences. This gives a huge head start on audio understanding.
- **Strengths:** Best quality, modern techniques, massive scale
- **Weaknesses:** osu! not Beat Saber (different game mechanics), closed-source

### BeatLearning (Innovative Small Model)
- **GitHub:** https://github.com/sedthh/BeatLearning
- **Architecture:** BERT-style bidirectional encoder
- **Key innovation:**
  - **Audio foresight** — the model can "see ahead" in the audio while generating current
    notes. Musical events are anticipated (build-ups before drops, tension before release).
  - Implementation: asymmetric context window — more future frames than past frames
    (e.g., 64 past + 192 future = 256 total)
  - **Joint onset + note generation** — single model predicts both WHEN and WHAT
- **Strengths:** Elegant design, captures musical anticipation
- **Weaknesses:** Small scale, limited output types

### Beat Sage (Most Popular, Closed Source)
- **Architecture:** 2 neural networks on mel spectrogram windows
- **Approach:** Focuses on percussion detection for onset placement
- **Quality:** "Fun but inconsistent" — widely used but quality complaints are common
- **Lesson:** Small windows of spectrogram → mediocre results. Full-song context matters.

### DDC (Dance Dance Convolution, 2017)
- **Architecture:** CNN + RNN for onset detection
- **Relevance:** Early proof that CNN onset detection on mel spectrograms works for rhythm
  games. Our CNN frontend design borrows from this lineage.

### Summary Table

| System | Architecture | Data | Onset Method | Sequence Method | Quality |
|--------|-------------|------|-------------|-----------------|---------|
| InfernoSaber | AE + TCN + DNN | ~300 curated | TCN on AE features | Pattern dictionary | Best OSS |
| DeepSaber | WaveNet + Transformer | Small curated | CNN onset detector | Beam search (17) | Academic PoC |
| Mapperatorinator | Whisper + Transformer | 39K maps | Whisper features | Token-by-token + DPO | Best overall |
| BeatLearning | BERT-style | Small | Joint w/ notes | Bidirectional | Innovative |
| Beat Sage | 2x NN | Unknown | NN on mel window | NN | Popular, mediocre |
| DDC | CNN + RNN | Variable | CNN on mel | RNN | Historical |

---

## Phase 6: Best of All Worlds

### From InfernoSaber
- **Audio autoencoder** for compact full-song representation
- **TCN for onset detection** — proven, efficient, large receptive fields via dilated convolutions
- **Heavy post-processing rules** — sanity checks, playability filters, pattern enforcement
- **Separate difficulty scaling** external to model

### From DeepSaber
- **Humaneness regularization** — penalize notes placed too close together:
  `loss += lambda * exp(-2 * dist / window)` for each predicted onset pair
- **beam_size=17** for coherent generation (our beam=8 may be too small)
- **Peak threshold 0.33** (not 0.5) — lower threshold + post-processing NMS

### From Mapperatorinator
- **Rhythm token weighting at 3x** in loss — timing tokens are the hardest and most important
- **Conditioning dropout (20%)** on all embeddings — enables CFG at inference
- **388 mel bands** instead of 80 — more frequency detail (we'll compromise at 192)
- **Whisper weight initialization** — pretrained audio features as starting point

### From BeatLearning
- **Audio foresight** — asymmetric context: 64 past + 192 future frames
- **Joint onset + note generation** — longer-term goal, eliminates Stage 1→2 error propagation

### Concrete Architecture V2 Plan

**Audio Encoder V2:**
- Increase mel bands: 80 → 192
- CNN frontend: 4 layers, stride=(2,1) on freq → 192/16 = 12 freq bins
- Projection: 256×12 = 3072 → d_model=512
- Transformer encoder: 6 layers, 8 heads
- Consider Whisper-small initialization (adapter layer for mel band mismatch)
- Full-song processing: CNN 4x freq downsample, 3-min song = 15,168 frames fits

**Onset Model V2:**
- Replace 2-layer Transformer decoder → Hybrid TCN + Transformer:
  - TCN (4 blocks, dilations 1,2,4,8,16,32, 128 filters) for local pattern detection
  - 2-layer Transformer on top for global context
- Remove genre embedding, keep difficulty
- Add humaneness regularization to loss
- pos_weight=1.0, Gaussian sigma=2 (sharper peaks)
- Window size: 2048 frames (~24 seconds)
- Conditioning dropout 20% on difficulty

**Sequence Model V2:**
- Keep 8-layer autoregressive Transformer decoder
- Add rhythm token weighting: timing tokens (EVENT_TYPE, SEP, EOS) at 3x in loss
- Audio foresight: 64 past + 192 future = 256 total context (asymmetric)
- Conditioning dropout 20% on difficulty + genre → enables CFG
- Pattern diversity loss: auxiliary term penalizing low-entropy output distributions
- Top-k constrained beam search: only consider playability-valid tokens at each step

**Post-Processing Pipeline:**
1. NPS enforcement (thin notes if density exceeds target)
2. Color rebalancing (force 45-55% red/blue)
3. Direction diversity (cap any direction at 40%)
4. Grid coverage (shift notes to unused cells)
5. Pattern deduplication (inject variation after 5+ repeats)
6. Bomb/wall injection (rule-based placement)
7. Parity check (swing direction alternation)

---

## Phase 7: Next-Gen Post-Boom Innovations

### Innovation 1: Mamba/SSM Audio Encoder — Full-Song, Linear Time

**Problem:** Transformer self-attention is O(n^2). A 3-minute song = 15,168 frames =
~230M attention pairs per layer.

**Solution:** Replace Transformer encoder with Mamba (Selective State Space Model).
- O(n) linear time with learned selective scan
- Processes entire song in one pass — no windowing
- The selective state naturally captures musical structure: remember beat patterns
  during verse, update on chorus, forget noise between sections
- Memory: O(n) vs O(n^2) — 15,168 frames uses ~60MB vs ~3.5GB

**Implementation:**
```
Audio Encoder V3:
  Mel spec [80, T] → CNN frontend (4 layers) → [T, 1280] → Linear → [T, 512]
  → Bidirectional Mamba (6 layers, d_state=64, d_conv=4, expand=2)
  → Output: [T, 512] frame embeddings with full-song context
```

**Package:** `pip install mamba-ssm` (CUDA-optimized selective scan kernels)

### Innovation 2: RoPE + GQA + SwiGLU — Modern Transformer Internals

**Rotary Position Embeddings (RoPE):**
- Encodes relative position via rotation matrices on Q and K
- Naturally handles variable-length sequences (no max_len buffer)
- Extrapolates to longer sequences than seen in training
- Eliminates PE buffer overflow bugs

**Grouped Query Attention (GQA):**
- Share K/V across groups of query heads
- 8 query heads, 2 KV groups → 4x smaller KV cache, 30% faster inference
- Critical for beam search speed

**SwiGLU Activation:**
- `SwiGLU(x) = Swish(xW1) * (xW2)`
- Used in LLaMA, PaLM, Mistral — consistently outperforms GELU/ReLU
- Same parameter count, free performance improvement

**RMSNorm:**
- Faster than LayerNorm (no mean subtraction)
- Used in all modern LLMs, drop-in replacement

### Innovation 3: KV-Cached Beam Search — 10x Faster Generation

**Problem:** 1688 onsets × 64 token steps × beam_size=8 = 864,000 forward passes,
each recomputing attention from scratch. Generation takes 11 minutes.

**Solution:** Cache self-attention K/V from previous positions. Each new step only
computes attention for the NEW token against cached K/V.

```python
class KVCache:
    def __init__(self, num_layers, max_seq_len, num_kv_heads, head_dim, device):
        self.k_cache = [torch.zeros(batch, num_kv_heads, max_seq_len, head_dim, device=device)
                        for _ in range(num_layers)]
        self.v_cache = [...]
        self.seq_pos = 0

    def update(self, layer_idx, new_k, new_v):
        self.k_cache[layer_idx][:, :, self.seq_pos] = new_k
        self.v_cache[layer_idx][:, :, self.seq_pos] = new_v
        self.seq_pos += 1
```

**Expected:** Generation from 11 minutes → 60-90 seconds.

### Innovation 4: Hierarchical Structure-Aware Generation

No existing automapper does this. Human mappers think hierarchically.

**Level 1 — Song Structure Segmentation:**
- Input: Full-song Mamba audio features
- Output: Segment boundaries + labels (intro, verse, chorus, bridge, drop, outro)
- Architecture: Linear classifier on Mamba features

**Level 2 — Phrase-Level Onset Density:**
- Input: Audio features + structure labels + difficulty
- Output: Per-phrase onset density curve
- Predicts "verse=4 NPS, chorus=7 NPS, bridge=2 NPS"

**Level 3 — Note-Level Generation:**
- Input: Audio features + density plan + difficulty
- Output: Individual onset frames + note tokens
- Has both local audio features AND global density target

**Training data for structure:** Bootstrap with pretrained music structure analysis
(MusicFM, MERT) or heuristic (spectral energy + novelty detection + k-means).

### Innovation 5: DPO for Map Quality

**Natural preference signals available:**
- BeatSaver upvote ratio (0-100%)
- ScoreSaber ranked status
- NPS appropriateness per difficulty
- Download/play count (popularity proxy)

**DPO for beatmaps:**
1. Generate map pairs for same song using different checkpoints/temperatures
2. Use BeatSaver signals to determine preferred map
3. Train: `L = -log sigma(beta * (log pi(y_w|x) - log pi(y_l|x)))`

**Or learned reward model:**
1. Train: AudioEncoder + MapEncoder → quality score (0-1)
2. Features: NPS, pattern diversity, grid coverage, direction distribution, color balance
3. Guide beam search with reward model scores

Requires working base model first — apply after supervised training (Phases 1-6).

### Innovation 6: Speculative Decoding

1. Train tiny "draft" model (2-layer, d=128) alongside main model
2. Draft generates N candidate tokens quickly
3. Main model verifies all N in one forward pass
4. Accept longest correct prefix
5. Typical acceptance: 70-90% → 2-3x additional speedup

For beatmaps: draft model = simple pattern lookup table (most common note configs).
Gets generation to ~20-30 seconds for a 3-minute song.

---

## Full V2 Architecture Diagram

```
============================================================
  BeatSaber Automapper v2 — "NextGen" Architecture (2026)
============================================================

  Audio (.mp3/.ogg/.wav)
          |
          v
  +-------------------------------------------------------+
  |  MEL SPECTROGRAM (192 bands, 1024 FFT, 512 hop)       |
  |  -> CNN Frontend (4 layers, freq downsample 16x)       |
  |  -> Bidirectional Mamba Encoder (6 layers, d=512)      |
  |    * Full-song context in O(n) linear time             |
  |    * No windowing -- processes entire 3-min song       |
  |  Output: [T, 512] frame embeddings                     |
  +-----------+--------------------------------------------+
              |
    +---------+---------+
    v         v         v
  +------+ +-------+ +-------+
  |STRUCT| |ONSET  | |LIGHT  |
  |LABEL | |DETECT | |GEN    |
  |      | |       | |       |
  |Seg-  | |Hybrid | |4-layer|
  |ment  | |TCN +  | |RoPE + |
  |into  | |RoPE   | |GQA    |
  |verse/| |Trans- | |decoder|
  |chorus| |former | |       |
  |/drop | |decoder| |       |
  +--+---+ +--+----+ +--+----+
     |        |         |
     v        v         |
  +---------------+     |
  | NOTE SEQUENCE |     |
  | GENERATION    |     |
  |               |     |
  | 8-layer RoPE +|     |
  | GQA + SwiGLU  |     |
  | decoder       |     |
  |               |     |
  | * KV-cached   |     |
  |   beam search |     |
  | * Audio fore- |     |
  |   sight (asym)|     |
  | * Structure-  |     |
  |   conditioned |     |
  | * CFG via     |     |
  |   cond dropout|     |
  +-------+-------+     |
          |             |
          v             v
  +------------------------------+
  |  POST-PROCESSING PIPELINE   |
  |  NPS enforcement, color     |
  |  rebalancing, direction     |
  |  diversity, parity check,   |
  |  pattern deduplication      |
  +--------------+---------------+
                 |
                 v
  +------------------------------+
  |  v3 JSON EXPORT -> .zip      |
  |  (After DPO refinement)     |
  +------------------------------+

What makes this unique vs ALL existing automappers:
  1. Mamba encoder -- no other mapper uses SSMs for audio
  2. Hierarchical structure -- no other mapper segments songs
  3. RoPE/GQA/SwiGLU -- modern LLM internals, not 2017 vanilla
  4. KV-cached beam search -- 10x faster inference
  5. DPO quality refinement -- RLHF-era alignment for beatmaps
  6. Speculative decoding -- another 2-3x inference speedup
  7. Full-song context -- most use small windows or autoencoders
============================================================
```

---

## Implementation Priority Tables

### Proven Techniques (Phase 6)

| Priority | Change | Source | Why |
|----------|--------|--------|-----|
| P0 | Fix pos_weight, window, genre | Analysis | Unblocks all learning |
| P0 | Post-processing pipeline | InfernoSaber | Immediately improves any output |
| P1 | Curated gold dataset | InfernoSaber | Clean signal >> more noise |
| P1 | Conditioning dropout 20% | Mapperatorinator | Enables CFG, better generalization |
| P1 | Rhythm token weighting 3x | Mapperatorinator | Proven critical improvement |
| P2 | Audio foresight (asymmetric) | BeatLearning | Build-up/drop anticipation |
| P2 | Humaneness regularization | DeepSaber | Playability constraint in loss |
| P2 | Pattern diversity loss | Novel | Prevents mode collapse |
| P3 | TCN hybrid onset model | InfernoSaber | Better architecture if Transformer stalls |
| P3 | 192 mel bands | Mapperatorinator | More audio detail |
| P3 | Whisper weight init | Mapperatorinator | Pretrained features |
| P4 | Joint onset+note model | BeatLearning | Research project, long-term |

### Next-Gen Innovations (Phase 7)

| Priority | Innovation | Effort | Expected Impact |
|----------|-----------|--------|-----------------|
| P1 | RoPE (replace sinusoidal PE) | 1 day | Variable-length handling, no overflow |
| P1 | KV-cached beam search | 2 days | 10x faster inference |
| P2 | GQA | 1 day | 4x smaller KV cache, 30% faster |
| P2 | SwiGLU + RMSNorm | 0.5 day | Free quality improvement |
| P3 | Mamba audio encoder | 3 days | Full-song O(n) processing |
| P3 | Hierarchical generation | 5 days | Musical structure awareness |
| P4 | DPO | 3 days | Quality refinement (needs base model) |
| P4 | Speculative decoding | 2 days | 2-3x additional inference speedup |

---

## Alternative Architectures Considered

### Option A: Pure TCN (InfernoSaber-style)
Replace all Transformers with TCNs + DNNs. Simpler, proven, but limited creativity.
Good fallback if Transformer-based approach continues to stall after P0 fixes.

### Option B: Diffusion-based Generation
Generate beatmaps via a denoising diffusion process over token space. Novel but
unproven for discrete structured outputs like beatmaps. Would require significant
research.

### Option C: Retrieval-Augmented Generation
At inference time, find the K most similar songs in the training set (by audio
embedding similarity), retrieve their beatmaps, and use them as "examples" in the
context. The model learns to adapt existing patterns rather than generating from
scratch. This is the InfernoSaber pattern dictionary approach taken further.

### Option D: Multi-Task Pretraining
Pretrain the audio encoder on multiple music understanding tasks (BPM detection,
key estimation, genre classification, instrument recognition) before fine-tuning
for beatmap generation. This gives richer audio representations.

### Option E: GAN-based Quality
Train a discriminator to distinguish human-mapped from AI-generated beatmaps.
Use adversarial training to push generated maps toward the distribution of
human-made maps. Risk: mode collapse, training instability.
