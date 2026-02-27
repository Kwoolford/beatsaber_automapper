# Beat Saber Automapper

An open-source AI system that generates high-quality Beat Saber levels from audio files. Given a song, the system produces a playable `.zip` level package containing notes, arcs, chains, bombs, obstacles, and a synchronized light show — targeting the v3 Beat Saber map format.

## Quick Start

```bash
# Install dependencies
uv venv --python 3.12
uv pip install --pre torch torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
uv sync

# Download training data
bsa-download --quota vanilla:10000 --quota chroma:2000 --quota noodle:1000

# Preprocess
bsa-preprocess --input data/raw --output data/processed

# Train all three stages
bsa-train stage=onset  data_dir=data/processed
bsa-train stage=sequence data_dir=data/processed
bsa-train stage=lighting data_dir=data/processed

# Generate a level
bsa-generate song.mp3 --difficulty Expert --output level.zip
```

## ML Pipeline Architecture

> **Note:** Keep this diagram in sync with the code. When changing model inputs,
> outputs, or stage structure, update this section and the `## Architecture`
> section in `CLAUDE.md`.

```
 INPUTS
 ──────
  song.wav/mp3/ogg       --difficulty Expert     --bpm 128 (or auto-detected)
  (raw audio)            (Easy→ExpertPlus)       --genre electronic (optional)


                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         AUDIO PREPROCESSING                                 │
│                                                                             │
│  Raw Audio → mono 44.1kHz → Mel Spectrogram + Structure Features           │
│              80 mel bands, 1024-pt FFT, 512 hop (~10ms/frame)              │
│              6 structure features: RMS, onset strength, bass/mid/high,      │
│              spectral centroid (all per-frame, librosa-derived)             │
│                                                                             │
│  Output: [80, T] mel + [6, T] structure   (~100 frames per second)         │
└─────────────────────────────────────────────────────────────────────────────┘
                                  │
                                  │  [80, T] + [6, T]
                                  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                      SHARED AUDIO ENCODER                                   │
│                    (used by all 3 stages)                                   │
│                                                                             │
│  4-layer CNN frontend  →  Linear projection (d_model=512)                  │
│  + Structure projection: Linear(6→512), added to CNN output                │
│  → Sinusoidal positional encoding                                           │
│  → Transformer Encoder (6 layers, 8 heads)                                 │
│                                                                             │
│  Output: contextualized frame embeddings  [T, 512]                         │
│          (enriched with song energy/structure information)                  │
└─────────────────────────────────────────────────────────────────────────────┘
                                  │
              ┌───────────────────┼────────────────────┐
              ▼                   ▼                     ▼
        STAGE 1              STAGE 2               STAGE 3
    ONSET DETECTION      NOTE GENERATION       LIGHTING GENERATION
    ───────────────      ──────────────        ───────────────────
    Inputs:              Inputs:               Inputs:
     audio embeddings     audio embeddings      audio embeddings
     difficulty emb       onset timestamps      note tokens (stage 2)
     genre emb            difficulty emb        difficulty emb
                          genre emb             genre emb
                          prev 8 onset seqs     beat grid timestamps
                                                slot embedding (4-pos)

    Arch:                Arch:                 Arch:
     6-block TCN          Transformer decoder   Transformer decoder
     + 2-layer Xfmr       8 layers, 8 heads     4 layers, 8 heads
     → Linear(1)          causal self-attn      cross-attn → audio
     → sigmoid            cross-attn → audio    note ctx = mean-pool
                          + prev_context[K=8]   slot emb (cycling 4)
                          512-frame context

    Loss:                Loss:                 Loss:
     BCE (Gaussian        CE (EOS 0.3x,         CE + label smooth
     smoothed labels)     rhythm 3x) +
                          flow loss (α=0.1)

    Output:              Output:               Output:
     per-frame onset      token sequences       lighting token seqs
     probability [T]      per onset             (constrained nucleus)
     → peak picking       (beam/nucleus,        → Chroma RGB colors
     → onset timestamps    min_length=3)         (energy→palette)

              │                   │                     │
              └───────────────────┴─────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                       TOKEN DECODER + EXPORT                                │
│                                                                             │
│  Note tokens  →  BeatmapTokenizer.decode_beatmap()  →  DifficultyBeatmap  │
│  Light tokens →  LightingTokenizer.decode_lighting() →  lighting events    │
│  Chroma       →  add_chroma_colors(events, energy)  →  RGB _customData    │
│                                                                             │
│  DifficultyBeatmap:                                                         │
│    colorNotes   bombNotes   obstacles   sliders   burstSliders              │
│    basicEvents  colorBoosts  (+ Chroma _color per event)                   │
│                                                                             │
│  beatmap_to_v3_dict() → v3 JSON .dat                                       │
│  build_info_dat()     → Info.dat (with Chroma suggestion)                  │
│  package_level()      → .zip                                               │
└─────────────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
                     ┌──────────────────────┐
                     │    OUTPUT: level.zip  │
                     ├──────────────────────┤
                     │  Info.dat            │
                     │  song.ogg            │
                     │  ExpertStandard.dat  │
                     │  cover.png (opt.)    │
                     └──────────────────────┘
                     → drag into ArcViewer to preview
```

### Conditioning Inputs

Every model stage is conditioned on:

| Input | Type | Values | Effect |
|-------|------|--------|--------|
| `difficulty` | learned embedding (5→512, additive) | Easy / Normal / Hard / Expert / ExpertPlus | Controls note density and pattern complexity |
| `genre` | learned embedding (11→512, additive) | electronic, rock, pop, anime, hip-hop, classical, jazz, country, video-game, other, unknown | Shapes map style and feel |
| `structure` | linear projection (6→512, additive) | RMS energy, onset strength, bass/mid/high energy, spectral centroid | Song energy awareness (drops, buildups, calm sections) |
| `prev_tokens` | mean-pool + project (Stage 2 only) | Previous K=8 onset token sequences | Inter-onset flow, color alternation, pattern diversity |
| `slot_emb` | learned embedding (Stage 3 only) | 4-position cycling (type/ET/VAL/BRIGHT) | Structural grammar enforcement for lighting events |

### Token Vocabulary (Stage 2 — 167 tokens)

```
Special:   PAD=0  EOS=1  SEP=2  BOS=3
Events:    NOTE=4  BOMB=5  WALL=6  ARC_START=7  ARC_END=8  CHAIN=9
Color:     RED=10  BLUE=11
Row:       Y0=12  Y1=13  Y2=14
Column:    X0=15  X1=16  X2=17  X3=18
Direction: UP=19 DOWN=20 LEFT=21 RIGHT=22 UL=23 UR=24 DL=25 DR=26 ANY=27
+ quantized bins for: angle offset, curvature (mu), wall dimensions, etc.

Example beat: NOTE RED Y0 X1 DOWN EOS
              → red note, bottom row, 2nd column, cut downward
```

## CLI Reference

### `bsa-generate`

```
bsa-generate song.mp3 [options]

Required:
  audio                     Input audio file (.mp3, .ogg, .wav)

Optional:
  --difficulty DIFF         Easy/Normal/Hard/Expert/ExpertPlus (default: Expert)
  --output PATH             Output .zip path (default: <audio>.zip)
  --bpm FLOAT               Song BPM — auto-detected via librosa if not set
  --genre GENRE             Genre hint: electronic, rock, pop, anime, hip-hop,
                            classical, jazz, country, video-game, other
                            (default: unknown)
  --onset-ckpt PATH         Trained Stage 1 checkpoint
  --seq-ckpt PATH           Trained Stage 2 checkpoint
  --lighting-ckpt PATH      Trained Stage 3 checkpoint (Stage 3 skipped if absent)
  --beam-size N             Beam search width (default: 8)
  --temperature FLOAT       Sampling temperature (default: 1.0)
  --nucleus-sampling        Use nucleus sampling instead of beam search
  --top-p FLOAT             Top-p for nucleus sampling (default: 0.9)
  --onset-threshold FLOAT   Onset detection threshold (default: 0.5)
```

### `bsa-download`

```
bsa-download [options]

  --quota CATEGORY:N        Per-category download quota (repeatable)
                            Categories: vanilla, chroma, noodle,
                                        mapping_extensions, vivify
  --count N                 Total count fallback (used if no --quota flags)
  --min-rating FLOAT        Minimum upvote ratio (default: 0.8)
  --min-year INT            Minimum upload year (default: 2022)
  --output PATH             Output directory (default: data/raw)

Example:
  bsa-download --quota vanilla:10000 --quota chroma:2000 --quota noodle:1000
```

### `bsa-preprocess`

```
bsa-preprocess [options]

  --input PATH              Raw data directory (default: data/raw)
  --output PATH             Output directory (default: data/processed)
  --exclude-categories ...  Skip mod categories (e.g. noodle mapping_extensions)
```

## Evaluation

Generated maps can be evaluated with browser-based tools — no installation needed:

| Tool | URL | Purpose |
|------|-----|---------|
| **ArcViewer** | https://allpoland.github.io/ArcViewer/ | 3D preview with game-accurate visuals |
| **BS Map Check** | https://kivalevan.me/BeatSaber-MapCheck/ | Structural error checking |
| **Map Inspector** | https://galaxymaster2.github.io/bs-parity/ | Parity / swing-direction check |

## Project Structure

```
beatsaber_automapper/
├── CLAUDE.md                    # Source of truth for architecture decisions
├── PROGRESS.md                  # Session-to-session handoff document
├── configs/model/               # Hydra configs for each model stage
├── src/beatsaber_automapper/
│   ├── data/                    # Download, parse, tokenize, dataset
│   ├── models/                  # AudioEncoder, OnsetModel, SequenceModel, LightingModel
│   ├── training/                # Lightning modules for each stage
│   ├── generation/              # Inference pipeline, beam search, export
│   └── evaluation/              # Metrics (onset F1, token accuracy)
├── scripts/                     # CLI entry points
└── tests/                       # pytest test suite (213 tests)
```

## Tech Stack

- **Python 3.12** · **PyTorch nightly cu128** (RTX 5090 / sm_120 support)
- **Lightning** for training · **Hydra** for config · **librosa** for BPM detection
- **soundfile** for audio I/O · **ruff** for linting · **pytest** for tests

## References

- [BSMG Map Format](https://bsmg.wiki/mapping/map-format.html)
- [BeatSaver API](https://api.beatsaver.com/docs/)
- [ArcViewer (3D preview)](https://allpoland.github.io/ArcViewer/)
