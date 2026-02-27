# CLAUDE.md — Beat Saber Automapper

This is the source-of-truth document for the beatsaber_automapper project. Read this file completely before starting any work.

## Architecture Diagram — Keep Updated

`README.md` contains the canonical ML pipeline flow diagram under "## ML Pipeline Architecture".
**Whenever you change model inputs, outputs, conditioning signals, or stage structure, update both:**
1. The ASCII diagram in `README.md` (the `## ML Pipeline Architecture` section)
2. The `## Architecture` section below in this file

Current conditioning inputs per stage:
- **All stages:** difficulty (5-class embedding) + genre (11-class embedding) + song structure features (6-dim per-frame projection), all additive to audio encoder output.
- **Stage 2 additionally:** previous K=8 onset token sequences (mean-pooled, projected, concatenated to cross-attention memory).
- **Stage 3 additionally:** structural slot embedding (4-position cycling) for event grammar.
- **Post-processing:** Chroma RGB lighting colors derived from song energy profile.

BPM is auto-detected via `detect_bpm()` in `data/audio.py` when not provided by the user.

## Project Overview

An open-source AI system that generates high-quality Beat Saber levels from audio files. Given a song, the system produces a playable .zip level package containing notes, arcs, chains, bombs, obstacles, and a synchronized light show — targeting the v3 Beat Saber map format.

Repository: https://github.com/Kwoolford/beatsaber_automapper
Platform: Windows (native), Python 3.12, PyTorch (nightly for RTX 5090 sm_120 support)
GPU: NVIDIA RTX 5090 (32GB VRAM, Blackwell/sm_120, CUDA 13.1 driver, CUDA Toolkit 12.9)

## Environment Setup (Verified Working)

```powershell
# Python 3.12 via uv (system Python is separate)
uv python install 3.12
uv venv --python 3.12
.venv\Scripts\activate

# PyTorch nightly with CUDA 12.8 (sm_120 support)
uv pip install --pre torch torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128

# Verify GPU
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_capability())"
# Expected: True (12, 0)
```

Always use `uv pip install` (not bare `pip`) inside the venv. Bare pip may install to the system Python.

## Priority Order

1. Generate playable Beat Saber maps from audio
2. State-of-the-art model quality
3. Clean modern codebase that is easy to extend
4. Working demo that can be shown off quickly

## Architecture

### Three-Stage Pipeline

```
Audio File (.mp3/.ogg/.wav)
        |
        v
+-------------------+
|  Audio Encoder    |  Mel spectrogram -> CNN frontend -> Transformer encoder
|  (shared across   |  Output: contextualized audio frame embeddings
|   all stages)     |  Config: 80 mel bands, 1024 FFT, 512 hop (~23ms/frame)
+--------+----------+
         |
   +-----+------+--------+
   v            v         v
+--------+ +----------+ +----------+
|Stage 1 | | Stage 2  | | Stage 3  |
|Onset   | | Note     | | Lighting |
|Predict | | Sequence | | Events   |
+--------+ +----------+ +----------+
   |            |             |
   v            v             v
 Timestamps   Note events   Light events
   |            |             |
   +-----+------+--------+---+
         v
   Beat Saber v3 .dat JSON -> .zip
```

### Stage 1: Onset / Beat Prediction
- Task: Binary classification per audio frame — "should a note appear here?"
- Architecture: Audio encoder output (with structure features) -> 6-block TCN (dilated convolutions, 128ch) -> 2-layer Transformer decoder (with difficulty + genre embedding) -> linear -> sigmoid
- Training: Binary cross-entropy with Gaussian-smoothed ("fuzzy") labels around true onsets
- Inference: Windowed prediction (1024-frame windows with overlap averaging) -> peak picking with configurable threshold

### Stage 2: Note Sequence Generation
- Task: Given onset timestamps + audio + previous onset context, generate the full note configuration at each onset
- Architecture: Transformer decoder (8 layers, 8 heads, d_model=512) with:
  - Causal self-attention over note token sequence (autoregressive)
  - Cross-attention to audio encoder output + previous K=8 onset context vectors
  - Difficulty + genre conditioning via learned embeddings (additive)
  - Inter-onset context: previous 8 onset token sequences are mean-pooled, projected, and concatenated to cross-attention memory alongside audio features
- Audio context: 512 frames (~6 seconds) per onset for musical phrase awareness
- Training: Teacher forcing with cross-entropy loss (EOS downweighted to 0.3x, rhythm tokens 3x) + flow-aware auxiliary loss (parity violation penalty, alpha=0.1)
- Inference: Beam search or nucleus sampling with min_length=3 (suppresses premature EOS)

### Stage 3: Lighting Generation
- Task: Given audio features + generated note sequence, produce lighting events
- Architecture: Transformer decoder (4 layers, 8 heads) with:
  - Structural slot embedding (4-position cycling: event_type -> ET -> VAL -> BRIGHT)
  - Cross-attention to audio encoder output (with structure features)
  - Note context via mean-pooled note token embeddings (additive)
- Training: CrossEntropyLoss with label_smoothing=0.1
- Inference: Constrained nucleus sampling (state machine enforces valid event grammar)
- Post-processing: Chroma RGB colors added from song energy profile (6 curated palettes)

### Audio Encoder (Shared)
- Input: Raw audio -> mono 44.1kHz -> Mel spectrogram (80 bands, 1024 FFT, 512 hop, ~10ms/frame)
- Architecture: 4-layer CNN frontend -> sinusoidal positional encoding -> Transformer encoder (6 layers, 8 heads, d_model=512)
- Song structure features: 6 per-frame features (RMS energy, onset strength, bass/mid/high energy, spectral centroid) projected via nn.Linear(6, 512) and added to CNN output
- Output: One embedding vector per ~10ms audio frame, enriched with song energy information
- This is a task-specific encoder (NOT a pretrained speech model) — we need low-level rhythmic features

### Token Vocabulary (Stage 2)

Each note event is tokenized as a sequence of tokens:

```
[EVENT_TYPE] [COLOR] [ROW] [COLUMN] [DIRECTION] [ANGLE_OFFSET]
```

Event types include:
- NOTE: Standard directional note
- BOMB: Avoid zone (no color/direction)
- WALL: Obstacle (has width, height, duration instead of direction)
- ARC_START: Beginning of a slider/arc (includes curvature multiplier)
- ARC_END: End of a slider/arc
- CHAIN: Burst slider (includes slice count, squish factor)
- SEP: Separator between events at the same timestamp
- EOS: End of sequence for this timestamp

Multiple notes at the same timestamp use SEP tokens between them, canonical ordering: left-to-right, bottom-to-top.

Grid coordinates: x=0-3 (columns, left to right), y=0-2 (rows, bottom to top)
Directions: 0=up, 1=down, 2=left, 3=right, 4=up-left, 5=up-right, 6=down-left, 7=down-right, 8=any

### Difficulty Conditioning
Both Stage 1 and Stage 2 receive a difficulty embedding. A single trained model can generate all difficulty levels. The difficulty affects:
- Stage 1: Note density (how many onsets)
- Stage 2: Pattern complexity, note types used (arcs/chains more common at higher difficulties)

## Beat Saber v3 Map Format

Target format version: v3 (introduced in Beat Saber 1.20.0). This has the widest community tooling support and includes arcs + chains.

### Map Structure (what we generate)
```
level.zip/
  Info.dat              # Song metadata, BPM, difficulty list, environment
  song.ogg              # Audio file (converted from input)
  cover.png             # Cover image (optional, can be auto-generated)
  ExpertStandard.dat    # Beatmap + lightshow for Expert difficulty
  ExpertPlusStandard.dat # Beatmap + lightshow for Expert+ difficulty
  ... (one .dat per difficulty)
```

### Key v3 JSON Collections (in each difficulty .dat)
```json
{
  "version": "3.3.0",
  "colorNotes": [{"b": 10.0, "x": 1, "y": 0, "c": 0, "d": 1, "a": 0}],
  "bombNotes": [{"b": 10.0, "x": 1, "y": 0}],
  "obstacles": [{"b": 10.0, "d": 5.0, "x": 1, "y": 0, "w": 1, "h": 5}],
  "sliders": [{"c": 0, "b": 10.0, "x": 1, "y": 0, "d": 1, "mu": 1.0, "tb": 15.0, "tx": 2, "ty": 2, "tc": 0, "tmu": 1.0, "m": 0}],
  "burstSliders": [{"c": 0, "b": 10.0, "x": 1, "y": 0, "d": 1, "tb": 15.0, "tx": 2, "ty": 2, "sc": 3, "s": 0.5}],
  "basicBeatmapEvents": [{"b": 10.0, "et": 1, "i": 3, "f": 1.0}],
  "colorBoostBeatmapEvents": [{"b": 10.0, "o": true}]
}
```

Field reference:
- b = beat, x = column (0-3), y = row (0-2), c = color (0=red,1=blue), d = direction (0-8), a = angle offset
- Sliders: mu/tmu = curvature multipliers, tb/tx/ty/tc = tail beat/position/direction, m = mid anchor mode
- Burst sliders: sc = slice count, s = squish factor
- Events: et = event type, i = value (on/off/flash/fade), f = float value (brightness)

### Canonical Info.dat Reference
```json
{
  "_version": "2.1.0",
  "_songName": "Song Title",
  "_songSubName": "",
  "_songAuthorName": "Artist",
  "_levelAuthorName": "beatsaber_automapper",
  "_beatsPerMinute": 120,
  "_shuffle": 0,
  "_shufflePeriod": 0.5,
  "_previewStartTime": 12,
  "_previewDuration": 10,
  "_songFilename": "song.ogg",
  "_coverImageFilename": "cover.png",
  "_environmentName": "DefaultEnvironment",
  "_songTimeOffset": 0,
  "_difficultyBeatmapSets": [
    {
      "_beatmapCharacteristicName": "Standard",
      "_difficultyBeatmaps": [
        {
          "_difficulty": "Expert",
          "_difficultyRank": 7,
          "_beatmapFilename": "ExpertStandard.dat",
          "_noteJumpMovementSpeed": 16,
          "_noteJumpStartBeatOffset": 0
        }
      ]
    }
  ]
}
```

## Data Strategy

### Source: BeatSaver API (https://beatsaver.com)

Public REST API. Key endpoints:
- GET /api/maps/latest — paginated maps
- GET /api/search/text/{page}?sortOrder=Rating — search by rating
- Download via: https://r2cdn.beatsaver.com/{hash}.zip (hash from API response)

### Curation Filters
- Rating: ≥80% upvote ratio (score field)
- Difficulty: Must have Expert or ExpertPlus
- Vintage: Post-2020 maps preferred (newer mapping conventions)
- NPS: Filter out >20 NPS (likely bad data)
- Ranked: ScoreSaber-ranked maps are gold standard

Target dataset size: 5,000-10,000 high-quality map+audio pairs

### Preprocessing Pipeline
1. Download maps via BeatSaver API with quality filters and rate limiting
2. Parse Info.dat for BPM, song offset, difficulty metadata
3. Parse difficulty .dat files -> extract all note/arc/chain/bomb/obstacle/lighting events
4. Convert audio to mono 44.1kHz WAV
5. Extract mel spectrograms (80 bands, 1024 FFT, 512 hop)
6. Tokenize note events using vocabulary defined above
7. Align audio frames with note timestamps using BPM and offset
8. Train/val/test split BY SONG (never split a song across sets)
9. Cache preprocessed tensors as .pt files for fast training iteration

### BeatSaver TOS Note
AI-generated maps uploaded to BeatSaver must be declared as AI-generated. They may be subject to a 90-day retention period. This doesn't affect our training data downloads or local generation, but keep in mind for any future uploads.

## Evaluation & Preview Tools

These are browser-based tools that require NO installation. Use them to evaluate generated maps.

### ArcViewer (Primary Previewer)
- URL: https://allpoland.github.io/ArcViewer/
- Desktop app: https://github.com/AllPoland/ArcViewer/releases
- Drag in a generated .zip file to see a full 3D preview with game-accurate visuals
- Supports all v3 features: arcs, chains, angle offset, walls, lighting events, multiple environments
- Adjustable playback speed, custom colors, hitsounds
- THIS IS THE PRIMARY WAY TO EVALUATE GENERATED MAPS

### BS Map Check (Error Checker)
- URL: https://kivalevan.me/BeatSaber-MapCheck/
- Flags structural errors, ranking criteria violations, and potential issues
- Use this to validate that generated .dat files are well-formed before visual preview
- Categories: ranking issues, errors, warnings, info

### Beat Saber Map Inspector (Parity Checker)
- URL: https://galaxymaster2.github.io/bs-parity/
- Checks note parity (swing direction flow) — important for playability
- Catches patterns that are technically valid JSON but feel wrong in-game

### Evaluation Workflow
1. Generate a .zip with `bsa-generate`
2. Upload to BS Map Check — fix any errors/warnings
3. Open in ArcViewer — visually inspect note placement, flow, lighting sync
4. Optionally check parity with Map Inspector
5. For quantitative eval: compute onset F1, token accuracy, pattern diversity metrics in code

## Tech Stack

| Tool | Purpose | Version |
|------|---------|---------|
| Python | Runtime | 3.12 (via uv venv) |
| PyTorch | ML framework | Nightly with cu128 (for sm_120) |
| torchaudio | Audio loading/transforms | Match PyTorch nightly |
| librosa | Audio analysis, onset detection | Latest stable |
| Lightning | Training loop, logging, checkpoints | ≥2.2 |
| Hydra | Configuration management | ≥1.3 |
| wandb | Experiment tracking (optional, TensorBoard fallback) | Latest |
| soundfile | Audio I/O | Latest |
| ruff | Linting + formatting | Latest |
| pytest | Testing | Latest |

### RTX 5090 / sm_120 Notes
- Stable PyTorch does NOT support sm_120 as of early 2026
- Use nightly: `uv pip install --pre torch torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128`
- Verified working: torch 2.11.0.dev+cu128 on Python 3.12, CUDA driver 13.1, toolkit 12.9
- Always verify: `python -c "import torch; print(torch.cuda.get_device_capability())"` should return `(12, 0)`
- 32GB VRAM allows batch_size=64+ for our model sizes
- TORCH_CUDA_ARCH_LIST="12.0" may be needed if building extensions from source

### No Pretrained Weights Required
This project trains all models from scratch on Beat Saber map data. There are no external pretrained weights to download. The audio encoder, onset model, sequence model, and lighting model are all trained end-to-end on our curated dataset.

## Project Structure

```
beatsaber_automapper/
├── CLAUDE.md                    # THIS FILE — source of truth
├── README.md                    # Public-facing docs
├── pyproject.toml               # Dependencies, project metadata, CLI entrypoints
├── .gitignore                   # Ignore .venv, data/, wandb/, __pycache__, etc.
├── configs/
│   ├── model/
│   │   ├── audio_encoder.yaml   # Shared audio encoder config
│   │   ├── onset.yaml           # Stage 1 model config
│   │   ├── sequence.yaml        # Stage 2 model config
│   │   └── lighting.yaml        # Stage 3 model config
│   ├── data/
│   │   └── default.yaml         # Data pipeline config
│   └── train.yaml               # Top-level Hydra training config
├── src/
│   └── beatsaber_automapper/
│       ├── __init__.py
│       ├── data/
│       │   ├── __init__.py
│       │   ├── download.py      # BeatSaver API scraper with rate limiting
│       │   ├── dataset.py       # PyTorch Dataset classes
│       │   ├── audio.py         # Mel spectrogram extraction, audio loading
│       │   ├── beatmap.py       # v3 .dat parser (notes, arcs, chains, bombs, walls, lights)
│       │   ├── tokenizer.py     # Note/lighting event <-> token vocabulary
│       │   └── augment.py       # Time stretch, pitch shift, noise injection
│       ├── models/
│       │   ├── __init__.py
│       │   ├── audio_encoder.py # CNN + Transformer encoder (shared)
│       │   ├── onset_model.py   # Stage 1: onset prediction
│       │   ├── sequence_model.py # Stage 2: note sequence generation
│       │   ├── lighting_model.py # Stage 3: lighting generation
│       │   └── components.py    # Shared building blocks (attention, positional encoding, etc.)
│       ├── training/
│       │   ├── __init__.py
│       │   ├── onset_module.py  # Lightning module for Stage 1
│       │   ├── seq_module.py    # Lightning module for Stage 2
│       │   └── light_module.py  # Lightning module for Stage 3
│       ├── generation/
│       │   ├── __init__.py
│       │   ├── generate.py      # End-to-end inference pipeline
│       │   ├── beam_search.py   # Beam search for Stage 2
│       │   └── export.py        # Tokens -> v3 JSON -> Beat Saber .zip
│       └── evaluation/
│           ├── __init__.py
│           ├── metrics.py       # Onset F1, token accuracy, pattern diversity
│           └── playability.py   # Heuristic rule checks (no impossible patterns)
├── scripts/
│   ├── download_data.py         # CLI: download maps from BeatSaver
│   ├── preprocess.py            # CLI: preprocess downloaded maps into tensors
│   ├── build_index.py           # CLI: rebuild frame_index.json after adding/removing .pt files
│   ├── train.py                 # CLI: train a model stage
│   ├── generate.py              # CLI: generate a level from audio
│   ├── dashboard.py             # Launch TensorBoard + print latest metrics summary
│   └── run_training_pipeline.py # Chain onset→sequence→lighting automatically
├── tests/
│   ├── test_tokenizer.py
│   ├── test_beatmap_parser.py
│   ├── test_dataset.py
│   ├── test_audio.py
│   └── test_generation.py
└── data/                        # Git-ignored, local data storage
    ├── raw/                     # Downloaded map zips
    ├── processed/               # Preprocessed tensors
    └── generated/               # Generated output levels
```

## pyproject.toml Skeleton

```toml
[project]
name = "beatsaber-automapper"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = [
    "torch>=2.7.0",
    "torchaudio>=2.7.0",
    "lightning>=2.2",
    "librosa>=0.10",
    "hydra-core>=1.3",
    "omegaconf",
    "soundfile",
    "numpy",
    "requests",
    "tqdm",
]

[project.optional-dependencies]
dev = ["pytest", "ruff", "mypy"]
wandb = ["wandb"]

[project.scripts]
bsa-download = "scripts.download_data:main"
bsa-preprocess = "scripts.preprocess:main"
bsa-train = "scripts.train:main"
bsa-generate = "scripts.generate:main"

[tool.ruff]
line-length = 100
target-version = "py312"

[tool.ruff.lint]
select = ["E", "F", "I", "N", "W", "UP"]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

## PR Roadmap

Execute these in order. Each PR should be a working increment.

### PR 1: Repo Scaffolding
Create the full project structure, pyproject.toml, configs, empty modules with docstrings, .gitignore, ruff config, basic CI. All tests pass (even if trivial). `uv sync` works.

Definition of done: `uv sync && ruff check . && pytest` all pass.

### PR 2: Data Pipeline
BeatSaver API client, Info.dat/beatmap.dat parsers (full v3 support including arcs, chains, bombs, obstacles, lighting events), audio preprocessing, tokenizer, PyTorch Dataset, DataLoader. Download ~500 maps for dev.

Definition of done: Can iterate a DataLoader and see correctly shaped batch tensors. Parser handles all v3 object types. Tokenizer round-trips correctly (encode -> decode produces identical beatmap data).

### PR 3: Audio Encoder + Stage 1 (Onset Detection)
Implement audio encoder (CNN + Transformer), onset prediction head, Lightning training module, Hydra configs, training script. Evaluate with onset F1 score.

Definition of done: Onset F1 > 0.6 on validation set. Can visualize predicted onsets overlaid on audio waveform.

### PR 4: Stage 2 (Note Sequence Generation)
Transformer decoder with cross-attention to audio, difficulty conditioning, token vocabulary embedding, beam search. Generates all v3 note types (notes, arcs, chains, bombs, walls).

Definition of done: Model generates syntactically valid v3-compatible note sequences. Beam search produces coherent, non-random patterns. Generated .dat files pass BS Map Check without errors.

### PR 5: End-to-End Generation + Export
Full pipeline: audio in -> Stage 1 -> Stage 2 -> v3 .dat JSON -> Beat Saber .zip. CLI command: `bsa-generate song.mp3 --difficulty Expert --output level.zip`

Definition of done: Generated .zip loads and previews correctly in ArcViewer. Notes are visibly synced to music.

### PR 6: Stage 3 (Lighting)
Lighting model, trained on audio + note sequence -> lighting events. Produces basicBeatmapEvents, colorBoostBeatmapEvents, and light event box groups.

Definition of done: Generated levels have synchronized, non-random lighting visible in ArcViewer.

### PR 7: Scale Training + Quality
Full dataset (5-10k maps), data augmentation, hyperparameter tuning, playability heuristics, post-processing rules. Compare against InfernoSaber output.

### PR 8: Documentation + Demo
README, pretrained weights on HuggingFace, Gradio/Streamlit web demo, example outputs.

## Coding Standards

- Python 3.12, type hints on all public functions
- ruff for linting and formatting (line length 100)
- Docstrings on all classes and public methods (Google style)
- Tests for all data processing and tokenization logic
- Hydra configs for all hyperparameters (no magic numbers in code)
- Lightning modules for all training (no raw training loops)
- Use pathlib.Path, not string paths
- Use logging module, not print statements
- Git commits: conventional commits (feat:, fix:, refactor:, docs:, test:)
- Always use `uv pip install` inside the venv, never bare `pip`

## Key Design Decisions

1. **Fresh build, not a fork** — The original DeepSaber (oxai/deepsaber) targets v1 map format and uses obsolete ML patterns. We reference its two-stage concept but share no code.

2. **v3 map format** — Widest community tooling support, includes arcs/chains/burst sliders. v4 exists but tooling is unstable.

3. **Three stages, not two** — Lighting is a separate stage because it depends on both audio AND note patterns. Can be trained independently after Stages 1+2 work.

4. **Task-specific audio encoder** — Pretrained models (wav2vec2, CLAP) are tuned for speech/semantics. We need low-level rhythmic features (transients, beat subdivisions). A custom encoder trained end-to-end will capture these better.

5. **Autoregressive token generation** — Notes at each timestamp are generated left-to-right as a token sequence with beam search. This captures dependencies between simultaneous notes (e.g., don't put red and blue in the same position).

6. **Difficulty as conditioning** — One model serves all difficulties. More training data, better generalization, and the user just passes a flag.

7. **Windows native** — RTX 5090 CUDA via WSL is unreliable. Training runs directly on Windows with PyTorch nightly cu128.

## Useful References

- BSMG Wiki Map Format: https://bsmg.wiki/mapping/map-format.html
- BSMG Beatmap Spec: https://bsmg.wiki/mapping/map-format/beatmap.html
- BSMG Lightshow Spec: https://bsmg.wiki/mapping/map-format/lightshow.html
- BeatSaver API Docs: https://api.beatsaver.com/docs/
- ArcViewer (3D previewer): https://allpoland.github.io/ArcViewer/
- ArcViewer Desktop: https://github.com/AllPoland/ArcViewer/releases
- BS Map Check (error checker): https://kivalevan.me/BeatSaber-MapCheck/
- BS Map Inspector (parity): https://galaxymaster2.github.io/bs-parity/
- InfernoSaber (competing automapper): https://github.com/fred-brenner/InfernoSaber---BeatSaber-Automapper
- BeatLearning (transformer approach for osu!): https://github.com/sedthh/BeatLearning
- Original DeepSaber (historical reference only): https://github.com/oxai/deepsaber
