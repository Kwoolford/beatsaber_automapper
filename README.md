# Beat Saber Automapper

An open-source AI system that generates high-quality Beat Saber levels from audio files. Given a song, the system produces a playable `.zip` level package containing notes, arcs, chains, bombs, obstacles, and a synchronized light show — targeting the v3 Beat Saber map format.

## Quick Start

```bash
# Install dependencies
uv venv --python 3.12
uv pip install --pre torch torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
uv sync

# Download training data (mapper-cohort sweep — V5 default workflow)
python scripts/download_cohorts.py

# Preprocess one cohort
python scripts/preprocess.py --cohort joetastic --workers 8

# Train a single cohort
python scripts/train.py stage=sequence cohort=joetastic

# Run the auto-researcher over a queue of experiments
python scripts/auto_research.py experiments/queue/initial.yaml
python scripts/leaderboard.py

# Generate a level from the best checkpoint
bsa-generate song.mp3 --difficulty Expert --output level.zip
```

## V6 — Swing-First Architecture (current)

V6 keeps V5's cohort + harness infrastructure but replaces the modeling stack.
The Stage 2 sequence model no longer emits chords at timestamps — it emits a
single ordered stream of **per-hand swing events**. Parity becomes structural,
the model sees an explicit 12-dim saber state at every step, and a
phrase-energy loss + style discriminator replace the V4/V5 aux-loss stack.

Full rationale and phase plan in [`docs/architecture_v6_plan.md`](docs/architecture_v6_plan.md) and [`TODO.md`](TODO.md).

- **Swing-event tokenization.** `[HAND][Δt][KIND][X][Y][DIR][ANGLE]`. HAND ∈ {LEFT, RIGHT, NONE}. Vocab ~70.
- **Saber-state proprioception.** 12-dim physical state `(L_pos, L_dir, L_dt, L_parity, R_pos, R_dir, R_dt, R_parity)` projected to `d_model`, additive at every decode step.
- **Phrase conditioning.** Mean-pooled 16-bar audio window projected to `d_model`, additive. Phrase-energy KL aux loss against song RMS curve.
- **Style discriminator.** Pretrained mapper-classifier provides `−λ log p_D(this_mapper | swings)` once F1 ≥ 0.6 on held-out swings.
- **Deleted in V6:** `_compute_flow_loss`, `_compute_intra_onset_parity_loss`, `_compute_follow_through_loss`, `_compute_ergo_loss`. These bandaids existed only because the chord representation hid physics.

### Preserved from V5 (no change)

- **Cohorts.** 18 mappers grouped into 9 style buckets (`data/reference/mappers.json`). Each mapper's full catalog is downloaded and preprocessed independently under `data/cohorts/{mapper}/`.
- **Auto-researcher.** `scripts/auto_research.py` reads a YAML queue, trains a small model per spec with a wall-clock cap, generates a fixed reference song, and scores the output with a playability + style-closeness composite. Results land in `experiments/leaderboard.jsonl`.
- **Composite score.** Weighted mix of parity / collision / note-density / wall-sanity (60%) and direction-KL / NPS-gap / parity-gap / color-gap against the cohort reference (40%). V5 and V6 rows are directly comparable.
- **Iteration target.** 10+ experiments per overnight run; the V6 pilot queue (`experiments/queue/v6_pilot.yaml`) sweeps top-3 cohorts at small preset.

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
│              8 structure features: RMS, onset strength, bass/mid/high,      │
│              spectral centroid, section_id, section_progress                │
│              (librosa-derived; sections via self-similarity + clustering)   │
│                                                                             │
│  Output: [80, T] mel + [8, T] structure   (~100 frames per second)         │
└─────────────────────────────────────────────────────────────────────────────┘
                                  │
                                  │  [80, T] + [8, T]
                                  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                      SHARED AUDIO ENCODER                                   │
│                    (used by all 3 stages)                                   │
│                                                                             │
│  4-layer CNN frontend  →  Linear projection (d_model=512)                  │
│  + Structure projection: Linear(8→512), added to CNN output                │
│  → Sinusoidal positional encoding                                           │
│  → Transformer Encoder (6 layers, 8 heads)                                 │
│                                                                             │
│  Output: contextualized frame embeddings  [T, 512]                         │
│          (enriched with song energy/structure information)                  │
└─────────────────────────────────────────────────────────────────────────────┘
                                  │
              ┌───────────────────┼────────────────────┐
              ▼                   ▼                     ▼
        STAGE 1              STAGE 2 (V6)          STAGE 3
    ONSET DETECTION      SWING-EVENT GEN       LIGHTING GENERATION
    ───────────────      ───────────────       ───────────────────
    Inputs:              Inputs:               Inputs:
     audio embeddings     audio embeddings      audio embeddings
     difficulty emb       difficulty emb        difficulty emb
     genre emb            genre emb             genre emb
                          mapper_id emb (cohort) beat grid timestamps
                          saber state [12-dim]  slot embedding (4-pos)
                          phrase emb (16-bar)   (RULE-BASED in V5+)
                          (legacy: onset times
                           guide event ordering)

    Arch:                Arch:                 Arch:
     6-block TCN          Transformer decoder   Rule-based engine
     + 2-layer Xfmr       8 layers, 8 heads      energy → events
     → Linear(1)          causal self-attn       chroma palette
     → sigmoid            cross-attn → audio     by section
                          + Linear(12→512)
                            saber-state proj
                          + Linear(d→d)
                            phrase proj

    Loss:                Loss:                 Loss:
     BCE (Gaussian        CE (rhythm-weighted)   n/a (rules)
     smoothed labels)     + phrase-energy KL
                          + style-disc (opt.,
                            once D F1 ≥ 0.6)
                          --- deleted in V6 ---
                          flow / intra-onset
                          parity / follow-thru
                          / ergo aux losses

    Output:              Output:               Output:
     per-frame onset      swing-event stream     lighting events
     probability [T]      [HAND][Δt][KIND]       (deterministic +
     → peak picking       [X][Y][DIR][ANGLE]      Chroma RGB colors)
     → onset timestamps   (beam search; saber
                           state recomputed
                           per step)

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

### Conditioning Inputs (V6)

Every model stage is conditioned on:

| Input | Type | Values | Effect |
|-------|------|--------|--------|
| `difficulty` | learned embedding (5→512, additive) | Easy / Normal / Hard / Expert / ExpertPlus | Controls note density and pattern complexity |
| `genre` | learned embedding (11→512, additive) | electronic, rock, pop, anime, hip-hop, classical, jazz, country, video-game, other, unknown | Shapes map style and feel |
| `structure` | linear projection (8→512, additive) | RMS energy, onset strength, bass/mid/high energy, spectral centroid, section_id, section_progress | Per-frame song energy + section awareness |
| `mapper_id` *(Stage 2, cohort training)* | learned embedding (18→512, additive) | One of 18 cohorts in `data/reference/mappers.json` | Style cohort conditioning |
| `saber_state` *(Stage 2, V6)* | `Linear(12→512)`, additive at every decode step | `(L_x, L_y, L_dx, L_dy, L_dt, L_parity, R_x, R_y, R_dx, R_dy, R_dt, R_parity)` | Physical proprioception — model knows where both sabers are |
| `phrase_emb` *(Stage 2, V6)* | mean-pool of 16-bar audio window → `Linear(d→d)`, additive | Pooled audio encoder output | Phrase-level musical context (replaces section_id as a *summary* signal) |
| `slot_emb` *(Stage 3 ML path, retained but rule-based ships)* | learned embedding | 4-position cycling (type/ET/VAL/BRIGHT) | Structural grammar enforcement for lighting events |

**Removed in V6:**
- `prev_tokens` (mean-pooled K=8 previous onset sequences) — replaced by `saber_state`.
- `plan_vector` (bidirectional song-level planner) — saber state subsumes its role for Stage 2 small preset. Kept wired but disabled by default.

### Token Vocabulary (Stage 2 — V6 Swing-Event Stream, ~70 tokens)

V6 emits one **swing event** per cut, ordered globally by time. A chord is just two swing events with `Δt=0`. Full grammar in [`docs/swing_event_grammar.md`](docs/swing_event_grammar.md).

```
SwingEvent := [HAND] [Δt_bin] [KIND] [GRID_X] [GRID_Y] [DIR] [ANGLE]

HAND:    LEFT (red saber), RIGHT (blue saber), NONE (bomb/wall)
Δt_bin:  beats since previous event, quantized (1/16-beat resolution + log tail)
KIND:    NOTE | ARC_HEAD | ARC_TAIL | CHAIN_HEAD | CHAIN_TAIL | BOMB | WALL
GRID_X:  column 0–3 (left to right)
GRID_Y:  row 0–2 (bottom to top)
DIR:     0=up 1=down 2=left 3=right 4=up-left 5=up-right
         6=down-left 7=down-right 8=any
ANGLE:   quantized angle offset bin (15° steps)

Example chord (red bottom-row down + blue bottom-row up, simultaneous):
   LEFT  Δt=0   NOTE  X=1 Y=0 DIR=down  ANGLE=0
   RIGHT Δt=0   NOTE  X=2 Y=0 DIR=up    ANGLE=0
```

**Parity is structural under this grammar:** consecutive same-`HAND` events alternate forehand/backhand by construction of the data. Chains and arcs self-connect via the next same-`HAND` `*_TAIL` event. The V4/V5 auxiliary parity / follow-through / ergo losses are deleted, not migrated.

The legacy V5 chord-grammar tokenizer (183 tokens, `data/tokenizer.py::BeatmapTokenizer`) is retained for backward-compatible decoding of pre-V6 checkpoints and for evaluation tooling.

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
│   └── research/                # Auto-researcher: spec, runner, metrics, leaderboard
├── scripts/                     # CLI entry points (download_cohorts, auto_research, ...)
├── experiments/                 # Queue YAMLs, per-run artifacts, leaderboard.jsonl
└── tests/                       # pytest test suite (241 tests)
```

## Tech Stack

- **Python 3.12** · **PyTorch nightly cu128** (RTX 5090 / sm_120 support)
- **Lightning** for training · **Hydra** for config · **librosa** for BPM detection
- **soundfile** for audio I/O · **ruff** for linting · **pytest** for tests

## References

- [BSMG Map Format](https://bsmg.wiki/mapping/map-format.html)
- [BeatSaver API](https://api.beatsaver.com/docs/)
- [ArcViewer (3D preview)](https://allpoland.github.io/ArcViewer/)
