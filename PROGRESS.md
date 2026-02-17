# Beat Saber Automapper — Progress Tracker

## Current Status: PR 2 Complete (Data Pipeline)

**Date:** 2026-02-17
**Branch:** main

## PR 2: Data Pipeline — DONE

All items complete and verified:

- [x] **Beatmap parser** (`data/beatmap.py`): Dataclasses for all v3 types (ColorNote, BombNote, Obstacle, Slider, BurstSlider, BasicEvent, ColorBoostEvent). File-based and in-memory JSON parsers. v2 detection returns None with warning.
- [x] **Tokenizer** (`data/tokenizer.py`): 167-token vocabulary covering all event types. Sliders split into ARC_START/ARC_END at head/tail beats. Canonical ordering (type priority → x → y). Quantization for angle offset, mu, squish, wall duration. Round-trip guarantee.
- [x] **Audio processing** (`data/audio.py`): Uses soundfile for I/O (avoids torchcodec dep), torchaudio transforms for resampling and mel spectrogram. beat_to_frame/frame_to_beat utilities.
- [x] **Datasets** (`data/dataset.py`): OnsetDataset (sliding windows + Gaussian-smoothed labels), SequenceDataset (per-onset context windows + padded tokens). Both support train/val/test splits and difficulty filtering.
- [x] **Download client** (`data/download.py`): BeatSaver API paginated search, quality filters (rating, NPS, year, difficulty), CDN download with atomic writes, resume support, rate limiting, 429 backoff.
- [x] **Preprocessing script** (`scripts/preprocess.py`): Processes .zip → .pt with mel spectrograms, tokenized events, Gaussian-smoothed onset labels. Deterministic hash-based splits (85/10/5).
- [x] **Exports** (`data/__init__.py`): Clean public API.
- [x] `.gitignore` fixed: was blocking `src/.../data/` files (changed `data/` → `/data/`)
- [x] `ruff check .` — all checks passed
- [x] `ruff format --check .` — all files formatted
- [x] `pytest` — 56/56 tests passed (13 parser, 17 tokenizer, 11 audio, 11 dataset, 3 components, 1 generation placeholder)

### Key Decisions

- **soundfile instead of torchaudio for I/O**: PyTorch nightly's torchaudio now requires torchcodec for load/save. soundfile is simpler, already a dependency, and sufficient.
- **Tokenizer vocab = 167**: Special(4) + Events(6) + Color(2) + Col(4) + Row(3) + Dir(9) + Angle(7) + Mu(9) + MidAnchor(3) + Slice(31) + Squish(11) + Width(4) + Height(5) + DurInt(65) + DurFrac(4).
- **Gaussian-smoothed onset labels**: σ=3.0 frames, ±4σ window for efficiency. Better than hard binary labels for training onset detection.

### Notes for Next Session

- Download client is untested against live API (no network tests). First real run: `bsa-download --output data/raw --count 500`
- After downloading, preprocess with: `bsa-preprocess --input data/raw --output data/processed`
- LightingDataset stays stub until PR 6
- `data/augment.py` stays stub until PR 7

## PR 1: Repo Scaffolding — DONE

**Date:** 2026-02-16

- Full project directory structure per CLAUDE.md spec
- `pyproject.toml` with all dependencies, CLI entrypoints, ruff/pytest config
- Hydra config files, all source modules with docstrings
- `SinusoidalPositionalEncoding` in `models/components.py` is only non-stub model code
- 8/8 tests passed

## Next Up: PR 3 — Audio Encoder + Stage 1 (Onset Detection)

What needs to happen:

1. **Audio encoder** (`models/audio_encoder.py`): 2D CNN frontend (3-4 conv layers) → sinusoidal positional encoding → Transformer encoder (6-8 layers, 8 heads, d_model=512)
2. **Onset model** (`models/onset_model.py`): Audio encoder output → 2-layer transformer decoder (with difficulty embedding) → linear → sigmoid
3. **Lightning module** (`training/onset_module.py`): Binary cross-entropy loss, onset F1 metric, LR scheduling
4. **Training script** integration with Hydra configs
5. **Evaluation**: Onset F1 > 0.6 on validation set

Definition of done: Onset F1 > 0.6 on validation set. Can visualize predicted onsets overlaid on audio waveform.

## PR Roadmap Reference

| PR | Status | Description |
|----|--------|-------------|
| 1  | **DONE** | Repo scaffolding |
| 2  | **DONE** | Data pipeline |
| 3  | NEXT   | Audio encoder + Stage 1 (onset detection) |
| 4  | —      | Stage 2 (note sequence generation) |
| 5  | —      | End-to-end generation + export |
| 6  | —      | Stage 3 (lighting) |
| 7  | —      | Scale training + quality |
| 8  | —      | Documentation + demo |
