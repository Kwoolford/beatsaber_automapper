# Beat Saber Automapper — Progress Tracker

## Current Status: PR 1 Complete (Repo Scaffolding)

**Date:** 2026-02-16
**Branch:** main

## PR 1: Repo Scaffolding — DONE

All items complete and verified:

- [x] Full project directory structure per CLAUDE.md spec
- [x] `pyproject.toml` with all dependencies, CLI entrypoints, ruff/pytest config
- [x] `.gitignore` covering Python, data, IDE, checkpoints, experiment logs
- [x] Hydra config files: `configs/model/{audio_encoder,onset,sequence,lighting}.yaml`, `configs/data/default.yaml`, `configs/train.yaml`
- [x] All source modules with docstrings under `src/beatsaber_automapper/`
  - `data/`: download, dataset, audio, beatmap, tokenizer, augment
  - `models/`: components (with working SinusoidalPositionalEncoding), audio_encoder, onset_model, sequence_model, lighting_model
  - `training/`: onset_module, seq_module, light_module
  - `generation/`: generate, beam_search, export
  - `evaluation/`: metrics, playability
- [x] CLI scripts: `scripts/{download_data,preprocess,train,generate}.py`
- [x] Test files with placeholders + 3 real tests for SinusoidalPositionalEncoding
- [x] `uv sync` — installs successfully
- [x] `ruff check .` — all checks passed
- [x] `ruff format --check .` — all files formatted
- [x] `pytest` — 8/8 tests passed

### Notes for Next Session

- `uv sync` installs stable torch (2.10.0), not nightly cu128. After scaffolding work, restore nightly with:
  ```
  uv pip install --pre torch torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
  ```
- The `SinusoidalPositionalEncoding` in `models/components.py` is the only fully implemented component — everything else raises `NotImplementedError` with the target PR noted.
- `test.py` at repo root is empty from initial commit — can be deleted when committing.

## Next Up: PR 2 — Data Pipeline

What needs to happen:

1. **BeatSaver API client** (`data/download.py`): Paginated download with rate limiting, quality filters (rating ≥ 80%, NPS ≤ 20, post-2020, Expert/ExpertPlus required)
2. **Beatmap parsers** (`data/beatmap.py`): Parse `Info.dat` and difficulty `.dat` files, extract all v3 objects (colorNotes, bombNotes, obstacles, sliders, burstSliders, events)
3. **Audio preprocessing** (`data/audio.py`): Load audio to mono 44.1kHz, extract mel spectrograms (80 bands, 1024 FFT, 512 hop)
4. **Tokenizer** (`data/tokenizer.py`): Encode/decode beatmap events ↔ token sequences, must round-trip correctly
5. **Dataset classes** (`data/dataset.py`): PyTorch Datasets for each stage, DataLoader integration
6. **Preprocessing script** (`scripts/preprocess.py`): Full pipeline: download → parse → mel spectrogram → tokenize → cache as .pt
7. **Tests**: Tokenizer round-trip, parser correctness, dataset shape verification

Definition of done: DataLoader iteration produces correctly shaped batch tensors. Parser handles all v3 types. Tokenizer round-trips perfectly.

## PR Roadmap Reference

| PR | Status | Description |
|----|--------|-------------|
| 1  | **DONE** | Repo scaffolding |
| 2  | NEXT   | Data pipeline |
| 3  | —      | Audio encoder + Stage 1 (onset detection) |
| 4  | —      | Stage 2 (note sequence generation) |
| 5  | —      | End-to-end generation + export |
| 6  | —      | Stage 3 (lighting) |
| 7  | —      | Scale training + quality |
| 8  | —      | Documentation + demo |
