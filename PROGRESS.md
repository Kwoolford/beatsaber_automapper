# Beat Saber Automapper — Progress Tracker

## Current Status: PR 6 Complete (Stage 3 — Lighting Generation)

**Date:** 2026-02-18
**Branch:** main

## PR 6: Stage 3 (Lighting Generation) — DONE

All items complete and verified:

- [x] **LightingTokenizer** (`data/tokenizer.py`): LIGHT_VOCAB_SIZE=35. Special tokens (PAD=0, EOS=1, SEP=2, BOS=3), event type tokens (BASIC=4, BOOST=5), attribute ranges: ET (6–20, 15 types), VAL (21–28, 8 values), BRIGHT (29–32, 4 brightness bins), ONOFF (33–34). `encode_lighting()` groups events by beat, SEP-separated, EOS-terminated. `decode_lighting()` is bounds-checked + clamped for robustness.
- [x] **LightingModel** (`models/lighting_model.py`): Light token embedding (scaled by √d_model) + SinusoidalPositionalEncoding + note context (mean-pool non-PAD note embeddings → Linear → add to each decoder position) → nn.TransformerDecoder (causal self-attn + cross-attn to audio) → LayerNorm → Linear(d_model, light_vocab_size). `forward()` and `decode_step()` methods.
- [x] **LightingLitModule** (`training/light_module.py`): Same pattern as SequenceLitModule. LIGHT_BOS teacher-forcing prepend. CrossEntropyLoss(ignore_index=LIGHT_PAD, label_smoothing=0.1). Logs train_loss, val_loss, val_token_acc. AdamW + linear warmup + cosine decay. `freeze_encoder` flag.
- [x] **LightingDataset** (`data/dataset.py`): Per-beat samples. Each sample: mel context window + nearest-onset note_tokens + light_tokens + difficulty. Expects `light_frames` and `light_token_sequences` in each difficulty's .pt data.
- [x] **preprocess.py update**: Runs LightingTokenizer on each beatmap, converts light beat→frame, stores `light_frames` + `light_token_sequences` in each difficulty's .pt output.
- [x] **train.py update**: `_build_lighting(cfg)` function + `stage=lighting` dispatch. Replaces prior `NotImplementedError`.
- [x] **Config** (`configs/model/lighting.yaml`): d_model=512, nhead=8, num_layers=4, dim_feedforward=2048, light_vocab_size=35, note_vocab_size=167, context_frames=128, max_note_len=64, max_light_len=32, label_smoothing=0.1, freeze_encoder=false.
- [x] **Stage 3 integration in generate.py**: `generate_lighting_events()` greedy decoder. `generate_level()` runs lighting on regular beat grid (lighting_beats_per_bar=2), uses nearest-onset note tokens for conditioning, extends beatmap.basic_events and color_boost_events before export.
- [x] **Exports**: `models/__init__.py` exports `LightingModel`. `training/__init__.py` exports `LightingLitModule`.
- [x] **Tests** (`tests/test_lighting_tokenizer.py`, `tests/test_lighting_model.py`): 35 new tests — all pass.
- [x] `ruff check .` — all checks passed.
- [x] `pytest` — 176/176 tests passed (35 new + 141 prior).

### Key Decisions

- **Note context as additive mean-pool**: Note tokens are embedded and mean-pooled into a single vector, added to every lighting decoder position. This avoids variable-length memory complexity while still conditioning lighting on note events.
- **Beat-grid lighting**: Lighting is generated on a regular beat grid (every 0.5 beats by default) rather than only at note onsets, so the light show covers the whole song.
- **Nearest-onset note context**: For each lighting beat, the nearest note-onset's token sequence is used as note conditioning — simple and avoids gaps when no notes are nearby.
- **Greedy decoding for lighting**: Lighting is less structured than note sequences (no canonical ordering, no parity constraints), so greedy decoding with temperature is sufficient. Beam search could be added later.
- **LIGHT_VOCAB_SIZE=35**: Covers BasicEvent (et 0–14, val 0–7, brightness 4 bins) + ColorBoostEvent (on/off) with tight vocabulary.

### Notes for Next Session

- To train lighting: `python scripts/train.py stage=lighting data_dir=data/processed` (after onset + sequence models are trained)
- To generate with lighting: `python scripts/generate.py song.mp3 --lighting-ckpt lighting.ckpt`
- All three stages are now fully implemented; next is training + quality evaluation

## PR 5: End-to-End Generation + Export — DONE

All items complete and verified:

- [x] **Export pipeline** (`generation/export.py`):
  - `beatmap_to_v3_dict()`: `DifficultyBeatmap` → v3 JSON dict (all object types)
  - `build_info_dat()`: builds `Info.dat` dict for any set of difficulties
  - `tokens_to_beatmap()`: wrapper around `BeatmapTokenizer.decode_beatmap()`
  - `package_level()`: packs `{difficulty: DifficultyBeatmap}` + audio + optional cover → `.zip`
- [x] **Full pipeline** (`generation/generate.py`):
  - `generate_level()`: audio → mel → AudioEncoder → OnsetModel → beam search → export
  - `predict_onsets()`: runs Stage 1 and peak-picks frame indices
  - `generate_note_sequence()`: beam search or nucleus sampling for a single onset context
  - Supports checkpoint loading or untrained random weights for testing
  - Auto-detects CUDA; accepts `device=` override
- [x] **CLI** (`scripts/generate.py`): full argparse CLI with all inference options
  - `python scripts/generate.py song.mp3 --difficulty Expert --output level.zip`
  - `--onset-ckpt` / `--seq-ckpt` for trained checkpoints
  - `--nucleus-sampling`, `--beam-size`, `--temperature`, `--top-p`
  - `--bpm`, `--song-name`, `--song-author`
- [x] **Bug fix** (`data/tokenizer.py`): Added bounds checks in `decode_beatmap()` for all event
  types (NOTE=6, BOMB=3, WALL=7, ARC_START=6, ARC_END=7, CHAIN=9 tokens minimum).
  Prevents `IndexError` on malformed/truncated token sequences from random models.
- [x] **Exports** (`generation/__init__.py`): exports `generate_level`, `beatmap_to_v3_dict`,
  `build_info_dat`, `package_level`, `tokens_to_beatmap`.
- [x] **Tests** (`tests/test_export.py`, `tests/test_generate.py`): 38 new tests — all pass
- [x] `ruff check .` — all checks passed
- [x] `pytest` — 141/141 tests passed (38 new + 103 prior)

  Also fixed two robustness bugs in `data/tokenizer.py` (found by testing with random model weights):
  - Added `_clamp()` helper so `_dequantize_*` functions never crash on out-of-range bin indices
  - Added `remaining < N` bounds checks before each event-type token consumption

### Key Decisions

- **Single-difficulty per call**: `generate_level()` generates one difficulty at a time; call
  multiple times with same audio for a multi-difficulty pack.
- **Audio encoded once**: `full_audio_features` is computed once; context windows are sliced
  per onset to avoid redundant encoder forward passes.
- **EOS appended in generate.py**: `decode_beatmap` expects EOS at end of each beat's token
  list; the pipeline appends it since beam search/sampling strips EOS from output.
- **Graceful decode on malformed tokens**: truncated token sequences (from untrained models or
  errors) now break cleanly rather than crashing with IndexError.
- **BPM defaults to 120**: No automatic BPM detection — caller must pass `bpm=` for real songs.
  This is intentional; BPM detection is a separate concern.

### Notes for Next Session

- To generate with trained models: `python scripts/generate.py song.mp3 --onset-ckpt onset.ckpt --seq-ckpt seq.ckpt`
- To generate with random weights (for testing structure): `python scripts/generate.py song.wav --bpm 120`
- Generated `.zip` loads in ArcViewer but notes will be random until models are trained
- Next step: train models on real data (PR 2 pipeline needed), then quality eval in ArcViewer

## PR 4: Stage 2 (Note Sequence Generation) — DONE

All items complete and verified:

- [x] **Sequence model** (`models/sequence_model.py`): Token embedding (scaled by √d_model, PAD=0 zeroed) + SinusoidalPositionalEncoding + difficulty embedding (additive) → nn.TransformerDecoder (8 layers, 8 heads, d_model=512, norm_first=True) with causal self-attention + cross-attention to audio → LayerNorm → Linear(d_model, vocab_size). `forward()` for teacher forcing, `decode_step()` for autoregressive inference (returns last-position logits).
- [x] **Beam search** (`generation/beam_search.py`): `beam_search_decode()` with length-normalized log probability scoring, configurable beam_size/temperature. `nucleus_sampling_decode()` with top-p filtering for creative diversity. Both strip BOS/EOS from output.
- [x] **Lightning module** (`training/seq_module.py`): SequenceLitModule wrapping AudioEncoder + SequenceModel. Teacher forcing with BOS prepend. CrossEntropyLoss(ignore_index=PAD, label_smoothing=0.1). Logs train_loss, val_loss, val_token_acc, val_eos_acc. AdamW + linear warmup + cosine decay. Optional freeze_encoder flag.
- [x] **Training CLI** (`scripts/train.py`): `stage=sequence` dispatch via `_build_sequence()`. Uses SequenceDataset with context_frames and max_seq_length from config. ModelCheckpoint(monitor=val_loss, mode=min), EarlyStopping(patience=10).
- [x] **Config updates**: `sequence.yaml` — vocab_size=167 (matches VOCAB_SIZE), added context_frames=128, label_smoothing=0.1, freeze_encoder=false. `train.yaml` — added `model/sequence` to defaults.
- [x] **Metrics** (`evaluation/metrics.py`): Added `token_accuracy()` utility for per-token accuracy ignoring PAD.
- [x] **Exports**: models/__init__.py exports SequenceModel. training/__init__.py exports SequenceLitModule. generation/__init__.py exports beam_search_decode, nucleus_sampling_decode.
- [x] `ruff check .` — all checks passed
- [x] `ruff format --check .` — all files formatted
- [x] `pytest` — 103/103 tests passed (7 sequence_model, 5 seq_module, 9 beam_search, 82 existing)

### Key Decisions

- **BOS prepend in Lightning module, not dataset**: Dataset provides raw tokens; shifting logic is training-specific.
- **CrossEntropyLoss with label_smoothing=0.1**: Prevents overconfident predictions; helps creative generation.
- **ignore_index=PAD in loss**: Padded positions don't contribute to gradients.
- **Difficulty as additive embedding**: Consistent with OnsetModel pattern.
- **decode_step returns last-position logits only**: Efficient for autoregressive inference.
- **Length-normalized log prob in beam search**: Prevents bias toward shorter sequences.
- **Nucleus sampling alongside beam search**: Better diversity for creative tasks.
- **freeze_encoder option**: Can load pre-trained Stage 1 encoder and freeze during Stage 2.
- **vocab_size=167**: Config was wrong at 256; matches tokenizer.VOCAB_SIZE.

### Notes for Next Session

- To train: `python scripts/train.py stage=sequence data_dir=data/processed`
- Need data from PR 2 pipeline first
- Definition of done for quality: Generated .dat files pass BS Map Check without errors
- Beam search produces coherent, non-random patterns (visual inspection needed)

## PR 3: Audio Encoder + Stage 1 — DONE

**Date:** 2026-02-17

All items complete and verified:

- [x] **Audio encoder** (`models/audio_encoder.py`): 4-layer CNN frontend (stride=(2,1) on freq, preserves time) → Linear projection → SinusoidalPositionalEncoding → 6-layer Transformer encoder. Input: `[B, n_mels, T]` → Output: `[B, T, d_model]`. Requires n_mels divisible by 16.
- [x] **Onset model** (`models/onset_model.py`): Difficulty embedding (5 levels, additive) → 2-layer Transformer encoder → LayerNorm → Linear(d_model, 1). Outputs raw logits (no sigmoid) for BCEWithLogitsLoss.
- [x] **Peak picking** (`models/components.py`): peak_picking() utility — threshold + local maxima + greedy distance suppression.
- [x] **Onset F1 metrics** (`evaluation/metrics.py`): onset_f1() for time-based matching, onset_f1_framewise() for frame-index validation loop use. Greedy matching (mir_eval approach).
- [x] **Lightning module** (`training/onset_module.py`): OnsetLitModule wrapping AudioEncoder + OnsetModel. BCEWithLogitsLoss(pos_weight=5.0). Training logs train_loss. Validation computes val_loss, val_f1, val_precision, val_recall via peak_picking + onset_f1_framewise. AdamW + linear warmup + cosine decay.
- [x] **Training CLI** (`scripts/train.py`): Hydra CLI with stage dispatch. Onset stage: builds OnsetDataset + OnsetLitModule, ModelCheckpoint(monitor=val_f1, mode=max), EarlyStopping(patience=10), LearningRateMonitor, TensorBoard/wandb logger.
- [x] **Config updates**: onset.yaml gains pos_weight, window_size, hop, min_onset_distance_frames. train.yaml checkpoint now monitors val_f1 (mode=max).
- [x] **Exports**: models/__init__.py exports AudioEncoder, OnsetModel, peak_picking. training/__init__.py exports OnsetLitModule.
- [x] 82/82 tests passed

## PR 2: Data Pipeline — DONE

**Date:** 2026-02-17

All items complete and verified:

- [x] **Beatmap parser** (`data/beatmap.py`): Dataclasses for all v3 types (ColorNote, BombNote, Obstacle, Slider, BurstSlider, BasicEvent, ColorBoostEvent). File-based and in-memory JSON parsers. v2 detection returns None with warning.
- [x] **Tokenizer** (`data/tokenizer.py`): 167-token vocabulary covering all event types. Sliders split into ARC_START/ARC_END at head/tail beats. Canonical ordering (type priority → x → y). Quantization for angle offset, mu, squish, wall duration. Round-trip guarantee.
- [x] **Audio processing** (`data/audio.py`): Uses soundfile for I/O (avoids torchcodec dep), torchaudio transforms for resampling and mel spectrogram. beat_to_frame/frame_to_beat utilities.
- [x] **Datasets** (`data/dataset.py`): OnsetDataset (sliding windows + Gaussian-smoothed labels), SequenceDataset (per-onset context windows + padded tokens). Both support train/val/test splits and difficulty filtering.
- [x] **Download client** (`data/download.py`): BeatSaver API paginated search, quality filters (rating, NPS, year, difficulty), CDN download with atomic writes, resume support, rate limiting, 429 backoff.
- [x] **Preprocessing script** (`scripts/preprocess.py`): Processes .zip → .pt with mel spectrograms, tokenized events, Gaussian-smoothed onset labels. Deterministic hash-based splits (85/10/5).
- [x] **Exports** (`data/__init__.py`): Clean public API.
- [x] 56/56 tests passed

## PR 1: Repo Scaffolding — DONE

**Date:** 2026-02-16

- Full project directory structure per CLAUDE.md spec
- `pyproject.toml` with all dependencies, CLI entrypoints, ruff/pytest config
- Hydra config files, all source modules with docstrings
- `SinusoidalPositionalEncoding` in `models/components.py` is only non-stub model code
- 8/8 tests passed

## PR 7: Scale Training + Quality — IN PROGRESS

**Date started:** 2026-02-19

### Genre tag conditioning (2026-02-20)

Added genre as a second conditioning signal alongside difficulty, wired through the full pipeline.

- [x] **`data/tokenizer.py`**: `GENRE_MAP` (11 classes: unknown=0, electronic, rock, pop, anime, hip-hop, classical, jazz, country, video-game, other), `NUM_GENRES=11`, `_GENRE_TAG_MAP`, `genre_from_tags()`.
- [x] **`data/download.py`**: `_extract_genre_tags()` reads BeatSaver API tag list. Manifest entries now include `genre_tags: list[str]` and `genre: str`. Backfilled entries default to `genre_tags=[]`, `genre="unknown"`.
- [x] **`scripts/preprocess.py`**: Reads `genre` from manifest; stores in `mod_requirements.genre` in every `.pt` file.
- [x] **`data/dataset.py`**: All three dataset classes (`OnsetDataset`, `SequenceDataset`, `LightingDataset`) now include `genre_idx` in their samples tuple and return `"genre": torch.tensor(genre_idx)` in each batch item.
- [x] **`models/onset_model.py`**: `genre_emb = nn.Embedding(num_genres, d_model)`, added additively to audio features. `forward(audio_features, difficulty, genre)`.
- [x] **`models/sequence_model.py`**: Same pattern — `genre_emb` added additively. `forward()` and `decode_step()` both accept `genre`.
- [x] **`models/lighting_model.py`**: Same pattern. `forward()` and `decode_step()` both accept `genre`.
- [x] **`generation/beam_search.py`**: `beam_search_decode()` and `nucleus_sampling_decode()` both accept `genre: torch.Tensor`.
- [x] **Training modules** (`onset_module.py`, `seq_module.py`, `light_module.py`): All accept `*_num_genres: int = 11` param, thread genre through forward/training/validation.
- [x] **`generation/generate.py`**: `generate_level()` accepts `genre: str = "unknown"`, converts to index via `GENRE_MAP`, passes as tensor through all three stages.
- [x] **`scripts/generate.py`**: `--genre` CLI arg with choices from GENRE_MAP keys.
- [x] **`configs/model/`**: `num_genres: 11` added to `onset.yaml`, `sequence.yaml`, `lighting.yaml`.
- [x] **Tests**: All test files updated — model fixtures gain `num_genres=11`, all forward/decode_step calls pass `genre` tensor, training batches include `"genre"` key. 3 new genre dataset tests.
- [x] `ruff check .` — all checks passed
- [x] `pytest` — 182/182 tests passed (6 new + 176 prior)

### Modding framework quotas + preprocessor tagging (2026-02-20)

Added per-category download quotas and mod_requirements tagging to support
clean separation of vanilla vs modded maps in the training pipeline.

- [x] **`download.py`**: New `_classify_map_api()` (pre-download, from API booleans), `_classify_map_zip()` (post-download, from Info.dat customData), `_load_manifest()`, `_save_manifest()` (atomic write). `download_maps()` now accepts `quotas: dict[str, int | None]` and maintains `data/raw/manifest.json` tracking every map's category, requirements, suggestions, and download timestamp. Existing 5k zips are backfilled on first run.
- [x] **`scripts/download_data.py`**: `--quota category:N` (repeatable) replaces `--count` as primary interface. `--count` kept as legacy fallback. Example: `bsa-download --quota vanilla:10000 --quota chroma:2000 --quota noodle:1000`
- [x] **`scripts/preprocess.py`**: Loads manifest at start; passes `manifest_entry` to `preprocess_single()`; embeds `mod_requirements: {category, requirements, suggestions}` in every `.pt` file. `--exclude-categories` CLI arg to skip entire categories during preprocessing.
- [x] **`data/dataset.py`**: `exclude_categories: list[str] | None = None` added to `OnsetDataset`, `SequenceDataset`, and `LightingDataset`. Category check (`mod_requirements.category`) applied during index construction (not at `__getitem__` time). Missing `mod_requirements` defaults to `"vanilla"`.
- [x] **`tests/test_dataset.py`**: `_make_test_pt()` updated with `category` param + `mod_requirements` in saved data. Three new tests: `test_onset_dataset_excludes_category`, `test_sequence_dataset_excludes_category`, `test_onset_dataset_excludes_unknown_category`.
- [x] `ruff check .` — all checks passed
- [x] `pytest` — 179/179 tests passed (3 new + 176 prior)

**Quota strategy for next download run:**
```
bsa-download --quota vanilla:10000 --quota chroma:2000 --quota noodle:1000 --min-rating 0.8 --min-year 2022
```
vivify and mapping_extensions are opportunistic (no cap). Existing 5k zips count toward quotas after backfill. Expected total: ~13k maps.

**Categories:**
- `vanilla` — no mod requirements
- `chroma` — Chroma in requirements/suggestions
- `noodle` — Noodle Extensions required
- `mapping_extensions` — Mapping Extensions required
- `vivify` — Vivify in requirements/suggestions (highest priority)
- `unknown` — no manifest entry (pre-backfill maps)

### Download client fixes (2026-02-19)

Three bugs found and fixed in `data/download.py` while running first real download:

- [x] **API URL fix**: BeatSaver dropped the `/api/` prefix — endpoint is now `/search/text/{page}`, not `/api/search/text/{page}`. Was returning 404 silently.
- [x] **`declaredAi` type bug**: API returns string `"None"` (not JSON `null`) for human-made maps. Comparing truthiness flagged every map as AI-generated, downloading 0 maps.
- [x] **NPS filter scope**: Was rejecting maps if any diff exceeded max_nps (including Easy). Now only enforces cap on Expert/ExpertPlus diffs.

### Difficulty filter expansion (2026-02-19)

- [x] **Accept all Standard difficulties**: Removed `require_difficulties=["Expert","ExpertPlus"]` default. Now accepts Easy/Normal/Hard/ExpertPlus as long as map has ≥1 Standard characteristic diff.
- [x] **Characteristic filter**: Require `characteristic=Standard` — excludes 360Degree, OneSaber, Lightshow, Lawless, etc. which would be noise for our Standard map generator.
- [x] **AI exclusion**: Added `exclude_ai=True` (default) using `automapper` + `declaredAi` API fields. Prevents training on AI-generated maps.
- [x] **`min_year` default**: 2020 → 2022 (v3 format era, avoiding v2 maps that get skipped in preprocessing anyway).

### Data collection status

- [x] **Full download**: 14,492 maps in `data/raw/` — exhausted full BeatSaver catalog under filters (≥80% rating, post-2022, Standard characteristic, no AI maps). Final category counts: vanilla=10,432, chroma=3,122, noodle=777, mapping_extensions=112, vivify=49. Manifest at `data/raw/manifest.json`.
- [ ] **Preprocess**: `python scripts/preprocess.py --input data/raw --output data/processed`
- [ ] **Train onset model**: `python scripts/train.py stage=onset data_dir=data/processed`
- [ ] **Train sequence model**: `python scripts/train.py stage=sequence data_dir=data/processed`
- [ ] **Train lighting model**: `python scripts/train.py stage=lighting data_dir=data/processed`
- [ ] **Generate + evaluate**: `python scripts/generate.py song.mp3 --onset-ckpt ... --seq-ckpt ... --lighting-ckpt ...`
- [ ] **Preview in ArcViewer**, check with BS Map Check, compute onset F1 and token accuracy

### Notes for next session

- Download is running — check `data/raw/` for .zip files; resume support means rerunning skips already-downloaded maps
- Model weights will go to HuggingFace Hub (PR 8); training data stays local (reproducible from BeatSaver public API)
- After download: run preprocess, then train all 3 stages in order

## PR Roadmap Reference

| PR | Status | Description |
|----|--------|-------------|
| 1  | **DONE** | Repo scaffolding |
| 2  | **DONE** | Data pipeline |
| 3  | **DONE** | Audio encoder + Stage 1 (onset detection) |
| 4  | **DONE** | Stage 2 (note sequence generation) |
| 5  | **DONE** | End-to-end generation + export |
| 6  | **DONE** | Stage 3 (lighting) |
| 7  | —      | Scale training + quality |
| 8  | —      | Documentation + demo |
