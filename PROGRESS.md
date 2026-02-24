# Beat Saber Automapper — Progress Tracker

## Current Status: PR 7 In Progress — Overnight Pipeline Running

**Date:** 2026-02-23 late night
**Branch:** main

### Session Handoff (2026-02-23 ~11:15 PM)

**Overnight Pipeline:** PID 31384, fully detached, will run onset → sequence → lighting
sequentially. Each stage: max_epochs=100, EarlyStopping(patience=10).
- Pipeline log: `logs/overnight_pipeline.log`
- Per-stage logs: `logs/train_{onset,sequence,lighting}_full.log`
- TensorBoard: `outputs/beatsaber_automapper/version_27/` (onset)
- GPU: 97% utilization, 12 GB VRAM, batch_size=32, 12 workers
- Dataset: 431,720 train / 50,651 val (Expert + ExpertPlus only)
- Blacklist: 1,324 maps excluded (647 modded, 642 no expert, 35 short)

**All P0 fixes applied:**
1. `pos_weight` 5.0 → 1.0 (onset.yaml + onset_module.py default)
2. `window_size` 256 → 1024, `hop` 128 → 512 (onset.yaml) — 12s context vs 3s
3. `num_genres` 11 → 1 (onset/sequence/lighting.yaml) — all maps are "unknown"
4. Windowed onset inference: `predict_onsets()` slides 1024-frame windows with overlap
   averaging — eliminates train/inference mismatch
5. Post-processing pipeline: `generation/postprocess.py` with 6 steps (NPS enforcement,
   color rebalancing, direction diversity, grid coverage, pattern dedup, parity fixing)
6. Architecture research saved to `docs/architecture_v2.md` for future pivots
7. Gaussian sigma 3 → 2 (sharper onset peaks)
8. onset_threshold 0.5 → 0.35 (model is conservative, high precision low recall)
9. Difficulty filtering: Expert + ExpertPlus only for onset AND sequence stages
10. Data blacklisting: 1,324 maps excluded (noodle/ME, no expert, short songs)
11. 205 tests pass (17 new postprocess tests)

**Lighting events NOT yet generated** — requires a trained lighting checkpoint.
The overnight pipeline will train lighting as Stage 3 after onset and sequence complete.
Once we have `--lighting-ckpt`, ArcViewer will show light events.

**Next actions (future session):**
- Check overnight training results — look at TensorBoard version_27+
- Run `evaluate_reference.py` with best checkpoints from overnight run
- Compare against baseline snapshot (`data/reference/snapshots/reference_20260223_180304.zip`)
- If onset val_f1 > 0.4: success, proceed to quality tuning
- If onset val_f1 < 0.3: consider Phase 2 (curated gold dataset) or Phase 3 (architecture)

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
- [x] **Training pipeline fixes** (2026-02-20):
  - Fixed Hydra config nesting: `# @package model.{name}` in each YAML so `cfg.model.audio_encoder` etc. resolve correctly
  - Fixed NaN loss: switched `precision: 16-mixed` → `bf16-mixed`, added `gradient_clip_val=1.0` to all Trainers
  - Added `torch.set_float32_matmul_precision("high")` for Blackwell Tensor Core hint
  - Wired `num_genres=11` through all three `_build_*()` functions in `train.py`
  - Smoke-test results: onset val_f1=0.248 after 3 epochs; sequence loss 5.3 (not NaN); lighting loss 3.6 (not NaN)
- [~] **Preprocess**: Running — `python scripts/preprocess.py --input data/raw --output data/processed` (~2 hrs, ~2 maps/s)
- [ ] **Train onset model**: `python scripts/train.py stage=onset data_dir=data/processed`
- [ ] **Train sequence model**: `python scripts/train.py stage=sequence data_dir=data/processed`
- [ ] **Train lighting model**: `python scripts/train.py stage=lighting data_dir=data/processed`
- [ ] **Generate + evaluate**: `python scripts/generate.py song.mp3 --onset-ckpt ... --seq-ckpt ... --lighting-ckpt ...`
- [ ] **Preview in ArcViewer**, check with BS Map Check, compute onset F1 and token accuracy

### Generation pipeline improvements (2026-02-23)

**Bug fixes in `generation/generate.py`:**
- Fixed BPM-to-frame conversion in lighting — was using inline formula that didn't match
  `beat_to_frame()`. Now uses the canonical function.
- Added error handling for checkpoint loading — `FileNotFoundError` and `RuntimeError` with
  clear messages instead of cryptic Lightning errors.
- Added warnings when no onsets detected or all token sequences are empty.
- Fixed docstring: BPM auto-detects via librosa (not "defaults to 120.0").

**Multi-difficulty generation:**
- `generate_level()` now accepts `difficulties: list[str]` to generate multiple diffs in one zip.
- Audio encoding is shared across all difficulties (computed once).
- CLI: `python scripts/generate.py song.mp3 --difficulty Expert ExpertPlus Hard`
- Extracted lighting generation to `_generate_lighting_for_beatmap()` helper.

**MP3/OGG audio support (`data/audio.py`):**
- Added ffmpeg fallback for formats soundfile can't handle natively (mp3 on Windows).
- Added `convert_to_ogg()` utility for Beat Saber zip packaging.
- Export pipeline now converts audio to `.ogg` in the zip (best BS compatibility).

**Gradio Web UI (`scripts/app.py`):**
- Full web interface for map generation: upload audio, pick difficulties/genre, generate .zip.
- Auto-discovers best checkpoints from `outputs/` directory.
- Links to ArcViewer, BS Map Check, and Parity Checker for previewing.
- Launch: `python scripts/app.py [--port 7860] [--share]`
- Added `gradio` to `pyproject.toml` optional deps: `uv pip install -e ".[ui]"`

**All tests pass:** 188/188, `ruff check .` clean.

### Full training run (2026-02-23)

**Memory stability fixes applied:**
1. Added `enable_model_summary=False` and `num_sanity_val_steps=0` to all Trainers
2. Added `_GarbageCollectCallback` — runs `gc.collect()` + `torch.cuda.empty_cache()` after
   each validation epoch to prevent memory creep
3. Reduced dataset LRU cache from 200 → 100 entries per worker (8 workers × 100 × ~6MB
   = ~4.8 GB total, down from ~9.6 GB)
4. Updated `run_training_pipeline.py` with optimal per-stage batch sizes, `--stages` and
   `--skip-onset` flags, and timing output

**Pipeline launched (PID 29760, detached):**
```
python scripts/run_training_pipeline.py --max-epochs 100
```
- Stage order: onset → sequence → lighting (sequential, full GPU)
- Onset: batch_size=64, 12 workers, ~5.8 it/s, 43,858 steps/epoch, ~2h/epoch, 6.6 GB VRAM
- Sequence: batch_size=32, 8 workers
- Lighting: batch_size=48, 8 workers
- EarlyStopping(patience=10) on all stages
- Log: `logs/pipeline_full.log`, per-stage: `logs/train_{onset,sequence,lighting}_full.log`
- TensorBoard: version_24 (onset)

**Existing checkpoints (from prior partial runs):**
```
outputs/beatsaber_automapper/version_22/checkpoints/onset-epoch=01-val_f1=0.229.ckpt
outputs/beatsaber_automapper/version_0/checkpoints/sequence-epoch=01-val_loss=1.329.ckpt
```

### Training pipeline notes (from prior sessions)

- Preprocessing complete: **12,014/14,492 .pt files** in `data/processed/`; remainder skipped (v2 maps)
- Dataset split: 10,213 train / 1,200 val / 599 test; `frame_index.json` present for fast init

**Bugs fixed (2026-02-22):**
1. `BCEWithLogitsLoss` + bf16 logits → `CUDNN_STATUS_EXECUTION_FAILED`. Fix: `logits.float()` in
   `onset_module.py` training_step and validation_step.
2. CUDA OOM when gaming: added gradient checkpointing (`use_checkpoint` flag) to AudioEncoder and
   OnsetModel, controlled by `model.onset.gradient_checkpointing=true` config flag.
3. Added `accumulate_grad_batches: 1` to train.yaml (overridable). Also added `+accumulate_grad_batches=4`
   CLI override pattern.
4. **CUDA device-side assert** in sequence training: 15 stale `.pt` files had token indices ≥ 167
   (old preprocessor missing `min(int(o.duration), 64)` wall-duration clamp). Token 1034 = `DUR_INT_OFFSET(98) + 936` (a 936-beat wall).
   - Fix A: `data/dataset.py` `SequenceDataset.__getitem__` clamps tokens: `.clamp(0, 166)` safety net.
   - Fix B: All 15 bad files deleted from `data/processed/`; their entries removed from `frame_index.json`.
   - Bad files: `15b49 15d52 15d87 160b8 161a9 1677f 1a037 1a53b 1a561 1ad83 1b068 1b66f 31dc5 38139 3ac33`
   - 11,997 clean `.pt` files remain.
5. **Triton spam** (`W... triton not found; flop counting will not work for triton kernels`) printed
   once per DataLoader worker on every run. Fixed by:
   - `scripts/train.py`: `logging.getLogger("torch.utils.flop_counter").setLevel(logging.ERROR)` in `main()`.
   - `data/dataset.py`: `_worker_init_fn()` sets same logger level in each worker, passed via `worker_init_fn=`.

**If you ever delete `.pt` files, also remove their entries from `frame_index.json`:**
```bash
python scripts/build_index.py --data-dir data/processed   # full rebuild (~20 min)
# or manually edit data/processed/frame_index.json to remove the bad keys
```

**WARNING — never delete `.pt` files while a training run is active.** The DataLoader indexes all
files at startup; deleting a file mid-run causes `FileNotFoundError` in a worker. Also purge deleted
entries from `frame_index.json` before next run.

**Training commands (full VRAM, no game, both stages in parallel):**
```
# Sequence (version_20 was running, ~12k steps into epoch 0)
python scripts/train.py stage=sequence data_dir=data/processed max_epochs=100 \
    data.dataset.batch_size=32 data.dataset.num_workers=8 low_priority=true accelerator=gpu

# Onset (version_21 was running, just started)
python scripts/train.py stage=onset data_dir=data/processed max_epochs=100 \
    data.dataset.batch_size=32 data.dataset.num_workers=8 low_priority=true accelerator=gpu
```
- Both stages fit on RTX 5090 32GB simultaneously (~8 GB onset + ~11 GB sequence)
- Sequence runs at ~5.36 it/s solo; ~2.17 it/s when sharing GPU with onset
- Epoch 0 for sequence = ~535k steps @ 5 it/s ≈ 30 hours solo, ~70 hours shared
- **No checkpoints saved yet** — epoch 0 not complete for either stage on full dataset
- TensorBoard: `python scripts/dashboard.py --no-browser` then open http://localhost:6006

**Prior smoke-test checkpoints** (11,997-file dataset, short run):
```
outputs/beatsaber_automapper/version_0/checkpoints/sequence-epoch=01-val_loss=1.329.ckpt
outputs/smoke_test/beatsaber_automapper/version_1/checkpoints/onset-epoch=02-val_f1=0.248.ckpt
```
These are usable for quick generation tests while full training runs.

- Checkpoints saved under `outputs/beatsaber_automapper/` after each epoch
- Each stage has EarlyStopping(patience=10), so actual epochs << 100 if model converges
- bf16-mixed + gradient_clip_val=1.0 committed to train.yaml and train.py
- Model weights will go to HuggingFace Hub (PR 8); training data stays local

---

## PLAN D: Comprehensive Training Overhaul (2026-02-23)

### The Problem

After ~8 hours on an RTX 5090 at full blast, the onset model shows:
- **Epoch 0:** val_f1=0.227, val_loss=1.080
- **Epoch 1:** val_f1=0.228, val_loss=1.100 (val loss went UP)
- **Epoch 2:** still training, no improvement visible
- Train loss plateau: 1.99 → 1.05 (fast), then stuck at ~1.0 for 2+ epochs

For reference, state-of-the-art musical onset detection achieves F1 ≥ 0.88. Even our
own smoke-test on fewer epochs with a smaller prior dataset got 0.248. The model is
essentially learning the base rate and then stalling.

### Root Cause Analysis

**Five critical issues identified (ordered by severity):**

#### Issue 1: pos_weight=5.0 is catastrophically wrong

The Gaussian-smoothed onset labels (sigma=3) create 11-frame-wide peaks around each
onset. With median ~660 onsets per song and median ~16,345 frames per song:
- Expected positive fraction: 660 × 11 / 16,345 = **44% of frames have label > 0**
- Actual measured: **30.2% median** onset label positive fraction
- With `pos_weight=5.0`, the model is told "positives are 5× more important than negatives"
- But positives are 30-44% of all frames — this is NEARLY BALANCED
- The model learns to predict "somewhat positive" for everything, which minimizes
  BCE loss but gives terrible F1 because peak_picking can't find real peaks in a sea
  of moderate predictions

**Fix:** `pos_weight=1.0` (or remove entirely). The Gaussian smoothing already handles
the timing tolerance — we don't need pos_weight to compensate for class imbalance when
there ISN'T much imbalance after smoothing.

#### Issue 2: 256-frame window = 3 seconds of context is far too small

The onset model sees a 256-frame sliding window (256 × 512 / 44100 = **2.97 seconds**).
This is shorter than a single musical phrase (typically 4-8 bars = 8-16 seconds at
120 BPM). The model cannot learn:
- Verse/chorus transitions
- Build-ups and drops
- Multi-bar rhythmic patterns
- Song structure (intro, verse, chorus, bridge, outro)

Beat Sage, the most popular automapper, also uses a "small window of the spectrogram"
but their results are widely considered mediocre — we should aim higher.

InfernoSaber uses a **deep convolutional autoencoder** to encode entire songs first,
giving full-song context to subsequent models. This is a fundamentally better approach.

**Fix:** Increase window to 1024+ frames (~12 seconds) or switch to a full-song
architecture where the CNN+Transformer processes the entire mel spectrogram.

#### Issue 3: Training on ALL 12k maps including noise

Our 11,997-map dataset includes:
- 777 Noodle Extension maps (5.4%) — wall art, decorative objects, non-standard gameplay
- 112 Mapping Extensions maps — extended grid, irrelevant to standard mapping
- ~270 maps with 0 lighting events
- ~69 broken/test maps under 15 seconds
- Maps with highly variable quality despite 80%+ rating filter

InfernoSaber trains on **curated high-quality maps** filtered by:
- Expert+ only (single difficulty focus)
- ≥90% like/dislike ratio (vs our 80%)
- NPS-based difficulty bands (separate models for different difficulty levels)
- Total training set: "hundreds" of maps, not thousands

**Key insight:** More data ≠ better when quality varies. A curated 500-1000 map
dataset of the absolute best maps may outperform 12k maps with variable quality.
The model spends capacity learning to average across wildly different mapping styles.

**Fix:** Create a "gold standard" curated subset. Filter criteria:
- Vanilla only (no Noodle/ME/Vivify)
- ≥92% upvote ratio
- Expert or ExpertPlus only (single difficulty to start)
- Map must have lighting events
- NPS between 3-12 (reasonable playable range)
- Song duration 90-300 seconds
- ScoreSaber-ranked maps preferred (community-validated quality)

#### Issue 4: Difficulty/genre conditioning is adding noise, not signal

The model receives difficulty and genre embeddings, but:
- **Genre is "unknown" for 100% of maps** — the embedding is pure noise
- **Difficulty distribution is heavily skewed**: ExpertPlus=36.6%, Expert=28.1%,
  Hard=19.2%, Normal=9.3%, Easy=6.8%
- The model is trying to learn ONE function that maps audio+difficulty → onsets for
  ALL difficulties simultaneously, but the mapping is highly nonlinear
- Easy maps have ~2× fewer onsets than ExpertPlus for the same song — the model must
  learn completely different onset densities per difficulty

**Fix for v1:** Train onset model on Expert/ExpertPlus only (single difficulty).
Remove genre conditioning entirely until genre labels are populated.
This eliminates a major source of confusion. Difficulty scaling can be added later
via inference-time threshold adjustment.

#### Issue 5: Train/inference mismatch — onset model sees different input lengths

During **training**, the onset model sees 256-frame windows (3 seconds).
During **inference** (`predict_onsets()`), it receives the FULL song mel spectrogram
(15,168 frames = 3 minutes for the reference song). The model was never trained on
sequences this long — positional encodings, attention patterns, and internal
representations are all calibrated for 256-frame inputs.

This explains why onset detection is even worse at inference than val_f1 suggests:
the model is running completely out of distribution.

**Fix:** Either (a) window the inference too (slide 256-frame windows with overlap,
aggregate predictions), or (b) train on longer windows so inference matches training.
Option (b) is better — increase window to 1024+ and train the model on what it will
see at inference time. For full-song inference, window and aggregate.

#### Issue 6: The model architecture may be undertrained, not underpowered

Current onset model: AudioEncoder(CNN + 6-layer Transformer encoder, d=512) →
OnsetModel(2-layer Transformer decoder, d=512) → Linear → sigmoid

This is ~25M parameters processing 256-frame windows. The issue isn't model size —
it's that 2 epochs on 12k maps ≈ 90k gradient steps, which should be plenty.
The learning rate (3e-4) and cosine schedule with 1000 warmup steps are reasonable.

The real problem is Issues 1-4 above preventing the model from learning the right thing.

### How Competing Automappers Work

| System | Architecture | Data | Onset Method | Quality |
|--------|-------------|------|-------------|---------|
| **Beat Sage** | 2 neural networks | Unknown (large) | NN on mel spec window, focuses on percussion | "Fun but inconsistent" |
| **InfernoSaber** | 4-stage: Autoencoder → TCN → DNN → DNN | Hundreds of curated expert+ maps | TCN on autoencoder features | Best open-source quality |
| **DeepSaber** (Oxford) | WaveNet + Transformer | Small curated set | CNN onset detector | Academic proof-of-concept |
| **Lolighter/ChroMapper** | Rule-based + heuristics | N/A | Audio analysis (librosa) | Decent for basic maps |
| **Ours (current)** | CNN+Transformer encoder → Transformer decoder | 12k maps (all qualities) | Transformer on 3s windows | F1=0.228 (not working) |

**Key takeaway:** Every successful system either uses (a) a much smaller curated dataset,
(b) simpler non-attention architectures (CNN/TCN/DNN), or (c) full-song context via
autoencoder. We're using the hardest approach (large Transformer on large noisy data)
without the infrastructure to make it work.

### The Revised Plan

#### Phase 1: Quick Wins (fix current run, no architecture changes)

1. **Stop current training** — it's not learning and burning GPU hours
2. **Fix pos_weight**: Change from 5.0 → 1.0 in `configs/model/onset.yaml`
3. **Increase window**: 256 → 1024 frames (~12 seconds of context)
   - Update `onset.yaml`: `window_size: 1024, hop: 512`
   - This 4× reduces samples per epoch but each sample is 4× more informative
4. **Filter dataset**: Apply Plan A outlier filters + restrict to vanilla/chroma only
5. **Drop genre conditioning**: Set `num_genres: 1` or bypass the embedding
6. **Restart onset training** with these fixes

Expected: val_f1 should break 0.4+ within 3 epochs if the core issues are fixed.

#### Phase 2: Curated Dataset Experiment

1. **Create a "gold" subset** of ~500-1000 maps:
   - Script: `scripts/curate_dataset.py`
   - Criteria: vanilla, ≥92% rating, Expert/ExpertPlus, has lighting, NPS 3-12,
     duration 90-300s, preferably ScoreSaber-ranked
   - Source: re-query BeatSaver API with tighter filters, or filter existing dataset
2. **Train onset model on gold subset** — if F1 > 0.5, the architecture works and
   the problem was data quality. If F1 still < 0.3, the architecture needs revision.
3. **Compare:** gold-500 vs full-12k vs full-12k-filtered

#### Phase 3: Architecture Improvements (if needed)

If Phase 1-2 don't break F1 > 0.5:

1. **Replace Transformer onset detector with TCN** — InfernoSaber's proven approach.
   Temporal Convolutional Networks handle 1D temporal patterns efficiently with large
   receptive fields via dilated convolutions. No attention overhead.
2. **Add an audio autoencoder stage** — Like InfernoSaber, pre-train a convolutional
   autoencoder to compress the mel spectrogram into a compact representation. Then
   train onset/sequence models on the compressed features.
3. **Consider full-song processing** — Use the CNN frontend to downsample 4-8×, then
   process entire songs with the Transformer. A 3-minute song at 4× downsampled =
   ~2000 frames, which fits in 512-dim Transformer attention.

#### Phase 4: Reference Song Evaluation System

Create a system to track model quality over time using a fixed reference song.

**Implementation: `scripts/evaluate_reference.py`**
```python
# Usage:
# python scripts/evaluate_reference.py --audio data/reference/test_song.ogg \
#     --onset-ckpt outputs/.../onset-epoch=XX.ckpt \
#     --seq-ckpt outputs/.../sequence-epoch=XX.ckpt \
#     --output-dir data/reference/snapshots/

# What it does:
# 1. Runs the full generation pipeline on the reference song
# 2. Saves the generated .zip to snapshots/ with timestamp
# 3. Computes and logs metrics:
#    - Number of onsets detected
#    - Onset density (notes per second)
#    - Note type distribution (notes/bombs/walls/arcs/chains)
#    - Unique patterns count
#    - Grid coverage (how many of 12 grid cells are used)
#    - Difficulty spread (if multi-diff)
# 4. Appends metrics to data/reference/history.json
# 5. Optionally generates a matplotlib chart of metrics over time
```

**Setup:**
1. Pick a reference song (user provides) and store at `data/reference/test_song.ogg`
2. Store a copy of the best human-mapped version of that song (if available) for
   comparison
3. After each training run or checkpoint, run the evaluation script
4. Over time, the snapshots directory builds a visual history of improvement

**Gradio integration:** Add a "Evaluate Reference" button that runs the reference
song through current best checkpoints and displays metrics + links to download the
generated .zip for ArcViewer comparison.

#### Phase 5: Training Speed Optimization

Current: 43,858 steps/epoch at 6.2 it/s = ~2 hours/epoch (onset only).

Optimizations:
1. **Larger batch size with window_size=1024**: GPU can still fit batch_size=32-48
   with 1024-frame windows (4× more data per sample, fewer steps per epoch)
2. **Gradient accumulation**: If batch_size must be reduced, use
   `accumulate_grad_batches=4` to simulate larger effective batches
3. **torch.compile()**: Add `torch.compile(model)` for 20-40% speedup on Blackwell
4. **Mixed precision**: Already using bf16-mixed, which is optimal for sm_120
5. **Pre-compute mel spectrograms**: Already cached in .pt files, so this is fine
6. **DataLoader prefetching**: Ensure `prefetch_factor=2` and `pin_memory=True`

With window=1024 and hop=512 on the gold-500 dataset:
- Samples per epoch ≈ 500 maps × ~30 windows × 2 diffs = ~30,000
- At batch_size=48: ~625 steps/epoch at ~6 it/s = **~100 seconds/epoch**
- Can run 100 epochs in under 3 hours

### Decision Matrix: What to Try First

| Change | Effort | Expected Impact | Risk |
|--------|--------|----------------|------|
| Fix pos_weight → 1.0 | 1 min | HIGH — fixes training signal | None |
| Window 256 → 1024 | 5 min config | HIGH — more musical context | Fewer steps/epoch |
| Drop genre conditioning | 5 min | MEDIUM — removes noise | None |
| Gold-500 curated subset | 2 hrs script | HIGH — cleaner signal | May need API re-query |
| Expert/ExpertPlus only | 5 min config | MEDIUM — focus on one task | Less data |
| TCN instead of Transformer | 1 day | MEDIUM — proven architecture | Code rewrite |
| Reference song evaluator | 2 hrs | META — enables comparison | None |

**Recommended order:** pos_weight → window → drop genre → curated subset → evaluate.
All config-level changes first, then data changes, then architecture if needed.

### Baseline Snapshot (pre-restructure, 2026-02-23)

Reference song: `data/reference/so_tired_rock.mp3` (rock, 123 BPM, 2:56)
Checkpoints: onset-epoch=01-val_f1=0.228, sequence-epoch=01-val_loss=1.329, no lighting

| Metric | Value | Target |
|--------|-------|--------|
| Notes | 1,643 (9.6 NPS) | 700-1050 (4-6 NPS for Expert) |
| Bombs | 0 | 20-60 |
| Walls | 21 | 30-80 |
| Arcs/Chains | 0 | 10-40 |
| Color balance | 82% red / 18% blue | ~50/50 |
| Grid coverage | 6/12 cells | 10-12/12 |
| Unique patterns | 9 | 50+ |
| Direction dist | 84% down | Spread across all 9 |
| Light events | 0 | 200+ |

Snapshot: `data/reference/snapshots/reference_20260223_180304.zip`

### Phase 6: "Best of All Worlds" Architecture (medium-term)

Goal: combine the strongest techniques from every competing automapper into
one system that surpasses them all.

#### What we take from each system

**From InfernoSaber (most successful open-source BS mapper):**
- Audio autoencoder for compact song representation — gives full-song context
  to downstream models without blowing up memory
- TCN for onset detection — proven, efficient, large receptive fields via
  dilated convolutions without attention overhead
- Heavy post-processing rules — sanity checks, playability filters, pattern
  enforcement. Our `playability.py` needs to be much more robust.
- Separate difficulty scaling external to model — simpler than embedding

**From DeepSaber (original academic approach):**
- "Humaneness regularization" — penalize notes placed too close together with
  exponential distance weighting. Add to our onset loss: `exp(-2*dist/window)`
  penalty for predicted onsets that violate minimum spacing
- beam_size=17 for coherent generation — our beam=8 may need to go higher
- Peak threshold 0.33 (not 0.5) — lower threshold + post-processing NMS
  may work better than trying to get sharp peaks

**From Mapperatorinator (best overall, osu!):**
- **Rhythm token weighting at 3×** in loss — timing is the hardest and most
  important thing. Weight onset-related tokens higher in sequence loss too.
- **Conditioning dropout (20%)** on all embeddings during training — enables
  classifier-free guidance at inference. "Show me what Expert looks like" vs
  "show me what NOT Easy looks like" = better difficulty control.
- **388 mel bands** instead of 80 — preserves more frequency detail. RTX 5090
  with 32GB VRAM can handle this easily.
- **Pretrained audio backbone** (Whisper) — we could initialize our audio
  encoder from Whisper weights rather than training from scratch. Whisper was
  designed for audio→text which is structurally similar to audio→beatmap.

**From BeatLearning (innovative small model):**
- **Audio foresight** — let the model "see ahead" in audio while predicting
  current tokens. Musical events are anticipated (build-ups before drops).
  Implementation: extend the audio context window asymmetrically — more
  future frames than past frames.
- **Joint onset + note generation** — longer-term: a single model that
  predicts both WHEN and WHAT in one pass. Eliminates error propagation
  from Stage 1 → Stage 2.

#### Concrete Architecture V2 Plan

**Audio Encoder V2:**
- Increase mel bands: 80 → 192 (compromise between 80 and 388)
- CNN frontend: 4 layers, stride=(2,1) on freq → 192/16 = 12 freq bins
- Projection: 256×12 = 3072 → d_model=512
- Transformer encoder: 6 layers, 8 heads (keep current)
- NEW: Consider initializing from Whisper-small encoder weights
  (Whisper-small uses 80 mel bands at 16kHz; we'd need an adapter layer)
- Full-song processing: with CNN 4× freq downsample, a 3-min song at
  44.1kHz/512 hop = 15,168 frames fits in the Transformer (already proven
  working in our generation pipeline)

**Onset Model V2:**
- Replace 2-layer Transformer decoder → **Hybrid TCN + Transformer:**
  - TCN (4 blocks, dilations 1,2,4,8,16,32, 128 filters) for local
    pattern detection — captures beat/sub-beat patterns with large
    receptive field efficiently
  - 2-layer Transformer on top for global context — verse/chorus/drop
    awareness
- Remove genre embedding (unused), keep difficulty embedding
- Add **humaneness regularization** to loss — penalty for onset
  predictions closer than `min_onset_distance` frames
- pos_weight=1.0 (or remove), Gaussian sigma=2 (sharper peaks)
- Window size: 2048 frames (~24 seconds, covers full musical phrases)
- NEW: Conditioning dropout 20% on difficulty during training

**Sequence Model V2:**
- Keep autoregressive Transformer decoder (8 layers) — most flexible
- Add **rhythm token weighting**: weight timing-sensitive tokens (EVENT_TYPE,
  SEP, EOS) at 3× in CrossEntropyLoss. These control WHEN notes appear;
  property tokens (color, direction) control WHAT and are less critical.
- Add **audio foresight**: extend context_frames asymmetrically — 64 past
  + 192 future = 256 total context (currently 64+64=128 symmetric)
- Add **conditioning dropout** 20% on difficulty + genre → enables CFG
- Add **pattern diversity loss**: auxiliary loss term that penalizes
  low-entropy output distributions. Prevents mode collapse (the "all red
  down" problem we see in the baseline).
- Consider **top-k constrained beam search**: at each step, only consider
  tokens that maintain game-playability constraints (e.g., no two notes
  in same grid cell, color alternation patterns)

**Lighting Model V2:**
- Keep current 4-layer decoder, expand for Chroma (Plan C)
- Priority: get onset + sequence right first

**Post-Processing Pipeline (NEW):**
- `generation/postprocess.py`:
  1. **NPS enforcement**: If NPS exceeds target for difficulty, thin
     notes by removing least musically-correlated onsets
  2. **Color rebalancing**: Force 45-55% red/blue split by flipping
     the least-constrained notes
  3. **Direction diversity**: If any direction > 40% of total, reassign
     some using playability-aware rules (avoid impossible wrist angles)
  4. **Grid coverage**: If < 8/12 cells used, shift some notes to
     unused positions
  5. **Pattern deduplication**: If identical note pattern repeats > 5×
     consecutively, inject variation
  6. **Bomb/wall injection**: Rule-based bomb and wall placement based
     on note patterns (between same-color clusters, during breaks)
  7. **Parity check**: Ensure swing direction alternation is physically
     possible (no 180° wrist flips)

#### Implementation Priority (Proven Techniques)

| Priority | Change | Why |
|----------|--------|-----|
| P0 | Fix pos_weight, window, genre | Unblocks all learning |
| P0 | Post-processing pipeline | Immediately improves any model output |
| P1 | Curated gold dataset | Clean signal >> more noise |
| P1 | Conditioning dropout | Enables CFG, improves generalization |
| P1 | Rhythm token weighting 3× | Proven by Mapperatorinator |
| P2 | Audio foresight (asymmetric context) | Build-up/drop anticipation |
| P2 | Humaneness regularization | Playability constraint in loss |
| P2 | Pattern diversity loss | Prevents mode collapse |
| P3 | TCN hybrid onset model | Better architecture if Transformer stalls |
| P3 | 192 mel bands | More audio detail |
| P3 | Whisper weight initialization | Pretrained features |
| P4 | Joint onset+note model | Research project, long-term |

### Phase 7: "Next-Gen" Architecture — Post-Boom Innovations

The competing automappers are all pre-boom (2019-2024) architectures. They use
vanilla Transformers with sinusoidal PE, no KV caching, no modern attention variants,
no preference optimization, no hierarchical generation. Here's what a 2025/2026
state-of-the-art architecture looks like — our unique twist.

#### Innovation 1: Mamba/SSM Audio Encoder — Full-Song, Linear Time

**The problem:** Transformer self-attention is O(n²) in sequence length. A 3-minute
song at 11.6ms/frame = 15,168 frames. Full self-attention on this is ~230M attention
pairs per layer. This is why every existing system either uses small windows (us,
Beat Sage) or compresses with an autoencoder (InfernoSaber).

**The solution:** Replace the Transformer encoder layers with **Mamba** (Selective
State Space Model). Mamba processes sequences in O(n) linear time with a learned
selective scan — it decides what to remember and what to forget at each timestep,
like an RNN but parallelizable during training.

**Why this is transformative for Beat Saber mapping:**
- Process the **entire song in one pass** — no windowing, no context truncation
- The selective state naturally captures musical structure: remember the beat pattern
  during a verse, update state when the chorus hits, forget noise between sections
- Audio Mamba has been validated for audio representation learning (2024 paper)
- Memory: O(n) vs O(n²) — a 15,168-frame song uses ~60MB vs ~3.5GB for attention
- Training on full songs means no train/inference mismatch

**Implementation:**
```
Audio Encoder V3:
  Mel spec [80, T] → CNN frontend (4 layers, same as now) → [T, 1280] → Linear → [T, 512]
  → Bidirectional Mamba (6 layers, d_state=64, d_conv=4, expand=2)
  → Output: [T, 512] frame embeddings with full-song context
```

The Mamba layers replace the 6 Transformer encoder layers. The CNN frontend stays
(it captures local frequency patterns). The bidirectional processing (forward + backward
Mamba, concatenated/projected) gives each frame context from the entire song in both
directions.

**Package:** `pip install mamba-ssm` (CUDA-optimized selective scan kernels)

#### Innovation 2: RoPE + GQA + SwiGLU — Modern Transformer Internals

Every component that remains a Transformer (onset decoder, sequence decoder, lighting
decoder) should use modern LLM-era internals, not 2017 "Attention is All You Need"
defaults.

**Rotary Position Embeddings (RoPE):**
- Replace sinusoidal PE everywhere
- RoPE encodes relative position directly in the attention computation via rotation
  matrices applied to Q and K
- Naturally handles variable-length sequences (no max_len buffer needed!)
- Extrapolates to longer sequences than seen in training — critical for us since
  songs vary from 30s to 38 minutes
- This eliminates the PE buffer overflow bug we just fixed

**Grouped Query Attention (GQA):**
- Instead of separate K/V heads per attention head (standard MHA), share K/V across
  groups of query heads
- E.g., 8 query heads, 2 KV groups → 4× smaller KV cache, 30% faster inference
- Critical for beam search speed — our 11-minute generation time for 1688 onsets is
  unacceptable. With GQA + KV cache, this could drop to 1-2 minutes.

**SwiGLU Activation:**
- Replace GELU in feedforward layers with SwiGLU: `SwiGLU(x) = Swish(xW₁) ⊙ (xW₂)`
- Used in LLaMA, PaLM, Mistral — consistently outperforms GELU/ReLU
- Free performance improvement, same parameter count

**RMSNorm:**
- Replace LayerNorm with RMSNorm (root mean square normalization)
- Faster (no mean subtraction), used in all modern LLMs
- Drop-in replacement

#### Innovation 3: KV-Cached Beam Search — 10× Faster Generation

**The problem:** Our sequence model generates 1688 onset tokens autoregressively.
Each onset needs up to 64 token steps of beam search with beam_size=8. That's
1688 × 64 × 8 = ~864,000 forward passes through the decoder. Currently each pass
recomputes attention from scratch. **This is why generation takes 11 minutes.**

**The solution:** Implement proper **KV caching** in the sequence decoder.

At each autoregressive step, the self-attention keys and values from all previous
positions are cached. The new step only computes attention for the NEW token position
against the cached K/V. This turns O(n²) per-step into O(n) per-step.

Combined with GQA (smaller K/V), beam search KV sharing (beams share prefix cache),
and our RTX 5090's memory bandwidth:

**Expected speedup:** Generation from 11 minutes → **60-90 seconds** for a 3-minute
song. This makes the Gradio UI actually usable.

**Implementation:**
```python
class KVCache:
    """Manages key/value caches across decoder layers for autoregressive inference."""
    def __init__(self, num_layers, max_seq_len, num_kv_heads, head_dim, device):
        self.k_cache = [torch.zeros(batch, num_kv_heads, max_seq_len, head_dim, device=device)
                        for _ in range(num_layers)]
        self.v_cache = [...]  # same
        self.seq_pos = 0  # current position

    def update(self, layer_idx, new_k, new_v):
        self.k_cache[layer_idx][:, :, self.seq_pos] = new_k
        self.v_cache[layer_idx][:, :, self.seq_pos] = new_v
        self.seq_pos += 1

# In beam search: beams share cache prefix, fork on divergence
```

#### Innovation 4: Hierarchical Structure-Aware Generation

**The problem:** All existing automappers (including ours) treat a song as a flat
sequence of audio frames → flat sequence of notes. But human mappers think
hierarchically: song structure → phrases → individual notes. A great mapper places
an intense pattern at the chorus drop, calms down during the verse, and builds
tension during the bridge. No flat model can learn this without enormous data.

**The solution:** A three-level hierarchical generation pipeline.

**Level 1 — Song Structure Segmentation (NEW):**
- Input: Full-song Mamba audio features
- Output: Segment boundaries + labels (intro, verse, pre-chorus, chorus, bridge,
  drop, breakdown, outro)
- Architecture: Linear classifier on Mamba features (fine-tuned from pre-trained
  music structure analysis models, or trained with our data using song-level labels
  from BeatSaver tags/metadata)
- This tells the model "beats 0-32 are intro, 32-96 are verse, 96-128 are chorus..."

**Level 2 — Phrase-Level Onset Density (modified Stage 1):**
- Input: Audio features + structure labels + difficulty
- Output: Per-phrase onset density curve (not individual onsets yet)
- The model predicts "verse should have 4 NPS, chorus should have 7 NPS, bridge
  should have 2 NPS" — a coarse plan before individual placement
- This replaces the flat sigmoid-per-frame approach with a musically-informed
  density prior

**Level 3 — Note-Level Generation (modified Stages 1+2):**
- Input: Audio features + density plan + difficulty
- Output: Individual onset frames + note tokens
- The onset model now has both local audio features AND a global density target
  from Level 2, so it knows how many onsets to place in each phrase
- The sequence model generates notes conditioned on structure label (e.g., "this
  is a chorus drop" → more dramatic patterns, wider grid usage, faster sequences)

**Why this is unique:** No existing automapper does hierarchical generation. They
all go directly from audio → notes. This mirrors how experienced mappers actually
work and should produce maps with much better musical coherence and flow variety.

**Training data for structure:** We can bootstrap structure labels:
- Use a pretrained music structure analysis model (MusicFM, MERT, or the ResNet-
  based approach from the 2025 paper) to auto-label song sections
- Or use a simpler heuristic: spectral energy + novelty detection to find
  transitions, k-means to cluster sections

#### Innovation 5: DPO (Direct Preference Optimization) for Map Quality

**The insight from the AI boom:** The biggest lesson from LLMs is that supervised
training (predicting the next token) gets you 80% of the way there, but preference
optimization (RLHF/DPO) is what makes outputs actually good. The same principle
applies to beatmaps.

**We have natural preference signals:**
- BeatSaver upvote ratio (0-100%) — community quality rating
- ScoreSaber ranked status — expert-validated playability
- NPS appropriateness — does the note density match the difficulty?
- Download count / play count — popularity (proxy for quality)

**DPO for beatmaps:**
1. Generate pairs of maps for the same song using different checkpoints/temperatures
2. Use BeatSaver quality signals to determine which is "preferred"
3. Train with DPO loss: `L = -log σ(β * (log π(y_w|x) - log π(y_l|x)))`
   where y_w = preferred map, y_l = rejected map

**Or use a learned reward model:**
1. Train a reward model: AudioEncoder + MapEncoder → quality score (0-1)
2. Training data: map features (NPS, pattern diversity, grid coverage, direction
   distribution, color balance) + BeatSaver quality signals
3. Use reward model to guide beam search: at each step, score partial sequences
   and prefer higher-reward beams
4. This is essentially **RLHF for beatmaps** — the model learns to generate maps
   that the community would upvote

**CLaMP-DPO analogy:** Recent 2025 work (CLaMP-DPO) shows DPO improves musicality
of symbolic music generation without human annotation, using a contrastive audio-music
model as the reward signal. We can do the same with BeatSaver community signals.

**Implementation timeline:** DPO requires a working base model first. Train with
supervised learning (Phases 1-6), then apply DPO as a quality refinement step.

#### Innovation 6: Speculative Decoding for Even Faster Inference

Once we have KV-cached beam search working (Innovation 3), we can go further with
**speculative decoding**:

1. Train a tiny "draft" model (2-layer decoder, d=128) alongside the main model
2. At inference: draft model generates N candidate tokens quickly
3. Main model verifies all N in one forward pass (parallel verification)
4. Accept the longest correct prefix, reject the rest
5. Typical acceptance rate: 70-90% → **2-3× additional speedup** on top of KV cache

For beatmap generation, the draft model can be a simple pattern lookup table
(most common note configurations) — it will be right for typical patterns and
the main model corrects the creative/unusual ones. This gets generation down to
**20-30 seconds** for a 3-minute song.

#### Innovation Summary: Our Unique Architecture

```
═══════════════════════════════════════════════════════════════
  BeatSaber Automapper v2 — "NextGen" Architecture (2026)
═══════════════════════════════════════════════════════════════

  Audio (.mp3/.ogg/.wav)
          │
          ▼
  ┌──────────────────────────────────────────────────────────┐
  │  MEL SPECTROGRAM (192 bands, 1024 FFT, 512 hop)         │
  │  → CNN Frontend (4 layers, freq downsample 16×)          │
  │  → Bidirectional Mamba Encoder (6 layers, d=512)         │
  │    ★ Full-song context in O(n) linear time               │
  │    ★ No windowing — processes entire 3-min song at once  │
  │  Output: [T, 512] frame embeddings                       │
  └───────────┬──────────────────────────────────────────────┘
              │
    ┌─────────┼─────────┐
    ▼         ▼         ▼
  ┌──────┐ ┌───────┐ ┌───────┐
  │STRUCT│ │ONSET  │ │LIGHT  │
  │LABEL │ │DETECT │ │GEN    │
  │      │ │       │ │       │
  │Seg-  │ │Hybrid │ │4-layer│
  │ment  │ │TCN +  │ │RoPE + │
  │into  │ │RoPE   │ │GQA    │
  │verse/│ │Trans- │ │decoder│
  │chorus│ │former │ │       │
  │/drop │ │decoder│ │       │
  └──┬───┘ └──┬────┘ └──┬────┘
     │        │         │
     ▼        ▼         │
  ┌───────────────┐     │
  │ NOTE SEQUENCE │     │
  │ GENERATION    │     │
  │               │     │
  │ 8-layer RoPE +│     │
  │ GQA + SwiGLU  │     │
  │ decoder       │     │
  │               │     │
  │ ★ KV-cached   │     │
  │   beam search │     │
  │ ★ Audio fore- │     │
  │   sight (asym)│     │
  │ ★ Structure-  │     │
  │   conditioned │     │
  │ ★ CFG via     │     │
  │   cond dropout│     │
  └───────┬───────┘     │
          │             │
          ▼             ▼
  ┌──────────────────────────────┐
  │  POST-PROCESSING PIPELINE   │
  │  NPS enforcement, color     │
  │  rebalancing, direction     │
  │  diversity, parity check,   │
  │  pattern deduplication      │
  └──────────────┬───────────────┘
                 │
                 ▼
  ┌──────────────────────────────┐
  │  v3 JSON EXPORT → .zip      │
  │  (After DPO refinement)     │
  └──────────────────────────────┘

What makes this unique vs ALL existing automappers:
  1. Mamba encoder — no other mapper uses SSMs for audio
  2. Hierarchical structure — no other mapper segments songs
  3. RoPE/GQA/SwiGLU — modern LLM internals, not 2017 vanilla
  4. KV-cached beam search — 10× faster inference
  5. DPO quality refinement — RLHF-era alignment for beatmaps
  6. Speculative decoding — another 2-3× inference speedup
  7. Full-song context — most use small windows or autoencoders

═══════════════════════════════════════════════════════════════
```

---

## Future Plans

### Plan A: Training Data Outlier Filtering

**Status:** Planned for next training run (do NOT apply to currently running pipeline)

Analysis of the 11,997-map dataset identified ~120 problematic maps (1% of dataset) that
may degrade training quality. Apply these filters before the next `bsa-preprocess` run.

**Tier 1 — Remove immediately (broken/test maps):**
- Songs < 15 seconds long (~69 maps) — these are test uploads or sound effects
- Filter: check audio duration in .pt metadata or re-derive from mel spectrogram frame count

**Tier 2 — Remove (extreme outliers):**
- Maps with < 20 total onsets (~30 maps) — too sparse to learn from
- Maps with > 2,000 onsets per minute (~21 maps) — vibro/spam maps
- Maps where `wall_count / (wall_count + note_count) > 0.90` — "wall art" maps (decorative
  obstacle sculptures with almost no playable notes, mostly Noodle Extension maps)

**Implementation:**
1. Add `scripts/filter_outliers.py` that scans `data/processed/*.pt` files
2. Compute per-map stats: duration, onset count, onset density, wall ratio
3. Output a `data/processed/outlier_blacklist.json` with map hashes and reasons
4. Modify `dataset.py` to skip blacklisted maps at load time (check `__init__`)
5. Rebuild `frame_index.json` after filtering

**Expected impact:** Removes ~120 maps, leaving ~11,877 clean maps. Should reduce
noise in onset model (fewer false positives from spam maps) and sequence model (fewer
degenerate patterns from wall art).

---

### Plan B: Post-Training Bomb & Obstacle Density Controls

**Status:** Planned feature for generation pipeline (post-training, no model changes needed)

Users want to control the density of bombs and obstacles independently from note patterns.
Two approaches, implement both:

#### Approach 1: Post-Processing Filter (no retraining)

Add parameters to `generate_level()` and the Gradio UI:

```python
def generate_level(
    ...,
    bomb_density: str = "medium",      # "none", "low", "medium", "high"
    obstacle_density: str = "medium",   # "none", "low", "medium", "high"
    decorative_walls: bool = False,     # if True, mark walls as uninteractable
)
```

**Implementation in `generation/generate.py`:**
1. After Stage 2 generates the full token sequence, decode to v3 JSON
2. Apply density filtering as a post-processing step:
   - `"none"`: Remove all bombs/obstacles from the decoded JSON
   - `"low"`: Keep only 25% of bombs/obstacles (randomly sample, preserving timing distribution)
   - `"medium"`: Keep as-is (model output)
   - `"high"`: Duplicate bomb/obstacle patterns at adjacent grid positions (heuristic)
3. If `decorative_walls=True`, add `"customData": {"uninteractable": true}` to all obstacles
4. Update `export.py` to pass through `customData` on obstacles

**Gradio UI changes (`scripts/app.py`):**
- Add two dropdowns: "Bomb Density" and "Obstacle Density" with choices
  `["None", "Low", "Medium (default)", "High"]`
- Add checkbox: "Decorative Walls Only (non-threatening)"

#### Approach 2: Conditioning Embedding (requires retraining)

For a future training run, add bomb/obstacle density as a conditioning signal:

1. Compute per-map bomb density percentile and obstacle density percentile during preprocessing
2. Quantize into 4 buckets: none (0), low (1-33%), medium (34-66%), high (67-100%)
3. Add two new embedding layers in SequenceModel (like difficulty/genre embeddings)
4. During training, pass ground-truth density bucket as conditioning
5. During inference, user selects desired density level

**This requires retraining** — implement Approach 1 first for immediate use, then add
Approach 2 conditioning in a future training run for better quality control.

---

### Plan C: Modded Mapping Framework Support

**Status:** Research complete, Chroma lighting is the actionable target

Dataset composition: 72% vanilla, 21.5% Chroma, 5.4% Noodle Extensions, 0.8% Mapping
Extensions, 0.3% Vivify.

#### Feasibility Assessment

| Framework | Feasibility | Worth It? | Reason |
|-----------|------------|-----------|--------|
| **Chroma (lighting)** | HIGH | **Yes** | 21.5% of maps; only affects Stage 3 lighting tokenizer |
| Chroma (note color) | Medium | Maybe | Rare on notes; could be post-processing heuristic |
| Noodle Extensions | Low | No (near-term) | Requires continuous 3D coordinates, animation system |
| Mapping Extensions | Low | No | Only 112 maps, obsoleted by Noodle |
| Vivify | Impossible | No | Requires Unity asset bundles, 3D modeling |

#### Chroma Lighting Support (recommended next step for Stage 3)

Chroma adds `customData` fields to `basicBeatmapEvents`:
- `color: [r, g, b, a]` — custom RGBA color (16.9M instances in dataset)
- `lightID: int | int[]` — target specific light(s) in a group (16.8M instances)
- `direction`, `speed`, `step`, `rotation`, `prop` — less common

**Implementation plan:**
1. **Parsing (`data/beatmap.py`):** Extract `customData.color` and `customData.lightID`
   from lighting events during preprocessing. Handle both v2 (`_customData._color`) and
   v3 (`customData.color`) naming conventions.
2. **Tokenizer (`data/tokenizer.py`):** Add Chroma tokens to the lighting vocabulary:
   - Color tokens: quantize RGBA to 8-bit per channel → `COLOR_R_0..255`, etc.
     (or use a smaller palette of ~64 colors clustered from training data)
   - LightID tokens: `LIGHT_ID_0..31` (cap at 32 individual lights)
3. **Stage 3 model:** No architecture changes needed — just a larger vocabulary
4. **Export (`generation/export.py`):** When color/lightID tokens are predicted,
   add `customData` dict to the exported `basicBeatmapEvents`
5. **Training:** Include Chroma maps in Stage 3 training data (adds ~3,122 maps)

**Estimated effort:** ~2 days of implementation + retraining Stage 3 only.

#### Current Handling of Modded Maps

- Noodle/ME maps with extended grid coordinates are **clamped to 4×3 grid** during parsing
  (beatmap.py line 321-322). This is correct — we lose precision but keep playable notes.
- Chroma lighting customData is currently **silently ignored** during parsing.
- Noodle `uninteractable` (fake) notes are included in training — Plan A's wall ratio
  filter catches the worst offenders, but a future improvement could skip fake notes entirely.

---

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
