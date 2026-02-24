"""End-to-end inference pipeline.

Orchestrates the full generation flow:
    Audio -> AudioEncoder -> Stage 1 (onsets) -> Stage 2 (notes)
    -> Stage 3 (lighting, optional) -> export

Supports loading trained Lightning checkpoints for each stage model,
or running in "random" mode with untrained weights for testing.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch

from beatsaber_automapper.data.audio import (
    beat_to_frame,
    detect_bpm,
    extract_mel_spectrogram,
    frame_to_beat,
    load_audio,
)
from beatsaber_automapper.data.tokenizer import DIFFICULTY_MAP, GENRE_MAP
from beatsaber_automapper.generation.beam_search import beam_search_decode, nucleus_sampling_decode
from beatsaber_automapper.generation.export import package_level, tokens_to_beatmap
from beatsaber_automapper.generation.postprocess import postprocess_beatmap
from beatsaber_automapper.models.components import peak_picking

logger = logging.getLogger(__name__)


def _load_onset_module(
    checkpoint_path: Path,
) -> Any:
    """Load a trained OnsetLitModule from a Lightning checkpoint.

    Args:
        checkpoint_path: Path to a .ckpt file saved by OnsetLitModule.

    Returns:
        Loaded OnsetLitModule in eval mode.

    Raises:
        FileNotFoundError: If checkpoint file does not exist.
        RuntimeError: If checkpoint is incompatible or corrupted.
    """
    from beatsaber_automapper.training.onset_module import OnsetLitModule

    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Onset checkpoint not found: {checkpoint_path}")
    try:
        module = OnsetLitModule.load_from_checkpoint(str(checkpoint_path))
    except Exception as e:
        raise RuntimeError(f"Failed to load onset checkpoint {checkpoint_path}: {e}") from e
    module.eval()
    return module


def _load_sequence_module(
    checkpoint_path: Path,
) -> Any:
    """Load a trained SequenceLitModule from a Lightning checkpoint.

    Args:
        checkpoint_path: Path to a .ckpt file saved by SequenceLitModule.

    Returns:
        Loaded SequenceLitModule in eval mode.

    Raises:
        FileNotFoundError: If checkpoint file does not exist.
        RuntimeError: If checkpoint is incompatible or corrupted.
    """
    from beatsaber_automapper.training.seq_module import SequenceLitModule

    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Sequence checkpoint not found: {checkpoint_path}")
    try:
        module = SequenceLitModule.load_from_checkpoint(str(checkpoint_path))
    except Exception as e:
        raise RuntimeError(
            f"Failed to load sequence checkpoint {checkpoint_path}: {e}"
        ) from e
    module.eval()
    return module


def _make_default_onset_module() -> Any:
    """Create a default (untrained) OnsetLitModule for testing.

    Returns:
        OnsetLitModule with default hyperparameters in eval mode.
    """
    from beatsaber_automapper.training.onset_module import OnsetLitModule

    module = OnsetLitModule()
    module.eval()
    return module


def _make_default_sequence_module() -> Any:
    """Create a default (untrained) SequenceLitModule for testing.

    Returns:
        SequenceLitModule with default hyperparameters in eval mode.
    """
    from beatsaber_automapper.training.seq_module import SequenceLitModule

    module = SequenceLitModule()
    module.eval()
    return module


def _load_lighting_module(checkpoint_path: Path) -> Any:
    """Load a trained LightingLitModule from a Lightning checkpoint.

    Args:
        checkpoint_path: Path to a .ckpt file saved by LightingLitModule.

    Returns:
        Loaded LightingLitModule in eval mode.

    Raises:
        FileNotFoundError: If checkpoint file does not exist.
        RuntimeError: If checkpoint is incompatible or corrupted.
    """
    from beatsaber_automapper.training.light_module import LightingLitModule

    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Lighting checkpoint not found: {checkpoint_path}")
    try:
        module = LightingLitModule.load_from_checkpoint(str(checkpoint_path))
    except Exception as e:
        raise RuntimeError(
            f"Failed to load lighting checkpoint {checkpoint_path}: {e}"
        ) from e
    module.eval()
    return module


def _make_default_lighting_module() -> Any:
    """Create a default (untrained) LightingLitModule for testing.

    Returns:
        LightingLitModule with default hyperparameters in eval mode.
    """
    from beatsaber_automapper.training.light_module import LightingLitModule

    module = LightingLitModule()
    module.eval()
    return module


def generate_lighting_events(
    lighting_module: Any,
    audio_features: torch.Tensor,
    note_tokens_tensor: torch.Tensor,
    genre_tensor: torch.Tensor,
    beam_size: int = 4,
    temperature: float = 1.0,
    max_length: int = 32,
    device: torch.device | None = None,
) -> list[int]:
    """Run Stage 3 greedy decoding to generate lighting tokens for one beat.

    Args:
        lighting_module: LightingLitModule (has audio_encoder + lighting_model).
        audio_features: Context audio features [1, T, d_model].
        note_tokens_tensor: Note tokens for context beat [1, N].
        genre_tensor: Genre index tensor [1].
        beam_size: Unused (kept for API compatibility).
        temperature: Sampling temperature.
        max_length: Maximum lighting token sequence length.
        device: Torch device.

    Returns:
        List of generated lighting tokens (without BOS/EOS).
    """
    from beatsaber_automapper.data.tokenizer import LIGHT_BOS, LIGHT_EOS

    if device is None:
        device = next(lighting_module.parameters()).device

    # Simple greedy decoding for lighting
    tokens = [LIGHT_BOS]
    model = lighting_module.lighting_model

    for _ in range(max_length):
        token_tensor = torch.tensor([tokens], dtype=torch.long, device=device)
        with torch.no_grad():
            logits = model.decode_step(
                token_tensor, audio_features, note_tokens_tensor, genre_tensor
            )  # [1, V]
        logits = logits.squeeze(0) / temperature
        next_token = int(logits.argmax().item())
        if next_token == LIGHT_EOS:
            break
        tokens.append(next_token)

    return tokens[1:]  # strip BOS


def predict_onsets(
    onset_module: Any,
    mel: torch.Tensor,
    difficulty_idx: int,
    genre_idx: int = 0,
    threshold: float = 0.5,
    min_distance: int = 5,
    device: torch.device | None = None,
    window_size: int = 1024,
    hop: int = 512,
) -> list[int]:
    """Run Stage 1 onset prediction on a mel spectrogram.

    Uses sliding-window inference to match training conditions. The model
    was trained on fixed-length windows, so we slide overlapping windows
    across the full song and average the probability predictions in
    overlapping regions before peak picking.

    Args:
        onset_module: OnsetLitModule (or object with audio_encoder + onset_model).
        mel: Mel spectrogram [n_mels, T].
        difficulty_idx: Integer difficulty index (0-4).
        genre_idx: Integer genre index (0-10).
        threshold: Peak picking probability threshold.
        min_distance: Minimum frames between peaks.
        device: Torch device for inference.
        window_size: Window size in frames (must match training).
        hop: Hop between windows in frames.

    Returns:
        List of frame indices where onsets are predicted.
    """
    if device is None:
        device = next(onset_module.parameters()).device

    total_frames = mel.shape[1]
    diff_tensor = torch.tensor([difficulty_idx], device=device)
    genre_tensor = torch.tensor([genre_idx], device=device)

    # If the song fits in a single window, process directly
    if total_frames <= window_size:
        mel_batch = mel.unsqueeze(0).to(device)
        with torch.no_grad():
            logits = onset_module(mel_batch, diff_tensor, genre_tensor)
            probs = torch.sigmoid(logits.squeeze(0))
        frames = peak_picking(probs, threshold=threshold, min_distance=min_distance)
        return frames.tolist()

    # Sliding window with overlap averaging
    prob_sum = torch.zeros(total_frames, device=device)
    hit_count = torch.zeros(total_frames, device=device)

    starts = list(range(0, total_frames - window_size + 1, hop))
    # Ensure we cover the tail end
    if starts[-1] + window_size < total_frames:
        starts.append(total_frames - window_size)

    for start in starts:
        end = start + window_size
        window_mel = mel[:, start:end].unsqueeze(0).to(device)  # [1, n_mels, W]
        with torch.no_grad():
            logits = onset_module(window_mel, diff_tensor, genre_tensor)  # [1, W]
            probs = torch.sigmoid(logits.squeeze(0))  # [W]
        prob_sum[start:end] += probs
        hit_count[start:end] += 1.0

    # Average overlapping predictions
    avg_probs = prob_sum / hit_count.clamp(min=1.0)

    frames = peak_picking(avg_probs, threshold=threshold, min_distance=min_distance)
    return frames.tolist()


def generate_note_sequence(
    seq_module: Any,
    audio_features: torch.Tensor,
    difficulty_idx: int,
    genre_idx: int = 0,
    beam_size: int = 8,
    temperature: float = 1.0,
    use_sampling: bool = False,
    top_p: float = 0.9,
    max_length: int = 64,
    device: torch.device | None = None,
) -> list[int]:
    """Run Stage 2 beam search to generate tokens for a single onset.

    Args:
        seq_module: SequenceLitModule (has audio_encoder + sequence_model).
        audio_features: Context audio features [1, T, d_model].
        difficulty_idx: Integer difficulty index (0-4).
        genre_idx: Integer genre index (0-10).
        beam_size: Beam search width.
        temperature: Sampling temperature.
        use_sampling: If True, use nucleus sampling instead of beam search.
        top_p: Nucleus sampling top-p threshold.
        max_length: Maximum token sequence length.
        device: Torch device for inference.

    Returns:
        List of generated tokens (without BOS/EOS).
    """
    if device is None:
        device = next(seq_module.parameters()).device

    diff_tensor = torch.tensor([difficulty_idx], device=device)
    genre_tensor = torch.tensor([genre_idx], device=device)

    if use_sampling:
        return nucleus_sampling_decode(
            model=seq_module.sequence_model,
            audio_features=audio_features,
            difficulty=diff_tensor,
            genre=genre_tensor,
            max_length=max_length,
            temperature=temperature,
            top_p=top_p,
        )
    else:
        return beam_search_decode(
            model=seq_module.sequence_model,
            audio_features=audio_features,
            difficulty=diff_tensor,
            genre=genre_tensor,
            beam_size=beam_size,
            max_length=max_length,
            temperature=temperature,
        )


def generate_level(
    audio_path: Path | str,
    output_path: Path | str,
    difficulty: str = "Expert",
    difficulties: list[str] | None = None,
    onset_checkpoint: Path | str | None = None,
    sequence_checkpoint: Path | str | None = None,
    lighting_checkpoint: Path | str | None = None,
    onset_threshold: float = 0.5,
    min_onset_distance: int = 5,
    beam_size: int = 8,
    temperature: float = 1.0,
    use_sampling: bool = False,
    top_p: float = 0.9,
    context_frames: int = 128,
    song_name: str | None = None,
    song_author: str = "Unknown Artist",
    bpm: float | None = None,
    genre: str = "unknown",
    device: str | None = None,
    n_mels: int = 80,
    n_fft: int = 1024,
    hop_length: int = 512,
    sample_rate: int = 44100,
    lighting_beats_per_bar: int = 2,
    onset_window_size: int = 1024,
    onset_hop: int = 512,
) -> Path:
    """Generate a complete Beat Saber level from an audio file.

    Runs the full Stage 1 + Stage 2 + optional Stage 3 pipeline:
    load audio, compute mel spectrogram, predict onset frames, generate note
    tokens at each onset via beam search, optionally generate lighting events
    at regular beat intervals, decode tokens, and export to .zip.

    Supports multi-difficulty generation: pass ``difficulties=["Hard", "Expert"]``
    to generate multiple difficulties in one zip. Audio encoding is shared across
    all difficulties. If ``difficulties`` is provided, ``difficulty`` is ignored.

    If no checkpoints are provided, models are initialized with random
    weights (useful for testing the pipeline structure).

    Args:
        audio_path: Path to input audio file (.mp3, .ogg, .wav).
        output_path: Path for the output .zip file.
        difficulty: Single difficulty name (ignored if ``difficulties`` is set).
        difficulties: List of difficulty names to generate (e.g. ["Expert", "ExpertPlus"]).
        onset_checkpoint: Path to trained OnsetLitModule .ckpt, or None for random weights.
        sequence_checkpoint: Path to trained SequenceLitModule .ckpt, or None for random.
        lighting_checkpoint: Path to trained LightingLitModule .ckpt, or None to skip.
        onset_threshold: Peak picking threshold for onset detection.
        min_onset_distance: Minimum frames between predicted onsets.
        beam_size: Beam search width for sequence generation.
        temperature: Sampling temperature.
        use_sampling: If True, use nucleus sampling instead of beam search.
        top_p: Nucleus sampling top-p threshold.
        context_frames: Number of audio frames as context window per onset.
        song_name: Song title for Info.dat (defaults to audio filename stem).
        song_author: Song artist name for Info.dat.
        bpm: BPM for Info.dat. If None, auto-detected via librosa (falls back to 120.0).
        genre: Genre string for conditioning (e.g. "electronic", "rock").
        device: Torch device string (e.g. "cuda", "cpu"). Auto-detected if None.
        n_mels: Number of mel bands (must match trained model).
        n_fft: FFT window size.
        hop_length: Hop length for spectrogram.
        sample_rate: Target audio sample rate.
        lighting_beats_per_bar: How many lighting beats to generate per bar.
        onset_window_size: Window size in frames for onset inference (must match training).
        onset_hop: Hop between windows for onset inference.

    Returns:
        Path to the generated .zip file.
    """
    from beatsaber_automapper.data.tokenizer import EOS

    audio_path = Path(audio_path)
    output_path = Path(output_path)

    # Resolve difficulty list
    diff_list = difficulties if difficulties else [difficulty]
    logger.info("Generating difficulties: %s", diff_list)

    if song_name is None:
        song_name = audio_path.stem

    # Device selection
    if device is None:
        resolved_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        resolved_device = torch.device(device)
    logger.info("Using device: %s", resolved_device)

    genre_idx = GENRE_MAP.get(genre, 0)

    # --- Load audio & mel ---
    logger.info("Loading audio: %s", audio_path)
    waveform, sr = load_audio(audio_path, target_sr=sample_rate)

    # Auto-detect BPM if not supplied
    if bpm is None:
        logger.info("No BPM provided — auto-detecting via librosa...")
        bpm = detect_bpm(waveform, sample_rate=sr)
    logger.info("Using BPM: %.1f", bpm)

    mel = extract_mel_spectrogram(
        waveform, sample_rate=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length
    )
    logger.info("Mel spectrogram shape: %s", list(mel.shape))

    # --- Load models ---
    if onset_checkpoint is not None:
        logger.info("Loading onset model from %s", onset_checkpoint)
        onset_module = _load_onset_module(Path(onset_checkpoint))
    else:
        logger.info("No onset checkpoint — using untrained model")
        onset_module = _make_default_onset_module()
    onset_module = onset_module.to(resolved_device)

    if sequence_checkpoint is not None:
        logger.info("Loading sequence model from %s", sequence_checkpoint)
        seq_module = _load_sequence_module(Path(sequence_checkpoint))
    else:
        logger.info("No sequence checkpoint — using untrained model")
        seq_module = _make_default_sequence_module()
    seq_module = seq_module.to(resolved_device)

    run_lighting = lighting_checkpoint is not None
    if run_lighting:
        logger.info("Loading lighting model from %s", lighting_checkpoint)
        lighting_module = _load_lighting_module(Path(lighting_checkpoint))
        lighting_module = lighting_module.to(resolved_device)
    else:
        lighting_module = None
        logger.info("No lighting checkpoint — Stage 3 skipped")

    # --- Shared audio encoding (computed once, reused for all difficulties) ---
    mel_batch = mel.unsqueeze(0).to(resolved_device)  # [1, n_mels, T]
    with torch.no_grad():
        full_audio_features = seq_module.audio_encoder(mel_batch)

    total_frames = mel.shape[1]
    half_ctx = context_frames // 2

    # Pre-compute lighting audio features if needed
    light_audio_features = None
    if run_lighting and lighting_module is not None:
        with torch.no_grad():
            light_audio_features = lighting_module.audio_encoder(mel_batch)

    # --- Generate each difficulty ---
    all_beatmaps: dict[str, Any] = {}

    for diff_name in diff_list:
        difficulty_idx = DIFFICULTY_MAP.get(diff_name, 3)
        logger.info(
            "=== Generating %s (idx=%d, genre=%s) ===",
            diff_name, difficulty_idx, genre,
        )

        # Stage 1: Onset prediction (per-difficulty — model outputs different densities)
        # Uses sliding-window inference to match training window size
        onset_frames = predict_onsets(
            onset_module=onset_module,
            mel=mel,
            difficulty_idx=difficulty_idx,
            genre_idx=genre_idx,
            threshold=onset_threshold,
            min_distance=min_onset_distance,
            device=resolved_device,
            window_size=onset_window_size,
            hop=onset_hop,
        )
        logger.info("Found %d onsets for %s", len(onset_frames), diff_name)

        if len(onset_frames) == 0:
            logger.warning(
                "No onsets for %s! Try lowering --onset-threshold (%.2f).",
                diff_name, onset_threshold,
            )

        # Stage 2: Note sequence generation per onset
        beat_tokens: dict[float, list[int]] = {}
        for onset_frame in onset_frames:
            start = max(0, onset_frame - half_ctx)
            end = min(total_frames, onset_frame + half_ctx)
            context_features = full_audio_features[:, start:end, :]

            tokens = generate_note_sequence(
                seq_module=seq_module,
                audio_features=context_features,
                difficulty_idx=difficulty_idx,
                genre_idx=genre_idx,
                beam_size=beam_size,
                temperature=temperature,
                use_sampling=use_sampling,
                top_p=top_p,
                device=resolved_device,
            )

            if tokens:
                beat = frame_to_beat(
                    onset_frame, bpm=bpm, sample_rate=sample_rate,
                    hop_length=hop_length,
                )
                beat_tokens[round(beat, 4)] = tokens + [EOS]

        logger.info("Generated tokens for %d/%d onsets", len(beat_tokens), len(onset_frames))

        if len(beat_tokens) == 0:
            logger.warning(
                "All token sequences empty for %s — map will have no notes.",
                diff_name,
            )

        # Decode tokens to beatmap
        beatmap = tokens_to_beatmap(beat_tokens)
        logger.info(
            "%s (raw): %d notes, %d bombs, %d walls, %d arcs, %d chains",
            diff_name,
            len(beatmap.color_notes),
            len(beatmap.bomb_notes),
            len(beatmap.obstacles),
            len(beatmap.sliders),
            len(beatmap.burst_sliders),
        )

        # Post-processing: improve playability and diversity
        song_dur_secs = None
        if beatmap.color_notes:
            max_beat = max(n.beat for n in beatmap.color_notes)
            song_dur_secs = max_beat / (bpm / 60.0) if bpm > 0 else None
        beatmap = postprocess_beatmap(
            beatmap, difficulty=diff_name, bpm=bpm, song_duration_secs=song_dur_secs,
        )
        logger.info(
            "%s (post): %d notes, %d bombs, %d walls, %d arcs, %d chains",
            diff_name,
            len(beatmap.color_notes),
            len(beatmap.bomb_notes),
            len(beatmap.obstacles),
            len(beatmap.sliders),
            len(beatmap.burst_sliders),
        )

        # Stage 3: Lighting (optional, per-difficulty)
        if run_lighting and lighting_module is not None and light_audio_features is not None:
            beatmap = _generate_lighting_for_beatmap(
                beatmap=beatmap,
                lighting_module=lighting_module,
                light_audio_features=light_audio_features,
                beat_tokens=beat_tokens,
                genre_idx=genre_idx,
                bpm=bpm,
                sample_rate=sample_rate,
                hop_length=hop_length,
                total_frames=total_frames,
                half_ctx=half_ctx,
                lighting_beats_per_bar=lighting_beats_per_bar,
                temperature=temperature,
                resolved_device=resolved_device,
            )

        all_beatmaps[diff_name] = beatmap

    # --- Export to .zip ---
    output_path = package_level(
        beatmaps=all_beatmaps,
        audio_path=audio_path,
        output_path=output_path,
        song_name=song_name,
        song_author=song_author,
        bpm=bpm,
    )

    return output_path


def _generate_lighting_for_beatmap(
    beatmap: Any,
    lighting_module: Any,
    light_audio_features: torch.Tensor,
    beat_tokens: dict[float, list[int]],
    genre_idx: int,
    bpm: float,
    sample_rate: int,
    hop_length: int,
    total_frames: int,
    half_ctx: int,
    lighting_beats_per_bar: int,
    temperature: float,
    resolved_device: torch.device,
) -> Any:
    """Generate lighting events and attach them to a beatmap."""
    from beatsaber_automapper.data.tokenizer import LIGHT_EOS, LightingTokenizer

    light_tokenizer = LightingTokenizer()
    note_token_lookup: dict[float, list[int]] = dict(beat_tokens)

    song_duration_beats = frame_to_beat(
        total_frames, bpm=bpm, sample_rate=sample_rate, hop_length=hop_length
    )
    beat_step = 1.0 / max(1, lighting_beats_per_bar)
    light_beats = [
        round(b * beat_step, 4)
        for b in range(int(song_duration_beats / beat_step) + 1)
    ]

    light_beat_tokens: dict[float, list[int]] = {}
    max_note_len = 64
    genre_tensor = torch.tensor([genre_idx], dtype=torch.long, device=resolved_device)

    sorted_note_beats = sorted(note_token_lookup.keys()) if note_token_lookup else []

    for lbeat in light_beats:
        if sorted_note_beats:
            nearest_note_beat = min(sorted_note_beats, key=lambda b: abs(b - lbeat))
            note_toks = note_token_lookup[nearest_note_beat]
        else:
            note_toks = []

        if len(note_toks) > max_note_len:
            note_toks = note_toks[:max_note_len]
        note_padded = note_toks + [0] * (max_note_len - len(note_toks))
        note_tensor = torch.tensor(
            [note_padded], dtype=torch.long, device=resolved_device
        )

        lframe = beat_to_frame(
            lbeat, bpm=bpm, sample_rate=sample_rate, hop_length=hop_length
        )
        lframe = max(0, min(total_frames - 1, lframe))
        lstart = max(0, lframe - half_ctx)
        lend = min(total_frames, lframe + half_ctx)
        light_ctx = light_audio_features[:, lstart:lend, :]

        ltokens = generate_lighting_events(
            lighting_module=lighting_module,
            audio_features=light_ctx,
            note_tokens_tensor=note_tensor,
            genre_tensor=genre_tensor,
            temperature=temperature,
            max_length=32,
            device=resolved_device,
        )

        if ltokens:
            light_beat_tokens[lbeat] = ltokens + [LIGHT_EOS]

    basic_events, color_boost_events = light_tokenizer.decode_lighting(
        light_beat_tokens
    )
    beatmap.basic_events.extend(basic_events)
    beatmap.color_boost_events.extend(color_boost_events)
    logger.info(
        "Lighting: %d basic events, %d boost events",
        len(basic_events),
        len(color_boost_events),
    )
    return beatmap
