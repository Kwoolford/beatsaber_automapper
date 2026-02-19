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

from beatsaber_automapper.data.audio import extract_mel_spectrogram, frame_to_beat, load_audio
from beatsaber_automapper.data.tokenizer import DIFFICULTY_MAP
from beatsaber_automapper.generation.beam_search import beam_search_decode, nucleus_sampling_decode
from beatsaber_automapper.generation.export import package_level, tokens_to_beatmap
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
    """
    from beatsaber_automapper.training.onset_module import OnsetLitModule

    module = OnsetLitModule.load_from_checkpoint(str(checkpoint_path))
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
    """
    from beatsaber_automapper.training.seq_module import SequenceLitModule

    module = SequenceLitModule.load_from_checkpoint(str(checkpoint_path))
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
    """
    from beatsaber_automapper.training.light_module import LightingLitModule

    module = LightingLitModule.load_from_checkpoint(str(checkpoint_path))
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
    beam_size: int = 4,
    temperature: float = 1.0,
    max_length: int = 32,
    device: torch.device | None = None,
) -> list[int]:
    """Run Stage 3 beam search to generate lighting tokens for one beat.

    Args:
        lighting_module: LightingLitModule (has audio_encoder + lighting_model).
        audio_features: Context audio features [1, T, d_model].
        note_tokens_tensor: Note tokens for context beat [1, N].
        beam_size: Beam search width.
        temperature: Sampling temperature.
        max_length: Maximum lighting token sequence length.
        device: Torch device.

    Returns:
        List of generated lighting tokens (without BOS/EOS).
    """
    from beatsaber_automapper.data.tokenizer import LIGHT_BOS, LIGHT_EOS

    if device is None:
        device = next(lighting_module.parameters()).device

    # Simple greedy decoding for lighting (beam search would need light-specific impl)
    tokens = [LIGHT_BOS]
    model = lighting_module.lighting_model

    for _ in range(max_length):
        token_tensor = torch.tensor([tokens], dtype=torch.long, device=device)
        with torch.no_grad():
            logits = model.decode_step(
                token_tensor, audio_features, note_tokens_tensor
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
    threshold: float = 0.5,
    min_distance: int = 5,
    device: torch.device | None = None,
) -> list[int]:
    """Run Stage 1 onset prediction on a mel spectrogram.

    Args:
        onset_module: OnsetLitModule (or object with audio_encoder + onset_model).
        mel: Mel spectrogram [n_mels, T].
        difficulty_idx: Integer difficulty index (0-4).
        threshold: Peak picking probability threshold.
        min_distance: Minimum frames between peaks.
        device: Torch device for inference.

    Returns:
        List of frame indices where onsets are predicted.
    """
    if device is None:
        device = next(onset_module.parameters()).device

    mel_batch = mel.unsqueeze(0).to(device)  # [1, n_mels, T]
    diff_tensor = torch.tensor([difficulty_idx], device=device)

    with torch.no_grad():
        logits = onset_module(mel_batch, diff_tensor)  # [1, T]
        probs = torch.sigmoid(logits.squeeze(0))  # [T]

    frames = peak_picking(probs, threshold=threshold, min_distance=min_distance)
    return frames.tolist()


def generate_note_sequence(
    seq_module: Any,
    audio_features: torch.Tensor,
    difficulty_idx: int,
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

    if use_sampling:
        return nucleus_sampling_decode(
            model=seq_module.sequence_model,
            audio_features=audio_features,
            difficulty=diff_tensor,
            max_length=max_length,
            temperature=temperature,
            top_p=top_p,
        )
    else:
        return beam_search_decode(
            model=seq_module.sequence_model,
            audio_features=audio_features,
            difficulty=diff_tensor,
            beam_size=beam_size,
            max_length=max_length,
            temperature=temperature,
        )


def generate_level(
    audio_path: Path | str,
    output_path: Path | str,
    difficulty: str = "Expert",
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
    device: str | None = None,
    n_mels: int = 80,
    n_fft: int = 1024,
    hop_length: int = 512,
    sample_rate: int = 44100,
    lighting_beats_per_bar: int = 2,
) -> Path:
    """Generate a complete Beat Saber level from an audio file.

    Runs the full Stage 1 + Stage 2 + optional Stage 3 pipeline:
    load audio, compute mel spectrogram, predict onset frames, generate note
    tokens at each onset via beam search, optionally generate lighting events
    at regular beat intervals, decode tokens, and export to .zip.

    If no checkpoints are provided, models are initialized with random
    weights (useful for testing the pipeline structure).

    Args:
        audio_path: Path to input audio file (.mp3, .ogg, .wav).
        output_path: Path for the output .zip file.
        difficulty: Difficulty name (Easy, Normal, Hard, Expert, ExpertPlus).
        onset_checkpoint: Path to trained OnsetLitModule .ckpt, or None for random weights.
        sequence_checkpoint: Path to trained SequenceLitModule .ckpt, or None for random weights.
        lighting_checkpoint: Path to trained LightingLitModule .ckpt, or None to skip Stage 3.
            If None and no checkpoint is given, Stage 3 is skipped entirely.
        onset_threshold: Peak picking threshold for onset detection.
        min_onset_distance: Minimum frames between predicted onsets.
        beam_size: Beam search width for sequence generation.
        temperature: Sampling temperature.
        use_sampling: If True, use nucleus sampling instead of beam search.
        top_p: Nucleus sampling top-p threshold.
        context_frames: Number of audio frames as context window per onset.
        song_name: Song title for Info.dat (defaults to audio filename stem).
        song_author: Song artist name for Info.dat.
        bpm: BPM for Info.dat. If None, defaults to 120.0 (no detection).
        device: Torch device string (e.g. "cuda", "cpu"). Auto-detected if None.
        n_mels: Number of mel bands (must match trained model).
        n_fft: FFT window size.
        hop_length: Hop length for spectrogram.
        sample_rate: Target audio sample rate.
        lighting_beats_per_bar: How many lighting beats to generate per bar (default 2).

    Returns:
        Path to the generated .zip file.
    """
    audio_path = Path(audio_path)
    output_path = Path(output_path)

    if song_name is None:
        song_name = audio_path.stem
    if bpm is None:
        bpm = 120.0
        logger.warning("No BPM provided — defaulting to 120.0. Pass bpm= for accurate timing.")

    # Device selection
    if device is None:
        resolved_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        resolved_device = torch.device(device)
    logger.info("Using device: %s", resolved_device)

    # Difficulty index
    difficulty_idx = DIFFICULTY_MAP.get(difficulty, 3)

    # --- Load audio & mel ---
    logger.info("Loading audio: %s", audio_path)
    waveform, sr = load_audio(audio_path, target_sr=sample_rate)
    mel = extract_mel_spectrogram(
        waveform, sample_rate=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length
    )
    # mel: [n_mels, T]
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

    # Stage 3 is optional — skip entirely if no checkpoint and running in test mode
    run_lighting = lighting_checkpoint is not None
    if run_lighting:
        logger.info("Loading lighting model from %s", lighting_checkpoint)
        lighting_module = _load_lighting_module(Path(lighting_checkpoint))
        lighting_module = lighting_module.to(resolved_device)
    else:
        lighting_module = None
        logger.info("No lighting checkpoint — Stage 3 skipped")

    # --- Stage 1: Onset prediction ---
    logger.info(
        "Predicting onsets (threshold=%.2f, min_distance=%d)...",
        onset_threshold,
        min_onset_distance,
    )
    onset_frames = predict_onsets(
        onset_module=onset_module,
        mel=mel,
        difficulty_idx=difficulty_idx,
        threshold=onset_threshold,
        min_distance=min_onset_distance,
        device=resolved_device,
    )
    logger.info("Found %d onsets", len(onset_frames))

    # --- Stage 2: Encode audio once, then decode per onset ---
    mel_batch = mel.unsqueeze(0).to(resolved_device)  # [1, n_mels, T]
    with torch.no_grad():
        full_audio_features = seq_module.audio_encoder(mel_batch)  # [1, T_audio, d_model]

    beat_tokens: dict[float, list[int]] = {}
    total_frames = mel.shape[1]
    half_ctx = context_frames // 2

    for onset_frame in onset_frames:
        # Extract context window around onset
        start = max(0, onset_frame - half_ctx)
        end = min(total_frames, onset_frame + half_ctx)
        context_features = full_audio_features[:, start:end, :]  # [1, ctx, d_model]

        tokens = generate_note_sequence(
            seq_module=seq_module,
            audio_features=context_features,
            difficulty_idx=difficulty_idx,
            beam_size=beam_size,
            temperature=temperature,
            use_sampling=use_sampling,
            top_p=top_p,
            device=resolved_device,
        )

        if tokens:
            beat = frame_to_beat(
                onset_frame, bpm=bpm, sample_rate=sample_rate, hop_length=hop_length
            )
            # Append EOS so tokenizer decode_beatmap works correctly
            from beatsaber_automapper.data.tokenizer import EOS
            beat_tokens[round(beat, 4)] = tokens + [EOS]

    logger.info("Generated token sequences for %d onsets", len(beat_tokens))

    # --- Decode note tokens -> DifficultyBeatmap ---
    beatmap = tokens_to_beatmap(beat_tokens)
    logger.info(
        "Decoded beatmap: %d notes, %d bombs, %d walls, %d arcs, %d chains",
        len(beatmap.color_notes),
        len(beatmap.bomb_notes),
        len(beatmap.obstacles),
        len(beatmap.sliders),
        len(beatmap.burst_sliders),
    )

    # --- Stage 3: Lighting generation (optional) ---
    if run_lighting and lighting_module is not None:
        from beatsaber_automapper.data.tokenizer import LIGHT_EOS, LightingTokenizer

        light_tokenizer = LightingTokenizer()

        # Build a per-beat note token lookup for conditioning
        note_token_lookup: dict[float, list[int]] = dict(beat_tokens)

        # Generate lighting on a regular beat grid
        # Estimate song duration from mel
        total_frames = mel.shape[1]
        song_duration_beats = frame_to_beat(
            total_frames, bpm=bpm, sample_rate=sample_rate, hop_length=hop_length
        )
        beat_step = 1.0 / max(1, lighting_beats_per_bar)
        light_beats = [
            round(b * beat_step, 4)
            for b in range(int(song_duration_beats / beat_step) + 1)
        ]

        # Encode audio once via lighting model's audio encoder
        light_audio_features: torch.Tensor
        with torch.no_grad():
            light_audio_features = lighting_module.audio_encoder(mel_batch)

        light_beat_tokens: dict[float, list[int]] = {}
        max_note_len = 64

        for lbeat in light_beats:
            # Get note context nearest to this lighting beat
            if note_token_lookup:
                nearest_note_beat = min(
                    note_token_lookup.keys(), key=lambda b: abs(b - lbeat)
                )
                note_toks = note_token_lookup[nearest_note_beat]
            else:
                note_toks = []

            # Pad/truncate note tokens
            if len(note_toks) > max_note_len:
                note_toks = note_toks[:max_note_len]
            note_padded = note_toks + [0] * (max_note_len - len(note_toks))
            note_tensor = torch.tensor([note_padded], dtype=torch.long, device=resolved_device)

            # Extract audio context for this beat
            lframe = int(round(lbeat * sample_rate * 60.0 / (bpm * hop_length)))
            lframe = max(0, min(total_frames - 1, lframe))
            lstart = max(0, lframe - half_ctx)
            lend = min(total_frames, lframe + half_ctx)
            light_ctx = light_audio_features[:, lstart:lend, :]

            ltokens = generate_lighting_events(
                lighting_module=lighting_module,
                audio_features=light_ctx,
                note_tokens_tensor=note_tensor,
                temperature=temperature,
                max_length=32,
                device=resolved_device,
            )

            if ltokens:
                light_beat_tokens[lbeat] = ltokens + [LIGHT_EOS]

        basic_events, color_boost_events = light_tokenizer.decode_lighting(light_beat_tokens)
        beatmap.basic_events.extend(basic_events)
        beatmap.color_boost_events.extend(color_boost_events)
        logger.info(
            "Generated %d basic events and %d boost events",
            len(basic_events),
            len(color_boost_events),
        )

    # --- Export to .zip ---
    output_path = package_level(
        beatmaps={difficulty: beatmap},
        audio_path=audio_path,
        output_path=output_path,
        song_name=song_name,
        song_author=song_author,
        bpm=bpm,
    )

    return output_path
