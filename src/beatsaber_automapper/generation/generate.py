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

import numpy as np
import torch

from beatsaber_automapper.data.audio import (
    beat_to_frame,
    compute_section_features,
    compute_structure_features,
    detect_bpm,
    detect_sections,
    extract_mel_spectrogram,
    frame_to_beat,
    load_audio,
)
from beatsaber_automapper.data.beatmap import ColorNote, DifficultyBeatmap
from beatsaber_automapper.data.tokenizer import DIFFICULTY_MAP, GENRE_MAP
from beatsaber_automapper.generation.beam_search import (
    ConstraintState,
    beam_search_decode,
    init_constraints,
    nucleus_sampling_decode,
)
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
        import torch as _torch
        ckpt = _torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
        sd = ckpt.get("state_dict", {})
        w = sd.get("audio_encoder.structure_proj.weight")
        kwargs = {}
        if w is not None:
            kwargs["n_structure_features"] = int(w.shape[1])
        module = OnsetLitModule.load_from_checkpoint(str(checkpoint_path), **kwargs)
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


def _load_note_pred_module(checkpoint_path: Path) -> Any:
    """Load a trained NotePredictionLitModule from a Lightning checkpoint."""
    from beatsaber_automapper.training.note_module import NotePredictionLitModule

    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Note predictor checkpoint not found: {checkpoint_path}")
    try:
        module = NotePredictionLitModule.load_from_checkpoint(str(checkpoint_path))
    except Exception as e:
        raise RuntimeError(
            f"Failed to load note predictor checkpoint {checkpoint_path}: {e}"
        ) from e
    module.eval()
    return module


# Angle offset index -> degree offset mapping
_ANGLE_OFFSETS = [-45, -30, -15, 0, 15, 30, 45]


def predict_notes_structured(
    note_pred_module: Any,
    audio_features: torch.Tensor,
    difficulty_idx: int,
    genre_idx: int = 0,
    device: torch.device | None = None,
    prev_tokens: torch.Tensor | None = None,
) -> list[ColorNote]:
    """Run NotePredictor to generate notes for a single onset.

    Converts the multi-head structured predictions into ColorNote objects.

    Args:
        note_pred_module: NotePredictionLitModule with audio_encoder + note_predictor.
        audio_features: Audio context features [1, T, d_model].
        difficulty_idx: Difficulty index (0-4).
        genre_idx: Genre index (0-10).
        device: Torch device.
        prev_tokens: Optional previous onset tokens [1, K, S].

    Returns:
        List of ColorNote objects for this onset (0-3 notes).
    """
    if device is None:
        device = next(note_pred_module.parameters()).device

    diff_tensor = torch.tensor([difficulty_idx], device=device)
    genre_tensor = torch.tensor([genre_idx], device=device)

    with torch.no_grad():
        preds = note_pred_module.note_predictor(
            audio_features=audio_features,
            difficulty=diff_tensor,
            genre=genre_tensor,
            prev_tokens=prev_tokens,
        )

    # Decode predictions — use per-slot color to determine active slots
    # (n_notes head may be unreliable; color==2 means "none"/inactive)
    colors = preds["color"].argmax(dim=-1).squeeze(0)     # [3]
    cols = preds["col"].argmax(dim=-1).squeeze(0)          # [3]
    rows = preds["row"].argmax(dim=-1).squeeze(0)          # [3]
    dirs = preds["direction"].argmax(dim=-1).squeeze(0)    # [3]
    angles = preds["angle"].argmax(dim=-1).squeeze(0)      # [3]

    notes = []
    used_positions: set[tuple[int, int]] = set()
    for slot in range(3):
        color = colors[slot].item()
        if color >= 2:  # "none" = inactive slot
            continue
        col = cols[slot].item()
        row = rows[slot].item()
        # Skip grid collisions
        pos = (col, row)
        if pos in used_positions:
            continue
        used_positions.add(pos)

        direction = dirs[slot].item()
        angle_idx = angles[slot].item()
        angle_offset = _ANGLE_OFFSETS[angle_idx] if angle_idx < len(_ANGLE_OFFSETS) else 0

        notes.append(ColorNote(
            beat=0.0,  # Will be set by the caller
            x=col,
            y=row,
            color=color,
            direction=direction,
            angle_offset=angle_offset,
        ))

    return notes


def generate_lighting_events(
    lighting_module: Any,
    audio_features: torch.Tensor,
    note_tokens_tensor: torch.Tensor,
    genre_tensor: torch.Tensor,
    beam_size: int = 4,
    temperature: float = 1.0,
    max_length: int = 32,
    device: torch.device | None = None,
    top_p: float = 0.9,
) -> list[int]:
    """Run Stage 3 constrained nucleus sampling to generate lighting tokens for one beat.

    Uses a state machine to enforce valid lighting event structure:
    LIGHT_BASIC(4) must be followed by ET(6-20), VAL(21-28), BRIGHT(29-32).
    After a complete event, only SEP(2), EOS(1), or a new event type marker is allowed.

    Args:
        lighting_module: LightingLitModule (has audio_encoder + lighting_model).
        audio_features: Context audio features [1, T, d_model].
        note_tokens_tensor: Note tokens for context beat [1, N].
        genre_tensor: Genre index tensor [1].
        beam_size: Unused (kept for API compatibility).
        temperature: Sampling temperature.
        max_length: Maximum lighting token sequence length.
        device: Torch device.
        top_p: Nucleus sampling probability threshold.

    Returns:
        List of generated lighting tokens (without BOS/EOS).
    """
    from beatsaber_automapper.data.tokenizer import (
        LIGHT_BASIC,
        LIGHT_BOOST,
        LIGHT_BOS,
        LIGHT_BRIGHT_OFFSET,
        LIGHT_EOS,
        LIGHT_ET_OFFSET,
        LIGHT_ONOFF_OFFSET,
        LIGHT_SEP,
        LIGHT_VAL_OFFSET,
        LIGHT_VOCAB_SIZE,
    )

    if device is None:
        device = next(lighting_module.parameters()).device

    tokens = [LIGHT_BOS]
    model = lighting_module.lighting_model

    # State machine for constrained decoding
    # States: "start" -> waiting for event type marker
    #         "et"    -> expecting ET token (6-20)
    #         "val"   -> expecting VAL token (21-28)
    #         "bright"-> expecting BRIGHT token (29-32)
    #         "onoff" -> expecting ONOFF token (33-34)
    state = "start"

    for _ in range(max_length):
        token_tensor = torch.tensor([tokens], dtype=torch.long, device=device)
        with torch.no_grad():
            logits = model.decode_step(
                token_tensor, audio_features, note_tokens_tensor, genre_tensor
            )  # [1, V]
        logits = logits.squeeze(0) / temperature

        # Apply structural constraints via logit masking
        mask = torch.full((LIGHT_VOCAB_SIZE,), float("-inf"), device=device)
        if state == "start":
            # Allow: LIGHT_BASIC(4), LIGHT_BOOST(5), EOS(1)
            mask[LIGHT_BASIC] = 0.0
            mask[LIGHT_BOOST] = 0.0
            mask[LIGHT_EOS] = 0.0
        elif state == "et":
            # Allow: ET tokens (6-20)
            mask[LIGHT_ET_OFFSET : LIGHT_ET_OFFSET + 15] = 0.0
        elif state == "val":
            # Allow: VAL tokens (21-28)
            mask[LIGHT_VAL_OFFSET : LIGHT_VAL_OFFSET + 8] = 0.0
        elif state == "bright":
            # Allow: BRIGHT tokens (29-32)
            mask[LIGHT_BRIGHT_OFFSET : LIGHT_BRIGHT_OFFSET + 4] = 0.0
        elif state == "onoff":
            # Allow: ONOFF tokens (33-34)
            mask[LIGHT_ONOFF_OFFSET : LIGHT_ONOFF_OFFSET + 2] = 0.0

        logits = logits + mask

        # Nucleus sampling
        probs = torch.softmax(logits, dim=-1)
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumulative = torch.cumsum(sorted_probs, dim=-1)
        # Remove tokens with cumulative > top_p (keep at least 1)
        remove_mask = cumulative - sorted_probs > top_p
        sorted_probs[remove_mask] = 0.0
        sorted_probs = sorted_probs / sorted_probs.sum()
        idx = torch.multinomial(sorted_probs, 1).item()
        next_token = int(sorted_indices[idx].item())

        if next_token == LIGHT_EOS:
            break
        tokens.append(next_token)

        # Update state machine — check special tokens first, then state transitions
        if next_token == LIGHT_SEP:
            state = "start"
        elif next_token == LIGHT_BASIC:
            state = "et"
        elif next_token == LIGHT_BOOST:
            state = "onoff"
        elif state == "et":
            state = "val"
        elif state == "val":
            state = "bright"
        elif state in ("bright", "onoff"):
            state = "start"  # event complete, expect SEP/EOS/new event

    return tokens[1:]  # strip BOS


# Target NPS ranges by difficulty (from training data analysis)
_NPS_RANGES: dict[int, tuple[float, float]] = {
    0: (1.0, 3.0),   # Easy
    1: (2.0, 5.0),   # Normal
    2: (3.0, 7.0),   # Hard
    3: (4.0, 10.0),  # Expert
    4: (5.0, 14.0),  # ExpertPlus
}


def _compute_adaptive_threshold(
    structure_features: torch.Tensor,
    base_threshold: float = 0.25,
    threshold_range: float = 0.20,
) -> torch.Tensor:
    """Compute per-frame adaptive onset threshold from energy.

    Loud sections get lower threshold (more onsets), quiet sections
    get higher threshold (fewer onsets).

    Args:
        structure_features: Song structure features [6, T].
        base_threshold: Threshold floor in loudest sections.
        threshold_range: Range added in quietest sections.

    Returns:
        Per-frame threshold tensor [T].
    """
    rms_energy = structure_features[0]  # [T]
    # Smooth with ~2-second window (200 frames at ~10ms/frame)
    kernel_size = min(200, rms_energy.shape[0])
    if kernel_size > 1:
        padding = kernel_size // 2
        smoothed = torch.nn.functional.avg_pool1d(
            rms_energy.unsqueeze(0).unsqueeze(0),
            kernel_size=kernel_size,
            padding=padding,
            count_include_pad=False,
        ).squeeze()[:rms_energy.shape[0]]
    else:
        smoothed = rms_energy

    # Normalize to 0-1 range
    e_min, e_max = smoothed.min(), smoothed.max()
    if e_max > e_min:
        normalized = (smoothed - e_min) / (e_max - e_min)
    else:
        normalized = torch.full_like(smoothed, 0.5)

    # High energy → low threshold (more notes), low energy → high threshold
    return base_threshold + (1.0 - normalized) * threshold_range


def _quantize_to_beat_grid(
    onset_frames: list[int],
    bpm: float,
    sample_rate: int,
    hop_length: int,
    max_subdivision: int = 8,
) -> list[int]:
    """Snap onset frames to nearest beat subdivision.

    Args:
        onset_frames: List of frame indices.
        bpm: Song BPM.
        sample_rate: Audio sample rate.
        hop_length: Spectrogram hop length.
        max_subdivision: Finest subdivision (8 = eighth notes).

    Returns:
        Sorted, deduplicated list of quantized frame indices.
    """
    if not onset_frames or bpm <= 0:
        return onset_frames

    frames_per_beat = (60.0 / bpm) * sample_rate / hop_length
    grid_spacing = frames_per_beat / max_subdivision

    if grid_spacing < 1:
        return onset_frames

    max_frame = max(onset_frames) + int(frames_per_beat)
    grid = np.arange(0, max_frame, grid_spacing)

    if len(grid) == 0:
        return onset_frames

    snapped = set()
    for f in onset_frames:
        nearest_idx = np.argmin(np.abs(grid - f))
        snapped.add(int(round(grid[nearest_idx])))

    return sorted(snapped)


def _apply_density_curve(
    onset_frames: list[int],
    difficulty_idx: int,
    structure_features: torch.Tensor,
    bpm: float,
    sample_rate: int,
    hop_length: int,
    avg_probs: torch.Tensor | None = None,
) -> list[int]:
    """Thin onsets to match difficulty-appropriate NPS based on energy.

    In high-energy sections, allow up to the upper NPS bound.
    In low-energy sections, target the lower NPS bound.
    Remove lowest-confidence onsets first when thinning.

    Args:
        onset_frames: Detected onset frame indices.
        difficulty_idx: 0-4 difficulty index.
        structure_features: [6, T] features.
        bpm: Song BPM.
        sample_rate: Audio sample rate.
        hop_length: Spectrogram hop length.
        avg_probs: Per-frame onset probabilities [T] for confidence ranking.

    Returns:
        Filtered onset frame list.
    """
    if not onset_frames or bpm <= 0:
        return onset_frames

    nps_min, nps_max = _NPS_RANGES.get(difficulty_idx, (4.0, 10.0))
    frames_per_second = sample_rate / hop_length
    total_frames = structure_features.shape[1]

    # Compute local energy for each onset
    rms = structure_features[0].cpu().numpy()  # [T]
    # Smooth energy
    kernel = min(200, len(rms))
    if kernel > 1:
        try:
            from scipy.ndimage import uniform_filter1d
            smoothed_rms = uniform_filter1d(rms, size=kernel)
        except ImportError:
            # Fallback: simple moving average
            cumsum = np.cumsum(np.insert(rms, 0, 0))
            smoothed_rms = (cumsum[kernel:] - cumsum[:-kernel]) / kernel
            # Pad to match original length
            pad_left = kernel // 2
            pad_right = len(rms) - len(smoothed_rms) - pad_left
            smoothed_rms = np.pad(smoothed_rms, (pad_left, max(0, pad_right)), mode="edge")
    else:
        smoothed_rms = rms

    # Normalize
    e_min, e_max = smoothed_rms.min(), smoothed_rms.max()
    if e_max > e_min:
        norm_rms = (smoothed_rms - e_min) / (e_max - e_min)
    else:
        norm_rms = np.full_like(smoothed_rms, 0.5)

    # For each onset, compute target NPS based on local energy
    # Use 2-second windows to check density
    window_frames = int(2.0 * frames_per_second)
    if window_frames < 1:
        return onset_frames

    # Build confidence scores for each onset
    onset_scores = []
    for f in onset_frames:
        energy = norm_rms[min(f, len(norm_rms) - 1)]
        target_nps = nps_min + energy * (nps_max - nps_min)
        conf = avg_probs[f].item() if avg_probs is not None and f < len(avg_probs) else 0.5
        onset_scores.append((f, target_nps, conf))

    # Check density in sliding windows and thin if needed
    # Simple approach: compute overall target and thin globally
    avg_energy = float(norm_rms.mean())
    target_nps = nps_min + avg_energy * (nps_max - nps_min)
    total_seconds = total_frames / frames_per_second
    target_count = int(target_nps * total_seconds)

    if len(onset_frames) <= target_count:
        return onset_frames

    # Sort by confidence and keep the highest-confidence ones
    onset_scores.sort(key=lambda x: x[2], reverse=True)
    kept = sorted([s[0] for s in onset_scores[:target_count]])

    logger.info(
        "Density curve: %d -> %d onsets (target NPS=%.1f for difficulty=%d, avg_energy=%.2f)",
        len(onset_frames), len(kept), target_nps, difficulty_idx, avg_energy,
    )
    return kept


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
    structure_features: torch.Tensor | None = None,
    adaptive_threshold: bool = True,
    base_threshold: float = 0.25,
    threshold_range: float = 0.20,
    bpm: float = 120.0,
    sample_rate: int = 44100,
    hop_length: int = 512,
) -> list[int]:
    """Run Stage 1 onset prediction on a mel spectrogram.

    Uses sliding-window inference to match training conditions. The model
    was trained on fixed-length windows, so we slide overlapping windows
    across the full song and average the probability predictions in
    overlapping regions before peak picking.

    Supports energy-adaptive thresholds, beat grid quantization, and
    difficulty-scaled density curves.

    Args:
        onset_module: OnsetLitModule (or object with audio_encoder + onset_model).
        mel: Mel spectrogram [n_mels, T].
        difficulty_idx: Integer difficulty index (0-4).
        genre_idx: Integer genre index (0-10).
        threshold: Peak picking probability threshold (used when adaptive is off).
        min_distance: Minimum frames between peaks.
        device: Torch device for inference.
        window_size: Window size in frames (must match training).
        hop: Hop between windows in frames.
        structure_features: Optional [6, T] song structure features.
        adaptive_threshold: Use energy-adaptive thresholds.
        base_threshold: Threshold floor for adaptive mode.
        threshold_range: Threshold range for adaptive mode.
        bpm: Song BPM for beat grid quantization.
        sample_rate: Audio sample rate.
        hop_length: Spectrogram hop length.

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
        structure_batch = None
        if structure_features is not None:
            sf_window = structure_features[:, :total_frames]
            if sf_window.shape[1] < window_size:
                pad_size = window_size - sf_window.shape[1]
                sf_window = torch.nn.functional.pad(sf_window, (0, pad_size))
            structure_batch = sf_window.unsqueeze(0).to(device)
        with torch.no_grad():
            logits = onset_module(mel_batch, diff_tensor, genre_tensor, structure=structure_batch)
            avg_probs = torch.sigmoid(logits.squeeze(0))
    else:
        # Sliding window with overlap averaging
        prob_sum = torch.zeros(total_frames, device=device)
        hit_count = torch.zeros(total_frames, device=device)

        starts = list(range(0, total_frames - window_size + 1, hop))
        # Ensure we cover the tail end
        if starts and starts[-1] + window_size < total_frames:
            starts.append(total_frames - window_size)

        for start in starts:
            end = start + window_size
            window_mel = mel[:, start:end].unsqueeze(0).to(device)  # [1, n_mels, W]
            structure_batch = None
            if structure_features is not None:
                sf_window = structure_features[:, start:end]
                if sf_window.shape[1] < window_size:
                    pad_size = window_size - sf_window.shape[1]
                    sf_window = torch.nn.functional.pad(sf_window, (0, pad_size))
                structure_batch = sf_window.unsqueeze(0).to(device)
            with torch.no_grad():
                logits = onset_module(
                    window_mel, diff_tensor, genre_tensor, structure=structure_batch
                )
                probs = torch.sigmoid(logits.squeeze(0))  # [W]
            prob_sum[start:end] += probs
            hit_count[start:end] += 1.0

        # Average overlapping predictions
        avg_probs = prob_sum / hit_count.clamp(min=1.0)

    # Peak picking with adaptive or fixed threshold
    if adaptive_threshold and structure_features is not None:
        adaptive_thresh = _compute_adaptive_threshold(
            structure_features.to(device), base_threshold, threshold_range
        )
        # Ensure aligned length
        if adaptive_thresh.shape[0] > avg_probs.shape[0]:
            adaptive_thresh = adaptive_thresh[:avg_probs.shape[0]]
        elif adaptive_thresh.shape[0] < avg_probs.shape[0]:
            pad = avg_probs.shape[0] - adaptive_thresh.shape[0]
            adaptive_thresh = torch.nn.functional.pad(adaptive_thresh, (0, pad), value=threshold)

        # Peak picking with per-frame threshold
        above_threshold = avg_probs > adaptive_thresh
        frames = peak_picking(
            avg_probs * above_threshold.float(),
            threshold=0.01,  # Already thresholded, just find peaks
            min_distance=min_distance,
        )
        frames = frames.tolist()
    else:
        frames = peak_picking(avg_probs, threshold=threshold, min_distance=min_distance)
        frames = frames.tolist()

    # Beat grid quantization — snap to nearest musical subdivision
    if bpm > 0:
        frames = _quantize_to_beat_grid(
            frames, bpm=bpm, sample_rate=sample_rate, hop_length=hop_length,
        )

    # Difficulty-scaled density curve — thin if too many for this difficulty
    if structure_features is not None:
        frames = _apply_density_curve(
            frames,
            difficulty_idx=difficulty_idx,
            structure_features=structure_features,
            bpm=bpm,
            sample_rate=sample_rate,
            hop_length=hop_length,
            avg_probs=avg_probs.cpu(),
        )

    # Beat-grid snapping can round a tail-end peak past total_frames. Clamp
    # and dedupe so every returned index is a valid frame into the mel.
    if frames:
        last = total_frames - 1
        frames = sorted({min(max(f, 0), last) for f in frames})
    return frames


def generate_note_sequence(
    seq_module: Any,
    audio_features: torch.Tensor,
    difficulty_idx: int,
    genre_idx: int = 0,
    beam_size: int = 8,
    temperature: float = 0.8,
    use_sampling: bool = True,
    top_p: float = 0.85,
    max_length: int = 64,
    device: torch.device | None = None,
    prev_tokens: torch.Tensor | None = None,
    min_length: int = 3,
    repetition_penalty: float = 1.5,
    constraints: ConstraintState | None = None,
    plan_vector: torch.Tensor | None = None,
) -> list[int]:
    """Run Stage 2 decoding to generate tokens for a single onset.

    Args:
        seq_module: SequenceLitModule (has audio_encoder + sequence_model).
        audio_features: Context audio features [1, T, d_model].
        difficulty_idx: Integer difficulty index (0-4).
        genre_idx: Integer genre index (0-10).
        beam_size: Beam search width.
        temperature: Sampling temperature (0.8 = less random than default).
        use_sampling: If True, use nucleus sampling instead of beam search.
        top_p: Nucleus sampling top-p threshold (0.85 = tighter nucleus).
        max_length: Maximum token sequence length.
        device: Torch device for inference.
        prev_tokens: Optional previous onset tokens [1, K, S] for inter-onset context.
        min_length: Minimum tokens before EOS is allowed.
        repetition_penalty: Penalize recently generated tokens (1.5 = more variety).
        constraints: Optional ConstraintState for grammar-constrained decoding.
        plan_vector: Optional plan vector [1, 1, d_model] from OnsetPlanner.

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
            prev_tokens=prev_tokens,
            min_length=min_length,
            repetition_penalty=repetition_penalty,
            constraints=constraints,
            plan_vector=plan_vector,
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
            prev_tokens=prev_tokens,
            min_length=min_length,
            constraints=constraints,
            plan_vector=plan_vector,
        )


def generate_level(
    audio_path: Path | str,
    output_path: Path | str,
    difficulty: str = "Expert",
    difficulties: list[str] | None = None,
    onset_checkpoint: Path | str | None = None,
    sequence_checkpoint: Path | str | None = None,
    note_pred_checkpoint: Path | str | None = None,
    lighting_checkpoint: Path | str | None = None,
    onset_threshold: float = 0.5,
    min_onset_distance: int = 5,
    beam_size: int = 8,
    temperature: float = 0.8,
    use_sampling: bool = True,
    top_p: float = 0.85,
    repetition_penalty: float = 1.5,
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
        note_pred_checkpoint: Path to trained NotePredictionLitModule .ckpt. If provided,
            uses structured prediction instead of autoregressive decoding for Stage 2.
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
    # Guard against out-of-bounds genre index: trained models may have num_genres=1
    # (all maps are "unknown"), so clamp to 0 if the model can't handle the requested genre.
    # This is checked after models are loaded below.

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

    # Determine Stage 2 mode: note predictor (structured) or autoregressive
    use_note_pred = note_pred_checkpoint is not None
    note_pred_module = None
    seq_module = None

    if use_note_pred:
        logger.info("Loading note predictor from %s", note_pred_checkpoint)
        note_pred_module = _load_note_pred_module(Path(note_pred_checkpoint))
        note_pred_module = note_pred_module.to(resolved_device)
        # Clamp genre_idx
        np_genre_size = note_pred_module.note_predictor.genre_emb.num_embeddings
        if genre_idx >= np_genre_size:
            logger.warning(
                "Genre '%s' (idx=%d) exceeds model's num_genres=%d — falling back to 'unknown' (0)",
                genre, genre_idx, np_genre_size,
            )
            genre_idx = 0
    else:
        if sequence_checkpoint is not None:
            logger.info("Loading sequence model from %s", sequence_checkpoint)
            seq_module = _load_sequence_module(Path(sequence_checkpoint))
        else:
            logger.info("No sequence checkpoint — using untrained model")
            seq_module = _make_default_sequence_module()
        seq_module = seq_module.to(resolved_device)
        # Clamp genre_idx
        seq_genre_size = seq_module.sequence_model.genre_emb.num_embeddings
        if genre_idx >= seq_genre_size:
            logger.warning(
                "Genre '%s' (idx=%d) exceeds model's num_genres=%d — falling back to 'unknown' (0)",
                genre, genre_idx, seq_genre_size,
            )
            genre_idx = 0

    # Lighting is now rule-based — ML checkpoint ignored
    if lighting_checkpoint is not None:
        logger.info(
            "Ignoring lighting checkpoint (rule-based): %s",
            lighting_checkpoint,
        )

    # --- Compute structure features (8 channels: 6 energy + 2 section) ---
    structure_features = compute_structure_features(
        waveform, sample_rate=sr, hop_length=hop_length, n_mels=n_mels
    )
    # Align to mel length
    if structure_features.shape[1] > mel.shape[1]:
        structure_features = structure_features[:, :mel.shape[1]]
    elif structure_features.shape[1] < mel.shape[1]:
        pad = mel.shape[1] - structure_features.shape[1]
        structure_features = torch.nn.functional.pad(structure_features, (0, pad))

    # Detect song sections and append section_id + section_progress channels
    n_mel_frames = mel.shape[1]
    sections = detect_sections(waveform, sample_rate=sr, hop_length=hop_length)
    section_ids, section_progress = compute_section_features(
        sections, n_frames=n_mel_frames, hop_length=hop_length, sample_rate=sr
    )
    section_id_norm = section_ids.float() / 5.0  # Normalize by max section type index (5)
    structure_features = torch.cat([
        structure_features,
        section_id_norm.unsqueeze(0),    # [1, T]
        section_progress.unsqueeze(0),   # [1, T]
    ], dim=0)  # [8, T]

    # --- Shared audio encoding (computed once, reused for all difficulties) ---
    mel_batch = mel.unsqueeze(0).to(resolved_device)  # [1, n_mels, T]
    structure_batch = structure_features.unsqueeze(0).to(resolved_device)  # [1, 8, T]
    encoder_module = note_pred_module if use_note_pred else seq_module
    with torch.no_grad():
        full_audio_features = encoder_module.audio_encoder(
            mel_batch, structure_features=structure_batch
        )

    total_frames = mel.shape[1]
    half_ctx = context_frames // 2

    # Check if sequence model has an onset planner
    has_planner = (
        seq_module is not None
        and hasattr(seq_module, "onset_planner")
        and seq_module.onset_planner is not None
    )

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
        # Slice structure_features to the channel count the onset encoder was trained on
        # (older checkpoints use 6 channels; newer ones use 8).
        onset_struct = structure_features
        try:
            onset_struct_ch = onset_module.audio_encoder.structure_proj.weight.shape[1]
            if onset_struct_ch != structure_features.shape[0]:
                onset_struct = structure_features[:onset_struct_ch]
        except AttributeError:
            pass
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
            structure_features=onset_struct,
            adaptive_threshold=True,
            base_threshold=0.25,
            threshold_range=0.20,
            bpm=bpm,
            sample_rate=sample_rate,
            hop_length=hop_length,
        )
        logger.info("Found %d onsets for %s", len(onset_frames), diff_name)

        if len(onset_frames) == 0:
            logger.warning(
                "No onsets for %s! Try lowering --onset-threshold (%.2f).",
                diff_name, onset_threshold,
            )

        # Compute plan vectors via OnsetPlanner (if available)
        plan_vectors = None
        if has_planner and len(onset_frames) > 0:
            onset_frame_indices = torch.tensor(onset_frames, dtype=torch.long)
            # Clamp to valid frame range
            onset_frame_indices = onset_frame_indices.clamp(0, full_audio_features.shape[1] - 1)
            # Extract audio embeddings at onset frames: [1, N_onsets, d_model]
            onset_embeddings = full_audio_features[:, onset_frame_indices, :]
            # Extract section features at onset frames for planner conditioning
            # structure_features[6] = normalized section_id, [7] = section_progress
            onset_section_ids = None
            onset_section_progress = None
            if structure_features.shape[0] >= 8:
                # Recover integer section IDs from normalized values
                sec_id_norm = structure_features[6]  # [T]
                sec_ids_int = (sec_id_norm * 5.0).round().long().clamp(0, 5)
                onset_section_ids = sec_ids_int[onset_frame_indices].unsqueeze(0).to(
                    resolved_device
                )  # [1, N_onsets]
                onset_section_progress = structure_features[7][onset_frame_indices].unsqueeze(
                    0
                ).to(resolved_device)  # [1, N_onsets]
            with torch.no_grad():
                plan_vectors = seq_module.onset_planner(
                    onset_embeddings,
                    section_ids=onset_section_ids,
                    section_progress=onset_section_progress,
                )
            logger.info("Computed plan vectors for %d onsets", len(onset_frames))

        # Stage 2: Note generation per onset
        if use_note_pred:
            # --- Structured prediction path (NotePredictor) ---
            all_notes: list[ColorNote] = []
            prev_context_k = getattr(
                note_pred_module.note_predictor, "prev_context_k", 0
            )

            for i, onset_frame in enumerate(onset_frames):
                start = max(0, onset_frame - half_ctx)
                end = min(total_frames, onset_frame + half_ctx)
                context_features = full_audio_features[:, start:end, :]

                beat = frame_to_beat(
                    onset_frame, bpm=bpm, sample_rate=sample_rate,
                    hop_length=hop_length,
                )

                notes = predict_notes_structured(
                    note_pred_module=note_pred_module,
                    audio_features=context_features,
                    difficulty_idx=difficulty_idx,
                    genre_idx=genre_idx,
                    device=resolved_device,
                )
                # Set beat time on each note
                for note in notes:
                    note.beat = round(beat, 4)
                all_notes.extend(notes)

            logger.info(
                "NotePredictor generated %d notes for %d onsets",
                len(all_notes), len(onset_frames),
            )

            # Build beatmap directly from notes
            beatmap = DifficultyBeatmap(version="3.3.0", color_notes=all_notes)

        else:
            # --- Autoregressive token path (SequenceModel) ---
            beat_tokens: dict[float, list[int]] = {}
            generated_sequences: list[list[int]] = []  # for building prev_tokens
            prev_context_k = getattr(seq_module.sequence_model, "prev_context_k", 0)
            max_token_len = 64

            # Track parity across onsets for constrained decoding
            parity_last_dirs: dict[int, int] = {}  # color -> last direction

            for i, onset_frame in enumerate(onset_frames):
                start = max(0, onset_frame - half_ctx)
                end = min(total_frames, onset_frame + half_ctx)
                context_features = full_audio_features[:, start:end, :]

                # Build prev_tokens from previously generated onsets
                prev_tokens_tensor = None
                if prev_context_k > 0:
                    prev_seqs = []
                    for k in range(prev_context_k):
                        prev_idx = i - (prev_context_k - k)
                        if prev_idx >= 0:
                            seq = list(generated_sequences[prev_idx])
                            if len(seq) > max_token_len:
                                seq = seq[:max_token_len]
                            seq = seq + [0] * (max_token_len - len(seq))
                        else:
                            seq = [0] * max_token_len
                        prev_seqs.append(seq)
                    prev_tokens_tensor = torch.tensor(
                        [prev_seqs], dtype=torch.long, device=resolved_device
                    )  # [1, K, S]

                # Create fresh constraints for this onset, carrying over parity
                onset_constraints = init_constraints(
                    difficulty=diff_name,
                    prev_last_dirs=parity_last_dirs,
                )

                # Extract per-onset plan vector if planner is active
                onset_plan_vector = None
                if plan_vectors is not None:
                    onset_plan_vector = plan_vectors[:, i:i + 1, :]  # [1, 1, d_model]

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
                    prev_tokens=prev_tokens_tensor,
                    min_length=7,  # BOS + 1 complete NOTE event (6 tokens) minimum
                    repetition_penalty=repetition_penalty,
                    constraints=onset_constraints,
                    plan_vector=onset_plan_vector,
                )

                # Update cross-onset parity tracking from constraint state
                parity_last_dirs.update(onset_constraints.last_dir)

                generated_sequences.append(tokens)

                if tokens:
                    beat = frame_to_beat(
                        onset_frame, bpm=bpm, sample_rate=sample_rate,
                        hop_length=hop_length,
                    )
                    beat_tokens[round(beat, 4)] = tokens + [EOS]

            logger.info(
                "Generated tokens for %d/%d onsets", len(beat_tokens), len(onset_frames),
            )

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

        # Stage 3: Rule-based lighting (replaces ML model)
        from beatsaber_automapper.generation.lighting_rules import (
            generate_lighting_events as gen_light,
        )
        basic_events, boost_events = gen_light(
            structure_features=structure_features,
            bpm=bpm,
            sample_rate=sample_rate,
            hop_length=hop_length,
        )
        beatmap.basic_events.extend(basic_events)
        beatmap.color_boost_events.extend(boost_events)
        logger.info(
            "Lighting: %d basic events, %d boost events (rule-based)",
            len(basic_events), len(boost_events),
        )

        all_beatmaps[diff_name] = beatmap

    # --- Apply Chroma colors to lighting events ---
    from beatsaber_automapper.generation.chroma import add_chroma_colors

    chroma_beatmap_dicts: dict[str, Any] = {}
    for diff_name, beatmap in all_beatmaps.items():
        if beatmap.basic_events:
            # Build plain event dicts first
            plain_events = [
                {"b": e.beat, "et": e.event_type, "i": e.value, "f": e.float_value}
                for e in beatmap.basic_events
            ]
            # Add Chroma RGB colors based on song structure
            chroma_events = add_chroma_colors(
                events=plain_events,
                structure_features=structure_features,
                bpm=bpm,
                sample_rate=sample_rate,
                hop_length=hop_length,
                genre=genre,
            )
            chroma_beatmap_dicts[diff_name] = chroma_events
        else:
            chroma_beatmap_dicts[diff_name] = None

    # --- Export to .zip ---
    output_path = package_level(
        beatmaps=all_beatmaps,
        audio_path=audio_path,
        output_path=output_path,
        song_name=song_name,
        song_author=song_author,
        bpm=bpm,
        chroma_events=chroma_beatmap_dicts,
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
