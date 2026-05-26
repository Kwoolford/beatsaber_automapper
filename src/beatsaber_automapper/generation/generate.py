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
    compute_section_features,
    compute_structure_features,
    detect_bpm,
    detect_sections,
    extract_mel_spectrogram,
    frame_to_beat,
    load_audio,
)
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


# ---------------------------------------------------------------------------
# V6 windowed inference helpers
# ---------------------------------------------------------------------------

_FRAMES_PER_SEC = 44100 / 512  # at sr=44100, hop=512


def _mel_window(
    full: torch.Tensor, center_frame: int, width: int, n_frames: int,
) -> torch.Tensor:
    """Extract a width-frame window of mel/structure centred on center_frame.

    Pads at the right edge if the song is shorter than the desired width.
    full shape: [B, C, T]; returns [B, C, width].
    """
    half = width // 2
    start = max(0, center_frame - half)
    end = start + width
    if end > n_frames:
        end = n_frames
        start = max(0, end - width)
    win = full[:, :, start:end]
    if win.shape[2] < width:
        win = torch.nn.functional.pad(win, (0, width - win.shape[2]))
    return win


def _compute_window_beats(context_frames: int, bpm: float) -> float:
    """How many beats fit comfortably inside an audio context window.

    Uses 60% of the context as the active region, leaving 20% buffer at each
    edge — this way every event in the window has at least ~0.3 seconds of
    audio context to either side.
    """
    seconds_per_window = context_frames / _FRAMES_PER_SEC
    beats_per_window = seconds_per_window * (bpm / 60.0)
    return max(1.0, beats_per_window * 0.6)


def _events_to_beatmap(events: list) -> Any:
    """Build a DifficultyBeatmap from a list of _SwingEvent objects.

    Matches ARC_HEAD/CHAIN_HEAD to their respective TAIL events by HAND FIFO,
    same policy as SwingEventTokenizer.decode_beatmap. Unmatched heads/tails
    are dropped.
    """
    from beatsaber_automapper.data.beatmap import (
        BombNote,
        BurstSlider,
        ColorNote,
        DifficultyBeatmap,
        Slider,
    )
    from beatsaber_automapper.data.swing_tokenizer import (
        _ANGLE_BINS,
        _MU_BINS,
        _SLICE_MIN,
        _SQUISH_BINS,
        ARC_HEAD,
        ARC_TAIL,
        BOMB,
        CHAIN_HEAD,
        CHAIN_TAIL,
        HAND_LEFT,
        HAND_RIGHT,
        NOTE,
    )

    def _color_from_hand(h: int) -> int:
        return 0 if h == HAND_LEFT else 1

    color_notes: list = []
    bomb_notes: list = []
    sliders: list = []
    burst_sliders: list = []
    arc_heads: dict = {HAND_LEFT: [], HAND_RIGHT: []}
    chain_heads: dict = {HAND_LEFT: [], HAND_RIGHT: []}

    # Events come from the sampler in time order already; sort by beat as a safety net.
    for evt in sorted(events, key=lambda e: e.beat):
        if evt.kind == NOTE and evt.hand in (HAND_LEFT, HAND_RIGHT):
            color_notes.append(ColorNote(
                beat=evt.beat,
                x=evt.x, y=evt.y,
                color=_color_from_hand(evt.hand),
                direction=evt.direction,
                angle_offset=int(_ANGLE_BINS[evt.field_d]),
            ))
        elif evt.kind == BOMB:
            bomb_notes.append(BombNote(beat=evt.beat, x=evt.x, y=evt.y))
        elif evt.kind == ARC_HEAD and evt.hand in (HAND_LEFT, HAND_RIGHT):
            arc_heads[evt.hand].append(
                (evt.beat, evt.x, evt.y, evt.direction, _MU_BINS[evt.field_d])
            )
        elif evt.kind == ARC_TAIL and evt.hand in (HAND_LEFT, HAND_RIGHT):
            if arc_heads[evt.hand]:
                hb, hx, hy, hdir, hmu = arc_heads[evt.hand].pop(0)
                sliders.append(Slider(
                    color=_color_from_hand(evt.hand),
                    beat=hb, x=hx, y=hy, direction=hdir, mu=hmu,
                    tail_beat=evt.beat, tail_x=evt.x, tail_y=evt.y,
                    tail_direction=evt.direction, tail_mu=_MU_BINS[evt.field_d],
                    mid_anchor_mode=0,
                ))
        elif evt.kind == CHAIN_HEAD and evt.hand in (HAND_LEFT, HAND_RIGHT):
            chain_heads[evt.hand].append(
                (evt.beat, evt.x, evt.y, evt.direction, evt.field_d + _SLICE_MIN)
            )
        elif evt.kind == CHAIN_TAIL and evt.hand in (HAND_LEFT, HAND_RIGHT):
            if chain_heads[evt.hand]:
                hb, hx, hy, hdir, hsc = chain_heads[evt.hand].pop(0)
                burst_sliders.append(BurstSlider(
                    color=_color_from_hand(evt.hand),
                    beat=hb, x=hx, y=hy, direction=hdir,
                    tail_beat=evt.beat, tail_x=evt.x, tail_y=evt.y,
                    slice_count=hsc, squish=_SQUISH_BINS[evt.field_d],
                ))

    return DifficultyBeatmap(
        version="3.3.0",
        color_notes=color_notes,
        bomb_notes=bomb_notes,
        obstacles=[],
        sliders=sliders,
        burst_sliders=burst_sliders,
    )


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

    if sequence_checkpoint is not None:
        logger.info("Loading sequence model from %s", sequence_checkpoint)
        seq_module = _load_sequence_module(Path(sequence_checkpoint))
    else:
        logger.info("No sequence checkpoint — using untrained model")
        seq_module = _make_default_sequence_module()
    seq_module = seq_module.to(resolved_device)
    seq_genre_size = seq_module.sequence_model.genre_emb.num_embeddings
    if genre_idx >= seq_genre_size:
        logger.warning(
            "Genre '%s' (idx=%d) exceeds model's num_genres=%d — falling back to 'unknown' (0)",
            genre, genre_idx, seq_genre_size,
        )
        genre_idx = 0

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
    with torch.no_grad():
        full_audio_features = seq_module.audio_encoder(
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

        # Stage 2: Autoregressive token generation
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


# ---------------------------------------------------------------------------
# V6 generation pipeline
# ---------------------------------------------------------------------------


def generate_swing_level(
    audio_path: Path | str,
    output_path: Path | str,
    difficulty: str = "Expert",
    sequence_checkpoint: Path | str | None = None,
    onset_checkpoint: Path | str | None = None,
    temperature: float = 0.9,
    top_p: float = 0.9,
    max_events: int = 2000,
    context_frames: int = 256,
    phrase_frames: int = 1024,
    song_name: str | None = None,
    song_author: str = "Unknown Artist",
    bpm: float | None = None,
    genre: str = "unknown",
    mapper_id: int = 0,
    device: str | None = None,
    n_mels: int = 80,
    n_fft: int = 1024,
    hop_length: int = 512,
    sample_rate: int = 44100,
    lighting_beats_per_bar: int = 2,
) -> Path:
    """Generate a complete Beat Saber level using the V6 swing-event model.

    Uses a single grammar-constrained nucleus sampling pass to generate the
    full per-hand swing-event stream for the song, rather than the V5
    per-onset approach.

    Args:
        audio_path: Input audio file (.mp3, .ogg, .wav).
        output_path: Output .zip path.
        difficulty: Difficulty name (Easy/Normal/Hard/Expert/ExpertPlus).
        sequence_checkpoint: Path to a V6 SequenceLitModule checkpoint.
            None = random untrained weights (useful for testing).
        onset_checkpoint: Path to an OnsetLitModule checkpoint (still used for
            BPM timing hints). None = skip onset model.
        temperature: Sampling temperature.
        top_p: Nucleus sampling top-p threshold.
        max_events: Maximum number of swing events to generate.
        context_frames: Audio context frames per generation window.
        phrase_frames: Wide audio context frames for phrase embedding.
        song_name: Song name override. Defaults to the audio filename stem.
        song_author: Song author name.
        bpm: BPM override. Auto-detected if None.
        genre: Genre string for conditioning.
        mapper_id: Cohort mapper index (0 = unknown/generic).
        device: Torch device string. Defaults to "cuda" if available.
        n_mels: Mel band count.
        n_fft: FFT size.
        hop_length: Mel hop length.
        sample_rate: Target sample rate.
        lighting_beats_per_bar: Lighting density (beats per light event).

    Returns:
        Path to the generated .zip file.
    """
    from beatsaber_automapper.data.tokenizer import DIFFICULTY_MAP, GENRE_MAP

    audio_path = Path(audio_path)
    output_path = Path(output_path)
    if song_name is None:
        song_name = audio_path.stem

    if device is None:
        device_obj = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device_obj = torch.device(device)

    logger.info(
        "V6 generate_swing_level: %s → %s (device=%s)",
        audio_path.name, output_path.name, device_obj,
    )

    # --- Audio preprocessing ---
    waveform, sr = load_audio(audio_path, target_sr=sample_rate)
    mel = extract_mel_spectrogram(waveform, sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length)

    if bpm is None:
        bpm = detect_bpm(waveform, sr)
    logger.info("BPM: %.1f", bpm)

    structure_features = compute_structure_features(
        waveform=waveform, sample_rate=sr, hop_length=hop_length, n_mels=n_mels,
    )
    n_mel_frames = mel.shape[1]
    # Align to mel length
    if structure_features.shape[1] > n_mel_frames:
        structure_features = structure_features[:, :n_mel_frames]
    elif structure_features.shape[1] < n_mel_frames:
        structure_features = torch.nn.functional.pad(
            structure_features, (0, n_mel_frames - structure_features.shape[1])
        )
    sections = detect_sections(waveform, sample_rate=sr, hop_length=hop_length)
    section_ids, section_progress = compute_section_features(
        sections, n_frames=n_mel_frames, hop_length=hop_length, sample_rate=sr,
    )
    structure_features = torch.cat([
        structure_features,
        section_ids.float().unsqueeze(0) / 5.0,
        section_progress.unsqueeze(0),
    ], dim=0)  # [8, T]

    song_duration_secs = waveform.shape[-1] / sr

    # --- Load / build models ---
    if sequence_checkpoint is not None:
        seq_module = _load_sequence_module(Path(sequence_checkpoint))
    else:
        seq_module = _make_default_sequence_module()
    seq_module = seq_module.to(device_obj).eval()

    # --- Windowed full-song inference ---
    # Hop through the song in fixed beat-windows; re-encode audio per window.
    # Saber state and song-absolute current_beat carry across window boundaries.
    n_frames = mel.shape[1]
    mel_tensor = mel.unsqueeze(0).to(device_obj)
    struct_tensor = (
        structure_features.unsqueeze(0).to(device_obj)
        if structure_features is not None else None
    )

    duration_beats = song_duration_secs * (bpm / 60.0)
    # context_frames ≈ 256 covers ~3 sec ≈ 6 beats at 120 BPM. Use a 4-beat
    # window so each window's events stay inside the audio context. Tunable.
    window_beats = float(_compute_window_beats(context_frames, bpm))
    diff_idx = DIFFICULTY_MAP.get(difficulty, 3)
    genre_idx = GENRE_MAP.get(genre, 0)
    diff_tensor = torch.tensor([diff_idx], device=device_obj)
    genre_tensor = torch.tensor([genre_idx], device=device_obj)
    mapper_tensor = torch.tensor([mapper_id], device=device_obj)
    use_mapper = (seq_module.sequence_model.num_mappers or 0) > 0

    from beatsaber_automapper.generation.beam_search_v6 import sample_swing_events

    all_events = []
    resume_state = None
    window_start_beat = 0.0

    logger.info(
        "V6 windowed inference: duration=%.1f beats, window=%.1f beats, max_events=%d",
        duration_beats, window_beats, max_events,
    )

    # The model's grammar state owns the song-absolute beat clock: each Δt token
    # advances `current_beat`, and events are timestamped with that value. We let
    # the model decide how far each window covers and advance the audio context
    # window from the resume state's beat after each pass. This is robust to
    # silent intros (large initial Δt) and lets the model self-pace.
    # Frames per beat (for activity context computation)
    _frames_per_beat = _FRAMES_PER_SEC * 60.0 / bpm

    while window_start_beat < duration_beats and len(all_events) < max_events:
        center_beat = window_start_beat + window_beats / 2.0
        center_frame = int(center_beat * _FRAMES_PER_SEC * 60.0 / bpm)

        mel_window = _mel_window(mel_tensor, center_frame, context_frames, n_frames)
        struct_window = (
            _mel_window(struct_tensor, center_frame, context_frames, n_frames)
            if struct_tensor is not None else None
        )
        phrase_window_mel = _mel_window(mel_tensor, center_frame, phrase_frames, n_frames)
        phrase_window_struct = (
            _mel_window(struct_tensor, center_frame, phrase_frames, n_frames)
            if struct_tensor is not None else None
        )

        with torch.no_grad():
            audio_features = seq_module.audio_encoder(
                mel_window, structure_features=struct_window,
            )
            phrase_audio = seq_module.audio_encoder(
                phrase_window_mel, structure_features=phrase_window_struct,
            )
            phrase_emb = phrase_audio.mean(dim=1)

        # Song position fraction (0→1 across full song)
        song_pos_frac_val = float(min(center_beat / max(duration_beats, 1.0), 1.0))
        song_pos_frac_tensor = torch.tensor(
            [song_pos_frac_val], dtype=torch.float32, device=device_obj,
        )

        # Section ID at the window centre frame
        section_id_val = 0
        if (struct_tensor is not None
                and struct_tensor.shape[1] >= 7 and struct_tensor.shape[2] >= 1):
            cf = max(0, min(center_frame, struct_tensor.shape[2] - 1))
            sec_norm = float(struct_tensor[0, 6, cf].item())
            section_id_val = min(int(round(sec_norm * 5.0)), 5)
        section_id_tensor = torch.tensor([section_id_val], dtype=torch.long, device=device_obj)

        # Activity prediction: suppress EOS in beat slots the predictor marks active
        activity_probs: torch.Tensor | None = None
        context_beats_window = context_frames / _FRAMES_PER_SEC * bpm / 60.0
        activity_beat_start = center_beat - context_beats_window / 2.0
        if (
            hasattr(seq_module, "activity_predictor")
            and seq_module.activity_predictor is not None
        ):
            with torch.no_grad():
                act_logits = seq_module.activity_predictor(audio_features)  # [1, N_BEATS]
                activity_probs = torch.sigmoid(act_logits).squeeze(0)       # [N_BEATS]

        # Per-window budget: scale with window size, cap to prevent budget exhaustion.
        # 256 was too high — a stall at Δt=0 could consume the entire budget in one window.
        remaining = max_events - len(all_events)
        per_window_cap = min(remaining, 128)

        # Stop sampling once the model crosses 1.5 window-widths past the audio
        # centre (so each pass covers ~one window of beats; rest comes from
        # subsequent windows with re-centred audio context).
        stop_beat = window_start_beat + 1.5 * window_beats

        result = sample_swing_events(
            model=seq_module.sequence_model,
            audio_features=audio_features,
            difficulty=diff_tensor,
            genre=genre_tensor,
            max_events=per_window_cap,
            max_tokens=per_window_cap * 8,
            temperature=temperature,
            top_p=top_p,
            device=device_obj,
            mapper_id=mapper_tensor if use_mapper else None,
            phrase_emb=phrase_emb,
            initial_state=resume_state,
            stop_at_beat=stop_beat,
            activity_probs=activity_probs,
            activity_beat_start=activity_beat_start,
            activity_beat_width=context_beats_window,
            song_pos_frac=song_pos_frac_tensor,
            section_id=section_id_tensor,
        )

        all_events.extend(result.events)
        resume_state = result.final_state

        next_beat = result.final_state.current_beat
        logger.info(
            "  window @beat=%.2f → emitted %d events, model advanced to beat %.2f",
            window_start_beat, len(result.events), next_beat,
        )

        # Advance to where the model actually is, never backwards
        if next_beat <= window_start_beat:
            # Model emitted EOS or got stuck — advance manually so we don't loop.
            # CRITICAL: also sync the grammar state's beat clock to the new window
            # start so that the next window's audio context stays aligned with the
            # model's internal time. Without this, every subsequent window sees audio
            # centered at beat N but the model's Δt predictions anchor on beat M<N,
            # causing all new events to stack at the stall point indefinitely.
            window_start_beat += window_beats
            resume_state.current_beat = window_start_beat
        else:
            window_start_beat = next_beat

    logger.info("Generated %d events across full song", len(all_events))

    # Convert events directly to a DifficultyBeatmap
    beatmap = _events_to_beatmap(all_events)
    logger.info(
        "Decoded: %d notes, %d arcs, %d chains, %d bombs",
        len(beatmap.color_notes), len(beatmap.sliders),
        len(beatmap.burst_sliders), len(beatmap.bomb_notes),
    )

    # --- Postprocess ---
    beatmap = postprocess_beatmap(beatmap, difficulty=difficulty, bpm=bpm,
                                  song_duration_secs=song_duration_secs)

    # --- Lighting (rule-based) ---
    from beatsaber_automapper.generation.lighting_rules import (
        generate_lighting_events as _gen_light,
    )
    basic_events, boost_events = _gen_light(
        structure_features=structure_features,
        bpm=bpm,
        sample_rate=sample_rate,
        hop_length=hop_length,
    )
    beatmap.basic_events.extend(basic_events)
    beatmap.color_boost_events.extend(boost_events)

    # --- Export ---
    diff_name = difficulty
    all_beatmaps = {diff_name: beatmap}

    chroma_beatmap_dicts = {}
    if beatmap.basic_events:
        from beatsaber_automapper.generation.chroma import add_chroma_colors
        plain_events = [
            {"b": e.beat, "et": e.event_type, "i": e.value, "f": e.float_value}
            for e in beatmap.basic_events
        ]
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


# ---------------------------------------------------------------------------
# V7 generation pipeline: Demucs + MERT + Stage1 + PhraseIndex + Stage2
# ---------------------------------------------------------------------------


def generate_v7_level(
    audio_path: Path | str,
    output_path: Path | str,
    beat_checkpoint: Path | str,
    layout_checkpoint: Path | str,
    difficulty: str = "Expert",
    genre: str = "unknown",
    song_name: str | None = None,
    song_author: str = "Unknown Artist",
    bpm: float | None = None,
    beat_threshold_left: float = 0.55,
    beat_threshold_right: float = 0.55,
    beat_nms_radius: int = 1,
    beat_energy_scale: float = 0.15,
    temperature: float = 0.9,
    top_p: float = 0.95,
    phrase_similarity: float = 0.85,
    device: str | None = None,
    sample_rate: int = 44100,
    hop_length: int = 512,
    n_mels: int = 80,
) -> Path:
    """Generate a Beat Saber level using the V7 pipeline.

    V7 pipeline:
      1. Demucs source separation → drum stem + melody stem
      2. MERT-v1-95M feature extraction → beat-aligned embeddings
      3. Stage 1 BeatClassifier → onset schedule (which beats have notes)
      4. PhraseIndex built from melody MERT fingerprints
      5. Stage 2 LayoutModel → spatial tokens per onset (with phrase retrieval)
      6. Assemble swing-event stream → postprocess → export

    Args:
        audio_path:          Input audio file.
        output_path:         Output .zip path.
        beat_checkpoint:     Path to BeatLitModule checkpoint.
        layout_checkpoint:   Path to LayoutPhraseLitModule checkpoint.
        difficulty:          Difficulty name.
        genre:               Genre string for conditioning.
        song_name:           Override song name (defaults to filename stem).
        song_author:         Song author for metadata.
        bpm:                 BPM override (auto-detected if None).
        beat_threshold_left: P(left note) threshold for Stage 1.
        beat_threshold_right: P(right note) threshold for Stage 1.
        temperature:         Stage 2 sampling temperature.
        top_p:               Stage 2 nucleus top-p.
        phrase_similarity:   PhraseIndex cosine similarity threshold.
        device:              Torch device (auto if None).
        sample_rate:         Audio sample rate.
        hop_length:          Mel spectrogram hop length.
        n_mels:              Mel bands.

    Returns:
        Path to the generated .zip file.
    """
    import torch
    from beatsaber_automapper.data.mert_encoder import (
        extract_features as mert_extract,
        pool_to_beat_grid,
        phrase_fingerprints as compute_fingerprints,
        BEAT_SUBDIV,
    )
    from beatsaber_automapper.data.stem_separator import separate as demucs_separate, DEMUCS_SR
    from beatsaber_automapper.data.audio import detect_bpm, load_audio
    from beatsaber_automapper.data.saber_state import compute_saber_states
    from beatsaber_automapper.data.swing_tokenizer import (
        HAND_LEFT, HAND_RIGHT,
        _DT_BINS, DT_BASE,
        SwingEventTokenizer,
    )
    from beatsaber_automapper.data.tokenizer import DIFFICULTY_MAP, GENRE_MAP
    from beatsaber_automapper.generation.phrase_index import PhraseIndex, NotePattern
    from beatsaber_automapper.generation.postprocess import postprocess_beatmap
    from beatsaber_automapper.generation.export import package_level
    from beatsaber_automapper.training.beat_module import BeatLitModule
    from beatsaber_automapper.training.layout_module import LayoutPhraseLitModule
    from beatsaber_automapper.data.beatmap import ColorNote, BombNote, DifficultyBeatmap
    from beatsaber_automapper.data.swing_tokenizer import (
        NOTE, ARC_HEAD, ARC_TAIL, CHAIN_HEAD, CHAIN_TAIL, BOMB,
        X_BASE, Y_BASE, DIR_BASE, ANGLE_BASE, MU_BASE, SLICE_BASE, SQUISH_BASE,
        _ANGLE_BINS, _MU_BINS, _SQUISH_BINS, _SLICE_MIN,
    )
    from beatsaber_automapper.data.layout_dataset import (
        HAND_LEFT_IDX, HAND_RIGHT_IDX, MAX_PHRASE_SLOTS,
    )

    audio_path  = Path(audio_path)
    output_path = Path(output_path)
    if song_name is None:
        song_name = audio_path.stem

    if device is None:
        device_obj = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device_obj = torch.device(device)

    logger.info("V7 generate_v7_level: %s → %s (device=%s)", audio_path.name, output_path.name, device_obj)

    # ---- 1. Audio loading + BPM ----
    waveform, src_sr = load_audio(audio_path, target_sr=DEMUCS_SR)
    if bpm is None:
        bpm = detect_bpm(waveform, sample_rate=src_sr)
    logger.info("BPM: %.1f", bpm)

    song_duration_secs = waveform.shape[-1] / src_sr
    total_beats = song_duration_secs * bpm / 60.0

    # ---- 2. Source separation ----
    logger.info("Separating audio with Demucs …")
    stems = demucs_separate(waveform, src_sr, device=str(device_obj))

    # ---- 3. MERT feature extraction ----
    logger.info("Extracting MERT features …")
    drum_mert = mert_extract(stems["drums"], DEMUCS_SR, device=str(device_obj))
    mix_mert  = mert_extract(stems["other"], DEMUCS_SR, device=str(device_obj))

    drum_beat = pool_to_beat_grid(drum_mert, bpm, total_beats, BEAT_SUBDIV)  # [N, 768]
    mix_beat  = pool_to_beat_grid(mix_mert,  bpm, total_beats, BEAT_SUBDIV)  # [N, 768]
    n_slots   = drum_beat.shape[0]

    song_emb_vec  = mix_beat.mean(0)   # [768] — full-song embedding

    fingerprints, boundaries = compute_fingerprints(mix_beat, beats_per_phrase=16, subdiv=BEAT_SUBDIV)

    # ---- 4. Load models ----
    logger.info("Loading Stage 1 BeatClassifier …")
    beat_module = BeatLitModule.load_from_checkpoint(str(beat_checkpoint))
    beat_module = beat_module.to(device_obj).eval()

    logger.info("Loading Stage 2 LayoutPhraseModel …")
    layout_module = LayoutPhraseLitModule.load_from_checkpoint(str(layout_checkpoint))
    layout_module = layout_module.to(device_obj).eval()

    diff_idx  = DIFFICULTY_MAP.get(difficulty, 3)
    genre_idx = GENRE_MAP.get(genre, 0)
    diff_t    = torch.tensor([diff_idx],  dtype=torch.long, device=device_obj)
    genre_t   = torch.tensor([genre_idx], dtype=torch.long, device=device_obj)

    # ---- 5. Stage 1: onset schedule (windowed, matching training window=128) ----
    logger.info("Running Stage 1 BeatClassifier …")
    _BEAT_WIN = 128   # training window size; pos_emb max_len=512, so ≤512 is safe
    drum_gpu = drum_beat.to(device_obj)   # [N, 768]
    mix_gpu  = mix_beat.to(device_obj)    # [N, 768]
    beat_probs_parts: list[torch.Tensor] = []
    with torch.no_grad():
        for s in range(0, n_slots, _BEAT_WIN):
            e   = min(s + _BEAT_WIN, n_slots)
            d_w = drum_gpu[s:e].unsqueeze(0)   # [1, W, 768]
            m_w = mix_gpu[s:e].unsqueeze(0)    # [1, W, 768]
            logits_w = beat_module(d_w, m_w, diff_t, slot_offset=s)  # [1, W, 2]
            beat_probs_parts.append(torch.sigmoid(logits_w.squeeze(0)))
    beat_probs = torch.cat(beat_probs_parts, dim=0)   # [N, 2]

    # ----- Section-aware threshold (replaces flat energy modulation) -----
    # The model outputs a near-uniform probability distribution with no clear
    # bimodal gap, so a fixed threshold produces a metronome. Instead we:
    #   1. Detect song sections (intro/verse/chorus/drop/bridge/outro) once.
    #   2. Map each section type to a threshold that reflects natural map density:
    #      drops get low thresholds (dense), intros/outros get high (sparse).
    #   3. Apply per-slot thresholds derived from section boundaries.
    # This matches how human mappers behave: they drop density in breakdowns and
    # go dense in drops, rather than sustaining a constant 6 NPS throughout.
    _SECTION_THRESHOLDS = {
        "drop":   0.38,   # loudest section — many notes
        "chorus": 0.44,
        "verse":  0.52,
        "bridge": 0.58,
        "intro":  0.68,   # sparse opening
        "outro":  0.72,   # sparse ending
    }

    try:
        from beatsaber_automapper.data.audio import detect_sections as _detect_sections
        sections = _detect_sections(waveform, sample_rate=src_sr)
    except Exception:
        sections = [("verse", 0.0, song_duration_secs)]

    logger.info("Sections: %s", [(t, f"{s:.0f}s", f"{e:.0f}s") for t, s, e in sections])

    # Build per-slot threshold vector from section labels
    beats_per_sec = bpm / 60.0
    thr_L = torch.full((n_slots,), beat_threshold_left,  device=device_obj)
    thr_R = torch.full((n_slots,), beat_threshold_right, device=device_obj)
    for sec_type, sec_start, sec_end in sections:
        base = _SECTION_THRESHOLDS.get(sec_type, beat_threshold_left)
        slot_s = max(0, int(sec_start * beats_per_sec * BEAT_SUBDIV))
        slot_e = min(n_slots, int(sec_end   * beats_per_sec * BEAT_SUBDIV) + 1)
        thr_L[slot_s:slot_e] = base
        thr_R[slot_s:slot_e] = base

    # ----- Non-maximum suppression within ±beat_nms_radius -----
    def _nms(probs: torch.Tensor, thresh: torch.Tensor, radius: int) -> set[int]:
        keep_mask = probs >= thresh
        if radius <= 0 or not keep_mask.any():
            return set(keep_mask.nonzero(as_tuple=True)[0].tolist())
        N = probs.shape[0]
        pooled = torch.nn.functional.max_pool1d(
            probs.unsqueeze(0).unsqueeze(0),
            kernel_size=2 * radius + 1, stride=1, padding=radius,
        ).squeeze(0).squeeze(0)[:N]
        keep = keep_mask & (probs >= pooled)
        return set(keep.nonzero(as_tuple=True)[0].tolist())

    left_onsets  = _nms(beat_probs[:, 0].to(device_obj), thr_L, beat_nms_radius)
    right_onsets = _nms(beat_probs[:, 1].to(device_obj), thr_R, beat_nms_radius)
    logger.info("Stage 1: %d left onsets, %d right onsets across %d slots",
                len(left_onsets), len(right_onsets), n_slots)

    # ---- 6. Build PhraseIndex ----
    phrase_index = PhraseIndex(similarity_threshold=phrase_similarity)
    phrase_index.build(mix_beat, boundaries)

    # ---- 7. Section embeddings (pre-compute per slot) ----
    # Simple: each phrase window is its own "section"
    # For finer-grained section boundaries use detect_sections if needed

    # ---- 8. Stage 2: phrase-level layout generation ----
    logger.info("Running Stage 2 LayoutPhraseModel …")

    from beatsaber_automapper.data.swing_tokenizer import _SwingEvent
    from beatsaber_automapper.data.beat_grid import BEAT_SUBDIV as _BEAT_SUBDIV

    all_events: list[_SwingEvent] = []
    max_phrase_slots_inf  = layout_module.model.max_phrase_slots
    max_song_phrases_inf  = getattr(layout_module.model, "max_song_phrases", 0)

    # Cross-phrase context: last ctx_len tokens from prior phrase.
    _ctx_len = getattr(layout_module.model, "ctx_len", 0)
    _prev_ctx_toks:  list[int] = []
    _prev_ctx_slots: list[int] = []
    _prev_ctx_hands: list[int] = []

    # Song-memory: all phrase fingerprints padded to max_song_phrases.
    # Computed once and reused for every phrase in this song.
    if max_song_phrases_inf > 0 and fingerprints is not None and len(fingerprints) > 0:
        N_fp = min(len(fingerprints), max_song_phrases_inf)
        _song_fps_pad  = torch.zeros(1, max_song_phrases_inf, 768, device=device_obj)
        _song_fp_mask_pad = torch.zeros(1, max_song_phrases_inf, dtype=torch.bool, device=device_obj)
        _song_fps_pad[0, :N_fp]      = fingerprints[:N_fp].to(device_obj, dtype=torch.float32)
        _song_fp_mask_pad[0, :N_fp]  = True
        _song_fps_t      = _song_fps_pad
        _song_fp_mask_t  = _song_fp_mask_pad
    else:
        _song_fps_t     = None
        _song_fp_mask_t = None

    for phrase_idx, (slot_start, slot_end) in enumerate(boundaries):
        # Collect onset schedule for this phrase: (slot_in_phrase, hand_idx) sorted
        # by (slot, hand) so LEFT comes before RIGHT at the same beat — matches
        # the training-data event ordering in LayoutPhraseDataset.
        onset_schedule: list[tuple[int, int]] = []
        for slot in range(slot_start, min(slot_end, n_slots)):
            sip = slot - slot_start
            if slot in left_onsets:
                onset_schedule.append((sip, HAND_LEFT_IDX))
            if slot in right_onsets:
                onset_schedule.append((sip, HAND_RIGHT_IDX))

        if not onset_schedule:
            continue

        # Build phrase MERT input tensor, padded to max_phrase_slots.
        real_slots = min(slot_end, n_slots) - slot_start
        p_len = min(real_slots, max_phrase_slots_inf)
        phrase_mert_t = mix_beat[slot_start:slot_start + p_len].unsqueeze(0).to(device_obj)  # [1,p,768]
        phrase_mask_t = torch.ones(1, p_len, dtype=torch.bool, device=device_obj)
        if p_len < max_phrase_slots_inf:
            pad_feat = torch.zeros(1, max_phrase_slots_inf - p_len, mix_beat.shape[-1],
                                   device=device_obj)
            pad_mask = torch.zeros(1, max_phrase_slots_inf - p_len, dtype=torch.bool,
                                   device=device_obj)
            phrase_mert_t = torch.cat([phrase_mert_t, pad_feat], dim=1)
            phrase_mask_t = torch.cat([phrase_mask_t, pad_mask], dim=1)

        with torch.no_grad():
            flat_tokens = layout_module.model.generate_phrase(
                phrase_mert    = phrase_mert_t,
                phrase_mask    = phrase_mask_t,
                onset_schedule = onset_schedule,
                difficulty     = diff_t,
                genre          = genre_t,
                temperature    = temperature,
                top_p          = top_p,
                context_tokens = _prev_ctx_toks  if _ctx_len > 0 else None,
                context_slots  = _prev_ctx_slots if _ctx_len > 0 else None,
                context_hands  = _prev_ctx_hands if _ctx_len > 0 else None,
                song_fps       = _song_fps_t,
                song_fp_mask   = _song_fp_mask_t,
            )

        phrase_events = _decode_phrase_tokens(flat_tokens, onset_schedule, slot_start)
        all_events.extend(phrase_events)

        # Update cross-phrase context buffer for next phrase.
        if _ctx_len > 0:
            _prev_ctx_toks  = flat_tokens[-_ctx_len:]
            # Slots/hands need to come from the onset_schedule alignment.
            # Rebuild from onset_schedule consumed by _decode_phrase_tokens.
            ctx_slots: list[int] = []
            ctx_hands: list[int] = []
            tok_per_onset = []
            ti = 0
            from beatsaber_automapper.data.swing_tokenizer import (
                BOMB as _BOMB, CHAIN_TAIL as _CHAIN_TAIL, KIND_BASE as _KB, KIND_COUNT as _KC,
            )
            for sip, hidx in onset_schedule:
                if ti >= len(flat_tokens): break
                k = flat_tokens[ti]
                n_tok = 3 if (k == _BOMB) else 4 if (k == _CHAIN_TAIL) else 5
                n_tok = min(n_tok, len(flat_tokens) - ti)
                tok_per_onset.append((sip, hidx, n_tok))
                ti += n_tok
            # Flatten slot/hand per token
            all_ctx_slots: list[int] = []
            all_ctx_hands: list[int] = []
            for sip, hidx, n in tok_per_onset:
                all_ctx_slots.extend([sip] * n)
                all_ctx_hands.extend([hidx] * n)
            _prev_ctx_slots = all_ctx_slots[-_ctx_len:]
            _prev_ctx_hands = all_ctx_hands[-_ctx_len:]

    logger.info("Generated %d events", len(all_events))

    # ---- 9. Assemble beatmap ----
    beatmap = _events_to_beatmap(all_events)
    logger.info("Decoded: %d notes, %d arcs, %d chains, %d bombs",
                len(beatmap.color_notes), len(beatmap.sliders),
                len(beatmap.burst_sliders), len(beatmap.bomb_notes))

    beatmap = postprocess_beatmap(beatmap, difficulty=difficulty, bpm=bpm,
                                  song_duration_secs=song_duration_secs)

    # ---- 10. Lighting + export ----
    from beatsaber_automapper.data.audio import (
        compute_structure_features, detect_sections, compute_section_features,
    )
    # Re-use existing structure features for lighting
    mel = extract_mel_spectrogram(waveform, src_sr, n_mels=n_mels, hop_length=hop_length)
    structure_features = compute_structure_features(waveform, src_sr, hop_length=hop_length, n_mels=n_mels)
    if structure_features.shape[1] > mel.shape[1]:
        structure_features = structure_features[:, :mel.shape[1]]

    from beatsaber_automapper.generation.lighting_rules import generate_lighting_events
    basic_events, boost_events = generate_lighting_events(
        structure_features=structure_features, bpm=bpm,
        sample_rate=sample_rate, hop_length=hop_length,
    )
    beatmap.basic_events.extend(basic_events)
    beatmap.color_boost_events.extend(boost_events)

    from beatsaber_automapper.generation.chroma import add_chroma_colors
    chroma_events = None
    if beatmap.basic_events:
        plain_events = [{"b": e.beat, "et": e.event_type, "i": e.value, "f": e.float_value}
                        for e in beatmap.basic_events]
        chroma_events = add_chroma_colors(
            events=plain_events, structure_features=structure_features,
            bpm=bpm, sample_rate=sample_rate, hop_length=hop_length, genre=genre,
        )

    output_path = package_level(
        beatmaps={difficulty: beatmap},
        audio_path=audio_path,
        output_path=output_path,
        song_name=song_name,
        song_author=song_author,
        bpm=bpm,
        chroma_events={difficulty: chroma_events},
    )
    return output_path


def _decode_phrase_tokens(
    tokens: list[int],
    onset_schedule: list[tuple[int, int]],
    slot_start: int,
) -> list["_SwingEvent"]:
    """Decode the flat token list from LayoutPhraseModel.generate_phrase into _SwingEvent objects.

    `tokens` is the BOS-stripped output of generate_phrase. Each entry in
    onset_schedule is (slot_in_phrase, hand_idx) and corresponds to one note;
    tokens are consumed in the same order the schedule was presented to the model:
      - BOMB      → 3 tokens: KIND X Y
      - CHAIN_TAIL → 4 tokens: KIND X Y FIELD_D(squish)
      - others     → 5 tokens: KIND X Y DIR FIELD_D
    """
    from beatsaber_automapper.data.swing_tokenizer import (
        _SwingEvent,
        BOMB, CHAIN_TAIL, NOTE, ARC_HEAD, ARC_TAIL,
        KIND_BASE, KIND_COUNT,
        X_BASE, X_COUNT, Y_BASE, Y_COUNT,
        DIR_BASE,
        ANGLE_BASE, ANGLE_COUNT, MU_BASE, MU_COUNT,
        SQUISH_BASE, SQUISH_COUNT, SLICE_BASE, SLICE_COUNT,
        HAND_LEFT, HAND_RIGHT,
    )
    from beatsaber_automapper.data.layout_dataset import HAND_LEFT_IDX
    from beatsaber_automapper.data.beat_grid import BEAT_SUBDIV

    events: list[_SwingEvent] = []
    i = 0
    for slot_in_phrase, hand_idx in onset_schedule:
        if i >= len(tokens):
            break
        beat_pos = (slot_start + slot_in_phrase) / BEAT_SUBDIV
        hand = HAND_LEFT if hand_idx == HAND_LEFT_IDX else HAND_RIGHT

        kind_tok = tokens[i]; i += 1
        if not (KIND_BASE <= kind_tok < KIND_BASE + KIND_COUNT):
            kind_tok = KIND_BASE  # clamp to NOTE

        x_tok = tokens[i]     if i     < len(tokens) else X_BASE
        y_tok = tokens[i + 1] if i + 1 < len(tokens) else Y_BASE
        i += 2

        x = max(0, min(x_tok - X_BASE, X_COUNT - 1))
        y = max(0, min(y_tok - Y_BASE, Y_COUNT - 1))
        direction = 0
        field_d   = 0

        if kind_tok == BOMB:
            pass
        elif kind_tok == CHAIN_TAIL:
            if i < len(tokens):
                fd_tok  = tokens[i]; i += 1
                field_d = max(0, min(fd_tok - SQUISH_BASE, SQUISH_COUNT - 1))
        else:
            if i < len(tokens):
                dir_tok   = tokens[i]; i += 1
                direction = max(0, min(dir_tok - DIR_BASE, 8))
            if i < len(tokens):
                fd_tok = tokens[i]; i += 1
                if kind_tok == NOTE:
                    field_d = max(0, min(fd_tok - ANGLE_BASE,  ANGLE_COUNT  - 1))
                elif kind_tok in (ARC_HEAD, ARC_TAIL):
                    field_d = max(0, min(fd_tok - MU_BASE,     MU_COUNT     - 1))
                else:  # CHAIN_HEAD
                    field_d = max(0, min(fd_tok - SLICE_BASE,  SLICE_COUNT  - 1))

        events.append(_SwingEvent(
            beat=beat_pos, hand=hand, kind=kind_tok,
            x=x, y=y, direction=direction, field_d=field_d,
        ))
    return events


def _decode_spatial_tokens(
    tokens: list[int],
    beat: float,
    hand: int,
) -> "_SwingEvent | None":
    """Decode a spatial token list into a _SwingEvent for assembly."""
    from beatsaber_automapper.data.swing_tokenizer import (
        _SwingEvent,
        NOTE, ARC_HEAD, ARC_TAIL, CHAIN_HEAD, CHAIN_TAIL, BOMB,
        KIND_BASE, KIND_COUNT,
        X_BASE, X_COUNT, Y_BASE, Y_COUNT,
        DIR_BASE, DIR_COUNT,
        ANGLE_BASE, ANGLE_COUNT, MU_BASE, MU_COUNT,
        SLICE_BASE, SLICE_COUNT, SQUISH_BASE, SQUISH_COUNT,
    )
    if len(tokens) < 3:
        return None

    kind_tok = tokens[0]
    if not (KIND_BASE <= kind_tok < KIND_BASE + KIND_COUNT):
        return None
    kind = kind_tok

    x = max(0, min(tokens[1] - X_BASE, 3)) if len(tokens) > 1 else 0
    y = max(0, min(tokens[2] - Y_BASE, 2)) if len(tokens) > 2 else 0
    direction = 0
    field_d   = 0

    if kind == BOMB:
        pass
    elif kind == CHAIN_TAIL:
        if len(tokens) > 3:
            field_d = max(0, min(tokens[3] - SQUISH_BASE, SQUISH_COUNT - 1))
    else:
        if len(tokens) > 3:
            direction = max(0, min(tokens[3] - DIR_BASE, 8))
        if len(tokens) > 4:
            fd = tokens[4]
            if kind == NOTE:
                field_d = max(0, min(fd - ANGLE_BASE, ANGLE_COUNT - 1))
            elif kind in (ARC_HEAD, ARC_TAIL):
                field_d = max(0, min(fd - MU_BASE, MU_COUNT - 1))
            else:
                field_d = max(0, min(fd - SLICE_BASE, SLICE_COUNT - 1))

    return _SwingEvent(beat=beat, hand=hand, kind=kind,
                       x=x, y=y, direction=direction, field_d=field_d)
