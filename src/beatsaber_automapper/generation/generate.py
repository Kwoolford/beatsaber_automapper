"""End-to-end inference pipeline.

Orchestrates the full generation flow:
    Audio -> AudioEncoder -> Stage 1 (onsets) -> Stage 2 (notes)
    -> Stage 3 (lighting, optional) -> export

Supports loading trained Lightning checkpoints for each stage model,
or running in "random" mode with untrained weights for testing.
"""

from __future__ import annotations

import logging
import os
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


# Per-section Stage-1 onset thresholds. Loud sections (drop/chorus) get a low
# threshold (dense); quiet ones (intro/outro) historically got a high one (sparse).
# The high intro/outro values are exactly what silenced real drops — see
# `_build_section_threshold_vector` and docs/v8_0_poc_findings.md.
_SECTION_THRESHOLDS: dict[str, float] = {
    "drop":   0.38,   # loudest section — many notes
    "chorus": 0.44,
    "verse":  0.52,
    "bridge": 0.58,
    "intro":  0.68,   # sparse opening
    "outro":  0.72,   # sparse ending
}


def _build_section_threshold_vector(
    sections: "list[tuple[str, float, float]]",
    n_slots: int,
    base_left: float,
    base_right: float,
    beats_per_sec: float,
    subdiv: int,
    section_gate: str = "loud_only",
) -> "tuple[torch.Tensor, torch.Tensor]":
    """Per-slot Stage-1 onset thresholds derived from detected sections.

    ``section_gate`` controls how sections modulate the base threshold:

    * ``"loud_only"`` (default): a section may only *lower* the threshold (make a
      loud part denser); it can never *raise* it above ``base_*``. This guarantees
      no section — however mislabeled — can silence a real onset. Fixes the
      V8-0-confirmed silent-drop failure mode.
    * ``"off"``: flat ``base_*`` everywhere; sections never touch notes.
    * ``"legacy"``: sections set the threshold outright (incl. raising intro/outro
      to 0.68/0.72). Kept for A/B comparison only.

    Returns ``(thr_left, thr_right)`` float tensors of shape ``[n_slots]``.
    """
    if section_gate not in ("off", "loud_only", "legacy"):
        raise ValueError(f"section_gate must be off|loud_only|legacy, got {section_gate!r}")
    thr_L = torch.full((n_slots,), float(base_left))
    thr_R = torch.full((n_slots,), float(base_right))
    if section_gate == "off":
        return thr_L, thr_R
    for sec_type, sec_start, sec_end in sections:
        base_L = _SECTION_THRESHOLDS.get(sec_type, base_left)
        base_R = _SECTION_THRESHOLDS.get(sec_type, base_right)
        if section_gate == "loud_only":
            base_L = min(base_L, base_left)     # never silence — only densify
            base_R = min(base_R, base_right)
        slot_s = max(0, int(sec_start * beats_per_sec * subdiv))
        slot_e = min(n_slots, int(sec_end * beats_per_sec * subdiv) + 1)
        thr_L[slot_s:slot_e] = base_L
        thr_R[slot_s:slot_e] = base_R
    return thr_L, thr_R


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


def _oracle_bpm(audio_path) -> float | None:
    """DIAGNOSTIC ONLY: the song's TRUE bpm, from a {song_id: bpm} JSON.

    ** NOT SHIPPABLE. ** This hands the generator an answer it cannot have at
    inference time. It exists to settle one question that no observational metric
    can: how much of our audio misalignment is the tempo estimate?

    The evidence that made it necessary (2026-08-02, axis A8): our detected bpm is
    exact on **1 of 21** eval songs. Median error is 0.74% and four songs land at
    2/3 of the true tempo. Every note is then placed on a 1/4-beat slot grid built
    from that bpm, so on most songs the grid slides against the music as the song
    plays. Human maps sit on the same 1/4-beat grid we do (557 of 561 notes on
    1f767) and score 0.930 onset precision to our 0.76 — **the grid is not too
    coarse, it is in the wrong place.**

    `detect_bpm` also throws away librosa's beat POSITIONS (`tempo, _ = beat_track`)
    and the grid is then anchored at t=0, so the phase is wrong independently of
    the tempo. This oracle fixes only the tempo; if alignment does not recover with
    a perfect tempo, phase is the remaining suspect.

    Set `BEAT_BPM_ORACLE=/path/to/bpm.json`, keyed by audio file stem.
    """
    path_env = os.environ.get("BEAT_BPM_ORACLE")
    if not path_env:
        return None
    try:
        import json as _json
        table = _json.loads(Path(path_env).read_text())
    except Exception as exc:  # noqa: BLE001
        logger.warning("BEAT_BPM_ORACLE unreadable (%s) — falling back to detection", exc)
        return None
    key = Path(audio_path).stem
    val = table.get(key)
    if val is None:
        # A silent miss here would look like a successful oracle run on a song that
        # never got one. Say so loudly instead.
        logger.warning("BEAT_BPM_ORACLE has no entry for %r — DETECTED bpm used", key)
        return None
    logger.info("BEAT_BPM_ORACLE: %s -> %.3f bpm (true tempo, diagnostic only)",
                key, float(val))
    return float(val)


def _quantize_to_beat_grid(
    onset_frames: list[int],
    bpm: float,
    sample_rate: int,
    hop_length: int,
    max_subdivision: int = 8,
) -> list[int]:
    """Snap onset frames to nearest beat subdivision.

    ★ THIS IS WHERE OUR TIMING SCATTER COMES FROM (measured 2026-08-01, axis A8).

    The grid spacing at the default 1/8 is ~46ms, so snapping displaces a note by
    up to +-23ms, uniformly. Predicted MAD 11.6ms; **measured 11.7ms** on the arms
    that reach an onset at all. Human maps sit at 8.7ms and their offsets are a
    unimodal peak on the onset, while ours are FLAT across the whole tolerance
    window — the signature of a grid, not of musical timing.

    It is worse than a fixed +-23ms, because the grid is built from the DETECTED
    bpm, which is exact on **1 of 21** songs in the eval set (median error 0.74%,
    and four songs land at 2/3 of the true tempo). A 0.74% error slides the grid
    ~1% of a beat per beat, so it walks away from the music as the song goes on.
    Together: "the consistent beat of the song is not where the notes are played"
    (Kyle, 2026-08-01) is manufactured here, downstream of the model.

    Stage-1 frames are hop_length/sample_rate = 11.6ms apart, so the frames
    themselves carry only ~2.9ms of quantisation MAD — the model's timing is four
    times finer than what this function leaves of it.

    `BEAT_GRID_SUBDIV` overrides the subdivision: 16 halves the displacement, 0
    disables snapping entirely (frame resolution only). Default 8 = prior
    behaviour, unchanged until a sweep says otherwise.

    Args:
        onset_frames: List of frame indices.
        bpm: Song BPM.
        sample_rate: Audio sample rate.
        hop_length: Spectrogram hop length.
        max_subdivision: Finest subdivision (8 = eighth notes).

    Returns:
        Sorted, deduplicated list of quantized frame indices.
    """
    subdiv = int(os.environ.get("BEAT_GRID_SUBDIV", str(max_subdivision)))
    if subdiv <= 0:
        # Snapping off: keep the model's own frame-resolution timing.
        return sorted(set(onset_frames))
    if not onset_frames or bpm <= 0:
        return onset_frames

    frames_per_beat = (60.0 / bpm) * sample_rate / hop_length
    grid_spacing = frames_per_beat / subdiv

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


def _audio_onset_times(waveform: torch.Tensor, sample_rate: int) -> "np.ndarray | None":
    """Onset times (s) detected directly from the audio, or None.

    Independent of Stage-1. Uses librosa on the mix — deliberately NOT the
    per-stem union that A8 scores against, since consuming the evaluation's own
    detector would be optimising the metric (the `h_dist` failure).
    """
    try:
        import librosa
    except ImportError:
        return None
    x = waveform.detach().cpu().float()
    if x.ndim > 1:
        x = x.mean(dim=0) if x.shape[0] <= 2 else x.reshape(-1)
    y = x.reshape(-1).numpy().astype(np.float32)
    if y.size < sample_rate:
        return None
    try:
        ons = librosa.onset.onset_detect(y=y, sr=sample_rate, units="time",
                                         backtrack=True)
    except Exception:  # noqa: BLE001
        return None
    return np.asarray(ons, dtype=np.float64) if len(ons) else None


def _last_onset_sec(waveform: torch.Tensor, sample_rate: int) -> float | None:
    """Time (s) of the last detected musical onset, or None if undetectable.

    Energy is the wrong criterion for K1 and this is why: on 1f8d6 the last
    onset is at 245.12 s but RMS energy persists to 249.78 s (outro decay,
    reverb, a held pad). The eleven stray notes live in exactly that gap, so an
    energy threshold structurally cannot remove them — measured, it dropped one.

    Uses librosa on the mix. Deliberately NOT the per-stem union that A8 scores
    against: cutting on the evaluation's own detector would be optimising the
    metric directly, the `h_dist` failure. This is a related but independent
    detector, and "no notes after the music stops" is a musical rule the human
    corpus independently confirms (humans place essentially none: p90 = 0.0 s).
    """
    ons = _audio_onset_times(waveform, sample_rate)
    return float(np.max(ons)) if ons is not None and len(ons) else None


def _music_end_sec(waveform: torch.Tensor, sample_rate: int, frac: float,
                   hop: int = 512) -> float | None:
    """Time (s) after which the song is effectively silent, or None if unclear.

    Serves K1. Kyle, on the tempo-fix maps: *"notes playing about 5 seconds
    after the song ends"*. Measured: 8/24 of our maps place notes past the last
    detected onset, the worst (1f8d6) running 11 notes 4.43 s past it, against a
    human corpus that essentially never does (p90 = 0.0 s).

    **Why energy, and not Stage-1's own probabilities.** Thresholding our own
    onset probability would be circular — every selected slot has already
    cleared that threshold, so nothing would be removed. The defect is precisely
    that Stage-1 fires where the evaluation's detector hears nothing. Energy is
    an independent signal, and "the song has ended" is a physical fact about the
    audio rather than a second opinion about onsets.

    The cut is the last frame whose smoothed RMS exceeds `frac` of the song's
    MEDIAN energy — median, not max, so one loud drop cannot drag the threshold
    above a quiet-but-real outro.

    Args:
        waveform: Audio, any shape reducible to mono along the last axis.
        sample_rate: Sample rate of `waveform`.
        frac: Fraction of median energy below which the song counts as over.
        hop: Frame hop in samples for the RMS envelope.

    Returns:
        Cut time in seconds, or None if the heuristic cannot decide.
    """
    if frac <= 0:
        return None
    x = waveform.detach().cpu().float()
    if x.ndim > 1:
        x = x.mean(dim=0) if x.shape[0] <= 2 else x.reshape(-1)
    x = x.reshape(-1).numpy().astype(np.float64)
    n = x.size // hop
    if n < 4:
        return None
    frames = x[: n * hop].reshape(n, hop)
    rms = np.sqrt((frames ** 2).mean(axis=1))

    # ~0.25 s of smoothing: long enough to ride over the gaps between beats,
    # short enough not to smear the end of the song by seconds.
    fps = sample_rate / hop
    k = max(1, int(round(0.25 * fps)))
    if k > 1 and rms.size >= k:
        c = np.cumsum(np.insert(rms, 0, 0.0))
        sm = (c[k:] - c[:-k]) / k
        pad = k // 2
        sm = np.pad(sm, (pad, max(0, rms.size - sm.size - pad)), mode="edge")[: rms.size]
    else:
        sm = rms

    med = float(np.median(sm))
    if not np.isfinite(med) or med <= 0:
        return None
    live = np.flatnonzero(sm > frac * med)
    if live.size == 0:
        return None
    return float(live[-1]) / fps


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


def _lead_multipliers(n_win: int, win_sec: float, bpm: float, asym: float,
                      swap_rate: float, seed: int = 0,
                      window_beats: float = 8.0) -> tuple["np.ndarray", "np.ndarray"]:
    """Per-window budget multipliers that give ONE hand the lead, then swap.

    Returns (left_mult, right_mult). Over a 2-bar block one hand is scaled up by
    (1+asym) and the other down by (1-asym); the block's leader flips with
    probability `swap_rate` (the measured human rate).

    Why multipliers on the window ALLOCATION rather than a post-hoc reassignment:
    `role_asymmetry` counts notes per hand per 8-beat window, so moving a note in
    TIME cannot change it — only changing how many notes each hand gets can. The
    previous `_assign_hand_roles` lever therefore had to DELETE one hand's note at
    shared slots to manufacture asymmetry, which cost ~24% of the notes and hurt
    rhythm. Scaling each hand's per-window share instead keeps every hand's TOTAL
    budget fixed, so no note is lost — the notes simply move to the windows where
    that hand leads. That is exactly the human pattern measured on 2026-07-27:
    balanced GLOBALLY, lopsided LOCALLY.
    """
    import random as _random

    rng = _random.Random(seed)
    block_sec = window_beats * (60.0 / bpm if bpm > 0 else 0.5)
    wins_per_block = max(int(round(block_sec / max(win_sec, 1e-6))), 1)
    lmul = np.ones(n_win); rmul = np.ones(n_win)
    lead = 0
    for b in range((n_win + wins_per_block - 1) // wins_per_block):
        lo, hi = b * wins_per_block, min((b + 1) * wins_per_block, n_win)
        lmul[lo:hi] = 1.0 + asym if lead == 0 else 1.0 - asym
        rmul[lo:hi] = 1.0 - asym if lead == 0 else 1.0 + asym
        if rng.random() < swap_rate:
            lead ^= 1
    return lmul, rmul


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
    section_gate: str = "loud_only",
    use_instr: bool | None = None,
    use_contour: bool | None = None,
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
        use_instr:           Whether to feed per-instrument layering features
            (Demucs→transcription→[n_slots, INSTR_FEATURE_DIM]) into the Stage-1
            BeatClassifier. ``None`` (default) auto-detects from the checkpoint
            (``model.use_instr``); pass ``True``/``False`` to force. Only the
            ``--use-instr`` checkpoints (e.g. version_7) consume this path; it is the
            TASK-2 inference DoD lever (does learned density track human density with
            the section gate OFF). Computed once per song at gen time (~adds Demucs +
            basic-pitch transcription cost).
        section_gate:        How section labels modulate the Stage-1 onset threshold.
            "loud_only" (default): only *lower* the threshold in loud sections
                (drop/chorus); never *raise* it above ``beat_threshold_*`` for quiet
                ones. This kills the V8-0-confirmed silent-drop failure mode (a
                mislabeled "intro"/"outro" can no longer gate a real drop at 0.68/0.72)
                while still letting drops get denser.
            "off": flat ``beat_threshold_*`` everywhere — sections never touch notes
                (they may still drive lighting). Most conservative re: the drop bug.
            "legacy": the old behavior (sections set the threshold outright, incl.
                raising intro/outro to 0.68/0.72). Kept for A/B comparison only.
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
        bpm = _oracle_bpm(audio_path)
    if bpm is None:
        bpm = detect_bpm(waveform, sample_rate=src_sr)
    logger.info("BPM: %.1f", bpm)

    song_duration_secs = waveform.shape[-1] / src_sr
    total_beats = song_duration_secs * bpm / 60.0

    # ---- 2. Source separation ----
    logger.info("Separating audio with Demucs …")
    stems = demucs_separate(waveform, src_sr, device=str(device_obj))

    # ---- 2b. Re-fit the tempo grid to the music (BEAT_TEMPO_FIT=1) ----
    # Every note lands on a 1/4-beat slot grid built from `bpm`, and axis A8
    # measured that grid as exact on 1 of 21 eval songs (median error 0.74%, four
    # at 2/3 tempo). A 0.74% error slides the grid through every phase as the song
    # plays, which is what Kyle hears as "the notes are off beat". Handing the
    # generator the true tempo took onset precision 0.803 -> 0.899 and scatter
    # 11.7 -> 8.5ms (better than the human map's 8.7ms) on 1f767.
    #
    # `data.tempo` recovers the tempo exactly on 21 of 23 eval songs where the
    # current detector manages 1. It runs HERE rather than at step 1 because it
    # scores candidate grids against per-stem onsets, and the stems only exist
    # once Demucs has run — the same stem-union onsets A8 scores against, so the
    # generator is now optimising the quantity the suite measures.
    if os.environ.get("BEAT_TEMPO_FIT") == "1" and bpm is not None:
        try:
            import librosa as _lr

            from beatsaber_automapper.data.tempo import estimate_tempo

            _stem_on: list[float] = []
            for _s in stems.values():
                _arr = _s.detach().cpu().numpy()
                if _arr.ndim > 1:
                    _arr = _arr.mean(axis=tuple(range(_arr.ndim - 1)))
                _stem_on.extend(
                    _lr.onset.onset_detect(y=_arr.astype("float32"), sr=DEMUCS_SR,
                                           units="time", backtrack=True).tolist())
            _on = np.array(sorted(set(np.round(_stem_on, 4))))
            _mono = waveform.squeeze().cpu().numpy()
            if _mono.ndim > 1:
                _mono = _mono.mean(axis=0)
            _fit = estimate_tempo(_mono.astype("float32"), src_sr, onsets=_on)
            if _fit.trusted:
                logger.info("BEAT_TEMPO_FIT: %.2f -> %.3f bpm (R=%.3f, phase %.1f ms)",
                            bpm, _fit.bpm, _fit.r, _fit.phase_s * 1000.0)
                bpm = _fit.bpm
                total_beats = song_duration_secs * bpm / 60.0
            else:
                # A weak fit means the grid is unrelated to the music. Keeping the
                # detector's answer is not obviously better, but silently swapping
                # in an untrusted one is how the original defect hid for months.
                logger.warning("BEAT_TEMPO_FIT: fit UNTRUSTED (R=%.3f) — keeping "
                               "detected bpm %.2f", _fit.r, bpm)
        except Exception as exc:  # noqa: BLE001
            logger.warning("BEAT_TEMPO_FIT failed (%s) — keeping detected bpm", exc)

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
    # ``strict=False`` so pre-struct-feature checkpoints (no struct_proj) still load.
    # Missing weights are reinitialised — fine because we only feed
    # struct_features when explicitly available, and the model treats the path
    # as a no-op when it isn't.
    beat_module = BeatLitModule.load_from_checkpoint(str(beat_checkpoint), strict=False)
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

    # ---- Optional: per-instrument layering features (TASK-2) + pitch contour (TASK-3) ----
    # Both derive from the same Demucs→transcription pass. instr_features feeds
    # the Stage-1 BeatClassifier (--use-instr / version_7+); the contour columns
    # (7:10 — lead_pitch/dpitch/bass_pitch) feed the Stage-2 LayoutModel encoder
    # when the layout ckpt was trained with --use-contour. Both auto-detect from
    # their checkpoints unless explicitly overridden. Compute the features once.
    from beatsaber_automapper.data.layout_dataset import CONTOUR_COLS
    model_has_instr   = bool(getattr(beat_module.model, "use_instr", False))
    model_has_contour = bool(getattr(layout_module.model, "use_contour", False))
    want_instr   = model_has_instr   if use_instr   is None else use_instr
    want_contour = model_has_contour if use_contour is None else use_contour
    instr_gpu:    torch.Tensor | None = None
    contour_full: torch.Tensor | None = None   # [N, 3] for Stage-2, lazily sliced per phrase
    if (want_instr and model_has_instr) or (want_contour and model_has_contour):
        from beatsaber_automapper.data.instrument_features import (
            compute_instrument_features,
        )
        logger.info("Computing per-instrument layering features (Demucs→transcription) …")
        instr_feats = compute_instrument_features(
            waveform, src_sr, bpm, n_slots, subdiv=BEAT_SUBDIV, device=str(device_obj),
        )   # [N, INSTR_FEATURE_DIM]
        instr_all = instr_feats.to(device_obj)
        logger.info("instr_features %s  nonzero_slots=%.2f",
                    tuple(instr_all.shape),
                    float((instr_all.abs().sum(-1) > 0).float().mean()))
        if want_instr and model_has_instr:
            instr_gpu = instr_all
        if want_contour and model_has_contour:
            contour_full = instr_all[:, CONTOUR_COLS].float()   # [N, 3]
    if want_instr and not model_has_instr:
        logger.warning("use_instr=True but beat checkpoint has no instr_proj path — ignoring.")
    elif model_has_instr and not want_instr:
        logger.warning("Beat checkpoint trained with --use-instr but use_instr=False — "
                       "feeding zeros (instr_proj path is a no-op).")
    if want_contour and not model_has_contour:
        logger.warning("use_contour=True but layout checkpoint has no contour_proj path — ignoring.")
    elif model_has_contour and not want_contour:
        logger.warning("Layout checkpoint trained with --use-contour but use_contour=False — "
                       "feeding zeros (contour_proj path is a no-op).")

    beat_probs_parts: list[torch.Tensor] = []
    with torch.no_grad():
        for s in range(0, n_slots, _BEAT_WIN):
            e   = min(s + _BEAT_WIN, n_slots)
            d_w = drum_gpu[s:e].unsqueeze(0)   # [1, W, 768]
            m_w = mix_gpu[s:e].unsqueeze(0)    # [1, W, 768]
            instr_w = instr_gpu[s:e].unsqueeze(0) if instr_gpu is not None else None
            logits_w = beat_module(d_w, m_w, diff_t, slot_offset=s,
                                   instr_features=instr_w)  # [1, W, 2]
            beat_probs_parts.append(torch.sigmoid(logits_w.squeeze(0)))
    beat_probs = torch.cat(beat_probs_parts, dim=0)   # [N, 2]

    # Oracle-ceiling PoC (2026-06-29): gated dump of raw Stage-1 probs BEFORE any
    # thresholding/NMS/density-curve, to measure how much density structure is
    # latent in the model's own probabilities. Env-gated; default behavior unchanged.
    _bp_dump = os.environ.get("BEAT_PROBS_DUMP")
    if _bp_dump:
        np.savez(
            _bp_dump,
            beat_probs=beat_probs.detach().cpu().numpy(),  # [N, 2] (left, right)
            bpm=float(bpm),
            beat_subdiv=int(BEAT_SUBDIV),
            n_slots=int(n_slots),
        )
        logger.info("BEAT_PROBS_DUMP wrote %s  (beat_probs %s, bpm=%.2f, subdiv=%d)",
                    _bp_dump, tuple(beat_probs.shape), float(bpm), int(BEAT_SUBDIV))

    # ----- Section-aware threshold (replaces flat energy modulation) -----
    # The model outputs a near-uniform probability distribution with no clear
    # bimodal gap, so a fixed threshold produces a metronome. Instead we:
    #   1. Detect song sections (intro/verse/chorus/drop/bridge/outro) once.
    #   2. Map each section type to a threshold that reflects natural map density:
    #      drops get low thresholds (dense), intros/outros get high (sparse).
    #   3. Apply per-slot thresholds derived from section boundaries.
    # This matches how human mappers behave: they drop density in breakdowns and
    # go dense in drops, rather than sustaining a constant 6 NPS throughout.
    # Per-section thresholds live in module-level `_SECTION_THRESHOLDS`; how they
    # apply is governed by `section_gate` (see `_build_section_threshold_vector`).

    # Energy-percentile section detector replaces the chroma/MFCC clustering
    # one. EDM and stable-timbre rock both collapsed into a single "outro"
    # cluster post-intro, which mapped to threshold 0.72 and produced a pause
    # at the drop in ArcViewer review. Raw RMS percentiles handle these tracks
    # cleanly. The clustering detector is kept as a fallback only.
    try:
        from beatsaber_automapper.data.audio import (
            detect_sections_energy_percentile as _detect_sections_energy,
        )
        sections = _detect_sections_energy(waveform, sample_rate=src_sr)
    except Exception:
        try:
            from beatsaber_automapper.data.audio import detect_sections as _detect_sections
            sections = _detect_sections(waveform, sample_rate=src_sr)
        except Exception:
            sections = [("verse", 0.0, song_duration_secs)]

    logger.info("Sections (energy-percentile): %s",
                [(t, f"{s:.1f}s", f"{e:.1f}s") for t, s, e in sections])

    # Build per-slot threshold vector from section labels.
    #
    # V8-0 PoC finding (docs/v8_0_poc_findings.md): the silent-drop bug is NOT a
    # representation problem — 94-99% of human notes sit on the existing BPM grid.
    # It is THIS gate: a mislabeled "intro"/"outro" raises the threshold to
    # 0.68/0.72 and silences a real drop. So by default we no longer let a section
    # RAISE the threshold above the base; we only LOWER it for loud sections.
    beats_per_sec = bpm / 60.0
    thr_L, thr_R = _build_section_threshold_vector(
        sections, n_slots, beat_threshold_left, beat_threshold_right,
        beats_per_sec, BEAT_SUBDIV, section_gate,
    )
    thr_L = thr_L.to(device_obj)
    thr_R = thr_R.to(device_obj)
    logger.info("Stage-1 section_gate=%s — thr range L[%.2f,%.2f] R[%.2f,%.2f]",
                section_gate, float(thr_L.min()), float(thr_L.max()),
                float(thr_R.min()), float(thr_R.max()))

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

    def _load_ioi_model():
        """Human P(next interval | current interval), mined from 300 human maps.

        Intervals are in 1/4-beat units (BEAT_SUBDIV), capped at 8. The model is
        strongly diagonal — P(1/8→1/8) 0.714, P(1/16→1/16) 0.618, P(1/4→1/4)
        0.561 — with real switching mass between 1/16, 1/8 and 1/4. Human rhythm
        holds a subdivision for a run and then changes gear; ours sits on one
        subdivision for a whole song (75% of intervals on 1/8 vs a human 49.5%),
        which is why A2 is one of our worst axes.
        """
        import json as _json
        import math as _math
        p = Path(__file__).resolve().parents[3] / "outputs" / "ioi_human_model.json"
        if not p.exists():
            return None
        try:
            raw = _json.loads(p.read_text())
        except Exception:  # noqa: BLE001
            return None
        floor = 1e-4
        logp: dict[int, dict[int, float]] = {}
        for a_s, row in raw["bigram"].items():
            rt = sum(row.values()) or 1
            logp[int(a_s)] = {int(b): _math.log(max(c / rt, floor))
                              for b, c in row.items()}
        uni = {int(k): v for k, v in raw["unigram"].items()}
        tot = sum(uni.values()) or 1
        back = {k: _math.log(max(v / tot, floor)) for k, v in uni.items()}
        return logp, back

    def _ioi_dp_select(p_arr, idxs: list[int], k: int, prev_cls: int, model,
                       lam: float, rng=None) -> tuple[list[int], int]:
        """Pick k slots from `idxs` by SAMPLING model prob + the human interval prior.

        Selecting purely by probability makes a map metronomic: the top-k slots of
        a periodic audio signal are themselves periodic, so the interval
        distribution collapses onto one value.

        The first version of this MAXIMISED prob + prior, and that was wrong in an
        instructive way. The human bigram is strongly diagonal (P(1/8→1/8) 0.714),
        so its argmax is "keep the current interval" — maximum-likelihood selection
        takes the diagonal nearly always and produces long homogeneous runs. On the
        24-song sweep it made rhythm WORSE than the baseline it was meant to fix
        (switch rate 5.38 → 3.18 against a human 13.65), even though the interval
        HISTOGRAM moved toward human. **The argmax of a distribution is not a
        sample from it**, and the human data switches 29% of the time.

        So this samples forward instead: at each step draw the next slot from
        softmax(log p + lam * log P(interval | previous)). That reproduces the
        prior's switching mass rather than collapsing onto its mode.
        """
        import math as _math
        import random as _random
        logp, back = model
        rng = rng or _random.Random(0)
        n = len(idxs)
        if k <= 0 or n == 0:
            return [], prev_cls
        k = min(k, n)

        def bi(prev: int, cur: int) -> float:
            row = logp.get(prev)
            return (row or back).get(cur, -9.0)

        lp = [_math.log(max(float(p_arr[s]), 1e-9)) for s in idxs]
        temp = float(os.environ.get("BEAT_IOI_TEMP", "0.35"))
        chosen: list[int] = []
        cur = prev_cls
        last = -1
        for step in range(k):
            # BUDGET GUARD: leave room for the notes still owed. Without it the
            # sampler can jump to a distant slot, run out of candidates and
            # under-fill the window -- free sampling lost 66% of the notes.
            need = k - step - 1
            hi = n - need
            cands = [i for i in range(last + 1, hi)]
            if not cands:
                break
            scores = []
            for i in cands:
                s = lp[i]
                if last >= 0:
                    s += lam * bi(cur, min(max(idxs[i] - idxs[last], 1), 8))
                scores.append(s)
            m = max(scores)
            # TEMPERATURE interpolates between the two failure modes: temp -> 0 is
            # the maximiser (too regular; long homogeneous runs), temp = 1 is a
            # free sample from the prior (too random). Human rhythm is neither.
            w = [_math.exp((s - m) / max(temp, 1e-3)) for s in scores]
            tot = sum(w) or 1.0
            r = rng.random() * tot
            acc = 0.0
            pick = cands[-1]
            for i, wi in zip(cands, w):
                acc += wi
                if acc >= r:
                    pick = i
                    break
            if last >= 0:
                cur = min(max(idxs[pick] - idxs[last], 1), 8)
            chosen.append(idxs[pick])
            last = pick
        return chosen, cur

    def _offset_hands(left: set[int], right: set[int], probs, rate: float,
                      seed: int = 0, spacing_aware: bool = False,
                      min_gap: int = 2) -> tuple[set[int], set[int]]:
        """Where both hands share a slot, MOVE one by a 16th instead of deleting it.

        Found 2026-07-27 by dumping beat_probs next to the human note times on the
        same grid. Our maps never place a note on an odd 16th — not once in 679
        slots; every note lands on a beat or an 8th. The human map puts 248 notes
        on odd 16ths, and those are exactly the slots we miss.

        The cause is hand LOCKSTEP. Nearest-right-hand-note offsets in 16ths:

            offset      -1      0     +1
            human     0.220  0.398  0.099     <- interleaved 32% of the time
            ours      0.002  0.945  0.000     <- lockstep

        The union of two hands can only reach an odd 16th if the hands are
        offset, and we never offset them. That makes the A2 rhythm gap and the A6
        hand-role gap **the same defect**: with both hands on the same slots the
        union rhythm is confined to the 8th-note grid, so intervals are forced to
        multiples of two slots and interval variety is impossible.

        This is also why BEAT_HAND_ROLE hurt rhythm — it DELETED one hand's note
        at a shared slot, leaving the odd slot empty and dropping ~24% of the
        notes. Moving preserves the note count and fills the odd slot.
        """
        import random as _random

        rng = _random.Random(seed)
        shared = sorted(left & right)
        if not shared or rate <= 0.0:
            return left, right
        new_right = set(right)
        n = len(probs)
        for s in shared:
            if rng.random() > rate:
                continue
            # prefer whichever neighbouring slot the model likes better for this
            # hand, and never collide with a note either hand already holds
            # MIN-GAP guard. Moving a note by a 16th can drop it right beside this
            # hand's neighbouring note, which spikes burst speed: the 24-song sweep
            # showed ebpm_burst going 243 -> 360 swings/min against a human 250,
            # and THAT -- not angle_change, which actually improved slightly -- is
            # what the flow regression was. Only offset when the moved note stays
            # at least `min_gap` slots away from this hand's other notes.
            cands = [d for d in (-1, 1)
                     if 0 <= s + d < n and (s + d) not in new_right and (s + d) not in left
                     and min((abs(s + d - o) for o in new_right if o != s), default=99) >= min_gap]
            if not cands:
                continue
            if spacing_aware:
                # Prefer the neighbour that leaves this hand's spacing more even.
                # Moving a note changes which hand plays when, which shifts the
                # wrist-rotation sequence: the 24-song sweep showed flow
                # regressing via angle_change (19.8 -> 23.1) while travel was
                # unchanged (5.73 -> 5.67). Choosing the side that keeps the
                # hand's own gaps regular is the cheapest available proxy for
                # keeping that sequence smooth.
                def _cost(dd: int) -> float:
                    t = s + dd
                    near = [abs(t - o) for o in new_right if abs(t - o) <= 8]
                    return -min(near) if near else -8.0
                d = min(cands, key=lambda dd: (_cost(dd), -float(probs[s + dd])))
            else:
                d = max(cands, key=lambda dd: float(probs[s + dd]))
            new_right.discard(s)
            new_right.add(s + d)
        return left, new_right

    def _assign_hand_roles(left: set[int], right: set[int], bpm: float,
                           strength: float = 1.0,
                           target_asym: float = 0.115,
                           swap_rate: float = 0.461,
                           double_rate: float = 0.175,
                           window_beats: float = 8.0,
                           seed: int = 0) -> tuple[set[int], set[int]]:
        """Give one hand the lead per 2-bar window, then swap — human role division.

        Onset TIMES are preserved exactly; only the hand each onset belongs to
        changes, so density and rhythm are untouched. Targets the measured human
        reference (`evaluation/handrole.py`): local asymmetry 0.115, dominant-hand
        swap rate 0.461. `strength` interpolates from current behaviour (0) to the
        full human target (1).
        """
        import random as _random

        rng = _random.Random(seed)
        slots_per_window = max(int(window_beats * BEAT_SUBDIV), 1)
        allslots = sorted(left | right)
        if not allslots:
            return left, right
        both = left & right          # slots where BOTH hands currently play

        by_win: dict[int, list[int]] = {}
        for s in allslots:
            by_win.setdefault(s // slots_per_window, []).append(s)

        new_left: set[int] = set()
        new_right: set[int] = set()
        lead = 0
        asym = target_asym * max(0.0, min(strength, 1.0))
        lead_share = (1.0 + asym) / 2.0
        for w in sorted(by_win):
            slots = by_win[w]
            n_lead = 0
            for i, s in enumerate(slots):
                # Keep genuine doubles at roughly the human rate. Taking the union
                # naively would collapse every simultaneous pair onto one hand and
                # silently delete ~45% of the notes (our maps are 85.6% doubles).
                if s in both and rng.random() < double_rate:
                    new_left.add(s)
                    new_right.add(s)
                    continue
                # Give the lead hand a majority SHARE, distributed through the
                # window rather than as one contiguous block: humans alternate
                # mostly (run length 1.36) while one hand takes more of the work.
                # A contiguous block overshoots run length to ~6.7 and reads as
                # one hand idling, which is not what human maps do.
                take_lead = (n_lead / (i + 1)) < lead_share if i else True
                to_left = (lead == 0) == take_lead
                (new_left if to_left else new_right).add(s)
                n_lead += take_lead
            if rng.random() < swap_rate:
                lead ^= 1
        return new_left, new_right

    # K1 decay lever. BEAT_ONSET_EVIDENCE is the exponent on per-window audio
    # onset density; 0 = OFF (prior behaviour). BEAT_ONSET_EVIDENCE_FLOOR keeps a
    # window with no detected onsets from being zeroed outright.
    _evid_beta = float(os.environ.get("BEAT_ONSET_EVIDENCE", "0.0"))
    _evid_floor = float(os.environ.get("BEAT_ONSET_EVIDENCE_FLOOR", "0.15"))
    _evid_onsets = _audio_onset_times(waveform, src_sr) if _evid_beta > 0.0 else None
    if _evid_beta > 0.0:
        logger.info("BEAT_ONSET_EVIDENCE=%.2f (floor %.2f): %s",
                    _evid_beta, _evid_floor,
                    f"{len(_evid_onsets)} audio onsets" if _evid_onsets is not None
                    else "NO audio onsets detected — lever inert")

    def _density_aware_select(
        probs: torch.Tensor, slot_sec: "np.ndarray", win_sec: float,
        gamma: float, budget: int, radius: int,
        win_mult: "np.ndarray | None" = None,
    ) -> set[int]:
        """Density-aware redistribution (2026-06-30 Phase-2 selection PoC).

        Keeps the SAME total count as the threshold method (``budget``) but
        re-allocates it across ``win_sec`` windows proportional to
        (window-mean prob)**gamma — so loud/dense windows keep more notes and
        quiet ones thin out, recovering the ~0.40 density structure the oracle
        ceiling showed is latent in beat_probs but flattened by the per-slot
        threshold + NMS. Within each window, picks the top slots by prob with a
        ``radius``-slot min-distance.
        """
        p = probs.detach().cpu().numpy()
        N = len(p)
        if budget <= 0 or N == 0:
            return set()
        win_idx = (slot_sec / win_sec).astype(int)
        n_win = int(win_idx.max()) + 1
        wsum = np.zeros(n_win); wcnt = np.zeros(n_win)
        np.add.at(wsum, win_idx, p); np.add.at(wcnt, win_idx, 1.0)
        wmean = wsum / np.clip(wcnt, 1.0, None)
        weight = np.power(np.clip(wmean, 1e-6, None), gamma)
        # ONSET EVIDENCE (K1 decay, 2026-08-03). Measured: on 1f8d6's outro,
        # windows with ZERO detected onsets carry wmean 0.28-0.42 -- as high as
        # the body of the song -- so this formula hands ~35 notes to a region
        # containing ~2 real onsets. wmean is the defect, so no ceiling computed
        # FROM wmean can fix it. This multiplies in an INDEPENDENT signal: how
        # many onsets the audio itself has in each window.
        #
        # Two mechanisms it is meant to catch, both measured:
        #   1f8d6 / 1f336 — music thins, Stage-1's probability does not follow
        #   1f333 / 1f3d7 — music does NOT thin, but probability RISES at the end
        #
        # C1 records three decode levers that failed to move precision, but all
        # three were functions of these same probabilities. This one is not,
        # which is why it is worth one more attempt -- it is still a hypothesis.
        if _evid_beta > 0.0 and _evid_onsets is not None:
            ev = np.zeros(n_win)
            ei = (_evid_onsets / win_sec).astype(int)
            ei = ei[(ei >= 0) & (ei < n_win)]
            np.add.at(ev, ei, 1.0)
            # Normalise to mean 1 so this re-shapes the budget without changing
            # its scale, then floor it so a window is never zeroed outright --
            # the detector missing a quiet passage should thin it, not delete it.
            m = ev.mean()
            if m > 0:
                ev = np.clip(ev / m, _evid_floor, None)
                weight = weight * np.power(ev, _evid_beta)
        # HAND LEAD (2026-08-01): scale this hand's share per window so it carries
        # more of the load where it leads. Applied AFTER gamma so the density curve
        # the oracle-ceiling PoC validated is preserved — this only changes how the
        # already-shaped budget is split between the hands, never its total.
        if win_mult is not None and len(win_mult) >= n_win:
            weight = weight * np.clip(win_mult[:n_win], 1e-6, None)
        if weight.sum() <= 0:
            return _nms(probs.to(device_obj), thr_L, radius)  # degenerate fallback
        raw = budget * weight / weight.sum()
        alloc = np.floor(raw).astype(int)
        rem = int(budget - alloc.sum())
        if rem > 0:
            frac = raw - np.floor(raw)
            for w in np.argsort(-frac)[:rem]:
                alloc[w] += 1
        # IOI PRIOR (2026-07-27). Within a window, picking the top-k slots by
        # probability reproduces the audio's own periodicity, so the interval
        # distribution collapses (75% of our intervals on 1/8 vs a human 49.5%).
        # With BEAT_IOI_PRIOR>0 the within-window pick instead maximises
        # prob + lambda * human P(interval | previous interval), carrying the
        # interval state ACROSS windows so phrase boundaries do not reset it.
        # The window ALLOCATION is untouched, so the validated density_corr
        # behaviour is preserved and only the interval structure changes.
        _lam = float(os.environ.get("BEAT_IOI_PRIOR", "0.0"))
        _model = _load_ioi_model() if _lam > 0.0 else None
        _prev_cls = 2                      # assume an 1/8 pulse to start

        selected: set[int] = set()
        for w in range(n_win):
            k = int(alloc[w])
            if k <= 0:
                continue
            idxs = np.where(win_idx == w)[0]
            if len(idxs) == 0:
                continue
            if _model is not None:
                # respect the same min-distance by thinning candidates first
                cand = [int(i) for i in idxs]
                chosen, _prev_cls = _ioi_dp_select(p, cand, k, _prev_cls,
                                                   _model, _lam)
            else:
                order = idxs[np.argsort(-p[idxs])]
                chosen = []
                for i in order:
                    if len(chosen) >= k:
                        break
                    if all(abs(int(i) - c) > radius for c in chosen):
                        chosen.append(int(i))
            selected.update(chosen)
        return selected

    left_thr  = _nms(beat_probs[:, 0].to(device_obj), thr_L, beat_nms_radius)
    right_thr = _nms(beat_probs[:, 1].to(device_obj), thr_R, beat_nms_radius)
    if os.environ.get("DENSITY_SELECT") == "1":
        _gamma = float(os.environ.get("DENSITY_SELECT_GAMMA", "1.5"))
        _win   = float(os.environ.get("DENSITY_SELECT_WIN", "2.0"))
        _slot_sec = (np.arange(n_slots) / BEAT_SUBDIV) * (60.0 / bpm)
        # If hand-role reassignment is on it will de-double most slots (our maps
        # currently play both hands on ~86% of beats vs a human 17.5%), which by
        # itself would delete ~38% of the notes and push density BELOW human.
        # Compensate by selecting proportionally more DISTINCT slots up front, so
        # the note budget survives and the extra notes buy rhythmic positions
        # rather than doubles.
        _hr_pre = float(os.environ.get("BEAT_HAND_ROLE", "0.0"))
        _bL, _bR = len(left_thr), len(right_thr)
        # DIFFICULTY SCALE (eval-suite v2 axis A7, 2026-07-28). Kyle played the
        # maps and called them "Expert+, not Expert": generated NPS is 6.18
        # against a human Expert median of 3.91-4.46. The window ALLOCATION
        # logic above already does the hard part (making density track song
        # structure); this only scales the TOTAL note budget those windows
        # compete for, so the shape of the density curve is preserved and only
        # its overall level drops. Default 1.0 = OFF (current behaviour).
        _diff_scale = float(os.environ.get("BEAT_DIFFICULTY_SCALE", "1.0"))
        if _diff_scale != 1.0:
            _bL = max(0, int(round(_bL * _diff_scale)))
            _bR = max(0, int(round(_bR * _diff_scale)))
        if _hr_pre > 0.0:
            _union = len(left_thr | right_thr) or 1
            _D = len(left_thr & right_thr) / _union
            _infl = (1.0 + _D) / (1.0 + 0.175)
            _bL, _bR = int(round(_bL * _infl)), int(round(_bR * _infl))
        # HAND LEAD (eval-suite v2 axis A6, 2026-08-01). `role_asymmetry` is human
        # 0.115 vs ours 0.026-0.046 and its cohort spread is 0.27 against the 0.35
        # bar — it is the SINGLE sub-metric that fails the handrole axis (see
        # scripts/eval_spread_breakdown.py). Unlike BEAT_HAND_ROLE this does not
        # reassign or delete any note: it biases each hand's per-window budget
        # share, so the hands stay globally balanced and the note count is exactly
        # preserved. Value = target local asymmetry; 0.0 = OFF (prior behaviour).
        _hl = float(os.environ.get("BEAT_HAND_LEAD", "0.0"))
        _lmul = _rmul = None
        if _hl > 0.0:
            _nw = int((_slot_sec / _win).astype(int).max()) + 1
            _lmul, _rmul = _lead_multipliers(
                _nw, _win, bpm, min(_hl, 0.95),
                float(os.environ.get("BEAT_HAND_LEAD_SWAP", "0.461")),
                # Re-seeding changes WHICH hand leads each block without changing
                # the target asymmetry — the check that a passing arm is not an
                # artefact of one particular arrangement of leads.
                seed=int(os.environ.get("BEAT_HAND_LEAD_SEED", "0")))
        left_onsets  = _density_aware_select(
            beat_probs[:, 0], _slot_sec, _win, _gamma, _bL, beat_nms_radius,
            win_mult=_lmul)
        # HAND INTERLEAVE (eval-suite v2 axis A2, 2026-07-27). The two hands are
        # selected from two probability channels driven by the SAME audio, so they
        # pick the same slots: our maps fire both hands simultaneously on 85.6% of
        # beats, against a human rate of 17.5%. That lockstep is what makes the
        # union rhythm metronomic — 75% of our intervals land on exactly 1/8,
        # vs 41% for humans — even though our PER-HAND intervals are already
        # human-like. Penalising the right hand on slots the left hand took lets
        # the hands interleave and the union rhythm breathe.
        # Strength 0.0 = OFF (prior behaviour). Human maps still play ~17.5% real
        # doubles, so this must be a soft penalty, never a hard exclusion.
        _il = float(os.environ.get("BEAT_HAND_INTERLEAVE", "0.0"))
        if _il > 0.0 and left_onsets:
            rp = beat_probs[:, 1].clone()
            idx = torch.tensor(sorted(left_onsets), dtype=torch.long, device=rp.device)
            rp[idx] = rp[idx] * (1.0 - min(_il, 1.0))
            right_onsets = _density_aware_select(
                rp, _slot_sec, _win, _gamma, _bR, beat_nms_radius, win_mult=_rmul)
        else:
            right_onsets = _density_aware_select(
                beat_probs[:, 1], _slot_sec, _win, _gamma, _bR, beat_nms_radius,
                win_mult=_rmul)
        # HAND ROLE (eval-suite v2 axis A6, 2026-07-27). Discovered by reading a
        # map next to its human counterpart: within a passage a human mapper gives
        # ONE hand the lead — a sustained run — while the other punctuates, then
        # they swap. Our maps split every bar evenly, so they are balanced at every
        # scale; human maps are balanced GLOBALLY but lopsided LOCALLY. Measured:
        # role_asymmetry 0.115 human vs 0.031 ours, swap rate 0.461 vs 0.269, and
        # A6 is our worst axis (3.50 vs a human 0.34).
        #
        # This REASSIGNS which hand plays each already-selected onset, per 2-bar
        # window, leaving the onset TIMES untouched — so rhythm and density are
        # unaffected and only the role structure changes. That is the difference
        # from the failed BEAT_HAND_INTERLEAVE lever, which pushed the hands apart
        # without giving either one a job and made rhythm worse.
        # HAND OFFSET (2026-07-27): shift one hand by a 16th at shared slots so
        # the union rhythm can reach odd-16th positions at all. Human hands are
        # interleaved 32% of the time; ours 0.2%. Default 0.0 = OFF.
        _ho = float(os.environ.get("BEAT_HAND_OFFSET", "0.0"))
        if _ho > 0.0 and left_onsets and right_onsets:
            left_onsets, right_onsets = _offset_hands(
                left_onsets, right_onsets, beat_probs[:, 1].detach().cpu().numpy(), _ho,
                spacing_aware=os.environ.get("BEAT_HAND_OFFSET_SPACING") == "1",
                min_gap=int(os.environ.get("BEAT_HAND_OFFSET_MINGAP", "2")))
        _hr = float(os.environ.get("BEAT_HAND_ROLE", "0.0"))
        if _hr > 0.0 and (left_onsets or right_onsets):
            left_onsets, right_onsets = _assign_hand_roles(
                left_onsets, right_onsets, bpm, strength=_hr)
        logger.info("DENSITY_SELECT on (gamma=%.2f win=%.1fs interleave=%.2f "
                    "role=%.2f diff_scale=%.2f): redistributed L %d->%d, R %d->%d",
                    _gamma, _win, _il, _hr, _diff_scale,
                    len(left_thr), len(left_onsets), len(right_thr), len(right_onsets))
    else:
        left_onsets, right_onsets = left_thr, right_thr

    # K1: drop slots landing after the music has stopped. Default OFF, like
    # every other lever here. BEAT_TRIM_TAIL is the GRACE in seconds allowed
    # after the last detected onset (0.5 is a sane starting point); the cut is
    # last_onset + grace, falling back to a silence cut if onset detection fails.
    # NB this is deliberately in the v7 path and not in predict_onsets(), which
    # only the legacy generate_level() calls -- a lever placed there would be a
    # silent no-op in production, which is exactly how BEAT_GRID_SUBDIV died.
    _tt = os.environ.get("BEAT_TRIM_TAIL", "")
    if _tt:
        try:
            _grace = float(_tt)
        except ValueError:
            _grace = -1.0
        _end = None
        if _grace >= 0:
            _lo = _last_onset_sec(waveform, src_sr)
            if _lo is not None:
                _end = _lo + _grace
            else:
                # Fall back to the silence heuristic rather than doing nothing.
                _end = _music_end_sec(waveform, src_sr, 0.15)
        if _end is not None:
            _slot_t = (np.arange(n_slots) / BEAT_SUBDIV) * (60.0 / bpm if bpm > 0 else 0.5)
            _keep = {int(i) for i in range(n_slots) if _slot_t[i] <= _end}
            _nl, _nr = len(left_onsets), len(right_onsets)
            _l2, _r2 = left_onsets & _keep, right_onsets & _keep
            # Never hand back an empty map because the heuristic misfired.
            if _l2 or _r2:
                left_onsets, right_onsets = _l2, _r2
                logger.info(
                    "BEAT_TRIM_TAIL grace=%.2fs: cut at %.2fs; dropped L %d->%d, "
                    "R %d->%d", _grace, _end, _nl, len(left_onsets), _nr,
                    len(right_onsets))
            else:
                logger.warning(
                    "BEAT_TRIM_TAIL would empty the map (cut %.2fs) — ignored",
                    _end)

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

        # TASK 3: per-slot pitch contour for this phrase, padded like phrase_mert.
        phrase_contour_t = None
        if contour_full is not None:
            pc = contour_full[slot_start:slot_start + p_len].unsqueeze(0)   # [1, p, 3]
            if p_len < max_phrase_slots_inf:
                pad_c = torch.zeros(1, max_phrase_slots_inf - p_len, pc.shape[-1],
                                    device=device_obj)
                pc = torch.cat([pc, pad_c], dim=1)
            phrase_contour_t = pc

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
                phrase_contour = phrase_contour_t,
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
    if os.environ.get("LAYOUT_DIAG") == "1":
        from beatsaber_automapper.data.swing_tokenizer import NOTE as _NOTE
        _ny = [0, 0, 0]
        for _e in all_events:
            if _e.kind == _NOTE and 0 <= _e.y <= 2:
                _ny[_e.y] += 1
        _t = max(sum(_ny), 1)
        logger.info("all_events NOTE-row dist = %s (n=%d)",
                    [round(c / _t, 2) for c in _ny], sum(_ny))

    # ---- 9. Assemble beatmap ----
    beatmap = _events_to_beatmap(all_events)
    logger.info("Decoded: %d notes, %d arcs, %d chains, %d bombs",
                len(beatmap.color_notes), len(beatmap.sliders),
                len(beatmap.burst_sliders), len(beatmap.bomb_notes))

    # Confound probe (TASK-3): dump the PRE-postprocess beatmap so the
    # contour-follow eval can see the model's raw swing directions before the
    # parity-fix rewrites ~48% of them. Gated behind an env var so production
    # behavior is unchanged when unset.
    import os as _os, copy as _copy
    _prepost_out = _os.environ.get("BS_PREPOST_OUT")
    _prepost_bm = _copy.deepcopy(beatmap) if _prepost_out else None

    beatmap = postprocess_beatmap(beatmap, difficulty=difficulty, bpm=bpm,
                                  song_duration_secs=song_duration_secs)

    if _prepost_out:
        package_level(
            beatmaps={difficulty: _prepost_bm},
            audio_path=audio_path,
            output_path=Path(_prepost_out),
            song_name=song_name,
            song_author=song_author,
            bpm=bpm,
            chroma_events={difficulty: None},
        )
        logger.info("Wrote PRE-postprocess beatmap to %s", _prepost_out)

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
