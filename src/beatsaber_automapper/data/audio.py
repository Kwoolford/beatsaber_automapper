"""Audio loading and mel spectrogram extraction.

Handles conversion to mono 44.1kHz and extraction of mel spectrograms
with parameters tuned for rhythmic feature detection in music.

Config: 80 mel bands, 1024 FFT, 512 hop (~11.6ms/frame at 44100Hz).
"""

from __future__ import annotations

import logging
from pathlib import Path

import soundfile as sf
import torch
import torchaudio

logger = logging.getLogger(__name__)


def load_audio(path: Path | str, target_sr: int = 44100) -> tuple[torch.Tensor, int]:
    """Load an audio file and convert to mono at the target sample rate.

    Uses soundfile for reading. Falls back to ffmpeg for formats soundfile
    can't handle natively (e.g. .mp3 on Windows without libsndfile extras).

    Args:
        path: Path to audio file (.mp3, .ogg, .wav, .egg, .flac).
        target_sr: Target sample rate in Hz.

    Returns:
        Tuple of (waveform tensor [1, samples], sample_rate).

    Raises:
        RuntimeError: If the audio file cannot be loaded by any backend.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")

    # Try soundfile first (fast, no ffmpeg dependency)
    try:
        data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    except Exception:
        # Fallback: use ffmpeg to convert to WAV in memory
        data, sr = _load_audio_ffmpeg(path)

    # data shape: [samples, channels]
    if data.shape[1] > 1:
        data = data.mean(axis=1, keepdims=True)

    waveform = torch.from_numpy(data.T)

    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
        waveform = resampler(waveform)

    return waveform, target_sr


def _load_audio_ffmpeg(path: Path) -> tuple:
    """Load audio via ffmpeg subprocess (fallback for mp3 etc).

    Args:
        path: Path to audio file.

    Returns:
        Tuple of (numpy array [samples, channels], sample_rate).

    Raises:
        RuntimeError: If ffmpeg is not installed or conversion fails.
    """
    import subprocess
    import tempfile

    try:
        # Convert to WAV via ffmpeg
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name

        cmd = [
            "ffmpeg", "-y", "-i", str(path),
            "-ar", "44100", "-ac", "1", "-f", "wav", tmp_path,
        ]
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=120,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"ffmpeg failed (code {result.returncode}): {result.stderr[:500]}"
            )

        data, sr = sf.read(tmp_path, dtype="float32", always_2d=True)
        return data, sr
    except FileNotFoundError:
        raise RuntimeError(
            f"Cannot load {path.suffix} files: soundfile failed and ffmpeg "
            "is not installed. Install ffmpeg or convert to .wav/.ogg first."
        ) from None
    finally:
        tmp_path_obj = Path(tmp_path) if "tmp_path" in dir() else None
        if tmp_path_obj and tmp_path_obj.exists():
            tmp_path_obj.unlink(missing_ok=True)


def extract_mel_spectrogram(
    waveform: torch.Tensor,
    sample_rate: int = 44100,
    n_mels: int = 80,
    n_fft: int = 1024,
    hop_length: int = 512,
) -> torch.Tensor:
    """Extract log-mel spectrogram from a waveform.

    Args:
        waveform: Audio tensor [1, samples] or [samples].
        sample_rate: Sample rate of the waveform.
        n_mels: Number of mel frequency bands.
        n_fft: FFT window size.
        hop_length: Hop length between frames.

    Returns:
        Log-mel spectrogram tensor [n_mels, n_frames].
    """
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)

    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
    )
    mel = mel_transform(waveform)  # [1, n_mels, n_frames]

    # Log scale with small epsilon to avoid log(0)
    mel = torch.log(mel.clamp(min=1e-9))

    return mel.squeeze(0)  # [n_mels, n_frames]


def detect_bpm(waveform: torch.Tensor, sample_rate: int = 44100) -> float:
    """Detect the tempo (BPM) of an audio waveform using librosa's beat tracker.

    Converts the waveform to a numpy array, runs librosa's beat_track(), and
    returns the estimated tempo. Falls back to 120.0 BPM on any error.

    Args:
        waveform: Audio tensor [1, samples] or [samples].
        sample_rate: Sample rate of the waveform.

    Returns:
        Estimated BPM as a float (e.g. 128.0).
    """
    import numpy as np

    try:
        import librosa
    except ImportError:
        logger.warning("librosa not installed — cannot detect BPM, defaulting to 120.0")
        return 120.0

    # Convert to mono numpy array
    audio_np = waveform.squeeze().numpy().astype(np.float32)

    try:
        tempo, _ = librosa.beat.beat_track(y=audio_np, sr=sample_rate)
        # tempo may be a numpy scalar or 0-d array
        bpm = float(np.atleast_1d(tempo)[0])
        if bpm <= 0:
            logger.warning("librosa returned invalid BPM %.1f — defaulting to 120.0", bpm)
            return 120.0
        logger.info("Auto-detected BPM: %.1f", bpm)
        return bpm
    except Exception as e:
        logger.warning("BPM detection failed (%s) — defaulting to 120.0", e)
        return 120.0


def compute_structure_features(
    waveform: torch.Tensor,
    sample_rate: int = 44100,
    hop_length: int = 512,
    n_mels: int = 80,
) -> torch.Tensor:
    """Compute per-frame song structure features aligned to mel spectrogram.

    Returns 6 normalized features that capture musical energy and section character.
    These help all three stages understand song structure (buildups, drops, breakdowns).

    Features:
        [0] rms_energy       — overall loudness (0-1)
        [1] onset_strength   — spectral flux / transient energy (0-1)
        [2] bass_energy      — mean energy in low mel bands (0-1)
        [3] mid_energy       — mean energy in mid mel bands (0-1)
        [4] high_energy      — mean energy in high mel bands (0-1)
        [5] spectral_centroid — brightness / frequency center (0-1)

    Args:
        waveform: Audio tensor [1, samples] or [samples].
        sample_rate: Sample rate of the waveform.
        hop_length: Hop length matching the mel spectrogram extraction.
        n_mels: Number of mel bands (for sub-band split points).

    Returns:
        Tensor of shape [6, n_frames] with the same time axis as the mel spectrogram.
    """
    import numpy as np

    try:
        import librosa
    except ImportError:
        logger.warning("librosa not installed — returning zero structure features")
        # Estimate n_frames from waveform length
        n_frames = waveform.shape[-1] // hop_length + 1
        return torch.zeros(6, n_frames)

    y = waveform.squeeze().numpy().astype(np.float32)

    # 1. RMS energy
    rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
    rms_norm = rms / (rms.max() + 1e-8)

    # 2. Onset strength (spectral flux)
    onset_env = librosa.onset.onset_strength(y=y, sr=sample_rate, hop_length=hop_length)
    onset_norm = onset_env / (onset_env.max() + 1e-8)

    # 3-5. Sub-band energy from mel spectrogram
    S = librosa.feature.melspectrogram(
        y=y, sr=sample_rate, n_mels=n_mels, hop_length=hop_length
    )
    S_db = librosa.power_to_db(S, ref=np.max)  # dB scale, max=0
    S_01 = np.clip((S_db + 80) / 80, 0, 1)  # rough normalization to [0, 1]

    bass_cutoff = n_mels // 4      # ~0-1kHz
    mid_cutoff = n_mels * 5 // 8   # ~1-4kHz
    bass = S_01[:bass_cutoff].mean(axis=0)
    mid = S_01[bass_cutoff:mid_cutoff].mean(axis=0)
    high = S_01[mid_cutoff:].mean(axis=0)

    # 6. Spectral centroid (normalized by Nyquist)
    centroid = librosa.feature.spectral_centroid(
        y=y, sr=sample_rate, hop_length=hop_length
    )[0]
    centroid_norm = centroid / (sample_rate / 2)

    # Align lengths (onset_strength can differ by 1 frame)
    n_frames = min(len(rms_norm), len(onset_norm), len(bass), len(centroid_norm))

    features = np.stack(
        [
            rms_norm[:n_frames],
            onset_norm[:n_frames],
            bass[:n_frames],
            mid[:n_frames],
            high[:n_frames],
            centroid_norm[:n_frames],
        ],
        axis=0,
    )

    return torch.from_numpy(features.astype(np.float32))


def convert_to_ogg(input_path: Path | str, output_path: Path | str) -> Path:
    """Convert an audio file to OGG Vorbis format using ffmpeg.

    Beat Saber expects .ogg or .egg audio. If the input is already .ogg,
    it is copied as-is. Otherwise ffmpeg converts it.

    Args:
        input_path: Path to source audio file.
        output_path: Path for the output .ogg file.

    Returns:
        Path to the output .ogg file.

    Raises:
        RuntimeError: If ffmpeg is not available and conversion is needed.
    """
    import shutil
    import subprocess

    input_path = Path(input_path)
    output_path = Path(output_path)

    if input_path.suffix.lower() in (".ogg", ".egg"):
        shutil.copy2(input_path, output_path)
        return output_path

    try:
        cmd = [
            "ffmpeg", "-y", "-i", str(input_path),
            "-c:a", "libvorbis", "-q:a", "6", str(output_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode != 0:
            logger.warning("ffmpeg ogg conversion failed, using original file")
            shutil.copy2(input_path, output_path)
    except FileNotFoundError:
        logger.warning("ffmpeg not found — using original audio format in zip")
        shutil.copy2(input_path, output_path)

    return output_path


def beat_to_frame(
    beat: float,
    bpm: float,
    sample_rate: int = 44100,
    hop_length: int = 512,
    offset: float = 0.0,
) -> int:
    """Convert a beat number to a spectrogram frame index (constant BPM).

    Args:
        beat: Beat number (e.g. 10.5).
        bpm: Beats per minute of the song.
        sample_rate: Audio sample rate.
        hop_length: Hop length used for spectrogram.
        offset: Song time offset in seconds.

    Returns:
        Frame index (integer).
    """
    time_seconds = (beat * 60.0 / bpm) + offset
    sample_index = time_seconds * sample_rate
    return int(round(sample_index / hop_length))


def beat_to_frame_variable_bpm(
    beat: float,
    base_bpm: float,
    bpm_changes: list[dict],
    sample_rate: int = 44100,
    hop_length: int = 512,
    offset: float = 0.0,
) -> int:
    """Convert a beat number to a frame index, accounting for BPM changes.

    Used for v2 maps that have ``_customData._BPMChanges``. Each entry in
    ``bpm_changes`` has ``_BPM`` (new tempo) and ``_time`` (beat position in
    base-BPM beats where the change takes effect).

    Args:
        beat: Beat number to convert.
        base_bpm: Song's base BPM from Info.dat.
        bpm_changes: List of dicts with ``_BPM`` and ``_time`` keys.
        sample_rate: Audio sample rate.
        hop_length: Hop length used for spectrogram.
        offset: Song time offset in seconds.

    Returns:
        Frame index (integer).
    """
    # Build sorted list of (change_beat, new_bpm) segments
    changes = sorted(
        [{"_time": c["_time"], "_BPM": c["_BPM"]} for c in bpm_changes],
        key=lambda c: c["_time"],
    )

    time_seconds = 0.0
    current_bpm = base_bpm
    prev_beat = 0.0

    for change in changes:
        change_beat = float(change["_time"])
        new_bpm = float(change["_BPM"])
        if change_beat >= beat:
            break
        # Accumulate time for the segment [prev_beat, change_beat]
        time_seconds += (change_beat - prev_beat) * 60.0 / current_bpm
        prev_beat = change_beat
        current_bpm = new_bpm

    # Remaining beats after last BPM change
    time_seconds += (beat - prev_beat) * 60.0 / current_bpm
    time_seconds += offset

    return int(round(time_seconds * sample_rate / hop_length))


def frame_to_beat(
    frame: int,
    bpm: float,
    sample_rate: int = 44100,
    hop_length: int = 512,
    offset: float = 0.0,
) -> float:
    """Convert a spectrogram frame index to a beat number.

    Args:
        frame: Frame index.
        bpm: Beats per minute of the song.
        sample_rate: Audio sample rate.
        hop_length: Hop length used for spectrogram.
        offset: Song time offset in seconds.

    Returns:
        Beat number (float).
    """
    time_seconds = (frame * hop_length / sample_rate) - offset
    return time_seconds * bpm / 60.0
