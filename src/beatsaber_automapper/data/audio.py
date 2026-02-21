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

    Uses soundfile for reading (avoids torchcodec dependency in torchaudio nightly).

    Args:
        path: Path to audio file (.mp3, .ogg, .wav).
        target_sr: Target sample rate in Hz.

    Returns:
        Tuple of (waveform tensor [1, samples], sample_rate).
    """
    path = Path(path)
    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    # data shape: [samples, channels]

    # Convert to mono by averaging channels
    if data.shape[1] > 1:
        data = data.mean(axis=1, keepdims=True)

    # Convert to torch: [channels, samples]
    waveform = torch.from_numpy(data.T)

    # Resample if needed
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
        waveform = resampler(waveform)

    return waveform, target_sr


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
