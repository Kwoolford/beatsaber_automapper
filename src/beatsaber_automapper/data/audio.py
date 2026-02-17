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


def beat_to_frame(
    beat: float,
    bpm: float,
    sample_rate: int = 44100,
    hop_length: int = 512,
    offset: float = 0.0,
) -> int:
    """Convert a beat number to a spectrogram frame index.

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
