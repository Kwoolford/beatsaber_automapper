"""Tests for audio loading and mel spectrogram extraction."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from beatsaber_automapper.data.audio import (
    beat_to_frame,
    extract_mel_spectrogram,
    frame_to_beat,
    load_audio,
)


def _make_sine_wav(path: Path, sr: int = 44100, duration: float = 1.0, freq: float = 440.0):
    """Create a mono sine wave WAV file using soundfile."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    waveform = np.sin(2 * np.pi * freq * t).astype(np.float32)
    sf.write(str(path), waveform, sr)


# ---------------------------------------------------------------------------
# load_audio
# ---------------------------------------------------------------------------


def test_load_audio_mono() -> None:
    """Load a mono WAV and verify shape."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test.wav"
        _make_sine_wav(path, sr=44100, duration=0.5)

        waveform, sr = load_audio(path)

    assert sr == 44100
    assert waveform.shape[0] == 1
    assert waveform.shape[1] > 0


def test_load_audio_stereo_to_mono() -> None:
    """Load a stereo WAV and verify it's converted to mono."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "stereo.wav"
        sr = 44100
        duration = 0.5
        n_samples = int(sr * duration)
        t = np.linspace(0, duration, n_samples, endpoint=False)
        left = np.sin(2 * np.pi * 440 * t).astype(np.float32)
        right = np.sin(2 * np.pi * 880 * t).astype(np.float32)
        stereo = np.column_stack([left, right])
        sf.write(str(path), stereo, sr)

        waveform, out_sr = load_audio(path)

    assert out_sr == 44100
    assert waveform.shape[0] == 1


def test_load_audio_resample() -> None:
    """Load audio at different sample rate and verify resampling."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test_22k.wav"
        _make_sine_wav(path, sr=22050, duration=1.0)

        waveform, sr = load_audio(path, target_sr=44100)

    assert sr == 44100
    # Resampled from 22050 to 44100 should roughly double sample count
    assert waveform.shape[1] > 22050


# ---------------------------------------------------------------------------
# extract_mel_spectrogram
# ---------------------------------------------------------------------------


def test_mel_spectrogram_shape() -> None:
    """Verify mel spectrogram output shape."""
    sr = 44100
    duration = 1.0
    n_samples = int(sr * duration)
    waveform = torch.randn(1, n_samples)

    mel = extract_mel_spectrogram(waveform, sample_rate=sr, n_mels=80, n_fft=1024, hop_length=512)

    assert mel.shape[0] == 80
    expected_frames = n_samples // 512 + 1
    assert abs(mel.shape[1] - expected_frames) <= 1


def test_mel_spectrogram_1d_input() -> None:
    """1D waveform input should also work."""
    waveform = torch.randn(44100)
    mel = extract_mel_spectrogram(waveform, sample_rate=44100)
    assert mel.dim() == 2
    assert mel.shape[0] == 80


def test_mel_spectrogram_no_nan() -> None:
    """Log-mel should not contain NaN or -inf values."""
    waveform = torch.randn(1, 44100)
    mel = extract_mel_spectrogram(waveform, sample_rate=44100)
    assert not torch.isnan(mel).any()
    assert not torch.isinf(mel).any()


def test_mel_spectrogram_silent_audio() -> None:
    """Silent audio should produce finite values (no log(0))."""
    waveform = torch.zeros(1, 44100)
    mel = extract_mel_spectrogram(waveform, sample_rate=44100)
    assert not torch.isnan(mel).any()
    assert not torch.isinf(mel).any()


# ---------------------------------------------------------------------------
# beat_to_frame / frame_to_beat
# ---------------------------------------------------------------------------


def test_beat_to_frame_basic() -> None:
    """Beat 0 at any BPM should be frame 0 (with no offset)."""
    assert beat_to_frame(0.0, bpm=120.0) == 0


def test_beat_to_frame_calculation() -> None:
    """Beat 1 at 120 BPM = 0.5 seconds = sample 22050 = frame 43."""
    frame = beat_to_frame(1.0, bpm=120.0, sample_rate=44100, hop_length=512)
    expected = round(0.5 * 44100 / 512)
    assert frame == expected


def test_frame_to_beat_inverse() -> None:
    """frame_to_beat should be the inverse of beat_to_frame."""
    bpm = 128.0
    original_beat = 10.5
    frame = beat_to_frame(original_beat, bpm=bpm)
    recovered_beat = frame_to_beat(frame, bpm=bpm)
    # Should be very close (rounding error from int frame)
    assert abs(recovered_beat - original_beat) < 0.1


def test_beat_frame_with_offset() -> None:
    """Verify offset shifts the frame index."""
    frame_no_offset = beat_to_frame(1.0, bpm=120.0)
    frame_with_offset = beat_to_frame(1.0, bpm=120.0, offset=0.5)
    # 0.5s offset adds ~44100*0.5/512 ≈ 43 frames
    assert frame_with_offset > frame_no_offset
