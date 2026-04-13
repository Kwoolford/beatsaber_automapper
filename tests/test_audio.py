"""Tests for audio loading and mel spectrogram extraction."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from beatsaber_automapper.data.audio import (
    SECTION_TYPES,
    beat_to_frame,
    compute_section_features,
    detect_sections,
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


# ---------------------------------------------------------------------------
# detect_sections / compute_section_features
# ---------------------------------------------------------------------------


def _make_synthetic_song(sr: int = 44100, duration: float = 30.0) -> torch.Tensor:
    """Create a synthetic song with distinct sections for testing.

    Alternates between quiet sine-wave and loud noise+sine sections
    to simulate verse/chorus structure.
    """
    n_samples = int(sr * duration)
    t = np.linspace(0, duration, n_samples, endpoint=False)
    waveform = np.zeros(n_samples, dtype=np.float32)

    section_dur = duration / 3
    for i in range(3):
        start = int(i * section_dur * sr)
        end = int((i + 1) * section_dur * sr)
        if i % 2 == 0:
            # Quiet section (verse-like)
            waveform[start:end] = 0.2 * np.sin(2 * np.pi * 220 * t[start:end])
        else:
            # Loud section (chorus-like)
            waveform[start:end] = 0.8 * np.sin(2 * np.pi * 440 * t[start:end])
            waveform[start:end] += 0.3 * np.random.randn(end - start).astype(np.float32)

    return torch.from_numpy(waveform).unsqueeze(0)


def test_detect_sections_returns_valid_types() -> None:
    """All returned section types should be from SECTION_TYPES."""
    waveform = _make_synthetic_song(duration=30.0)
    sections = detect_sections(waveform, sample_rate=44100)

    assert len(sections) >= 3
    for section_type, start, end in sections:
        assert section_type in SECTION_TYPES
        assert end > start
        assert start >= 0.0


def test_detect_sections_covers_full_duration() -> None:
    """Sections should cover the full song duration without gaps."""
    duration = 30.0
    waveform = _make_synthetic_song(duration=duration)
    sections = detect_sections(waveform, sample_rate=44100)

    # First section starts at 0
    assert sections[0][1] == 0.0
    # Last section ends at duration
    assert abs(sections[-1][2] - duration) < 0.5
    # No gaps between consecutive sections
    for i in range(len(sections) - 1):
        assert abs(sections[i][2] - sections[i + 1][1]) < 0.01


def test_detect_sections_n_segments() -> None:
    """Specifying n_segments should control the number of sections."""
    waveform = _make_synthetic_song(duration=30.0)
    sections = detect_sections(waveform, sample_rate=44100, n_segments=5)
    assert len(sections) >= 3  # May merge some, but should be close to 5


def test_compute_section_features_shapes() -> None:
    """Section features should have correct shapes."""
    sections = [("intro", 0.0, 5.0), ("verse", 5.0, 15.0), ("chorus", 15.0, 30.0)]
    n_frames = 2584  # ~30s at 44100/512

    section_ids, section_progress = compute_section_features(
        sections, n_frames=n_frames, hop_length=512, sample_rate=44100
    )

    assert section_ids.shape == (n_frames,)
    assert section_progress.shape == (n_frames,)
    assert section_ids.dtype == torch.long
    assert section_progress.dtype == torch.float32


def test_compute_section_features_values() -> None:
    """Section IDs should match SECTION_TYPES indices, progress should be 0-1."""
    sections = [("intro", 0.0, 5.0), ("chorus", 5.0, 10.0)]
    n_frames = 862  # ~10s

    section_ids, section_progress = compute_section_features(
        sections, n_frames=n_frames, hop_length=512, sample_rate=44100
    )

    # intro = index 0, chorus = index 2
    assert section_ids[0].item() == SECTION_TYPES.index("intro")
    # Midpoint of chorus section should have chorus ID
    chorus_start_frame = int(5.0 * 44100 / 512)
    if chorus_start_frame < n_frames:
        assert section_ids[chorus_start_frame].item() == SECTION_TYPES.index("chorus")

    # Progress should be in [0, 1]
    assert section_progress.min() >= 0.0
    assert section_progress.max() <= 1.0


def test_detect_sections_short_audio() -> None:
    """Very short audio should still return at least one section."""
    waveform = torch.randn(1, 44100 * 3)  # 3 seconds
    sections = detect_sections(waveform, sample_rate=44100)
    assert len(sections) >= 1
