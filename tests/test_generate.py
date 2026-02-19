"""Tests for generation/generate.py — end-to-end pipeline."""

from __future__ import annotations

import wave
import zipfile
from pathlib import Path

import torch

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_wav(path: Path, duration_s: float = 0.5, sample_rate: int = 44100) -> Path:
    """Write a minimal WAV file (silence) at the given path."""
    n_samples = int(duration_s * sample_rate)
    with wave.open(str(path), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(b"\x00\x00" * n_samples)
    return path


# ---------------------------------------------------------------------------
# Tests for predict_onsets
# ---------------------------------------------------------------------------


class TestPredictOnsets:
    def test_returns_list(self, tmp_path):
        from beatsaber_automapper.generation.generate import (
            _make_default_onset_module,
            predict_onsets,
        )

        # Short synthetic mel
        mel = torch.randn(80, 64)
        module = _make_default_onset_module()

        frames = predict_onsets(
            onset_module=module,
            mel=mel,
            difficulty_idx=3,
            threshold=0.0,  # Low threshold so we get some hits
            min_distance=2,
            device=torch.device("cpu"),
        )
        assert isinstance(frames, list)

    def test_high_threshold_returns_empty(self):
        from beatsaber_automapper.generation.generate import (
            _make_default_onset_module,
            predict_onsets,
        )

        mel = torch.randn(80, 32)
        module = _make_default_onset_module()

        frames = predict_onsets(
            onset_module=module,
            mel=mel,
            difficulty_idx=3,
            threshold=2.0,  # Impossible threshold
            min_distance=5,
            device=torch.device("cpu"),
        )
        assert frames == []

    def test_frame_indices_in_range(self):
        from beatsaber_automapper.generation.generate import (
            _make_default_onset_module,
            predict_onsets,
        )

        n_frames = 64
        mel = torch.randn(80, n_frames)
        module = _make_default_onset_module()

        frames = predict_onsets(
            onset_module=module,
            mel=mel,
            difficulty_idx=0,
            threshold=0.0,
            min_distance=2,
            device=torch.device("cpu"),
        )
        for f in frames:
            assert 0 <= f < n_frames


# ---------------------------------------------------------------------------
# Tests for generate_note_sequence
# ---------------------------------------------------------------------------


class TestGenerateNoteSequence:
    def test_returns_list(self):
        from beatsaber_automapper.generation.generate import (
            _make_default_sequence_module,
            generate_note_sequence,
        )

        module = _make_default_sequence_module()
        # Tiny audio feature context
        audio_features = torch.randn(1, 8, 512)

        tokens = generate_note_sequence(
            seq_module=module,
            audio_features=audio_features,
            difficulty_idx=3,
            beam_size=2,
            temperature=1.0,
            use_sampling=False,
            max_length=16,
            device=torch.device("cpu"),
        )
        assert isinstance(tokens, list)

    def test_nucleus_sampling_returns_list(self):
        from beatsaber_automapper.generation.generate import (
            _make_default_sequence_module,
            generate_note_sequence,
        )

        module = _make_default_sequence_module()
        audio_features = torch.randn(1, 8, 512)

        tokens = generate_note_sequence(
            seq_module=module,
            audio_features=audio_features,
            difficulty_idx=2,
            use_sampling=True,
            top_p=0.9,
            max_length=16,
            device=torch.device("cpu"),
        )
        assert isinstance(tokens, list)

    def test_no_bos_or_eos_in_output(self):
        from beatsaber_automapper.data.tokenizer import BOS, EOS
        from beatsaber_automapper.generation.generate import (
            _make_default_sequence_module,
            generate_note_sequence,
        )

        module = _make_default_sequence_module()
        audio_features = torch.randn(1, 8, 512)

        tokens = generate_note_sequence(
            seq_module=module,
            audio_features=audio_features,
            difficulty_idx=3,
            beam_size=2,
            max_length=16,
            device=torch.device("cpu"),
        )
        assert BOS not in tokens
        assert EOS not in tokens


# ---------------------------------------------------------------------------
# Tests for generate_level (end-to-end)
# ---------------------------------------------------------------------------


class TestGenerateLevel:
    def test_creates_zip(self, tmp_path):
        from beatsaber_automapper.generation.generate import generate_level

        wav = _make_wav(tmp_path / "song.wav")
        out = tmp_path / "level.zip"

        result = generate_level(
            audio_path=wav,
            output_path=out,
            difficulty="Expert",
            bpm=120.0,
            device="cpu",
        )
        assert result == out
        assert out.exists()
        assert zipfile.is_zipfile(out)

    def test_zip_has_info_dat(self, tmp_path):
        from beatsaber_automapper.generation.generate import generate_level

        wav = _make_wav(tmp_path / "song.wav")
        out = tmp_path / "level.zip"
        generate_level(audio_path=wav, output_path=out, bpm=120.0, device="cpu")

        with zipfile.ZipFile(out) as zf:
            assert "Info.dat" in zf.namelist()

    def test_zip_has_difficulty_dat(self, tmp_path):
        from beatsaber_automapper.generation.generate import generate_level

        wav = _make_wav(tmp_path / "song.wav")
        out = tmp_path / "level.zip"
        generate_level(
            audio_path=wav, output_path=out, difficulty="Hard", bpm=120.0, device="cpu"
        )

        with zipfile.ZipFile(out) as zf:
            assert "HardStandard.dat" in zf.namelist()

    def test_default_song_name_from_filename(self, tmp_path):
        import json
        import zipfile

        from beatsaber_automapper.generation.generate import generate_level

        wav = _make_wav(tmp_path / "mysong.wav")
        out = tmp_path / "level.zip"
        generate_level(audio_path=wav, output_path=out, bpm=120.0, device="cpu")

        with zipfile.ZipFile(out) as zf:
            info = json.loads(zf.read("Info.dat"))
        assert info["_songName"] == "mysong"

    def test_custom_song_name(self, tmp_path):
        import json
        import zipfile

        from beatsaber_automapper.generation.generate import generate_level

        wav = _make_wav(tmp_path / "song.wav")
        out = tmp_path / "level.zip"
        generate_level(
            audio_path=wav, output_path=out, song_name="My Custom Song", bpm=120.0, device="cpu"
        )

        with zipfile.ZipFile(out) as zf:
            info = json.loads(zf.read("Info.dat"))
        assert info["_songName"] == "My Custom Song"

    def test_all_difficulties(self, tmp_path):
        from beatsaber_automapper.generation.generate import generate_level

        for diff in ["Easy", "Normal", "Hard", "Expert", "ExpertPlus"]:
            wav = _make_wav(tmp_path / f"song_{diff}.wav")
            out = tmp_path / f"level_{diff}.zip"
            result = generate_level(
                audio_path=wav, output_path=out, difficulty=diff, bpm=120.0, device="cpu"
            )
            assert result.exists(), f"Failed for difficulty {diff}"
