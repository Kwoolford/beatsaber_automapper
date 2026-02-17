"""Tests for PyTorch dataset classes."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import torch

from beatsaber_automapper.data.dataset import (
    DIFFICULTY_MAP,
    OnsetDataset,
    SequenceDataset,
    create_dataloader,
)


def _make_test_pt(tmpdir: Path, song_id: str = "song001", n_frames: int = 512) -> Path:
    """Create a minimal preprocessed .pt file for testing."""
    n_mels = 80
    mel = torch.randn(n_mels, n_frames)

    # Create some onset frames and labels
    onset_frames = torch.tensor([50, 100, 150, 200, 250])
    onset_labels = torch.zeros(n_frames)
    for f in onset_frames:
        onset_labels[f] = 1.0

    # Simple token sequences (one per onset)
    token_sequences = [
        [4, 10, 12, 16, 19, 28, 1],  # NOTE + attrs + EOS
        [4, 11, 13, 17, 20, 29, 1],
        [5, 14, 18, 1],  # BOMB + attrs + EOS
        [4, 10, 15, 16, 22, 30, 1],
        [4, 11, 12, 17, 19, 28, 1],
    ]

    data = {
        "song_id": song_id,
        "bpm": 128.0,
        "mel_spectrogram": mel,
        "difficulties": {
            "Expert": {
                "onset_frames": onset_frames,
                "onset_labels": onset_labels,
                "token_sequences": token_sequences,
            },
        },
    }

    pt_path = tmpdir / f"{song_id}.pt"
    torch.save(data, pt_path)
    return pt_path


def _make_splits(tmpdir: Path, train: list[str], val: list[str], test: list[str]) -> None:
    splits = {"train": train, "val": val, "test": test}
    with open(tmpdir / "splits.json", "w") as f:
        json.dump(splits, f)


# ---------------------------------------------------------------------------
# OnsetDataset
# ---------------------------------------------------------------------------


def test_onset_dataset_length() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        _make_test_pt(tmpdir, "song001", n_frames=512)
        _make_splits(tmpdir, train=["song001"], val=[], test=[])

        ds = OnsetDataset(tmpdir, split="train", window_size=128, hop=64)
        assert len(ds) > 0


def test_onset_dataset_shapes() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        _make_test_pt(tmpdir, "song001", n_frames=512)
        _make_splits(tmpdir, train=["song001"], val=[], test=[])

        ds = OnsetDataset(tmpdir, split="train", window_size=128, hop=64)
        sample = ds[0]

        assert sample["mel"].shape == (80, 128)
        assert sample["labels"].shape == (128,)
        assert sample["difficulty"].shape == ()
        assert sample["difficulty"].dtype == torch.long


def test_onset_dataset_difficulty_filter() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        _make_test_pt(tmpdir, "song001", n_frames=512)
        _make_splits(tmpdir, train=["song001"], val=[], test=[])

        ds_all = OnsetDataset(tmpdir, split="train", window_size=128, hop=64)
        ds_expert = OnsetDataset(
            tmpdir, split="train", window_size=128, hop=64, difficulties=["Expert"]
        )
        ds_easy = OnsetDataset(
            tmpdir, split="train", window_size=128, hop=64, difficulties=["Easy"]
        )

        assert len(ds_expert) == len(ds_all)  # only Expert exists
        assert len(ds_easy) == 0  # Easy doesn't exist


def test_onset_dataset_split_filtering() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        _make_test_pt(tmpdir, "song001", n_frames=512)
        _make_test_pt(tmpdir, "song002", n_frames=512)
        _make_splits(tmpdir, train=["song001"], val=["song002"], test=[])

        ds_train = OnsetDataset(tmpdir, split="train", window_size=128, hop=64)
        ds_val = OnsetDataset(tmpdir, split="val", window_size=128, hop=64)

        # Both should have data but from different songs
        assert len(ds_train) > 0
        assert len(ds_val) > 0


# ---------------------------------------------------------------------------
# SequenceDataset
# ---------------------------------------------------------------------------


def test_sequence_dataset_length() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        _make_test_pt(tmpdir, "song001", n_frames=512)
        _make_splits(tmpdir, train=["song001"], val=[], test=[])

        ds = SequenceDataset(tmpdir, split="train", context_frames=64, max_token_len=32)
        # Should have 5 onsets
        assert len(ds) == 5


def test_sequence_dataset_shapes() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        _make_test_pt(tmpdir, "song001", n_frames=512)
        _make_splits(tmpdir, train=["song001"], val=[], test=[])

        ds = SequenceDataset(tmpdir, split="train", context_frames=64, max_token_len=32)
        sample = ds[0]

        assert sample["mel"].shape == (80, 64)
        assert sample["tokens"].shape == (32,)
        assert sample["tokens"].dtype == torch.long
        assert sample["token_length"].shape == ()
        assert sample["difficulty"].shape == ()


def test_sequence_dataset_token_padding() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        _make_test_pt(tmpdir, "song001", n_frames=512)
        _make_splits(tmpdir, train=["song001"], val=[], test=[])

        ds = SequenceDataset(tmpdir, split="train", context_frames=64, max_token_len=32)
        sample = ds[0]

        # Token length should be <= max_token_len
        token_len = sample["token_length"].item()
        assert token_len <= 32
        # Tokens after token_length should be 0 (PAD)
        assert (sample["tokens"][token_len:] == 0).all()


# ---------------------------------------------------------------------------
# DataLoader integration
# ---------------------------------------------------------------------------


def test_onset_dataloader_batching() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        _make_test_pt(tmpdir, "song001", n_frames=512)
        _make_splits(tmpdir, train=["song001"], val=[], test=[])

        ds = OnsetDataset(tmpdir, split="train", window_size=128, hop=64)
        dl = create_dataloader(ds, batch_size=2, shuffle=False, num_workers=0)

        batch = next(iter(dl))
        assert batch["mel"].shape[0] == 2
        assert batch["mel"].shape[1] == 80
        assert batch["mel"].shape[2] == 128
        assert batch["labels"].shape == (2, 128)
        assert batch["difficulty"].shape == (2,)


def test_sequence_dataloader_batching() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        _make_test_pt(tmpdir, "song001", n_frames=512)
        _make_splits(tmpdir, train=["song001"], val=[], test=[])

        ds = SequenceDataset(tmpdir, split="train", context_frames=64, max_token_len=32)
        dl = create_dataloader(ds, batch_size=3, shuffle=False, num_workers=0)

        batch = next(iter(dl))
        assert batch["mel"].shape == (3, 80, 64)
        assert batch["tokens"].shape == (3, 32)
        assert batch["token_length"].shape == (3,)
        assert batch["difficulty"].shape == (3,)


# ---------------------------------------------------------------------------
# DIFFICULTY_MAP
# ---------------------------------------------------------------------------


def test_difficulty_map() -> None:
    assert DIFFICULTY_MAP["Easy"] == 0
    assert DIFFICULTY_MAP["ExpertPlus"] == 4
    assert len(DIFFICULTY_MAP) == 5


# ---------------------------------------------------------------------------
# No splits.json fallback
# ---------------------------------------------------------------------------


def test_onset_dataset_no_splits() -> None:
    """Without splits.json, all songs should be included."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        _make_test_pt(tmpdir, "song001", n_frames=512)
        # No splits.json

        ds = OnsetDataset(tmpdir, split="train", window_size=128, hop=64)
        assert len(ds) > 0
