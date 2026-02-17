"""PyTorch Dataset classes for Beat Saber map training data.

Provides datasets for each training stage:
    - OnsetDataset: Audio frames + binary onset labels (Stage 1)
    - SequenceDataset: Audio frames + onset times + token sequences (Stage 2)
    - LightingDataset: Audio frames + note sequence + lighting tokens (Stage 3)

Preprocessed .pt format (one per song):
    {"song_id": str, "bpm": float, "mel_spectrogram": Tensor[n_mels, n_frames],
     "difficulties": {"Expert": {"onset_frames": Tensor[n_onsets],
       "onset_labels": Tensor[n_frames], "token_sequences": list[list[int]]}}}
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)

DIFFICULTY_MAP: dict[str, int] = {
    "Easy": 0,
    "Normal": 1,
    "Hard": 2,
    "Expert": 3,
    "ExpertPlus": 4,
}


class OnsetDataset(Dataset):
    """Dataset for Stage 1 onset prediction training.

    Produces sliding windows over mel spectrogram frames with
    Gaussian-smoothed onset labels for binary classification.

    Each sample: {mel: [n_mels, window_size], labels: [window_size], difficulty: int}
    """

    def __init__(
        self,
        data_dir: Path | str,
        split: str = "train",
        window_size: int = 256,
        hop: int = 128,
        difficulties: list[str] | None = None,
    ) -> None:
        """Initialize OnsetDataset.

        Args:
            data_dir: Directory with preprocessed .pt files and splits.json.
            split: One of "train", "val", "test".
            window_size: Number of frames per sliding window.
            hop: Hop between consecutive windows.
            difficulties: Which difficulties to include. Defaults to all available.
        """
        self.data_dir = Path(data_dir)
        self.window_size = window_size
        self.hop = hop
        self.target_difficulties = difficulties

        # Load split info
        splits_path = self.data_dir / "splits.json"
        if splits_path.exists():
            with open(splits_path) as f:
                splits = json.load(f)
            song_ids = set(splits.get(split, []))
        else:
            song_ids = None

        # Index all windows across all songs
        self.samples: list[tuple[Path, str, int, int]] = []  # (pt_path, diff, start_frame, diff_id)
        pt_files = sorted(self.data_dir.glob("*.pt"))
        for pt_path in pt_files:
            song_id = pt_path.stem
            if song_ids is not None and song_id not in song_ids:
                continue
            data = torch.load(pt_path, weights_only=False)
            mel = data["mel_spectrogram"]
            n_frames = mel.shape[1]

            for diff_name, diff_data in data.get("difficulties", {}).items():
                if self.target_difficulties and diff_name not in self.target_difficulties:
                    continue
                diff_id = DIFFICULTY_MAP.get(diff_name, 3)
                # Generate sliding windows
                start = 0
                while start + window_size <= n_frames:
                    self.samples.append((pt_path, diff_name, start, diff_id))
                    start += hop
                # Include last partial window if there's remaining data
                if start < n_frames and n_frames > window_size:
                    self.samples.append((pt_path, diff_name, n_frames - window_size, diff_id))

        # Cache loaded data to avoid re-reading
        self._cache: dict[str, dict] = {}

    def _load(self, pt_path: Path) -> dict:
        key = str(pt_path)
        if key not in self._cache:
            self._cache[key] = torch.load(pt_path, weights_only=False)
        return self._cache[key]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        pt_path, diff_name, start, diff_id = self.samples[idx]
        data = self._load(pt_path)
        mel = data["mel_spectrogram"]
        labels = data["difficulties"][diff_name]["onset_labels"]

        end = start + self.window_size
        return {
            "mel": mel[:, start:end],
            "labels": labels[start:end],
            "difficulty": torch.tensor(diff_id, dtype=torch.long),
        }


class SequenceDataset(Dataset):
    """Dataset for Stage 2 note sequence generation training.

    Produces per-onset samples with a context window of mel frames
    centered on the onset, plus the token sequence for that onset.

    Each sample: {mel: [n_mels, context_frames], tokens: [max_token_len],
                  token_length: int, difficulty: int}
    """

    def __init__(
        self,
        data_dir: Path | str,
        split: str = "train",
        context_frames: int = 128,
        max_token_len: int = 64,
        difficulties: list[str] | None = None,
    ) -> None:
        """Initialize SequenceDataset.

        Args:
            data_dir: Directory with preprocessed .pt files and splits.json.
            split: One of "train", "val", "test".
            context_frames: Number of mel frames for audio context around each onset.
            max_token_len: Maximum token sequence length (padded/truncated).
            difficulties: Which difficulties to include. Defaults to all available.
        """
        self.data_dir = Path(data_dir)
        self.context_frames = context_frames
        self.max_token_len = max_token_len
        self.target_difficulties = difficulties

        # Load split info
        splits_path = self.data_dir / "splits.json"
        if splits_path.exists():
            with open(splits_path) as f:
                splits = json.load(f)
            song_ids = set(splits.get(split, []))
        else:
            song_ids = None

        # Index all onset samples
        # (pt_path, diff_name, onset_idx, diff_id)
        self.samples: list[tuple[Path, str, int, int]] = []
        pt_files = sorted(self.data_dir.glob("*.pt"))
        for pt_path in pt_files:
            song_id = pt_path.stem
            if song_ids is not None and song_id not in song_ids:
                continue
            data = torch.load(pt_path, weights_only=False)

            for diff_name, diff_data in data.get("difficulties", {}).items():
                if self.target_difficulties and diff_name not in self.target_difficulties:
                    continue
                diff_id = DIFFICULTY_MAP.get(diff_name, 3)
                n_onsets = len(diff_data.get("token_sequences", []))
                for onset_idx in range(n_onsets):
                    self.samples.append((pt_path, diff_name, onset_idx, diff_id))

        self._cache: dict[str, dict] = {}

    def _load(self, pt_path: Path) -> dict:
        key = str(pt_path)
        if key not in self._cache:
            self._cache[key] = torch.load(pt_path, weights_only=False)
        return self._cache[key]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        pt_path, diff_name, onset_idx, diff_id = self.samples[idx]
        data = self._load(pt_path)
        mel = data["mel_spectrogram"]
        diff_data = data["difficulties"][diff_name]

        onset_frame = int(diff_data["onset_frames"][onset_idx].item())
        token_seq = diff_data["token_sequences"][onset_idx]

        # Extract context window centered on onset
        n_frames = mel.shape[1]
        half = self.context_frames // 2
        start = max(0, onset_frame - half)
        end = start + self.context_frames
        if end > n_frames:
            end = n_frames
            start = max(0, end - self.context_frames)

        mel_window = mel[:, start:end]
        # Pad if context is shorter than expected (very short audio)
        if mel_window.shape[1] < self.context_frames:
            pad_size = self.context_frames - mel_window.shape[1]
            mel_window = torch.nn.functional.pad(mel_window, (0, pad_size))

        # Pad/truncate token sequence
        if len(token_seq) > self.max_token_len:
            token_seq = token_seq[: self.max_token_len]
        token_length = len(token_seq)
        padded = token_seq + [0] * (self.max_token_len - len(token_seq))

        return {
            "mel": mel_window,
            "tokens": torch.tensor(padded, dtype=torch.long),
            "token_length": torch.tensor(token_length, dtype=torch.long),
            "difficulty": torch.tensor(diff_id, dtype=torch.long),
        }


class LightingDataset(Dataset):
    """Dataset for Stage 3 lighting generation training."""

    def __init__(self) -> None:
        raise NotImplementedError("LightingDataset will be implemented in PR 6")


def create_dataloader(
    dataset: Dataset,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = True,
) -> DataLoader:
    """Create a DataLoader with sensible defaults.

    Args:
        dataset: PyTorch dataset to wrap.
        batch_size: Batch size.
        shuffle: Whether to shuffle.
        num_workers: Number of data loading workers.
        pin_memory: Pin memory for GPU transfer.

    Returns:
        Configured DataLoader.
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
