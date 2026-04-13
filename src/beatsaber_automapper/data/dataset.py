"""PyTorch Dataset classes for Beat Saber map training data.

Provides datasets for each training stage:
    - OnsetDataset: Audio frames + binary onset labels (Stage 1)
    - SequenceDataset: Audio frames + onset times + token sequences (Stage 2)
    - LightingDataset: Audio frames + note sequence + lighting tokens (Stage 3)

Preprocessed .pt format (one per song):
    {"song_id": str, "bpm": float, "mel_spectrogram": Tensor[n_mels, n_frames],
     "difficulties": {"Expert": {"onset_frames": Tensor[n_onsets],
       "onset_labels": Tensor[n_frames], "token_sequences": list[list[int]]}},
     "mod_requirements": {"category": str, "requirements": list[str],
       "suggestions": list[str], "genre": str}}

The ``exclude_categories`` parameter on each dataset class filters out maps
whose ``mod_requirements.category`` matches, enabling clean separation of
vanilla maps from modded ones (noodle, mapping_extensions, chroma, vivify).

Genre conditioning: each sample includes a ``"genre"`` int tensor (0-10) drawn
from ``mod_requirements.genre`` via ``GENRE_MAP`` in the tokenizer.
"""

from __future__ import annotations

import json
import logging
import random
from collections import OrderedDict
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset, RandomSampler

from beatsaber_automapper.data.tokenizer import (
    ANGLE_OFFSET_COUNT,
    ANGLE_OFFSET_OFFSET,
    ARC_END,
    ARC_START,
    BOMB,
    CHAIN,
    COL_COUNT,
    COL_OFFSET,
    COLOR_COUNT,
    COLOR_OFFSET,
    DIR_COUNT,
    DIR_OFFSET,
    GENRE_MAP,
    NOTE,
    WALL,
)

logger = logging.getLogger(__name__)

# Number of per-frame structure feature channels.
# Channels 0-5: energy features (RMS, onset_strength, bass, mid, high, centroid)
# Channels 6-7: section_id (normalized), section_progress
N_STRUCTURE_FEATURES = 8


def _pad_structure(
    structure: torch.Tensor, target_channels: int = N_STRUCTURE_FEATURES,
) -> torch.Tensor:
    """Pad structure features to target channel count (backward compat for [6, T] -> [8, T])."""
    if structure.shape[0] >= target_channels:
        return structure[:target_channels]
    pad = torch.zeros(target_channels - structure.shape[0], structure.shape[1])
    return torch.cat([structure, pad], dim=0)


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

    Each sample: {mel: [n_mels, window_size], labels: [window_size],
                  difficulty: int, genre: int}
    """

    def __init__(
        self,
        data_dir: Path | str,
        split: str = "train",
        window_size: int = 256,
        hop: int = 128,
        difficulties: list[str] | None = None,
        exclude_categories: list[str] | None = None,
    ) -> None:
        """Initialize OnsetDataset.

        Args:
            data_dir: Directory with preprocessed .pt files and splits.json.
            split: One of "train", "val", "test".
            window_size: Number of frames per sliding window.
            hop: Hop between consecutive windows.
            difficulties: Which difficulties to include. Defaults to all available.
            exclude_categories: Mod categories to exclude (e.g. ["noodle",
                "mapping_extensions"]). Filters by mod_requirements.category
                embedded in each .pt file. None includes all categories.
        """
        self.data_dir = Path(data_dir)
        self.window_size = window_size
        self.hop = hop
        self.target_difficulties = difficulties
        self.exclude_categories = set(exclude_categories) if exclude_categories else set()

        # Load split info
        splits_path = self.data_dir / "splits.json"
        if splits_path.exists():
            with open(splits_path) as f:
                splits = json.load(f)
            song_ids = set(splits.get(split, []))
        else:
            song_ids = None

        # Load blacklist if present (outlier filtering)
        blacklist_path = self.data_dir / "blacklist.json"
        blacklist: set[str] = set()
        if blacklist_path.exists():
            with open(blacklist_path) as f:
                blacklist = set(json.load(f).keys())

        # Load whitelist if present (curated subset — only train on these maps)
        whitelist_path = self.data_dir / "whitelist.json"
        whitelist: set[str] | None = None
        if whitelist_path.exists():
            with open(whitelist_path) as f:
                whitelist = set(json.load(f).keys())

        # Load frame index if available (avoids reading every .pt file at init)
        frame_index_path = self.data_dir / "frame_index.json"
        frame_index: dict | None = None
        if frame_index_path.exists():
            with open(frame_index_path) as f:
                frame_index = json.load(f)

        # Index all windows across all songs
        # (pt_path, diff_name, start_frame, diff_id, genre_idx)
        self.samples: list[tuple[Path, str, int, int, int]] = []
        pt_files = sorted(self.data_dir.glob("*.pt"))
        for pt_path in pt_files:
            song_id = pt_path.stem
            if song_ids is not None and song_id not in song_ids:
                continue
            if song_id in blacklist:
                continue
            if whitelist is not None and song_id not in whitelist:
                continue

            if frame_index is not None and song_id in frame_index:
                # Fast path: use pre-built index, no file I/O
                entry = frame_index[song_id]
                cat = entry.get("category", "vanilla")
                if self.exclude_categories and cat in self.exclude_categories:
                    continue
                genre_idx = GENRE_MAP.get(entry.get("genre", "unknown"), 0)
                n_frames = entry["n_frames"]
                diff_names = entry.get("difficulties", [])
            else:
                # Slow path: load .pt file to get metadata
                data = torch.load(pt_path, weights_only=False)
                mod_reqs = data.get("mod_requirements", {})
                cat = mod_reqs.get("category", "vanilla")
                if self.exclude_categories and cat in self.exclude_categories:
                    continue
                genre_idx = GENRE_MAP.get(mod_reqs.get("genre", "unknown"), 0)
                n_frames = data["mel_spectrogram"].shape[1]
                diff_names = list(data.get("difficulties", {}).keys())

            for diff_name in diff_names:
                if self.target_difficulties and diff_name not in self.target_difficulties:
                    continue
                diff_id = DIFFICULTY_MAP.get(diff_name, 3)
                # Generate sliding windows
                start = 0
                while start + window_size <= n_frames:
                    self.samples.append((pt_path, diff_name, start, diff_id, genre_idx))
                    start += hop
                # Include last partial window if there's remaining data
                if start < n_frames and n_frames > window_size:
                    tail_start = n_frames - window_size
                    self.samples.append((pt_path, diff_name, tail_start, diff_id, genre_idx))

        # Bounded LRU cache: keeps the most-recently-used files in memory.
        # 100 files × ~6 MB each ≈ 600 MB per worker — safe with 8+ workers overnight.
        self._cache: OrderedDict[str, dict] = OrderedDict()
        self._cache_max = 100

    def _load(self, pt_path: Path) -> dict:
        key = str(pt_path)
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        data = torch.load(pt_path, weights_only=False)
        self._cache[key] = data
        if len(self._cache) > self._cache_max:
            self._cache.popitem(last=False)
        return data

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        pt_path, diff_name, start, diff_id, genre_idx = self.samples[idx]
        data = self._load(pt_path)
        mel = data["mel_spectrogram"]
        diff_data = data["difficulties"][diff_name]
        labels = diff_data["onset_labels"]

        end = start + self.window_size

        # Extract actual onset frame positions within this window (for proper F1 eval).
        # Padded to fixed size so default collation works; n_onsets tracks the real count.
        onset_frames = diff_data["onset_frames"]
        mask = (onset_frames >= start) & (onset_frames < end)
        window_onsets = onset_frames[mask] - start  # relative to window start
        max_onsets = 256  # generous upper bound per 1024-frame window
        n_real = min(len(window_onsets), max_onsets)
        padded_onsets = torch.zeros(max_onsets, dtype=torch.long)
        padded_onsets[:n_real] = window_onsets[:n_real]

        # Structure features (backward compat: zeros if not in .pt file)
        structure = data.get("structure_features", None)
        if structure is not None:
            structure = _pad_structure(structure)
            structure_window = structure[:, start:end]
            if structure_window.shape[1] < self.window_size:
                pad_size = self.window_size - structure_window.shape[1]
                structure_window = torch.nn.functional.pad(structure_window, (0, pad_size))
        else:
            structure_window = torch.zeros(N_STRUCTURE_FEATURES, self.window_size)

        return {
            "mel": mel[:, start:end],
            "labels": labels[start:end],
            "onset_frames": padded_onsets,
            "n_onsets": torch.tensor(n_real, dtype=torch.long),
            "difficulty": torch.tensor(diff_id, dtype=torch.long),
            "genre": torch.tensor(genre_idx, dtype=torch.long),
            "structure": structure_window,
        }


# ---------------------------------------------------------------------------
# Mirror augmentation: flip left↔right, swap red↔blue, mirror directions
# ---------------------------------------------------------------------------

# Direction mirror map: left↔right (2↔3, 4↔5, 6↔7), up/down/any unchanged
_DIR_MIRROR = {0: 0, 1: 1, 2: 3, 3: 2, 4: 5, 5: 4, 6: 7, 7: 6, 8: 8}

# Column mirror map: 0↔3, 1↔2
_COL_MIRROR = {0: 3, 1: 2, 2: 1, 3: 0}

# Event types that have COLOR, COL, DIR fields
_EVENTS_WITH_COLOR = frozenset({NOTE, ARC_START, ARC_END, CHAIN})


def _mirror_token_sequence(tokens: list[int]) -> list[int]:
    """Mirror a token sequence: flip columns, swap colors, mirror directions.

    This is equivalent to reflecting the play field left↔right, which swaps
    red/blue hands and mirrors all spatial positions and directions.

    Token grammar per event type:
        NOTE:      [NOTE] [COLOR] [COL] [ROW] [DIR] [ANGLE]
        BOMB:      [BOMB] [COL] [ROW]
        WALL:      [WALL] [COL] [ROW] [W] [H] [DUR_INT] [DUR_FRAC]
        ARC_START: [ARC_START] [COLOR] [COL] [ROW] [DIR] [MU]
        ARC_END:   [ARC_END] [COLOR] [COL] [ROW] [DIR] [MU] [MID]
        CHAIN:     [CHAIN] [COLOR] [COL] [ROW] [DIR] [TAIL_COL] [TAIL_ROW] [SLICE] [SQUISH]
    """
    result = list(tokens)
    i = 0
    while i < len(result):
        tok = result[i]
        if tok == NOTE and i + 5 < len(result):
            # Swap color: 0↔1
            color_val = result[i + 1] - COLOR_OFFSET
            if 0 <= color_val < COLOR_COUNT:
                result[i + 1] = COLOR_OFFSET + (1 - color_val)
            # Mirror column
            col_val = result[i + 2] - COL_OFFSET
            if 0 <= col_val < COL_COUNT:
                result[i + 2] = COL_OFFSET + _COL_MIRROR[col_val]
            # Mirror direction
            dir_val = result[i + 4] - DIR_OFFSET
            if 0 <= dir_val < DIR_COUNT:
                result[i + 4] = DIR_OFFSET + _DIR_MIRROR[dir_val]
            # Mirror angle offset: negate (bin index mirrors around center=3)
            # Bins: 0=-45, 1=-30, 2=-15, 3=0, 4=15, 5=30, 6=45 → 6-idx
            angle_val = result[i + 5] - ANGLE_OFFSET_OFFSET
            if 0 <= angle_val < ANGLE_OFFSET_COUNT:
                result[i + 5] = ANGLE_OFFSET_OFFSET + (ANGLE_OFFSET_COUNT - 1 - angle_val)
            i += 6
        elif tok == BOMB and i + 2 < len(result):
            col_val = result[i + 1] - COL_OFFSET
            if 0 <= col_val < COL_COUNT:
                result[i + 1] = COL_OFFSET + _COL_MIRROR[col_val]
            i += 3
        elif tok == WALL and i + 6 < len(result):
            col_val = result[i + 1] - COL_OFFSET
            if 0 <= col_val < COL_COUNT:
                result[i + 1] = COL_OFFSET + _COL_MIRROR[col_val]
            i += 7
        elif tok == ARC_START and i + 5 < len(result):
            color_val = result[i + 1] - COLOR_OFFSET
            if 0 <= color_val < COLOR_COUNT:
                result[i + 1] = COLOR_OFFSET + (1 - color_val)
            col_val = result[i + 2] - COL_OFFSET
            if 0 <= col_val < COL_COUNT:
                result[i + 2] = COL_OFFSET + _COL_MIRROR[col_val]
            dir_val = result[i + 4] - DIR_OFFSET
            if 0 <= dir_val < DIR_COUNT:
                result[i + 4] = DIR_OFFSET + _DIR_MIRROR[dir_val]
            i += 6
        elif tok == ARC_END and i + 6 < len(result):
            color_val = result[i + 1] - COLOR_OFFSET
            if 0 <= color_val < COLOR_COUNT:
                result[i + 1] = COLOR_OFFSET + (1 - color_val)
            col_val = result[i + 2] - COL_OFFSET
            if 0 <= col_val < COL_COUNT:
                result[i + 2] = COL_OFFSET + _COL_MIRROR[col_val]
            dir_val = result[i + 4] - DIR_OFFSET
            if 0 <= dir_val < DIR_COUNT:
                result[i + 4] = DIR_OFFSET + _DIR_MIRROR[dir_val]
            i += 7
        elif tok == CHAIN and i + 9 < len(result):
            color_val = result[i + 1] - COLOR_OFFSET
            if 0 <= color_val < COLOR_COUNT:
                result[i + 1] = COLOR_OFFSET + (1 - color_val)
            col_val = result[i + 2] - COL_OFFSET
            if 0 <= col_val < COL_COUNT:
                result[i + 2] = COL_OFFSET + _COL_MIRROR[col_val]
            dir_val = result[i + 4] - DIR_OFFSET
            if 0 <= dir_val < DIR_COUNT:
                result[i + 4] = DIR_OFFSET + _DIR_MIRROR[dir_val]
            # Mirror tail_col
            tail_col = result[i + 5] - COL_OFFSET
            if 0 <= tail_col < COL_COUNT:
                result[i + 5] = COL_OFFSET + _COL_MIRROR[tail_col]
            # Token at i+9 is chain_tail_beat — no mirroring needed
            i += 10
        else:
            i += 1
    return result


class SequenceDataset(Dataset):
    """Dataset for Stage 2 note sequence generation training.

    Produces per-onset samples with a context window of mel frames
    centered on the onset, plus the token sequence for that onset.

    Each sample: {mel: [n_mels, context_frames], tokens: [max_token_len],
                  token_length: int, difficulty: int, genre: int}
    """

    def __init__(
        self,
        data_dir: Path | str,
        split: str = "train",
        context_frames: int = 128,
        max_token_len: int = 64,
        difficulties: list[str] | None = None,
        exclude_categories: list[str] | None = None,
        prev_context_k: int = 0,
        mirror_augment: bool = False,
    ) -> None:
        """Initialize SequenceDataset.

        Args:
            data_dir: Directory with preprocessed .pt files and splits.json.
            split: One of "train", "val", "test".
            context_frames: Number of mel frames for audio context around each onset.
            max_token_len: Maximum token sequence length (padded/truncated).
            difficulties: Which difficulties to include. Defaults to all available.
            exclude_categories: Mod categories to exclude (e.g. ["noodle",
                "mapping_extensions"]). Filters by mod_requirements.category
                embedded in each .pt file. None includes all categories.
            prev_context_k: Number of previous onset token sequences to include
                as inter-onset context (0 = disabled, 8 = recommended).
            mirror_augment: If True, 50% of samples are randomly mirrored
                (flip columns, swap colors, mirror directions). Only for training.
        """
        self.data_dir = Path(data_dir)
        self.context_frames = context_frames
        self.max_token_len = max_token_len
        self.prev_context_k = prev_context_k
        self.mirror_augment = mirror_augment
        self.target_difficulties = difficulties
        self.exclude_categories = set(exclude_categories) if exclude_categories else set()

        # Load split info
        splits_path = self.data_dir / "splits.json"
        if splits_path.exists():
            with open(splits_path) as f:
                splits = json.load(f)
            song_ids = set(splits.get(split, []))
        else:
            song_ids = None

        # Load blacklist if present (outlier filtering)
        blacklist_path = self.data_dir / "blacklist.json"
        blacklist: set[str] = set()
        if blacklist_path.exists():
            with open(blacklist_path) as f:
                blacklist = set(json.load(f).keys())

        # Load whitelist if present (curated subset — only train on these maps)
        whitelist_path = self.data_dir / "whitelist.json"
        whitelist: set[str] | None = None
        if whitelist_path.exists():
            with open(whitelist_path) as f:
                whitelist = set(json.load(f).keys())

        # Load frame index if available (avoids reading every .pt file at init)
        frame_index_path = self.data_dir / "frame_index.json"
        frame_index: dict | None = None
        if frame_index_path.exists():
            with open(frame_index_path) as f:
                frame_index = json.load(f)

        # Index all onset samples
        # (pt_path, diff_name, onset_idx, diff_id, genre_idx)
        self.samples: list[tuple[Path, str, int, int, int]] = []
        pt_files = sorted(self.data_dir.glob("*.pt"))
        for pt_path in pt_files:
            song_id = pt_path.stem
            if song_ids is not None and song_id not in song_ids:
                continue
            if song_id in blacklist:
                continue
            if whitelist is not None and song_id not in whitelist:
                continue

            if frame_index is not None and song_id in frame_index:
                entry = frame_index[song_id]
                cat = entry.get("category", "vanilla")
                if self.exclude_categories and cat in self.exclude_categories:
                    continue
                genre_idx = GENRE_MAP.get(entry.get("genre", "unknown"), 0)
                diff_meta = entry.get("difficulties", {})
                for diff_name, dmeta in diff_meta.items():
                    if self.target_difficulties and diff_name not in self.target_difficulties:
                        continue
                    diff_id = DIFFICULTY_MAP.get(diff_name, 3)
                    n_onsets = dmeta.get("n_onsets", 0)
                    for onset_idx in range(n_onsets):
                        self.samples.append((pt_path, diff_name, onset_idx, diff_id, genre_idx))
                continue  # skip slow path below

            data = torch.load(pt_path, weights_only=False)

            mod_reqs = data.get("mod_requirements", {})
            cat = mod_reqs.get("category", "vanilla")

            # Filter by mod category
            if self.exclude_categories and cat in self.exclude_categories:
                continue

            genre_idx = GENRE_MAP.get(mod_reqs.get("genre", "unknown"), 0)

            for diff_name, diff_data in data.get("difficulties", {}).items():
                if self.target_difficulties and diff_name not in self.target_difficulties:
                    continue
                diff_id = DIFFICULTY_MAP.get(diff_name, 3)
                n_onsets = len(diff_data.get("token_sequences", []))
                for onset_idx in range(n_onsets):
                    self.samples.append((pt_path, diff_name, onset_idx, diff_id, genre_idx))

        self._cache: OrderedDict[str, dict] = OrderedDict()
        self._cache_max = 100

    def _load(self, pt_path: Path) -> dict:
        key = str(pt_path)
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        data = torch.load(pt_path, weights_only=False)
        self._cache[key] = data
        if len(self._cache) > self._cache_max:
            self._cache.popitem(last=False)
        return data

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        pt_path, diff_name, onset_idx, diff_id, genre_idx = self.samples[idx]
        data = self._load(pt_path)
        mel = data["mel_spectrogram"]
        diff_data = data["difficulties"][diff_name]

        onset_frame = int(diff_data["onset_frames"][onset_idx].item())
        token_seq = list(diff_data["token_sequences"][onset_idx])

        # Mirror augmentation: 50% chance to flip left↔right, swap colors, mirror dirs
        do_mirror = self.mirror_augment and random.random() < 0.5
        if do_mirror:
            token_seq = _mirror_token_sequence(token_seq)

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

        # Structure features (backward compat: zeros if not in .pt file)
        structure = data.get("structure_features", None)
        if structure is not None:
            structure = _pad_structure(structure)
            structure_window = structure[:, start:end]
            if structure_window.shape[1] < self.context_frames:
                pad_size = self.context_frames - structure_window.shape[1]
                structure_window = torch.nn.functional.pad(structure_window, (0, pad_size))
        else:
            structure_window = torch.zeros(N_STRUCTURE_FEATURES, self.context_frames)

        # Pad/truncate token sequence
        if len(token_seq) > self.max_token_len:
            token_seq = token_seq[: self.max_token_len]
        token_length = len(token_seq)
        padded = token_seq + [0] * (self.max_token_len - len(token_seq))

        result = {
            "mel": mel_window,
            "tokens": torch.tensor(padded, dtype=torch.long).clamp(0, 182),
            "token_length": torch.tensor(token_length, dtype=torch.long),
            "difficulty": torch.tensor(diff_id, dtype=torch.long),
            "genre": torch.tensor(genre_idx, dtype=torch.long),
            "structure": structure_window,
        }

        # Previous onset context (Phase 3: inter-onset context)
        if self.prev_context_k > 0:
            all_seqs = diff_data["token_sequences"]
            prev_tokens = []
            for k in range(self.prev_context_k):
                prev_idx = onset_idx - (self.prev_context_k - k)
                if prev_idx >= 0:
                    seq = list(all_seqs[prev_idx])
                    if do_mirror:
                        seq = _mirror_token_sequence(seq)
                    if len(seq) > self.max_token_len:
                        seq = seq[: self.max_token_len]
                    seq = seq + [0] * (self.max_token_len - len(seq))
                else:
                    seq = [0] * self.max_token_len
                prev_tokens.append(seq)
            result["prev_tokens"] = torch.tensor(prev_tokens, dtype=torch.long)

            # Time gap from previous onset (seconds) for flow loss
            onset_frames = diff_data["onset_frames"]
            if onset_idx > 0:
                prev_frame = int(onset_frames[onset_idx - 1].item())
                # ~86 frames/sec at sr=44100, hop=512
                time_gap = (onset_frame - prev_frame) / 86.0
            else:
                time_gap = 999.0  # large gap = first onset
            result["time_gap"] = torch.tensor(time_gap, dtype=torch.float32)

        return result


class LightingDataset(Dataset):
    """Dataset for Stage 3 lighting generation training.

    Produces per-beat samples for beats that have lighting events.
    Each sample provides a mel context window, note tokens for the beat
    (for note conditioning), and lighting tokens as the target sequence.

    Each sample:
        {mel: [n_mels, context_frames], note_tokens: [max_note_len],
         light_tokens: [max_light_len], difficulty: int, genre: int}

    The .pt files must contain ``light_frames`` and ``light_token_sequences``
    entries (produced by the preprocessing pipeline with LightingTokenizer).
    """

    def __init__(
        self,
        data_dir: Path | str,
        split: str = "train",
        context_frames: int = 128,
        max_note_len: int = 64,
        max_light_len: int = 32,
        difficulties: list[str] | None = None,
        exclude_categories: list[str] | None = None,
    ) -> None:
        """Initialize LightingDataset.

        Args:
            data_dir: Directory with preprocessed .pt files and splits.json.
            split: One of "train", "val", "test".
            context_frames: Number of mel frames for audio context around each beat.
            max_note_len: Maximum note token sequence length (padded/truncated).
            max_light_len: Maximum lighting token sequence length (padded/truncated).
            difficulties: Which difficulties to include. Defaults to all available.
            exclude_categories: Mod categories to exclude (e.g. ["noodle",
                "mapping_extensions"]). Filters by mod_requirements.category
                embedded in each .pt file. None includes all categories.
        """
        self.data_dir = Path(data_dir)
        self.context_frames = context_frames
        self.max_note_len = max_note_len
        self.max_light_len = max_light_len
        self.target_difficulties = difficulties
        self.exclude_categories = set(exclude_categories) if exclude_categories else set()

        # Load split info
        splits_path = self.data_dir / "splits.json"
        if splits_path.exists():
            with open(splits_path) as f:
                splits = json.load(f)
            song_ids = set(splits.get(split, []))
        else:
            song_ids = None

        # Load blacklist if present (outlier filtering)
        blacklist_path = self.data_dir / "blacklist.json"
        blacklist: set[str] = set()
        if blacklist_path.exists():
            with open(blacklist_path) as f:
                blacklist = set(json.load(f).keys())

        # Load whitelist if present (curated subset — only train on these maps)
        whitelist_path = self.data_dir / "whitelist.json"
        whitelist: set[str] | None = None
        if whitelist_path.exists():
            with open(whitelist_path) as f:
                whitelist = set(json.load(f).keys())

        # Load frame index if available (avoids reading every .pt file at init)
        frame_index_path = self.data_dir / "frame_index.json"
        frame_index: dict | None = None
        if frame_index_path.exists():
            with open(frame_index_path) as f:
                frame_index = json.load(f)

        # Index all lighting-event samples: (pt_path, diff_name, light_idx, diff_id, genre_idx)
        self.samples: list[tuple[Path, str, int, int, int]] = []
        pt_files = sorted(self.data_dir.glob("*.pt"))
        for pt_path in pt_files:
            song_id = pt_path.stem
            if song_ids is not None and song_id not in song_ids:
                continue
            if song_id in blacklist:
                continue
            if whitelist is not None and song_id not in whitelist:
                continue

            if frame_index is not None and song_id in frame_index:
                entry = frame_index[song_id]
                cat = entry.get("category", "vanilla")
                if self.exclude_categories and cat in self.exclude_categories:
                    continue
                genre_idx = GENRE_MAP.get(entry.get("genre", "unknown"), 0)
                diff_meta = entry.get("difficulties", {})
                for diff_name, dmeta in diff_meta.items():
                    if self.target_difficulties and diff_name not in self.target_difficulties:
                        continue
                    n_light = dmeta.get("n_lights", 0)
                    if n_light == 0:
                        continue
                    diff_id = DIFFICULTY_MAP.get(diff_name, 3)
                    for light_idx in range(n_light):
                        self.samples.append((pt_path, diff_name, light_idx, diff_id, genre_idx))
                continue  # skip slow path below

            data = torch.load(pt_path, weights_only=False)

            mod_reqs = data.get("mod_requirements", {})
            cat = mod_reqs.get("category", "vanilla")

            # Filter by mod category
            if self.exclude_categories and cat in self.exclude_categories:
                continue

            genre_idx = GENRE_MAP.get(mod_reqs.get("genre", "unknown"), 0)

            for diff_name, diff_data in data.get("difficulties", {}).items():
                if self.target_difficulties and diff_name not in self.target_difficulties:
                    continue
                if "light_token_sequences" not in diff_data:
                    continue
                diff_id = DIFFICULTY_MAP.get(diff_name, 3)
                n_light = len(diff_data["light_token_sequences"])
                for light_idx in range(n_light):
                    self.samples.append((pt_path, diff_name, light_idx, diff_id, genre_idx))

        self._cache: OrderedDict[str, dict] = OrderedDict()
        self._cache_max = 100

    def _load(self, pt_path: Path) -> dict:
        key = str(pt_path)
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        data = torch.load(pt_path, weights_only=False)
        self._cache[key] = data
        if len(self._cache) > self._cache_max:
            self._cache.popitem(last=False)
        return data

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        pt_path, diff_name, light_idx, diff_id, genre_idx = self.samples[idx]
        data = self._load(pt_path)
        mel = data["mel_spectrogram"]
        diff_data = data["difficulties"][diff_name]

        light_frame = int(diff_data["light_frames"][light_idx].item())
        light_seq = diff_data["light_token_sequences"][light_idx]

        # Find note tokens for the nearest onset to this beat
        # (use the note token sequence closest in frame to light_frame)
        onset_frames = diff_data.get("onset_frames", torch.tensor([]))
        note_tokens: list[int] = []
        if len(onset_frames) > 0:
            dists = (onset_frames - light_frame).abs()
            nearest_idx = int(dists.argmin().item())
            note_tokens = diff_data["token_sequences"][nearest_idx]

        # Extract mel context window
        n_frames = mel.shape[1]
        half = self.context_frames // 2
        start = max(0, light_frame - half)
        end = start + self.context_frames
        if end > n_frames:
            end = n_frames
            start = max(0, end - self.context_frames)
        mel_window = mel[:, start:end]
        if mel_window.shape[1] < self.context_frames:
            pad_size = self.context_frames - mel_window.shape[1]
            mel_window = torch.nn.functional.pad(mel_window, (0, pad_size))

        # Pad/truncate note tokens
        if len(note_tokens) > self.max_note_len:
            note_tokens = note_tokens[: self.max_note_len]
        note_padded = note_tokens + [0] * (self.max_note_len - len(note_tokens))

        # Pad/truncate lighting tokens
        if len(light_seq) > self.max_light_len:
            light_seq = light_seq[: self.max_light_len]
        light_padded = light_seq + [0] * (self.max_light_len - len(light_seq))

        # Structure features (backward compat: zeros if not in .pt file)
        structure = data.get("structure_features", None)
        if structure is not None:
            structure = _pad_structure(structure)
            structure_window = structure[:, start:end]
            if structure_window.shape[1] < self.context_frames:
                pad_size = self.context_frames - structure_window.shape[1]
                structure_window = torch.nn.functional.pad(structure_window, (0, pad_size))
        else:
            structure_window = torch.zeros(N_STRUCTURE_FEATURES, self.context_frames)

        return {
            "mel": mel_window,
            "note_tokens": torch.tensor(note_padded, dtype=torch.long),
            "light_tokens": torch.tensor(light_padded, dtype=torch.long),
            "difficulty": torch.tensor(diff_id, dtype=torch.long),
            "genre": torch.tensor(genre_idx, dtype=torch.long),
            "structure": structure_window,
        }


def _worker_init_fn(worker_id: int) -> None:  # noqa: ARG001
    """Suppress noisy per-worker PyTorch log spam."""
    import logging

    logging.getLogger("torch.utils.flop_counter").setLevel(logging.ERROR)


# ---------------------------------------------------------------------------
# Song-level batching for onset planner training
# ---------------------------------------------------------------------------


class SongBatchDataset(Dataset):
    """Dataset that yields ALL onsets for one song+difficulty as a single sample.

    Used for training with the OnsetPlanner, which needs to see all onsets
    in a song simultaneously for bidirectional context.

    Each sample: {
        mel: [n_mels, T_full],              # full song mel spectrogram
        onset_frames: [N_onsets],            # frame positions of all onsets
        token_sequences: [N_onsets, max_len], # token sequence per onset
        token_lengths: [N_onsets],           # actual token lengths
        difficulty: int,
        genre: int,
        structure: [6, T_full],             # optional structure features
        n_onsets: int,                       # actual onset count
    }
    """

    def __init__(
        self,
        data_dir: Path | str,
        split: str = "train",
        max_token_len: int = 64,
        difficulties: list[str] | None = None,
        exclude_categories: list[str] | None = None,
        mirror_augment: bool = False,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.max_token_len = max_token_len
        self.mirror_augment = mirror_augment
        self.target_difficulties = difficulties
        self.exclude_categories = set(exclude_categories) if exclude_categories else set()

        # Load split info
        splits_path = self.data_dir / "splits.json"
        if splits_path.exists():
            with open(splits_path) as f:
                splits = json.load(f)
            song_ids = set(splits.get(split, []))
        else:
            song_ids = None

        # Load blacklist/whitelist
        blacklist: set[str] = set()
        blacklist_path = self.data_dir / "blacklist.json"
        if blacklist_path.exists():
            with open(blacklist_path) as f:
                blacklist = set(json.load(f).keys())

        whitelist: set[str] | None = None
        whitelist_path = self.data_dir / "whitelist.json"
        if whitelist_path.exists():
            with open(whitelist_path) as f:
                whitelist = set(json.load(f).keys())

        # Index: (pt_path, diff_name, diff_id, genre_idx)
        self.samples: list[tuple[Path, str, int, int]] = []
        pt_files = sorted(self.data_dir.glob("*.pt"))

        # Load frame index for fast init
        frame_index: dict | None = None
        frame_index_path = self.data_dir / "frame_index.json"
        if frame_index_path.exists():
            with open(frame_index_path) as f:
                frame_index = json.load(f)

        for pt_path in pt_files:
            song_id = pt_path.stem
            if song_ids is not None and song_id not in song_ids:
                continue
            if song_id in blacklist:
                continue
            if whitelist is not None and song_id not in whitelist:
                continue

            if frame_index is not None and song_id in frame_index:
                entry = frame_index[song_id]
                cat = entry.get("category", "vanilla")
                if self.exclude_categories and cat in self.exclude_categories:
                    continue
                genre_idx = GENRE_MAP.get(entry.get("genre", "unknown"), 0)
                diff_meta = entry.get("difficulties", {})
                for diff_name in diff_meta:
                    if self.target_difficulties and diff_name not in self.target_difficulties:
                        continue
                    n_onsets = diff_meta[diff_name].get("n_onsets", 0)
                    if n_onsets == 0:
                        continue
                    diff_id = DIFFICULTY_MAP.get(diff_name, 3)
                    self.samples.append((pt_path, diff_name, diff_id, genre_idx))
            else:
                data = torch.load(pt_path, weights_only=False)
                mod_reqs = data.get("mod_requirements", {})
                cat = mod_reqs.get("category", "vanilla")
                if self.exclude_categories and cat in self.exclude_categories:
                    continue
                genre_idx = GENRE_MAP.get(mod_reqs.get("genre", "unknown"), 0)
                for diff_name, diff_data in data.get("difficulties", {}).items():
                    if self.target_difficulties and diff_name not in self.target_difficulties:
                        continue
                    n_onsets = len(diff_data.get("token_sequences", []))
                    if n_onsets == 0:
                        continue
                    diff_id = DIFFICULTY_MAP.get(diff_name, 3)
                    self.samples.append((pt_path, diff_name, diff_id, genre_idx))

        self._cache: OrderedDict[str, dict] = OrderedDict()
        self._cache_max = 50

    def _load(self, pt_path: Path) -> dict:
        key = str(pt_path)
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        data = torch.load(pt_path, weights_only=False)
        self._cache[key] = data
        if len(self._cache) > self._cache_max:
            self._cache.popitem(last=False)
        return data

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        pt_path, diff_name, diff_id, genre_idx = self.samples[idx]
        data = self._load(pt_path)
        mel = data["mel_spectrogram"]
        diff_data = data["difficulties"][diff_name]

        onset_frames = diff_data["onset_frames"]
        token_sequences = diff_data["token_sequences"]
        n_onsets = len(token_sequences)

        # Pad/truncate all token sequences to max_token_len
        padded_tokens = []
        token_lengths = []
        do_mirror = self.mirror_augment and random.random() < 0.5
        for seq in token_sequences:
            seq = list(seq)
            if do_mirror:
                seq = _mirror_token_sequence(seq)
            token_lengths.append(min(len(seq), self.max_token_len))
            if len(seq) >= self.max_token_len:
                seq = seq[: self.max_token_len]
            else:
                seq = seq + [0] * (self.max_token_len - len(seq))
            padded_tokens.append(seq)

        # Structure features
        structure = data.get("structure_features", torch.zeros(N_STRUCTURE_FEATURES, mel.shape[1]))
        structure = _pad_structure(structure)

        result = {
            "mel": mel,
            "onset_frames": onset_frames[:n_onsets].long(),
            "token_sequences": torch.tensor(padded_tokens, dtype=torch.long).clamp(0, 182),
            "token_lengths": torch.tensor(token_lengths, dtype=torch.long),
            "difficulty": torch.tensor(diff_id, dtype=torch.long),
            "genre": torch.tensor(genre_idx, dtype=torch.long),
            "structure": structure,
            "n_onsets": torch.tensor(n_onsets, dtype=torch.long),
        }
        return result


def song_batch_collate(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor | list]:
    """Collate function for SongBatchDataset.

    Since each sample is a full song with variable onset counts,
    this collates by padding to the maximum onset count in the batch.
    Typically batch_size=1 for song-level training.
    """
    if len(batch) == 1:
        # Common case: batch_size=1 for song-level training
        sample = batch[0]
        return {
            "mel": sample["mel"].unsqueeze(0),
            "onset_frames": sample["onset_frames"].unsqueeze(0),
            "token_sequences": sample["token_sequences"].unsqueeze(0),
            "token_lengths": sample["token_lengths"].unsqueeze(0),
            "difficulty": sample["difficulty"].unsqueeze(0),
            "genre": sample["genre"].unsqueeze(0),
            "structure": sample["structure"].unsqueeze(0),
            "n_onsets": sample["n_onsets"].unsqueeze(0),
        }

    # Multi-song batching: pad to max dimensions
    max_onsets = max(s["n_onsets"].item() for s in batch)
    max_frames = max(s["mel"].shape[1] for s in batch)
    n_mels = batch[0]["mel"].shape[0]
    max_token_len = batch[0]["token_sequences"].shape[1]
    n_structure = batch[0]["structure"].shape[0]
    b = len(batch)

    mel_batch = torch.zeros(b, n_mels, max_frames)
    onset_batch = torch.zeros(b, max_onsets, dtype=torch.long)
    token_batch = torch.zeros(b, max_onsets, max_token_len, dtype=torch.long)
    length_batch = torch.zeros(b, max_onsets, dtype=torch.long)
    diff_batch = torch.zeros(b, dtype=torch.long)
    genre_batch = torch.zeros(b, dtype=torch.long)
    structure_batch = torch.zeros(b, n_structure, max_frames)
    n_onsets_batch = torch.zeros(b, dtype=torch.long)

    for i, s in enumerate(batch):
        n = s["n_onsets"].item()
        t = s["mel"].shape[1]
        mel_batch[i, :, :t] = s["mel"]
        onset_batch[i, :n] = s["onset_frames"][:n]
        token_batch[i, :n] = s["token_sequences"][:n]
        length_batch[i, :n] = s["token_lengths"][:n]
        diff_batch[i] = s["difficulty"]
        genre_batch[i] = s["genre"]
        structure_batch[i, :, :t] = s["structure"]
        n_onsets_batch[i] = s["n_onsets"]

    return {
        "mel": mel_batch,
        "onset_frames": onset_batch,
        "token_sequences": token_batch,
        "token_lengths": length_batch,
        "difficulty": diff_batch,
        "genre": genre_batch,
        "structure": structure_batch,
        "n_onsets": n_onsets_batch,
    }


def create_dataloader(
    dataset: Dataset,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = True,
    max_samples_per_epoch: int | None = None,
) -> DataLoader:
    """Create a DataLoader with sensible defaults.

    Args:
        dataset: PyTorch dataset to wrap.
        batch_size: Batch size.
        shuffle: Whether to shuffle.
        num_workers: Number of data loading workers.
        pin_memory: Pin memory for GPU transfer.
        max_samples_per_epoch: If set, cap the number of samples seen per epoch.
            Uses RandomSampler with replacement=False so each epoch sees a
            different random subset. Useful for very large datasets (17M+ samples)
            where a full epoch would take days.

    Returns:
        Configured DataLoader.
    """
    sampler = None
    effective_shuffle = shuffle
    if max_samples_per_epoch is not None and max_samples_per_epoch < len(dataset):
        sampler = RandomSampler(dataset, replacement=False, num_samples=max_samples_per_epoch)
        effective_shuffle = False  # sampler handles shuffling

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=effective_shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        # Keep workers alive between epochs so their file caches persist.
        # prefetch_factor=2 keeps data prefetched ahead of the GPU.
        persistent_workers=(num_workers > 0),
        prefetch_factor=2 if num_workers > 0 else None,
        worker_init_fn=_worker_init_fn if num_workers > 0 else None,
    )
