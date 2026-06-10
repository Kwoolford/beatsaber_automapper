"""V7-3: Dataset for Stage 1 BeatClassifier training.

Yields sliding windows of beat-aligned drum + mix MERT features paired with
binary left/right note-presence labels. Skips .pt files that have not yet
been preprocessed with V7 features (run scripts/preprocess_v7.py first).

Each sample:
    drum_features [window_size, 768]  drum MERT (float32)
    mix_features  [window_size, 768]  mix MERT  (float32)
    left_labels   [window_size]       binary int64
    right_labels  [window_size]       binary int64
    slot_offset   scalar int64        absolute first-slot index of the window
                                       (for phase-in-bar embedding)
    difficulty    scalar int64
    genre         scalar int64
"""

from __future__ import annotations

import json
import logging
from collections import OrderedDict
from pathlib import Path

import torch
from torch.utils.data import Dataset

from beatsaber_automapper.data.beat_grid import extract_beat_labels, BEAT_SUBDIV
from beatsaber_automapper.data.instrument_features import INSTR_FEATURE_DIM
from beatsaber_automapper.data.tokenizer import GENRE_MAP
from beatsaber_automapper.data.dataset import DIFFICULTY_MAP

logger = logging.getLogger(__name__)


class BeatDataset(Dataset):
    """Sliding-window dataset over beat-grid MERT features + binary labels.

    Args:
        data_dir:        Directory containing .pt files with V7 features.
        split:           "train" or "val".
        window_size:     Number of beat slots per sample (default 128 = 32 beats).
        hop:             Stride between windows (default 64).
        difficulties:    Which difficulties to include (None = all).
        exclude_categories: Mod categories to skip.
        min_note_density: Minimum fraction of positive slots in a window
                          (sample-level filter; per-slot weighting handles the rest).
    """

    def __init__(
        self,
        data_dir: Path | str,
        split: str = "train",
        window_size: int = 128,
        hop: int = 64,
        difficulties: list[str] | None = None,
        exclude_categories: list[str] | None = None,
        min_note_density: float = 0.02,
        require_instr: bool = False,
    ) -> None:
        self.data_dir       = Path(data_dir)
        self.window_size    = window_size
        self.hop            = hop
        self.target_diffs   = set(difficulties) if difficulties else None
        self.exclude_cats   = set(exclude_categories) if exclude_categories else set()
        self.min_density    = min_note_density
        # When True, only index songs that already have cached per-instrument
        # layering features (skips songs not yet covered by the transcription pass).
        self.require_instr  = require_instr

        splits_path = self.data_dir / "splits.json"
        song_ids: set[str] | None = None
        if splits_path.exists():
            with open(splits_path) as f:
                song_ids = set(json.load(f).get(split, []))

        blacklist: set[str] = set()
        if (bp := self.data_dir / "blacklist.json").exists():
            with open(bp) as f:
                blacklist = set(json.load(f).keys())

        # Index: (pt_path, diff_name, start_slot, diff_id, genre_idx)
        self.samples: list[tuple[Path, str, int, int, int]] = []

        for pt_path in sorted(self.data_dir.glob("*.pt")):
            song_id = pt_path.stem
            if song_ids is not None and song_id not in song_ids:
                continue
            if song_id in blacklist:
                continue

            try:
                meta = torch.load(pt_path, weights_only=False, mmap=True)
            except Exception:
                continue

            if "drum_beat_features" not in meta or "mix_beat_features" not in meta:
                continue  # not yet preprocessed by V7
            if self.require_instr and "instr_beat_features" not in meta:
                continue  # not yet covered by the per-instrument transcription pass

            mod_reqs = meta.get("mod_requirements", {})
            if self.exclude_cats and mod_reqs.get("category") in self.exclude_cats:
                continue

            genre_idx = GENRE_MAP.get(mod_reqs.get("genre", "unknown"), 0)
            n_slots   = meta["drum_beat_features"].shape[0]

            for diff_name, diff_data in meta.get("difficulties", {}).items():
                if self.target_diffs and diff_name not in self.target_diffs:
                    continue
                if not diff_data.get("swing_tokens"):
                    continue
                diff_id = DIFFICULTY_MAP.get(diff_name, 3)

                # Slide windows
                start = 0
                while start + window_size <= n_slots:
                    self.samples.append((pt_path, diff_name, start, diff_id, genre_idx))
                    start += hop
                # Include tail window
                if n_slots > window_size and (n_slots - window_size) % hop != 0:
                    self.samples.append(
                        (pt_path, diff_name, n_slots - window_size, diff_id, genre_idx)
                    )

        logger.info("BeatDataset[%s]: %d windows from %d songs",
                    split, len(self.samples), len({s[0] for s in self.samples}))

        # Cache: pt_path → dict of tensors
        self._cache: OrderedDict[str, dict] = OrderedDict()
        self._cache_max = 64
        # Label cache: (pt_path, diff_name) → (left_all, right_all) full-song tensors
        self._label_cache: OrderedDict[tuple[str, str], tuple[torch.Tensor, torch.Tensor]] = OrderedDict()
        self._label_cache_max = 128

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

    def _labels_for(
        self, pt_path: Path, diff_name: str, data: dict,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        key = (str(pt_path), diff_name)
        if key in self._label_cache:
            self._label_cache.move_to_end(key)
            return self._label_cache[key]

        bpm  = float(data.get("bpm", 120.0))
        n_slots_full = data["drum_beat_features"].shape[0]
        swing_tokens = data["difficulties"][diff_name]["swing_tokens"]
        left_all, right_all, _, _ = extract_beat_labels(
            swing_tokens, bpm, n_slots_full,
        )
        left_t  = torch.from_numpy(left_all).long()
        right_t = torch.from_numpy(right_all).long()
        self._label_cache[key] = (left_t, right_t)
        if len(self._label_cache) > self._label_cache_max:
            self._label_cache.popitem(last=False)
        return left_t, right_t

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        pt_path, diff_name, start, diff_id, genre_idx = self.samples[idx]
        data = self._load(pt_path)

        end = start + self.window_size

        drum_window = data["drum_beat_features"][start:end].float()   # [W, 768]
        mix_window  = data["mix_beat_features"][start:end].float()    # [W, 768]

        # Pad to window_size if needed (tail window)
        if drum_window.shape[0] < self.window_size:
            pad = self.window_size - drum_window.shape[0]
            drum_window = torch.nn.functional.pad(drum_window, (0, 0, 0, pad))
            mix_window  = torch.nn.functional.pad(mix_window,  (0, 0, 0, pad))

        left_all, right_all = self._labels_for(pt_path, diff_name, data)
        left_w  = left_all[start:end]
        right_w = right_all[start:end]
        if left_w.shape[0] < self.window_size:
            pad = self.window_size - left_w.shape[0]
            left_w  = torch.nn.functional.pad(left_w,  (0, pad))
            right_w = torch.nn.functional.pad(right_w, (0, pad))

        # ---- Structure features: pool [8, N_frames] → [N_slots, 8] ----
        # adaptive_avg_pool1d handles variable frame-rate → beat-grid mapping
        # without needing BPM arithmetic.  All .pt files have structure_features.
        sf_raw = data.get("structure_features")   # [8, N_frames] float32
        if sf_raw is not None and sf_raw.shape[0] > 0:
            n_slots_full = data["drum_beat_features"].shape[0]
            sf_grid = torch.nn.functional.adaptive_avg_pool1d(
                sf_raw.unsqueeze(0).float(), n_slots_full,
            ).squeeze(0).T                                    # [N_slots, 8]
            struct_w = sf_grid[start:end]
            if struct_w.shape[0] < self.window_size:
                pad = self.window_size - struct_w.shape[0]
                struct_w = torch.nn.functional.pad(struct_w, (0, 0, 0, pad))
        else:
            struct_w = torch.zeros(self.window_size, 8)

        # ---- Per-instrument layering features [N_slots, INSTR_FEATURE_DIM] ----
        # Already on the same 1/subdiv-note grid as drum/mix (written by the
        # transcription preprocessing pass). Zeros when a song isn't covered yet.
        instr_raw = data.get("instr_beat_features")
        if instr_raw is not None:
            instr_w = instr_raw[start:end].float()
            if instr_w.shape[0] < self.window_size:
                pad = self.window_size - instr_w.shape[0]
                instr_w = torch.nn.functional.pad(instr_w, (0, 0, 0, pad))
        else:
            instr_w = torch.zeros(self.window_size, INSTR_FEATURE_DIM)

        return {
            "drum_features":   drum_window,                               # [W, 768]
            "mix_features":    mix_window,                                # [W, 768]
            "struct_features": struct_w,                                  # [W, 8]
            "instr_features":  instr_w,                                   # [W, INSTR_FEATURE_DIM]
            "left_labels":     left_w,                                    # [W]
            "right_labels":    right_w,                                   # [W]
            "slot_offset":     torch.tensor(start,     dtype=torch.long),
            "difficulty":      torch.tensor(diff_id,   dtype=torch.long),
            "genre":           torch.tensor(genre_idx, dtype=torch.long),
        }
