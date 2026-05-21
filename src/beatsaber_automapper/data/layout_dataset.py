"""V7-4: Dataset for Stage 2 LayoutModel training.

For each confirmed note onset (beat, hand) pair derived from Expert swing_tokens,
yields:
  - local_mert:      mix MERT features at the onset beat  [768]
  - song_emb:        mean mix MERT over full song          [768]
  - section_emb:     mean mix MERT over current section    [768]
  - saber_state:     12-dim physical state at this onset   [12]
  - layout_tokens:   spatial token sequence [KIND X Y DIR FIELD_D] padded to max_len
  - retrieval_feats: phrase fingerprint for the window containing this onset [768]
  - difficulty, genre, mapper_id

The retrieval_feats vector is the phrase fingerprint; at inference the PhraseIndex
does the actual cosine lookup. During training it's used to condition the model on
which musical phrase it's in (so the model learns phrase-consistent mapping).
"""

from __future__ import annotations

import json
import logging
from collections import OrderedDict
from pathlib import Path

import torch
from torch.utils.data import Dataset

from beatsaber_automapper.data.beat_grid import extract_beat_labels, BEAT_SUBDIV
from beatsaber_automapper.data.saber_state import compute_saber_states
from beatsaber_automapper.data.swing_tokenizer import (
    _DT_BINS,
    ANGLE_BASE, ANGLE_COUNT,
    ARC_HEAD, ARC_TAIL,
    BOMB,
    BOS,
    CHAIN_HEAD, CHAIN_TAIL,
    DIR_BASE,
    DT_BASE, DT_COUNT,
    EOS,
    HAND_LEFT, HAND_NONE, HAND_RIGHT,
    KIND_BASE, KIND_COUNT,
    MU_BASE, MU_COUNT,
    NOTE,
    PAD,
    SLICE_BASE, SLICE_COUNT,
    SQUISH_BASE, SQUISH_COUNT,
    X_BASE, X_COUNT,
    Y_BASE, Y_COUNT,
    SwingEventTokenizer,
)
from beatsaber_automapper.data.tokenizer import GENRE_MAP
from beatsaber_automapper.data.dataset import DIFFICULTY_MAP

logger = logging.getLogger(__name__)

# Layout token vocabulary: spatial-only subset (no HAND/Δt)
# KIND(6) + X(4) + Y(3) + DIR(9) + ANGLE(7) + MU(9) + SLICE(31) + SQUISH(11) + BOS/EOS/PAD = ~83
LAYOUT_PAD = 0
LAYOUT_BOS = 1
LAYOUT_EOS = 2

# Remap original token IDs to compact layout vocab
# We keep the same integer IDs from swing_tokenizer but skip HAND (3,4,5) and DT (6-37).
# The layout model uses: PAD=0, BOS=1, EOS=2, KIND_BASE=38..43, X_BASE=44..47,
# Y_BASE=48..50, DIR=51..59, ANGLE=60..66, MU=67..75, SLICE=76..106, SQUISH=107..117
# These IDs are already compact and don't overlap — just use them directly.
LAYOUT_VOCAB_SIZE = 118   # same vocab, subset actually used

MAX_LAYOUT_LEN = 32   # max tokens per note event (very generous; NOTE = 5 tokens)


def _encode_layout_event(
    kind: int, x: int, y: int, direction: int, field_d: int,
) -> list[int]:
    """Encode a single swing event as spatial tokens (no HAND/Δt).

    Returns a list of 3–5 token IDs depending on event kind.
    """
    tokens = [kind, X_BASE + x, Y_BASE + y]
    if kind == BOMB:
        pass  # 3 tokens: KIND X Y
    elif kind == CHAIN_TAIL:
        tokens.append(SQUISH_BASE + min(field_d, SQUISH_COUNT - 1))  # +SQUISH = 4
    else:
        tokens.append(DIR_BASE + direction)
        if kind == NOTE:
            tokens.append(ANGLE_BASE + min(field_d, ANGLE_COUNT - 1))
        elif kind in (ARC_HEAD, ARC_TAIL):
            tokens.append(MU_BASE + min(field_d, MU_COUNT - 1))
        else:  # CHAIN_HEAD
            tokens.append(SLICE_BASE + min(field_d, SLICE_COUNT - 1))
    return tokens


def _parse_events_from_tokens(swing_tokens: list[int]) -> list[dict]:
    """Parse swing_tokens into a list of event dicts with beat positions."""
    events = []
    i, n = 0, len(swing_tokens)
    current_beat = 0.0

    while i < n:
        tok = swing_tokens[i]
        if tok in (PAD, BOS):
            i += 1; continue
        if tok == EOS:
            break
        if tok not in (HAND_LEFT, HAND_RIGHT, HAND_NONE):
            i += 1; continue

        hand = tok
        if i + 1 >= n:
            break
        dt_tok = swing_tokens[i + 1]
        if not (DT_BASE <= dt_tok < DT_BASE + DT_COUNT):
            i += 1; continue

        dt = _DT_BINS[dt_tok - DT_BASE]
        current_beat += dt

        if i + 2 >= n:
            break
        kind_tok = swing_tokens[i + 2]
        if not (KIND_BASE <= kind_tok < KIND_BASE + KIND_COUNT):
            i += 1; continue
        kind = kind_tok

        # Extract X, Y, DIR, FIELD_D based on kind
        x = y = direction = field_d = 0
        if i + 4 < n:
            x = max(0, min(swing_tokens[i + 3] - X_BASE, 3))
            y = max(0, min(swing_tokens[i + 4] - Y_BASE, 2))

        if kind == BOMB:
            step = 5
        elif kind == CHAIN_TAIL:
            if i + 5 < n:
                field_d = max(0, min(swing_tokens[i + 5] - SQUISH_BASE, SQUISH_COUNT - 1))
            step = 6
        else:
            if i + 5 < n:
                direction = max(0, min(swing_tokens[i + 5] - DIR_BASE, 8))
            if i + 6 < n:
                fd_tok = swing_tokens[i + 6]
                if kind == NOTE:
                    field_d = max(0, min(fd_tok - ANGLE_BASE, ANGLE_COUNT - 1))
                elif kind in (ARC_HEAD, ARC_TAIL):
                    field_d = max(0, min(fd_tok - MU_BASE, MU_COUNT - 1))
                else:
                    field_d = max(0, min(fd_tok - SLICE_BASE, SLICE_COUNT - 1))
            step = 7

        events.append({
            "beat": current_beat, "hand": hand, "kind": kind,
            "x": x, "y": y, "direction": direction, "field_d": field_d,
        })
        i += step

    return events


class LayoutDataset(Dataset):
    """Per-onset dataset for Stage 2 Layout Model training.

    Each sample represents one note (left or right hand) at a specific beat position.
    The model is trained to predict the spatial token sequence given MERT context.

    Args:
        data_dir:           Directory with V7-preprocessed .pt files.
        split:              "train" or "val".
        difficulties:       Difficulties to include (None = all).
        exclude_categories: Mod categories to skip.
        max_len:            Max layout token sequence length.
    """

    def __init__(
        self,
        data_dir: Path | str,
        split: str = "train",
        difficulties: list[str] | None = None,
        exclude_categories: list[str] | None = None,
        max_len: int = MAX_LAYOUT_LEN,
    ) -> None:
        self.data_dir     = Path(data_dir)
        self.max_len      = max_len
        self.target_diffs = set(difficulties) if difficulties else None
        self.exclude_cats = set(exclude_categories) if exclude_categories else set()

        splits_path = self.data_dir / "splits.json"
        song_ids: set[str] | None = None
        if splits_path.exists():
            with open(splits_path) as f:
                song_ids = set(json.load(f).get(split, []))

        blacklist: set[str] = set()
        if (bp := self.data_dir / "blacklist.json").exists():
            with open(bp) as f:
                blacklist = set(json.load(f).keys())

        # Index: (pt_path, diff_name, event_idx, diff_id, genre_idx)
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

            if "mix_beat_features" not in meta:
                continue  # not yet V7-preprocessed

            mod_reqs = meta.get("mod_requirements", {})
            if self.exclude_cats and mod_reqs.get("category") in self.exclude_cats:
                continue

            genre_idx = GENRE_MAP.get(mod_reqs.get("genre", "unknown"), 0)

            for diff_name, diff_data in meta.get("difficulties", {}).items():
                if self.target_diffs and diff_name not in self.target_diffs:
                    continue
                tokens = diff_data.get("swing_tokens", [])
                if not tokens:
                    continue

                diff_id  = DIFFICULTY_MAP.get(diff_name, 3)
                n_events = sum(1 for t in tokens if t in (HAND_LEFT, HAND_RIGHT, HAND_NONE))
                for evt_idx in range(n_events):
                    self.samples.append((pt_path, diff_name, evt_idx, diff_id, genre_idx))

        logger.info("LayoutDataset[%s]: %d note samples", split, len(self.samples))

        self._cache: OrderedDict[str, dict] = OrderedDict()
        self._cache_max = 32
        # Per-(song, difficulty) cache of decoded events + saber states.
        # Avoids the O(n²) re-decode that dominated dataset throughput when
        # the cache was per-file only.
        self._evt_cache: OrderedDict[tuple[str, str], tuple[list, torch.Tensor]] = OrderedDict()
        self._evt_cache_max = 64

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

    def _events_and_states(
        self, pt_path: Path, diff_name: str, swing_tokens: list[int],
    ) -> tuple[list, torch.Tensor]:
        """Return (decoded events, saber-state-before-each-event tensor).

        Cached per (song, difficulty) — the previous per-sample recompute was
        O(n²) over the song's event count.
        """
        key = (str(pt_path), diff_name)
        if key in self._evt_cache:
            self._evt_cache.move_to_end(key)
            return self._evt_cache[key]

        tok = SwingEventTokenizer()
        all_events = tok.decode_events(swing_tokens)
        # states[i] is the saber state BEFORE event i — exactly the conditioning
        # we want when generating the spatial tokens for event i.
        states = compute_saber_states(all_events)
        self._evt_cache[key] = (all_events, states)
        if len(self._evt_cache) > self._evt_cache_max:
            self._evt_cache.popitem(last=False)
        return all_events, states

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        pt_path, diff_name, evt_idx, diff_id, genre_idx = self.samples[idx]
        data = self._load(pt_path)

        mix_beat = data["mix_beat_features"].float()       # [N_slots, 768]
        fp_tensor = data["phrase_fingerprints"].float()    # [N_phrases, 768]
        pb = data["phrase_boundaries"]                     # list of (start, end)
        bpm = float(data.get("bpm", 120.0))
        n_slots = mix_beat.shape[0]

        swing_tokens = data["difficulties"][diff_name]["swing_tokens"]
        events = _parse_events_from_tokens(swing_tokens)

        if evt_idx >= len(events):
            # Safety: return a zero sample if index is out of range
            return self._zero_sample(diff_id, genre_idx)

        evt = events[evt_idx]

        # Decoded events + saber states (cached per song/diff)
        all_events, all_states = self._events_and_states(pt_path, diff_name, swing_tokens)

        # Beat → slot index
        beat_slot = min(int(round(evt["beat"] * BEAT_SUBDIV)), n_slots - 1)

        # Local MERT at this beat
        local_mert = mix_beat[beat_slot]                   # [768]

        # Song-level embedding
        song_emb = mix_beat.mean(0)                        # [768]

        # Section embedding: section that contains beat_slot
        # Use the existing section detection from structure_features if available,
        # otherwise fall back to a ±32-slot window around the beat
        struct = data.get("structure_features")
        if struct is not None and struct.shape[0] >= 7:
            # Structure channel 6 = normalised section_id
            cf = min(beat_slot, struct.shape[1] - 1)
            sec_norm = float(struct[6, cf].item())
            sec_id   = min(int(round(sec_norm * 5.0)), 5)
            # Find all slots with the same section_id and average their mix features
            sec_ids_all = (struct[6] * 5.0).round().long().clamp(0, 5)
            # Pool mix_beat where section matches — resample struct to beat grid length
            T_struct = sec_ids_all.shape[0]
            if T_struct != n_slots:
                # Nearest-neighbour upsample/downsample
                ratio = T_struct / max(n_slots, 1)
                sec_at_beat = torch.tensor([
                    int(sec_ids_all[min(int(s * ratio), T_struct - 1)].item())
                    for s in range(n_slots)
                ])
            else:
                sec_at_beat = sec_ids_all
            sec_mask = (sec_at_beat == sec_id)
            if sec_mask.any():
                section_emb = mix_beat[sec_mask].mean(0)
            else:
                section_emb = local_mert
        else:
            # Fallback: ±64-beat window
            half = 64 * BEAT_SUBDIV
            s_start = max(0, beat_slot - half)
            s_end   = min(n_slots, beat_slot + half)
            section_emb = mix_beat[s_start:s_end].mean(0)

        # Saber state BEFORE the current event (= state after all prior events).
        # compute_saber_states emits state[i] = state BEFORE event i, so we
        # want all_states[evt_idx] directly — no off-by-one slicing needed.
        if evt_idx < all_states.shape[0]:
            saber = all_states[evt_idx]
        else:
            saber = torch.zeros(12)

        # Layout tokens (spatial only, no HAND/Δt)
        layout_toks = _encode_layout_event(
            evt["kind"], evt["x"], evt["y"], evt["direction"], evt["field_d"],
        )
        # Wrap with BOS/EOS
        layout_toks = [LAYOUT_BOS] + layout_toks + [LAYOUT_EOS]
        if len(layout_toks) > self.max_len:
            layout_toks = layout_toks[:self.max_len]
        padded = layout_toks + [LAYOUT_PAD] * (self.max_len - len(layout_toks))

        # Phrase fingerprint: which phrase does this beat belong to?
        phrase_feat = torch.zeros(768)
        for pi, (s, e) in enumerate(pb):
            if s <= beat_slot < e:
                phrase_feat = fp_tensor[pi]
                break

        return {
            "local_mert":    local_mert,                                    # [768]
            "song_emb":      song_emb,                                      # [768]
            "section_emb":   section_emb,                                   # [768]
            "saber_state":   saber.float(),                                  # [12]
            "phrase_feat":   phrase_feat,                                   # [768]
            "layout_tokens": torch.tensor(padded, dtype=torch.long),        # [max_len]
            "token_length":  torch.tensor(len(layout_toks), dtype=torch.long),
            "difficulty":    torch.tensor(diff_id,   dtype=torch.long),
            "genre":         torch.tensor(genre_idx, dtype=torch.long),
        }

    def _zero_sample(self, diff_id: int, genre_idx: int) -> dict[str, torch.Tensor]:
        return {
            "local_mert":    torch.zeros(768),
            "song_emb":      torch.zeros(768),
            "section_emb":   torch.zeros(768),
            "saber_state":   torch.zeros(12),
            "phrase_feat":   torch.zeros(768),
            "layout_tokens": torch.zeros(self.max_len, dtype=torch.long),
            "token_length":  torch.tensor(0, dtype=torch.long),
            "difficulty":    torch.tensor(diff_id,   dtype=torch.long),
            "genre":         torch.tensor(genre_idx, dtype=torch.long),
        }
