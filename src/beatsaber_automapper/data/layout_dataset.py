"""V7-5b: Per-phrase Layout dataset.

Each sample is one phrase (16-beat / 64-slot window) of one (song, difficulty).
The decoder will emit the spatial tokens for ALL notes in the phrase as a single
autoregressive sequence; per-token metadata (slot, hand, role) lets the model
know which onset each token belongs to. The hand-engineered 12-dim saber state
is gone — the decoder learns position/direction/parity from its own prior-token
attention within the phrase.

Each sample:
    phrase_mert       [P_slots, 768]      mix MERT for the phrase, slot-pooled
    phrase_mask       [P_slots]            1 = real slot, 0 = padding
    layout_tokens     [S]                  [BOS, ...event tokens..., EOS, PAD...]
    token_slot        [S]                  per-token slot index (0..P-1), PAD value for BOS/EOS/PAD
    token_hand        [S]                  per-token hand (LEFT=0, RIGHT=1), 2 for special
    token_role        [S]                  per-token role: KIND=0, X=1, Y=2, DIR=3, FIELD_D=4, special=5
    target            [S]                  next-token target (shifted by 1, -100 on PAD)
    n_notes           int                  number of note events in phrase (audit aid)
    difficulty        int
    genre             int
    song_id           str (for debugging only)
"""

from __future__ import annotations

import json
import logging
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import Dataset

from beatsaber_automapper.data.beat_grid import BEAT_SUBDIV
from beatsaber_automapper.data.dataset import DIFFICULTY_MAP
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
)
from beatsaber_automapper.data.tokenizer import GENRE_MAP

logger = logging.getLogger(__name__)

# Layout vocab — re-uses the swing-event vocab integer IDs. The HAND and Δt
# tokens never appear in the layout sequence (timing comes from per-token slot
# metadata, hand from per-token hand metadata).
LAYOUT_PAD = 0
LAYOUT_BOS = 1
LAYOUT_EOS = 2
LAYOUT_VOCAB_SIZE = 118  # full swing vocab — only the spatial subset is reachable

# Per-token role labels (small embedding alongside the token embedding).
ROLE_KIND    = 0
ROLE_X       = 1
ROLE_Y       = 2
ROLE_DIR     = 3
ROLE_FIELD_D = 4
ROLE_SPECIAL = 5  # BOS / EOS / PAD
N_ROLES      = 6

# Hand sentinel for special tokens
HAND_LEFT_IDX  = 0
HAND_RIGHT_IDX = 1
HAND_SPECIAL_IDX = 2
N_HANDS_EMB = 3

# Phrase slot embedding size default. Observed phrases are 64 slots; allow some
# headroom. The "special slot" sentinel index for BOS/EOS/PAD is always equal
# to whatever max_phrase_slots is configured on the dataset/model instance —
# they must use the same value (the train script wires both consistently).
MAX_PHRASE_SLOTS = 96
SPECIAL_SLOT_IDX = MAX_PHRASE_SLOTS  # default-value alias kept for tests/imports

# Max layout-token sequence length per phrase. Worst case Expert+ phrase:
# ~50 events × 5 tokens = 250 + BOS/EOS = 252. Round up.
DEFAULT_MAX_LAYOUT_LEN = 384

# Loss ignore index (matches PyTorch CE default)
IGNORE_INDEX = -100


@dataclass
class _Event:
    beat:      float
    slot:      int
    hand:      int   # HAND_LEFT or HAND_RIGHT or HAND_NONE
    kind:      int
    x:         int
    y:         int
    direction: int
    field_d:   int


def _parse_events_from_tokens(swing_tokens: list[int], subdiv: int = BEAT_SUBDIV) -> list[_Event]:
    """Decode the V6 swing-event token stream into structured _Event records."""
    events: list[_Event] = []
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
        if i + 1 >= n: break
        dt_tok = swing_tokens[i + 1]
        if not (DT_BASE <= dt_tok < DT_BASE + DT_COUNT):
            i += 1; continue
        dt = _DT_BINS[dt_tok - DT_BASE]
        current_beat += dt

        if i + 2 >= n: break
        kind_tok = swing_tokens[i + 2]
        if not (KIND_BASE <= kind_tok < KIND_BASE + KIND_COUNT):
            i += 1; continue
        kind = kind_tok

        x = y = direction = field_d = 0
        if i + 4 < n:
            x = max(0, min(swing_tokens[i + 3] - X_BASE, X_COUNT - 1))
            y = max(0, min(swing_tokens[i + 4] - Y_BASE, Y_COUNT - 1))

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
                else:  # CHAIN_HEAD
                    field_d = max(0, min(fd_tok - SLICE_BASE, SLICE_COUNT - 1))
            step = 7

        slot = int(round(current_beat * subdiv))
        events.append(_Event(
            beat=current_beat, slot=slot, hand=hand, kind=kind,
            x=x, y=y, direction=direction, field_d=field_d,
        ))
        i += step

    return events


def _event_to_tokens(e: _Event) -> tuple[list[int], list[int]]:
    """Return (token_ids, role_ids) for one event's spatial tokens.

    Roles tell the decoder what each token MEANS (KIND vs X vs Y vs DIR vs
    FIELD_D), independent of event kind. Variable-length events skip DIR
    (chain_tail, bomb) — those positions just don't appear in the sequence.
    """
    tokens: list[int] = [e.kind,            X_BASE + e.x, Y_BASE + e.y]
    roles:  list[int] = [ROLE_KIND,         ROLE_X,       ROLE_Y]

    if e.kind == BOMB:
        return tokens, roles
    if e.kind == CHAIN_TAIL:
        tokens.append(SQUISH_BASE + min(e.field_d, SQUISH_COUNT - 1))
        roles.append(ROLE_FIELD_D)
        return tokens, roles

    tokens.append(DIR_BASE + e.direction);    roles.append(ROLE_DIR)
    if e.kind == NOTE:
        tokens.append(ANGLE_BASE  + min(e.field_d, ANGLE_COUNT  - 1))
    elif e.kind in (ARC_HEAD, ARC_TAIL):
        tokens.append(MU_BASE     + min(e.field_d, MU_COUNT     - 1))
    else:  # CHAIN_HEAD
        tokens.append(SLICE_BASE  + min(e.field_d, SLICE_COUNT  - 1))
    roles.append(ROLE_FIELD_D)
    return tokens, roles


def _hand_idx(hand: int) -> int:
    if hand == HAND_LEFT:  return HAND_LEFT_IDX
    if hand == HAND_RIGHT: return HAND_RIGHT_IDX
    return HAND_SPECIAL_IDX   # HAND_NONE (bomb) — bombs have no "hand," group with special


class LayoutPhraseDataset(Dataset):
    """Per-phrase dataset for Stage 2 phrase-level autoregressive training.

    Args:
        data_dir:           Directory containing V7-preprocessed .pt files.
        split:              "train" / "val" — uses splits.json if present.
        difficulties:       Difficulties to include (None = all).
        exclude_categories: Mod categories to skip.
        max_layout_len:     Max layout-token sequence length per phrase.
        max_phrase_slots:   Max phrase length in slots (pads encoder input).
        min_notes:          Drop phrases with fewer than this many note events.
    """

    def __init__(
        self,
        data_dir: Path | str,
        split: str = "train",
        difficulties: list[str] | None = None,
        exclude_categories: list[str] | None = None,
        max_layout_len: int = DEFAULT_MAX_LAYOUT_LEN,
        max_phrase_slots: int = MAX_PHRASE_SLOTS,
        min_notes: int = 1,
    ) -> None:
        self.data_dir         = Path(data_dir)
        self.target_diffs     = set(difficulties) if difficulties else None
        self.exclude_cats     = set(exclude_categories) if exclude_categories else set()
        self.max_layout_len   = max_layout_len
        self.max_phrase_slots = max_phrase_slots
        # Special-slot index = max_phrase_slots; the model's slot embedding has
        # `max_phrase_slots + 1` rows, with the last one reserved for this sentinel.
        self._special_slot    = max_phrase_slots
        self.min_notes        = min_notes

        splits_path = self.data_dir / "splits.json"
        song_ids: set[str] | None = None
        if splits_path.exists():
            with open(splits_path) as f:
                song_ids = set(json.load(f).get(split, []))

        blacklist: set[str] = set()
        if (bp := self.data_dir / "blacklist.json").exists():
            with open(bp) as f:
                blacklist = set(json.load(f).keys())

        # Index: (pt_path, diff_name, phrase_idx, diff_id, genre_idx, n_notes)
        self.samples: list[tuple[Path, str, int, int, int, int]] = []
        n_skip_short = 0
        n_skip_long  = 0

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
                continue

            mod_reqs = meta.get("mod_requirements", {})
            if self.exclude_cats and mod_reqs.get("category") in self.exclude_cats:
                continue
            genre_idx = GENRE_MAP.get(mod_reqs.get("genre", "unknown"), 0)

            phrase_b = meta.get("phrase_boundaries") or []
            if not phrase_b:
                continue

            for diff_name, diff_data in meta.get("difficulties", {}).items():
                if self.target_diffs and diff_name not in self.target_diffs:
                    continue
                tokens = diff_data.get("swing_tokens") or []
                if not tokens:
                    continue
                diff_id = DIFFICULTY_MAP.get(diff_name, 3)

                events = _parse_events_from_tokens(tokens)
                if not events:
                    continue

                for pi, (s, e) in enumerate(phrase_b):
                    n_in = sum(1 for ev in events if s <= ev.slot < e and ev.kind != BOMB)
                    if n_in < min_notes:
                        n_skip_short += 1
                        continue
                    if e - s > max_phrase_slots:
                        n_skip_long += 1
                        continue
                    self.samples.append(
                        (pt_path, diff_name, pi, diff_id, genre_idx, n_in)
                    )

        logger.info("LayoutPhraseDataset[%s]: %d phrases (skip_short=%d skip_long=%d)",
                    split, len(self.samples), n_skip_short, n_skip_long)

        self._meta_cache: OrderedDict[str, dict] = OrderedDict()
        self._events_cache: OrderedDict[tuple[str, str], list[_Event]] = OrderedDict()
        self._cache_max = 64

    # ------------------------------------------------------------------
    # Caching
    # ------------------------------------------------------------------
    def _meta(self, pt_path: Path) -> dict:
        key = str(pt_path)
        if key in self._meta_cache:
            self._meta_cache.move_to_end(key)
            return self._meta_cache[key]
        data = torch.load(pt_path, weights_only=False)
        self._meta_cache[key] = data
        if len(self._meta_cache) > self._cache_max:
            self._meta_cache.popitem(last=False)
        return data

    def _events(self, pt_path: Path, diff_name: str, swing_tokens: list[int]) -> list[_Event]:
        key = (str(pt_path), diff_name)
        if key in self._events_cache:
            self._events_cache.move_to_end(key)
            return self._events_cache[key]
        evts = _parse_events_from_tokens(swing_tokens)
        self._events_cache[key] = evts
        if len(self._events_cache) > self._cache_max * 4:
            self._events_cache.popitem(last=False)
        return evts

    # ------------------------------------------------------------------
    # Sample assembly
    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        pt_path, diff_name, phrase_idx, diff_id, genre_idx, _n_notes = self.samples[idx]
        data = self._meta(pt_path)

        mix_beat   = data["mix_beat_features"].float()    # [N_slots, 768]
        phrase_b   = data["phrase_boundaries"]
        swing_toks = data["difficulties"][diff_name]["swing_tokens"]
        events     = self._events(pt_path, diff_name, swing_toks)

        s, e = phrase_b[phrase_idx]
        phrase_len = min(e - s, self.max_phrase_slots)

        # ---- Encoder input ----
        phrase_mert = torch.zeros(self.max_phrase_slots, 768, dtype=torch.float32)
        phrase_mask = torch.zeros(self.max_phrase_slots, dtype=torch.bool)
        clipped_e = s + phrase_len
        if clipped_e <= mix_beat.shape[0]:
            phrase_mert[:phrase_len] = mix_beat[s:clipped_e]
        else:
            real = mix_beat.shape[0] - s
            real = max(0, real)
            phrase_mert[:real] = mix_beat[s:mix_beat.shape[0]]
            phrase_mask[:real] = True
            phrase_mask_set = True
        phrase_mask[:phrase_len] = True

        # ---- Decoder sequence (token / per-token metadata / target) ----
        token_ids: list[int] = [LAYOUT_BOS]
        token_slot: list[int] = [self._special_slot]
        token_hand: list[int] = [HAND_SPECIAL_IDX]
        token_role: list[int] = [ROLE_SPECIAL]

        phrase_events = sorted(
            (ev for ev in events if s <= ev.slot < e),
            key=lambda ev: (ev.slot, _hand_idx(ev.hand), ev.kind),
        )

        for ev in phrase_events:
            ev_tokens, ev_roles = _event_to_tokens(ev)
            slot_in_phrase = min(ev.slot - s, self.max_phrase_slots - 1)
            h_idx = _hand_idx(ev.hand)
            for tid, rid in zip(ev_tokens, ev_roles):
                if len(token_ids) >= self.max_layout_len - 1:  # leave room for EOS
                    break
                token_ids.append(tid)
                token_slot.append(slot_in_phrase)
                token_hand.append(h_idx)
                token_role.append(rid)
            if len(token_ids) >= self.max_layout_len - 1:
                break

        token_ids.append(LAYOUT_EOS)
        token_slot.append(self._special_slot)
        token_hand.append(HAND_SPECIAL_IDX)
        token_role.append(ROLE_SPECIAL)

        # Pad to max_layout_len
        L = len(token_ids)
        pad_n = self.max_layout_len - L
        if pad_n > 0:
            token_ids.extend([LAYOUT_PAD] * pad_n)
            token_slot.extend([self._special_slot] * pad_n)
            token_hand.extend([HAND_SPECIAL_IDX] * pad_n)
            token_role.extend([ROLE_SPECIAL]    * pad_n)

        tok_t  = torch.tensor(token_ids,  dtype=torch.long)
        slot_t = torch.tensor(token_slot, dtype=torch.long)
        hand_t = torch.tensor(token_hand, dtype=torch.long)
        role_t = torch.tensor(token_role, dtype=torch.long)

        # Targets are next-token prediction; ignore loss on shifted PAD positions.
        target = tok_t.clone()
        target = torch.cat([target[1:], torch.tensor([LAYOUT_PAD], dtype=torch.long)])
        target[target == LAYOUT_PAD] = IGNORE_INDEX

        return {
            "phrase_mert":   phrase_mert,                                # [P, 768]
            "phrase_mask":   phrase_mask,                                # [P]
            "layout_tokens": tok_t,                                      # [S]
            "token_slot":    slot_t,                                     # [S]
            "token_hand":    hand_t,                                     # [S]
            "token_role":    role_t,                                     # [S]
            "target":        target,                                     # [S]
            "difficulty":    torch.tensor(diff_id,   dtype=torch.long),
            "genre":         torch.tensor(genre_idx, dtype=torch.long),
        }
