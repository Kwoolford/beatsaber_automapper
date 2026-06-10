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
ROLE_CONTEXT = 6  # cross-phrase context prefix tokens (loss masked)
N_ROLES      = 7  # includes ROLE_CONTEXT

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

# TASK 3 (S2 pitch-contour conditioning): the per-slot melodic anchor channel.
# Columns 7,8,9 of the cached `instr_beat_features` [N_slots, 10] (written by
# scripts/preprocess_instruments.py): lead_pitch, lead_dpitch, bass_pitch. We
# slice them into a per-slot [N_slots, 3] contour tensor and feed it into the
# encoder alongside the MERT slots, so the decoder's cross-attention can bias
# swing DIRECTION to follow the melodic line (ascending→up/right, etc.). No new
# preprocess pass — these columns already ship in every .pt.
CONTOUR_COLS = slice(7, 10)
CONTOUR_DIM = 3


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
        ctx_len: int = 0,
        max_song_phrases: int = 150,
        min_nps: float | None = None,
        max_nps: float | None = None,
        use_contour: bool = False,
    ) -> None:
        self.data_dir         = Path(data_dir)
        self.target_diffs     = set(difficulties) if difficulties else None
        # Cohort quality filter (V8-0 follow-up / orthogonal data fix): drop whole
        # (song, difficulty) pairs whose overall notes-per-second is outside this
        # band. ExpertPlus maps that run 9-15 NPS teach ergonomically hard
        # "for-sport" swings the user flagged; capping at ~4-8 NPS keeps the cohort
        # to musical, human-playable density regardless of the difficulty label.
        self.min_nps          = min_nps
        self.max_nps          = max_nps
        self.exclude_cats     = set(exclude_categories) if exclude_categories else set()
        self.max_layout_len   = max_layout_len
        self.max_phrase_slots = max_phrase_slots
        # Special-slot index = max_phrase_slots; the model's slot embedding has
        # `max_phrase_slots + 1` rows, with the last one reserved for this sentinel.
        self._special_slot    = max_phrase_slots
        self.min_notes        = min_notes
        self.max_song_phrases = max_song_phrases
        # ctx_len > 0: prepend the last ctx_len spatial tokens from the prior
        # phrase into the decoder sequence (with ROLE_CONTEXT; loss is masked).
        self.ctx_len          = ctx_len
        # TASK 3: feed per-slot pitch contour (instr_beat_features cols 7:10) into
        # the encoder. Off by default → prior behaviour / old ckpts unchanged.
        self.use_contour      = use_contour

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
        n_skip_nps   = 0

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

                # Cohort NPS filter. Duration from the last event slot (slots are
                # 1/BEAT_SUBDIV-beat) and the song bpm. Counts non-bomb notes only.
                if self.min_nps is not None or self.max_nps is not None:
                    bpm = float(meta.get("bpm") or 0.0)
                    note_events = [ev for ev in events if ev.kind != BOMB]
                    max_slot = max((ev.slot for ev in note_events), default=0)
                    dur_sec = (max_slot / BEAT_SUBDIV) * (60.0 / bpm) if bpm > 0 else 0.0
                    nps = len(note_events) / dur_sec if dur_sec > 1.0 else 0.0
                    if (self.min_nps is not None and nps < self.min_nps) or \
                       (self.max_nps is not None and nps > self.max_nps):
                        n_skip_nps += 1
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

        logger.info("LayoutPhraseDataset[%s]: %d phrases (skip_short=%d skip_long=%d "
                    "skip_nps=%d, nps_band=[%s,%s])",
                    split, len(self.samples), n_skip_short, n_skip_long, n_skip_nps,
                    self.min_nps, self.max_nps)

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
        # Only positions that are actually backed by mix_beat data should be
        # marked True in the mask. Otherwise the encoder attends to zero
        # vectors at the end of the last phrase of every song.
        phrase_mert = torch.zeros(self.max_phrase_slots, 768, dtype=torch.float32)
        phrase_mask = torch.zeros(self.max_phrase_slots, dtype=torch.bool)
        clipped_e = min(s + phrase_len, mix_beat.shape[0])
        real = max(0, clipped_e - s)
        if real > 0:
            phrase_mert[:real] = mix_beat[s:clipped_e]
            phrase_mask[:real] = True

        # ---- Cross-phrase context prefix ----
        # Collect the last ctx_len spatial tokens from the previous phrase (same
        # song, same difficulty). These are prepended to the decoder input with
        # role=ROLE_CONTEXT so the model can see the trailing note pattern from
        # the prior phrase. Loss is masked on these positions (they're context,
        # not predictions). For phrase_idx==0, or if ctx_len==0, this is empty.
        ctx_token_ids: list[int] = []
        ctx_token_slot: list[int] = []
        ctx_token_hand: list[int] = []
        ctx_token_role: list[int] = []

        if self.ctx_len > 0 and phrase_idx > 0:
            prev_s, prev_e = phrase_b[phrase_idx - 1]
            prev_events = sorted(
                (ev for ev in events if prev_s <= ev.slot < prev_e),
                key=lambda ev: (ev.slot, _hand_idx(ev.hand), ev.kind),
            )
            # Build the full token sequence for the previous phrase
            all_prev_toks: list[int] = []
            all_prev_slots: list[int] = []
            all_prev_hands: list[int] = []
            all_prev_roles: list[int] = []
            for ev in prev_events:
                ev_tokens, ev_roles = _event_to_tokens(ev)
                sp = min(ev.slot - prev_s, self.max_phrase_slots - 1)
                h_idx = _hand_idx(ev.hand)
                for tid, rid in zip(ev_tokens, ev_roles):
                    all_prev_toks.append(tid)
                    all_prev_slots.append(sp)
                    all_prev_hands.append(h_idx)
                    all_prev_roles.append(rid)
            # Take the last ctx_len tokens and relabel with ROLE_CONTEXT
            tail = all_prev_toks[-self.ctx_len:]
            tail_slots = all_prev_slots[-self.ctx_len:]
            tail_hands = all_prev_hands[-self.ctx_len:]
            ctx_token_ids   = tail
            ctx_token_slot  = tail_slots
            ctx_token_hand  = tail_hands
            ctx_token_role  = [ROLE_CONTEXT] * len(tail)

        # ---- Decoder sequence (context prefix + BOS + event tokens) ----
        token_ids: list[int] = ctx_token_ids + [LAYOUT_BOS]
        token_slot: list[int] = ctx_token_slot + [self._special_slot]
        token_hand: list[int] = ctx_token_hand + [HAND_SPECIAL_IDX]
        token_role: list[int] = ctx_token_role + [ROLE_SPECIAL]

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

        # Targets are next-token prediction; ignore loss on shifted PAD positions
        # AND on the context prefix (we don't want to predict those — they're
        # read-only context for the decoder, not training targets).
        ctx_n  = len(ctx_token_ids)
        target = tok_t.clone()
        target = torch.cat([target[1:], torch.tensor([LAYOUT_PAD], dtype=torch.long)])
        target[target == LAYOUT_PAD] = IGNORE_INDEX
        # Mask loss on context prefix positions (positions 0 .. ctx_n-1)
        if ctx_n > 0:
            target[:ctx_n] = IGNORE_INDEX

        # ---- Song-memory: all phrase fingerprints for this song ----
        # Padded to max_song_phrases so batches collate cleanly. The model's
        # encoder concatenates these to the local phrase memory so the decoder
        # can dynamically attend to whichever prior phrase had similar melody —
        # replacing the hard-threshold PhraseIndex with learned soft retrieval.
        MAX_FP = self.max_song_phrases
        fp_raw = data.get("phrase_fingerprints")  # [N, 768] float16 | None
        if fp_raw is not None:
            N_fp = min(fp_raw.shape[0], MAX_FP)
            song_fps      = torch.zeros(MAX_FP, 768, dtype=torch.float32)
            song_fp_mask  = torch.zeros(MAX_FP, dtype=torch.bool)
            song_fps[:N_fp]     = fp_raw[:N_fp].float()
            song_fp_mask[:N_fp] = True
        else:
            song_fps     = torch.zeros(MAX_FP, 768, dtype=torch.float32)
            song_fp_mask = torch.zeros(MAX_FP, dtype=torch.bool)

        # ---- TASK 3: per-slot pitch contour (lead_pitch/dpitch/bass_pitch) ----
        # Same slot grid + same slicing/padding as phrase_mert so it lines up
        # 1:1 with the encoder positions. Zeros where the song has no cached
        # instr features (1/5320 songs) or beyond the real slot count.
        out: dict[str, torch.Tensor]
        if self.use_contour:
            phrase_contour = torch.zeros(self.max_phrase_slots, CONTOUR_DIM, dtype=torch.float32)
            instr = data.get("instr_beat_features")
            if instr is not None and real > 0:
                contour_slice = instr[s:clipped_e, CONTOUR_COLS].float()
                phrase_contour[:contour_slice.shape[0]] = contour_slice

        out = {
            "phrase_mert":   phrase_mert,                                # [P, 768]
            "phrase_mask":   phrase_mask,                                # [P]
            "layout_tokens": tok_t,                                      # [S]
            "token_slot":    slot_t,                                     # [S]
            "token_hand":    hand_t,                                     # [S]
            "token_role":    role_t,                                     # [S]
            "target":        target,                                     # [S]
            "difficulty":    torch.tensor(diff_id,   dtype=torch.long),
            "genre":         torch.tensor(genre_idx, dtype=torch.long),
            "ctx_len":       torch.tensor(ctx_n,     dtype=torch.long),
            "song_fps":      song_fps,                                   # [MAX_FP, 768]
            "song_fp_mask":  song_fp_mask,                               # [MAX_FP]
        }
        if self.use_contour:
            out["phrase_contour"] = phrase_contour                       # [P, 3]
        return out
