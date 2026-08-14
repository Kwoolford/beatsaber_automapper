"""V7-2: Beat grid label extraction from V6 swing_tokens.

Converts the existing flat swing-event token stream into a 2D binary grid:
  [N_slots, 2]  — columns are (left_note_present, right_note_present)

One row per 1/subdiv-note beat slot. Used as ground truth for Stage 1
BeatClassifier training.

Also extracts per-slot note kind (NOTE / ARC_HEAD / CHAIN_HEAD / BOMB)
for richer Stage 2 conditioning, though Stage 1 only uses binary presence.
"""

from __future__ import annotations

import logging

import numpy as np
import torch

from beatsaber_automapper.data.swing_tokenizer import (
    _DT_BINS,
    BOS,
    BOMB,
    DT_BASE,
    DT_COUNT,
    EOS,
    HAND_LEFT,
    HAND_NONE,
    HAND_RIGHT,
    KIND_BASE,
    KIND_COUNT,
    NOTE,
    PAD,
)

logger = logging.getLogger(__name__)

# 1/4-note resolution. Imported from `mert_encoder` rather than redeclared so the two
# CANNOT drift apart — generate.py reads BEAT_SUBDIV from both modules, in three
# separate places, and a disagreement would silently break the slot→beat conversion
# while still producing a map. See mert_encoder for why this is env-overridable.
from beatsaber_automapper.data.mert_encoder import BEAT_SUBDIV  # noqa: E402

# Kind IDs that count as a "real note" for the beat presence label.
# ARC_TAIL and CHAIN_TAIL are tails of events — the head already set the label.
_NOTE_LIKE_KINDS = frozenset({NOTE, KIND_BASE + 1, KIND_BASE + 3})  # NOTE, ARC_HEAD, CHAIN_HEAD


def extract_beat_labels(
    swing_tokens: list[int],
    bpm: float,
    n_slots: int,
    subdiv: int = BEAT_SUBDIV,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Derive binary note-presence labels per beat slot from swing_tokens.

    Parses the flat token stream, accumulates beat position from Δt tokens,
    and sets left/right binary labels at the nearest 1/subdiv-note slot.

    Args:
        swing_tokens: Flat swing-event token list (from .pt["difficulties"][d]["swing_tokens"]).
        bpm:          Song tempo in BPM.
        n_slots:      Total number of beat slots (= int(total_beats * subdiv)).
        subdiv:       Beat subdivisions per beat (4 = 1/4 note).

    Returns:
        left_labels  [n_slots] int32 binary — 1 if left hand note at this slot
        right_labels [n_slots] int32 binary — 1 if right hand note at this slot
        left_kinds   [n_slots] int32 — KIND token ID for the note (0 if empty)
        right_kinds  [n_slots] int32 — KIND token ID for the note (0 if empty)
    """
    left_labels  = np.zeros(n_slots, dtype=np.int32)
    right_labels = np.zeros(n_slots, dtype=np.int32)
    left_kinds   = np.zeros(n_slots, dtype=np.int32)
    right_kinds  = np.zeros(n_slots, dtype=np.int32)

    tokens = list(swing_tokens)
    i = 0
    n = len(tokens)
    current_beat = 0.0

    while i < n:
        tok = tokens[i]

        if tok in (PAD, BOS):
            i += 1
            continue
        if tok == EOS:
            break
        if tok not in (HAND_LEFT, HAND_RIGHT, HAND_NONE):
            i += 1
            continue

        hand = tok
        if i + 1 >= n:
            break

        dt_tok = tokens[i + 1]
        if not (DT_BASE <= dt_tok < DT_BASE + DT_COUNT):
            i += 1
            continue

        dt = _DT_BINS[dt_tok - DT_BASE]
        current_beat += dt

        # Snap to nearest beat slot
        slot = int(round(current_beat * subdiv))
        slot = max(0, min(slot, n_slots - 1))

        # Read KIND token (i+2)
        kind = 0
        if i + 2 < n:
            k_tok = tokens[i + 2]
            if KIND_BASE <= k_tok < KIND_BASE + KIND_COUNT:
                kind = k_tok

        # Set label: only for note-like events (not ARC_TAIL / CHAIN_TAIL)
        if hand == HAND_LEFT and kind in _NOTE_LIKE_KINDS:
            left_labels[slot] = 1
            left_kinds[slot]  = kind
        elif hand == HAND_RIGHT and kind in _NOTE_LIKE_KINDS:
            right_labels[slot] = 1
            right_kinds[slot]  = kind
        # HAND_NONE = bomb; intentionally excluded from note labels

        # Skip to next event: advance past HAND + DT + rest of event tokens
        i += 2
        while i < n and tokens[i] not in (HAND_LEFT, HAND_RIGHT, HAND_NONE, EOS, BOS, PAD):
            i += 1

    n_left  = int(left_labels.sum())
    n_right = int(right_labels.sum())
    logger.debug("Beat labels: %d L + %d R notes across %d slots", n_left, n_right, n_slots)
    return left_labels, right_labels, left_kinds, right_kinds


def beat_labels_from_pt(
    data: dict,
    difficulty: str = "Expert",
    subdiv: int = BEAT_SUBDIV,
) -> dict[str, torch.Tensor] | None:
    """Extract beat grid labels from a loaded .pt file dict.

    Returns a dict with:
        drum_beat_features  [N_slots, 768]  (from V7 preprocessing)
        mix_beat_features   [N_slots, 768]
        left_labels         [N_slots]       binary int
        right_labels        [N_slots]       binary int
        n_slots             int

    Returns None if required data is missing.
    """
    # V7 features must already be present (run preprocess_v7.py first)
    if "drum_beat_features" not in data:
        return None

    diff_data = data.get("difficulties", {}).get(difficulty)
    if diff_data is None:
        # Fall back to any available difficulty
        for d in ("ExpertPlus", "Expert", "Hard", "Normal", "Easy"):
            diff_data = data.get("difficulties", {}).get(d)
            if diff_data is not None:
                break
    if diff_data is None or not diff_data.get("swing_tokens"):
        return None

    drum_beat = data["drum_beat_features"].float()  # [N_slots, 768]
    mix_beat  = data["mix_beat_features"].float()
    n_slots   = drum_beat.shape[0]

    bpm = float(data.get("bpm", 120.0))
    swing_tokens = diff_data["swing_tokens"]

    left_labels, right_labels, left_kinds, right_kinds = extract_beat_labels(
        swing_tokens, bpm, n_slots, subdiv,
    )

    return {
        "drum_beat_features": drum_beat,
        "mix_beat_features":  mix_beat,
        "left_labels":        torch.from_numpy(left_labels),
        "right_labels":       torch.from_numpy(right_labels),
        "left_kinds":         torch.from_numpy(left_kinds),
        "right_kinds":        torch.from_numpy(right_kinds),
        "n_slots":            n_slots,
        "bpm":                bpm,
    }
