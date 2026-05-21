"""V7-6: PhraseIndex — cross-song consistency memory for generation.

Maintains a list of (phrase_fingerprint, note_pattern) pairs accumulated
during left-to-right song generation. When a new phrase has high cosine
similarity to a prior phrase, hard retrieval replays the stored note
pattern as conditioning for Stage 2.

This enforces that the second chorus produces the same note patterns as
the first chorus, and that any repeating musical motif gets consistent
treatment throughout the song.

Usage:
    index = PhraseIndex(similarity_threshold=0.85)
    index.build(mix_beat_features, phrase_boundaries)

    for window_start, window_end in phrase_boundaries:
        fingerprint = mix_beat[window_start:window_end].mean(0)
        prior = index.query(fingerprint)
        if prior is not None:
            # hard retrieval: use prior.events for this window
            ...
        else:
            # generate freely
            events = stage2.generate(...)
            index.record(fingerprint, events)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import torch

logger = logging.getLogger(__name__)


@dataclass
class NotePattern:
    """Stored note pattern for one phrase window.

    Keyed by (beat_slot, hand) → list of spatial token IDs.
    beat_slot is relative to the phrase's start slot.
    """
    window_start_slot: int
    tokens_by_position: dict[tuple[int, int], list[int]] = field(default_factory=dict)

    def add_event(self, relative_slot: int, hand: int, spatial_tokens: list[int]) -> None:
        self.tokens_by_position[(relative_slot, hand)] = spatial_tokens

    def get_event(self, relative_slot: int, hand: int) -> list[int] | None:
        return self.tokens_by_position.get((relative_slot, hand))

    def __len__(self) -> int:
        return len(self.tokens_by_position)


class PhraseIndex:
    """Cosine-similarity-based phrase memory for consistent song generation.

    Args:
        similarity_threshold: Minimum cosine similarity to trigger hard retrieval
                              (default 0.85 — experimentally chosen starting point).
    """

    def __init__(self, similarity_threshold: float = 0.85) -> None:
        self.threshold   = similarity_threshold
        self._fingerprints: list[torch.Tensor] = []   # [768] per entry
        self._patterns:     list[NotePattern]  = []
        self._boundaries:   list[tuple[int, int]] = []

    def build(
        self,
        mix_beat_features: torch.Tensor,
        phrase_boundaries: list[tuple[int, int]],
    ) -> None:
        """Pre-compute phrase fingerprints from full-song beat features.

        Called once at the start of generation before the main loop.
        Populates internal fingerprint list; patterns are populated by
        record() calls during generation.

        Args:
            mix_beat_features: [N_slots, 768] beat-aligned mix MERT features.
            phrase_boundaries: list of (start_slot, end_slot) pairs.
        """
        self._fingerprints = []
        self._patterns     = []
        self._boundaries   = list(phrase_boundaries)

        for start, end in phrase_boundaries:
            chunk = mix_beat_features[start:end]
            if chunk.shape[0] > 0:
                fp = chunk.mean(0)  # [768]
            else:
                fp = torch.zeros(mix_beat_features.shape[1])
            self._fingerprints.append(fp.float())
            self._patterns.append(None)   # unfilled until record() is called

        logger.debug("PhraseIndex: %d phrase windows indexed.", len(self._boundaries))

    def query(
        self,
        phrase_emb: torch.Tensor,
        exclude_last: int = 1,
    ) -> NotePattern | None:
        """Find the most similar previously-generated phrase.

        Only considers phrases that have already been recorded (not None) and
        excludes the most recent `exclude_last` entries (to avoid self-match
        with the immediately preceding phrase).

        Args:
            phrase_emb:   [768] fingerprint of the phrase to look up.
            exclude_last: Don't match against the N most-recently-added phrases.

        Returns:
            The stored NotePattern if max cosine similarity ≥ threshold, else None.
        """
        best_sim  = -1.0
        best_idx  = -1

        n = len(self._fingerprints)
        cutoff = max(0, n - exclude_last)

        for i in range(cutoff):
            if self._patterns[i] is None:
                continue   # not yet generated
            sim = float(torch.nn.functional.cosine_similarity(
                phrase_emb.unsqueeze(0),
                self._fingerprints[i].unsqueeze(0),
            ).item())
            if sim > best_sim:
                best_sim = sim
                best_idx = i

        if best_sim >= self.threshold:
            logger.debug("PhraseIndex: hit (sim=%.3f, idx=%d)", best_sim, best_idx)
            return self._patterns[best_idx]

        logger.debug("PhraseIndex: miss (best_sim=%.3f)", best_sim)
        return None

    def record(
        self,
        phrase_emb: torch.Tensor,
        pattern: NotePattern,
    ) -> None:
        """Store a generated note pattern for future retrieval.

        Finds the closest pre-indexed fingerprint slot and fills it with
        the generated pattern. If the phrase wasn't pre-indexed (shouldn't
        happen in normal operation), appends it.

        Args:
            phrase_emb: [768] fingerprint of the phrase that was generated.
            pattern:    NotePattern produced by Stage 2 for this phrase.
        """
        if not self._fingerprints:
            self._fingerprints.append(phrase_emb.float())
            self._patterns.append(pattern)
            return

        # Find the closest unrecorded slot
        sims = torch.stack([
            torch.nn.functional.cosine_similarity(
                phrase_emb.unsqueeze(0), fp.unsqueeze(0)
            )
            for fp in self._fingerprints
        ])
        # Prefer unrecorded (None) slots
        candidates = [i for i, p in enumerate(self._patterns) if p is None]
        if candidates:
            # Among unrecorded, pick highest similarity
            best = max(candidates, key=lambda i: float(sims[i]))
        else:
            best = int(sims.argmax())

        self._patterns[best] = pattern
        logger.debug("PhraseIndex: recorded at slot %d (len=%d)", best, len(pattern))

    def clear(self) -> None:
        """Reset the index (call between songs)."""
        self._fingerprints.clear()
        self._patterns.clear()
        self._boundaries.clear()

    def fingerprint_for_slot(self, beat_slot: int) -> torch.Tensor | None:
        """Return the pre-computed fingerprint for the phrase containing beat_slot."""
        for i, (start, end) in enumerate(self._boundaries):
            if start <= beat_slot < end:
                return self._fingerprints[i]
        return None

    def boundary_for_slot(self, beat_slot: int) -> tuple[int, int] | None:
        """Return (start, end) boundary of the phrase containing beat_slot."""
        for start, end in self._boundaries:
            if start <= beat_slot < end:
                return start, end
        return None

    def __len__(self) -> int:
        return len(self._fingerprints)
