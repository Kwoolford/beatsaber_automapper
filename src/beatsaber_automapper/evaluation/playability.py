"""Heuristic playability checks for generated Beat Saber maps.

Validates that generated maps don't contain impossible or unplayable
patterns. Checks include:
    - No overlapping notes at the same grid position
    - Valid direction sequences (parity)
    - Reasonable spacing between notes
    - No out-of-bounds grid coordinates
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def check_playability(beatmap_data: dict) -> list[dict]:
    """Run playability heuristic checks on a beatmap.

    Args:
        beatmap_data: v3 beatmap JSON dictionary.

    Returns:
        List of issue dictionaries with 'severity' and 'message' fields.
    """
    raise NotImplementedError("Playability checks will be implemented in PR 5")
