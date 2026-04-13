"""Heuristic playability checks for generated Beat Saber maps.

Validates that generated maps don't contain impossible or unplayable
patterns. Returns a list of issues found, each with severity and details.

This is a read-only analysis tool — it does NOT modify the beatmap.
Use postprocess.py for fixing issues.
"""

from __future__ import annotations

import logging
from collections import defaultdict

logger = logging.getLogger(__name__)

# Parity direction groups
_FOREHAND_DIRS = frozenset({1, 6, 7})  # down, down-left, down-right
_BACKHAND_DIRS = frozenset({0, 4, 5})  # up, up-left, up-right
_NEUTRAL_DIRS = frozenset({2, 3, 8})   # left, right, any/dot


def check_playability(beatmap_data: dict) -> list[dict]:
    """Run playability heuristic checks on a beatmap.

    Args:
        beatmap_data: v3 beatmap JSON dictionary with colorNotes, bombNotes, etc.

    Returns:
        List of issue dictionaries with 'severity', 'check', 'message',
        and optionally 'beat' fields. Severity is one of: 'error', 'warning', 'info'.
    """
    issues: list[dict] = []
    notes = beatmap_data.get("colorNotes", [])

    if not notes:
        issues.append({
            "severity": "warning",
            "check": "empty_map",
            "message": "Beatmap has no color notes",
        })
        return issues

    issues.extend(_check_grid_bounds(notes))
    issues.extend(_check_overlapping_notes(notes))
    issues.extend(_check_note_density(notes))
    issues.extend(_check_parity(notes))
    issues.extend(_check_color_balance(notes))
    issues.extend(_check_dot_note_overuse(notes))

    return issues


def summarize_issues(issues: list[dict]) -> dict[str, int]:
    """Summarize issue counts by severity.

    Args:
        issues: List of issue dicts from check_playability.

    Returns:
        Dict with counts per severity level and total.
    """
    counts: dict[str, int] = {"error": 0, "warning": 0, "info": 0}
    for issue in issues:
        counts[issue["severity"]] = counts.get(issue["severity"], 0) + 1
    counts["total"] = len(issues)
    return counts


def _check_grid_bounds(notes: list[dict]) -> list[dict]:
    """Check for notes outside the valid 4x3 grid."""
    issues = []
    for note in notes:
        x, y = note.get("x", 0), note.get("y", 0)
        if not (0 <= x <= 3 and 0 <= y <= 2):
            issues.append({
                "severity": "error",
                "check": "grid_bounds",
                "message": f"Note at beat {note.get('b', '?')} has invalid position ({x}, {y})",
                "beat": note.get("b"),
            })
        d = note.get("d", 0)
        if not (0 <= d <= 8):
            issues.append({
                "severity": "error",
                "check": "direction_bounds",
                "message": f"Note at beat {note.get('b', '?')} has invalid direction {d}",
                "beat": note.get("b"),
            })
    return issues


def _check_overlapping_notes(notes: list[dict]) -> list[dict]:
    """Check for multiple notes at the same grid position on the same beat."""
    issues = []
    # Group by beat (within tolerance of 0.01 beats)
    by_beat: dict[float, list[dict]] = defaultdict(list)
    for note in notes:
        beat = round(note.get("b", 0), 2)
        by_beat[beat].append(note)

    for beat, beat_notes in by_beat.items():
        positions: set[tuple[int, int]] = set()
        for note in beat_notes:
            pos = (note.get("x", 0), note.get("y", 0))
            if pos in positions:
                issues.append({
                    "severity": "error",
                    "check": "overlap",
                    "message": f"Overlapping notes at beat {beat} position {pos}",
                    "beat": beat,
                })
            positions.add(pos)

        # Check same-color duplicates
        color_counts: dict[int, int] = defaultdict(int)
        for note in beat_notes:
            color_counts[note.get("c", 0)] += 1
        for color, count in color_counts.items():
            if count > 2:
                color_name = "red" if color == 0 else "blue"
                issues.append({
                    "severity": "error",
                    "check": "same_color_excess",
                    "message": (
                        f"Beat {beat}: {count} {color_name} notes "
                        f"(max 2 for ExpertPlus)"
                    ),
                    "beat": beat,
                })

    return issues


def _check_note_density(notes: list[dict]) -> list[dict]:
    """Check for excessive note density (clumping)."""
    issues = []
    by_beat: dict[float, int] = defaultdict(int)
    for note in notes:
        beat = round(note.get("b", 0), 2)
        by_beat[beat] += 1

    clump_beats = sum(1 for count in by_beat.values() if count > 3)
    if clump_beats > 0:
        issues.append({
            "severity": "warning",
            "check": "note_clumping",
            "message": f"{clump_beats} beats have >3 notes (potential clumping)",
        })

    extreme_beats = sum(1 for count in by_beat.values() if count > 4)
    if extreme_beats > 0:
        issues.append({
            "severity": "error",
            "check": "extreme_clumping",
            "message": f"{extreme_beats} beats have >4 notes (unplayable clumping)",
        })

    return issues


def _check_parity(notes: list[dict]) -> list[dict]:
    """Check parity (swing direction) violations per color."""
    issues = []
    violations = 0
    total_checks = 0

    # Sort by beat, then process per color
    sorted_notes = sorted(notes, key=lambda n: (n.get("b", 0), n.get("c", 0)))

    for color in (0, 1):
        color_notes = [n for n in sorted_notes if n.get("c", 0) == color]
        last_dir = None
        last_beat = -999.0

        for note in color_notes:
            d = note.get("d", 8)
            beat = note.get("b", 0)

            # Skip neutral directions for parity tracking
            if d in _NEUTRAL_DIRS:
                continue

            # Reset parity after long gaps (>3 beats)
            if beat - last_beat > 3.0:
                last_dir = None

            if last_dir is not None:
                total_checks += 1
                last_is_forehand = last_dir in _FOREHAND_DIRS
                curr_is_forehand = d in _FOREHAND_DIRS

                if last_is_forehand == curr_is_forehand:
                    violations += 1

            last_dir = d
            last_beat = beat

    if total_checks > 0:
        rate = violations / total_checks
        if rate > 0.1:
            issues.append({
                "severity": "error",
                "check": "parity",
                "message": (
                    f"{violations}/{total_checks} parity violations "
                    f"({rate:.1%}) — should be <10%"
                ),
            })
        elif rate > 0.02:
            issues.append({
                "severity": "warning",
                "check": "parity",
                "message": (
                    f"{violations}/{total_checks} parity violations "
                    f"({rate:.1%})"
                ),
            })
        else:
            issues.append({
                "severity": "info",
                "check": "parity",
                "message": (
                    f"Parity: {violations}/{total_checks} violations "
                    f"({rate:.1%}) — good"
                ),
            })

    return issues


def _check_color_balance(notes: list[dict]) -> list[dict]:
    """Check red/blue color balance."""
    issues = []
    red = sum(1 for n in notes if n.get("c", 0) == 0)
    blue = sum(1 for n in notes if n.get("c", 0) == 1)
    total = red + blue

    if total == 0:
        return issues

    ratio = red / total
    if ratio < 0.3 or ratio > 0.7:
        issues.append({
            "severity": "warning",
            "check": "color_balance",
            "message": (
                f"Color imbalance: {red} red ({ratio:.0%}) vs "
                f"{blue} blue ({1 - ratio:.0%}) — target 40-60%"
            ),
        })
    else:
        issues.append({
            "severity": "info",
            "check": "color_balance",
            "message": f"Color balance: {red} red ({ratio:.0%}) / {blue} blue ({1 - ratio:.0%})",
        })

    return issues


def _check_dot_note_overuse(notes: list[dict]) -> list[dict]:
    """Check for excessive dot/any direction (direction 8) usage."""
    issues = []
    dot_count = sum(1 for n in notes if n.get("d", 0) == 8)
    total = len(notes)

    if total == 0:
        return issues

    rate = dot_count / total
    if rate > 0.15:
        issues.append({
            "severity": "warning",
            "check": "dot_overuse",
            "message": (
                f"{dot_count}/{total} notes ({rate:.1%}) use direction 8 (dot) "
                f"— should be <15%"
            ),
        })

    return issues
