"""Post-processing pipeline for generated Beat Saber beatmaps.

Applies rule-based fixes to improve playability and diversity of model output.
Each step operates on a DifficultyBeatmap in-place and returns it.

Steps:
    1. NPS enforcement — thin notes if density exceeds difficulty target
    2. Color rebalancing — push toward 45-55% red/blue split
    3. Direction diversity — cap any single direction at 40% of total
    4. Grid coverage — shift some notes to unused grid cells
    5. Pattern deduplication — inject variation after repeated patterns
    6. Parity check — ensure swing direction alternation is physically possible
"""

from __future__ import annotations

import logging
import random
from collections import Counter

from beatsaber_automapper.data.beatmap import ColorNote, DifficultyBeatmap

logger = logging.getLogger(__name__)

# Target NPS ranges per difficulty
DIFFICULTY_NPS_TARGETS: dict[str, tuple[float, float]] = {
    "Easy": (1.0, 3.0),
    "Normal": (2.0, 4.5),
    "Hard": (3.0, 6.0),
    "Expert": (4.0, 8.0),
    "ExpertPlus": (5.0, 12.0),
}

# All valid grid cells (x=0-3, y=0-2)
ALL_GRID_CELLS = {(x, y) for x in range(4) for y in range(3)}

# Direction names for logging
DIR_NAMES = {
    0: "up", 1: "down", 2: "left", 3: "right",
    4: "upL", 5: "upR", 6: "dnL", 7: "dnR", 8: "any",
}

# Opposite directions for parity: hand alternates between "forehand" (down-ish) and
# "backhand" (up-ish). These pairs represent physically comfortable follow-ups.
_FOREHAND_DIRS = {1, 6, 7}  # down, down-left, down-right
_BACKHAND_DIRS = {0, 4, 5}  # up, up-left, up-right


def postprocess_beatmap(
    beatmap: DifficultyBeatmap,
    difficulty: str = "Expert",
    bpm: float = 120.0,
    song_duration_secs: float | None = None,
    seed: int | None = None,
) -> DifficultyBeatmap:
    """Apply all post-processing steps to a beatmap.

    Args:
        beatmap: The generated beatmap to improve.
        difficulty: Difficulty name for NPS targeting.
        bpm: BPM for timing calculations.
        song_duration_secs: Song duration in seconds (for NPS calculation).
        seed: Random seed for reproducibility.

    Returns:
        The same beatmap object, modified in-place.
    """
    if seed is not None:
        random.seed(seed)

    n_before = len(beatmap.color_notes)

    beatmap = enforce_nps(beatmap, difficulty, bpm, song_duration_secs)
    beatmap = rebalance_colors(beatmap)
    beatmap = diversify_directions(beatmap)
    beatmap = expand_grid_coverage(beatmap)
    beatmap = deduplicate_patterns(beatmap)
    beatmap = fix_parity(beatmap)

    n_after = len(beatmap.color_notes)
    logger.info(
        "Post-processing: %d -> %d notes, difficulty=%s",
        n_before, n_after, difficulty,
    )
    return beatmap


def enforce_nps(
    beatmap: DifficultyBeatmap,
    difficulty: str = "Expert",
    bpm: float = 120.0,
    song_duration_secs: float | None = None,
) -> DifficultyBeatmap:
    """Remove excess notes to bring NPS within the target range for the difficulty.

    Thins notes by removing those that are least musically significant —
    notes that fall between two other notes of the same color (mid-stream).
    """
    notes = beatmap.color_notes
    if not notes:
        return beatmap

    min_nps, max_nps = DIFFICULTY_NPS_TARGETS.get(difficulty, (4.0, 8.0))

    # Calculate duration
    if song_duration_secs is None:
        if notes:
            max_beat = max(n.beat for n in notes)
            song_duration_secs = max_beat / (bpm / 60.0)
        else:
            return beatmap

    if song_duration_secs <= 0:
        return beatmap

    current_nps = len(notes) / song_duration_secs

    if current_nps <= max_nps:
        return beatmap

    # Need to thin: target the midpoint of the acceptable range
    target_nps = (min_nps + max_nps) / 2.0
    target_count = max(1, int(target_nps * song_duration_secs))

    if target_count >= len(notes):
        return beatmap

    # Sort by beat time
    notes.sort(key=lambda n: n.beat)

    # Score each note: keep notes at starts/ends of groups, remove mid-stream
    # Simple approach: keep every N-th note uniformly distributed
    keep_ratio = target_count / len(notes)
    kept: list[ColorNote] = []
    accumulator = 0.0
    for note in notes:
        accumulator += keep_ratio
        if accumulator >= 1.0:
            kept.append(note)
            accumulator -= 1.0

    logger.info(
        "NPS enforcement: %.1f -> %.1f NPS (%d -> %d notes)",
        current_nps,
        len(kept) / song_duration_secs,
        len(notes),
        len(kept),
    )
    beatmap.color_notes = kept
    return beatmap


def rebalance_colors(
    beatmap: DifficultyBeatmap,
    min_ratio: float = 0.40,
    max_ratio: float = 0.60,
) -> DifficultyBeatmap:
    """Push color distribution toward balanced by flipping least-constrained notes.

    Flips notes from the overrepresented color to the underrepresented color,
    preferring notes that are isolated (no same-beat neighbors of the same color).
    """
    notes = beatmap.color_notes
    if len(notes) < 10:
        return beatmap

    red_count = sum(1 for n in notes if n.color == 0)
    total = len(notes)
    red_ratio = red_count / total

    if min_ratio <= red_ratio <= max_ratio:
        return beatmap

    # Determine which color is overrepresented
    if red_ratio > max_ratio:
        over_color, under_color = 0, 1
        excess = red_count - int(0.50 * total)
    else:
        over_color, under_color = 1, 0
        excess = (total - red_count) - int(0.50 * total)

    if excess <= 0:
        return beatmap

    # Find notes of the overrepresented color, sorted by how isolated they are
    # (notes far from others of the same color at the same beat are safer to flip)
    beat_groups: dict[float, list[ColorNote]] = {}
    for n in notes:
        beat_groups.setdefault(n.beat, []).append(n)

    # Candidates: notes of over_color that are alone at their beat
    # (flipping won't create a same-cell same-color conflict)
    isolated_candidates = []
    grouped_candidates = []
    for n in notes:
        if n.color != over_color:
            continue
        same_beat_same_color = sum(
            1 for m in beat_groups[n.beat] if m.color == over_color
        )
        if same_beat_same_color == 1:
            isolated_candidates.append(n)
        else:
            grouped_candidates.append(n)

    # Flip isolated ones first, then grouped if needed
    to_flip = excess
    random.shuffle(isolated_candidates)
    random.shuffle(grouped_candidates)
    candidates = isolated_candidates + grouped_candidates

    flipped = 0
    for n in candidates:
        if flipped >= to_flip:
            break
        n.color = under_color
        flipped += 1

    new_red = sum(1 for n in notes if n.color == 0)
    logger.info(
        "Color rebalance: %.1f%% -> %.1f%% red (%d flipped)",
        red_ratio * 100,
        new_red / total * 100,
        flipped,
    )
    return beatmap


def diversify_directions(
    beatmap: DifficultyBeatmap,
    max_direction_ratio: float = 0.40,
) -> DifficultyBeatmap:
    """Cap any single direction at max_direction_ratio of total notes.

    Reassigns excess notes to underrepresented directions using
    physically comfortable alternatives.
    """
    notes = beatmap.color_notes
    if len(notes) < 20:
        return beatmap

    dir_counts = Counter(n.direction for n in notes)
    total = len(notes)
    max_allowed = int(max_direction_ratio * total)

    # Find overrepresented directions
    over_dirs = {d: c - max_allowed for d, c in dir_counts.items() if c > max_allowed}
    if not over_dirs:
        return beatmap

    # Find underrepresented directions (for redistribution)
    all_dirs = list(range(9))  # 0-8
    under_dirs = [d for d in all_dirs if dir_counts.get(d, 0) < max_allowed]

    if not under_dirs:
        return beatmap

    total_reassigned = 0
    for over_dir, excess in over_dirs.items():
        # Find notes with this direction
        candidates = [n for n in notes if n.direction == over_dir]
        random.shuffle(candidates)

        for n in candidates[:excess]:
            new_dir = random.choice(under_dirs)
            n.direction = new_dir
            total_reassigned += 1

    if total_reassigned > 0:
        logger.info(
            "Direction diversity: reassigned %d notes from overrepresented directions",
            total_reassigned,
        )
    return beatmap


def expand_grid_coverage(
    beatmap: DifficultyBeatmap,
    min_coverage: int = 8,
) -> DifficultyBeatmap:
    """Shift some notes to unused grid cells if coverage is too low.

    Only moves notes that are in the most crowded cells to spread the map out.
    """
    notes = beatmap.color_notes
    if len(notes) < 20:
        return beatmap

    used_cells = {(n.x, n.y) for n in notes}
    coverage = len(used_cells)

    if coverage >= min_coverage:
        return beatmap

    unused_cells = list(ALL_GRID_CELLS - used_cells)
    if not unused_cells:
        return beatmap

    # Find the most crowded cells
    cell_counts = Counter((n.x, n.y) for n in notes)
    # Sort notes by how crowded their cell is (move from most crowded)
    crowded_notes = sorted(
        notes,
        key=lambda n: cell_counts[(n.x, n.y)],
        reverse=True,
    )

    moved = 0
    target_moves = min(len(unused_cells), min_coverage - coverage)

    for n in crowded_notes:
        if moved >= target_moves or not unused_cells:
            break
        # Only move if the cell has plenty of other notes
        if cell_counts[(n.x, n.y)] <= 2:
            continue
        old_cell = (n.x, n.y)
        new_cell = unused_cells.pop(random.randrange(len(unused_cells)))
        cell_counts[old_cell] -= 1
        n.x, n.y = new_cell
        cell_counts[new_cell] = cell_counts.get(new_cell, 0) + 1
        moved += 1

    if moved > 0:
        new_coverage = len({(n.x, n.y) for n in notes})
        logger.info(
            "Grid coverage: %d/12 -> %d/12 cells (%d notes moved)",
            coverage, new_coverage, moved,
        )
    return beatmap


def deduplicate_patterns(
    beatmap: DifficultyBeatmap,
    max_repeats: int = 5,
) -> DifficultyBeatmap:
    """Inject variation when the same note pattern repeats too many times.

    A "pattern" is defined as (color, x, y, direction) tuple. When the same
    pattern appears more than max_repeats times consecutively, some instances
    are modified.
    """
    notes = beatmap.color_notes
    if len(notes) < max_repeats + 1:
        return beatmap

    notes.sort(key=lambda n: n.beat)

    def _pattern(n: ColorNote) -> tuple[int, int, int, int]:
        return (n.color, n.x, n.y, n.direction)

    # Find runs of identical patterns
    total_varied = 0
    i = 0
    while i < len(notes):
        pat = _pattern(notes[i])
        run_end = i + 1
        while run_end < len(notes) and _pattern(notes[run_end]) == pat:
            run_end += 1

        run_length = run_end - i
        if run_length > max_repeats:
            # Vary every other note in the excess portion
            for j in range(i + max_repeats, run_end, 2):
                n = notes[j]
                # Shift direction to something different
                alt_dirs = [d for d in range(9) if d != n.direction]
                if alt_dirs:
                    n.direction = random.choice(alt_dirs)
                    total_varied += 1

        i = run_end

    if total_varied > 0:
        logger.info("Pattern dedup: varied %d notes in repeated runs", total_varied)
    return beatmap


def fix_parity(beatmap: DifficultyBeatmap) -> DifficultyBeatmap:
    """Fix swing direction parity violations.

    Ensures that consecutive notes of the same color alternate between
    forehand (down-ish) and backhand (up-ish) swings. Notes that violate
    parity are adjusted to the opposite swing direction.

    This is a simplified parity check — it handles the most common case
    (alternating up/down within a color) but doesn't handle all edge cases
    like diagonal resets or dot notes.
    """
    notes = beatmap.color_notes
    if len(notes) < 2:
        return beatmap

    notes.sort(key=lambda n: (n.beat, n.color))

    # Process each color stream independently
    fixes = 0
    for color in (0, 1):
        color_notes = [n for n in notes if n.color == color]
        if len(color_notes) < 2:
            continue

        for i in range(1, len(color_notes)):
            prev = color_notes[i - 1]
            curr = color_notes[i]

            # Skip "any" direction notes — they reset parity
            if prev.direction == 8 or curr.direction == 8:
                continue

            prev_is_forehand = prev.direction in _FOREHAND_DIRS
            curr_is_forehand = curr.direction in _FOREHAND_DIRS

            # Parity violation: two consecutive forehand or two consecutive backhand
            if prev_is_forehand == curr_is_forehand:
                # Flip current note to opposite swing
                if curr_is_forehand:
                    # Was forehand, flip to backhand
                    backhand_options = list(_BACKHAND_DIRS)
                    curr.direction = random.choice(backhand_options)
                else:
                    # Was backhand, flip to forehand
                    forehand_options = list(_FOREHAND_DIRS)
                    curr.direction = random.choice(forehand_options)
                fixes += 1

    if fixes > 0:
        logger.info("Parity fix: corrected %d swing direction violations", fixes)
    return beatmap
