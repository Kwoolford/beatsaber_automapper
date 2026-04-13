"""Post-processing pipeline for generated Beat Saber beatmaps.

Applies rule-based fixes to improve playability and flow of model output.
Each step operates on a DifficultyBeatmap in-place and returns it.

Pipeline order (each step depends on clean input from previous):
    1. Cap non-note objects — keep bombs/walls/arcs/chains at reasonable density
    2. Max notes per beat — cap to difficulty limit (Expert=2, E+=3)
    3. NPS enforcement — thin notes if density exceeds difficulty target
    4. Color rebalancing — push toward 45-55% red/blue split
    5. Color separation — red to left cols, blue to right cols
    6. Remove unplayable patterns — overlaps, vision blocks, 180° flips
    7. Convert dot notes — replace direction 8 with proper swing directions
    8. Fix parity — flow-aware forehand/backhand alternation with look-ahead
    9. Fix arc/chain connectivity — connect arcs/chains to actual notes
   10. Lighting postprocess — deduplicate, cap density, smooth brightness
"""

from __future__ import annotations

import logging
import random
from collections import Counter

from beatsaber_automapper.data.beatmap import (
    BasicEvent,
    BurstSlider,
    ColorNote,
    DifficultyBeatmap,
    Slider,
)

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

    beatmap = cap_non_note_objects(beatmap, song_duration_secs)
    beatmap = enforce_max_notes_per_beat(beatmap, difficulty)
    beatmap = enforce_nps(beatmap, difficulty, bpm, song_duration_secs)
    beatmap = rebalance_colors(beatmap)
    beatmap = enforce_color_separation(beatmap)
    beatmap = remove_unplayable_patterns(beatmap, bpm)
    beatmap = convert_dot_notes(beatmap)
    beatmap = fix_parity(beatmap)
    beatmap = fix_arc_chain_connectivity(beatmap, bpm)
    beatmap = postprocess_lighting(beatmap, bpm)

    n_after = len(beatmap.color_notes)
    logger.info(
        "Post-processing: %d -> %d notes, difficulty=%s",
        n_before, n_after, difficulty,
    )
    return beatmap


def strip_non_note_objects(beatmap: DifficultyBeatmap) -> DifficultyBeatmap:
    """Remove all bombs, walls, arcs, and chains — notes only."""
    removed = (
        len(beatmap.bomb_notes) + len(beatmap.obstacles)
        + len(beatmap.sliders) + len(beatmap.burst_sliders)
    )
    beatmap.bomb_notes = []
    beatmap.obstacles = []
    beatmap.sliders = []
    beatmap.burst_sliders = []
    if removed > 0:
        logger.info("Stripped %d non-note objects (bombs/walls/arcs/chains)", removed)
    return beatmap


def cap_non_note_objects(
    beatmap: DifficultyBeatmap,
    song_duration_secs: float | None = None,
) -> DifficultyBeatmap:
    """Cap bombs, walls, arcs, and chains to reasonable densities.

    Undertrained models produce excessive non-note objects that clutter the map.
    This removes the excess, keeping only the most evenly spaced instances.
    """
    duration = song_duration_secs or 180.0

    # Max density per second for each object type
    limits = {
        "bombs": int(max(10, duration * 0.15)),      # ~0.15/sec
        "walls": int(max(5, duration * 0.10)),        # ~0.10/sec
        "arcs": int(max(5, duration * 0.05)),         # ~0.05/sec
        "chains": int(max(5, duration * 0.05)),       # ~0.05/sec
    }

    def _thin(items: list, max_count: int, label: str) -> list:
        if len(items) <= max_count:
            return items
        # Keep evenly spaced items by beat
        items.sort(key=lambda x: x.beat)
        step = len(items) / max_count
        kept = [items[int(i * step)] for i in range(max_count)]
        logger.info("Cap %s: %d -> %d", label, len(items), len(kept))
        return kept

    beatmap.bomb_notes = _thin(beatmap.bomb_notes, limits["bombs"], "bombs")
    beatmap.obstacles = _thin(beatmap.obstacles, limits["walls"], "walls")
    beatmap.sliders = _thin(beatmap.sliders, limits["arcs"], "arcs")
    beatmap.burst_sliders = _thin(beatmap.burst_sliders, limits["chains"], "chains")

    return beatmap


def enforce_max_notes_per_beat(
    beatmap: DifficultyBeatmap,
    difficulty: str = "Expert",
) -> DifficultyBeatmap:
    """Cap notes per beat to difficulty-appropriate limits.

    Expert allows max 2 notes per beat, ExpertPlus allows max 3.
    When over the limit, removes notes that are least ergonomic
    (same-color duplicates first, then furthest from center).
    """
    notes = beatmap.color_notes
    if not notes:
        return beatmap

    max_per_beat = 3 if difficulty == "ExpertPlus" else 2
    max_per_color = 2 if difficulty == "ExpertPlus" else 1

    # Group by beat
    beat_groups: dict[float, list[ColorNote]] = {}
    for n in notes:
        beat_groups.setdefault(n.beat, []).append(n)

    removed_total = 0
    kept_notes: list[ColorNote] = []

    for beat in sorted(beat_groups.keys()):
        group = beat_groups[beat]

        if len(group) <= max_per_beat:
            # Check same-color duplicates within limit
            color_counts: dict[int, list[ColorNote]] = {}
            for n in group:
                color_counts.setdefault(n.color, []).append(n)

            beat_kept: list[ColorNote] = []
            for color, cnotes in color_counts.items():
                if len(cnotes) <= max_per_color:
                    beat_kept.extend(cnotes)
                else:
                    # Keep the ones closest to their color's preferred side
                    # Red prefers cols 0-1, Blue prefers cols 2-3
                    preferred_center = 0.5 if color == 0 else 2.5
                    cnotes.sort(key=lambda n: abs(n.x - preferred_center))
                    beat_kept.extend(cnotes[:max_per_color])
                    removed_total += len(cnotes) - max_per_color

            kept_notes.extend(beat_kept)
        else:
            # Too many notes total — thin to max_per_beat
            # Prefer keeping one of each color, then pick by ergonomic position
            by_color: dict[int, list[ColorNote]] = {}
            for n in group:
                by_color.setdefault(n.color, []).append(n)

            beat_kept: list[ColorNote] = []
            for color in sorted(by_color.keys()):
                cnotes = by_color[color]
                preferred_center = 0.5 if color == 0 else 2.5
                cnotes.sort(key=lambda n: abs(n.x - preferred_center))
                # Keep up to max_per_color per color, up to max_per_beat total
                for n in cnotes[:max_per_color]:
                    if len(beat_kept) < max_per_beat:
                        beat_kept.append(n)

            removed_total += len(group) - len(beat_kept)
            kept_notes.extend(beat_kept)

    if removed_total > 0:
        beatmap.color_notes = kept_notes
        logger.info(
            "Max notes/beat: removed %d excess notes (max %d/beat for %s)",
            removed_total, max_per_beat, difficulty,
        )
    return beatmap


def enforce_color_separation(
    beatmap: DifficultyBeatmap,
) -> DifficultyBeatmap:
    """Push red notes toward left (cols 0-1) and blue toward right (cols 2-3).

    Only moves notes that are on the "wrong" side and not at the same beat
    as another note of the opposite color in the target position.
    """
    notes = beatmap.color_notes
    if len(notes) < 10:
        return beatmap

    # Group by beat for collision checking
    beat_groups: dict[float, list[ColorNote]] = {}
    for n in notes:
        beat_groups.setdefault(n.beat, []).append(n)

    moved = 0
    for n in notes:
        # Red in cols 2-3 → try to move to cols 0-1
        if n.color == 0 and n.x >= 2:
            group = beat_groups[n.beat]
            occupied = {(m.x, m.y) for m in group if m is not n}
            # Try col 1, then col 0 (same row)
            for target_col in (1, 0):
                if (target_col, n.y) not in occupied:
                    n.x = target_col
                    moved += 1
                    break
        # Blue in cols 0-1 → try to move to cols 2-3
        elif n.color == 1 and n.x <= 1:
            group = beat_groups[n.beat]
            occupied = {(m.x, m.y) for m in group if m is not n}
            for target_col in (2, 3):
                if (target_col, n.y) not in occupied:
                    n.x = target_col
                    moved += 1
                    break

    if moved > 0:
        logger.info("Color separation: moved %d notes to preferred side", moved)
    return beatmap


def _beat_strength(beat: float) -> float:
    """Score how musically strong a beat position is (0.0-1.0).

    Downbeats (whole beats) are strongest, then half-beats, quarter-beats, etc.
    """
    # Check from strongest to weakest subdivision
    frac = beat % 1.0
    if abs(frac) < 0.01 or abs(frac - 1.0) < 0.01:
        return 1.0  # Downbeat (whole beat)
    if abs(frac - 0.5) < 0.01:
        return 0.75  # Half-beat
    if abs(frac - 0.25) < 0.01 or abs(frac - 0.75) < 0.01:
        return 0.5  # Quarter-beat
    return 0.25  # Offbeat / subdivision


def _note_importance(
    note: ColorNote,
    idx: int,
    notes: list[ColorNote],
) -> float:
    """Score a note's musical importance for NPS thinning (higher = keep).

    Considers beat strength, gap impact, and color pairing.
    """
    score = 0.0

    # 1. Beat strength: notes on strong beats are more important
    score += _beat_strength(note.beat) * 3.0

    # 2. Gap penalty: removing this note would create a large gap — keep it
    prev_beat = notes[idx - 1].beat if idx > 0 else note.beat - 2.0
    next_beat = notes[idx + 1].beat if idx < len(notes) - 1 else note.beat + 2.0
    gap_if_removed = next_beat - prev_beat
    if gap_if_removed > 2.0:
        score += 2.0  # Large gap — strongly prefer keeping
    elif gap_if_removed > 1.0:
        score += 1.0

    # 3. Paired notes: notes that share a beat with their opposite color
    #    are more important (red+blue pairs are the bread and butter of mapping)
    has_pair = False
    for j in range(max(0, idx - 3), min(len(notes), idx + 4)):
        if j != idx and abs(notes[j].beat - note.beat) < 0.01 and notes[j].color != note.color:
            has_pair = True
            break
    if has_pair:
        score += 1.5

    # 4. First and last notes of the song get a boost
    if idx < 2 or idx >= len(notes) - 2:
        score += 2.0

    return score


def enforce_nps(
    beatmap: DifficultyBeatmap,
    difficulty: str = "Expert",
    bpm: float = 120.0,
    song_duration_secs: float | None = None,
) -> DifficultyBeatmap:
    """Remove excess notes to bring NPS within the target range for the difficulty.

    Uses importance-based scoring to preserve musically significant notes:
    downbeats over offbeats, paired notes over solo notes, and notes that
    would create large gaps if removed.
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

    # Target the midpoint of the acceptable range
    target_nps = (min_nps + max_nps) / 2.0
    target_count = max(1, int(target_nps * song_duration_secs))

    if target_count >= len(notes):
        return beatmap

    # Sort by beat time
    notes.sort(key=lambda n: n.beat)

    # Score each note by musical importance
    scored = [(i, _note_importance(notes[i], i, notes)) for i in range(len(notes))]

    # Sort by importance ascending — remove least important first
    scored.sort(key=lambda x: x[1])

    # Mark the least important notes for removal
    n_to_remove = len(notes) - target_count
    remove_indices = {scored[i][0] for i in range(n_to_remove)}

    kept = [n for i, n in enumerate(notes) if i not in remove_indices]

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
    max_direction_ratio: float = 0.35,
) -> DifficultyBeatmap:
    """Push direction distribution toward training data targets.

    Target distribution (from training data analysis):
        Down (1): ~25%, Up (0): ~20%, Left (2): ~15%, Right (3): ~15%,
        Diagonals (4-7): ~5% each, Any (8): ~5%

    Only reassigns from overrepresented directions, preferring physically
    comfortable alternatives (forehand <-> backhand).
    """
    notes = beatmap.color_notes
    if len(notes) < 20:
        return beatmap

    # Target distribution from training data
    target_ratios = {
        1: 0.25,  # down
        0: 0.20,  # up
        2: 0.15,  # left
        3: 0.15,  # right
        4: 0.05,  # up-left
        5: 0.05,  # up-right
        6: 0.05,  # down-left
        7: 0.05,  # down-right
        8: 0.05,  # any
    }

    dir_counts = Counter(n.direction for n in notes)
    total = len(notes)

    # Cap each direction at its target + margin
    margin = 0.10  # Allow 10% above target before redistributing
    total_reassigned = 0

    for d in range(9):
        target = target_ratios.get(d, 0.05)
        max_allowed = int((target + margin) * total)
        current = dir_counts.get(d, 0)

        if current <= max_allowed:
            continue

        excess = current - max_allowed

        # Find underrepresented directions to redistribute to
        under_dirs = [
            dd for dd in range(9)
            if dir_counts.get(dd, 0) < int(target_ratios.get(dd, 0.05) * total)
        ]
        if not under_dirs:
            # Fall back to any direction below its cap
            under_dirs = [
                dd for dd in range(9)
                if dd != d
                and dir_counts.get(dd, 0) < int(
                    (target_ratios.get(dd, 0.05) + margin) * total
                )
            ]
        if not under_dirs:
            continue

        candidates = [n for n in notes if n.direction == d]
        random.shuffle(candidates)

        for n in candidates[:excess]:
            new_dir = random.choice(under_dirs)
            dir_counts[d] -= 1
            dir_counts[new_dir] = dir_counts.get(new_dir, 0) + 1
            n.direction = new_dir
            total_reassigned += 1

    if total_reassigned > 0:
        logger.info(
            "Direction diversity: reassigned %d notes toward target distribution",
            total_reassigned,
        )
    return beatmap


def expand_grid_coverage(
    beatmap: DifficultyBeatmap,
    min_coverage: int = 8,
) -> DifficultyBeatmap:
    """Redistribute notes toward training data row distribution.

    Training data distribution: ~47% bottom (y=0), ~28% mid (y=1), ~25% top (y=2).
    Only redistributes if the model output is extremely skewed (>70% in one row).
    Also spreads notes to unused grid cells when coverage is too low.
    """
    notes = beatmap.color_notes
    if len(notes) < 20:
        return beatmap

    # Target row distribution from training data
    target_row_dist = {0: 0.47, 1: 0.28, 2: 0.25}
    total = len(notes)

    # Check current row distribution
    row_counts = Counter(n.y for n in notes)
    row_ratios = {y: row_counts.get(y, 0) / total for y in range(3)}

    # Only redistribute if severely skewed (>70% in any single row)
    max_ratio = max(row_ratios.values())
    if max_ratio > 0.70:
        over_row = max(row_ratios, key=row_ratios.get)
        target_count = {y: int(target_row_dist[y] * total) for y in range(3)}

        # Move excess notes from overrepresented row to underrepresented rows
        excess = row_counts.get(over_row, 0) - target_count[over_row]
        under_rows = [y for y in range(3) if row_counts.get(y, 0) < target_count[y]]

        if excess > 0 and under_rows:
            candidates = [n for n in notes if n.y == over_row]
            random.shuffle(candidates)

            moved = 0
            for n in candidates:
                if moved >= excess or not under_rows:
                    break
                new_row = under_rows[moved % len(under_rows)]
                n.y = new_row
                moved += 1

            logger.info(
                "Grid row rebalance: moved %d notes from row %d "
                "(%.0f%% -> target ~%.0f%%)",
                moved, over_row, max_ratio * 100,
                target_row_dist[over_row] * 100,
            )

    # Also spread to unused cells if coverage is too low
    used_cells = {(n.x, n.y) for n in notes}
    coverage = len(used_cells)

    if coverage < min_coverage:
        unused_cells = list(ALL_GRID_CELLS - used_cells)
        if unused_cells:
            cell_counts = Counter((n.x, n.y) for n in notes)
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


def remove_unplayable_patterns(
    beatmap: DifficultyBeatmap,
    bpm: float = 120.0,
) -> DifficultyBeatmap:
    """Remove or fix notes that create unplayable patterns.

    Detects and fixes:
    1. Vision blocks — note directly behind another at the same beat
    2. Handclaps — same color, same position, same beat
    3. Stacked notes — different color, same position, same beat (overlap)
    4. Rapid impossible angles — >180 degree direction change within 100ms

    Args:
        beatmap: The beatmap to check.
        bpm: Song BPM for timing checks.

    Returns:
        The same beatmap with unplayable patterns removed.
    """
    notes = beatmap.color_notes
    if len(notes) < 2:
        return beatmap

    notes.sort(key=lambda n: (n.beat, n.x, n.y))

    # Group notes by beat
    beat_groups: dict[float, list[ColorNote]] = {}
    for n in notes:
        beat_groups.setdefault(n.beat, []).append(n)

    to_remove: set[int] = set()

    for beat, group in beat_groups.items():
        positions = {}
        for i, n in enumerate(group):
            pos = (n.x, n.y)
            if pos in positions:
                # Same position at same beat — remove the duplicate
                # Keep the first one, remove later ones
                orig_idx = notes.index(n)
                to_remove.add(orig_idx)
            else:
                positions[pos] = n

        # Vision blocks: note at (x, y) with direction pointing away from
        # another note at (x, y-1) or (x, y+1) at the same beat
        for n in group:
            if n.direction == 0:  # up
                # Check if there's a note directly above that blocks vision
                blocker = next(
                    (m for m in group if m.x == n.x and m.y == n.y + 1 and m is not n),
                    None,
                )
                if blocker:
                    # Move the blocker to a different column
                    alt_cols = [c for c in range(4) if c != n.x]
                    if alt_cols:
                        blocker.x = random.choice(alt_cols)

    # Remove flagged notes
    if to_remove:
        beatmap.color_notes = [n for i, n in enumerate(notes) if i not in to_remove]
        logger.info("Removed %d unplayable notes (overlaps/vision blocks)", len(to_remove))

    # Check rapid direction changes (>180 degree within 100ms)
    notes = beatmap.color_notes
    notes.sort(key=lambda n: (n.beat, n.color))
    ms_per_beat = 60000.0 / max(bpm, 1)
    fixes = 0

    for color in (0, 1):
        color_notes = [n for n in notes if n.color == color]
        for i in range(1, len(color_notes)):
            prev_n = color_notes[i - 1]
            curr_n = color_notes[i]
            gap_ms = (curr_n.beat - prev_n.beat) * ms_per_beat

            if gap_ms < 100 and gap_ms > 0:
                # Very rapid succession — ensure directions are compatible
                # "Opposite" directions: up<->down, left<->right
                opposites = {0: 1, 1: 0, 2: 3, 3: 2, 4: 7, 7: 4, 5: 6, 6: 5}
                if opposites.get(prev_n.direction) == curr_n.direction:
                    # 180 degree flip in <100ms — fix by using same direction
                    curr_n.direction = prev_n.direction
                    fixes += 1

    if fixes > 0:
        logger.info("Fixed %d rapid impossible angle changes (<100ms)", fixes)

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


def _choose_flow_direction(
    need_forehand: bool,
    curr_x: int,
    next_x: int | None = None,
    curr_y: int = 0,
) -> int:
    """Choose a swing direction based on parity, position, and flow.

    Uses grid position to create natural swing variety:
    - Edge columns swing inward (col 0 → right-leaning, col 3 → left-leaning)
    - Center columns swing straight or follow flow to next note
    - Next-note flow takes priority when there's significant lateral movement
    - Mixes straight and diagonal for variety based on position

    Args:
        need_forehand: True if this swing should be forehand (down-ish).
        curr_x: Current note's column (0-3).
        next_x: Next same-color note's column, or None if unknown.
        curr_y: Current note's row (0-2).

    Returns:
        Direction integer (0-8).
    """
    # Flow to next note takes priority for lateral movement
    if next_x is not None:
        dx = next_x - curr_x
        if abs(dx) >= 2:
            if need_forehand:
                return 7 if dx > 0 else 6  # down-right or down-left
            else:
                return 5 if dx > 0 else 4  # up-right or up-left
        if abs(dx) == 1:
            # Moderate lateral: use diagonal ~50% of the time based on position
            if (curr_x + curr_y) % 2 == 0:  # deterministic variety
                if need_forehand:
                    return 7 if dx > 0 else 6
                else:
                    return 5 if dx > 0 else 4

    # Position-aware defaults: edge columns swing inward
    if curr_x == 0:
        # Far left → swing right-leaning
        if need_forehand:
            return 7 if curr_y % 2 == 0 else 1  # down-right or straight down
        else:
            return 5 if curr_y % 2 == 0 else 0  # up-right or straight up
    elif curr_x == 3:
        # Far right → swing left-leaning
        if need_forehand:
            return 6 if curr_y % 2 == 0 else 1  # down-left or straight down
        else:
            return 4 if curr_y % 2 == 0 else 0  # up-left or straight up
    elif curr_x == 1:
        # Center-left: mix of straight and slight right-lean
        if need_forehand:
            return 1 if curr_y != 2 else 7  # straight down or down-right at top
        else:
            return 0 if curr_y != 0 else 5  # straight up or up-right at bottom
    else:  # curr_x == 2
        # Center-right: mix of straight and slight left-lean
        if need_forehand:
            return 1 if curr_y != 2 else 6  # straight down or down-left at top
        else:
            return 0 if curr_y != 0 else 4  # straight up or up-left at bottom


def convert_dot_notes(beatmap: DifficultyBeatmap) -> DifficultyBeatmap:
    """Convert direction 8 (any/dot) notes to proper swing directions.

    Dot notes (direction 8) are overused by the model (14.9% vs ~5% target).
    Convert them to real directional notes based on:
    - Parity context (what the previous same-color note direction was)
    - Flow to the next same-color note position

    Preserves a small percentage (~5%) of dots at natural rest points
    (after long gaps of 2+ beats with no same-color notes).
    """
    notes = beatmap.color_notes
    if not notes:
        return beatmap

    notes.sort(key=lambda n: (n.beat, n.color))
    converted = 0

    for color in (0, 1):
        color_notes = [n for n in notes if n.color == color]
        if not color_notes:
            continue

        for i, n in enumerate(color_notes):
            if n.direction != 8:
                continue

            # Keep dots after long gaps (2+ beats) — these are natural rest points
            if i > 0:
                gap = n.beat - color_notes[i - 1].beat
                if gap >= 2.0:
                    continue  # keep as dot — parity reset point

            # Determine parity from previous note
            prev_dir = None
            if i > 0:
                prev_dir = color_notes[i - 1].direction

            if prev_dir is None or prev_dir == 8:
                # No parity context — default to forehand (down)
                need_forehand = True
            elif prev_dir in _FOREHAND_DIRS:
                need_forehand = False  # alternate to backhand
            elif prev_dir in _BACKHAND_DIRS:
                need_forehand = True  # alternate to forehand
            else:
                # Horizontal or unknown — default forehand
                need_forehand = True

            # Get next note position for flow
            next_x = color_notes[i + 1].x if i + 1 < len(color_notes) else None

            n.direction = _choose_flow_direction(need_forehand, n.x, next_x, n.y)
            converted += 1

    if converted > 0:
        logger.info("Dot note conversion: converted %d direction-8 to real directions", converted)
    return beatmap


def fix_parity(beatmap: DifficultyBeatmap) -> DifficultyBeatmap:
    """Fix swing direction parity with flow-aware look-ahead.

    Ensures consecutive notes of the same color alternate between forehand
    (down-ish) and backhand (up-ish) swings. Uses look-ahead to the NEXT
    note to choose directions that create natural flow.

    Key design: prefers straight up/down. Only uses diagonal directions
    when the next same-color note is 2+ columns away laterally.

    Parity groups:
        - Forehand (down-ish): {1=down, 6=down-left, 7=down-right}
        - Backhand (up-ish):   {0=up, 4=up-left, 5=up-right}
        - Neutral:             {2=left, 3=right, 8=any} — don't violate parity
    """
    notes = beatmap.color_notes
    if len(notes) < 2:
        return beatmap

    notes.sort(key=lambda n: (n.beat, n.color))

    fixes = 0
    for color in (0, 1):
        color_notes = [n for n in notes if n.color == color]
        if len(color_notes) < 2:
            continue

        # First note: ensure it starts with a forehand (down) if it's not neutral
        first = color_notes[0]
        if first.direction in _BACKHAND_DIRS:
            # Unusual to start backhand — but allow if model wants it
            pass

        for i in range(1, len(color_notes)):
            prev = color_notes[i - 1]
            curr = color_notes[i]

            # Neutral directions don't violate parity — skip
            if prev.direction in (2, 3, 8) or curr.direction in (2, 3, 8):
                continue

            prev_is_forehand = prev.direction in _FOREHAND_DIRS
            curr_is_forehand = curr.direction in _FOREHAND_DIRS

            # Parity violation: two consecutive same-parity swings
            if prev_is_forehand == curr_is_forehand:
                need_forehand = not curr_is_forehand

                # Look ahead to next note for flow direction
                next_x = None
                if i + 1 < len(color_notes):
                    next_x = color_notes[i + 1].x

                curr.direction = _choose_flow_direction(
                    need_forehand, curr.x, next_x, curr.y
                )
                fixes += 1

    if fixes > 0:
        logger.info("Parity fix: corrected %d swing direction violations", fixes)
    return beatmap


def fix_arc_chain_connectivity(
    beatmap: DifficultyBeatmap,
    bpm: float = 120.0,
) -> DifficultyBeatmap:
    """Connect arcs and chains to actual notes in the beatmap.

    Arcs (sliders) should connect from one note to the next note of the
    same color. Chains (burst sliders) should have their tail at the next
    note's position. This fixes cases where the model generates arc/chain
    events with random tail positions that don't match actual notes.

    Also removes orphaned arcs/chains that have no nearby notes.
    """
    notes = beatmap.color_notes
    if not notes:
        # No notes to connect to — remove all arcs and chains
        if beatmap.sliders or beatmap.burst_sliders:
            removed = len(beatmap.sliders) + len(beatmap.burst_sliders)
            beatmap.sliders = []
            beatmap.burst_sliders = []
            logger.info("Removed %d orphaned arcs/chains (no notes)", removed)
        return beatmap

    notes.sort(key=lambda n: n.beat)

    # Build lookup: for each color, sorted list of notes
    notes_by_color: dict[int, list[ColorNote]] = {0: [], 1: []}
    for n in notes:
        if n.color in notes_by_color:
            notes_by_color[n.color].append(n)

    def _find_next_note(color: int, after_beat: float) -> ColorNote | None:
        """Find the next note of the given color after the given beat."""
        cnotes = notes_by_color.get(color, [])
        for n in cnotes:
            if n.beat > after_beat + 0.01:  # small epsilon to avoid same-beat
                return n
        return None

    def _find_nearest_note(
        color: int, beat: float, max_beats: float = 2.0,
    ) -> ColorNote | None:
        """Find the nearest note of the given color within max_beats."""
        cnotes = notes_by_color.get(color, [])
        best = None
        best_dist = max_beats
        for n in cnotes:
            dist = abs(n.beat - beat)
            if dist < best_dist:
                best_dist = dist
                best = n
        return best

    # Fix sliders (arcs): connect head to the nearest note, tail to next note
    fixed_sliders: list[Slider] = []
    arc_fixes = 0
    for s in beatmap.sliders:
        # Find a note at/near the head position
        head_note = _find_nearest_note(s.color, s.beat, max_beats=1.0)
        if head_note is None:
            continue  # orphaned — drop it

        # Find the next note of the same color for the tail
        tail_note = _find_next_note(s.color, head_note.beat)
        if tail_note is None:
            continue  # no tail target — drop it

        # Update arc to connect head note → tail note
        s.beat = head_note.beat
        s.x = head_note.x
        s.y = head_note.y
        s.direction = head_note.direction
        s.tail_beat = tail_note.beat
        s.tail_x = tail_note.x
        s.tail_y = tail_note.y
        s.tail_direction = tail_note.direction
        fixed_sliders.append(s)
        arc_fixes += 1

    beatmap.sliders = fixed_sliders

    # Fix burst sliders (chains): connect to next note
    fixed_chains: list[BurstSlider] = []
    chain_fixes = 0
    for bs in beatmap.burst_sliders:
        # Find a note at/near the head
        head_note = _find_nearest_note(bs.color, bs.beat, max_beats=1.0)
        if head_note is None:
            continue  # orphaned

        # Find next note for the tail
        tail_note = _find_next_note(bs.color, head_note.beat)
        if tail_note is None:
            continue  # no tail target

        # Update chain to connect head → tail
        bs.beat = head_note.beat
        bs.x = head_note.x
        bs.y = head_note.y
        bs.direction = head_note.direction
        bs.tail_beat = tail_note.beat
        bs.tail_x = tail_note.x
        bs.tail_y = tail_note.y
        fixed_chains.append(bs)
        chain_fixes += 1

    orig_arcs = len(beatmap.sliders)
    orig_chains = len(beatmap.burst_sliders)
    beatmap.sliders = fixed_sliders
    beatmap.burst_sliders = fixed_chains

    if arc_fixes > 0 or chain_fixes > 0 or orig_arcs > 0 or orig_chains > 0:
        logger.info(
            "Arc/chain connectivity: %d->%d arcs, %d->%d chains",
            orig_arcs, len(fixed_sliders),
            orig_chains, len(fixed_chains),
        )
    return beatmap


# ---------------------------------------------------------------------------
# Lighting post-processing
# ---------------------------------------------------------------------------

_MAX_LIGHT_EVENTS_PER_BEAT = 4
_MAX_BRIGHTNESS_JUMP = 0.6


def postprocess_lighting(
    beatmap: DifficultyBeatmap,
    bpm: float = 120.0,
) -> DifficultyBeatmap:
    """Post-process lighting events for quality.

    Steps:
        1. Deduplicate events at same beat + event_type
        2. Cap events per beat to prevent strobing
        3. Smooth brightness transitions (no jarring jumps)

    Args:
        beatmap: Beatmap with lighting events to clean up.
        bpm: BPM for timing context.

    Returns:
        The same beatmap, modified in-place.
    """
    events = beatmap.basic_events
    if not events:
        return beatmap

    n_before = len(events)

    # Step 1: Deduplicate — keep last event per (beat, event_type)
    seen: dict[tuple[float, int], int] = {}
    for i, ev in enumerate(events):
        key = (round(ev.beat, 4), ev.event_type)
        seen[key] = i  # last occurrence wins
    events = [events[i] for i in sorted(seen.values())]

    # Step 2: Cap events per beat
    events.sort(key=lambda e: e.beat)
    beat_groups: dict[float, list[BasicEvent]] = {}
    for ev in events:
        rounded = round(ev.beat * 4.0) / 4.0  # quantize to quarter-beat
        beat_groups.setdefault(rounded, []).append(ev)

    capped: list[BasicEvent] = []
    for _beat, group in sorted(beat_groups.items()):
        if len(group) <= _MAX_LIGHT_EVENTS_PER_BEAT:
            capped.extend(group)
        else:
            # Keep events with highest brightness and diverse event types
            seen_types: set[int] = set()
            kept: list[BasicEvent] = []
            group.sort(key=lambda e: e.float_value, reverse=True)
            for ev in group:
                if ev.event_type not in seen_types or len(kept) < _MAX_LIGHT_EVENTS_PER_BEAT:
                    kept.append(ev)
                    seen_types.add(ev.event_type)
                if len(kept) >= _MAX_LIGHT_EVENTS_PER_BEAT:
                    break
            capped.extend(kept)

    # Step 3: Smooth brightness per event_type channel
    capped.sort(key=lambda e: (e.event_type, e.beat))
    by_type: dict[int, list[BasicEvent]] = {}
    for ev in capped:
        by_type.setdefault(ev.event_type, []).append(ev)

    for _et, type_events in by_type.items():
        for i in range(1, len(type_events)):
            prev_brightness = type_events[i - 1].float_value
            curr_brightness = type_events[i].float_value
            jump = curr_brightness - prev_brightness
            if abs(jump) > _MAX_BRIGHTNESS_JUMP:
                # Clamp to max allowed jump in the same direction
                if jump > 0:
                    type_events[i].float_value = min(
                        1.0, prev_brightness + _MAX_BRIGHTNESS_JUMP
                    )
                else:
                    type_events[i].float_value = max(
                        0.0, prev_brightness - _MAX_BRIGHTNESS_JUMP
                    )

    # Reassemble and sort by beat
    capped.sort(key=lambda e: e.beat)
    beatmap.basic_events = capped

    if n_before != len(capped):
        logger.info(
            "Lighting postprocess: %d -> %d events",
            n_before, len(capped),
        )
    return beatmap
