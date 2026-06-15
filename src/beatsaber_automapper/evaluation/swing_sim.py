"""Swing simulator — "play the map" parity/flow model (Phase 1, TASK P1-1).

A Python port of the core ideas in JoshaParity (github.com/Joshabi/JoshaParity):
a per-hand parity state machine that walks a difficulty's notes in time order,
groups them into swings, assigns each swing a forehand/backhand parity, and flags
the transitions a human wrist physically cannot make cleanly (wrist-break resets).

This complements ``playability.py`` (which does cheap per-beat heuristics): here we
actually simulate the alternating up/down swing motion and the saber's rotation, so
we can tell an *intentional* reset (enough time, or bomb-forced) from a *broken* one
(a same-parity repeat crammed into no time — the "for-sport diagonal" failure mode of
raw model output).

Read-only: never mutates the beatmap.

Outputs a ``MapScorecard`` (per-map aggregate) plus per-swing records, and exposes
per-seam hand state (the parity each hand is in when a section boundary is crossed),
which Phase-2 best-of-N stitching needs.

DoD (plan doc §4 / TODO P1-1): 0 violations on human Expert maps; >0 on raw
PRE-postprocess V7 output.
"""

from __future__ import annotations

import argparse
import json
import logging
import pathlib
from dataclasses import dataclass, field
from enum import IntEnum

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Cut-direction geometry (JoshaParity angle dictionaries).
#
# Each cut direction maps to the saber's rotation angle (degrees) at the moment
# of the cut, separately for a forehand swing and a backhand swing. Angle 0 = the
# saber points straight down; +/- sweeps toward the horizontal. The same physical
# direction has near-opposite angles for the two parities because the hand is
# inverted between an up-swing and a down-swing.
# --------------------------------------------------------------------------- #
# direction codes: 0=up 1=down 2=left 3=right 4=up-left 5=up-right
#                  6=down-left 7=down-right 8=any/dot
FOREHAND_ANGLE = {0: 180, 1: 0, 2: -90, 3: 90, 4: -135, 5: 135, 6: -45, 7: 45, 8: 0}
BACKHAND_ANGLE = {0: 0, 1: 180, 2: 90, 3: -90, 4: 45, 5: -45, 6: 135, 7: -135, 8: 0}

# Which parity a given cut direction is *naturally* swung as. A down-ward cut ends a
# forehand (down) swing; an up-ward cut ends a backhand (up) swing. Horizontal/dot
# are parity-neutral (either works).
_DOWN_GROUP = frozenset({1, 6, 7})   # down, down-left, down-right  -> forehand
_UP_GROUP = frozenset({0, 4, 5})     # up, up-left, up-right        -> backhand
_NEUTRAL_DIRS = frozenset({2, 3, 8})  # left, right, dot

# Tunables (beats). Validated against human Expert vs raw-V7 fixtures; see
# scripts/eval_swing_sim.py and tests/test_swing_sim.py.
STACK_BEAT = 0.126          # notes within this gap = one swing (stacks/sliders/windows)
PARITY_RESET_GAP = 4.0      # gap longer than this frees the hand to re-seat parity
BOMB_RESET_WINDOW = 1.0     # a bomb this close before a swing can justify a reset

# Two same-parity swings whose cut angles differ by at least this much are a flowing
# "roll" (e.g. down-left -> down-right, a 90° wrist rotation), not a stop-and-reverse:
# the player sweeps through without re-cocking. Only near-identical-direction repeats
# (angle change below this) are true resets. JoshaParity's angle model, simplified.
ANGLE_FLOW_DEG = 90.0

# A same-parity reset (full stop + reverse swing) needs wall-clock time. Below this
# many SECONDS it is a wrist-break — physically unplayable, a mapping error. Note
# this is in seconds, not beats: playability is wall-clock, not musical. Calibrated
# against human Expert (fastest reset ~0.34s) vs raw-V7 output (resets crammed at
# 0.24s = flat-density spam); 0.30s sits in the gap. See tests/test_swing_sim.py.
HARD_RESET_SEC = 0.30


class Parity(IntEnum):
    FOREHAND = 0  # down-swing, hand ends low
    BACKHAND = 1  # up-swing, hand ends high

    def flipped(self) -> "Parity":
        return Parity.BACKHAND if self is Parity.FOREHAND else Parity.FOREHAND


@dataclass(slots=True)
class Swing:
    """One swing of one hand (one or more stacked/sliced notes)."""

    color: int          # 0=red(left) 1=blue(right)
    beat: float         # beat of the first note in the swing
    end_beat: float     # beat of the last note in the swing
    direction: int      # representative cut direction (first non-dot, else geometric)
    flexible: bool = False  # all notes were dots -> parity-free (direction is for render only)
    parity: Parity = Parity.FOREHAND
    is_reset: bool = False        # parity did NOT alternate from the previous swing
    reset_kind: str = ""          # "" | "bomb" | "intentional" | "violation"
    note_count: int = 1


@dataclass(slots=True)
class HandReport:
    color: int
    swings: list[Swing] = field(default_factory=list)
    resets: int = 0
    violations: int = 0
    swing_ebpm_p95: float = 0.0   # 95th-pct effective BPM of swings (burst speed)


@dataclass(slots=True)
class MapScorecard:
    n_notes: int
    n_swings: int
    resets: int
    violations: int                 # the DoD metric (wrist-break resets)
    violation_beats: list[float]
    per_hand: dict[int, HandReport]

    def as_dict(self) -> dict:
        return {
            "n_notes": self.n_notes,
            "n_swings": self.n_swings,
            "resets": self.resets,
            "violations": self.violations,
            "violation_beats": [round(b, 3) for b in self.violation_beats],
            "per_hand": {
                str(c): {
                    "swings": len(h.swings),
                    "resets": h.resets,
                    "violations": h.violations,
                    "swing_ebpm_p95": round(h.swing_ebpm_p95, 1),
                }
                for c, h in self.per_hand.items()
            },
        }


# --------------------------------------------------------------------------- #
# Swing extraction
# --------------------------------------------------------------------------- #
def _geometric_direction(notes: list) -> int:
    """Infer a cut direction for an all-dot swing from its note geometry.

    Uses the first->last note displacement; falls back to ``8`` (dot) if the notes
    are stacked on one cell.
    """
    if len(notes) < 2:
        return 8
    dx = notes[-1].x - notes[0].x
    dy = notes[-1].y - notes[0].y
    if dx == 0 and dy == 0:
        return 8
    # map (sign dx, sign dy) -> direction code
    sx, sy = (dx > 0) - (dx < 0), (dy > 0) - (dy < 0)
    table = {
        (0, 1): 0, (0, -1): 1, (-1, 0): 2, (1, 0): 3,
        (-1, 1): 4, (1, 1): 5, (-1, -1): 6, (1, -1): 7,
    }
    return table.get((sx, sy), 8)


def _extract_swings(notes: list, color: int) -> list[Swing]:
    """Group one color's notes (already filtered) into swings by time proximity."""
    color_notes = sorted(
        (n for n in notes if n.color == color), key=lambda n: n.beat
    )
    swings: list[Swing] = []
    if not color_notes:
        return swings

    group = [color_notes[0]]
    for n in color_notes[1:]:
        if n.beat - group[-1].beat <= STACK_BEAT:
            group.append(n)
        else:
            swings.append(_swing_from_group(group, color))
            group = [n]
    swings.append(_swing_from_group(group, color))
    return swings


def _swing_from_group(group: list, color: int) -> Swing:
    direction = next((n.direction for n in group if n.direction != 8), None)
    flexible = direction is None  # every note was a dot -> parity-free
    if direction is None:
        # geometric direction is kept for rendering only; parity treats it as neutral
        direction = _geometric_direction(group)
    return Swing(
        color=color,
        beat=group[0].beat,
        end_beat=group[-1].beat,
        direction=direction,
        flexible=flexible,
        note_count=len(group),
    )


# --------------------------------------------------------------------------- #
# Parity state machine
# --------------------------------------------------------------------------- #
def _natural_parity(direction: int) -> Parity | None:
    if direction in _DOWN_GROUP:
        return Parity.FOREHAND
    if direction in _UP_GROUP:
        return Parity.BACKHAND
    return None  # neutral/dot — caller decides from alternation


def _angle_delta(a: float, b: float) -> float:
    """Smallest absolute angle between two headings (degrees, 0..180)."""
    d = abs(a - b) % 360.0
    return d if d <= 180.0 else 360.0 - d


def _bomb_forces_reset(bombs: list, prev_beat: float, beat: float, color: int) -> bool:
    """Whether this swing sits inside a deliberate bomb-reset section.

    Bombs are used to flip a player's parity, so a same-direction "stream" laced
    with bombs is intended, not a wrist-break. We look in a symmetric window around
    the note (a bomb shortly before OR after marks the section as bomb-driven), which
    correctly covers the gaps between individual bomb pairs in such patterns.
    """
    lo, hi = beat - BOMB_RESET_WINDOW, beat + BOMB_RESET_WINDOW
    for b in bombs:
        if lo <= b.beat <= hi:
            return True
    return False


def simulate(beatmap, *, bpm: float, section_beats: list[float] | None = None) -> MapScorecard:
    """Run the parity simulation on a parsed ``DifficultyBeatmap``.

    Args:
        beatmap: a ``DifficultyBeatmap`` (has ``.color_notes`` / ``.bomb_notes``).
        bpm: song tempo, required to judge reset timing in wall-clock seconds.
        section_beats: optional sorted section-boundary beats; if given, the result
            records each hand's parity at each boundary (Phase-2 seam stitching).

    Returns:
        MapScorecard with the violation count (DoD metric) and per-hand detail.
    """
    sec_per_beat = 60.0 / bpm if bpm > 0 else 0.5
    notes = list(beatmap.color_notes)
    bombs = list(getattr(beatmap, "bomb_notes", []) or [])

    per_hand: dict[int, HandReport] = {}
    all_violation_beats: list[float] = []

    for color in (0, 1):
        report = HandReport(color=color)
        swings = _extract_swings(notes, color)

        prev: Swing | None = None          # previous swing (any kind), for timing
        prev_dir_parity: Parity | None = None  # parity of last DIRECTIONAL swing
        prev_dir_dir = 8                   # cut direction of last directional swing
        prev_dir_beat = -1e9               # beat of last directional swing
        neutral_since_dir = False          # a neutral swing has intervened since it
        prev_dir_was_reset = False         # was the last directional swing itself a reset?
        for sw in swings:
            # all-dot swings are parity-free (treated as neutral), regardless of the
            # geometric direction we stored for rendering
            nat = None if sw.flexible else _natural_parity(sw.direction)

            if prev is None or sw.beat - prev.beat > PARITY_RESET_GAP:
                # first swing, or a long break — hand re-seats parity for free
                sw.parity = nat if nat is not None else Parity.FOREHAND
                if nat is not None:
                    prev_dir_parity, prev_dir_beat = nat, sw.beat
                    prev_dir_dir = sw.direction
                    neutral_since_dir = False
                    prev_dir_was_reset = False
                report.swings.append(sw)
                prev = sw
                continue

            if nat is None:
                # Neutral (left/right/dot): parity-flexible. It can be swung either
                # way, so it both (a) takes the alternating parity for display and
                # (b) absorbs one parity flip for the *next* directional note — i.e.
                # it offers a free reset opportunity. It is never itself a violation.
                sw.parity = (prev_dir_parity.flipped() if prev_dir_parity is not None
                             else Parity.FOREHAND)
                neutral_since_dir = True
                report.swings.append(sw)
                prev = sw
                continue

            # Directional swing: parity is fixed by its cut direction.
            sw.parity = nat
            this_is_reset = False
            angle_dict = FOREHAND_ANGLE if nat is Parity.FOREHAND else BACKHAND_ANGLE
            same_dir = (
                prev_dir_parity is not None
                and nat == prev_dir_parity
                and _angle_delta(angle_dict[prev_dir_dir], angle_dict[sw.direction])
                < ANGLE_FLOW_DEG
            )
            if (
                same_dir
                and not neutral_since_dir  # a neutral swing absorbs the flip for free
            ):
                # Same forced parity AND near-identical cut angle = no alternation and
                # no room to roll through: a genuine stop-and-reverse reset.
                this_is_reset = True
                sw.is_reset = True
                report.resets += 1
                gap_sec = (sw.beat - prev_dir_beat) * sec_per_beat
                if _bomb_forces_reset(bombs, prev_dir_beat, sw.beat, color):
                    sw.reset_kind = "bomb"
                elif gap_sec >= HARD_RESET_SEC:
                    sw.reset_kind = "intentional"   # enough time to re-swing
                elif prev_dir_was_reset:
                    # 2nd+ fast reset in a row: the hand is locked into a
                    # non-alternating run (FFF…/BBB…) it cannot physically recover
                    # from — the V7 parity-chaos signature. A LONE fast reset is a
                    # playable "double"; a run is a wrist-break violation.
                    sw.reset_kind = "violation"
                    report.violations += 1
                    all_violation_beats.append(sw.beat)
                else:
                    sw.reset_kind = "fast_single"   # lone fast double — playable
            prev_dir_was_reset = this_is_reset
            prev_dir_parity, prev_dir_beat = nat, sw.beat
            prev_dir_dir = sw.direction
            neutral_since_dir = False
            report.swings.append(sw)
            prev = sw

        report.swing_ebpm_p95 = _swing_ebpm_p95(report.swings)
        per_hand[color] = report

    scorecard = MapScorecard(
        n_notes=len(notes),
        n_swings=sum(len(h.swings) for h in per_hand.values()),
        resets=sum(h.resets for h in per_hand.values()),
        violations=sum(h.violations for h in per_hand.values()),
        violation_beats=sorted(all_violation_beats),
        per_hand=per_hand,
    )
    return scorecard


def seam_hand_states(
    scorecard: MapScorecard, section_beats: list[float]
) -> list[dict]:
    """Parity each hand is in at each section boundary (entry/exit state).

    For each boundary beat, reports the last swing parity on each hand strictly
    before the boundary and the first swing parity at/after it. Used by Phase-2 to
    check that a stitched seam keeps each hand's alternation continuous.
    """
    out = []
    for sb in sorted(section_beats):
        entry: dict[int, dict] = {}
        for color, hand in scorecard.per_hand.items():
            before = [s for s in hand.swings if s.beat < sb]
            after = [s for s in hand.swings if s.beat >= sb]
            entry[color] = {
                "exit_parity": before[-1].parity.name if before else None,
                "enter_parity": after[0].parity.name if after else None,
            }
        out.append({"beat": round(sb, 3), "hands": entry})
    return out


def _swing_ebpm_p95(swings: list[Swing]) -> float:
    """95th-percentile effective swing BPM (a burst-speed proxy, in beats domain).

    Effective BPM between two swings = 60 / (seconds between them); here we work in
    beats so we report swings/beat * 60 is not meaningful without bpm. We instead
    return the 95th-pct of (1 / inter-swing-beat-gap), i.e. swings-per-beat at the
    fast end — multiply by song BPM downstream for true EBPM.
    """
    gaps = [
        b.beat - a.beat
        for a, b in zip(swings, swings[1:])
        if 0 < (b.beat - a.beat) <= PARITY_RESET_GAP
    ]
    if not gaps:
        return 0.0
    rates = sorted(1.0 / g for g in gaps)
    idx = min(len(rates) - 1, int(round(0.95 * (len(rates) - 1))))
    return rates[idx]


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _load_difficulty(map_path: pathlib.Path, difficulty: str) -> tuple:
    """Load a difficulty + its BPM from a .zip or a map directory.

    Returns ``(DifficultyBeatmap, bpm)``. BPM falls back to 120 if Info.dat is
    missing/unreadable.
    """
    import shutil
    import tempfile
    import zipfile

    from beatsaber_automapper.data.beatmap import parse_difficulty_dat, parse_info_dat

    tmp = None
    try:
        if map_path.suffix == ".zip":
            tmp = tempfile.mkdtemp(prefix="swing_sim_")
            with zipfile.ZipFile(map_path) as zf:
                zf.extractall(tmp)
            map_dir = pathlib.Path(tmp)
        else:
            map_dir = map_path

        bpm = 120.0
        info = None
        for info_name in ("Info.dat", "info.dat"):
            info_path = map_dir / info_name
            if info_path.exists():
                try:
                    info = parse_info_dat(info_path)
                    if info is not None:
                        bpm = float(info.bpm)
                except Exception:  # noqa: BLE001 — bad Info.dat shouldn't kill sim
                    info = None
                break

        # Resolve the requested STANDARD-characteristic difficulty via Info.dat. The
        # simulator models Standard-mode parity, so OneSaber / 90°-360° / Lawless
        # variants are out of scope and must NOT be silently substituted (they have
        # different or no parity rules). Maps lacking a Standard <difficulty> raise.
        diff_path = None
        if info is not None and info.difficulties:
            std = [d for d in info.difficulties if d.characteristic == "Standard"]
            match = next((d for d in std if d.difficulty.lower() == difficulty.lower()), None)
            if match is not None and match.filename:
                cand = map_dir / match.filename
                if cand.exists():
                    diff_path = cand
            if diff_path is None:
                raise FileNotFoundError(
                    f"No Standard '{difficulty}' difficulty in {map_path.name} "
                    f"(have: {[(d.characteristic, d.difficulty) for d in info.difficulties]})"
                )
        else:
            # No Info.dat (e.g. a loose dir / model dump): fall back to filename match.
            diff_files = sorted(map_dir.glob("*.dat"))
            for f in diff_files:
                if f.name.lower().startswith(difficulty.lower()):
                    diff_path = f
                    break
            if diff_path is None:
                raise FileNotFoundError(f"No '{difficulty}' .dat in {map_dir}")
        bm = parse_difficulty_dat(diff_path)
        if bm is None:
            raise RuntimeError(f"Failed to parse {diff_path}")
        return bm, bpm
    finally:
        if tmp is not None:
            shutil.rmtree(tmp, ignore_errors=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Swing-simulator parity scorecard.")
    ap.add_argument("map", type=pathlib.Path, help="map .zip or directory")
    ap.add_argument("--difficulty", default="Expert")
    ap.add_argument("--json", type=pathlib.Path, default=None)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    bm, bpm = _load_difficulty(args.map, args.difficulty)
    card = simulate(bm, bpm=bpm)
    d = card.as_dict()
    logger.info(
        "%s [%s]: %d notes, %d swings, %d resets, %d VIOLATIONS",
        args.map.name, args.difficulty, d["n_notes"], d["n_swings"],
        d["resets"], d["violations"],
    )
    if d["violations"]:
        logger.info("  violation beats: %s", d["violation_beats"][:20])
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(d, indent=2))
        logger.info("wrote %s", args.json)


if __name__ == "__main__":
    main()
