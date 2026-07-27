#!/usr/bin/env python
"""Write a Beat Saber map FROM the text score — the reverse of map_view.py.

Phase 3 of docs/map_authoring_plan.md. `map_view.py` made maps readable; this
makes them writable, which closes the loop: I can read a map, edit the score,
and get a playable level back. That is what lets me hand-author a map, and
hand-authoring is the strongest available test of whether the evaluation suite
is right — if a map I build scores human-range but plays badly (or the reverse),
the suite has another blind spot.

Two input forms, because note-by-note authoring does not scale to the 1300+
notes in a real map:

  SCORE   the exact text `map_view.py` prints. Round-trips losslessly, so the
          workflow is: view a passage, edit the rows, write it back.

  BLOCKS  the practical form. A line per phrase:

              <start_beat> <idiom_id|name> x<repeats> [hand]

          which expands using the mined human vocabulary — the same vocabulary
          `rule_mapper.py` samples from. Composing at the phrase level is also
          how human mappers actually work.

Usage:
  python scripts/map_view.py <map.zip> --bars 33-40 > passage.txt
  # edit passage.txt
  python scripts/map_write.py --score passage.txt --audio song.ogg --bpm 128 \
      --out edited.zip
  python scripts/map_write.py --score passage.txt --check     # parse only
"""
from __future__ import annotations

import argparse
import pathlib
import re
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from beatsaber_automapper.data.beatmap import ColorNote, DifficultyBeatmap  # noqa: E402

# inverse of map_view.ARROW
DIR_OF = {"↑": 0, "↓": 1, "←": 2, "→": 3, "↖": 4, "↗": 5, "↙": 6, "↘": 7, "•": 8}

# " 34  132.00 │ 0,2 ↑ B    │ 2,0 ↙ F    │ ..."  — trailing lanes ignored
ROW = re.compile(
    r"^\s*(?P<bar>\d+)\s+(?P<beat>[\d.]+)\s*│(?P<left>[^│]*)│(?P<right>[^│]*)"
)
CELL = re.compile(r"(?P<x>[0-3])\s*,\s*(?P<y>[0-2])\s*(?P<dir>[↑↓←→↖↗↙↘•])")


class ScoreParseError(ValueError):
    pass


def parse_score(text: str) -> list[ColorNote]:
    """Parse the text score back into notes.

    Deliberately strict about the note cell and lenient about everything else:
    parity tags, idiom annotations and audio lanes are DERIVED columns, so they
    are ignored on the way in rather than trusted. A typo in a cell raises
    instead of silently producing a different map.
    """
    notes: list[ColorNote] = []
    for lineno, line in enumerate(text.splitlines(), 1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        m = ROW.match(line)
        if not m:
            continue                      # headers, rules, labels
        try:
            beat = float(m.group("beat"))
        except ValueError:
            continue
        for color, key in ((0, "left"), (1, "right")):
            cell = m.group(key)
            if not cell.strip():
                continue
            c = CELL.search(cell)
            if not c:
                raise ScoreParseError(
                    f"line {lineno}: cannot read a note cell from {cell.strip()!r}. "
                    f"Expected e.g. '0,2 ↑'.")
            notes.append(ColorNote(beat=beat, x=int(c.group("x")), y=int(c.group("y")),
                                   color=color, direction=DIR_OF[c.group("dir")]))
    return sorted(notes, key=lambda n: (n.beat, n.color))


def parse_blocks(text: str) -> list[ColorNote]:
    """Expand phrase-level block lines using the human idiom vocabulary.

        <start_beat> <idiom_rank> x<repeats> [L|R]

    Each repeat applies the idiom's (dx, dy, dir_to) to the running hand
    position, at the spacing implied by the idiom's dt class.
    """
    from beatsaber_automapper.evaluation import idiom as idm
    _counts, ranked, _probs = idm.load_vocab()
    if not ranked:
        raise ScoreParseError("no idiom vocabulary — run scripts/calibrate_idiom.py")

    dt_of = {0: 0.125, 1: 0.25, 2: 0.5, 3: 1.0, 4: 2.0}
    pos = {0: (1, 1), 1: (2, 1)}
    notes: list[ColorNote] = []
    for lineno, line in enumerate(text.splitlines(), 1):
        line = line.split("#")[0].strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 2:
            raise ScoreParseError(f"line {lineno}: expected '<beat> <idiom> [xN] [L|R]'")
        try:
            beat = float(parts[0])
            rank = int(parts[1].lstrip("#"))
        except ValueError as e:
            raise ScoreParseError(f"line {lineno}: {e}") from e
        if not 0 <= rank < len(ranked):
            raise ScoreParseError(f"line {lineno}: idiom #{rank} out of range "
                                  f"(0..{len(ranked) - 1})")
        reps = 1
        color = 0
        for p in parts[2:]:
            if p.lower().startswith("x"):
                reps = int(p[1:])
            elif p.upper() in ("L", "R"):
                color = 0 if p.upper() == "L" else 1
        dx, dy, d_from, d_to, cls = ranked[rank]
        step = dt_of.get(cls, 0.5)
        # CHAIN rather than repeat. Applying one idiom N times emits the same cut
        # direction every note, so parity never alternates and every second swing
        # is a wrist-break (measured: 24 violations in 38 swings). A phrase is a
        # SEQUENCE of compatible idioms — each one's `d_from` must match the
        # previous one's `d_to` — which is also how the vocabulary is structured.
        cur_dir = d_from
        for i in range(reps):
            if i == 0:
                step_idiom = ranked[rank]
            else:
                step_idiom = next(
                    (t for t in ranked
                     if t[2] == cur_dir and t[4] == cls
                     and 0 <= pos[color][0] + t[0] <= 3
                     and 0 <= pos[color][1] + t[1] <= 2),
                    None)
                if step_idiom is None:
                    break          # nothing idiomatic continues this phrase
            sdx, sdy, _sf, sd_to, _sc = step_idiom
            x, y = pos[color]
            nx = max(0, min(3, x + sdx))
            ny = max(0, min(2, y + sdy))
            notes.append(ColorNote(beat=beat + i * step, x=nx, y=ny,
                                   color=color, direction=sd_to))
            pos[color] = (nx, ny)
            cur_dir = sd_to
    return sorted(notes, key=lambda n: (n.beat, n.color))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--score", help="text score file (map_view.py output)")
    src.add_argument("--blocks", help="phrase-level block file")
    ap.add_argument("--audio", help="song audio, required to package a playable zip")
    ap.add_argument("--bpm", type=float, default=120.0)
    ap.add_argument("--out", help="output .zip")
    ap.add_argument("--difficulty", default="Expert")
    ap.add_argument("--check", action="store_true",
                    help="parse and score only; do not write a zip")
    a = ap.parse_args()

    text = pathlib.Path(a.score or a.blocks).read_text()
    notes = parse_score(text) if a.score else parse_blocks(text)
    if not notes:
        raise SystemExit("no notes parsed — is this a map_view.py score?")
    print(f"parsed {len(notes)} notes, beats {notes[0].beat:.2f}–{notes[-1].beat:.2f}")

    # always report playability + how the suite sees it, since that is the point
    from beatsaber_automapper.evaluation import swing_sim as ss

    class _BM:
        color_notes = notes
        bomb_notes: list = []

    card = ss.simulate(_BM(), bpm=a.bpm)
    print(f"swing simulator: {card.n_swings} swings, {card.violations} violations")
    try:
        from beatsaber_automapper.evaluation import handrole, idiom, rhythm
        m = {**idiom.idiom_metrics(_BM()).metrics,
             **rhythm.rhythm_metrics(_BM()).metrics,
             **handrole.handrole_metrics(_BM()).metrics}
        for k in ("idiom_coverage", "idiom_local", "pulse_stability", "role_asymmetry"):
            v = m.get(k)
            if v is not None and v == v:
                print(f"  {k:18s} {v:.3f}")
    except Exception:  # noqa: BLE001
        pass

    if a.check:
        return
    if not (a.audio and a.out):
        raise SystemExit("--audio and --out are required to package a zip "
                         "(or pass --check to parse only)")

    from beatsaber_automapper.generation.export import package_level
    bm = DifficultyBeatmap(color_notes=notes, bomb_notes=[], obstacles=[])
    out = package_level({a.difficulty: bm}, pathlib.Path(a.audio),
                        pathlib.Path(a.out), bpm=a.bpm,
                        song_name="Hand-authored", song_author="claude")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
