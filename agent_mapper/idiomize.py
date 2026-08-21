#!/usr/bin/env python
"""Re-choose every note's POSITION and DIRECTION from the mined human vocabulary.

**The defect this fixes, in Kyle's words.** He played the first agent-built map and
said: *"I was expecting the agent's song to be much better. The main problem is the
notes flow in a really odd way."* The per-map judge, which was calibrated on 1 100
human maps and never told what he said, names the same thing in numbers:

    idiom_coverage   0.503   human percentile  0.4     (human median 0.909)
    idiom_jsd        0.731   human percentile 98.5     (human median 0.430)
    angle_change    37.6 deg human percentile 95.8     (human median 19.5)

Half of the map's hand-to-hand transitions are moves **no human mapper makes**, and
the wrist rotation between swings is double the human median. `mapctl auto` chose
each note's cell from a geometric rule of its own; it never knew that 130 395 human
transitions collapse to 2 510 idioms whose top 500 cover ~90 % of everything human
mappers do (`docs/eval_suite_v2.md` A3). That vocabulary is mined and checked in,
and `scripts/rule_mapper.py` already samples from it well enough to beat our trained
model on the idiom axis from rules alone.

**Why this is a post-pass and not a rewrite of `auto`.** The obvious move -- generate
the whole map with `rule_mapper.build_map` -- was tried first and measured: idiom
coverage went 0.503 -> 0.901 and angle_change 37.6 -> 22.8, exactly as intended, but
`ebpm_burst` went **376 -> 752** against a human 376 and the map got *worse* overall.
`rule_mapper` picks its own note times and hands and has never heard of the per-hand
floor that `agent_mapper` measured over **31 723 human gaps** (cohort p5 = 148 ms) --
the same 752 regression recorded in `agent_mapper/PROGRESS.md`, arriving by a new
route.

So this pass changes **only `x`, `y` and `direction`**. Note times, hand assignment
and note count come out byte-identical, which means `ebpm_burst`, `nps`, `peak_nps`,
every rhythm metric and every hand-role metric **cannot move** -- the A/B isolates
one thing, the way the walls/arcs/chains ladder does. It keeps the musical decisions
(which onset, which instrument, which hand) where they belong: with the agent that
can see the whole song.

Usage:
    python agent_mapper/idiomize.py in.zip --out out.zip
    python agent_mapper/idiomize.py in.zip --out out.zip --crossover 0.21 --seed 1
"""
from __future__ import annotations

import argparse
import pathlib
import random
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from beatsaber_automapper.evaluation import idiom as idm  # noqa: E402
from beatsaber_automapper.evaluation import swing_sim as ss  # noqa: E402

# Measured human values. Every one of these is a median from the human corpus, not
# a chosen target -- a point target is what Goodharted `h_dist`.
CROSSOVER_TARGET = 0.208   # judge reference median over 1100 maps
TRAVEL_TARGET = 4.167      # grid-units per second, same source
# Vocabulary depth to sample from. ★Not 500, even though the top 500 idioms cover
# ~90 % of human transitions -- BECAUSE they do. Sampling only from the top 500
# forces `idiom_coverage` to ~1.0 by construction, and humans sit at 0.909; the map
# then looks *more* vocabulary-pure than a human map, which is the "more human than
# human" tell, not a win. Measured over a depth sweep on the same map and seed:
#   500 -> coverage 0.995   1000 -> 0.903   2000 -> 0.883   4000 -> 0.887
# 1000 reproduces the human's own top-500 coverage, so that is the default.
VOCAB_DEPTH = 1000

# ★**How often a hand REPEATS a figure it just played.** Measured need: with
# independent sampling at every note, `idiom_local` (distinct idioms inside a
# 16-note window) sat at the **98.2nd human percentile on 23 of 23** autobuilt maps
# -- our maps were *more varied* locally than almost every human map. That is the
# same "globally right, locally wrong" shape the suite already records for hand
# roles, arriving on the vocabulary axis: A3's founding result is that human mapping
# is **a small vocabulary deployed deliberately**, and deliberate means a figure gets
# repeated for a few beats before it changes. Sampling fresh every time is maximum
# entropy, which docs/eval_suite_v2.md Finding 3 already established is NOT human.
REPEAT_P = 0.55
REPEAT_WINDOW = 6

DOWN_DIRS = (1, 6, 7)
UP_DIRS = (0, 4, 5)
HOME = {0: (0, 1), 1: (2, 3)}   # red left, blue right


def _parity_of(direction: int) -> int | None:
    if direction in DOWN_DIRS:
        return 0
    if direction in UP_DIRS:
        return 1
    return None


class _Hand:
    __slots__ = ("color", "x", "y", "direction", "parity", "beat")

    def __init__(self, color: int):
        self.color = color
        self.x = HOME[color][0]
        self.y = 1
        self.direction = 1
        self.parity = 0
        self.beat = -99.0


def _candidates(ranked, counts, h: _Hand, dt_beats: float, spb: float,
                top_k: int, cross_ok: bool):
    """Vocabulary moves legal from this hand's state, with their human weights.

    Returns `[(idiom, weight)]`. The weight is the idiom's **frequency in the human
    corpus**, damped by how far its implied travel is from the human median.

    ★**Why frequency and not flow rank.** The first version sorted candidates by
    flow alone and sampled the best 6. It produced `idiom_coverage` **0.996** where
    humans sit at 0.909, and `idiom_top50` **0.207** where humans sit at 0.404 --
    i.e. it drew almost everything from the vocabulary, but from the *long tail* of
    it, using rare-but-legal moves instead of the ones human mappers actually
    reach for. Coverage overshooting the human value is not a win; it is the
    "more human than human" signature that saturated `h_dist`, arriving on a new
    axis. The vocabulary ships its counts -- using them is free and is the whole
    point of having mined it.
    """
    cls = idm.dt_class(dt_beats)
    dt_sec = max(dt_beats * spb, 1e-6)
    out = []
    for entry in ranked[:top_k]:
        dx, dy, d_from, d_to, c = entry
        if c != cls or d_from != h.direction:
            continue
        nx, ny = h.x + dx, h.y + dy
        if not (0 <= nx <= 3 and 0 <= ny <= 2):
            continue
        crosses = nx not in HOME[h.color]
        if crosses and not cross_ok:
            continue
        p = _parity_of(d_to)
        if p is not None and p == h.parity and dt_sec < ss.HARD_RESET_SEC:
            continue
        dist = (dx * dx + dy * dy) ** 0.5
        # A1: prefer travel near the human median, but as a soft weight rather than
        # a sort key -- a hard sort is what discarded the common idioms.
        speed_err = abs(dist / dt_sec - TRAVEL_TARGET) / TRAVEL_TARGET
        w = counts.get(entry, 1) / (1.0 + speed_err) ** 2
        out.append((entry, w, crosses))
    return out


def _pick(cands, rng: random.Random, prefer_cross: bool, width: int = 0):
    """Frequency-weighted choice, preferring a crossover when one was asked for.

    🔴🔴**`width` WAS A DEAD PARAMETER UNTIL 2026-08-21.** It was accepted by
    `idiomize()`, threaded through `idiomize_zip()`, and advertised in `--help` as
    *"sample from the best N candidates (1 = greedy)"* -- and never referenced in the
    body. `--width 1` and `--width 12` produced **byte-identical** maps. Same shape as
    the `BEAT_GRID_SUBDIV` no-op this project already retired: a knob that reads as a
    lever and silently is not one.

    ★**What it does now**: restrict the frequency-weighted draw to the `width` most
    common candidates. `1` is greedy (always the single most human cell, minimum
    variety); a large width samples the whole tail. **This is a VARIETY dial, and
    variety is part of transition difficulty** -- Kyle: *"difficulty isn't always just
    NPS, it's how hard are the notes to get to from the last note as well."*
    ⚠️`0` means "no restriction", which is the pre-fix behaviour, so nothing that did
    not pass `width` changes.
    """
    pool = cands
    if prefer_cross:
        crossing = [c for c in cands if c[2]]
        if crossing:
            pool = crossing
    if width and width > 0 and len(pool) > width:
        pool = sorted(pool, key=lambda c: -c[1])[:width]
    total = sum(w for _e, w, _x in pool)
    if total <= 0:
        return None
    r = rng.random() * total
    for e, w, _x in pool:
        r -= w
        if r <= 0:
            return e
    return pool[-1][0]


def idiomize(records, bpm: float, *, seed: int = 0, top_k: int = VOCAB_DEPTH,
             width: int = 0, crossover: float = CROSSOVER_TARGET,
             repeat_p: float = REPEAT_P):
    """Redraw (x, y, direction) for every note from the human vocabulary.

    `records` is a list of dicts with keys b/x/y/c/d (the v3 `colorNotes` shape).
    Returns `(new_records, n_fallback)`. Beat, colour, order and count are
    preserved exactly -- that invariant is the whole point of the pass and it is
    asserted by the caller.
    """
    counts, ranked, _ = idm.load_vocab()
    if not ranked:
        raise SystemExit("no idiom vocabulary -- run scripts/calibrate_idiom.py")
    rng = random.Random(seed)
    spb = 60.0 / bpm if bpm > 0 else 0.5

    hands = {0: _Hand(0), 1: _Hand(1)}
    recent: dict[int, list] = {0: [], 1: []}
    order = sorted(range(len(records)),
                   key=lambda i: (float(records[i].get("b", 0.0)),
                                  int(records[i].get("c", 0))))
    out = [None] * len(records)
    n_fallback = 0

    for i in order:
        r = records[i]
        beat = float(r.get("b", 0.0))
        color = int(r.get("c", 0))
        if color not in (0, 1):
            out[i] = (int(r.get("x", 0)), int(r.get("y", 0)), int(r.get("d", 8)))
            continue
        h = hands[color]
        dt = beat - h.beat
        if dt <= 0:
            dt = 1e-3
        # Crossovers are DELIBERATE and occasional: humans cross on ~21 % of notes,
        # and `enforce_color_separation` in our production path forbids them
        # entirely, which the judge reports as the single most non-human property
        # of every map we ship (crossover 0.000, human percentile 0.4).
        cross_ok = rng.random() < crossover
        cands = _candidates(ranked, counts, h, min(dt, 2.0), spb, top_k, cross_ok)
        # Prefer a figure this hand has just played, when one still fits from its
        # current state. This is what makes the local vocabulary small.
        if cands and recent[color] and rng.random() < repeat_p:
            legal = {c[0] for c in cands}
            again = [e for e in recent[color] if e in legal]
            if again:
                cands = [c for c in cands if c[0] in set(again)]
        # ★A crossover knob that only PERMITS crossing does not produce crossing.
        # Set to the human 0.208 it realised 0.063, because most legal candidates
        # stay on-side and a permissive filter never changes the odds. When the
        # draw asks for a crossover, pick from the crossing candidates.
        pick = _pick(cands, rng, prefer_cross=cross_ok, width=width)
        if pick is None and cross_ok:
            cands = _candidates(ranked, counts, h, min(dt, 2.0), spb, top_k, False)
            pick = _pick(cands, rng, prefer_cross=False, width=width)
        if pick is not None:
            dx, dy, _df, d_to, _c = pick
            nx, ny = h.x + dx, h.y + dy
        else:
            # No idiom fits this state. Keep the ORIGINAL cell rather than invent
            # one -- an invented cell is exactly what this pass exists to remove.
            n_fallback += 1
            nx, ny, d_to = int(r.get("x", 0)), int(r.get("y", 0)), int(r.get("d", 8))

        out[i] = (nx, ny, d_to)
        if pick is not None:
            recent[color].append(pick)
            del recent[color][:-REPEAT_WINDOW]
        h.x, h.y, h.direction, h.beat = nx, ny, d_to, beat
        p = _parity_of(d_to)
        h.parity = p if p is not None else (h.parity ^ 1)

    new = []
    for r, (nx, ny, nd) in zip(records, out):
        q = dict(r)
        q["x"], q["y"], q["d"] = int(nx), int(ny), int(nd)
        new.append(q)
    return new, n_fallback


def _reparity(notes: list[dict], bpm: float) -> list[dict]:
    """Run `postprocess.fix_parity` over v3 note dicts, preserving times and colours.

    Returns the notes unchanged if the fixer is unavailable or disagrees about the
    note count -- a parity pass that silently drops notes would be worse than the
    violation it fixes.
    """
    try:
        from beatsaber_automapper.data.beatmap import ColorNote, DifficultyBeatmap
        from beatsaber_automapper.generation.postprocess import fix_parity
    except Exception:  # noqa: BLE001
        return notes
    order = sorted(range(len(notes)), key=lambda i: (notes[i].get("b", 0.0),
                                                     notes[i].get("c", 0)))
    cn = [ColorNote(beat=float(notes[i].get("b", 0.0)), x=int(notes[i].get("x", 0)),
                    y=int(notes[i].get("y", 0)), color=int(notes[i].get("c", 0)),
                    direction=int(notes[i].get("d", 0))) for i in order]
    try:
        fixed = fix_parity(DifficultyBeatmap(version="3.0.0", color_notes=cn))
    except Exception:  # noqa: BLE001
        return notes
    out_notes = list(getattr(fixed, "color_notes", []) or [])
    if len(out_notes) != len(cn):
        return notes
    out = [dict(n) for n in notes]
    for slot, fn in zip(order, out_notes):
        out[slot]["x"] = int(fn.x)
        out[slot]["y"] = int(fn.y)
        out[slot]["d"] = int(fn.direction)
    return out


def idiomize_zip(src: pathlib.Path, dst: pathlib.Path, *, seed: int = 0,
                 top_k: int = VOCAB_DEPTH, width: int = 0,
                 crossover: float = CROSSOVER_TARGET,
                 repeat_p: float = REPEAT_P) -> tuple[int, int]:
    """Copy `src` to `dst` with only note cells redrawn. Returns (n_notes, n_fallback)."""
    import json
    import shutil
    import tempfile
    import zipfile

    tmp = pathlib.Path(tempfile.mkdtemp(prefix="idiomize_"))
    try:
        with zipfile.ZipFile(src) as zf:
            zf.extractall(tmp)
            names = zf.namelist()
        # EXACT basename: "BPMInfo.dat" also ends with "info.dat" and sorts first in
        # 73 of 300 corpus zips, where picking it yields a silent bpm of 120.
        info = next((n for n in names
                     if n.split("/")[-1].lower() == "info.dat"), None)
        dat = next((n for n in names
                    if n.lower().split("/")[-1].startswith("expert")
                    and n.lower().endswith("standard.dat")), None)
        if dat is None:
            dat = next((n for n in names if n.lower().endswith("standard.dat")), None)
        if dat is None or info is None:
            raise ValueError("no Expert Standard difficulty / info.dat in the zip")

        bpm = 120.0
        try:
            meta = json.loads((tmp / info).read_text(encoding="utf-8-sig"))
            for k in ("_beatsPerMinute", "beatsPerMinute", "bpm"):
                if k in meta:
                    bpm = float(meta[k])
                    break
                audio = meta.get("audio") or {}
                if k in audio:
                    bpm = float(audio[k])
                    break
        except Exception:  # noqa: BLE001
            pass

        f = tmp / dat
        d = json.loads(f.read_text(encoding="utf-8-sig"))
        if not str(d.get("version", "")).startswith("3"):
            raise ValueError("only v3 maps are supported (ours are 3.3.0)")
        notes = d.get("colorNotes") or []
        if len(notes) < 20:
            raise ValueError("too few notes to re-place")

        new, nfb = idiomize(notes, bpm, seed=seed, top_k=top_k,
                            width=width, crossover=crossover, repeat_p=repeat_p)
        # The invariant the whole design rests on: this pass moves cells and
        # nothing else. If it ever changes a time, a colour or the count, the A/B
        # stops isolating one thing and the comparison is worthless.
        assert len(new) == len(notes)
        for o, q in zip(notes, new):
            assert o.get("b") == q.get("b") and o.get("c") == q.get("c")

        # ★★RE-FIX PARITY. `mapctl export` runs `postprocess.fix_parity` and then THIS
        # pass rewrites every direction, so the fixer's work is undone downstream and
        # nothing re-checks. Measured on 1fb3f: 0 violations and 0 resets before this
        # pass, **1 violation and 30 resets after** -- and `mapjudge` FAILs on
        # `viol > 0` regardless of the p-value, so a single unplayable transition sinks
        # an otherwise-passing map (that one scored p=0.746).
        # ⚠️Reuse the pipeline's fixer; hand-rolled parity repair already cost this
        # project 380 notes and still left violations.
        new = _reparity(new, bpm)
        # The invariant survives: the fixer changes DIRECTIONS, never times, colours
        # or the count -- re-asserted here because that is what makes the A/B valid.
        assert len(new) == len(notes)
        for o, q in zip(notes, new):
            assert o.get("b") == q.get("b") and o.get("c") == q.get("c")

        d["colorNotes"] = new
        f.write_text(json.dumps(d), encoding="utf-8")
        dst.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(dst, "w", zipfile.ZIP_DEFLATED) as zo:
            for pth in sorted(tmp.rglob("*")):
                if pth.is_file():
                    zo.write(pth, pth.relative_to(tmp).as_posix())
        return len(new), nfb
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("zip_in", type=pathlib.Path)
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--width", type=int, default=0,
                    help="restrict the draw to the N most common cells; 1 = greedy "
                         "(least variety), 0 = no restriction. A VARIETY dial — was a "
                         "dead no-op before 2026-08-21")
    ap.add_argument("--top-k", type=int, default=VOCAB_DEPTH,
                    help="vocabulary depth (default %(default)s; see VOCAB_DEPTH)")
    ap.add_argument("--crossover", type=float, default=CROSSOVER_TARGET)
    ap.add_argument("--repeat-p", type=float, default=REPEAT_P,
                    help="chance a hand repeats a figure it just played "
                         "(0 = resample independently, which is what put "
                         "idiom_local at the 98th human percentile)")
    a = ap.parse_args()

    n, nfb = idiomize_zip(a.zip_in, a.out, seed=a.seed, top_k=a.top_k,
                          width=a.width, crossover=a.crossover,
                          repeat_p=a.repeat_p)
    print(f"{a.zip_in.name}: re-placed {n - nfb}/{n} notes from the human vocabulary "
          f"({nfb} kept their original cell: no idiom fit)")
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
