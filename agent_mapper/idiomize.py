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
# ★**LEG 3's second travel lever** (2026-08-24). `travel` and `angle_change`
# anti-correlate under `width`, and the `technical` style asks for BOTH at p75, so a
# width mapping is over-specified -- three were measured and none beat leaving `width`
# alone. This is the orthogonal knob: `width` chooses HOW MANY candidates are
# considered, `travel_target` re-weights them by HOW FAR each move travels, without
# touching which cut directions are on offer.
# ⚠️Do NOT re-attempt a width->style mapping; it is refuted three times over.
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
# ★★**0.55 → 0.25 on 2026-09-13aa, the first time this was ever measured on a BUILD.** It was
# reachable only from `idiomize.py`'s own CLI, so every map this project shipped used 0.55
# unexamined. It controls `idiom_local` (distinct transitions per 16-transition window) almost
# linearly -- isolated, 6 seeds: **0.900 / 0.865 / 0.812 / 0.755** at p = 0 / .25 / .55 / .80 --
# and leaves 4-bar block echo FLAT (0.454 / 0.450 / 0.441 / 0.436), which is the defect it was
# written for. ⇒at 0.55 it was paying the axis this repo calls *"globally right, locally wrong"*
# and buying nothing. Human median `idiom_local` is **0.867**; 0.25 lands on it.
# Full builds, 4 songset songs x 3 seeds: `idiom_local` **+0.020 to +0.043, outside 2se on all
# four**, with echo not falling anywhere (+0.000 to +0.023), `idiom_coverage` ±0.02 and judge p
# ±0.04 (both inside noise), and no map gaining a red.
REPEAT_P = 0.25
REPEAT_WINDOW = 6

# ★How strongly a palette landing is preferred, as a WEIGHT on the frequency-weighted
# draw. ⚠️Never a hard filter: see the note in `idiomize()`. 2026-09-12h shipped the
# filter by mistake and it put `idiom_coverage` at the 1.7th human percentile.
PALETTE_BOOST = 6.0

# ★**How strongly a landing THIS MAP HAS ALREADY PLAYED is preferred** (2026-09-13r). Same
# banded form as the palette, and for the same reason: a hard filter leaves whatever idioms
# happen to land on a remembered cell (the long tail) and put `idiom_coverage` at the 1.7th
# percentile, while a flat boost overshoots past "more human than human".
# ⚠️This is NOT the palette and NOT `REPEAT_P`. The palette is a set of landings decided
# BEFORE the map is drawn; `REPEAT_P` remembers the last **6 notes** where the defect lives at
# **4 bars**. This remembers every landing the hand has played SO FAR, which is what makes a
# vocabulary narrow without anyone choosing it in advance -- and it can never force an
# out-of-vocabulary transition, because it only reweights candidates `_candidates` already
# offered. Measured over 500 human Experts (`scripts/exp_vocabulary.py`): figure-vocabulary
# entropy explains **r = -0.734 (r2 0.538)** of 4-bar block echo while NOTE COUNT explains
# r = -0.006, and humans span 4.17-5.53 bits (p10-p90) where our four builds span 5.68-5.86
# -- the 95th-98th percentile on every song, another builder constant across a per-song axis.
# ★**A STYLE LEVER, OFF BY DEFAULT** (2026-09-13y). With the parity leak closed
# (`STRICT_PARITY`) it is stable: 12 builds, `idiom_coverage` inside the control band on every
# seed with a TIGHTER spread. What it actually costs is **local** variety -- `idiom_local` 36.8
# → 21.9 pct, `diagonal_share` 17.6 → 5.7, `angle_change` 15.3 → 5.2, judge p −0.2 to −0.3
# (~4.3 se) -- while `idiom_jsd` and `idiom_top50` improve. A coherent trade: the whole-map
# distribution bought with local repetition, pushing the wrong way on the one idiom axis this
# repo already calls "globally right, locally wrong". It does not clear SCATTER (room 0.46 →
# 0.64 against a line at 1.00), so it cannot be a default.
# 🔴The earlier verdict below was REFUTED -- the collapse was the parity leak, not this lever:
# **(superseded 2026-09-13v note)** Over 18 builds per arm it makes
# `idiom_coverage` **bimodal**: the control never leaves 0.79-0.86, and about a THIRD of treated
# seeds land at 0.47-0.67 — the palette FILTER's failure range, arriving through a weight. Three
# seeds called that "noise" because the difference was tested against the treated arm's own sd,
# which the treatment had inflated ~20x; against the SE of the difference it is a real drop.
# It does what it says on its own axis (entropy −0.29 to −0.49, block echo +0.02 to +0.05, 1f333's
# SCATTER margin 0.43 → 0.77) and it never clears a red. Keep it off.
MEMORY_BOOST = 0.0

# ★★**Alternate unconditionally, like the pass that runs after us.** `idiomize_zip` finishes
# with `fix_parity`, which alternates every consecutive same-hand pair with **no** time
# condition -- so a same-parity repeat placed here (a RESET, legal and human) is guaranteed to
# be rewritten by a pass that does not know the vocabulary. Measured 2026-09-13x on one song:
# with resets allowed the fixer rewrote **319-351 of 728 directions** and took the share of
# transitions in the human top-500 from 0.998 to **0.587**; with this on it rewrites **0** and
# the share stays at 0.998, with fallbacks still 0 in every arm.
# ⚠️This does NOT change how many resets the SHIPPED map has -- `fix_parity` already removes
# them all (every songset build reads `resets 0` against a human's 2). It changes only WHO
# chooses the direction: the vocabulary-aware sampler, or the blind repair.
STRICT_PARITY = True

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
                top_k: int, cross_ok: bool, travel_target: float = TRAVEL_TARGET,
                strict_parity: bool = False):
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
        # ★★`strict_parity` matches `fix_parity`, which alternates UNCONDITIONALLY. This pass
        # deliberately allows a same-parity repeat when there is time to re-cock (a reset,
        # which is legal and which humans play) -- but `idiomize_zip` runs `fix_parity` after
        # it, so **every reset placed here is guaranteed to be rewritten** by a pass that does
        # not know the vocabulary. An option the next pass always overrides is not an option,
        # it is a leak. See PROGRESS 2026-09-13w-x.
        if p is not None and p == h.parity and (strict_parity or dt_sec < ss.HARD_RESET_SEC):
            continue
        dist = (dx * dx + dy * dy) ** 0.5
        # A1: prefer travel near the human median, but as a soft weight rather than
        # a sort key -- a hard sort is what discarded the common idioms.
        speed_err = abs(dist / dt_sec - travel_target) / max(travel_target, 1e-6)
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

    ★★**DEFAULT 3, validated n=23 (2026-08-21).** Found by READING two maps side by
    side: the human plays a small recurring vocabulary and we played a scatter. Top-5
    cell share **0.342 -> 0.492** (human 0.577) and recurrence-within-8-notes
    **0.319 -> 0.434** (human 0.496), 23/23 still PASS. ⚠️`idiom_local` falls to the
    15th percentile and that is CONVERGENCE, not a cost -- the human map of the dogfood
    song sits at the **8.7th**.

    ★**What it does**: restrict the frequency-weighted draw to the `width` most
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


def _palette_of(records, n_per_hand: int) -> dict[int, set]:
    """The `n_per_hand` most-played `(x, y, dir)` shapes per hand in these records."""
    from collections import Counter
    per = {0: Counter(), 1: Counter()}
    for r in records:
        c = int(r.get("c", 0))
        if c in (0, 1):
            per[c][(int(r.get("x", 0)), int(r.get("y", 0)), int(r.get("d", 8)))] += 1
    return {c: {s for s, _ in per[c].most_common(n_per_hand)} for c in (0, 1)}


def idiomize(records, bpm: float, *, seed: int = 0, top_k: int = VOCAB_DEPTH,
             width: int = 3, crossover: float = CROSSOVER_TARGET,
             repeat_p: float = REPEAT_P,
             travel_target: float = TRAVEL_TARGET,
             palette: dict[int, set] | None = None,
             memory_boost: float = MEMORY_BOOST,
             strict_parity: bool = False):
    """Redraw (x, y, direction) for every note from the human vocabulary.

    `records` is a list of dicts with keys b/x/y/c/d (the v3 `colorNotes` shape).
    Returns `(new_records, n_fallback)`. Beat, colour, order and count are
    preserved exactly -- that invariant is the whole point of the pass and it is
    asserted by the caller.

    ★★**`palette`** (2026-09-12g) restricts each hand to a fixed set of landing
    `(x, y, dir)` shapes, falling back to the unrestricted draw whenever no palette
    move is legal from the hand's current state. **Why a palette and not a knob.**
    Measured 2026-09-12f over 400 human Experts, a human map uses a median of **28.5**
    distinct shapes per hand with its top ten covering **82 %** of its notes; ours use
    34-60 with a top-ten share of 53-65 %, and vocabulary concentration correlates with
    4-bar block echo at **r = +0.654**. Four knobs were swept against that gap and none
    moved it -- `REPEAT_P`/`REPEAT_WINDOW` (unordered, and its window is 6 NOTES where
    the defect is 4 BARS) and `top_k` from 200 to 2000, across which the resulting
    per-map vocabulary stayed **38-55 per hand**. ⇒Sampling a fresh idiom PER NOTE makes
    a map's vocabulary a function of its NOTE COUNT, not the pool depth. A human does not
    sample per note; **he commits to a palette and plays it all map.** That is a change
    to the sampling structure, which is why no knob could find it.
    """
    counts, ranked, _ = idm.load_vocab()
    if not ranked:
        raise SystemExit("no idiom vocabulary -- run scripts/calibrate_idiom.py")
    rng = random.Random(seed)
    spb = 60.0 / bpm if bpm > 0 else 0.5

    hands = {0: _Hand(0), 1: _Hand(1)}
    recent: dict[int, list] = {0: [], 1: []}
    played: dict[int, set] = {0: set(), 1: set()}   # every landing this hand has used
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
        cands = _candidates(ranked, counts, h, min(dt, 2.0), spb, top_k, cross_ok,
                            travel_target, strict_parity)
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
        # ★The palette narrows WHERE a swing may land, never which idioms exist.
        # 🔴🔴**IT IS A BANDED WEIGHT, NOT A FILTER — two wrong forms cost a bad default.**
        # Measured on one full build of 1f913, same seed (`PROGRESS.md 2026-09-12i`):
        #   no palette            idiom_coverage 0.992  human pct 94.1   judge p 0.572
        #   hard FILTER           idiom_coverage 0.618  human pct  1.7 ! judge p 0.538
        #   flat BOOST x6         idiom_coverage 0.998  human pct 97.5 ! judge p 0.333
        # Filtering leaves whatever idioms happen to land on a palette cell -- the long
        # tail, the exact failure `_candidates` weights by frequency to avoid. A flat boost
        # overshoots the other way, past the human 0.909, into the "more human than human"
        # range `VOCAB_DEPTH` warns about. ⇒**Boost only candidates that are ALREADY at or
        # above the median frequency of this state's candidates**, so the palette can shift
        # the choice among common idioms and can never promote a rare one.
        if memory_boost > 1.0 and cands and played[color]:
            # the same median guard the palette needs: shift the choice among COMMON
            # idioms, never promote a rare one onto a remembered cell
            mid = sorted(w for _e, w, _x in cands)[len(cands) // 2]
            cands = [(e, w * (memory_boost
                              if w >= mid and (h.x + e[0], h.y + e[1], e[3]) in played[color]
                              else 1.0), x)
                     for e, w, x in cands]
        if palette is not None and cands:
            pal = palette.get(color, ())
            mid = sorted(w for _e, w, _x in cands)[len(cands) // 2]
            cands = [(e, w * (PALETTE_BOOST
                              if w >= mid and (h.x + e[0], h.y + e[1], e[3]) in pal else 1.0), x)
                     for e, w, x in cands]
        pick = _pick(cands, rng, prefer_cross=cross_ok, width=width)
        if pick is None and cross_ok:
            cands = _candidates(ranked, counts, h, min(dt, 2.0), spb, top_k, False,
                                travel_target, strict_parity)
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
            played[color].add((nx, ny, d_to))
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
    return _revocab(notes, out)


def _revocab(before: list[dict], after: list[dict]) -> list[dict]:
    """Re-pick the fixer's new directions INSIDE THEIR OWN PARITY CLASS, preferring the
    vocabulary.

    ★★**Why this is safe by construction.** Parity depends only on whether a direction is
    up-ish (`UP_DIRS`) or down-ish (`DOWN_DIRS`), so swapping 5 for 0 or 4 cannot change
    whether the swing alternates. The fixer's verdict about *which way the hand must go* is
    kept exactly; only *which of that class's directions* is re-chosen, by how often the
    human corpus plays the resulting transition. Notes the fixer did not touch are never
    considered, so a map it leaves alone comes back byte-identical.

    ⇒**The bug this closes** (2026-09-13w): `fix_parity` is vocabulary-blind, and on a
    parity-hostile map it rewrote 319-351 directions of 728, turning **0** out-of-vocabulary
    transitions into **114-149** and taking `idiom_coverage` from 0.99 to 0.53-0.59. Nothing
    downstream noticed -- the verdict page does not read that axis. The repair is not wrong to
    fire; it was choosing among directions without being told which ones humans use.
    """
    changed = {i for i, (o, n) in enumerate(zip(before, after)) if o.get("d") != n.get("d")}
    if not changed:
        return after
    counts, _ranked, _ = idm.load_vocab()
    out = [dict(n) for n in after]
    for color in {int(n.get("c", 0)) for n in out}:
        idx = sorted((i for i in range(len(out)) if int(out[i].get("c", 0)) == color),
                     key=lambda i: float(out[i].get("b", 0.0)))
        for a_i, b_i in zip(idx, idx[1:]):
            if b_i not in changed:
                continue
            dt = round(float(out[b_i]["b"]) - float(out[a_i]["b"]), 3)
            if dt <= 0 or dt > idm.MAX_DT:
                continue
            cls = idm.dt_class(dt)
            dx = int(out[b_i]["x"]) - int(out[a_i]["x"])
            dy = int(out[b_i]["y"]) - int(out[a_i]["y"])
            d_from, d_fix = int(out[a_i]["d"]), int(out[b_i]["d"])
            par = _parity_of(d_fix)
            if par is None:
                continue
            same = [d for d in (DOWN_DIRS if par == 0 else UP_DIRS)]
            best = max(same, key=lambda d: (counts.get((dx, dy, d_from, d, cls), 0),
                                            d == d_fix))
            if counts.get((dx, dy, d_from, best, cls), 0) > 0:
                out[b_i]["d"] = int(best)
    return out


def idiomize_zip(src: pathlib.Path, dst: pathlib.Path, *, seed: int = 0,
                 top_k: int = VOCAB_DEPTH, width: int = 3,
                 crossover: float = CROSSOVER_TARGET,
                 repeat_p: float = REPEAT_P,
                 travel_target: float = TRAVEL_TARGET,
                 palette: int = 0,
                 memory_boost: float = MEMORY_BOOST,
                 strict_parity: bool = STRICT_PARITY) -> tuple[int, int]:
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

        # ⚠️Every parameter added to `idiomize_zip` MUST be threaded into this call.
        # `width` was accepted here, advertised in --help, and never passed for weeks:
        # `--width 1` and `--width 12` produced byte-identical maps. `travel_target`
        # was inert for exactly the same reason the moment it was added, and was
        # caught only because the sweep printed three identical rows.
        # ★A knob whose arms are identical to 3 decimals is not a weak lever, it is an
        # UNWIRED one.
        new, nfb = idiomize(notes, bpm, seed=seed, top_k=top_k,
                            width=width, crossover=crossover, repeat_p=repeat_p,
                            travel_target=travel_target, memory_boost=memory_boost,
                            strict_parity=strict_parity)
        # ★TWO PASSES when a palette size is asked for: the first pass says which shapes
        # this song's rhythm actually reaches, the top `palette` of those become the
        # palette, and the second pass replays the map inside it. Deriving the palette
        # from the map's own first pass (rather than from a corpus list) keeps it a
        # mechanism: it commits to shapes this song was already going to play.
        if palette:
            new, nfb = idiomize(notes, bpm, seed=seed, top_k=top_k,
                                width=width, crossover=crossover, repeat_p=repeat_p,
                                travel_target=travel_target, memory_boost=memory_boost,
                                strict_parity=strict_parity,
                                palette=_palette_of(new, palette))
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
        # 🔴🔴**AND IT IS VOCABULARY-BLIND — a latent bug in the pipeline** (2026-09-13w).
        # It rewrites DIRECTIONS with no reference to the mined vocabulary, so a map that is
        # parity-hostile leaves here full of transitions no human plays. Measured on one song,
        # 4 seeds per arm: the control triggers **0** rewrites, and with `--map-memory 4` two
        # seeds trigger **319 and 351** — turning 0 out-of-vocabulary transitions into 114 and
        # 149, which is the whole of that flag's `idiom_coverage` collapse (0.99 → 0.53-0.59).
        # ⇒Any change that makes a map more parity-hostile silently degrades coverage and
        # nothing notices: the verdict page does not read it, only `mapjudge` would.
        # ✅CLOSED 2026-09-13x/y: `STRICT_PARITY` stops the sampler placing the resets the fixer
        # rewrites, and `_revocab` re-picks any rewrite it still makes inside its parity class.
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
    ap.add_argument("--width", type=int, default=3,
                    help="restrict the draw to the N most common cells; 1 = greedy "
                         "(least variety), 0 = no restriction. A VARIETY dial — was a "
                         "dead no-op before 2026-08-21")
    ap.add_argument("--top-k", type=int, default=VOCAB_DEPTH,
                    help="vocabulary depth (default %(default)s; see VOCAB_DEPTH)")
    ap.add_argument("--crossover", type=float, default=CROSSOVER_TARGET)
    ap.add_argument("--allow-resets", action="store_true",
                    help="let the sampler place a same-parity repeat when there is time to "
                         "re-cock. ⚠️`fix_parity` runs after this pass and removes every one, "
                         "so this only hands the direction choice to a vocabulary-blind "
                         "repair (see STRICT_PARITY)")
    ap.add_argument("--map-memory", type=float, default=MEMORY_BOOST, dest="map_memory",
                    help="prefer a landing this map has ALREADY played, as a weight on the "
                         "frequency-weighted draw (1.0 = off; see MEMORY_BOOST). Unlike "
                         "--palette nothing is decided in advance and no candidate is "
                         "removed, so it cannot force an out-of-vocabulary transition")
    ap.add_argument("--palette", type=int, default=0,
                    help="commit the map to N landing shapes per hand (0 = off, the "
                         "pre-2026-09-12 behaviour). Two passes: the first says which "
                         "shapes this song reaches, the top N become the palette, the "
                         "second replays inside it. 20 is the measured operating point "
                         "and realises ~33 shapes/hand, the human median being 28.5")
    ap.add_argument("--travel-target", type=float, default=TRAVEL_TARGET,
                    help="grid-units/sec the sampler prefers a move to cover. Higher "
                         "= wider reaches. Orthogonal to --width, which sets how many "
                         "candidates are considered rather than how far they go")
    ap.add_argument("--repeat-p", type=float, default=REPEAT_P,
                    help="chance a hand repeats a figure it just played "
                         "(0 = resample independently, which is what put "
                         "idiom_local at the 98th human percentile)")
    a = ap.parse_args()

    # 🔴🔴**`--travel-target` WAS DEAD HERE UNTIL 2026-09-12** — accepted by the parser,
    # documented in --help, and never passed on, which is the third time this exact bug
    # has shipped in this file (`width`, then `travel_target` inside `idiomize_zip`, now
    # `travel_target` at the CLI). ⚠️Any sweep of `--travel-target` run before today was
    # sweeping nothing; its arms were identical by construction.
    n, nfb = idiomize_zip(a.zip_in, a.out, seed=a.seed, top_k=a.top_k,
                          width=a.width, crossover=a.crossover,
                          repeat_p=a.repeat_p, travel_target=a.travel_target,
                          palette=a.palette, memory_boost=a.map_memory,
                          strict_parity=not a.allow_resets)
    print(f"{a.zip_in.name}: re-placed {n - nfb}/{n} notes from the human vocabulary "
          f"({nfb} kept their original cell: no idiom fit)")
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
