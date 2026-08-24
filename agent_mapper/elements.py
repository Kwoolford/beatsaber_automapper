#!/usr/bin/env python
"""THE OTHER THREE ELEMENTS, MADE READABLE — walls, arcs and chains.

🔴🔴**THE GAP THIS CLOSES.** The agent can WRITE five map elements and could READ only
one. `walls.py`, `arcs.py` and `chains.py` generate obstacles, sliders and burst
sliders; **not one reading tool could see them back** -- not `map_view`, not
`notesheet`, not `overlay`, not `flowview`, and not `mapjudge`, whose 23 metrics move by
**exactly 0.000** when 89 walls + 90 arcs + 16 chains are added. An agent reading a
finished `[FULL]` map saw a notes-only map and had no way to know otherwise.

That is why *"is `[FULL]` less empty than `[V2]`?"* sat unanswerable for days: it is a
question about the three elements nothing could read.

★**Kyle, 2026-08-24:** *"beef up the tooling so the agent calling the skill has as much
visibility as a person playing and listening… don't need to rely on me to audit."* A
player SEES walls coming and moves out of the lane. This module is that channel.

**What a player perceives, and what this exposes:**
  * **which lanes are blocked right now** -- `lane_map()` renders the 4 columns as
    `██··`, exactly the dodge decision the player is making.
  * **how long they had to get out** -- `dodge_windows()`. A wall that appears under
    the hand with no warning is the difference between a dodge and a hit.
  * **notes trapped inside walls** -- `collisions()`. This has shipped before: wiring
    walls BEFORE `idiomize` put **12 notes inside walls** on the first attempt, and
    only a collision check caught it.
  * **arcs and chains as gestures**, not rows in a JSON file.

⚠️**Beat Saber v3 schema, straight from the map:**
  obstacle    {b, d(uration), x, y, w(idth), h(eight)}   lanes x .. x+w-1
  slider/arc  {b, c(olor), x, y, d, tb(tail beat), tx, ty, tc}
  burst/chain {b, c, x, y, d, tb, tx, ty, sc(segment count), s(quish)}
"""
from __future__ import annotations

import json
import pathlib
import zipfile

GRID_W = 4          # columns 0..3
GRID_H = 3          # rows 0..2


def load_elements(map_path: pathlib.Path) -> dict:
    """Walls, arcs, chains and notes from a map zip, as plain dicts.

    ⚠️Exact basename match and an explicit BPMInfo exclusion -- "BPMInfo.dat" also ends
    with "info.dat" and sorts FIRST in 73 of 300 corpus zips, where picking it yields a
    silent bpm of 120 and stretches every time in this module.
    """
    with zipfile.ZipFile(map_path) as zf:
        names = zf.namelist()
        info = next((n for n in names
                     if n.split("/")[-1].lower() == "info.dat"), None)
        # Prefer Expert(+)Standard, then any Standard, then ANY difficulty file.
        # ⚠️Plenty of corpus maps ship only Lawless/360/OneSaber, and demanding
        # "*standard.dat" silently skipped them -- over a 5 373-map corpus that is a
        # selection effect on the very reference this builds.
        def pick(pred):
            return next((n for n in names
                         if pred(n.split("/")[-1].lower())
                         and "bpminfo" not in n.lower()), None)
        diff = (pick(lambda b: b == "expertstandard.dat")
                or pick(lambda b: b == "expertplusstandard.dat")
                or pick(lambda b: b.endswith("standard.dat"))
                or pick(lambda b: b.endswith(".dat") and b != "info.dat"))
        if info is None or diff is None:
            # ⚠️ValueError, NOT SystemExit: SystemExit inherits BaseException and
            # sails straight through `except Exception`, killing any batch caller.
            raise ValueError(f"could not read a difficulty from {map_path}")
        meta = json.loads(zf.read(info).decode("utf-8-sig"))
        dat = json.loads(zf.read(diff).decode("utf-8-sig"))
    bpm = next((float(v) for k, v in meta.items()
                if "beatsperminute" in k.lower()), 120.0)
    offset = next((float(v) for k, v in meta.items()
                   if "songtimeoffset" in k.lower().replace("_", "")), 0.0)
    return {
        "bpm": bpm, "offset": offset,
        "notes": dat.get("colorNotes") or [],
        "bombs": dat.get("bombNotes") or [],
        "walls": dat.get("obstacles") or [],
        "arcs": dat.get("sliders") or [],
        "chains": dat.get("burstSliders") or [],
    }


def walls_at(walls, beat: float) -> list[dict]:
    """Every wall covering this beat. A wall spans [b, b+d)."""
    return [w for w in walls
            if float(w.get("b", 0)) <= beat < float(w.get("b", 0))
            + max(float(w.get("d", 0)), 1e-9)]


def lane_map(walls, beat: float) -> str:
    """The 4 columns as `██··` — which lanes are blocked at this instant.

    ★This is the player's actual decision surface: not "there are 89 walls" but "right
    now, can I stand here?"
    """
    blocked = [False] * GRID_W
    for w in walls_at(walls, beat):
        x = int(w.get("x", 0))
        for c in range(x, min(x + max(int(w.get("w", 1)), 1), GRID_W)):
            if c >= 0:
                blocked[c] = True
    return "".join("█" if b else "·" for b in blocked)


def collisions(elems: dict) -> list[dict]:
    """Notes that sit INSIDE a wall — unplayable, and it has shipped before.

    A wall occupies lanes x..x+w-1 and rows y..y+h-1 for its duration. A note in that
    box at that time cannot be hit.
    """
    out = []
    for n in elems["notes"]:
        nb, nx, ny = float(n.get("b", 0)), int(n.get("x", 0)), int(n.get("y", 0))
        for w in walls_at(elems["walls"], nb):
            wx, wy = int(w.get("x", 0)), int(w.get("y", 0))
            ww, wh = max(int(w.get("w", 1)), 1), max(int(w.get("h", 1)), 1)
            if wx <= nx < wx + ww and wy <= ny < wy + wh:
                out.append({"beat": nb, "x": nx, "y": ny, "color": int(n.get("c", 0)),
                            "wall_beat": float(w.get("b", 0)),
                            "wall_dur": float(w.get("d", 0))})
                break
    return out


def dodge_windows(elems: dict) -> list[dict]:
    """For each wall, how long the player had to leave the lane it blocks.

    ★**What a player feels.** The warning is the gap between the last note that put a
    hand in a lane this wall blocks and the moment the wall arrives. A short window is
    a wall that appears under the hand.
    ⚠️Reported in SECONDS, because reaction time is wall-clock, not musical.
    """
    spb = 60.0 / max(elems["bpm"], 1e-6)
    notes = sorted(elems["notes"], key=lambda n: float(n.get("b", 0)))
    out = []
    for w in elems["walls"]:
        wb = float(w.get("b", 0))
        wx, ww = int(w.get("x", 0)), max(int(w.get("w", 1)), 1)
        lanes = set(range(wx, wx + ww))
        prev = None
        for n in notes:
            nb = float(n.get("b", 0))
            if nb >= wb:
                break
            if int(n.get("x", 0)) in lanes:
                prev = nb
        out.append({"wall_beat": wb, "lanes": sorted(lanes),
                    "window_s": None if prev is None else (wb - prev) * spb})
    return out


def summary(elems: dict) -> dict:
    """The one-screen element audit: is this map actually wearing its other elements?"""
    spb = 60.0 / max(elems["bpm"], 1e-6)
    walls, arcs, chains = elems["walls"], elems["arcs"], elems["chains"]
    last = max([float(n.get("b", 0)) for n in elems["notes"]] or [0.0])
    dur_beats = max(last, 1.0)
    blocked = sum(max(float(w.get("d", 0)), 0.0) for w in walls)
    cols = collisions(elems)
    wins = [d["window_s"] for d in dodge_windows(elems) if d["window_s"] is not None]
    tight = [w for w in wins if w < 0.5]
    return {
        "walls": len(walls), "arcs": len(arcs), "chains": len(chains),
        "notes": len(elems["notes"]), "bombs": len(elems["bombs"]),
        "wall_beats_blocked": round(blocked, 1),
        "wall_duty": round(blocked / dur_beats, 4),
        "walls_per_min": round(len(walls) / max(dur_beats * spb / 60.0, 1e-6), 1),
        "notes_in_walls": len(cols),
        "chain_segments": sum(int(c.get("sc", 0)) for c in chains),
        "arc_share_of_notes": round(len(arcs) / max(len(elems["notes"]), 1), 4),
        "tight_dodges_lt_0p5s": len(tight),
        "min_dodge_s": round(min(wins), 2) if wins else None,
    }


# ---------------------------------------------------------------------------
# JUDGING, not just reading. `summary()` reports what the map HAS; this places
# each quantity against what human mappers do, which is the difference between a
# number and a verdict. Reference built by `scripts/calibrate_elements.py`.
# ---------------------------------------------------------------------------

REFERENCE_PATH = pathlib.Path(__file__).resolve().parents[1] / "outputs" / "element_reference.json"

# What "too far" means per quantity. ★These are NOT symmetric, because the defects are
# not symmetric: too FEW walls is the shipped defect (we emitted zero for months), too
# many is a style. And a note inside a wall is unplayable at ANY count.
_LOW_IS_BAD = {"walls", "arcs", "chains", "wall_duty", "walls_per_min",
               "arc_share_of_notes", "chain_segments"}


def load_reference() -> dict | None:
    if not REFERENCE_PATH.exists():
        return None
    return json.loads(REFERENCE_PATH.read_text())


def _pct_of(dist: dict, v: float) -> float:
    """Approximate human percentile of `v` by interpolating the stored quantiles."""
    pts = [(0.05, dist["p5"]), (0.25, dist["p25"]), (0.50, dist["median"]),
           (0.75, dist["p75"]), (0.95, dist["p95"])]
    if v <= pts[0][1]:
        return 0.05
    if v >= pts[-1][1]:
        return 0.95
    for (p0, v0), (p1, v1) in zip(pts, pts[1:]):
        if v0 <= v <= v1:
            if v1 == v0:
                return p1
            return p0 + (p1 - p0) * (v - v0) / (v1 - v0)
    return 0.5


def judge(elems: dict, reference: dict | None = None) -> dict:
    """Place this map's elements against the human corpus, with a verdict per line.

    ★**This is the piece that lets the agent audit its own map.** Reading `wall_duty
    0.042` says nothing; *"3rd percentile, human median 0.152, our walls are far too
    short"* is an instruction.
    """
    reference = reference or load_reference()
    s = summary(elems)
    if not reference:
        return {"summary": s, "lines": [], "note": "no element reference; run "
                "scripts/calibrate_elements.py"}
    dist, present = reference.get("dist", {}), reference.get("present", {})
    lines = []
    for k, d in dist.items():
        if k not in s or s[k] is None:
            continue
        v = float(s[k])
        pct = _pct_of(d, v)
        bad = (k in _LOW_IS_BAD and pct <= 0.05) or (k == "notes_in_walls" and v > 0)
        warn = k in _LOW_IS_BAD and 0.05 < pct <= 0.25
        lines.append({"key": k, "value": v, "pct": pct, "median": d["median"],
                      "flag": "🔴" if bad else ("🟡" if warn else "✅")})
    # 🔴Presence is separate from amount: shipping ZERO of an element 96 % of human
    # maps carry is the defect that went unnoticed for months, and no percentile over
    # "maps that use it" can see it.
    for k, share in present.items():
        if s.get(k, 0) == 0 and share >= 0.5:
            lines.append({"key": f"{k}_present", "value": 0, "pct": 0.0,
                          "median": share, "flag": "🔴"})
    return {"summary": s, "lines": lines}


def format_judgement(j: dict) -> list[str]:
    out = ["", f"{'element quantity':22s}{'ours':>10}{'human med':>11}{'pct':>7}  "]
    out.append("─" * 56)
    for ln in j["lines"]:
        out.append(f"{ln['flag']} {ln['key']:20s}{ln['value']:>10.4g}"
                   f"{ln['median']:>11.4g}{100 * ln['pct']:>6.0f}%")
    if j.get("note"):
        out.append(f"  ⚠️{j['note']}")
    return out
