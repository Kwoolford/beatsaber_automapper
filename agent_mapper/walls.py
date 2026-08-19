#!/usr/bin/env python
"""WALLS — the element we have never emitted, and 93 % of human maps have.

★**Why this exists.** Measured 2026-08-19j over 147 paired maps: **137 of 147 human maps contain
walls (median 86 per map) and we emit ZERO in every map we have ever produced.** On Fallen Kingdom
the human map has **124** obstacles and ours has **0** — and that is the song Kyle called *"really
empty"*, a complaint five separate instruments have failed to explain, every one of them by
looking at notes. A map missing an entire physical layer would feel empty at any note count.

This is deliberately a **post-processor on a finished zip**, not a generator change: it can be
given to his ear as a `[WALLS]` arm without touching a single note of the map it is compared to,
so the A/B isolates exactly one thing.

## The vocabulary it copies, measured from 135 human maps (16,504 vanilla walls)

⚠️**54 % of the corpus's walls are MODDED** (Mapping/Noodle Extensions repurpose the fields —
negative durations, lane −4750, width 1000) **and are discarded.** Reading them as vanilla gives a
median duration of *minus 2.5 beats*, which is how this was caught.

| property | human |
|---|---|
| walls per map | median **84** (p10 19, p90 222) |
| duration | median **0.12** beats (p90 1.25) |
| width | **90 % are 1 lane** |
| lane | **52 % at x=0, 41 % at x=3** — 93 % in an OUTER lane |
| height | 62 % crouch, 38 % full |
| notes inside a wall's own lanes while it is active | median **0.000**, any overlap only **8 %** |

⇒The idiom is a **short, one-lane wall hugging an outer edge, in a lane the hands are not using**.
That last line is the hard constraint: a wall where a note is is unplayable, and humans essentially
never do it.

Usage:
    python agent_mapper/walls.py in.zip --out out.zip
    python agent_mapper/walls.py in.zip --out out.zip --per-map 84 --seed 0
"""

from __future__ import annotations

import argparse
import json
import pathlib
import shutil
import tempfile
import zipfile

import numpy as np

# Measured human medians (vanilla only) — see the table above.
# ⚠️Sampled log-uniformly between the human p10 and p90 — NOT between the median and p90,
# which cannot reproduce the median (that first attempt gave 0.38 against a human 0.12).
DUR_BEATS = (0.03, 1.25)      # (p10, p90); human median 0.12, this yields ≈0.19
OUTER_LANES = (0, 3)
SAFETY_BEATS = 0.30           # keep this clear of any note in the wall's lane, both sides
MIN_GAP_BEATS = 2.0           # do not stack walls on top of each other in one lane


def _difficulty_dat(names: list[str]) -> str | None:
    cands = [n for n in names if n.lower().endswith(".dat") and "info" not in n.lower()]
    return next((n for n in cands if "expert" in n.lower()), cands[0] if cands else None)


def plan_walls(note_beats: np.ndarray, note_x: np.ndarray, span: tuple[float, float],
               n_walls: int, rng: np.random.Generator) -> list[dict]:
    """Choose wall placements that no note collides with.

    ⚠️The collision test is per LANE and includes a `SAFETY_BEATS` margin either side: a wall
    that ends a hair before a note in the same lane is still a wall the player is inside when
    they have to swing there.
    """
    b0, b1 = span
    out: list[dict] = []
    placed: dict[int, list[float]] = {x: [] for x in OUTER_LANES}
    # candidate starts on the half-beat, shuffled so the walls do not march in lockstep
    cands = np.arange(b0 + 4.0, max(b1 - 4.0, b0 + 4.0), 0.5)
    rng.shuffle(cands)
    for start in cands:
        if len(out) >= n_walls:
            break
        lane = int(OUTER_LANES[rng.integers(0, len(OUTER_LANES))])
        dur = float(np.exp(rng.uniform(np.log(DUR_BEATS[0]), np.log(DUR_BEATS[1]))))
        lo, hi = start - SAFETY_BEATS, start + dur + SAFETY_BEATS
        # any note in THIS lane overlapping the wall (plus margin)?
        m = (note_beats >= lo) & (note_beats <= hi) & (note_x == lane)
        if m.any():
            continue
        if any(abs(start - p) < MIN_GAP_BEATS for p in placed[lane]):
            continue
        placed[lane].append(float(start))
        # 62 % crouch (h=3, sitting high) / 38 % full — matching the human split
        crouch = rng.random() < 0.62
        out.append({"b": round(float(start), 3), "d": round(dur, 3), "x": lane, "w": 1,
                    "y": 2 if crouch else 0, "h": 3 if crouch else 5})
    out.sort(key=lambda o: o["b"])
    return out


def add_walls(src: pathlib.Path, dst: pathlib.Path, per_map: int = 84,
              seed: int = 0) -> int:
    """Copy `src` to `dst` with walls added to its Expert difficulty. Returns the count."""
    rng = np.random.default_rng(seed)
    tmp = pathlib.Path(tempfile.mkdtemp(prefix="walls_"))
    try:
        with zipfile.ZipFile(src) as zf:
            zf.extractall(tmp)
            names = zf.namelist()
        dat = _difficulty_dat(names)
        if dat is None:
            raise ValueError("no difficulty .dat in the zip")
        f = tmp / dat
        d = json.loads(f.read_text(encoding="utf-8-sig"))
        if not str(d.get("version", "")).startswith("3"):
            raise ValueError("only v3 maps are supported (ours are 3.3.0)")
        notes = d.get("colorNotes") or []
        if len(notes) < 20:
            raise ValueError("too few notes to place walls around")
        nb = np.array([n.get("b", 0.0) for n in notes], dtype=float)
        nx = np.array([n.get("x", 0) for n in notes], dtype=int)
        walls = plan_walls(nb, nx, (float(nb.min()), float(nb.max())), per_map, rng)
        d["obstacles"] = walls
        f.write_text(json.dumps(d), encoding="utf-8")
        dst.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(dst, "w", zipfile.ZIP_DEFLATED) as zo:
            for p in sorted(tmp.rglob("*")):
                if p.is_file():
                    zo.write(p, p.relative_to(tmp).as_posix())
        return len(walls)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("zip", type=pathlib.Path)
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--per-map", type=int, default=84, help="human median is 84")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    n = add_walls(a.zip, a.out, a.per_map, a.seed)
    print(f"{a.zip.name}: added {n} walls -> {a.out}")
    if n < a.per_map:
        print(f"  (asked for {a.per_map}; the rest had no note-free slot — that is the "
              f"constraint working, not a failure)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
