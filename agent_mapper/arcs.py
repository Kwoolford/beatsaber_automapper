#!/usr/bin/env python
"""ARCS — the second element we never emit, added without touching a single note.

★**Why arcs and not chains, first.** Measured over 51 v3 human maps (2026-08-19l):

| | arcs (`sliders`) | chains (`burstSliders`) |
|---|---|---|
| maps using it | **45/51 = 88 %** | 25/51 = 49 % |
| per map | median **48** (p10 15, p90 93) | median 16 |
| span | median **1.00 beat** (p90 2.00) | median 0.062 beats |
| shape | head and tail in **different cells 93 %** of the time | **4-5 slices** (51 % use 4) |

**An arc is ADDITIVE**: in v3 it is its own object drawn between two positions, so it can be laid
over a finished map without altering, moving or removing any note. **A chain is not** — it turns a
note into a head plus segments, which changes what the hand has to do, and therefore has to clear
the swing simulator before it is safe. That is why this module does arcs and leaves chains for a
build that can be parity-checked.

★**Chains remain the more interesting idea, though, and the measurement says why**: *a chain is
one swing carrying 4 segments — density without a new distinct time.* We currently buy density
with **doubles** (39.6 % of our vocal notes, against a human 20.7 %), which costs 21 % of the vocal
budget. Chains are how a human buys it instead.

## The rule
Connect **consecutive same-colour notes** whose gap is near the human median (1 beat) and whose
head and tail sit in different cells — an arc between two notes in the same cell is a hold, which
humans do only 7 % of the time. Nothing else in the map changes.

Usage:
    python agent_mapper/arcs.py in.zip --out out.zip
"""

from __future__ import annotations

import argparse
import json
import pathlib
import shutil
import tempfile
import zipfile

# Human medians, measured — see the table above.
SPAN = (0.5, 2.0)          # beats between head and tail; human median 1.00, p90 2.00
TARGET = 48                # arcs per map; human median
MIN_SPACING = 2.0          # beats between arcs on one hand, so they do not chain up


def _dat(names: list[str]) -> str | None:
    c = [n for n in names if n.lower().endswith(".dat") and "info" not in n.lower()]
    return next((n for n in c if "expert" in n.lower()), c[0] if c else None)


def plan_arcs(notes: list[dict], target: int = TARGET) -> list[dict]:
    """Arcs over consecutive same-colour note pairs. Adds nothing else, changes nothing."""
    out: list[dict] = []
    last: dict[int, float] = {}
    # ⚠️Budget PER HAND. A single shared counter fills the quota with whichever colour is
    # iterated first and leaves the other hand with zero arcs — caught in verification, where
    # all 48 landed on red.
    per_hand = max(target // 2, 1)
    for color in (0, 1):
        made = 0
        ns = sorted((n for n in notes if n.get("c", 0) == color), key=lambda n: n.get("b", 0.0))
        for a, b in zip(ns, ns[1:]):
            if made >= per_hand:
                break
            span = b.get("b", 0.0) - a.get("b", 0.0)
            if not (SPAN[0] <= span <= SPAN[1]):
                continue
            # an arc between two notes in the SAME cell is a hold — humans do that 7 % of the
            # time, so it is not the idiom to copy
            if a.get("x") == b.get("x") and a.get("y") == b.get("y"):
                continue
            if color in last and a.get("b", 0.0) - last[color] < MIN_SPACING:
                continue
            last[color] = a.get("b", 0.0)
            made += 1
            out.append({
                "b": round(float(a.get("b", 0.0)), 3), "c": color,
                "x": a.get("x", 0), "y": a.get("y", 0), "d": a.get("d", 1), "mu": 1.0,
                "tb": round(float(b.get("b", 0.0)), 3),
                "tx": b.get("x", 0), "ty": b.get("y", 0), "tc": b.get("d", 1), "tmu": 1.0,
                "m": 0,
            })
    out.sort(key=lambda o: o["b"])
    return out


def add_arcs(src: pathlib.Path, dst: pathlib.Path, target: int = TARGET) -> int:
    tmp = pathlib.Path(tempfile.mkdtemp(prefix="arcs_"))
    try:
        with zipfile.ZipFile(src) as zf:
            zf.extractall(tmp)
            names = zf.namelist()
        dat = _dat(names)
        if dat is None:
            raise ValueError("no difficulty .dat")
        f = tmp / dat
        d = json.loads(f.read_text(encoding="utf-8-sig"))
        if not str(d.get("version", "")).startswith("3"):
            raise ValueError("v3 only")
        notes = d.get("colorNotes") or []
        arcs = plan_arcs(notes, target)
        d["sliders"] = arcs
        f.write_text(json.dumps(d), encoding="utf-8")
        dst.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(dst, "w", zipfile.ZIP_DEFLATED) as zo:
            for p in sorted(tmp.rglob("*")):
                if p.is_file():
                    zo.write(p, p.relative_to(tmp).as_posix())
        return len(arcs)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("zip", type=pathlib.Path)
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--target", type=int, default=TARGET)
    a = ap.parse_args()
    print(f"{a.zip.name}: added {add_arcs(a.zip, a.out, a.target)} arcs -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
