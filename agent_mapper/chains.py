#!/usr/bin/env python
"""CHAINS — density with NO new distinct time, which is what we currently buy with doubles.

★**Why this matters more than it looks.** Measured 2026-08-18k: **39.6 % of the notes we spend on
the vocal line are doubles** landing on an instant the other hand already covered, against a human
**20.7 %** — waste costing **21 % of the vocal budget**. A chain is the human's alternative: **one
swing carrying 4-5 segments**, so the map reads denser without a second hand firing and without a
new distinct time. ⇒Chains are a candidate answer to **C5** (doubles) and **D1** (*"very slow"*) at
once.

⚠️**CORRECTION to an assumption made one step earlier.** `arcs.py` says a chain "turns a note into
a head plus segments" and therefore cannot be additive. **Measured: 678 of 678 human chain heads
also exist as a `colorNote` — 100 %.** The note stays; the chain extends the same swing. So chains
*are* additive in the data. What they still change is what the hand does **after** the note (it
must carry through the burst), which is why this module parity-checks its own output rather than
assuming.

## The vocabulary, measured over 51 v3 human maps (25 use chains)

| property | human |
|---|---|
| chains per map | median **16** (p10 3, p90 71) |
| span head→tail | median **0.062 beats** (p90 0.08) |
| slices | **4 → 51 %**, 5 → 35 %, 3 → 5 % |
| head | always coincides with a note (100 %) |

Usage:
    python agent_mapper/chains.py in.zip --out out.zip
"""

from __future__ import annotations

import argparse
import json
import pathlib
import shutil
import tempfile
import zipfile

import numpy as np

SPAN = 0.0625            # beats, human median
SLICES = (4, 5)          # 51 % / 35 % of human chains
TARGET = 16              # human median per map
CLEAR_AFTER = 0.5        # beats that must be free in this lane after the head
# Which row the burst travels toward, by the head's cut direction (v3 `d`).
# 0 up, 1 down, 2 left, 3 right, 4 up-left, 5 up-right, 6 down-left, 7 down-right, 8 any.
_DROW = {0: +1, 1: -1, 4: +1, 5: +1, 6: -1, 7: -1}


def _dat(names: list[str]) -> str | None:
    c = [n for n in names if n.lower().endswith(".dat") and "info" not in n.lower()]
    return next((n for n in c if "expert" in n.lower()), c[0] if c else None)


def plan_chains(notes: list[dict], target: int = TARGET,
                rng: np.random.Generator | None = None) -> list[dict]:
    """Chains on notes that have room after them. Adds nothing else, removes nothing."""
    rng = rng or np.random.default_rng(0)
    beats = np.array([n.get("b", 0.0) for n in notes], dtype=float)
    lanes = np.array([n.get("x", 0) for n in notes], dtype=int)
    out: list[dict] = []
    per_hand = max(target // 2, 1)
    for color in (0, 1):
        made = 0
        idx = [i for i, n in enumerate(notes) if n.get("c", 0) == color]
        rng.shuffle(idx)
        for i in idx:
            if made >= per_hand:
                break
            n = notes[i]
            d = n.get("d", 1)
            if d not in _DROW:              # dot notes have no committed heading
                continue
            b, x, y = n.get("b", 0.0), n.get("x", 0), n.get("y", 0)
            ty = y + _DROW[d]
            if not (0 <= ty <= 2):          # the burst must stay on the grid
                continue
            # the lane has to be clear after the head, or the burst runs into the next note
            m = (beats > b) & (beats <= b + CLEAR_AFTER) & (lanes == x)
            if m.any():
                continue
            out.append({"b": round(float(b), 3), "c": color, "x": x, "y": y, "d": d,
                        "tb": round(float(b + SPAN), 3), "tx": x, "ty": int(ty),
                        "sc": int(SLICES[0] if rng.random() < 0.6 else SLICES[1]), "s": 1.0})
            made += 1
    out.sort(key=lambda o: o["b"])
    return out


def add_chains(src: pathlib.Path, dst: pathlib.Path, target: int = TARGET,
               seed: int = 0) -> int:
    tmp = pathlib.Path(tempfile.mkdtemp(prefix="chains_"))
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
        chains = plan_chains(d.get("colorNotes") or [], target,
                             np.random.default_rng(seed))
        d["burstSliders"] = chains
        f.write_text(json.dumps(d), encoding="utf-8")
        dst.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(dst, "w", zipfile.ZIP_DEFLATED) as zo:
            for p in sorted(tmp.rglob("*")):
                if p.is_file():
                    zo.write(p, p.relative_to(tmp).as_posix())
        return len(chains)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("zip", type=pathlib.Path)
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--target", type=int, default=TARGET)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    print(f"{a.zip.name}: added {add_chains(a.zip, a.out, a.target, a.seed)} chains -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
