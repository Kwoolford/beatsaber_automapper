#!/usr/bin/env python
"""Reachability of note-to-note transitions — the real K2, per Kyle 2026-08-03.

His correction, after seeing a global diagonal thin:

    "I don't like the global thin diagonal, they can be fun in fast passages, but
     not outside corner in swings followed by another swing that's hard to reach.
     [...] They should still be playable though that's the core problem not that
     they are diagonal."

So diagonal *share* was the wrong target. A diagonal is only a problem when it
leaves the hand somewhere the next note punishes, and those two things come
apart: a diagonal in open space is fine, a corner diagonal before a far note is
not — and a *non*-diagonal can be just as bad.

**The model.** A cut carries the saber through the note in the cut direction, so
after cutting at `p` with direction `d` the hand is around `p + d`. The cost of
the next same-hand note is the travel from **there**, not from `p`. A corner note
cut outward puts `p + d` outside the grid entirely, which is exactly the
"outside corner in-swing" he named.

  reach          grid distance from the follow-through point to the next
                 same-hand note.
  reach_rate     share of transitions needing reach >= HARD_REACH within
                 HARD_SEC — far AND soon, which is what makes it unplayable
                 rather than merely wide.
  corner_exit    share of notes cut from an outer column in a direction that
                 carries the hand further outward. The specific pattern he named.

Reported against the human corpus, since "playable" has to mean "as reachable as
what people actually play", not zero.

Usage:
    python scripts/eval_reachability.py --maps 'outputs/eval_sweep_cache/arm#s0__*.zip'
"""

from __future__ import annotations

import argparse
import glob
import json
import pathlib
import random
import shutil
import statistics as st
import sys
import tempfile
import zipfile

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from beatsaber_automapper.data.beatmap import (  # noqa: E402
    parse_difficulty_dat, parse_info_dat,
)
from beatsaber_automapper.evaluation import scorecard  # noqa: E402

# dx, dy per cut direction (y increases upward). 8 = dot, no follow-through.
DIRV = {0: (0, 1), 1: (0, -1), 2: (-1, 0), 3: (1, 0),
        4: (-1, 1), 5: (1, 1), 6: (-1, -1), 7: (1, -1), 8: (0, 0)}
DIAGONAL = (4, 5, 6, 7)
HARD_REACH = 3.0
HARD_SEC = 0.30


def load_expert(zp: pathlib.Path):
    """Strict Expert + exact Info.dat — both loader traps found 2026-08-03."""
    tmp = pathlib.Path(tempfile.mkdtemp(prefix="reach_"))
    try:
        with zipfile.ZipFile(zp) as zf:
            names = zf.namelist()
            info = next((n for n in names
                         if n.split("/")[-1].lower() == "info.dat"), None)
            diff = next((n for n in names
                         if n.split("/")[-1].lower() == "expertstandard.dat"), None)
            if info is None or diff is None:
                return None
            for n in (info, diff):
                (tmp / pathlib.Path(n).name).write_bytes(zf.read(n))
        meta = parse_info_dat(tmp / pathlib.Path(info).name)
        bm = parse_difficulty_dat(tmp / pathlib.Path(diff).name)
        if meta is None or bm is None or len(bm.color_notes) < 100:
            return None
        return bm, float(meta.bpm)
    except Exception:  # noqa: BLE001
        return None
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def reach_metrics(beatmap, bpm: float) -> dict | None:
    if bpm <= 0:
        return None
    spb = 60.0 / bpm
    notes = sorted(beatmap.color_notes, key=lambda n: n.beat)
    if len(notes) < 100:
        return None

    reaches, hard, corner_exits, diag_hard, n_diag = [], 0, 0, 0, 0
    for color in (0, 1):
        hand = [n for n in notes if n.color == color]
        for a, b in zip(hand, hand[1:]):
            dx, dy = DIRV.get(a.direction, (0, 0))
            ex, ey = a.x + dx, a.y + dy          # follow-through point
            r = float(np.hypot(b.x - ex, b.y - ey))
            dt = (b.beat - a.beat) * spb
            reaches.append(r)
            tough = r >= HARD_REACH and 0 < dt <= HARD_SEC
            hard += int(tough)
            if a.direction in DIAGONAL:
                n_diag += 1
                diag_hard += int(tough)
            # "outside corner in-swing": outer column, cut carrying it further out
            if (a.x == 0 and dx < 0) or (a.x == 3 and dx > 0):
                corner_exits += 1

    n_tr = len(reaches)
    if n_tr < 50:
        return None
    return {
        "reach_median": round(float(np.median(reaches)), 4),
        "reach_p90": round(float(np.percentile(reaches, 90)), 4),
        "hard_rate": round(hard / n_tr, 4),
        "corner_exit_rate": round(corner_exits / n_tr, 4),
        "hard_given_diagonal": round(diag_hard / n_diag, 4) if n_diag else None,
        "n_transitions": n_tr,
    }


def report(rows: list[dict], label: str) -> dict:
    if not rows:
        print(f"{label}: nothing scored")
        return {}
    print(f"\n=== {label} (n={len(rows)}) ===")
    out = {}
    for k in ("reach_median", "reach_p90", "hard_rate", "corner_exit_rate",
              "hard_given_diagonal"):
        v = [r[k] for r in rows if r.get(k) is not None]
        if not v:
            continue
        out[k] = round(st.median(v), 4)
        print(f"  {k:22s} median {st.median(v):.4f}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--maps", action="append", default=[])
    ap.add_argument("--label", action="append", default=[])
    ap.add_argument("--human", type=int, default=0, help="N human Expert maps")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()

    out = {}
    for i, g in enumerate(a.maps):
        lab = a.label[i] if i < len(a.label) else f"cohort{i}"
        rows = []
        for p in sorted(glob.glob(g)):
            try:
                L = scorecard._load_any(pathlib.Path(p))
            except Exception:  # noqa: BLE001
                continue
            if not L:
                continue
            r = reach_metrics(L[0], L[1])
            if r:
                rows.append(r)
        out[lab] = report(rows, lab.upper())

    if a.human:
        raws = sorted((REPO / "data" / "raw").glob("*.zip"))
        random.Random(0).shuffle(raws)
        rows, n = [], 0
        for zp in raws:
            if n >= a.human:
                break
            L = load_expert(zp)
            if not L:
                continue
            n += 1
            r = reach_metrics(L[0], L[1])
            if r:
                rows.append(r)
        out["human"] = report(rows, "HUMAN (strict Expert)")

    if len(out) >= 2:
        labs = list(out)
        print("\n=== COMPARISON ===")
        keys = ["reach_median", "reach_p90", "hard_rate", "corner_exit_rate",
                "hard_given_diagonal"]
        print(f"{'metric':24s}" + "".join(f"{l[:13]:>15s}" for l in labs))
        for k in keys:
            print(f"{k:24s}" + "".join(
                f"{out[l][k]:>15.4f}" if out[l].get(k) is not None else f"{'--':>15s}"
                for l in labs))
        print(f"\nhard = reach >= {HARD_REACH} grid units within {HARD_SEC}s "
              "(far AND soon).")
        print("If hard_rate matches human while diagonal share does not, then the")
        print("diagonals are NOT the playability problem and thinning them is the")
        print("wrong lever -- keep it as a stylistic knob instead.")

    if a.json:
        pathlib.Path(a.json).write_text(json.dumps(out, indent=2))
        print(f"\nwrote {a.json}")


if __name__ == "__main__":
    main()
