#!/usr/bin/env python
"""How much of a KNOWN flaw does the outlier sweep see? Inject flaws into held-out human maps at a
dose, and measure detection — map-wide (any flag on the flaw's own features) and local (windows).

A tool that only ever rediscovers what it was shown is anecdote. This puts a number on its reach:
for each flaw kind and dose, the share of 150 human maps on which the injected flaw is FLAGGED,
against the share on which the same features flag with nothing injected (the false-alarm floor).

    python scripts/sweep_sensitivity.py [n_maps]
"""
from __future__ import annotations

import copy
import json
import pathlib
import random
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor

import numpy as np

warnings.filterwarnings("ignore")
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))

UP2DOWN = {0: 1, 1: 0, 4: 7, 7: 4, 5: 6, 6: 5}
Note = None


def _cp(m):
    from agent_mapper.score import Note as N
    w = copy.copy(m)
    w.notes = [N(n.beat, n.x, n.y, n.color, n.direction) for n in m.notes]
    return w


def inj_collide(m, dose, rng):
    from agent_mapper.score import Note as N
    w = _cp(m)
    for n in rng.sample(w.notes, max(1, int(dose))):
        w.notes.append(N(n.beat, n.x, n.y, 1 - n.color, n.direction))
    return w


def inj_inward(m, dose, rng):
    w = _cp(m)
    for n in w.notes:
        if n.direction == 0 and rng.random() < dose:
            n.direction = 5 if n.color == 0 else 4
    return w


def inj_parity(m, dose, rng):
    w = _cp(m)
    for n in w.notes:
        if n.direction in UP2DOWN and rng.random() < dose:
            n.direction = UP2DOWN[n.direction]
    return w


def inj_offgrid(m, dose, rng):
    w = _cp(m)
    for n in w.notes:
        if rng.random() < dose:
            n.beat += 0.25
    w.notes.sort(key=lambda n: (n.beat, n.color))
    return w


def inj_halves(m, dose, rng):
    w = _cp(m)
    for n in w.notes:
        if rng.random() < dose:
            if n.color == 0 and n.x > 1:
                n.x = n.x - 2
            elif n.color == 1 and n.x < 2:
                n.x = n.x + 2
    return w


def inj_doubles(m, dose, rng):
    from agent_mapper.score import Note as N
    w = _cp(m)
    at = {}
    for n in w.notes:
        at.setdefault(round(n.beat * 48), []).append(n)
    for t, ns in at.items():
        if len(ns) == 1 and rng.random() < dose:
            n = ns[0]
            w.notes.append(N(n.beat, 3 - n.x, n.y, 1 - n.color, n.direction))
    w.notes.sort(key=lambda n: (n.beat, n.color))
    return w


def inj_gap(m, dose, rng):
    """Delete `dose` bars of notes from the middle of the song (a dead section)."""
    w = _cp(m)
    last = max(n.beat for n in w.notes)
    a = rng.uniform(0.3, 0.6) * last
    w.notes = [n for n in w.notes if not (a <= n.beat < a + 4 * dose)]
    return w


def inj_vocab(m, dose, rng):
    """Collapse a share of each hand's notes onto its most common cell for that parity."""
    w = _cp(m)
    import collections
    top = {}
    for c in (0, 1):
        for par, ds in (("u", {0, 4, 5}), ("d", {1, 6, 7})):
            cc = collections.Counter((n.x, n.y, n.direction) for n in w.notes if n.color == c and n.direction in ds)
            if cc:
                top[(c, par)] = cc.most_common(1)[0][0]
    for n in w.notes:
        par = "u" if n.direction in (0, 4, 5) else "d" if n.direction in (1, 6, 7) else None
        if par and (n.color, par) in top and rng.random() < dose:
            n.x, n.y, n.direction = top[(n.color, par)]
    return w


FLAWS = {  # name: (injector, doses, the features that SHOULD carry it)
    "collide": (inj_collide, [1, 3, 10], ["cell_collisions", "two_colour_cells"]),
    "inward": (inj_inward, [0.1, 0.25, 0.5], ["L_dir_UR", "R_dir_UL", "L_rot", "R_rot"]),
    "parity": (inj_parity, [0.03, 0.08, 0.2], ["_reset", "_fast_reset", "_rot0", "_rot"]),
    "offgrid": (inj_offgrid, [0.05, 0.15, 0.3], ["on_16th_odd", "on_beat", "on_8th_off", "gap_"]),
    "halves": (inj_halves, [0.5, 0.8, 1.0], ["_col", "crossover", "cell_entropy", "dbl_dx"]),
    "doubles": (inj_doubles, [0.1, 0.25, 0.5], ["double_share", "notes_per_event", "dbl_", "alternation",
                                                 "events_ps"]),
    "gap": (inj_gap, [4, 8, 16], ["longest_gap", "gap_long", "win_"]),
    "vocab": (inj_vocab, [0.3, 0.6, 0.9], ["shapes", "cell_entropy", "top5", "_col", "_row", "_move"]),
}


def _relevant(nm, keys):
    # two_colour_cells contains "_col": only the collide flaw may claim it (2026-10-05: it inflated
    # vocab 55 % -> 85 % because collapsing cells also stacks colours)
    if nm == "two_colour_cells" and "two_colour_cells" not in keys:
        return False
    return any(k in nm for k in keys)


def run_one(args):
    path, seed = args
    from agent_mapper.score import load_map
    from agent_mapper.fingerprint import fingerprint
    import outlier_sweep as O
    names, X, ids = O.load_table()
    ln, lX, llnps, lids = O.lload()
    try:
        m = load_map(pathlib.Path(path))
    except Exception:
        return None
    if len(m.notes) < 200 or any(not (0 <= n.x <= 3 and 0 <= n.y <= 2 and 0 <= n.direction <= 8) for n in m.notes):
        return None
    sid = pathlib.Path(path).stem
    rng = random.Random(seed)

    def read(mm, keys):
        fl = O.flags(O.place(fingerprint(mm), names, X, ids, exclude=sid))
        rel = [r for r in fl if _relevant(r[0], keys)]
        wins = 0
        for b, f in O.windows(mm):
            lf = O.lplace(f, ln, lX, llnps, lids, exclude=sid)
            wins += any(_relevant(r[0], keys) for r in lf)
        return dict(map=bool(rel), nflags=len(fl), win=wins)

    out = {}
    for fname, (fn, doses, keys) in FLAWS.items():
        out[fname] = {"base": read(m, keys)}
        for d in doses:
            out[fname][str(d)] = read(fn(m, d, rng), keys)
    return out


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 150
    import outlier_sweep as O
    used = {r["id"] for r in json.loads(O.LTABLE.read_text())}   # not in the window table
    zips = [z for z in sorted((ROOT / "data/raw").glob("*.zip")) if z.stem not in used]
    rng = np.random.default_rng(7)
    zips = [str(zips[i]) for i in rng.choice(len(zips), size=min(n * 2, len(zips)), replace=False)]
    with ProcessPoolExecutor(16) as ex:
        res = [r for r in ex.map(run_one, [(z, i) for i, z in enumerate(zips)], chunksize=2) if r][:n]
    lctl = json.loads((O.TABLE.parent / "lcontrol.json").read_text())["flagged_windows_per_map"]["95"]
    print(f"{len(res)} held-out human maps (map-wide: any flag on the flaw's features; local: >= 2 windows"
          f" flagged on them)")
    summary = {}
    for fname, (fn, doses, keys) in FLAWS.items():
        cells = []
        for d in ["base"] + [str(x) for x in doses]:
            mp = np.mean([r[fname][d]["map"] for r in res])
            lw = np.mean([r[fname][d]["win"] >= 2 for r in res])
            cells.append(f"{d:>5}: map {mp:4.0%} local {lw:4.0%}")
            summary.setdefault(fname, {})[d] = dict(map=float(mp), local=float(lw))
        print(f"  {fname:8} " + " | ".join(cells))
    (O.TABLE.parent / "sensitivity.json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
