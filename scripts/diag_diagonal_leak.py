#!/usr/bin/env python
"""WHERE inside the sampler do the diagonals go?

P1.2 is the largest non-circular gap: `vertical 0.773 vs 0.480 · diagonal 0.223 vs
0.415`. Four explanations are already refuted -- the vocabulary HAS the diagonals
(34.4 % stationary), 0 fallbacks fire, `_reparity` changes 0 directions, and doubles
are not the cause. TODO records the loss as living **"inside the sampler and NOT yet
explained"**, and names two untested suspects: the crossover filter and `REPEAT_P`.

★**This traces the candidate pool through EVERY stage instead of testing the two
suspects one at a time**, because a leak is a place, not a hypothesis. Per note, per
hand, the diagonal share of `d_to` is recorded at:

    A  after `_candidates`   -- d_from match + reachability + parity + cross filter
    B  after the REPEAT_P restriction to recently-played figures
    C  after `prefer_cross` narrowing to crossing candidates
    D  after the `width` truncation to the most FREQUENT candidates
    E  the direction actually picked

⚠️**`width` is a third suspect TODO does not name, and it is the most mechanically
likely one.** `_pick` keeps only the `width` (default **3**) highest-weight
candidates, and the weight is the human corpus COUNT. If human idiom frequency is
dominated by verticals, a top-3-by-frequency truncation cannot help but strip
diagonals -- which is exactly "the vocabulary has them and we never accumulate them".
It became the default on 2026-08-21, validated on recurrence and top-5 cell share;
its effect on cut direction was never measured.

★A stage where the share DROPS is the leak. A stage where it is already low means the
loss happened upstream. If A is already at ~0.22 the sampler is innocent and the loss
is in the vocabulary match itself.

Usage:
    python scripts/diag_diagonal_leak.py --songs 1f767 1f333
"""
from __future__ import annotations

import argparse
import json
import pathlib
import statistics as st
import subprocess
import sys
import tempfile
import zipfile

REPO = pathlib.Path(__file__).resolve().parents[1]
AM = REPO / "agent_mapper"
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO))

DIAG = {4, 5, 6, 7}
VERT = {0, 1}


def share(entries, want) -> float | None:
    """Share of candidate entries whose PLACED direction `d_to` is in `want`."""
    if not entries:
        return None
    return sum(1 for e in entries if e[3] in want) / len(entries)


def notes_of(zp: pathlib.Path) -> tuple[list, float] | None:
    """The generated map's colorNotes as idiomize `records`, plus bpm."""
    with zipfile.ZipFile(zp) as zf:
        names = zf.namelist()
        info = next((n for n in names
                     if n.split("/")[-1].lower() == "info.dat"), None)
        diff = next((n for n in names
                     if n.split("/")[-1].lower().endswith("standard.dat")
                     and "bpminfo" not in n.lower()), None)
        if info is None or diff is None:
            return None
        meta = json.loads(zf.read(info))
        dat = json.loads(zf.read(diff))
    bpm = None
    for k in ("_beatsPerMinute", "beatsPerMinute"):
        if k in meta.get("_songTimeOffset", {}) if False else k in meta:
            bpm = float(meta[k])
            break
    if bpm is None:
        for k, v in meta.items():
            if "beatsperminute" in k.lower():
                bpm = float(v)
                break
    notes = dat.get("colorNotes") or dat.get("_notes") or []
    recs = []
    for n in notes:
        recs.append({"b": n.get("b", n.get("_time", 0.0)),
                     "x": n.get("x", n.get("_lineIndex", 0)),
                     "y": n.get("y", n.get("_lineLayer", 0)),
                     "c": n.get("c", n.get("_type", 0)),
                     "d": n.get("d", n.get("_cutDirection", 1))})
    return (recs, float(bpm or 120.0)) if recs else None


def trace(recs, bpm, seed=0):
    """Re-run the sampler with every stage instrumented. Mirrors `idiomize`."""
    import random

    from agent_mapper import idiomize as IZ
    from beatsaber_automapper.evaluation import idiom as idm

    counts, ranked, _ = idm.load_vocab()
    rng = random.Random(seed)
    spb = 60.0 / bpm if bpm > 0 else 0.5
    hands = {0: IZ._Hand(0), 1: IZ._Hand(1)}
    recent: dict[int, list] = {0: [], 1: []}
    order = sorted(range(len(recs)),
                   key=lambda i: (float(recs[i].get("b", 0.0)),
                                  int(recs[i].get("c", 0))))
    stages: dict[str, list] = {k: [] for k in "ABCD"}
    picked: list[int] = []

    for i in order:
        r = recs[i]
        beat, color = float(r.get("b", 0.0)), int(r.get("c", 0))
        if color not in (0, 1):
            continue
        h = hands[color]
        dt = beat - h.beat
        if dt <= 0:
            dt = 1e-3
        cross_ok = rng.random() < IZ.CROSSOVER_TARGET
        cands = IZ._candidates(ranked, counts, h, min(dt, 2.0), spb,
                               IZ.VOCAB_DEPTH, cross_ok)
        s = share([c[0] for c in cands], DIAG)
        if s is not None:
            stages["A"].append(s)

        if cands and recent[color] and rng.random() < IZ.REPEAT_P:
            legal = {c[0] for c in cands}
            again = [e for e in recent[color] if e in legal]
            if again:
                cands = [c for c in cands if c[0] in set(again)]
        s = share([c[0] for c in cands], DIAG)
        if s is not None:
            stages["B"].append(s)

        pool = cands
        if cross_ok:
            crossing = [c for c in cands if c[2]]
            if crossing:
                pool = crossing
        s = share([c[0] for c in pool], DIAG)
        if s is not None:
            stages["C"].append(s)

        width = 3
        if width and len(pool) > width:
            pool = sorted(pool, key=lambda c: -c[1])[:width]
        s = share([c[0] for c in pool], DIAG)
        if s is not None:
            stages["D"].append(s)

        pick = IZ._pick(cands, rng, prefer_cross=cross_ok, width=width)
        if pick is None and cross_ok:
            c2 = IZ._candidates(ranked, counts, h, min(dt, 2.0), spb,
                                IZ.VOCAB_DEPTH, False)
            pick = IZ._pick(c2, rng, prefer_cross=False, width=width)
        if pick is not None:
            dx, dy, _df, d_to, _c = pick
            h.x, h.y = h.x + dx, h.y + dy
            h.direction = d_to
            p = IZ._parity_of(d_to)
            if p is not None:
                h.parity = p
            picked.append(d_to)
            recent[color].append(pick)
            del recent[color][:-IZ.REPEAT_WINDOW]
        h.beat = beat

    return stages, picked


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--songs", nargs="*",
                    default=["1f767", "1f333", "1f8d6", "1f913"])
    ap.add_argument("--json", default="")
    a = ap.parse_args()

    tmp = pathlib.Path(tempfile.mkdtemp(prefix="diagleak_"))
    rows = []
    print("diagonal share of the candidate pool, by sampler stage")
    print(f"{'song':8s}{'A cand':>9s}{'B repeat':>10s}{'C cross':>9s}"
          f"{'D width':>9s}{'E picked':>10s}{'vert out':>10s}")
    print("-" * 66)
    for sid in a.songs:
        audio = REPO / "data" / "eval_songset" / f"{sid}.ogg"
        if not audio.exists():
            print(f"{sid:8s}  no audio")
            continue
        out = tmp / f"LEAK__{sid}.zip"
        subprocess.run(
            [sys.executable, str(AM / "autobuild.py"), str(audio), "--pulse",
             "--lead-bias", "0.2", "--name", f"lk_{sid}", "--out", str(out)],
            capture_output=True, text=True, cwd=REPO)
        got = notes_of(out) if out.exists() else None
        if not got:
            print(f"{sid:8s}  build failed")
            continue
        recs, bpm = got
        stages, picked = trace(recs, bpm)
        if not picked:
            print(f"{sid:8s}  no picks")
            continue
        d_out = sum(1 for d in picked if d in DIAG) / len(picked)
        v_out = sum(1 for d in picked if d in VERT) / len(picked)
        cells = [st.mean(stages[k]) if stages[k] else float("nan")
                 for k in "ABCD"]
        print(f"{sid:8s}{cells[0]:9.3f}{cells[1]:10.3f}{cells[2]:9.3f}"
              f"{cells[3]:9.3f}{d_out:10.3f}{v_out:10.3f}")
        rows.append(dict(song=sid, A=cells[0], B=cells[1], C=cells[2], D=cells[3],
                         picked_diag=d_out, picked_vert=v_out, n=len(picked)))

    if not rows:
        return 1
    print("-" * 66)
    m = {k: st.mean([r[k] for r in rows]) for k in ("A", "B", "C", "D")}
    md = st.mean([r["picked_diag"] for r in rows])
    mv = st.mean([r["picked_vert"] for r in rows])
    print(f"{'MEAN':8s}{m['A']:9.3f}{m['B']:10.3f}{m['C']:9.3f}{m['D']:9.3f}"
          f"{md:10.3f}{mv:10.3f}")
    print(f"\nhuman: diagonal 0.415 · vertical 0.480      "
          f"(vocabulary stationary diagonal 0.344)")

    print("\nWHERE THE LEAK IS")
    drops = [("A->B  REPEAT_P", m["B"] - m["A"]),
             ("B->C  crossover", m["C"] - m["B"]),
             ("C->D  width trunc", m["D"] - m["C"]),
             ("D->E  the draw", md - m["D"])]
    for name, d in sorted(drops, key=lambda x: x[1]):
        print(f"  {name:22s} {d:+.3f}")
    worst = min(drops, key=lambda x: x[1])
    print(f"\n  biggest single loss: {worst[0]} ({worst[1]:+.3f})")
    print("  ⚠️If stage A is ALREADY near the output share, no sampler stage is the")
    print("     leak -- the loss is in the d_from/reachability match itself, and the")
    print("     'reachability gradient' note (from a vertical only 18-31% of reachable")
    print("     idioms are diagonal) is the whole explanation.")

    if a.json:
        p = pathlib.Path(a.json)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(rows, indent=2))
        print(f"\nwrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
