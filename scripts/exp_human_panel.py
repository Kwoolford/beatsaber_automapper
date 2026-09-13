#!/usr/bin/env python
"""How often does each per-song threshold fire when a HUMAN is read against another HUMAN?

★★**The control this project never had.** Every human row in `bench.py` is scored against
**itself** (`--vs auto` finds its own map), and the `humanplus-*` / `humanexp-*` rows are one
mapper's two difficulties of one song. Neither can answer *"would this code call a top mapper
defective?"* — for that you need two DIFFERENT mappers on the same music.

Scanning all 5 373 corpus zips by song title and artist finds **172 songs mapped by 2+ different
mappers** (`outputs/dup_songs_2026-09-13.json`). Read as ordered pairs — map A judged against map B
exactly as our builds are judged against a human — that is a real negative control.

The first threshold put through it, ELEMENTS' 0.50x wall coverage, **fired on 34 % of pairs** and
had to move to 0.10 (2026-09-13ad). This script does the same for the other map-only codes.

⚠️**These are ANALOGUES of the queries, not the queries themselves.** Two mappers notate the same
song at their own BPM, so there is no shared bar lattice to run `queries.py` on. Windows here are
fixed **seconds** measured from each map's first note, and SCATTER's echo is each map's own mean
over its own blocks rather than over shared block indices. That makes the numbers indicative of
the threshold's behaviour, not a re-run of the gate — which is enough to decide whether a line
fires on humans, and not enough to set one to three decimals.

Run:
    python scripts/exp_human_panel.py
    python scripts/exp_human_panel.py --out outputs/human_panel_<date>.json
"""
from __future__ import annotations

import argparse
import itertools
import json
import pathlib
import sys
import zipfile
from collections import Counter

import numpy as np

REPO = pathlib.Path(__file__).resolve().parent.parent
RAW = REPO / "data" / "raw"
DUPS = REPO / "outputs" / "dup_songs_2026-09-13.json"

WINDOW_SEC = 8.0      # stands in for q_events' 4-bar window
BLOCK_SEC = 8.0       # stands in for q_scatter's 4-bar block
MIN_NOTES_BLOCK = 6   # queries._echo
EMPTY_LOW = 0.6       # queries.q_events
DENSE_HIGH = 2.0
EMPTY_MIN_H = 12
DENSE_MIN_H = 8
SCATTER_MARGIN = 0.15


def load(zp: pathlib.Path, want: str | None = None):
    """(difficulty name, note times in seconds + figures), or None.

    ⚠️**`want` matters more than it looks.** EMPTY and D6 are LEVEL claims and `queries.py`
    refuses to ask them across difficulties — a top mapper's own Expert draws SEVEN EMPTY
    against his own ExpertPlus (2026-09-10). Pairing "whichever difficulty each zip happens to
    have" would reproduce exactly that and read as a threshold failure. The first run of this
    script did precisely that; the difficulty-matched numbers are the ones to use.
    """
    with zipfile.ZipFile(zp) as zf:
        names = zf.namelist()
        info = next((n for n in names if n.split("/")[-1].lower() == "info.dat"), None)
        std = [n for n in names if n.lower().split("/")[-1].endswith("standard.dat")]
        by = {}
        for n in std:
            b = n.lower().split("/")[-1]
            if b.startswith("expertplus"):
                by["ExpertPlus"] = n
            elif b.startswith("expert"):
                by["Expert"] = n
            elif b.startswith("hard"):
                by["Hard"] = n
        if want is not None:
            diff = by.get(want)
            name = want
        else:
            name = next((k for k in ("Expert", "ExpertPlus", "Hard") if k in by), None)
            diff = by.get(name) if name else None
        if diff is None or info is None:
            return None
        meta = json.loads(zf.read(info).decode("utf-8-sig"))
        d = json.loads(zf.read(diff).decode("utf-8-sig"))
    bpm = 0.0
    for k in ("_beatsPerMinute", "beatsPerMinute"):
        if k in meta:
            bpm = float(meta[k])
            break
    else:
        bpm = float((meta.get("audio") or {}).get("bpm") or 0.0)
    if bpm <= 0:
        return None
    notes = d.get("colorNotes") or d.get("_notes") or []
    if len(notes) < 150:
        return None
    out = []
    for n in notes:
        b = n.get("b", n.get("_time"))
        if b is None:
            continue
        out.append((float(b) * 60.0 / bpm,
                    (int(n.get("c", n.get("_type", 0))),
                     int(n.get("x", n.get("_lineIndex", 0))),
                     int(n.get("y", n.get("_lineLayer", 0))),
                     int(n.get("d", n.get("_cutDirection", 0))))))
    out.sort()
    if not out:
        return None
    t0 = out[0][0]
    return name, [(t - t0, f) for t, f in out]


def echo(seq) -> float:
    """Mean block echo, `queries._echo` on time blocks instead of bars."""
    blocks: dict[int, Counter] = {}
    for t, f in seq:
        blocks.setdefault(int(t // BLOCK_SEC), Counter())[f] += 1
    ks = sorted(b for b in blocks if sum(blocks[b].values()) >= MIN_NOTES_BLOCK)
    if len(ks) < 9:
        return float("nan")
    vals = []
    for i, b in enumerate(ks[1:], 1):
        A = blocks[b]
        vals.append(max(sum((A & blocks[c]).values())
                        / max(sum(A.values()), sum(blocks[c].values())) for c in ks[:i]))
    return float(np.mean(vals))


def events(seq) -> Counter:
    """Player EVENTS per window: notes on one instant count once, as `queries` reads them."""
    c: Counter = Counter()
    for w, ts in itertools.groupby(seq, key=lambda x: int(x[0] // WINDOW_SEC)):
        c[w] = len({round(t, 3) for t, _ in ts})
    return c


def fires(a, b) -> dict:
    """Would each code fire on map A when B is its reference human?"""
    ea, eb = events(a), events(b)
    empty = dense = False
    for w, h in eb.items():
        m = ea.get(w, 0)
        if h >= EMPTY_MIN_H and m < EMPTY_LOW * h:
            empty = True
        if h >= DENSE_MIN_H and m >= DENSE_HIGH * h:
            dense = True
    ka, kb = echo(a), echo(b)
    scatter = (kb - ka) >= SCATTER_MARGIN if ka == ka and kb == kb else False
    return dict(EMPTY=empty, D6=dense, SCATTER=scatter,
                echo_gap=(kb - ka) if ka == ka and kb == kb else float("nan"))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", help="write the per-pair records here")
    a = ap.parse_args()

    dups = json.loads(DUPS.read_text())
    rows, skipped = [], 0
    for song, entries in dups.items():
        # ★pair only maps of the SAME declared difficulty -- see `load`'s note
        for want in ("Expert", "ExpertPlus"):
            per = {}
            for stem, mapper in entries:
                zp = RAW / f"{stem}.zip"
                if not zp.exists() or mapper in per:
                    continue
                try:
                    s = load(zp, want)
                except Exception:  # noqa: BLE001
                    s = None
                if s is None:
                    skipped += 1
                    continue
                per[mapper] = (stem, s[1])
            for (ma, (sa, A)), (mb, (sb, B)) in itertools.permutations(per.items(), 2):
                r = fires(A, B)
                r.update(song=song.split("|")[0][:40], a=sa, b=sb, difficulty=want)
                rows.append(r)

    print(f"{len(rows)} ordered human-vs-human pairs ({skipped} maps unreadable)\n")
    print(f"{'code':<9s} {'fires on':>10s}   what that means")
    for code, note in (("EMPTY", "we are not playing the song"),
                       ("D6", "nps wasted / over-dense"),
                       ("SCATTER", "nothing comes back to lock into")):
        n = sum(1 for r in rows if r[code])
        print(f"{code:<9s} {n:4d}/{len(rows):<5d} {n/len(rows):5.1%}   {note}")
    gaps = np.array([r["echo_gap"] for r in rows if r["echo_gap"] == r["echo_gap"]])
    print(f"\nSCATTER echo gap (his − ours) over {len(gaps)} pairs, red at +{SCATTER_MARGIN}:")
    for q in (50, 75, 90, 95, 99):
        print(f"  p{q:<3d} {np.percentile(gaps, q):+.3f}")
    for thr in (0.15, 0.20, 0.25, 0.30):
        print(f"  a +{thr:.2f} line would fire on {(gaps >= thr).mean():5.1%} of human pairs")

    if a.out:
        pathlib.Path(a.out).write_text(json.dumps(rows, indent=1))
        print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
