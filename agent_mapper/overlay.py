#!/usr/bin/env python
"""THE OVERLAY — our map judged against the music under it: HIT / MISSED / WASTED.

★**This is the view Kyle's 2026-08-17 review asked for.** His central complaint was
not that the maps score badly, it was:

> *"I think the nps is generally wasted on every few non main notes… there is a good
> deal of notes that are on beat and I can tell play part of the song, but they aren't
> hitting that main flow that mappers can generally see."*

That is a claim about **allocation**: the note budget is spent on events that are not
the song. It has never been measurable, because nothing in this project has ever
defined *which events are the song*. This module defines it — **out loud, in one
place, so he can disagree with it** — and classifies every note against it:

| verdict | meaning | his words |
|---|---|---|
| **HIT** | our note sits on a main musical event | *"notes that are on beat and play part of the song"* |
| **MISSED** | a main event with no note on it | *"not following the main vocals"* |
| **WASTED** | our note with no main event under it | ★*"nps wasted on every few non main notes"* |

⚠️⚠️**THE DEFINITION OF "MAIN" IS THE WHOLE ARGUMENT, AND IT IS A GUESS UNTIL HE
CORRECTS IT.** This project's standing failure mode is inventing an axis from first
principles and then discovering it does not track his ear (measured: 13 of 26 axes
agreed with his one recorded verdict = a coin flip). So this is deliberately **not**
presented as a metric. It is presented as three colours on a picture, with the rule
printed beside them, so that the thing he corrects is the *definition* — which is a far
better object to argue about than a score.

## The rule, v1

A **main event** is any of:

1. a **pitched vocal onset** — the sung line, note by note;
2. a **kick or snare** strike — the backbeat, the pulse a player moves to;
3. during a **vocal rest** (no vocal onset for `REST_BEATS` beats), the **lead**
   (`other` stem) pitched onsets — because in an instrumental passage the lead *is*
   the main line, and a rule that only knows about singing would call an entire guitar
   solo "wasted".

Hats, toms, ghost notes, bass runs under a vocal, and every unpitched onset are
**not** main. They are real events — that is why a WASTED note reports *what* it landed
on — but they are not what a listener would hum back.

⚠️**A WASTED note is not necessarily a bad note.** Landing on a hi-hat is a normal
mapping choice; landing on nothing at all is not. The two are counted separately
(`wasted_on_nothing`) rather than merged, because collapsing them would smuggle in a
judgement this module has no standing to make.

Usage:
    python agent_mapper/overlay.py <audio.ogg> --map <map.zip>
    python agent_mapper/overlay.py <audio.ogg> --map <map.zip> --main vocals,kick,snare
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

REPO = pathlib.Path(__file__).resolve().parents[1]

# Half a sixteenth at 138 bpm is 54 ms; a human mapper's own onset scatter is ~10 ms and
# ours ~10-23 ms. 70 ms is wide enough that a correctly placed note always counts and
# narrow enough that a note on the next sixteenth never does.
TOL = 0.070
REST_BEATS = 2.0          # a vocal gap this long hands the main line to the lead
MAIN_DEFAULT = "vocals,kick,snare,lead_in_rests"


def load_map(zip_path: pathlib.Path) -> tuple[np.ndarray, float]:
    """Note times in SECONDS, and the map's own bpm.

    ⚠️Times come from the **map's** declared bpm, not from our detected tempo: that is
    the clock the game plays the note on. Where the two disagree the notes will visibly
    drift against the audio lanes — which is a defect worth seeing, not one to hide by
    quietly re-timing them.
    """
    sys.path.insert(0, str(REPO / "scripts"))
    from eval_contour_follow import _load_notes_with_direction
    from feel_disc_poc import _zip_bpm

    recs = _load_notes_with_direction(zip_path, "Expert")
    bpm = float(_zip_bpm(str(zip_path)) or 120.0)
    if not recs:
        return np.zeros(0), bpm
    beats = np.array([r[0] for r in recs], dtype=float)
    return np.sort(beats) * 60.0 / bpm, bpm


def main_events(d: dict, parts: str = MAIN_DEFAULT) -> list[dict]:
    """The events the rule above calls "the song", each tagged with its source."""
    want = {p.strip() for p in parts.split(",") if p.strip()}
    ev: list[dict] = []

    vox = d["melody"]["stems"].get("vocals", [])
    if "vocals" in want:
        ev += [{"t": e["t"], "src": "vox", "lane": "vocals"} for e in vox]

    for piece in ("kick", "snare"):
        if piece in want:
            ev += [{"t": h["t"], "src": piece, "lane": "kit"}
                   for h in d["perc"]["hits"] if h["piece"] == piece]

    if "lead_in_rests" in want:
        gap = REST_BEATS * d["grid"]["spb"]
        vt = np.array([e["t"] for e in vox], dtype=float)
        for e in d["melody"]["stems"].get("other", []):
            # nearest vocal onset either side; a lead note inside a vocal rest is main
            if len(vt) == 0 or float(np.min(np.abs(vt - e["t"]))) > gap:
                ev.append({"t": e["t"], "src": "lead", "lane": "other"})

    ev.sort(key=lambda x: x["t"])
    return ev


def _minor_events(d: dict) -> list[dict]:
    """Everything real but not main — what a WASTED note may still have landed on."""
    ev = [{"t": h["t"], "src": h["piece"]} for h in d["perc"]["hits"]
          if h["piece"] in ("hat", "crash")]
    ev += [{"t": e["t"], "src": "bass"} for e in d["melody"]["stems"].get("bass", [])]
    ev += [{"t": e["t"], "src": "lead"} for e in d["melody"]["stems"].get("other", [])]
    ev.sort(key=lambda x: x["t"])
    return ev


def _nearest(ts: np.ndarray, t: float) -> int | None:
    """Index of the closest time in a sorted array, or None if it is empty."""
    if len(ts) == 0:
        return None
    i = int(np.searchsorted(ts, t))
    cands = [j for j in (i - 1, i) if 0 <= j < len(ts)]
    return min(cands, key=lambda j: abs(ts[j] - t)) if cands else None


def classify(notes: np.ndarray, d: dict, parts: str = MAIN_DEFAULT,
             tol: float = TOL) -> dict:
    """Every note a verdict, every main event covered or not.

    ⚠️Notes are classified against **distinct times**, not one-to-one: a double is two
    notes on one event and both are HIT, and the event is covered once. Pairing them
    greedily instead would call the second half of every double WASTED and manufacture
    a defect out of a normal mapping idiom.
    """
    main = main_events(d, parts)
    mt = np.array([e["t"] for e in main], dtype=float)
    minor = _minor_events(d)
    nt = np.array([e["t"] for e in minor], dtype=float)

    verdicts: list[dict] = []
    covered = np.zeros(len(main), dtype=bool)
    for t in notes:
        j = _nearest(mt, t)
        if j is not None and abs(mt[j] - t) <= tol:
            covered[j] = True
            verdicts.append({"t": float(t), "v": "hit", "on": main[j]["src"]})
            continue
        k = _nearest(nt, t)
        on = minor[k]["src"] if k is not None and abs(nt[k] - t) <= tol else "nothing"
        verdicts.append({"t": float(t), "v": "wasted", "on": on})

    missed = [main[i] for i in range(len(main)) if not covered[i]]
    n = max(len(notes), 1)
    nhit = sum(1 for v in verdicts if v["v"] == "hit")
    nothing = sum(1 for v in verdicts if v["v"] == "wasted" and v["on"] == "nothing")
    return {
        "verdicts": verdicts, "missed": missed, "main": main,
        "n_notes": int(len(notes)), "n_main": len(main),
        "hit": nhit, "wasted": len(verdicts) - nhit, "n_missed": len(missed),
        "precision": round(nhit / n, 3),                       # share of notes that are main
        "recall": round(int(covered.sum()) / max(len(main), 1), 3),   # share of main played
        "wasted_on_nothing": nothing,
        "nothing_share": round(nothing / n, 3),
        "rule": parts, "tol": tol,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("audio", type=pathlib.Path)
    ap.add_argument("--map", type=pathlib.Path, required=True)
    ap.add_argument("--main", default=MAIN_DEFAULT)
    ap.add_argument("--tol", type=float, default=TOL)
    a_ = ap.parse_args()

    import notesheet as _ns

    d = _ns.collect(a_.audio)
    notes, bpm = load_map(a_.map)
    r = classify(notes, d, a_.main, a_.tol)

    dur = d["dur"]
    print(f"{a_.audio.stem}  map {a_.map.name}")
    print(f"  map bpm {bpm:.2f} vs detected {d['grid']['bpm']:.2f}"
          + ("   ⚠️DISAGREE — notes will drift against the lanes"
             if abs(bpm - d["grid"]["bpm"]) > 0.6 else ""))
    print(f"  rule: main = {r['rule']}   tolerance ±{r['tol']*1000:.0f} ms")
    print(f"  notes {r['n_notes']} ({r['n_notes']/dur:.2f} nps)   "
          f"main events {r['n_main']} ({r['n_main']/dur:.2f}/s)")
    print(f"  HIT    {r['hit']:>5}   {r['precision']:.1%} of our notes are on a main event")
    print(f"  WASTED {r['wasted']:>5}   of which {r['wasted_on_nothing']} on NOTHING "
          f"({r['nothing_share']:.1%} of all notes)")
    print(f"  MISSED {r['n_missed']:>5}   we play {r['recall']:.1%} of the main line")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
