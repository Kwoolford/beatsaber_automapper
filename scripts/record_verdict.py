#!/usr/bin/env python
"""Record one of Kyle's listening verdicts into the tracked ledger.

**Why this exists.** His ear is the only ground truth this project has, and the P0
finding is that the suite does not track it — the masterpiece axes rank the map he
called *"really empty"* second-best and the map he graded **A+** fifth-worst. The fix
proposed there is a structured A/B preference loop. That needs somewhere to put the
answers, and until now there was nowhere: one verdict lived hardcoded inside
`preference_screen.py` and the rest as prose scattered through `PROGRESS.md`.

Verdicts go to `docs/eval_references/preference_verdicts.json` — **tracked in git**,
because `outputs/` is gitignored (TODO C6) and losing this would mean asking him to
listen to everything again.

★**A same-song A/B is worth far more than a cross-song comparison.** With two arms on
one song, every axis can be read directly. Across songs, each axis has to be
normalised by that song's own human map to be commensurable at all — which is the
mistake this project has made more often than any other.

★**"Can't tell" is a real verdict.** Record it. It says the difference is inaudible,
which is evidence about whatever axis claimed the difference was large.

Usage:
    # same-song A/B (the good kind)
    python scripts/record_verdict.py --song 2c352 --name BEcause \\
        --better PHASE --worse BEFORE \\
        --quote "the phase one locks onto the beat, the other drifts"

    # he could not hear a difference
    python scripts/record_verdict.py --song 2c352 --name BEcause \\
        --tie PHASE BEFORE --quote "can't tell them apart"

    python scripts/record_verdict.py --list
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]
LEDGER = REPO / "docs" / "eval_references" / "preference_verdicts.json"


def load() -> dict:
    if LEDGER.exists():
        return json.loads(LEDGER.read_text())
    return {"_README": [], "verdicts": []}


def save(d: dict) -> None:
    LEDGER.write_text(json.dumps(d, indent=1, ensure_ascii=False) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--song", help="song id, e.g. 2c352")
    ap.add_argument("--name", default=None, help="human-readable song name")
    ap.add_argument("--better", help="arm he preferred, e.g. PHASE")
    ap.add_argument("--worse", help="arm he liked less, e.g. BEFORE")
    ap.add_argument("--tie", nargs=2, metavar=("ARM_A", "ARM_B"),
                    help="he could not tell them apart — a real verdict")
    ap.add_argument("--quote", default="", help="his words, verbatim if possible")
    ap.add_argument("--map-dir", default="outputs/kyle_review_2026-08-14",
                    help="directory holding the maps he actually played")
    ap.add_argument("--note", default="")
    ap.add_argument("--list", action="store_true", help="show the ledger and exit")
    a = ap.parse_args()

    d = load()
    if a.list:
        print(f"{len(d['verdicts'])} verdict(s) in {LEDGER.relative_to(REPO)}\n")
        for v in d["verdicts"]:
            if v.get("kind") == "tie":
                print(f"  {v['date']}  {v['id']}\n      TIE: {v['tie']}  "
                      f"\"{v.get('quote','')}\"")
            else:
                print(f"  {v['date']}  {v['id']}\n"
                      f"      better: {v['better'].get('name', v['better']['song'])} "
                      f"\"{v['better'].get('quote','')}\"\n"
                      f"      worse : {v['worse'].get('name', v['worse']['song'])} "
                      f"\"{v['worse'].get('quote','')}\"")
        return 0

    if not a.song:
        ap.error("--song is required (or use --list)")
    name = a.name or a.song

    if a.tie:
        arm_a, arm_b = a.tie
        v = {"id": f"{_dt.date.today()}/{a.song}-{arm_a}-vs-{arm_b}-tie",
             "date": str(_dt.date.today()), "kind": "tie", "song": a.song,
             "name": name, "tie": [arm_a, arm_b],
             "maps": [f"{a.map_dir}/{name}_{arm_a}.zip",
                      f"{a.map_dir}/{name}_{arm_b}.zip"],
             "quote": a.quote, "note": a.note}
    else:
        if not (a.better and a.worse):
            ap.error("give --better and --worse, or --tie ARM_A ARM_B")
        v = {"id": f"{_dt.date.today()}/{a.song}-{a.better}-over-{a.worse}",
             "date": str(_dt.date.today()), "kind": "same_song", "song": a.song,
             "name": name,
             "better": {"song": a.song, "name": name, "arm": a.better,
                        "map": f"{a.map_dir}/{name}_{a.better}.zip",
                        "quote": a.quote},
             "worse": {"song": a.song, "name": name, "arm": a.worse,
                       "map": f"{a.map_dir}/{name}_{a.worse}.zip", "quote": ""},
             "note": a.note}

    # ⚠️Refuse to silently record a verdict about a map that is not on disk. Naming a
    # map he did not play is the ExpertPlus-contamination error in a new costume.
    missing = [p for p in ([v["maps"]] if a.tie else
                           [[v["better"]["map"], v["worse"]["map"]]])[0]
               if not (REPO / p).exists()]
    if missing:
        print("⚠️ these map files do not exist — check --map-dir and --name:",
              file=sys.stderr)
        for m in missing:
            print(f"     {m}", file=sys.stderr)
        print("   recording anyway would attach his words to a map he did not play.",
              file=sys.stderr)
        return 2

    d.setdefault("verdicts", []).append(v)
    save(d)
    print(f"recorded {v['id']}  ({len(d['verdicts'])} total)")
    print(f"  -> {LEDGER.relative_to(REPO)}  (tracked; commit it)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
