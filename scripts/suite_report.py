#!/usr/bin/env python
"""★ ONE COMMAND, ONE SONG, EVERYTHING — the eval suite's front door (P0.5).

Kyle, 2026-08-04: *"Create a way for you to see the song and map in a way that
gives you my vision… make it as robust as possible."*

The suite had grown to six separate scripts, each answering one question. That is
fine for building and bad for reviewing: a review means holding the picture, the
numbers and the timestamps together. This runs them all for one song and prints a
single consolidated report, plus the PNG that shows the worst stretch.

    METRICS      main-beat coverage / continuity / notes-on-main, ours vs human,
                 with the metrical level the song was judged on and the fit's
                 confidence — because a number computed on a LOW-confidence grid
                 must not be read as if it were solid.
    FINDINGS     review_map.py's ranked timestamps (STARVED / MISSED_HIT /
                 OFFBEAT / PHRASE_HOLE / MAPPING_SILENCE / ENDING).
    PICTURE      view_main_beat.py at the worst stretch, so the top finding and
                 the image are about the same moment.

⚠️**Read the confidence line first.** On 1fa48 and 1f9a0 the grid is right but
Stage-1 is one slot out of phase, so coverage reads ~0.00 — a real defect, but one
that would be misread as "we play nothing" without the diagnosis in TODO S1.

Usage:
    python scripts/suite_report.py --song 1f8d6
    python scripts/suite_report.py --song 1f333 --map outputs/some_arm.zip
    python scripts/suite_report.py --all --arm tf_trim_ev03_rc05     # cohort sweep
"""

from __future__ import annotations

import argparse
import collections
import glob
import pathlib
import subprocess
import sys

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from beatsaber_automapper.evaluation import alignment, scorecard  # noqa: E402
from calibrate_playfeel import load_expert_only  # noqa: E402
from main_beat import coverage, find_main_beat  # noqa: E402
from review_map import mmss, review  # noqa: E402


def load(song: str, map_path: str | None, arm: str):
    if map_path is None:
        hits = sorted(glob.glob(str(REPO / f"outputs/eval_sweep_cache/{arm}#s0__{song}.zip")))
        if not hits:
            return None
        map_path = hits[0]
    L = scorecard._load_any(pathlib.Path(map_path))
    if not L:
        return None
    bpm = float(L[1])
    ours = np.sort(np.asarray(alignment.note_times(L[0], bpm), dtype=float))
    hz = REPO / "data" / "raw" / f"{song}.zip"
    human = None
    if hz.exists():
        H = load_expert_only(hz)
        if H:
            human = np.sort(np.asarray(alignment.note_times(H[0], float(H[1])), dtype=float))
    return map_path, bpm, ours, human


def report(song: str, map_path: str | None, arm: str, top: int, png: bool) -> dict | None:
    got = load(song, map_path, arm)
    if got is None:
        print(f"{song}: no map found")
        return None
    map_path, bpm, ours, human = got
    end = float(max(ours.max(), human.max() if human is not None else 0))
    mb = find_main_beat(song, bpm, end)
    if mb is None:
        print(f"{song}: no main beat (missing stem cache?)")
        return None

    cov = coverage(ours, mb)
    covh = coverage(human, mb) if human is not None else {}
    F = review(song, ours, bpm, human)
    counts = collections.Counter(f["kind"] for f in F)

    print(f"\n{'='*78}\n{song}   {pathlib.Path(map_path).name}   bpm {bpm:g}\n{'='*78}")
    print(f"  MAIN BEAT: {mb.ratio:g}× the fitted beat ({mb.period*1000:.0f}ms), "
          f"fit {mb.confidence}")
    print(f"  {'metric':18s}{'ours':>9s}{'human':>9s}")
    for k, lab in (("main_covered", "covered"), ("main_continuity", "continuity"),
                   ("notes_on_main", "notes on main")):
        o = cov.get(k, float("nan"))
        h = covh.get(k, float("nan"))
        flag = ""
        if k == "main_covered" and h == h and o == o and o < h - 0.08:
            flag = "  ← below human"
        if k == "notes_on_main" and h == h and o == o and o > h + 0.08:
            flag = "  ← ABOVE human (metronome risk)"
        print(f"  {lab:18s}{o:9.3f}{h:9.3f}{flag}")
    print(f"  notes: ours {len(ours)}   human {len(human) if human is not None else '—'}")
    print(f"  findings: {dict(counts)}")
    print(f"\n  top {top} moments:")
    for f in sorted(F, key=lambda f: -f["sev"])[:top]:
        print(f"    {mmss(f['t']):>9s}  {f['kind']:<16s} {f['msg'][:96]}")

    if png:
        out = REPO / "outputs" / "suite_2026-08-04" / f"report_{song}.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run([str(REPO / ".venv/bin/python"), str(REPO / "scripts/view_main_beat.py"),
                        "--song", song, "--map", map_path, "--worst", "--span", "24",
                        "--out", str(out)], capture_output=True, timeout=300)
        print(f"\n  picture: {out}")
    return {"song": song, "cov": cov, "covh": covh, "counts": counts, "mb": mb}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--song")
    ap.add_argument("--map")
    ap.add_argument("--arm", default="tf_trim_ev03_rc05")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--top", type=int, default=6)
    ap.add_argument("--no-png", action="store_true")
    a = ap.parse_args()

    if a.all:
        rows = []
        for p in sorted(glob.glob(str(REPO / f"outputs/eval_sweep_cache/{a.arm}#s0__*.zip"))):
            sid = pathlib.Path(p).name.split("__")[1][:-4]
            r = report(sid, p, a.arm, a.top, png=not a.no_png)
            if r:
                rows.append(r)
        if rows:
            print(f"\n{'='*78}\nCOHORT SUMMARY  ({len(rows)} songs, arm {a.arm})\n{'='*78}")
            for k in ("main_covered", "main_continuity", "notes_on_main"):
                o = [r["cov"][k] for r in rows if k in r["cov"]]
                h = [r["covh"][k] for r in rows if k in r["covh"]]
                print(f"  {k:18s} ours {np.median(o):.3f}"
                      + (f"   human {np.median(h):.3f}" if h else ""))
            agg = collections.Counter()
            for r in rows:
                agg.update(r["counts"])
            print(f"  findings per song: "
                  + ", ".join(f"{k} {v/len(rows):.1f}" for k, v in agg.most_common()))
    elif a.song:
        report(a.song, a.map, a.arm, a.top, png=not a.no_png)
    else:
        ap.error("pass --song or --all")


if __name__ == "__main__":
    main()
