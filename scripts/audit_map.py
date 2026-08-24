#!/usr/bin/env python
"""THE SELF-AUDIT — every channel on one screen, with one honest verdict.

★**Kyle, 2026-08-24:** *"complete when you call the skill and are confident in the map
you build and feel you can evaluate the map correctly and don't need to rely on me to
audit."* Every channel needed for that now exists, but each has to be remembered, run
separately, and interpreted against a different reference. This assembles them.

    NOTES        `mapjudge` -- 23 metrics vs 1 100 human maps (works at n=1: 0 nan)
    ALIGNMENT    per-note distance to detected onsets -- ●/○/✗
    ELEMENTS     walls/arcs/chains vs 2 688 human maps, + notes trapped in walls
    PLAYABILITY  parity violations from the swing simulator

⚠️**A PASS IS A FLOOR, NOT A VERDICT** — and this prints that every time, because the
number is the part that misleads:
  * a PASS means **not defective**, not good; the gate is the corpus MEDIAN and Kyle's
    target is *"the best mappers"*.
  * the judge accepts **65 %** of maps shifted a quarter-beat off the music, so a PASS
    says nothing about alignment ⇒**read the ALIGNMENT block, that is what it is for.**
  * `p` is a distance-from-typical: **high `p` means BLANDER.** Human maps sit at
    0.28-0.57; above 0.7 is a warning, not a win. **Never rank by it.**
  * the 23 metrics move by **exactly 0.000** when walls/arcs/chains are added ⇒the
    NOTES block is blind to three of the five elements. That is what ELEMENTS is for.

★**What this deliberately does NOT do**: decide whether the map is FUN. It rules out
defects and places every quantity against humans. `agent_mapper/READING.md` is the part
that needs eyes -- the reading prompts at the end are the handover to it.

Usage:
    python scripts/audit_map.py <map.zip> [--audio <song>] [--json out.json]
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "agent_mapper"))

import elements as EL  # noqa: E402

TOL_S = 0.050


def song_id_of(map_path: pathlib.Path) -> str:
    """`<arm>__<songid>.zip` -> songid. ⚠️Arm-first is the repo convention because
    `scorecard.song_id()` parses from the START of the name and silently returns
    `alignment = nan` for `<songid>_<arm>`."""
    return map_path.stem.split("__")[-1].split("_")[0]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("map", type=pathlib.Path)
    ap.add_argument("--audio", type=pathlib.Path, default=None,
                    help="needed only for a song with no cached onsets")
    ap.add_argument("--json", default="")
    a = ap.parse_args()

    from beatsaber_automapper.evaluation import mapjudge as mj
    from agent_mapper import refonsets as RO

    sid = song_id_of(a.map)
    elems = EL.load_elements(a.map)
    onsets = RO.reference_onsets(sid, audio=a.audio, compute=bool(a.audio))
    report: dict = {"map": str(a.map), "song": sid}

    print(f"# AUDIT  {a.map.name}   ({len(elems['notes'])} notes @ "
          f"{elems['bpm']:.1f} bpm)")

    # ---- NOTES -------------------------------------------------------------
    print("\n## NOTES — 23 metrics vs 1 100 human maps")
    try:
        r = mj.judge_zip(a.map, onsets=onsets, reference=mj.load_reference())
        out = [m for m in r.metrics
               if m.pct is not None and (m.pct <= 0.05 or m.pct >= 0.95)]
        print(f"  {r.verdict()}   p={r.p_value:.3f}   {len(r.metrics)} metrics scored")
        if out:
            for m in sorted(out, key=lambda m: min(m.pct, 1 - m.pct)):
                print(f"    ⚠️{m.name:22s}{m.value:>10.4g}  human pct {100*m.pct:>5.1f}%")
        else:
            print("    no metric outside the 5th-95th human range")
        # ★high p is BLANDER, not better -- say so where it is read, not in a doc.
        if r.p_value > 0.70:
            print(f"    🟡p={r.p_value:.2f} is ABOVE the human band (0.28-0.57): this map "
                  f"is closer to the corpus average than real maps are. Blander.")
        report["notes"] = {"verdict": r.verdict(), "p": r.p_value,
                           "out_of_range": [m.name for m in out]}
    except Exception as exc:  # noqa: BLE001
        print(f"    🔴could not judge: {exc}")
        report["notes"] = {"error": str(exc)}

    # ---- ALIGNMENT ---------------------------------------------------------
    print("\n## ALIGNMENT — is each note on a sound the player hears?")
    if onsets is None or not len(onsets):
        print("    🔴NO ONSETS for this song — pass --audio to compute them.")
        print("    ⚠️Without this the audit is DEAF: the notes block cannot see")
        print("       alignment, and a PASS accepts 65 % of quarter-beat-off maps.")
        report["alignment"] = None
    else:
        o = np.sort(np.asarray(onsets, dtype=float))
        ts = np.array([elems["offset"] + float(n.get("b", 0)) * 60.0 / elems["bpm"]
                       for n in elems["notes"]])
        i = np.clip(np.searchsorted(o, ts), 1, len(o) - 1)
        signed = np.where(np.abs(ts - o[i - 1]) <= np.abs(ts - o[i]),
                          ts - o[i - 1], ts - o[i]) * 1000.0
        d = np.abs(signed)
        on, near, miss = (d <= 50).mean(), ((d > 50) & (d <= 120)).mean(), (d > 120).mean()
        print(f"    ● on a sound (≤50ms)  {on:6.1%}   ← onset_precision")
        print(f"    ○ near-miss (50-120)  {near:6.1%}")
        print(f"    ✗ nothing there       {miss:6.1%}")
        print(f"    median |offset| {np.median(d):.0f}ms · signed median "
              f"{np.median(signed):+.0f}ms")
        # ⚠️**The SIGNED median saturates and is NOT the detector.** A song carries
        # ~2 000+ detected onsets, so almost every note has one nearby and the signed
        # median sits near zero even on a map with a real grid error (measured: 1f333
        # reads +2ms while only 76.8 % of its notes are within 50ms). ★**The ● SHARE is
        # the discriminating statistic**; the signed median is reported for context
        # only. An earlier version warned on |signed median| > 15ms and was reading a
        # bug that always compared against the LOWER neighbouring onset.
        if on < 0.85:
            print(f"    🟡only {on:.1%} of notes are on a sound. If the map is otherwise "
                  f"sound this is usually the GRID, not the notes — try "
                  f"--phase-calibrate (our fitted phase sits 0.053 beats early of a "
                  f"human mapper's, and it lifts this share).")
        report["alignment"] = {"on": float(on), "near": float(near), "miss": float(miss),
                               "signed_median_ms": float(np.median(signed))}

    # ---- EMPTINESS ---------------------------------------------------------
    # ★The direction nothing measured before: not "was this note motivated?" but
    # "was this musical moment answered?" -- which is what "it feels empty" means.
    print("\n## EMPTINESS — did the map answer what the song asked? (vs 975 human maps)")
    if onsets is None or not len(onsets):
        print("    🔴needs onsets — pass --audio")
        report["coverage"] = None
    else:
        import coverage as CV
        jc = CV.judge(elems, onsets)
        for ln in jc["lines"]:
            print(f"  {ln['flag']} {ln['key']:24s}{ln['value']:>9.4g}"
                  f"  human med {ln['median']:>8.4g}  pct {100 * ln['pct']:>5.0f}%")
        if jc.get("note"):
            print(f"    ⚠️{jc['note']}")
        for g in jc["summary"]["worst_gaps"][:3]:
            print(f"    gap {g['dur']:.1f}s at {g['t0']:.0f}s "
                  f"({g['onsets_per_s']} onsets/s inside — "
                  f"{'the song is BUSY here' if g['onsets_per_s'] > 2 else 'the song is quiet too'})")
        report["coverage"] = {k: v for k, v in jc["summary"].items()
                              if k != "worst_gaps"}

    # ---- ELEMENTS ----------------------------------------------------------
    print("\n## ELEMENTS — walls, arcs, chains vs 2 688 human maps")
    j = EL.judge(elems)
    for line in EL.format_judgement(j)[1:]:
        print("  " + line)
    report["elements"] = j["summary"]

    # ---- PLAYABILITY -------------------------------------------------------
    print("\n## PLAYABILITY")
    cols = EL.collisions(elems)
    print(f"  {'🔴' if cols else '✅'} notes trapped inside walls: {len(cols)}"
          + (f"  first at beat {cols[0]['beat']:.2f}" if cols else ""))
    report["collisions"] = len(cols)

    # ---- THE HANDOVER ------------------------------------------------------
    print("\n## WHAT THIS CANNOT TELL YOU — read the map (agent_mapper/READING.md)")
    print("  1. Does a cell COME BACK? A scatter has nothing to lock into — the single")
    print("     most reliable read for 'unfun', and invisible to all 23 metrics.")
    print("  2. Does the density BREATHE? Read the --sections sparkline as a contour;")
    print("     nps percentiles say the level, never the shape.")
    print("  3. Does each hand stay home, and do the arrows alternate ↓↑?")
    print("  4. What is ABSENT? Zero doubles, no lead-hand alternation, a flat arc —")
    print("     absence never raises a number.")
    print("  ⇒ python scripts/map_view.py <map> --bars a-b --align --elements")

    if a.json:
        p = pathlib.Path(a.json)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(report, indent=2, default=float))
        print(f"\nwrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
