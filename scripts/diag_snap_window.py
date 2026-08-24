#!/usr/bin/env python
"""Should the snap window scale with the grid slot instead of being a fixed 60 ms?

**Why ask.** `diag_snap_reselection` found `corr(bpm, survival) = -0.868`: the snap
rewrites most of a fast song's map and barely touches a slow one. The mechanism is
arithmetic -- a fixed **60 ms** window against a 1/4-beat slot of `15000/bpm` ms is
**75 %** of a slot at 188 bpm and **38 %** at 93 bpm, so on a fast song the event lands
in a DIFFERENT slot and a different map gets built.

**The trade.** A narrower window reselects less, but the 60 ms was chosen because
*"88 % of our missed notes sat 50-120 ms out"* -- the near-miss band is where the mass
is. Narrowing it may simply stop finding the onsets that made the snap worth doing.

★**This asks the CHEAP half first.** The alignment gain is an EVENT-level property
(measured against the human mapper's own note times, before any note selection), and
that needs no map builds at all -- just arithmetic over cached events. If the gain
collapses at narrower windows, the bpm-relative idea is dead for a few seconds of
compute instead of a cohort of rebuilds.

⚠️**This does NOT answer the map-level half** (does survival actually rise?). That
needs builds, and is only worth paying for if the gain survives here.

Usage:
    python scripts/diag_snap_window.py --json outputs/snap_window.json
"""
from __future__ import annotations

import argparse
import json
import pathlib
import statistics as st
import sys

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO))

from agent_mapper import refonsets  # noqa: E402
from diag_snap_independent import (event_times, human_note_times,  # noqa: E402
                                   nearest_dist, snap_to)

TOL_S = 0.050

# Fixed windows, in ms, spanning well below and above today's 60.
FIXED_MS = (20, 30, 40, 60, 80, 120)

# Bpm-relative windows, as a FRACTION of the 1/4-beat slot (15000/bpm ms).
# 0.375 reproduces today's 60 ms at 160 bpm, the cohort's rough middle, so it is the
# closest bpm-relative analogue of the current setting.
SLOT_FRACS = (0.25, 0.375, 0.50)


def slot_ms(bpm: float) -> float:
    """A 1/4-beat slot in ms. Below 150 bpm this exceeds the axis' 50 ms tolerance."""
    return 15000.0 / max(bpm, 1e-6)


def bpm_of(sid: str) -> float | None:
    f = REPO / "outputs" / "event_cache" / f"{sid}.6s.json"
    if not f.exists():
        return None
    return float(json.loads(f.read_text())["bpm"])


def score(ev, ref, hum, window: float) -> dict | None:
    """Event-level lift for one song at one window.

    ⚠️**`lift` is measured over ALL events, not just the moved ones.** A wider window
    moves a larger and more distant population, so a lift computed over the moved
    subset has a different denominator in every arm and cannot be compared across
    windows -- it would flatter wide windows by construction. Over the whole event set
    the question is the comparable one: *how much closer to the human's note times does
    this window get the map's raw material?*
    `lift_moved` is kept alongside it as the per-move efficiency.
    """
    snapped, moved = snap_to(ev, ref, window)
    if moved.sum() < 10:
        return None
    db_all = nearest_dist(ev, hum)
    da_all = nearest_dist(snapped, hum)
    db = nearest_dist(ev[moved], hum)
    da = nearest_dist(snapped[moved], hum)
    wins = int((da < db - 1e-9).sum())
    losses = int((da > db + 1e-9).sum())
    return dict(
        n_moved=int(moved.sum()),
        moved_frac=float(moved.mean()),
        lift=float((da_all <= TOL_S).mean() - (db_all <= TOL_S).mean()),
        lift_moved=float((da <= TOL_S).mean() - (db <= TOL_S).mean()),
        win_pct=100.0 * wins / max(wins + losses, 1),
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="")
    ap.add_argument("--songs", nargs="*", default=None)
    a = ap.parse_args()

    sids = a.songs or sorted(p.stem for p in
                             (REPO / "data" / "eval_songset").glob("*.ogg"))
    songs = []
    for sid in sids:
        ev, h, ref, bpm = (event_times(sid), human_note_times(sid),
                           refonsets.reference_onsets(sid), bpm_of(sid))
        if ev is None or h is None or ref is None or bpm is None:
            continue
        songs.append((sid, ev, ref, h[0], bpm))
    if not songs:
        print("no songs")
        return 1
    print(f"cohort n={len(songs)}   (today's setting is a fixed "
          f"{int(refonsets.SNAP_WINDOW_S * 1000)} ms)\n")

    arms: dict[str, dict] = {}

    print(f"{'window':>16s}{'lift(all)':>12s}{'win%':>8s}{'moved':>9s}"
          f"{'songs+':>9s}{'≤0.5 slot':>11s}")
    print("-" * 66)
    for ms in FIXED_MS:
        rows = [(sid, score(ev, ref, hum, ms / 1000.0), bpm)
                for sid, ev, ref, hum, bpm in songs]
        rows = [(s, r, b) for s, r, b in rows if r]
        if not rows:
            continue
        lifts = [r["lift"] for _, r, _ in rows]
        # How many songs keep the window inside HALF a slot -- the regime where a snap
        # cannot push an event into the neighbouring slot.
        safe = sum(1 for _, _, b in rows if ms <= 0.5 * slot_ms(b))
        arms[f"fixed_{ms}ms"] = dict(
            kind="fixed", ms=ms, lift=st.mean(lifts),
            win=st.mean([r["win_pct"] for _, r, _ in rows]),
            moved=st.mean([r["moved_frac"] for _, r, _ in rows]),
            songs_positive=sum(1 for x in lifts if x > 0), n=len(rows), safe=safe)
        m = arms[f"fixed_{ms}ms"]
        star = "  <-- today" if ms == int(refonsets.SNAP_WINDOW_S * 1000) else ""
        print(f"{f'fixed {ms} ms':>16s}{m['lift']:+12.4f}{m['win']:8.1f}"
              f"{m['moved']:9.3f}{m['songs_positive']:>6d}/{m['n']:<3d}"
              f"{safe:>7d}/{m['n']:<3d}{star}")

    print()
    for frac in SLOT_FRACS:
        rows = []
        for sid, ev, ref, hum, bpm in songs:
            r = score(ev, ref, hum, frac * slot_ms(bpm) / 1000.0)
            if r:
                rows.append((sid, r, bpm))
        if not rows:
            continue
        lifts = [r["lift"] for _, r, _ in rows]
        arms[f"slot_{frac}"] = dict(
            kind="slot", frac=frac, lift=st.mean(lifts),
            win=st.mean([r["win_pct"] for _, r, _ in rows]),
            moved=st.mean([r["moved_frac"] for _, r, _ in rows]),
            songs_positive=sum(1 for x in lifts if x > 0), n=len(rows),
            safe=len(rows) if frac <= 0.5 else 0)
        m = arms[f"slot_{frac}"]
        lo = frac * slot_ms(max(b for _, _, b in rows))
        hi = frac * slot_ms(min(b for _, _, b in rows))
        print(f"{f'{frac:g}x slot':>16s}{m['lift']:+12.4f}{m['win']:8.1f}"
              f"{m['moved']:9.3f}{m['songs_positive']:>6d}/{m['n']:<3d}"
              f"{m['safe']:>7d}/{m['n']:<3d}   ({lo:.0f}-{hi:.0f} ms)")

    today = arms.get(f"fixed_{int(refonsets.SNAP_WINDOW_S * 1000)}ms")
    print("\nVERDICT LOGIC")
    if today:
        print(f"  today's fixed 60 ms buys lift {today['lift']:+.4f}.")
        best_slot = max((v for k, v in arms.items() if v["kind"] == "slot"),
                        key=lambda v: v["lift"], default=None)
        if best_slot:
            keep = best_slot["lift"] / today["lift"] if today["lift"] else float("nan")
            print(f"  best bpm-relative arm ({best_slot['frac']:g}x slot) buys "
                  f"{best_slot['lift']:+.4f}  = {keep:.0%} of it.")
            print("  KEEPS THE GAIN if that share is ~>=90% -- then the map-level half is")
            print("     worth building, because a <=0.5-slot window cannot push an event")
            print("     into the neighbouring slot and survival should rise on fast songs.")
            print("  DEAD if the share is materially lower: narrowing the window stops")
            print("     finding the near-miss onsets that made the snap worth doing, and")
            print("     the reselection on fast songs is the PRICE of the gain, not a bug.")

    if a.json:
        p = pathlib.Path(a.json)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(arms, indent=2))
        print(f"\nwrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
