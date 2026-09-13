#!/usr/bin/env python
"""Assert every reported margin agrees with the code that reports it.

★**A margin that disagrees with its own query is worse than no margin** (2026-09-13q). Each
query takes an optional write-only `report` dict and records how close it came to firing, as
a `room`: **1 + the signed slack as a fraction of the line**, so 1.00 sits exactly on the line
in both directions (a FLOOR — red BELOW L — is `value/L`; a CEILING — red AT/ABOVE L — is
`2 - value/L`). That gives one invariant worth asserting:

    room < 1.00  **iff**  the query fired that code

It caught two real bugs the day it was written, both in margins that had shipped and read
plausibly: `q_events` reported EMPTY and D1 across difficulties where the query does not ask
them, and read D6's margin off the `h >= 12` window population while the over-dense branch
fires on `h >= 8` — calling `1f8d6` safe at 1.88x on a window that had fired.

Run it after touching any query or any threshold:

    python scripts/check_margins.py            # the 19 bench rows
    python scripts/check_margins.py <map.zip> --song <sid>

Exit 1 on any disagreement.
"""
from __future__ import annotations

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import bench  # noqa: E402
import queries as Q  # noqa: E402

QUERIES = (Q.q_events, Q.q_flow, Q.q_vocals, Q.q_drops,
           Q.q_elements, Q.q_breathing, Q.q_scatter)


def check(arrs: dict, label: str, verbose: bool = False) -> list[str]:
    """Every margin this map's queries report, against whether the code fired."""
    bad = []
    for q in QUERIES:
        rep: dict = {}
        fired = {h[0] for h in q(arrs, report=rep)}
        for code, (txt, room) in sorted(rep.items()):
            hit = code in fired
            if (room < 1.0) != hit:
                bad.append(f"{label:<20s} {code:<9s} room={room:6.2f} fired={hit}  {txt}")
            elif verbose:
                print(f"  {label:<20s} {code:<9s} {room:6.2f} {'FIRED' if hit else '     '}  {txt}")
    return bad


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("map", nargs="?", help="a map zip; default is every bench row")
    ap.add_argument("--song", help="song id, when reading a zip")
    ap.add_argument("-v", "--verbose", action="store_true", help="print every margin")
    a = ap.parse_args()

    bad, n = [], 0
    if a.map:
        from verdict import load_arrays
        arrs, _, _ = load_arrays(pathlib.Path(a.map), a.song, "auto")
        n = 1
        bad += check(arrs, pathlib.Path(a.map).stem, a.verbose)
    else:
        for row in bench.load_rows():
            arrs = bench.arrays_for(row)
            if arrs is None:
                continue
            n += 1
            bad += check(arrs, row["id"], a.verbose)

    if bad:
        print(f"\n🔴 {len(bad)} margin(s) disagree with their own query, over {n} map(s):")
        for line in bad:
            print("  " + line)
        print("\nA reported `room` must be below 1.00 exactly when the code fires. Copy the "
              "query's GATE (a branch it skips is not a margin) and its POPULATION (the same "
              "windows it actually reads).")
        return 1
    print(f"✅ every margin agrees with its query, over {n} map(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
