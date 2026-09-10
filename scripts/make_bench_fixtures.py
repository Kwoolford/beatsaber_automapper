#!/usr/bin/env python
"""Rebuild the DERIVED bench fixtures — maps the bench needs that are not runs of ours.

    python scripts/make_bench_fixtures.py            # writes outputs/bench_fixtures/

★**The difficulty control (P5b, 2026-09-10).** Until today every CLEAN row on the bench was a
human map scored **against itself** (`--vs auto` resolves to the same zip), so any query written
as *"ours differs from his"* — which is every reference-relative query in `scripts/queries.py` —
was zero there **by construction**. Its "0 false fires on 4 humans" line was vacuous.

The corpus holds one map per song, but two of the songset maps carry **two difficulties by the
same mapper**. His ExpertPlus, read against his own Expert, is the first non-self-referential
human negative this bench has: genuinely different notes, indisputably good mapping, same song,
same author. A locator that fires on it is measuring *difference from a particular reference*,
not a defect — which is exactly what it must not do once P6 lets a user ask for a harder map.

⚠️It caught two things the same afternoon: the first `q_scatter` (vocabulary breadth per
hand-window) fired 5× on it and was rewritten because of that, and **`q_events` fires D6
"over-dense" six times on `1f333` ExpertPlus** — a true statement about density and a false
statement about quality. That is why the rows built here do NOT forbid `D1`/`D6`: those two
codes are density claims, and a harder difficulty is legitimately denser.

The audio is dropped: `score.py` resolves the song from `data/eval_songset` by id, so the
fixture only needs `Info.dat` + the one difficulty, which keeps it small enough to keep around.
"""

from __future__ import annotations

import pathlib
import zipfile

REPO = pathlib.Path(__file__).resolve().parents[1]
OUT = REPO / "outputs" / "bench_fixtures"

# song id -> the difficulty to KEEP as the fixture's map, and the prefix it is filed under.
# HUMANPLUS is read against data/raw/<id>.zip (where load_map prefers ExpertStandard), so it is
# the DENSER side. HUMANEXP is the same mapper's Expert read against HUMANPLUS — the SPARSER
# side, and the harsher of the two: before P5c it drew 7 EMPTY, 3 D4 and 1 D3 on 1f333.
DIFFICULTY_CONTROL = {
    ("HUMANPLUS", "1f333"): "expertplusstandard.dat",
    ("HUMANPLUS", "1f8d6"): "expertplusstandard.dat",
    ("HUMANEXP", "1f333"): "expertstandard.dat",
    ("HUMANEXP", "1f8d6"): "expertstandard.dat",
}


def difficulty_control(sid: str, keep: str, prefix: str = "HUMANPLUS") -> pathlib.Path:
    src = REPO / "data" / "raw" / f"{sid}.zip"
    dst = OUT / f"{prefix}__{sid}.zip"
    OUT.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(src) as z:
        names = [n for n in z.namelist()
                 if n.split("/")[-1].lower() in (keep, "info.dat")]
        if len(names) < 2:
            raise SystemExit(f"{src} has no {keep}")
        with zipfile.ZipFile(dst, "w", zipfile.ZIP_DEFLATED) as o:
            for n in names:
                o.writestr(n, z.read(n))
    return dst


def main() -> int:
    for (prefix, sid), keep in DIFFICULTY_CONTROL.items():
        p = difficulty_control(sid, keep, prefix)
        print(f"{p.relative_to(REPO)}  {p.stat().st_size / 1024:.0f} KB  "
              f"({zipfile.ZipFile(p).namelist()})")
    print("\nHUMANPLUS__<sid>  --vs <sid>                            (his ExpertPlus vs his Expert)"
          "\nHUMANEXP__<sid>   --vs outputs/bench_fixtures/HUMANPLUS__<sid>.zip   (the reverse)"
          "\na query that fires on either is reading DIFFICULTY, not a defect.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
