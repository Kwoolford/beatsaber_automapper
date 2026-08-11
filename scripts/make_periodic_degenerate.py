#!/usr/bin/env python
"""THE DEGENERATE THE BATTERY IS MISSING — a map that repeats at a FIXED LAG.

🔴WHY THIS HAD TO BE BUILT, AND WHAT IT PUTS AT RISK. `diag_full` moved `rhy_rhythm`
+0.0423 and `harm_rhythm` +0.0542 at n=149 — the first resolvable movement of a
masterpiece axis in this project. Then `view_structure.py` on アリスブルー showed our
AFTER panel is a **rigid periodic checkerboard** while the music's and the human's are
both irregular. The map is repeating *mechanically*, and the axes applauded.

The M-axes were designed as CONTRASTS precisely so degenerates score ~0, and they do —
against the degenerates `audit_masterpiece.py` contains: a metronome, random note
times, a bar-rotated map, another song's map. **A structurally periodic map is not one
of them.** And it is the one that matters here, because musical repeats are themselves
often periodic (8- and 16-bar phrases), so a map that copies bar *i* from bar *i−k* at a
fixed *k* can look like it "repeats where the music repeats" while ignoring the music
completely.

**This script builds exactly that map**, from the control cohort, using the evaluator's
own bar grid, with the audio ignored entirely. Score it with `masterpiece_report.py`.

🔴**THE PRE-REGISTERED READING** — written before the numbers exist:
  degenerate scores NEAR OR ABOVE our arm on `rhy_rhythm` / `harm_rhythm` / `harm_place`
      ⇒ those axes are **NOT steer-safe for this class of lever**. The 2026-08-11
        headline is then partly an artifact and every claim built on it must be re-read.
        This would be the most useful possible outcome, and it costs one hour.
  degenerate scores WELL BELOW our arm
      ⇒ the axes distinguish musical repetition from mechanical repetition, the
        headline stands as measured, and the checkerboard is a dose problem rather than
        a measurement problem.
⚠️Either way this does NOT settle whether the map is good — that is Kyle's ear. It
  settles whether the ruler can tell structure from regularity.

Usage:
    python scripts/make_periodic_degenerate.py --lag 8
    python scripts/masterpiece_report.py --arm periodic_k8 --wide \\
        --wide-dir outputs/wide_cohort_periodic_k8 --vs prod --vs-wide-dir outputs/wide_cohort
"""

from __future__ import annotations

import argparse
import glob
import json
import pathlib
import shutil
import sys
import zipfile

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))


def rebuild(src_zip: pathlib.Path, dst_zip: pathlib.Path, lag: int) -> str:
    """Copy the zip, replacing ExpertStandard.dat with its fixed-lag repeat."""
    import song_structure as ss
    from beatsaber_automapper.evaluation import scorecard

    L = scorecard._load_any(src_zip)
    if not L:
        return "unreadable"
    bpm = float(L[1])
    B = ss.bars(src_zip.stem, bpm, ss.song_end(src_zip.stem))
    if B is None or B.n < 24:
        return "no bar grid"

    with zipfile.ZipFile(src_zip) as zf:
        names = zf.namelist()
        dat = next((n for n in names if n.lower().endswith("expertstandard.dat")), None)
        if dat is None:
            return "no ExpertStandard.dat"
        doc = json.loads(zf.read(dat).decode("utf-8", "ignore"))

    notes = doc.get("colorNotes", [])
    if not notes:
        return "no notes"
    spb = 60.0 / bpm
    edges = B.edges
    dur = float(B.dur)

    by_bar: dict[int, list] = {}
    for n in notes:
        t = float(n["b"]) * spb
        if t < edges[0] or t >= edges[-1]:
            continue
        bi = int((t - edges[0]) // dur)
        if 0 <= bi < B.n:
            by_bar.setdefault(bi, []).append(n)

    out, kept = [], 0
    for bi in range(B.n):
        src = bi - lag
        if src < 0:
            out.extend(by_bar.get(bi, []))
            kept += 1
            continue
        for n in by_bar.get(src, []):
            t = float(n["b"]) * spb + lag * dur          # shift forward by `lag` bars
            if t >= edges[-1]:
                continue
            m = dict(n)
            m["b"] = t / spb
            out.append(m)
    # notes outside the grid entirely are preserved so the map stays well-formed
    for n in notes:
        t = float(n["b"]) * spb
        if t < edges[0] or t >= edges[-1]:
            out.append(n)
    doc["colorNotes"] = sorted(out, key=lambda n: n["b"])

    dst_zip.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_zip, dst_zip)
    # rewrite the one member, preserving everything else (audio, Info.dat, lighting)
    tmp = dst_zip.with_suffix(".tmp.zip")
    with zipfile.ZipFile(src_zip) as zin, zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zout:
        for it in zin.infolist():
            data = zin.read(it.filename)
            if it.filename == dat:
                data = json.dumps(doc).encode("utf-8")
            zout.writestr(it, data)
    tmp.replace(dst_zip)
    return f"ok ({len(notes)} -> {len(doc['colorNotes'])} notes, {kept} bars kept as-is)"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lag", type=int, default=8, help="fixed repeat period, in bars")
    ap.add_argument("--src", default="outputs/wide_cohort")
    ap.add_argument("--out", default="")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    out = pathlib.Path(a.out or f"outputs/wide_cohort_periodic_k{a.lag}")
    files = sorted(glob.glob(f"{a.src}/*.zip"))
    if a.limit:
        files = files[: a.limit]
    print(f"building a fixed-lag-{a.lag} degenerate from {len(files)} control maps")
    print("⚠️the audio is IGNORED — that is the point\n")
    ok = bad = 0
    for i, f in enumerate(files, 1):
        src = pathlib.Path(f)
        r = rebuild(src, out / src.name, a.lag)
        if r.startswith("ok"):
            ok += 1
        else:
            bad += 1
        if i % 25 == 0:
            print(f"  [{i}/{len(files)}] {ok} built, {bad} skipped")
    print(f"\nDONE: {ok} built, {bad} skipped -> {out}")
    print(f"\nNow score it:\n  python scripts/masterpiece_report.py --arm periodic_k{a.lag} "
          f"--wide --wide-dir {out} --vs prod --vs-wide-dir outputs/wide_cohort")


if __name__ == "__main__":
    main()
