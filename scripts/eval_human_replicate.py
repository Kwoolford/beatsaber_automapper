#!/usr/bin/env python
"""THE HUMAN SIDE'S NOISE FLOOR — how much do these axes move between two human maps
of the same song?

We now know our own floor: the same config at a different seed moves every
steer-safe axis by ≤0.004 over 149 songs. The human side has no such number, and
without it a human value is a point with no error bar — which is exactly the trap
`h_dist` fell into.

The corpus gives a natural replicate: many maps ship **Expert** *and* **ExpertPlus**
of the same song, usually by the same mapper. That is the same person reading the
same music twice, at a different difficulty. Scoring both tells us:

    how much of an axis is the SONG and the mapper's reading of it   (stable)
    how much is the particular map they were writing at the time     (moves)

★An axis that swings wildly between one mapper's two takes cannot support a claim
that *our* map is below *their* map by a small margin — the margin has to clear this
number, not just clear zero.

⚠️ExpertPlus is denser by construction, so part of any movement here is density.
The battery measured what density does to these axes (a 30 % thin retains 0.6–0.9),
so read this as an UPPER bound on the human replicate noise, not a clean one.

Usage:
    python scripts/eval_human_replicate.py --json outputs/human_replicate.json
"""

from __future__ import annotations

import argparse
import glob
import json
import pathlib
import shutil
import statistics as st
import sys
import tempfile
import zipfile

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

import masterpiece_report as mr  # noqa: E402
import song_structure as ss  # noqa: E402
from beatsaber_automapper.data.beatmap import (parse_difficulty_dat,  # noqa: E402
                                               parse_info_dat)


def load_difficulty(zp: pathlib.Path, basename: str):
    """Load one named difficulty. ⚠️Exact basename match — "BPMInfo.dat" also ends
    with "info.dat", and 73 of 300 corpus zips list it FIRST, which silently makes
    bpm fall back to 120 and stretches every note time."""
    tmp = pathlib.Path(tempfile.mkdtemp(prefix="repl_"))
    try:
        with zipfile.ZipFile(zp) as zf:
            names = zf.namelist()
            info = next((n for n in names
                         if n.split("/")[-1].lower() == "info.dat"), None)
            diff = next((n for n in names
                         if n.split("/")[-1].lower() == basename), None)
            if info is None or diff is None:
                return None
            for n in (info, diff):
                (tmp / pathlib.Path(n).name).write_bytes(zf.read(n))
        meta = parse_info_dat(tmp / pathlib.Path(info).name)
        bm = parse_difficulty_dat(tmp / pathlib.Path(diff).name)
        if meta is None or bm is None or len(bm.color_notes) < 100:
            return None
        return bm, float(meta.bpm)
    except Exception:
        return None
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json", default="")
    ap.add_argument("--limit", type=int, default=200)
    a = ap.parse_args()

    songs = [pathlib.Path(f).stem
             for f in sorted(glob.glob(str(REPO / "outputs/wide_cohort/*.zip")))][: a.limit]
    rows = []
    for song in songs:
        zp = REPO / "data" / "raw" / f"{song}.zip"
        if not zp.exists():
            continue
        e = load_difficulty(zp, "expertstandard.dat")
        ep = load_difficulty(zp, "expertplusstandard.dat")
        if not e or not ep:
            continue
        from beatsaber_automapper.evaluation import alignment
        t = np.asarray(alignment.note_times(e[0], e[1]), dtype=float)
        B = ss.bars(song, e[1], ss.song_end(song, float(t.max())))
        if B is None or B.n < 24:
            continue
        A = ss.bar_audio_matrix(song, B)
        stems = mr.m2.stem_onsets(song)
        if A is None or len(stems) < 3:
            continue
        nov = mr.m4.novelty(A)
        bnds = mr.m4.boundaries(nov) if nov is not None else []
        s_e = mr.score_one(song, mr.m1.notes_xydc(e[0], e[1]), B, A, stems, bnds)
        s_ep = mr.score_one(song, mr.m1.notes_xydc(ep[0], ep[1]), B, A, stems, bnds)
        rows.append({"song": song, "expert": s_e, "expertplus": s_ep})
        if len(rows) % 20 == 0:
            print(f"  {len(rows)} songs with both difficulties")

    if len(rows) < 20:
        print(f"only {len(rows)} songs have both Expert and ExpertPlus")
        return

    print(f"\n{'='*100}")
    print(f"HUMAN REPLICATE — the same mapper's Expert vs ExpertPlus, {len(rows)} songs")
    print(f"{'='*100}")
    print(f"{'axis':<20} {'Expert':>9} {'Expert+':>9} {'paired Δ':>10} {'resolv':>7} "
          f"{'|Δ| vs OUR gap':>15}")

    # our gap to the human, from the same wide cohort, for scale
    wide = mr.collect("prod", "s0", rebuild=False, wide=True)
    out = {}
    for _, k in mr.REPORT_KEYS:
        d = ss.paired_delta(rows, k, a="expertplus", b="expert")
        if not d:
            continue
        ours = ss.paired_delta(wide, k)
        ratio = (abs(d["delta"]) / abs(ours["delta"])
                 if ours and ours.get("delta") else None)
        ve = [r["expert"][k] for r in rows if r["expert"].get(k) is not None]
        vp = [r["expertplus"][k] for r in rows if r["expertplus"].get(k) is not None]
        out[k] = {"expert": round(st.median(ve), 4),
                  "expertplus": round(st.median(vp), 4),
                  "paired": d, "vs_our_gap": round(ratio, 3) if ratio else None}
        print(f"{k:<20} {st.median(ve):>+9.4f} {st.median(vp):>+9.4f} "
              f"{d['delta']:>+10.4f} {('YES' if d['resolvable'] else 'no'):>7} "
              f"{(ratio if ratio is not None else float('nan')):>15.2f}")

    print("\nHOW TO READ: the last column is |this human-vs-human delta| ÷ |our gap to the")
    print("human| on the same axis. Below ~0.3 means our gap is comfortably larger than")
    print("the same mapper's own variation between two takes. Near or above 1 means the")
    print("axis cannot tell us apart from a mapper having a different day.")

    if a.json:
        pathlib.Path(a.json).write_text(json.dumps(
            {"n_songs": len(rows), "axes": out}, indent=2))
        print(f"\nwrote {a.json}")


if __name__ == "__main__":
    main()
