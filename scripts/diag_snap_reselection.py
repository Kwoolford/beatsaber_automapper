#!/usr/bin/env python
"""Does `--snap-onsets` REALIGN a map, or RESELECT it?

The snap costs 2 of 23 songs their PASS (`1f9a0`, `1fb3f`), and on both the damage is
on IDIOM axes while `onset_precision` RISES -- `idiom_coverage` 0.675 -> 0.350 on
`1fb3f`, and 0.99 -> 0.23 on `1f9a0`. An idiom is
`(dx, dy, dir_from, dir_to, dt_class)`, so a coverage collapse of that size is not a
timing change: **it means a different map got built.**

★**This is the exact shape of P0.4**, where a grid-phase shift looked like a
realignment and turned out to be a RESELECTION -- only 45.9 % of note beat positions
survived, so it was two different maps rather than one map moved. That was retracted
after being filed as a realignment, and the lesson was to measure SURVIVAL before
believing an alignment story.

**The measurement**: what fraction of BASE note beats still exist in SNAP.

    survival = |beats(BASE) & beats(SNAP)| / |beats(BASE)|

⚠️**Compared per song against the cohort**, because the cohort-wide geometry does not
move at all (`vertical_share` 0.754 -> 0.760, `diagonal_share` 0.243 -> 0.239). If the
two failing songs show ordinary survival, the idiom collapse is NOT reselection and
this hypothesis is dead. If they are outliers, the snap reselects on some songs and
the PASS cost is a note-selection effect, not an alignment one.

Usage:
    python scripts/diag_snap_reselection.py --json outputs/snap_reselection.json
"""
from __future__ import annotations

import argparse
import json
import pathlib
import statistics as st
import subprocess
import sys
import tempfile
import zipfile

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[1]
AM = REPO / "agent_mapper"
sys.path.insert(0, str(REPO / "src"))

from beatsaber_automapper.data.beatmap import (parse_difficulty_dat,  # noqa: E402
                                               parse_info_dat)

FAILED = {"1f9a0", "1fb3f"}  # the two songs the snap costs a PASS


def load_map(zp: pathlib.Path):
    """(note beats, bpm) from a generated zip."""
    tmp = pathlib.Path(tempfile.mkdtemp(prefix="resel_"))
    try:
        with zipfile.ZipFile(zp) as zf:
            names = zf.namelist()
            info = next((n for n in names
                         if n.split("/")[-1].lower() == "info.dat"), None)
            diff = next((n for n in names
                         if n.split("/")[-1].lower().endswith("standard.dat")
                         and "bpminfo" not in n.lower()), None)
            if info is None or diff is None:
                return None
            for n in (info, diff):
                (tmp / pathlib.Path(n).name).write_bytes(zf.read(n))
        meta = parse_info_dat(tmp / pathlib.Path(info).name)
        bm = parse_difficulty_dat(tmp / pathlib.Path(diff).name)
        if meta is None or bm is None:
            return None
        return bm, float(meta.bpm)
    except Exception:  # noqa: BLE001
        return None
    finally:
        import shutil
        shutil.rmtree(tmp, ignore_errors=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--songs", nargs="*", default=None)
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    from beatsaber_automapper.evaluation import idiom as ID

    sids = a.songs or [p.stem for p in
                       sorted((REPO / "data" / "eval_songset").glob("*.ogg"))]
    tmp = pathlib.Path(tempfile.mkdtemp(prefix="snapresel_"))
    rows = []
    print(f"{'song':8s}{'bpm':>7s}{'n_base':>8s}{'n_snap':>8s}"
          f"{'survival':>10s}{'idiomCov b->a':>18s}{'':>4s}")
    print("-" * 60)
    for sid in sids:
        audio = REPO / "data" / "eval_songset" / f"{sid}.ogg"
        got = {}
        for arm, extra in (("BASE", []), ("SNAP", ["--snap-onsets"])):
            out = tmp / f"{arm}__{sid}.zip"
            subprocess.run(
                [sys.executable, str(AM / "autobuild.py"), str(audio), "--pulse",
                 "--lead-bias", "0.2", "--name", f"rs_{sid}_{arm}",
                 "--out", str(out), *extra],
                capture_output=True, text=True, cwd=REPO)
            m = load_map(out) if out.exists() else None
            if m is not None:
                got[arm] = m
        if len(got) != 2:
            print(f"{sid:8s}  incomplete")
            continue

        (bm_b, bpm), (bm_s, _) = got["BASE"], got["SNAP"]
        beats_b = {round(n.beat, 3) for n in bm_b.color_notes}
        beats_s = {round(n.beat, 3) for n in bm_s.color_notes}
        surv = len(beats_b & beats_s) / max(len(beats_b), 1)

        cov_b = ID.idiom_metrics(bm_b).metrics["idiom_coverage"]
        cov_s = ID.idiom_metrics(bm_s).metrics["idiom_coverage"]

        mark = "  <-- FAILS" if sid in FAILED else ""
        print(f"{sid:8s}{bpm:7.0f}{len(bm_b.color_notes):8d}"
              f"{len(bm_s.color_notes):8d}{surv:10.3f}"
              f"{cov_b:8.3f} ->{cov_s:7.3f}{mark}")
        rows.append(dict(song=sid, bpm=bpm, n_base=len(bm_b.color_notes),
                         n_snap=len(bm_s.color_notes), survival=surv,
                         cov_base=cov_b, cov_snap=cov_s, fails=sid in FAILED))

    if not rows:
        print("no rows")
        return 1

    ok = [r for r in rows if not r["fails"]]
    bad = [r for r in rows if r["fails"]]
    print("-" * 60)
    if not ok:
        print("⚠️no non-failing songs in this run — the cohort comparison needs the "
              "full songset, so the numbers above are per-song only.")
    else:
        sd = st.pstdev([r["survival"] for r in ok])
        print(f"survival: cohort median {st.median([r['survival'] for r in ok]):.3f} "
              f"(n={len(ok)}), sd {sd:.3f}")
        # ★Also report the idiom_coverage DELTA spread. Survival can be ordinary while
        # coverage collapses -- that combination would mean the times were kept and the
        # GEOMETRY was re-drawn, which is a different mechanism from P0.4.
        dcov = [r["cov_snap"] - r["cov_base"] for r in ok]
        print(f"Δidiom_coverage: cohort median {st.median(dcov):+.3f}, "
              f"sd {st.pstdev(dcov):.3f}")
        for r in bad:
            z = (r["survival"] - st.mean([x["survival"] for x in ok])) / max(sd, 1e-9)
            zc = ((r["cov_snap"] - r["cov_base"] - st.mean(dcov))
                  / max(st.pstdev(dcov), 1e-9))
            print(f"  {r['song']}: survival {r['survival']:.3f} ({z:+.1f} sd)   "
                  f"Δcoverage {r['cov_snap'] - r['cov_base']:+.3f} ({zc:+.1f} sd)")

    print("\nVERDICT LOGIC")
    print("  Failing songs' survival INSIDE the cohort spread => the idiom collapse is")
    print("     NOT reselection; this hypothesis is dead and the cause is elsewhere.")
    print("  Failing songs clear OUTLIERS on survival => the snap RESELECTS on some")
    print("     songs, exactly as the grid phase did in P0.4, and the PASS cost is a")
    print("     note-SELECTION effect wearing an alignment change's clothes.")
    print("  ⚠️A correlation between survival and bpm would say WHICH songs are at")
    print("     risk -- the snap window is 60 ms against a 1/4-beat slot of 15000/bpm ms.")

    corr = np.corrcoef([r["bpm"] for r in rows],
                       [r["survival"] for r in rows])[0, 1]
    print(f"\n  corr(bpm, survival) = {corr:+.3f} over {len(rows)} songs")

    if a.json:
        p = pathlib.Path(a.json)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(rows, indent=2))
        print(f"wrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
