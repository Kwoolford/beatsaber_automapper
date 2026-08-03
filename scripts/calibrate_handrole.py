#!/usr/bin/env python
"""Calibrate the hand-role human reference (eval suite v2, axis A6).

Computes the raw hand-role metrics over N human Standard maps from data/raw and writes
the median/MAD reference that `evaluation.rhythm` scoring scores
against. Scoring against the human *spread* (rather than a point target) is what
keeps "more extreme than human" from reading as "better" — the failure that
saturated h_dist. See docs/eval_suite_v2.md.

Usage:
  python scripts/calibrate_handrole.py --n 200
"""
from __future__ import annotations

import argparse
import json
import pathlib
import random
import shutil
import sys
import tempfile
import zipfile

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from beatsaber_automapper.data.beatmap import (  # noqa: E402
    parse_difficulty_dat, parse_info_dat,
)
from beatsaber_automapper.evaluation import handrole  # noqa: E402

RAW = REPO / "data" / "raw"


def load_human(zip_path: pathlib.Path, difficulty: str = "Expert"):
    """Parsed human Standard beatmap + bpm, preferring `difficulty` then ExpertPlus."""
    tmp = pathlib.Path(tempfile.mkdtemp(prefix="handrole_calib_"))
    try:
        with zipfile.ZipFile(zip_path) as zf:
            names = zf.namelist()
            # EXACT basename: "BPMInfo.dat" also ends with "info.dat", and 73 of 300
            # corpus zips list it FIRST -- picking it makes parse_info_dat find no
            # bpm and silently fall back to 120, which stretches every note time.
            info = next((n for n in names
                         if n.split("/")[-1].lower() == "info.dat"), None)
            std = [n for n in names if n.lower().split("/")[-1].endswith("standard.dat")]
            diff = None
            for cand in (difficulty.lower(), "expertplus"):
                for n in std:
                    b = n.lower().split("/")[-1]
                    if b.startswith(cand) and "plus" not in b.replace(cand, "", 1):
                        diff = n
                        break
                if diff:
                    break
            if info is None or diff is None:
                return None
            for n in (info, diff):
                (tmp / pathlib.Path(n).name).write_bytes(zf.read(n))
        meta = parse_info_dat(tmp / pathlib.Path(info).name)
        bm = parse_difficulty_dat(tmp / pathlib.Path(diff).name)
        if meta is None or bm is None or len(bm.color_notes) < 100:
            return None
        return bm, float(meta.bpm)
    except Exception:  # noqa: BLE001
        return None
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip", type=int, default=32,
                    help="skip this many maps before calibrating. audit_eval_suite.py "
                         "draws its human cohort from the head of the same seed-0 "
                         "shuffle, so skipping keeps the reference DISJOINT from the "
                         "cohort it will be used to judge (no in-sample flattery)")
    ap.add_argument("--out", default=str(handrole.REFERENCE_PATH))
    a = ap.parse_args()

    raws = sorted(RAW.glob("*.zip"))
    random.Random(a.seed).shuffle(raws)
    raws = raws[a.skip:]

    records = []
    for zp in raws:
        if len(records) >= a.n:
            break
        loaded = load_human(zp)
        if loaded is None:
            continue
        bm, bpm = loaded
        try:
            rep = handrole.handrole_metrics(bm)
        except Exception:  # noqa: BLE001
            continue
        records.append(rep.metrics)
        if len(records) % 25 == 0:
            print(f"  {len(records)}/{a.n} maps …", flush=True)

    ref = handrole.calibrate(records)
    out = pathlib.Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(ref, indent=2))

    print(f"\ncalibrated on {len(records)} human maps -> {out}\n")
    print(f"{'metric':20s} {'median':>10s} {'MAD':>10s} {'n':>6s}")
    for k, v in ref.items():
        print(f"{k:20s} {v['median']:10.3f} {v['mad']:10.3f} {v['n']:6d}")


if __name__ == "__main__":
    main()

