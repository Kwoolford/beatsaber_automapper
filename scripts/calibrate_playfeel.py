#!/usr/bin/env python
"""Calibrate the difficulty + direction-idiom reference (eval suite v2, axis A7).

STRICTLY Expert-only. The other calibrators fall back to ExpertPlus when a map
has no Expert difficulty, which is fine for pattern axes but would defeat this
one: difficulty is exactly what it measures, and ExpertPlus maps are denser by
definition. A contaminated reference here would tell us our too-dense maps are
fine.
"""
from __future__ import annotations
import argparse, json, pathlib, random, shutil, sys, tempfile, zipfile
REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from beatsaber_automapper.data.beatmap import parse_difficulty_dat, parse_info_dat  # noqa: E402
from beatsaber_automapper.evaluation import playfeel  # noqa: E402
RAW = REPO / "data" / "raw"

def load_expert_only(zp: pathlib.Path):
    tmp = pathlib.Path(tempfile.mkdtemp(prefix="playfeel_"))
    try:
        with zipfile.ZipFile(zp) as zf:
            names = zf.namelist()
            info = next((n for n in names if n.lower().endswith("info.dat")), None)
            diff = next((n for n in names
                         if (b := n.lower().split("/")[-1]) == "expertstandard.dat"), None)
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

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip", type=int, default=32)
    a = ap.parse_args()
    raws = sorted(RAW.glob("*.zip")); random.Random(a.seed).shuffle(raws)
    recs = []
    for zp in raws[a.skip:]:
        if len(recs) >= a.n: break
        r = load_expert_only(zp)
        if r is None: continue
        recs.append(playfeel.playfeel_metrics(r[0], bpm=r[1]).metrics)
        if len(recs) % 50 == 0: print(f"  {len(recs)}/{a.n} …", flush=True)
    ref = playfeel.calibrate(recs)
    playfeel.REFERENCE_PATH.write_text(json.dumps(ref, indent=2))
    print(f"\nEXPERT-ONLY reference from {len(recs)} maps -> {playfeel.REFERENCE_PATH}\n")
    print(f"{'metric':18s}{'median':>10s}{'MAD':>10s}{'n':>6s}")
    for k, v in ref.items():
        print(f"{k:18s}{v['median']:10.3f}{v['mad']:10.3f}{v['n']:6d}")

if __name__ == "__main__":
    main()
