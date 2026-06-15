#!/usr/bin/env python3
"""Generate a V7 cohort from real songs for the AUC(human vs V7) reward test (2026-06-10).

The 06-10 n=1500 gate showed the handcrafted-feature reward can't separate human
from our 4 sample V7 maps (DoD-B collapsed GREEN->AMBER). The build plan's real
gate is AUC(human vs V7) >= 0.75. That needs MANY V7 maps over real songs.

UNLOCK: data/raw/*.zip each bundle Song.egg (audio) + Info.dat (BPM), so we are NOT
limited to the one test song. This extracts audio+bpm from N raw maps and generates a
production-config V7 Expert map for each (beat version_4 + layout version_10,
section_gate=loud_only). Output zips land in --out-dir, named <songid>.zip so each
can be matched to its human map (data/processed/<songid>.pt) downstream.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import random
import subprocess
import sys
import tempfile
import time
import zipfile

REPO = pathlib.Path(__file__).resolve().parent.parent
BEAT_CKPT = REPO / "logs/beat_classifier/version_4/checkpoints/beat-epoch=11-val_f1_avg_tol=0.603.ckpt"
LAYOUT_CKPT = REPO / "logs/layout_phrase/version_10/checkpoints/layout-epoch=09-val_token_acc=0.865.ckpt"


def extract_song(zip_path: pathlib.Path, work: pathlib.Path):
    """Extract Song.egg -> <stem>.ogg and read bpm from Info.dat. Returns (ogg, bpm) or None."""
    try:
        with zipfile.ZipFile(zip_path) as zf:
            names = zf.namelist()
            # match the basename exactly: "info.dat" (NOT "BPMInfo.dat", which also
            # ends with "info.dat" but carries no _beatsPerMinute).
            info_name = next(
                (n for n in names if pathlib.PurePosixPath(n).name.lower() == "info.dat"),
                None,
            )
            if not info_name:
                return None
            info = json.loads(zf.read(info_name).decode("utf-8", "ignore"))
            bpm = float(info.get("_beatsPerMinute") or info.get("_beatsPerMinute".lower()) or 0)
            song_fn = info.get("_songFilename", "Song.egg")
            if bpm <= 0 or song_fn not in names:
                # fall back to any .egg/.ogg
                song_fn = next((n for n in names if n.lower().endswith((".egg", ".ogg"))), None)
                if not song_fn or bpm <= 0:
                    return None
            ogg = work / (zip_path.stem + ".ogg")
            ogg.write_bytes(zf.read(song_fn))
            return ogg, bpm
    except Exception as e:
        print(f"  [extract-fail] {zip_path.name}: {e}")
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--difficulty", default="Expert")
    ap.add_argument("--out-dir", type=pathlib.Path, default=REPO / "outputs/v7_cohort_2026-06-10")
    ap.add_argument("--raw-glob", default=str(REPO / "data/raw/*.zip"))
    ap.add_argument("--timeout", type=int, default=600, help="per-song generation timeout (s)")
    args = ap.parse_args()

    assert BEAT_CKPT.exists(), f"missing {BEAT_CKPT}"
    assert LAYOUT_CKPT.exists(), f"missing {LAYOUT_CKPT}"
    args.out_dir.mkdir(parents=True, exist_ok=True)

    import glob
    raws = sorted(glob.glob(args.raw_glob))
    random.Random(args.seed).shuffle(raws)

    done, failed, gen_ms = [], [], []
    t_all = time.time()
    for zp in raws:
        if len(done) >= args.n:
            break
        zip_path = pathlib.Path(zp)
        out_zip = args.out_dir / f"{zip_path.stem}.zip"
        if out_zip.exists():
            done.append(zip_path.stem)
            continue
        with tempfile.TemporaryDirectory() as td:
            ex = extract_song(zip_path, pathlib.Path(td))
            if ex is None:
                failed.append((zip_path.stem, "extract"))
                continue
            ogg, bpm = ex
            cmd = [
                sys.executable, str(REPO / "scripts/generate.py"), str(ogg),
                "--v7", "--difficulty", args.difficulty,
                "--beat-ckpt", str(BEAT_CKPT), "--layout-ckpt", str(LAYOUT_CKPT),
                "--bpm", str(bpm), "--section-gate", "loud_only",
                "--output", str(out_zip),
            ]
            t0 = time.time()
            try:
                r = subprocess.run(cmd, cwd=str(REPO), capture_output=True,
                                   text=True, timeout=args.timeout)
            except subprocess.TimeoutExpired:
                failed.append((zip_path.stem, "timeout"))
                print(f"  [timeout] {zip_path.stem}")
                continue
            dt = time.time() - t0
            if r.returncode != 0 or not out_zip.exists():
                failed.append((zip_path.stem, "gen"))
                print(f"  [gen-fail rc={r.returncode}] {zip_path.stem} bpm={bpm}\n{r.stderr[-500:]}")
                continue
            gen_ms.append(dt)
            done.append(zip_path.stem)
            print(f"  [ok {len(done)}/{args.n}] {zip_path.stem} bpm={bpm:.0f} {dt:.0f}s")

    avg = sum(gen_ms) / len(gen_ms) if gen_ms else 0
    print(f"\n[cohort] done={len(done)} failed={len(failed)} avg_gen={avg:.0f}s "
          f"total={time.time()-t_all:.0f}s -> {args.out_dir}")
    if failed:
        print(f"[cohort] failures: {failed[:20]}")
    (args.out_dir / "_manifest.json").write_text(json.dumps(
        {"done": done, "failed": failed, "avg_gen_s": avg}, indent=2))


if __name__ == "__main__":
    main()
