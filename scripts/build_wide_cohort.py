#!/usr/bin/env python
"""WIDEN THE PAIRED COHORT — the cheapest large upgrade the suite can get.

Every masterpiece-axis claim from 2026-08-04 rests on **13 songs**: the eval songset
is 24 songs and only 13 of them ship a strict Expert human map. Thirteen paired
observations is enough to resolve a 3x effect and nowhere near enough to resolve the
levers we actually want to rank, and this project has already been burned once by
n=3 lying about a spread (idiom's sd went 0.043 -> 0.107 on adding two seeds).

Nothing was blocking a bigger cohort except our own maps. The corpus already has
**250 songs** that have (a) a strict `ExpertStandard.dat`, (b) a seeded stem-onset
cache, and (c) their audio inside the map zip. Generation is ~18 s per song at the
promoted defaults, so the paired cohort can go from 13 to ~150 in about an hour of
otherwise idle GPU.

⚠️This does NOT replace the eval songset. The songset is the fixed ruler that every
historical arm was scored against and it must stay comparable. This is a second,
wider cohort used to ask one question: **do the findings replicate at n=150?**

Usage:
    python scripts/build_wide_cohort.py --n 150            # extract + generate
    python scripts/build_wide_cohort.py --n 150 --resume   # skip what exists
"""

from __future__ import annotations

import argparse
import glob
import pathlib
import subprocess
import sys
import time
import zipfile

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from calibrate_playfeel import load_expert_only  # noqa: E402

OUT = REPO / "outputs" / "wide_cohort"
AUDIO = OUT / "audio"
BEAT_CKPT = ("logs/beat_classifier/version_4/checkpoints/"
             "beat-epoch=11-val_f1_avg_tol=0.603.ckpt")
LAYOUT_CKPT = ("logs/layout_phrase/version_10/checkpoints/"
               "layout-epoch=09-val_token_acc=0.865.ckpt")


def candidates() -> list[str]:
    have_stem = {pathlib.Path(p).stem
                 for p in glob.glob(str(REPO / "outputs/stem_onset_cache/*.npz"))}
    songset = {pathlib.Path(p).stem
               for p in glob.glob(str(REPO / "data/eval_songset/*"))}
    out = []
    for sid in sorted(have_stem):
        if sid in songset:
            continue
        z = REPO / "data" / "raw" / f"{sid}.zip"
        if not z.exists():
            continue
        if load_expert_only(z):
            out.append(sid)
    return out


def extract_audio(sid: str) -> pathlib.Path | None:
    AUDIO.mkdir(parents=True, exist_ok=True)
    dst = AUDIO / f"{sid}.ogg"
    if dst.exists() and dst.stat().st_size > 10_000:
        return dst
    try:
        with zipfile.ZipFile(REPO / "data" / "raw" / f"{sid}.zip") as zf:
            name = next((n for n in zf.namelist()
                         if n.lower().endswith((".egg", ".ogg"))), None)
            if name is None:
                return None
            dst.write_bytes(zf.read(name))
        return dst
    except Exception:
        return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", action="store_true", default=True)
    a = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    cands = candidates()[: a.n]
    print(f"{len(cands)} candidate songs (strict Expert + stem cache + audio)")
    t0 = time.time()
    made = skipped = failed = 0
    for i, sid in enumerate(cands, 1):
        zp = OUT / f"{sid}.zip"
        if a.resume and zp.exists() and zp.stat().st_size > 1000:
            skipped += 1
            continue
        au = extract_audio(sid)
        if au is None:
            failed += 1
            continue
        cmd = [str(REPO / ".venv/bin/python"), str(REPO / "scripts/generate.py"),
               str(au), "--v7", "--beat-ckpt", BEAT_CKPT, "--layout-ckpt", LAYOUT_CKPT,
               "--difficulty", "Expert", "--section-gate", "loud_only",
               "--song-name", sid, "--seed", str(a.seed), "--output", str(zp)]
        r = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)
        if r.returncode != 0 or not zp.exists():
            failed += 1
            print(f"  [{i}/{len(cands)}] {sid} FAILED: "
                  f"{(r.stderr or '').strip().splitlines()[-1:] }")
            continue
        made += 1
        if made % 10 == 0:
            el = time.time() - t0
            print(f"  [{i}/{len(cands)}] {made} made, {skipped} cached, {failed} failed, "
                  f"{el/60:.1f} min, {el/max(made,1):.1f}s/song")
    print(f"\nDONE: {made} generated, {skipped} already present, {failed} failed")
    print(f"maps in {OUT}")


if __name__ == "__main__":
    main()
