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
# ★A SECOND ARM ON THE SAME 149 SONGS. v8 (the instrument model) is the only arm
# that moved a masterpiece axis on the songset -- follow_vocals +0.008, and +0.015
# with the main-beat bonus, both >2sd across seeds -- so it is the acceptance
# metric for Track B. Confirming it at n=149 is worth an hour of otherwise idle GPU.
# ⚠️Same audio, same songs, same seed as the prod cohort, so the comparison is
# PAIRED by song and differs in exactly one thing.
V8_CKPT = ("logs/beat_classifier/version_8/checkpoints/"
           "beat-epoch=12-val_f1_avg_tol=0.598.ckpt")
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
    ap.add_argument("--variant", default="prod", choices=("prod", "v8"),
                    help="prod = the promoted defaults; v8 = the instrument model")
    ap.add_argument("--tag", default=None,
                    help="output-dir suffix for a lever arm, e.g. 'me_z20' -> "
                         "outputs/wide_cohort_me_z20. Required with --env so two "
                         "arms cannot resume into each other's files.")
    ap.add_argument("--env", action="append", default=[], metavar="KEY=VAL",
                    help="env var for the generator subprocess, repeatable. This is "
                         "how a default-OFF lever becomes an arm without a new "
                         "--variant per lever.")
    a = ap.parse_args()

    extra_env = {}
    for kv in a.env:
        k, _, v = kv.partition("=")
        if not _:
            ap.error(f"--env expects KEY=VAL, got {kv!r}")
        extra_env[k] = v
    if extra_env and not a.tag:
        ap.error("--env without --tag would write a lever arm into the control's "
                 "directory; pass --tag")

    # ⚠️The seed must be in the directory name or a second seed silently "resumes"
    # into the first one's files and you get one cohort labelled as two.
    tag = a.variant + ("" if a.seed == 0 else f"_s{a.seed}") + \
        (f"_{a.tag}" if a.tag else "")
    out_dir = OUT if tag == "prod" else REPO / "outputs" / f"wide_cohort_{tag}"
    out_dir.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    cands = candidates()[: a.n]
    if out_dir != OUT:
        # ⚠️ANY secondary cohort — another arm OR another seed — must cover exactly
        # the songs the primary one produced. Otherwise the difference between two
        # arms is partly a difference between two song SETS, which is this
        # project's most repeated mistake. Caught when the seed-1 run started
        # walking 200 candidates against the primary cohort's 149.
        have = {p.stem for p in OUT.glob("*.zip")}
        cands = [c for c in cands if c in have]
    print(f"{len(cands)} candidate songs (strict Expert + stem cache + audio)")
    if extra_env:
        print(f"arm env: {extra_env}  ->  {out_dir.name}")
    t0 = time.time()
    made = skipped = failed = 0
    for i, sid in enumerate(cands, 1):
        zp = out_dir / f"{sid}.zip"
        if a.resume and zp.exists() and zp.stat().st_size > 1000:
            skipped += 1
            continue
        au = extract_audio(sid)
        if au is None:
            failed += 1
            continue
        ckpt = BEAT_CKPT if a.variant == "prod" else V8_CKPT
        cmd = [str(REPO / ".venv/bin/python"), str(REPO / "scripts/generate.py"),
               str(au), "--v7", "--beat-ckpt", ckpt, "--layout-ckpt", LAYOUT_CKPT,
               "--difficulty", "Expert", "--section-gate", "loud_only",
               "--song-name", sid, "--seed", str(a.seed), "--output", str(zp)]
        if a.variant == "v8":
            cmd.append("--use-instr")
        run_env = None
        if extra_env:
            import os
            run_env = {**os.environ, **extra_env}
        r = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, env=run_env)
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
    print(f"maps in {out_dir}")


if __name__ == "__main__":
    main()
