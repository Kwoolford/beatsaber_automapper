#!/usr/bin/env python3
"""Eval a layout checkpoint across the eval_songset (2026-06-30).

Generates with density-select γ + a given layout ckpt + top_p, then reports
per-song row_concentration, column distribution, swing violations, and the
density_corr DoD (reusing eval_sweep's cached references). Used to evaluate the
entropy-reg fine-tuned layout models for the Stage-2 collapse fix.
"""
from __future__ import annotations
import argparse, os, pathlib, subprocess, sys
import numpy as np

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
from feel_disc_poc import load_v7
from best_of_n_poc import swing_violations
from eval_sweep import _get_ref, _list_songs, SONGSET, WIN_SEC, BEAT_CKPT
from eval_alignment import _load_generated_beatmap, _beat_to_seconds
from eval_density_corr import _bin_counts, _spearman


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layout-ckpt", required=True)
    ap.add_argument("--gamma", default="2.5")
    ap.add_argument("--top-p", default="0.999")
    ap.add_argument("--temperature", default="1.0")
    ap.add_argument("--tag", default="ft")
    args = ap.parse_args()

    out = REPO / "outputs" / f"layout_eval_{args.tag}"
    out.mkdir(parents=True, exist_ok=True)
    songs = _list_songs()
    print(f"=== eval {args.tag}: ckpt={pathlib.Path(args.layout_ckpt).name} "
          f"gamma={args.gamma} top_p={args.top_p} temp={args.temperature} ===")
    print(f"{'song':14s} {'row_conc':>9s} {'rows':>16s} {'cols':>22s} {'viol':>5s} {'dens_corr':>10s}")
    rcs, dcs, vls = [], [], []
    for s in songs:
        zp = out / f"{s.stem}.zip"
        env = dict(os.environ)
        env.update({"DENSITY_SELECT": "1", "DENSITY_SELECT_GAMMA": args.gamma})
        cmd = [sys.executable, "scripts/generate.py", str(s), "--v7", "--difficulty", "Expert",
               "--beat-ckpt", BEAT_CKPT, "--layout-ckpt", args.layout_ckpt,
               "--section-gate", "loud_only", "--temperature", args.temperature,
               "--top-p", args.top_p, "--output", str(zp)]
        r = subprocess.run(cmd, cwd=REPO, env=env, capture_output=True, text=True)
        if not zp.exists():
            print(f"{s.stem:14s}  GEN FAIL: {r.stderr.strip().splitlines()[-1][:60] if r.stderr.strip() else r.returncode}")
            continue
        seq = load_v7(str(zp), "Expert")
        ys = np.rint(seq[:, 2] * 2).astype(int); xs = np.rint(seq[:, 1] * 3).astype(int)
        yb = np.bincount(ys, minlength=3) / len(ys); xb = np.bincount(np.clip(xs, 0, 3), minlength=4) / len(xs)
        rc = float(max((ys == r).mean() for r in (0, 1, 2)))
        viol = swing_violations(zp, "Expert")
        # density_corr vs cached ref
        ref_times, dur = _get_ref(s)
        notes, bpm = _load_generated_beatmap(zp, "Expert")
        gt = np.array(sorted(_beat_to_seconds(b, bpm) for b, _x, _c in notes), dtype=np.float64)
        d = float(max(dur, gt.max() if len(gt) else 0))
        gd = _bin_counts(gt, d, WIN_SEC); rd = _bin_counts(ref_times, d, WIN_SEC)
        n = min(len(gd), len(rd)); dc = _spearman(gd[:n], rd[:n])
        rcs.append(rc); dcs.append(dc); vls.append(viol or 0)
        print(f"{s.stem:14s} {rc:9.3f} {str(yb.round(2)):>16s} {str(xb.round(2)):>22s} {str(viol):>5s} {dc:10.3f}")
    if rcs:
        print(f"\n  MEAN row_conc={np.mean(rcs):.3f}  density_corr={np.mean(dcs):.3f} "
              f"(#pass {sum(d>=0.41 for d in dcs)}/{len(dcs)})  total_viol={sum(vls)}")
        print(f"  (baseline v10: row_conc 0.94, human ~0.47; density_corr target >=0.41)")


if __name__ == "__main__":
    main()
