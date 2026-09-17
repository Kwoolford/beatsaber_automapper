#!/usr/bin/env python
"""Read `price_palette_strict.sh` (2026-09-16): control check FIRST, then the arms.

1. CONTROL: a fresh default build (`CTRL_s1__1f333.zip`, no --repeat-p) must equal the 2026-09-13
   `repeatp` baseline `rp0.25_s1__1f333.zip` on the difficulty JSON. ⚠️memory_cov2 is NOT a valid
   baseline: it predates REPEAT_P 0.55 -> 0.25 (commit 92a40a1), and its M0 builds differ on 558 of
   1433 notes. Nothing below is read if the control differs.
2. Per build: `mapjudge --top 30` (the tool that judges `idiom_coverage`, not a reimplementation)
   for idiom_coverage / idiom_local / idiom_jsd + percentiles and p; `q_scatter`'s own room; the
   verdict's red count.
3. Per song: the per-arm DISTRIBUTION, then the difference of means against the SE OF THE
   DIFFERENCE (never an arm's own sd).

Run: python scripts/eval_palette_strict.py
"""
from __future__ import annotations

import json
import pathlib
import re
import subprocess
import sys
import zipfile

import numpy as np

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
OUT = REPO / "outputs" / "palette_strict_2026-09-16"
CTRL = REPO / "outputs" / "repeatp_2026-09-13" / "rp0.25_s1__1f333.zip"
SONGS, SEEDS, ARMS = ("1f333", "1f8d6"), range(4, 10), (0, 20)
AXES = ("idiom_coverage", "idiom_local", "idiom_jsd", "idiom_top50")


def dat(zp):
    with zipfile.ZipFile(zp) as zf:
        n = next(x for x in zf.namelist() if x.lower().endswith("standard.dat"))
        return json.loads(zf.read(n))


def judge(zp):
    txt = subprocess.run([sys.executable, "-m", "beatsaber_automapper.evaluation.mapjudge",
                          str(zp), "--top", "30"], capture_output=True, text=True,
                         cwd=REPO).stdout
    row = {"p": float(re.search(r"p=([\d.]+)", txt).group(1))}
    for ax in AXES:
        m = re.search(rf"^\s*{ax}\s+([\d.]+)\s+human pct\s+([\d.]+)%", txt, re.M)
        row[ax] = float(m.group(1)) if m else float("nan")
        row[ax + "_pct"] = float(m.group(2)) if m else float("nan")
    return row


def scatter_and_reds(zp, sid):
    import queries as Q
    from verdict import load_arrays
    arrs, _, _ = load_arrays(zp, sid, "auto")
    rep: dict = {}
    fired = bool(Q.q_scatter(arrs, report=rep))
    room = rep.get("SCATTER", ("", float("nan")))[1]
    js = OUT / (zp.stem + ".verdict.json")
    subprocess.run([sys.executable, "scripts/verdict.py", str(zp), "--song", sid, "--no-bench",
                    "--json", str(js)], capture_output=True, cwd=REPO)
    reds = json.loads(js.read_text())["reds"] if js.exists() else -1
    return room, fired, reds


def main() -> int:
    if dat(OUT / "CTRL_s1__1f333.zip") != dat(CTRL):
        print("🔴 CONTROL FAILED — the default builder no longer reproduces the 09-13 baseline")
        return 1
    print("✅ control: today's default build reproduces repeatp rp0.25_s1 byte-for-byte\n")

    rows = []
    for sid in SONGS:
        for s in SEEDS:
            for p in ARMS:
                zp = OUT / f"P{p}_s{s}__{sid}.zip"
                r = judge(zp)
                r["room"], r["fired"], r["reds"] = scatter_and_reds(zp, sid)
                r.update(sid=sid, seed=s, arm=p)
                rows.append(r)
                print(f"{sid} s{s} P{p:<2d} cov {r['idiom_coverage']:.3f} ({r['idiom_coverage_pct']:4.1f}%)"
                      f"  local {r['idiom_local']:.3f} ({r['idiom_local_pct']:4.1f}%)"
                      f"  jsd {r['idiom_jsd']:.3f}  p {r['p']:.3f}  SCATTER room {r['room']:.2f}"
                      f"{' RED' if r['fired'] else ''}  reds {r['reds']}", flush=True)
    (OUT / "rows.json").write_text(json.dumps(rows, indent=1))

    print("\n== per song: arm distributions, and diff of means vs SE of the difference")
    for sid in SONGS:
        print(f"-- {sid}")
        for key in ("idiom_coverage", "idiom_local", "idiom_jsd", "p", "room", "reds"):
            x0 = np.array([r[key] for r in rows if r["sid"] == sid and r["arm"] == 0], float)
            x1 = np.array([r[key] for r in rows if r["sid"] == sid and r["arm"] == 20], float)
            se = np.sqrt(x0.var(ddof=1) / len(x0) + x1.var(ddof=1) / len(x1))
            d = x1.mean() - x0.mean()
            print(f"  {key:<15} P0 [{' '.join(f'{v:.2f}' for v in sorted(x0))}]"
                  f"  P20 [{' '.join(f'{v:.2f}' for v in sorted(x1))}]"
                  f"  Δ {d:+.3f}  ({d / se if se > 0 else float('nan'):+.1f} se)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
