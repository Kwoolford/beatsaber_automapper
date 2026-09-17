#!/usr/bin/env python
"""What would it cost to BAKE the fitted phase into the beats? (2026-09-16g, option b)

Export writes the grid phase only to `_songTimeOffset`, which BSMG calls deprecated and which
`mapjudge` ignores. Human maps carry 0 and put the timing in the beat numbers. This copies each
build in `outputs/phase16_2026-09-16/` with every beat-timed object (notes, bombs, walls, arcs,
chains, including arc/chain tail beats) moved by `offset / spb` and the offset set to 0, then judges
both copies with `mapjudge --top 30`.

Expected: `onset_precision` moves to the "apply" column of `exp_phase_sweep.py` (the judge can now
see the phase), and `offgrid_frac` jumps wherever |shift| > 0.01 beats (its tolerance) — the guard
is what this measures the price of.

Run: python scripts/exp_bake_phase.py
"""
from __future__ import annotations

import json
import pathlib
import re
import subprocess
import sys
import zipfile

REPO = pathlib.Path(__file__).resolve().parent.parent
SRC = REPO / "outputs" / "phase16_2026-09-16"
DST = REPO / "outputs" / "bakephase_2026-09-16"
KEYS = ("colorNotes", "bombNotes", "obstacles", "sliders", "burstSliders")


def bake(src: pathlib.Path, dst: pathlib.Path) -> float:
    with zipfile.ZipFile(src) as zin:
        items = {n: zin.read(n) for n in zin.namelist()}
    info_n = next(n for n in items if n.split("/")[-1].lower() == "info.dat")
    info = json.loads(items[info_n])
    off = float(info.get("_songTimeOffset") or 0.0)
    shift = off * float(info["_beatsPerMinute"]) / 60.0          # seconds -> beats
    for n in list(items):
        if not n.lower().endswith(".dat") or n == info_n:
            continue
        try:
            d = json.loads(items[n])
        except Exception:  # noqa: BLE001
            continue
        if not isinstance(d, dict) or not any(k in d for k in KEYS):
            continue
        for k in KEYS:
            for o in d.get(k, []) or []:
                for f in ("b", "tb"):
                    if f in o:
                        o[f] = round(float(o[f]) + shift, 4)
        items[n] = json.dumps(d).encode()
    info["_songTimeOffset"] = 0.0
    items[info_n] = json.dumps(info).encode()
    dst.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(dst, "w", zipfile.ZIP_DEFLATED) as zout:
        for n, b in items.items():
            zout.writestr(n, b)
    return off


def judge(zp):
    txt = subprocess.run([sys.executable, "-m", "beatsaber_automapper.evaluation.mapjudge",
                          str(zp), "--top", "30"], capture_output=True, text=True, cwd=REPO).stdout
    out = {"p": float(re.search(r"p=([\d.]+)", txt).group(1)),
           "verdict": "PASS" if ": PASS" in txt else "FAIL"}
    for ax in ("onset_precision", "offgrid_frac"):
        m = re.search(rf"^\s*{ax}\s+([\d.]+)\s+human pct\s+([\d.]+)%", txt, re.M)
        out[ax] = (float(m.group(1)), float(m.group(2))) if m else (float("nan"), float("nan"))
    return out


def main() -> int:
    print(f"{'song':<6}{'offset':>8} | {'as built':^30} | {'baked':^30}")
    flips = []
    for zp in sorted(SRC.glob("B__*.zip")):
        sid = zp.stem.split("__")[1]
        dst = DST / f"K__{sid}.zip"
        off = bake(zp, dst)
        a, b = judge(zp), judge(dst)
        row = []
        for r in (a, b):
            row.append(f"{r['verdict']} p {r['p']:.3f} on {r['onset_precision'][0]:.3f} "
                       f"og {r['offgrid_frac'][0]:.2f}({r['offgrid_frac'][1]:3.0f}%)")
        print(f"{sid:<6}{off * 1000:>+6.0f}ms | {row[0]} | {row[1]}")
        if a["verdict"] != b["verdict"]:
            flips.append((sid, a["verdict"], b["verdict"]))
    print(f"\nverdict flips: {flips or 'none'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
