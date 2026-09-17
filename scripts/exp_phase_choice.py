#!/usr/bin/env python
"""P1.0: when should a song keep its RAW phase instead of the cohort calibration? (2026-09-16i)

Reads the calibrated builds (`outputs/phase16_2026-09-16/`) and the `--no-phase-calibrate` builds
(`outputs/phase16raw_2026-09-16/`) of the same 23 songs. Precision is taken with the offset APPLIED
(what export intends; the judge cannot see it, 09-16g). The residual is each build's best global
shift minus the HUMAN map's best shift, so the shared onset-detector bias cancels; 0 = our grid
sits where his does.

Asks whether the event cache's `grid_r` (how periodic the song's events are on the fitted grid)
separates the songs where raw wins — the candidate confidence gate.

Run: python scripts/exp_phase_choice.py
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from exp_phase_sweep import best, load, prec  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parent.parent
CAL = REPO / "outputs" / "phase16_2026-09-16"
RAW = REPO / "outputs" / "phase16raw_2026-09-16"


def main() -> int:
    rows = []
    for zc in sorted(CAL.glob("B__*.zip")):
        sid = zc.stem.split("__")[1]
        zr = RAW / zc.name
        ev = REPO / "outputs" / "event_cache" / f"{sid}.6s.json"
        hz = REPO / "data" / "raw" / f"{sid}.zip"
        if not (zr.exists() and ev.exists() and hz.exists()):
            continue
        on = np.load(REPO / "outputs" / "onset_cache" / f"{sid}.npz")["onsets"]
        g = json.loads(ev.read_text())["grid_r"]
        _, ho, ht = load(hz)
        hs = best(ht + ho, on)[1]
        r = dict(sid=sid, grid_r=g)
        for tag, zp in (("cal", zc), ("raw", zr)):
            _, off, t = load(zp)
            r[tag] = prec(t + off, on)
            r[tag + "_off"] = off
            r[tag + "_res"] = best(t + off, on)[1] - hs
        rows.append(r)
    rows.sort(key=lambda r: r["grid_r"])
    print(f"{'song':<6}{'grid_r':>7}{'cal off':>9}{'raw off':>9}{'cal':>7}{'raw':>7}"
          f"{'Δ raw':>8}{'cal res':>9}{'raw res':>9}")
    for r in rows:
        print(f"{r['sid']:<6}{r['grid_r']:>7.3f}{r['cal_off'] * 1000:>+7.0f}ms{r['raw_off'] * 1000:>+7.0f}ms"
              f"{r['cal']:>7.3f}{r['raw']:>7.3f}{r['raw'] - r['cal']:>+8.3f}"
              f"{r['cal_res'] * 1000:>+7.0f}ms{r['raw_res'] * 1000:>+7.0f}ms")
    g = np.array([r["grid_r"] for r in rows])
    d = np.array([r["raw"] - r["cal"] for r in rows])
    print(f"\n{len(rows)} songs; raw beats calibrated on {(d > 0.005).sum()}, loses on "
          f"{(d < -0.005).sum()}; r(grid_r, Δ raw) = {np.corrcoef(g, d)[0, 1]:+.3f}")
    cr = np.array([abs(r["cal_res"]) for r in rows])
    rr = np.array([abs(r["raw_res"]) for r in rows])
    print(f"|residual vs human|: calibrated median {np.median(cr) * 1000:.0f} ms (≤ 20 ms on "
          f"{(cr <= 0.02).sum()}), raw median {np.median(rr) * 1000:.0f} ms (≤ 20 ms on {(rr <= 0.02).sum()})")
    for cut in sorted(set(np.round(g, 3)))[:6]:
        pick = np.where(g <= cut, [r["raw"] for r in rows], [r["cal"] for r in rows])
        cal = np.array([r["cal"] for r in rows])
        print(f"  gate 'raw if grid_r <= {cut:.3f}': ≥ calibrated on {(pick >= cal - 1e-9).sum()}, "
              f"mean Δ {np.mean(pick - cal):+.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
