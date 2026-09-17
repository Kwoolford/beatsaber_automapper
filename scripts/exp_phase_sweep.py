#!/usr/bin/env python
"""Per song: how far is our grid phase from the one the music wants? (2026-09-16)

1f9a0 scores `onset_precision` 0.48 and a CONSTANT shift of about -110 ms, the same in every fifth
of the song, lifts it to 0.75. A constant optimum along the whole song is a phase error, not a
selection error (P1.0 had recorded the opposite). This reads every build in
`outputs/phase16_2026-09-16/` (`price_phase_songset16.sh`):

  * `offset` — the `_songTimeOffset` export wrote (the fitted phase). ⚠️BSMG: that field is
    "deprecated due to unstable behavior in recent versions of the game, unsupported from v4";
    human maps carry 0 and bake timing into beats. `mapjudge` IGNORES it.
  * precision ignoring it (what the judge scores) and applying it (what export intends);
  * the best global shift on top of the applied offset, ±200 ms, and the precision there;
  * the HUMAN map's own best shift, the independent check that the onsets are not simply biased.

Same matcher as the judge (`alignment.match_offsets`, 50 ms, one note per onset).

Run: python scripts/exp_phase_sweep.py
"""
from __future__ import annotations

import json
import pathlib
import zipfile

import numpy as np

from beatsaber_automapper.evaluation.alignment import match_offsets

REPO = pathlib.Path(__file__).resolve().parent.parent
OUT = REPO / "outputs" / "phase16_2026-09-16"
SHIFTS = np.round(np.arange(-0.20, 0.2001, 0.01), 3)


def load(zp):
    with zipfile.ZipFile(zp) as z:
        names = z.namelist()
        info = json.loads(z.read(next(x for x in names if x.split("/")[-1].lower() == "info.dat")))
        std = [x for x in names if x.lower().endswith("standard.dat")]
        n = next((x for x in std if x.split("/")[-1].lower().startswith("expert")
                  and "plus" not in x.lower()), std[0] if std else None)
        d = json.loads(z.read(n))
    ns = d.get("colorNotes") or d.get("_notes") or []
    bpm = float(info.get("_beatsPerMinute") or 0)
    off = float(info.get("_songTimeOffset") or 0)
    b = np.array(sorted({round(float(x.get("b", x.get("_time"))), 4) for x in ns}))
    return bpm, off, b * 60.0 / bpm


def prec(t, on):
    return match_offsets(list(t), on)[0] / len(t)


def best(t, on):
    sc = [(prec(t + s, on), s) for s in SHIFTS]
    return max(sc)


def main() -> int:
    print(f"{'song':<6}{'offset':>8}{'ignore':>8}{'apply':>8}{'best shift':>12}{'at best':>9}"
          f"{'human':>8}{'his best':>10}")
    rows = []
    for zp in sorted(OUT.glob("B__*.zip")):
        sid = zp.stem.split("__")[1]
        f = REPO / "outputs" / "onset_cache" / f"{sid}.npz"
        if not f.exists():
            continue
        on = np.load(f)["onsets"]
        bpm, off, t = load(zp)
        pi, pa = prec(t, on), prec(t + off, on)
        pb, sb = best(t + off, on)
        hz = REPO / "data" / "raw" / f"{sid}.zip"
        hp = hs = float("nan")
        if hz.exists():
            try:
                _hb, hoff, ht = load(hz)
                hp = prec(ht + hoff, on)
                hs = best(ht + hoff, on)[1]
            except Exception:  # noqa: BLE001
                pass
        rows.append(dict(sid=sid, bpm=bpm, offset=off, ignore=pi, apply=pa, shift=sb, best=pb,
                         human=hp, human_shift=hs))
        print(f"{sid:<6}{off * 1000:>+7.0f}ms{pi:>8.3f}{pa:>8.3f}{sb * 1000:>+10.0f}ms{pb:>9.3f}"
              f"{hp:>8.3f}{hs * 1000:>+8.0f}ms")
    (OUT / "phase_rows.json").write_text(json.dumps(rows, indent=1))
    s = np.array([r["shift"] for r in rows])
    g = np.array([r["best"] - r["apply"] for r in rows])
    hs = np.array([r["human_shift"] for r in rows])
    print(f"\n{len(rows)} songs. our best shift: median {np.median(s) * 1000:+.0f} ms, "
          f"|shift| >= 30 ms on {(abs(s) >= 0.03).sum()}; gain at best: median {np.median(g):+.3f}, "
          f">= 0.05 on {(g >= 0.05).sum()}")
    print(f"human best shift: median {np.nanmedian(hs) * 1000:+.0f} ms, |shift| >= 30 ms on "
          f"{(abs(hs[~np.isnan(hs)]) >= 0.03).sum()} of {(~np.isnan(hs)).sum()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
