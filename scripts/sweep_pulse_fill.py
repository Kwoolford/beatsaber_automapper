#!/usr/bin/env python
"""Trade the lattice fill against staying on the music, and price both sides.

`diag_onset_precision.py` found that the pulse pass costs `onset_precision`
0.868 -> 0.829 against a human 0.919 -- it nearly doubled our distance from the
human on an axis that **did not exist when the pulse fix was priced**. The
mechanism is `MAX_EMPTY_RUN`: a lattice point held across a quiet gap has no source
event by definition, so it can land where there is no audio onset.

This prices both sides of that trade at once, because either metric alone is
misleading: fill 0 buys precision and may give the pulse back, fill 2 does the
reverse. **Both are two-sided** -- `pulse_stability` past the human is a metronome,
and the human's `onset_precision` is 0.919, not 1.0.

Usage:
    python scripts/sweep_pulse_fill.py --fill 0 1 2
"""
from __future__ import annotations

import argparse
import pathlib
import statistics
import subprocess
import sys
import tempfile

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[1]
AM = REPO / "agent_mapper"
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

WATCH = ("onset_precision", "pulse_stability", "dominant_share", "ioi_switch_rate")


def onsets_for(sid: str):
    f = REPO / "outputs" / "onset_cache" / f"{sid}.npz"
    if not f.exists():
        return None
    z = np.load(f)
    return z[list(z.keys())[0]]


def raw(zp: pathlib.Path, on):
    from audit_eval_suite import _load_generated, _load_human
    from beatsaber_automapper.evaluation import alignment, mapjudge as mj, rhythm
    notes = bpm = None
    for loader in (_load_human, _load_generated):
        try:
            got = loader(zp)
        except Exception:  # noqa: BLE001
            continue
        if got and got[0]:
            notes, bpm = got
            break
    if not notes:
        return None
    bm = mj._BM(notes)
    m = dict(rhythm.rhythm_metrics(bm).metrics)
    if on is not None:
        try:
            m.update(alignment.alignment_metrics(bm, bpm=bpm, onsets=on).metrics)
        except Exception:  # noqa: BLE001
            pass
    m["n_notes"] = float(len(notes))
    return m


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fill", nargs="+", type=int, default=[1])
    ap.add_argument("--sync", nargs="+", type=float, default=[0.5, 0.4, 0.3, 0.2])
    ap.add_argument("--songs", nargs="*", default=None)
    a = ap.parse_args()

    from beatsaber_automapper.evaluation import mapjudge as mj
    ref = mj.load_reference()
    dists = ref["distributions"]

    sids = a.songs or [p.stem for p in
                       sorted((REPO / "data" / "eval_songset").glob("*.ogg"))]
    tmp = pathlib.Path(tempfile.mkdtemp(prefix="fill_"))
    combos = [(f, s) for f in a.fill for s in a.sync]
    rows: dict[tuple, list[dict]] = {c: [] for c in combos}
    human: list[dict] = []
    for sid in sids:
        audio = REPO / "data" / "eval_songset" / f"{sid}.ogg"
        on = onsets_for(sid)
        for (f, sy) in combos:
            tag = f"{f}_{str(sy).replace('.', '')}"
            out = tmp / f"fill{tag}__{sid}.zip"
            subprocess.run(
                [sys.executable, str(AM / "autobuild.py"), str(audio), "--pulse",
                 "--no-idiomize", "--lead-bias", "0.3", "--pulse-fill", str(f),
                 "--pulse-sync", str(sy),
                 "--name", f"pf_{sid}_{tag}", "--out", str(out)],
                capture_output=True, text=True, cwd=REPO)
            if out.exists():
                m = raw(out, on)
                if m:
                    rows[(f, sy)].append(m)
        hz = REPO / "data" / "raw" / f"{sid}.zip"
        if hz.exists():
            m = raw(hz, on)
            if m:
                human.append(m)

    def pctstr(name, v):
        d = dists.get(name)
        return f"({100 * mj.percentile_of(v, d):.0f}%)" if d else ""

    print(f"\n{'fill/sync':<11}" + "".join(f"{k[:14]:>22}" for k in WATCH) + f"{'n':>8}")
    print("-" * 108)
    for c in combos:
        rs = rows[c]
        if not rs:
            continue
        cells = []
        for k in WATCH:
            v = statistics.median([r[k] for r in rs if r.get(k) == r.get(k)])
            cells.append(f"{v:8.3f} {pctstr(k, v):>7}")
        n = statistics.median([r["n_notes"] for r in rs])
        print(f"{str(c[0]) + '/' + str(c[1]):<11}"
              + "".join(f"{x:>22}" for x in cells) + f"{n:>8.0f}")
    if human:
        cells = []
        for k in WATCH:
            v = statistics.median([r[k] for r in human if r.get(k) == r.get(k)])
            cells.append(f"{v:8.3f} {'':>7}")
        n = statistics.median([r["n_notes"] for r in human])
        print(f"{'HUMAN':<11}" + "".join(f"{x:>22}" for x in cells) + f"{n:>8.0f}")
    print("\n★Both axes are two-sided: pulse_stability past the human is a metronome, "
          "and the human's onset_precision is 0.919, not 1.0.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
