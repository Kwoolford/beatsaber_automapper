#!/usr/bin/env python
"""Where do our notes stop landing on the music? And did the PULSE FIX cost us?

The audio axis went in tonight and immediately reported `onset_precision` at the
**10.5th human percentile** over 23 songs: our notes sit on a detected onset less
often than ~90 % of human maps. On the agent path that is surprising, because we
choose note times FROM detected onsets -- so something between "an onset" and "a
note" is moving them.

Three suspects, two of which this session INTRODUCED:

  GRID    `mapctl` snaps every onset to the 1/4-beat build grid. At 160 bpm a slot
          is 93.75 ms, so a snap can displace a note by up to ~47 ms -- against a
          50 ms matching tolerance. This has always been there.
  PULSE   `pulse.py` moves notes onto a per-phrase lattice AND fills quiet lattice
          points to hold the run. A filled point has no source event by definition.
  LEAD    `--lead-bias` changes which HAND plays a note, never when, so it cannot
          move this at all -- included as a NULL ARM. If it moves, the measurement
          is wrong, not the map.

★**The null arm is the point.** An arm that cannot affect the metric and does not is
what makes the arms that do move interpretable.

Every arm is scored against the SAME cached onsets, and the song's human map is
scored against them too -- both sides of the comparison on one footing, which is the
rule this axis exists to enforce.

Usage:
    python scripts/diag_onset_precision.py            # all 23 songset songs
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

ARMS = {
    "NOPULSE": [],
    "PULSE": ["--pulse"],
    "PULSE+LEAD": ["--pulse", "--lead-bias", "0.3"],
}
KEYS = ("onset_precision", "offset_mad_ms", "onset_lag_ms", "onset_recall")


def onsets_for(sid: str):
    f = REPO / "outputs" / "onset_cache" / f"{sid}.npz"
    if not f.exists():
        return None
    z = np.load(f)
    return z[list(z.keys())[0]]


def score(zp: pathlib.Path, on) -> dict | None:
    from audit_eval_suite import _load_generated, _load_human
    from beatsaber_automapper.evaluation import alignment, mapjudge as mj
    notes = bpm = None
    for loader in (_load_human, _load_generated):
        try:
            got = loader(zp)
        except Exception:  # noqa: BLE001
            continue
        if got and got[0]:
            notes, bpm = got
            break
    if not notes or on is None:
        return None
    m = dict(alignment.alignment_metrics(mj._BM(notes), bpm=bpm, onsets=on).metrics)
    m["n_notes"] = float(len(notes))
    return m


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--songs", nargs="*", default=None)
    a = ap.parse_args()

    sids = a.songs or [p.stem for p in
                       sorted((REPO / "data" / "eval_songset").glob("*.ogg"))]
    tmp = pathlib.Path(tempfile.mkdtemp(prefix="onsetprec_"))
    rows: dict[str, list[dict]] = {k: [] for k in ARMS}
    human: list[dict] = []
    for sid in sids:
        audio = REPO / "data" / "eval_songset" / f"{sid}.ogg"
        on = onsets_for(sid)
        if on is None:
            continue
        for arm, extra in ARMS.items():
            out = tmp / f"{arm.replace('+', '')}__{sid}.zip"
            subprocess.run(
                [sys.executable, str(AM / "autobuild.py"), str(audio),
                 "--name", f"op_{sid}_{arm.replace('+', '')}", "--no-idiomize",
                 "--out", str(out), *extra],
                capture_output=True, text=True, cwd=REPO)
            if out.exists():
                m = score(out, on)
                if m:
                    rows[arm].append(m)
        hz = REPO / "data" / "raw" / f"{sid}.zip"
        if hz.exists():
            m = score(hz, on)
            if m:
                human.append(m)

    print(f"\n{'arm':<13}" + "".join(f"{k[:14]:>16}" for k in KEYS) + f"{'n':>8}")
    print("-" * 90)
    for arm in (*ARMS, "HUMAN"):
        rs = human if arm == "HUMAN" else rows[arm]
        if not rs:
            continue
        cells = [statistics.median([r[k] for r in rs if r.get(k) == r.get(k)])
                 if any(r.get(k) == r.get(k) for r in rs) else float("nan")
                 for k in KEYS]
        n = statistics.median([r["n_notes"] for r in rs])
        print(f"{arm:<13}" + "".join(f"{c:>16.3f}" for c in cells) + f"{n:>8.0f}")

    p = {k: (statistics.median([r["onset_precision"] for r in rows[k]])
             if rows[k] else float("nan")) for k in ARMS}
    hp = statistics.median([r["onset_precision"] for r in human]) if human else float("nan")
    print(f"\nsongs: {len(rows['PULSE'])}   human onset_precision {hp:.3f}")
    if p["NOPULSE"] == p["NOPULSE"] and p["PULSE"] == p["PULSE"]:
        d = p["PULSE"] - p["NOPULSE"]
        print(f"PULSE cost on onset_precision: {d:+.3f} "
              f"({p['NOPULSE']:.3f} -> {p['PULSE']:.3f})")
        if abs(d) < 0.01:
            print("⇒ THE PULSE FIX IS NOT WHAT MOVED IT. The shortfall predates "
                  "tonight and the grid snap is the remaining suspect.")
        elif d < 0:
            print("⇒ ★THE PULSE FIX COST ONSET PRECISION. P0.5's win was measured "
                  "before the audio axis existed, so this cost was invisible at the "
                  "time. Re-price it: the lattice fill is the mechanism.")
        else:
            print("⇒ the pulse fix IMPROVED onset precision (holding a lattice keeps "
                  "notes nearer real onsets than the raw union did).")
    if p["PULSE"] == p["PULSE"] and p["PULSE+LEAD"] == p["PULSE+LEAD"]:
        dl = abs(p["PULSE+LEAD"] - p["PULSE"])
        print(f"NULL ARM check (lead-bias cannot move note times): |Δ| {dl:.4f} "
              f"{'✅ as expected' if dl < 0.005 else '🔴 MOVED — the measurement is wrong'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
