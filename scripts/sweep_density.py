#!/usr/bin/env python
"""Can we hit a REQUESTED difficulty, and what does the density cost?

Kyle, 2026-08-21: *"The objective is to be able to map whatever difficulty we want.
Difficulty isn't always just NPS, it's how hard are the notes to get from the last
note as well."*

Measured before this: we built every song at ~4 nps (3.15-4.46) while the human maps
of the same songs range 1.88-7.51 — we flattened every song to one difficulty, and
`--nps 9` silently delivered 6.25 because `autobuild` only ever looked at drums plus
ONE carrier class (1 965 candidate events of the 4 813 the song actually has).

This measures the two things that matter together:
  DELIVERY  requested nps vs realised nps — does the dial do what it says?
  COST      does the extra density land on the music (`onset_precision`) and stay
            playable (`viol`, `ebpm_burst`), or is it filler?

★A dial that delivers density by placing notes off the music is not a difficulty
dial, it is a padding dial. Both columns have to be read together.
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


def onsets_for(sid):
    f = REPO / "outputs" / "onset_cache" / f"{sid}.npz"
    if not f.exists():
        return None
    z = np.load(f)
    return z[list(z.keys())[0]]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--songs", nargs="*", default=None)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--nps", nargs="+", type=float, default=[4.17, 6.5, 9.0])
    a = ap.parse_args()

    from beatsaber_automapper.evaluation import mapjudge as mj
    ref = mj.load_reference()

    sids = a.songs or [p.stem for p in
                       sorted((REPO / "data" / "eval_songset").glob("*.ogg"))][: a.n]
    tmp = pathlib.Path(tempfile.mkdtemp(prefix="dens_"))
    rows: dict[float, list] = {n: [] for n in a.nps}
    for sid in sids:
        audio = REPO / "data" / "eval_songset" / f"{sid}.ogg"
        on = onsets_for(sid)
        for want in a.nps:
            out = tmp / f"n{want}__{sid}.zip"
            subprocess.run(
                [sys.executable, str(AM / "autobuild.py"), str(audio), "--pulse",
                 "--lead-bias", "0.2", "--nps", str(want),
                 "--name", f"dn_{sid}_{str(want).replace('.', '')}", "--out", str(out)],
                capture_output=True, text=True, cwd=REPO)
            if not out.exists():
                continue
            try:
                rows[want].append(mj.judge_zip(out, onsets=on, reference=ref))
            except Exception:  # noqa: BLE001
                pass

    print(f"\nsongs {len(sids)}\n")
    print(f"{'asked':>7}{'got nps':>10}{'delivery':>10}{'peak':>8}{'onset_prec':>12}"
          f"{'viol':>7}{'PASS':>8}")
    print("-" * 62)
    for want in a.nps:
        rs = rows[want]
        if not rs:
            continue
        got = statistics.median(m.value for r in rs for m in r.metrics if m.name == "nps")
        peak = statistics.median(m.value for r in rs for m in r.metrics
                                 if m.name == "peak_nps")
        op = statistics.median(m.value for r in rs for m in r.metrics
                               if m.name == "onset_precision")
        nviol = sum(1 for r in rs if (r.viol or 0) > 0)
        npass = sum(1 for r in rs if r.verdict() == "PASS")
        print(f"{want:>7.2f}{got:>10.2f}{got/want:>10.0%}{peak:>8.2f}{op:>12.3f}"
              f"{nviol:>7}{npass:>5}/{len(rs):<2}")
    print("\n★delivery = realised / requested. ★onset_precision is the check that the "
          "extra notes are ON the music, not filler.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
