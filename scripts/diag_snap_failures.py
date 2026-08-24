#!/usr/bin/env python
"""WHICH songs does `--snap-onsets` cost a PASS, and on WHAT?

The 23-song sweep (2026-08-24) says the snap raises `onset_precision` 0.866 -> 0.891
and, at current defaults, costs NO density (`nps` 4.412 -> 4.464 -- the recorded
3.43 -> 3.29 predates the two-stream carrier fix). But PASS falls **23/23 -> 21/23**,
and that is what blocks making it the default.

★**A PASS count is not a ranking statistic** -- an identical config has scored 4, 4, 2
in this repo -- and **a FAIL can mean NOT TYPICAL rather than bad**. So the question is
not "how many failed" but **which metric moved, in which direction, and is that metric
one we actually want centred**.

Prints the per-song verdict for both arms and, for every song whose verdict CHANGES,
the metrics that are out of range in each arm.

Usage:
    python scripts/diag_snap_failures.py --json outputs/snap_failures.json
"""
from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys
import tempfile

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[1]
AM = REPO / "agent_mapper"
sys.path.insert(0, str(REPO / "src"))


def onsets_for(sid):
    f = REPO / "outputs" / "onset_cache" / f"{sid}.npz"
    if not f.exists():
        return None
    z = np.load(f)
    return z[list(z.keys())[0]]


def offenders(r):
    """Metrics the judge counts against this map, with their human percentile."""
    out = []
    for m in r.metrics:
        pct = getattr(m, "pct", None)
        if pct is None:
            continue
        # Out at either tail: the judge asks "is this typical?", so both ends count.
        if pct <= 0.05 or pct >= 0.95:
            out.append((m.name, float(m.value), float(pct)))
    return sorted(out, key=lambda x: min(x[2], 1 - x[2]))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--songs", nargs="*", default=None)
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    from beatsaber_automapper.evaluation import mapjudge as mj
    ref = mj.load_reference()

    arms = {"BASE": [], "SNAP": ["--snap-onsets"]}
    sids = a.songs or [p.stem for p in
                       sorted((REPO / "data" / "eval_songset").glob("*.ogg"))]
    tmp = pathlib.Path(tempfile.mkdtemp(prefix="snapfail_"))
    rows = []
    print(f"{'song':8s}{'BASE':>10s}{'SNAP':>10s}{'onsetP b->a':>18s}{'nps b->a':>16s}")
    print("-" * 64)
    for sid in sids:
        audio = REPO / "data" / "eval_songset" / f"{sid}.ogg"
        on = onsets_for(sid)
        got = {}
        for arm, extra in arms.items():
            out = tmp / f"{arm}__{sid}.zip"
            subprocess.run(
                [sys.executable, str(AM / "autobuild.py"), str(audio), "--pulse",
                 "--lead-bias", "0.2", "--name", f"sf_{sid}_{arm}",
                 "--out", str(out), *extra],
                capture_output=True, text=True, cwd=REPO)
            if not out.exists():
                continue
            try:
                got[arm] = mj.judge_zip(out, onsets=on, reference=ref)
            except Exception:  # noqa: BLE001
                pass
        if len(got) != 2:
            print(f"{sid:8s}  incomplete")
            continue

        def pick(r, name):
            for m in r.metrics:
                if m.name == name:
                    return float(m.value)
            return float("nan")

        vb, va = got["BASE"].verdict(), got["SNAP"].verdict()
        ob, oa = pick(got["BASE"], "onset_precision"), pick(got["SNAP"], "onset_precision")
        nb, na = pick(got["BASE"], "nps"), pick(got["SNAP"], "nps")
        flag = "  <-- CHANGED" if vb != va else ""
        print(f"{sid:8s}{vb:>10s}{va:>10s}"
              f"{ob:8.3f} ->{oa:7.3f}{nb:7.2f} ->{na:6.2f}{flag}")
        rows.append(dict(song=sid, base=vb, snap=va,
                         onset_base=ob, onset_snap=oa, nps_base=nb, nps_snap=na,
                         offenders_base=offenders(got["BASE"]),
                         offenders_snap=offenders(got["SNAP"])))

    changed = [r for r in rows if r["base"] != r["snap"]]
    print("\n" + "=" * 64)
    print(f"verdict changed on {len(changed)} of {len(rows)} songs")
    for r in changed:
        print(f"\n{r['song']}: {r['base']} -> {r['snap']}   "
              f"onset_precision {r['onset_base']:.3f} -> {r['onset_snap']:.3f}")
        print(f"   BASE out-of-range: {r['offenders_base'] or 'none'}")
        print(f"   SNAP out-of-range: {r['offenders_snap'] or 'none'}")

    print("\nHOW TO READ THIS")
    print("  If the NEW offender is `nps`/`peak_nps`, the snap changed the note BUDGET")
    print("     by collapsing events onto a shared onset -- a density change wearing an")
    print("     alignment change's clothes, and a real reason not to default it.")
    print("  If the NEW offender is a GEOMETRY axis, the snap moved times and the")
    print("     sampler re-drew around them; that is a reselection, not a defect, and")
    print("     is the same class of effect as P0.4's grid-phase reselection.")
    print("  If a song FAILS while `onset_precision` ROSE, the FAIL is 'not typical',")
    print("     not 'worse on the thing we changed'.")

    if a.json:
        p = pathlib.Path(a.json)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(rows, indent=2))
        print(f"\nwrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
