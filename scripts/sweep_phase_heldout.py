#!/usr/bin/env python
"""Does the phase correction survive on songs that did NOT set its constant?

Established so far: our fitted bar-grid phase sits **0.053 beats early** of the human
mapper's grid (18/18 songs), shifting **+0.05** peaks `onset_precision` (0.857 ->
0.894) and human agreement (0.610 -> 0.678) while **+0.10 overshoots and hurts**, and
the estimator provably tracks the MUSIC rather than the corpus `t=0` convention
(padded-audio control, 6/6).

⚠️**But 0.053 was measured on the same 18 songs it would be applied to.** A constant
fitted and evaluated on one cohort is not evidence that it transfers -- it is the
winner's-curse shape this repo has already been bitten by (`travel`'s "defect" came
from an experimental arm; the `h_dist` saturation came from fitting the reference).

**This splits the cohort.** The bias is fitted on half the songs, then applied,
untouched, to the other half -- and both directions are run, so neither half is
privileged:

    fold A -> constant -> evaluated on fold B
    fold B -> constant -> evaluated on fold A

Three arms per held-out song:
    NONE    no shift (today's default)
    HELDOUT the constant fitted on the OTHER fold        <- the honest number
    ORACLE  that song's own -phi                          <- the ceiling, not shippable

★**ORACLE is the ceiling and is deliberately not a candidate**: it uses the song's own
fitted phase, which is exactly what the correction is trying to repair, so it says how
much is available, not how much is earned.

⚠️Note times include **`_songTimeOffset`** -- it carries the grid phase, and omitting it
made an earlier probe report that shifting the phase moved nothing at all.

Usage:
    python scripts/sweep_phase_heldout.py --json outputs/phase_heldout.json
"""
from __future__ import annotations

import argparse
import json
import pathlib
import statistics as st
import subprocess
import sys
import tempfile

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[1]
AM = REPO / "agent_mapper"
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO))

from diag_grid_vs_human import human_grid, wrapped_phase  # noqa: E402
from diag_phase_sign import note_times, onsets_for  # noqa: E402
from diag_snap_independent import human_note_times, nearest_dist  # noqa: E402

TOL_S = 0.050

# Session scratchpad rather than /tmp, per the project convention.
SCRATCH = pathlib.Path(tempfile.mkdtemp(prefix="phaseheldout_"))


def cohort():
    """Songs usable for this test: bpm-matched, human grid clean, event cache present."""
    out = []
    for sid in sorted(p.stem for p in (REPO / "data" / "eval_songset").glob("*.ogg")):
        hg = human_grid(sid)
        ec = REPO / "outputs" / "event_cache" / f"{sid}.6s.json"
        if hg is None or not ec.exists():
            continue
        hbpm, hnotes = hg
        d = json.loads(ec.read_text())
        obpm, ophase = float(d["bpm"]), float(d["phase"])
        if abs(hbpm - obpm) >= 0.5:
            continue                      # half-tempo: a different ruler
        if float(np.median(np.abs(wrapped_phase(hnotes, hbpm)))) >= 0.01:
            continue                      # human notes not on their own grid
        our_beats = float(wrapped_phase(np.array([ophase % (60.0 / obpm)]), obpm)[0])
        out.append(dict(song=sid, bpm=obpm, our_phase_beats=our_beats,
                        correction=-our_beats))
    return out


def build_and_score(sid: str, shift: float, tmp: pathlib.Path):
    audio = REPO / "data" / "eval_songset" / f"{sid}.ogg"
    out = tmp / f"S{shift:+.3f}__{sid}.zip"
    cmd = [sys.executable, str(AM / "autobuild.py"), str(audio), "--pulse",
           "--lead-bias", "0.2", "--name", f"ho_{sid}", "--out", str(out)]
    if abs(shift) > 1e-9:
        cmd += ["--phase-shift", f"{shift:.4f}"]
    subprocess.run(cmd, capture_output=True, text=True, cwd=REPO)
    got = note_times(out) if out.exists() else None
    if not got:
        return None
    t, n, _ = got
    ons = onsets_for(sid)
    h = human_note_times(sid)
    if ons is None or h is None:
        return None
    return dict(
        onset_prec=float((nearest_dist(t, ons) <= TOL_S).mean()),
        human_agree=float((nearest_dist(t, h[0]) <= TOL_S).mean()),
        notes=n,
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="")
    a = ap.parse_args()

    songs = cohort()
    if len(songs) < 6:
        print(f"cohort too small ({len(songs)})")
        return 1
    # Deterministic interleaved split -- alternating by sorted id, so neither fold is
    # a contiguous block of one tempo range.
    fold = {s["song"]: i % 2 for i, s in enumerate(songs)}
    consts = {}
    for f in (0, 1):
        vals = [s["correction"] for s in songs if fold[s["song"]] != f]
        consts[f] = st.median(vals)   # constant for fold f comes from the OTHER fold
    print(f"cohort n={len(songs)}  (fold0 n={sum(1 for v in fold.values() if v == 0)}, "
          f"fold1 n={sum(1 for v in fold.values() if v == 1)})")
    print(f"constant applied to fold0 (fitted on fold1) = {consts[0]:+.4f} beats")
    print(f"constant applied to fold1 (fitted on fold0) = {consts[1]:+.4f} beats\n")

    rows = []
    print(f"{'song':8s}{'fold':>5s}{'oracle':>9s}"
          f"{'  onset: none  held  orac':>28s}"
          f"{'  human: none  held  orac':>28s}")
    print("-" * 78)
    for s in songs:
        sid, f = s["song"], fold[s["song"]]
        arms = {"none": 0.0, "heldout": consts[f], "oracle": s["correction"]}
        got = {k: build_and_score(sid, v, SCRATCH)
               for k, v in arms.items()}
        if any(v is None for v in got.values()):
            print(f"{sid:8s}  incomplete")
            continue
        # ⚠️`s` already carries `song`; splatting it alongside `song=sid` is a
        # duplicate-keyword TypeError.
        rows.append(dict(fold=f, **s,
                         **{f"{k}_{m}": got[k][m] for k in arms
                            for m in ("onset_prec", "human_agree", "notes")}))
        print(f"{sid:8s}{f:>5d}{s['correction']:+9.3f}"
              f"{got['none']['onset_prec']:>10.3f}{got['heldout']['onset_prec']:>7.3f}"
              f"{got['oracle']['onset_prec']:>7.3f}"
              f"{got['none']['human_agree']:>12.3f}"
              f"{got['heldout']['human_agree']:>7.3f}"
              f"{got['oracle']['human_agree']:>7.3f}", flush=True)

    if not rows:
        return 1
    print("-" * 78)
    for m in ("onset_prec", "human_agree"):
        base = st.mean([r[f"none_{m}"] for r in rows])
        held = st.mean([r[f"heldout_{m}"] for r in rows])
        orac = st.mean([r[f"oracle_{m}"] for r in rows])
        bw = sum(1 for r in rows if r[f"heldout_{m}"] > r[f"none_{m}"])
        print(f"{m:14s} none {base:.4f}   held-out {held:.4f} ({held - base:+.4f}, "
              f"better on {bw}/{len(rows)})   oracle {orac:.4f} ({orac - base:+.4f})")
        if orac > base:
            print(f"{'':14s} held-out captures "
                  f"{100 * (held - base) / (orac - base):.0f}% of the oracle gain")

    print("\nVERDICT")
    dh = (st.mean([r["heldout_onset_prec"] for r in rows])
          - st.mean([r["none_onset_prec"] for r in rows]))
    da = (st.mean([r["heldout_human_agree"] for r in rows])
          - st.mean([r["none_human_agree"] for r in rows]))
    if dh > 0.01 and da > 0.01:
        print("  ✅THE CORRECTION TRANSFERS. A constant fitted on songs it was not")
        print("     evaluated on still improves both the axis and the INDEPENDENT")
        print("     human reference ⇒ this is a calibration, not a per-cohort fit,")
        print("     and it is ready to propose as a real change.")
    elif dh > 0 or da > 0:
        print("  🟡PARTIAL: it moves the right way but under the 0.01 bar on one")
        print("     reference. Report it as such; do not promote it to a default.")
    else:
        print("  🔴IT DOES NOT TRANSFER: the gain lived in fitting the same songs it")
        print("     was scored on. The per-song oracle column says how much is really")
        print("     there; the constant is not the way to get it.")
    print("  ⚠️ORACLE is a CEILING, not a candidate — it uses each song's own fitted")
    print("     phase, which is the very quantity the correction is meant to repair.")

    if a.json:
        p = pathlib.Path(a.json)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(dict(rows=rows, constants=consts), indent=2))
        print(f"\nwrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
