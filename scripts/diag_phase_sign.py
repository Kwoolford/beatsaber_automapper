#!/usr/bin/env python
"""P0.4: is the grid-phase gain a PHASE CORRECTION or a RESELECTION LOTTERY?

**The standing state.** Shifting the bar grid **+0.05-0.10 beats** improves
`onset_precision` 0.866 -> 0.916 and agreement with the human's own note times
0.645 -> 0.686, on 16/23 songs, inverted-U on both references. But it is **NOT a
realignment**: only **45.9 %** of note beat positions survive the shift and counts
move +-14 %, so it builds a *different map* rather than moving the same one. TODO
records it as *"a reproducible grid-dependent selection effect with no known
mechanism"*.

★**The test comes from a result found earlier the same day.** `--snap-onsets` turned
out to RESELECT rather than realign whenever its window is a large fraction of a
1/4-beat slot (`corr(bpm, survival) = -0.868`). A phase shift is the same kind of
perturbation -- it changes which event lands in which slot -- and it improves the same
metric. **Two different perturbations of the event->slot assignment both improving
alignment suggests the DEFAULT assignment is simply a poor selection, and that almost
any re-roll finds a better one.**

**A REAL PHASE ERROR HAS A SIGN. A LOTTERY DOES NOT.** That was the intended test:

    if only +delta helps  ->  the grid is genuinely misplaced; fix the estimator.
    if -delta helps too   ->  zero is not special, the gain is RESELECTION, and the
                              thing to fix is note SELECTION -- which is exactly what
                              C1 and P1.0 already concluded by other routes.

🔴🔴**THE SIGN TEST AS ORIGINALLY WRITTEN IS INVALID, AND THIS SCRIPT NOW MEASURES WHY.**
Note times do NOT move continuously with the phase. Measured on 1f767 (160 bpm, one
1/4-beat slot = 93.8 ms), median displacement of the shifted note set from the
unshifted one:

    -0.20 beats -> 93.8 ms      (a FULL SLOT)
    -0.05 beats -> 93.8 ms      (a FULL SLOT)
    -0.01 beats ->  0.0 ms
    +0.05 beats ->  0.0 ms
    +0.20 beats ->  0.0 ms

⇒**the negative arm is not "the same perturbation with the opposite sign"** -- it
displaces the median note by a whole slot, which is far outside the 50 ms tolerance
and guarantees a collapse. Reading that collapse as evidence of a signed phase error
would be reading a quantisation artifact. ★**`displacement` is therefore reported
alongside every arm: an arm whose median displacement is a full slot is not a
perturbation of the same map and its scores are not comparable.**

★★**AND THE POSITIVE ARMS SHARPEN P0.4.** At +0.05 and +0.20 the surviving notes sit
at median displacement **0.0** -- exactly on the unshifted absolute grid -- while the
note COUNT changes (986 -> 1039). So the phase shift does not move notes at all; it
changes **which events get selected into which slot**. That confirms and sharpens
TODO's retraction: the grid phase is a **SELECTION knob, not an alignment knob**, and
whatever P0.4 is buying, it is buying by re-picking notes.

⚠️Anything that changes the grid changes the map, so `nps` and note count are reported
too: a "gain" bought by emitting fewer, easier notes is not a gain.
⚠️Scored against BOTH references -- the onsets the axis uses AND the human mapper's own
note times, which no grid search can flatter.

Usage:
    python scripts/diag_phase_sign.py --json outputs/phase_sign.json
"""
from __future__ import annotations

import argparse
import json
import pathlib
import statistics as st
import subprocess
import sys
import tempfile
import zipfile

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[1]
AM = REPO / "agent_mapper"
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO))

from diag_snap_independent import human_note_times, nearest_dist  # noqa: E402

# ★Symmetric about zero ON PURPOSE. The published effect is +0.05..+0.10; the
# negatives are the control that decides whether zero is special at all.
SHIFTS = (-0.10, -0.05, 0.0, 0.05, 0.10)
TOL_S = 0.050


def onsets_for(sid):
    f = REPO / "outputs" / "onset_cache" / f"{sid}.npz"
    if not f.exists():
        return None
    z = np.load(f)
    return np.sort(np.asarray(z[list(z.keys())[0]], dtype=float))


def note_times(zp: pathlib.Path) -> tuple[np.ndarray, int] | None:
    """Note times in seconds, and the note count."""
    with zipfile.ZipFile(zp) as zf:
        names = zf.namelist()
        info = next((n for n in names
                     if n.split("/")[-1].lower() == "info.dat"), None)
        diff = next((n for n in names
                     if n.split("/")[-1].lower().endswith("standard.dat")
                     and "bpminfo" not in n.lower()), None)
        if info is None or diff is None:
            return None
        meta = json.loads(zf.read(info))
        dat = json.loads(zf.read(diff))
    bpm = None
    for k, v in meta.items():
        if "beatsperminute" in k.lower():
            bpm = float(v)
            break
    if not bpm:
        return None
    notes = dat.get("colorNotes") or []
    if not notes:
        return None
    t = np.array([float(n.get("b", 0.0)) * 60.0 / bpm for n in notes])
    return np.sort(t), len(notes), bpm


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--songs", nargs="*", default=None)
    ap.add_argument("--json", default="")
    a = ap.parse_args()

    sids = a.songs or [p.stem for p in
                       sorted((REPO / "data" / "eval_songset").glob("*.ogg"))]
    tmp = pathlib.Path(tempfile.mkdtemp(prefix="phasesign_"))
    per_shift: dict[float, list] = {s: [] for s in SHIFTS}

    for sid in sids:
        audio = REPO / "data" / "eval_songset" / f"{sid}.ogg"
        ons = onsets_for(sid)
        h = human_note_times(sid)
        if ons is None or h is None:
            continue
        hum = h[0]
        base_t = None
        slot_ms = None
        # 🔴**Run the 0.0 baseline FIRST.** `SHIFTS` is ordered negative-first, and the
        # displacement of every other arm is measured against this one -- iterating in
        # order left the negative arms with no baseline and silently recorded `nan`,
        # which made the full-slot guard read 0/23 and let the invalid verdict through.
        for sh in sorted(SHIFTS, key=lambda s: (s != 0.0, s)):
            out = tmp / f"PH{sh:+.2f}__{sid}.zip"
            cmd = [sys.executable, str(AM / "autobuild.py"), str(audio), "--pulse",
                   "--lead-bias", "0.2", "--name", f"ph_{sid}_{sh:+.2f}",
                   "--out", str(out)]
            if sh:
                cmd += ["--phase-shift", str(sh)]
            subprocess.run(cmd, capture_output=True, text=True, cwd=REPO)
            got = note_times(out) if out.exists() else None
            if not got:
                continue
            t, n, bpm = got
            if sh == 0.0:
                base_t = t
                # ⚠️One 1/4-beat slot is `15000/bpm` ms. An earlier version estimated it
                # from the median inter-note GAP, which is several slots wide, so the
                # full-slot guard could never fire.
                slot_ms = 15000.0 / bpm
            d_ons = nearest_dist(t, ons)
            d_hum = nearest_dist(t, hum)
            # ★How far the shifted map's notes sit from the UNSHIFTED map's notes. An
            # arm at a full slot is a displaced map, not a perturbed one.
            disp = (float(np.median(nearest_dist(t, base_t)) * 1000)
                    if base_t is not None and len(base_t) else float("nan"))
            per_shift[sh].append(dict(
                song=sid, n=n, disp_ms=disp, slot_ms=slot_ms,
                onset_prec=float((d_ons <= TOL_S).mean()),
                human_agree=float((d_hum <= TOL_S).mean()),
            ))
        print(f"  {sid} done", flush=True)

    print(f"\n{'shift':>8}{'onset_prec':>13}{'human_agree':>14}"
          f"{'notes':>9}{'displaced':>11}{'songs':>7}")
    print("-" * 64)
    rows = []
    for sh in SHIFTS:
        rs = per_shift[sh]
        if not rs:
            continue
        op = st.mean([r["onset_prec"] for r in rs])
        ha = st.mean([r["human_agree"] for r in rs])
        nn = st.mean([r["n"] for r in rs])
        # share of songs whose median note sits a FULL SLOT from the unshifted map
        full = sum(1 for r in rs
                   if r["slot_ms"] and r["disp_ms"] >= 0.5 * r["slot_ms"])
        star = "  <-- today" if sh == 0.0 else ""
        print(f"{sh:>+8.2f}{op:>13.4f}{ha:>14.4f}{nn:>9.0f}"
              f"{full:>7}/{len(rs):<3}{len(rs):>7}{star}")
        rows.append(dict(shift=sh, onset_prec=op, human_agree=ha, notes=nn,
                         n_songs=len(rs), songs_displaced_a_slot=full,
                         disp_ms=st.mean([r["disp_ms"] for r in rs]),
                         per_song={r["song"]: r["onset_prec"] for r in rs}))

    base = next((r for r in rows if r["shift"] == 0.0), None)
    if not base:
        return 1

    print("\nSIGN TEST — is zero special?")
    for r in rows:
        if r["shift"] == 0.0:
            continue
        d_op = r["onset_prec"] - base["onset_prec"]
        d_ha = r["human_agree"] - base["human_agree"]
        better = sum(1 for s, v in r["per_song"].items()
                     if v > base["per_song"].get(s, 0))
        print(f"  {r['shift']:+.2f}:  onset_prec {d_op:+.4f}   "
              f"human_agree {d_ha:+.4f}   better on {better}/{r['n_songs']} songs")

    neg = [r for r in rows if r["shift"] < 0]
    pos = [r for r in rows if r["shift"] > 0]
    if neg and pos:
        dn = st.mean([r["onset_prec"] for r in neg]) - base["onset_prec"]
        dp = st.mean([r["onset_prec"] for r in pos]) - base["onset_prec"]
        print(f"\n  mean Δonset_prec:  negative shifts {dn:+.4f}   "
              f"positive shifts {dp:+.4f}")
        print("\nVERDICT")
        # 🔴Guard the trap this script exists to document: an arm that displaces the
        # median note by a full slot is a DIFFERENT map, and comparing its score to
        # the unshifted one measures quantisation, not phase.
        bad = [r for r in rows
               if r["shift"] != 0.0 and r["songs_displaced_a_slot"] > 0.5 * r["n_songs"]]
        if bad:
            arms = ", ".join(f"{r['shift']:+.2f}" for r in bad)
            print(f"  🔴THE SIGN TEST CANNOT BE READ: arms [{arms}] displace the median")
            print("     note by a FULL SLOT on most songs. That is not the same map")
            print("     perturbed, it is a map moved off the grid, so their collapse is")
            print("     a quantisation artifact and NOT evidence of a signed phase error.")
            print("  ★What IS readable: arms with displacement 0.0 change the note")
            print("     COUNT while leaving surviving notes exactly in place ⇒ the grid")
            print("     phase is a SELECTION knob, not an alignment knob.")
        elif dn > 0 and dp > 0:
            print("  🔴BOTH DIRECTIONS HELP ⇒ zero is NOT a special phase. The gain is")
            print("     RESELECTION, not a phase correction. Do not build a phase")
            print("     estimator fix; the target is note SELECTION (C1, P1.0).")
        elif dp > 0 >= dn:
            print("  ✅ONLY THE POSITIVE DIRECTION HELPS ⇒ a real, signed grid offset.")
            print("     Wiring a phase estimate through is worth building.")
        else:
            print("  ⚠️Neither direction reproduces the published gain on this build —")
            print("     check whether the effect survives the current defaults at all.")

    if a.json:
        p = pathlib.Path(a.json)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(rows, indent=2))
        print(f"\nwrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
