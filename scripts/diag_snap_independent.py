#!/usr/bin/env python
"""Is `--snap-onsets` a real realignment, or is it Goodharting its own axis?

**The problem.** P0.7 records `onset_precision` 0.856 -> 0.890 from snapping placed
events onto `outputs/onset_cache`. But `onset_precision` IS the share of our notes
within 50 ms of that same cache. Moving notes onto X and then scoring
distance-to-X is close to circular by construction: the metric cannot tell a
genuine realignment from the act of snapping.

⚠️It is not FULLY circular -- the human map scores 0.919 on the same onsets without
being snapped, so the ceiling is real. But our +0.034 is bought by construction and
has never been checked against anything the snap does not define.

**The independent reference.** The HUMAN MAPPER'S OWN NOTE TIMES. The snap knows
nothing about them, so they can adjudicate: if moving an event onto the judge's
onset also moves it toward where a human put a note, the snap is finding the music.
If the moves scatter randomly with respect to the human, the snap is only satisfying
the axis and generalising it to every song would ship a metric artifact.

**The test is PAIRED and SIGN-BASED**, over only the events the snap actually MOVED
(an unmoved event contributes nothing and would dilute the effect toward zero):

    d_before = |t          - nearest human note time|
    d_after  = |snap(t)    - nearest human note time|

Then `wins` = moved events that got closer, `losses` = further. A snap that finds
the music wins well above half; a snap that only feeds its own axis sits at ~50 %
with a median signed delta inside noise.

★**Why the event set and not the built map**: the snap is a pure function on times,
so applying it to the cached events isolates the operation from note selection,
grid quantisation and the sampler. A confound there would be a confound about
those, not about the snap.

⚠️**Reads the cohort, not one song** -- the single-song probe trap has now caught two
hypotheses in this repo, and 1f333 is half-tempo.

Usage:
    python scripts/diag_snap_independent.py --json outputs/snap_independent.json
"""
from __future__ import annotations

import argparse
import json
import pathlib
import shutil
import statistics as st
import sys
import tempfile
import zipfile

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from beatsaber_automapper.data.beatmap import (parse_difficulty_dat,  # noqa: E402
                                               parse_info_dat)

sys.path.insert(0, str(REPO))
from agent_mapper import refonsets  # noqa: E402

TOL_S = 0.050  # the alignment axis' own tolerance, reused for the human reference

# ★Sweep it. A threshold-defined gap that exists only at the value I chose is a
# construction: the "pendulum lock" axis looked decisive at MINRUN=6 and REVERSED
# SIGN by MINRUN=10 (TODO landmine, 2026-08-22). 30 ms is tighter than the snap
# window, 120 ms is the far edge of the near-miss band P0.7 was built for.
TOL_SWEEP_S = (0.020, 0.030, 0.040, 0.050, 0.070, 0.100, 0.120)


def human_note_times(sid: str) -> tuple[np.ndarray, str] | None:
    """Human note times for `sid`, in seconds, plus which difficulty they came from.

    ⚠️Exact basename match on info.dat -- "BPMInfo.dat" also ends with "info.dat" and
    73 of 300 corpus zips list it FIRST, which silently falls back to bpm 120 and
    stretches every note time (landmine from eval_human_replicate).

    ★**Falls back to ExpertPlus**: 10 of the 23 songset maps ship NO Expert at all, and
    demanding Expert silently dropped them -- a 23-song cohort quietly scoring 13. The
    test is PAIRED within a song (both arms hit the same reference), so a denser
    reference cannot favour either arm; it only makes that song's absolute
    `near_human` rate higher, which is why the difficulty is reported per song.
    """
    zp = REPO / "data" / "raw" / f"{sid}.zip"
    if not zp.exists():
        return None
    tmp = pathlib.Path(tempfile.mkdtemp(prefix="snapind_"))
    try:
        with zipfile.ZipFile(zp) as zf:
            names = zf.namelist()
            info = next((n for n in names
                         if n.split("/")[-1].lower() == "info.dat"), None)
            diff = dname = None
            for want in ("expertstandard.dat", "expertplusstandard.dat"):
                diff = next((n for n in names
                             if n.split("/")[-1].lower() == want), None)
                if diff is not None:
                    dname = want.replace("standard.dat", "")
                    break
            if info is None or diff is None:
                return None
            for n in (info, diff):
                (tmp / pathlib.Path(n).name).write_bytes(zf.read(n))
        meta = parse_info_dat(tmp / pathlib.Path(info).name)
        bm = parse_difficulty_dat(tmp / pathlib.Path(diff).name)
        if meta is None or bm is None or len(bm.color_notes) < 100:
            return None
        from beatsaber_automapper.evaluation import alignment
        t = np.asarray(alignment.note_times(bm, float(meta.bpm)), dtype=float)
        return np.sort(np.unique(t)), dname
    except Exception:  # noqa: BLE001
        return None
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def nearest_dist(t: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """|t - nearest ref| elementwise."""
    i = np.clip(np.searchsorted(ref, t), 1, len(ref) - 1)
    lo, hi = ref[i - 1], ref[i]
    return np.minimum(np.abs(t - lo), np.abs(t - hi))


def snap_to(ev: np.ndarray, ref: np.ndarray,
            window: float) -> tuple[np.ndarray, np.ndarray]:
    """Snap `ev` onto `ref` within `window`, WITHOUT dedup so pairing survives.

    ⚠️`refonsets.snap` sorts and deduplicates, which destroys the input/output
    pairing this test is built on. Returns (snapped, really_moved_mask).
    """
    i = np.clip(np.searchsorted(ref, ev), 1, len(ref) - 1)
    lo, hi = ref[i - 1], ref[i]
    near = np.where(np.abs(ev - lo) <= np.abs(ev - hi), lo, hi)
    d = np.abs(ev - near)
    out = np.where(d <= window, near, ev)
    return out, (d <= window) & (np.abs(out - ev) > 1e-9)


# ★NEGATIVE CONTROLS. A snap CONCENTRATES event times onto a discrete set, and
# concentration alone can raise a "share within X ms of a human note" -- the human's
# times are themselves quantised to a beat grid, so ANY discretisation might score
# better. These two destroy the onsets' alignment to the music while preserving the
# structure that would drive such an artifact:
#   shift  -- the same onset set, rigidly displaced. Identical count, identical
#             spacing distribution, musically wrong.
#   random -- uniform times at the same count. Kills the spacing structure too.
# If the real onsets beat BOTH, the gain is musical. If `shift` matches it, the gain
# is concentration and P0.7 is an artifact.
CONTROL_SHIFT_S = 0.200


def event_times(sid: str) -> np.ndarray | None:
    f = REPO / "outputs" / "event_cache" / f"{sid}.6s.json"
    if not f.exists():
        return None
    d = json.loads(f.read_text())
    return np.sort(np.unique(np.asarray([e["t"] for e in d["events"]], dtype=float)))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json", default="")
    ap.add_argument("--songs", nargs="*", default=None)
    a = ap.parse_args()

    sids = a.songs or sorted(p.stem for p in
                             (REPO / "data" / "eval_songset").glob("*.ogg"))
    rows = []
    print(f"{'song':8s}{'diff':>6s}{'moved':>8s}{'/ev':>7s}{'win%':>7s}"
          f"{'d_before':>10s}{'d_after':>9s}{'Δmed_ms':>9s}"
          f"{'nearH_b':>9s}{'nearH_a':>9s}")
    print("-" * 82)
    for sid in sids:
        ev = event_times(sid)
        h = human_note_times(sid)
        ref = refonsets.reference_onsets(sid)
        if ev is None or h is None or ref is None or len(h[0]) < 2:
            print(f"{sid:8s}  skipped (events={ev is not None} "
                  f"human={h is not None} onsets={ref is not None})")
            continue
        hum, dname = h

        W = refonsets.SNAP_WINDOW_S
        snapped, really_moved = snap_to(ev, ref, W)

        # Negative controls, scored on the SAME events over the SAME window.
        rng = np.random.default_rng(abs(hash(sid)) % (2 ** 32))
        ctl_refs = {
            "shift": np.sort(ref + CONTROL_SHIFT_S),
            "random": np.sort(rng.uniform(float(ref.min()), float(ref.max()), len(ref))),
        }

        # Only the events the snap actually moved carry information. An event that
        # did not move has d_before == d_after by construction.
        if really_moved.sum() < 10:
            print(f"{sid:8s}  skipped (only {int(really_moved.sum())} moved)")
            continue

        db = nearest_dist(ev[really_moved], hum)
        da = nearest_dist(snapped[really_moved], hum)
        wins = int((da < db - 1e-9).sum())
        losses = int((da > db + 1e-9).sum())
        decided = wins + losses
        winpct = 100.0 * wins / decided if decided else float("nan")

        row = dict(
            song=sid, difficulty=dname,
            n_events=int(len(ev)), n_moved=int(really_moved.sum()),
            wins=wins, losses=losses, win_pct=winpct,
            d_before_ms=float(np.median(db) * 1000),
            d_after_ms=float(np.median(da) * 1000),
            delta_med_ms=float(np.median(da - db) * 1000),
            # share landing within the axis' own tolerance of a HUMAN note
            near_human_before=float((db <= TOL_S).mean()),
            near_human_after=float((da <= TOL_S).mean()),
            # ★the same share swept across the tolerance. A gap that exists only at
            # the threshold I picked is a construction, not a result -- the pendulum
            # axis died exactly this way (TODO landmine, 2026-08-22).
            sweep={f"{int(t*1000)}": [float((db <= t).mean()), float((da <= t).mean())]
                   for t in TOL_SWEEP_S},
        )
        # Controls: score each on ITS OWN moved set, then compare the lift it buys
        # against the real onsets' lift. Comparing on the real set would ask the
        # control to explain moves it never makes.
        for cname, cref in ctl_refs.items():
            csnap, cmoved = snap_to(ev, cref, W)
            if cmoved.sum() < 10:
                row[f"lift_{cname}"] = float("nan")
                continue
            cb = nearest_dist(ev[cmoved], hum)
            ca = nearest_dist(csnap[cmoved], hum)
            row[f"lift_{cname}"] = float((ca <= TOL_S).mean() - (cb <= TOL_S).mean())
            row[f"win_{cname}"] = float(
                100.0 * (ca < cb - 1e-9).sum()
                / max(int((ca < cb - 1e-9).sum() + (ca > cb + 1e-9).sum()), 1))
        row["lift_real"] = row["near_human_after"] - row["near_human_before"]
        rows.append(row)
        print(f"{sid:8s}{dname:>6s}{row['n_moved']:8d}{row['n_events']:7d}{winpct:7.1f}"
              f"{row['d_before_ms']:10.1f}{row['d_after_ms']:9.1f}"
              f"{row['delta_med_ms']:9.2f}"
              f"{row['near_human_before']:9.3f}{row['near_human_after']:9.3f}")

    if not rows:
        print("\nno songs scored")
        return 1

    print("-" * 76)
    winp = [r["win_pct"] for r in rows]
    dlt = [r["delta_med_ms"] for r in rows]
    nhb = [r["near_human_before"] for r in rows]
    nha = [r["near_human_after"] for r in rows]
    n = len(rows)
    print(f"cohort n={n}")
    print(f"  win%           median {st.median(winp):6.1f}   "
          f"mean {st.mean(winp):6.1f}   sd {st.pstdev(winp):5.2f}")
    print(f"  Δ median ms    median {st.median(dlt):+6.2f}   "
          f"mean {st.mean(dlt):+6.2f}   sd {st.pstdev(dlt):5.2f}")
    print(f"  near-human     {st.mean(nhb):.4f} -> {st.mean(nha):.4f}   "
          f"(Δ {st.mean(nha) - st.mean(nhb):+.4f})")
    above = sum(1 for w in winp if w > 50.0)
    print(f"  songs with win% > 50: {above}/{n}")

    print(f"\n  near-human share vs tolerance  (does the effect live at ONE threshold?)")
    print(f"  {'tol_ms':>8s}{'before':>9s}{'after':>9s}{'Δ':>9s}{'songs+':>9s}")
    swept = []
    for t in TOL_SWEEP_S:
        k = f"{int(t*1000)}"
        b = st.mean([r["sweep"][k][0] for r in rows])
        aa = st.mean([r["sweep"][k][1] for r in rows])
        pos = sum(1 for r in rows if r["sweep"][k][1] > r["sweep"][k][0])
        swept.append(dict(tol_ms=int(t * 1000), before=b, after=aa,
                          delta=aa - b, songs_positive=pos))
        print(f"  {int(t*1000):8d}{b:9.4f}{aa:9.4f}{aa-b:+9.4f}{pos:>6d}/{n}")
    # ★Judge the sign only INSIDE the band the snap can act on. Past ~2x the 60 ms
    # snap window the measure has saturated (>0.78 before the snap even runs) and a
    # <=60 ms move can only push already-near events outward -- a reversal there is
    # mechanical, not evidence about the effect. Reversal INSIDE the band would be
    # the pendulum failure and would kill this.
    band = [s for s in swept if s["tol_ms"] <= 100]
    past = [s for s in swept if s["tol_ms"] > 100]
    band_signs = {np.sign(round(s["delta"], 4)) for s in band} - {0.0}
    print(f"  sign inside the snap's band (<=100 ms): "
          f"{'STABLE +' if band_signs == {1.0} else '🔴REVERSES — threshold artifact'}"
          f"   ({len(band)} tolerances)")
    if past:
        tail = ", ".join("{}ms {:+.4f}".format(s["tol_ms"], s["delta"]) for s in past)
        print(f"  past saturation (>100 ms): {tail}"
              f"  — expected: a <=60 ms move cannot help an event already that close")

    print(f"\n  NEGATIVE CONTROLS — is the lift musical, or just concentration?")
    print(f"  {'reference':>12s}{'lift@50ms':>12s}{'win%':>9s}")
    real_lift = st.mean([r["lift_real"] for r in rows])
    real_win = st.mean(winp)
    print(f"  {'real onsets':>12s}{real_lift:+12.4f}{real_win:9.1f}")
    ctl_summary = {}
    for cname in ("shift", "random"):
        lifts = [r[f"lift_{cname}"] for r in rows
                 if not np.isnan(r.get(f"lift_{cname}", float("nan")))]
        wins_c = [r[f"win_{cname}"] for r in rows if f"win_{cname}" in r]
        if not lifts:
            continue
        ctl_summary[cname] = dict(lift=st.mean(lifts),
                                  win=st.mean(wins_c) if wins_c else float("nan"))
        print(f"  {cname:>12s}{st.mean(lifts):+12.4f}"
              f"{st.mean(wins_c) if wins_c else float('nan'):9.1f}")
    beats_all = all(real_lift > c["lift"] + 0.005 for c in ctl_summary.values())
    print(f"  real onsets beat BOTH controls by >0.005: "
          f"{'✅YES — the lift is musical' if beats_all else '🔴NO — the lift is concentration'}")

    # Sign test over songs: how surprising is `above` under a fair coin?
    from math import comb
    p_two = sum(comb(n, k) for k in range(0, n + 1)
                if abs(k - n / 2) >= abs(above - n / 2)) / 2 ** n
    print(f"  sign test over songs: p = {p_two:.4g}")

    print("\nVERDICT LOGIC")
    print("  win% ~50 AND near-human Δ inside ±0.005  => the snap is AXIS-ONLY.")
    print("     It moves notes onto the judge's onsets without moving them toward")
    print("     the music a human heard. Generalising it ships a metric artifact.")
    print("  win% clearly >50 AND near-human Δ positive => the snap REALIGNS.")
    print("     P0.7's gain is genuine and generalising --snap-onsets is worth the")
    print("     second Demucs pass it costs on a non-corpus song.")

    if a.json:
        p = pathlib.Path(a.json)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(dict(
            rows=rows,
            win_pct_median=st.median(winp), delta_med_ms_median=st.median(dlt),
            near_human_before=st.mean(nhb), near_human_after=st.mean(nha),
            songs_above_half=above, n=n, sign_test_p=p_two,
            tolerance_sweep=swept, controls=ctl_summary,
            real_lift=real_lift, beats_controls=bool(beats_all)), indent=2))
        print(f"\nwrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
