#!/usr/bin/env python
"""Is our missing pulse caused by the UNION of two event streams, or by each stream?

P0.5 says our autobuilt maps do not hold a pulse (`pulse_stability` 0.329 vs human
0.560 on 23 songs) and names a suspected mechanism: `autobuild` places notes by
running `mapctl auto` TWICE per section -- once following the drums, once following
the melodic carrier -- so the note times are the union of two independently
accent-filtered streams, and **the union of two rhythms is not a rhythm**.

That is a hypothesis, not a measurement. It has an obvious alternative: each stream
may already be un-pulsed on its own (the accent percentile keeps the loudest events,
which need not be evenly spaced), in which case merging is innocent and the fix would
be aimed at the wrong place.

This separates them by BUILDING the arms rather than reasoning about them:

    DRUMS    only the drums pass, at its planned accent percentile
    CARRIER  only the melodic carrier pass, at its planned accent percentile
    UNION    both, exactly as `autobuild` ships them  (= the current map)

★If UNION is much worse than BOTH of its parts, the merge is the defect and P0.5's
proposed fix (choose one stream to define the pulse per section) is aimed correctly.
★If DRUMS and CARRIER are already near UNION, the defect is in event SELECTION and
picking a carrier per section will not fix it.

The human map for the same song is scored as the ceiling, so the arms are read
against the spread that actually matters rather than against 1.0.

Runs on cached events (no Demucs, no GPU) once `stem_cache`/`event_cache` are warm.

Usage:
    python scripts/diag_pulse_union.py --songs 1f767 1f8d6 --json outputs/pulse.json
    python scripts/diag_pulse_union.py --n 8
"""
from __future__ import annotations

import argparse
import contextlib
import json
import pathlib
import shutil
import statistics
import subprocess
import sys
import tempfile

REPO = pathlib.Path(__file__).resolve().parents[1]
AM = REPO / "agent_mapper"
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(AM))

SONGSET = REPO / "data" / "eval_songset"
RAW = REPO / "data" / "raw"
ARMS = ("DRUMS", "CARRIER", "UNION")
KEYS = ("pulse_stability", "dominant_share", "ioi_switch_rate", "ioi_cond_entropy")


def run(args: list[str]) -> str:
    r = subprocess.run([sys.executable, *args], capture_output=True, text=True,
                       cwd=REPO)
    if r.returncode not in (0, 1):
        raise RuntimeError(f"{' '.join(args[-5:])}\n{r.stdout}\n{r.stderr}")
    return r.stdout


def build_arm(audio: pathlib.Path, name: str, rows: list[dict], arm: str,
              out: pathlib.Path) -> None:
    """One arm of the same plan: drums only, carrier only, or both."""
    run([str(AM / "mapctl.py"), "init", str(audio), "--name", name, "--fresh"])
    for r in rows:
        bars = f"{r['bar0']}-{r['bar1']}"
        if arm in ("DRUMS", "UNION") and r["drums_n"]:
            cmd = [str(AM / "mapctl.py"), "auto", name, "--bars", bars,
                   "--follow", "drums", "--wide"]
            if r["drums_pct"]:
                cmd += ["--accent-pct", str(r["drums_pct"])]
            run(cmd)
        if arm in ("CARRIER", "UNION") and r["carrier"]:
            cmd = [str(AM / "mapctl.py"), "auto", name, "--bars", bars,
                   "--follow", r["carrier"], "--wide"]
            if r["carrier_pct"]:
                cmd += ["--accent-pct", str(r["carrier_pct"])]
            run(cmd)
    run([str(AM / "mapctl.py"), "export", name, "--out", str(out)])


def score(zp: pathlib.Path) -> dict | None:
    """Rhythm metrics for a map zip, or None if it will not load.

    ⚠️`scorecard._load_any` returns **0 notes** for a `mapctl export` zip -- the same
    silent-empty failure that hid the alignment gap. Use the pair of loaders
    `mapjudge.judge_zip` uses (human layout first, then ours), which is the only
    loader verified on both sides of a human-vs-ours comparison.
    """
    from beatsaber_automapper.evaluation import mapjudge as mj
    from beatsaber_automapper.evaluation import rhythm
    sys.path.insert(0, str(REPO / "scripts"))
    from audit_eval_suite import _load_generated, _load_human  # noqa: PLC0415

    notes = None
    for loader in (_load_human, _load_generated):
        try:
            got = loader(zp)
        except Exception:  # noqa: BLE001
            continue
        if got and got[0]:
            notes = got[0]
            break
    if not notes or len(notes) < 50:
        print(f"    ⚠️ {zp.name}: {0 if not notes else len(notes)} notes loaded — "
              f"not scored")
        return None
    # `rhythm_metrics` scores DISTINCT note beats, so doubles cannot move it —
    # which is why the pulse defect is about note TIMES and not note count.
    rep = rhythm.rhythm_metrics(mj._BM(notes))
    m = dict(rep.metrics)
    m["n_notes"] = len(notes)
    beats = sorted({round(n.beat, 4) for n in notes})
    m["n_times"] = len(beats)
    # The IOI histogram itself, in beats. `pulse_stability` says whether we HOLD an
    # interval; this says WHICH intervals we hold, which is what a fix has to change.
    iois = [round(b - a, 3) for a, b in zip(beats, beats[1:])]
    m["_iois"] = [x for x in iois if 0 < x <= 4.0]
    return m


def human_zip(sid: str) -> pathlib.Path | None:
    p = RAW / f"{sid}.zip"
    return p if p.exists() else None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--songs", nargs="*", default=None)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--nps", type=float, default=None)
    ap.add_argument("--json", type=pathlib.Path)
    a = ap.parse_args()

    import autobuild as AB

    sids = a.songs or [p.stem for p in sorted(SONGSET.glob("*.ogg"))][: a.n]
    nps = a.nps if a.nps is not None else AB.HUMAN_NPS

    tmp = pathlib.Path(tempfile.mkdtemp(prefix="pulseunion_"))
    rows_out: list[dict] = []
    for sid in sids:
        audio = SONGSET / f"{sid}.ogg"
        if not audio.exists():
            print(f"  {sid}: no audio in the songset, skipped")
            continue
        try:
            plan = AB.plan(audio, nps)
        except Exception as exc:  # noqa: BLE001
            print(f"  {sid}: plan failed ({exc}), skipped")
            continue
        rec: dict = {"song": sid}
        for arm in ARMS:
            name = f"pu_{sid}_{arm.lower()}"
            zp = tmp / f"{arm}__{sid}.zip"
            try:
                build_arm(audio, name, plan, arm, zp)
            except Exception as exc:  # noqa: BLE001
                print(f"  {sid}/{arm}: build failed ({exc})")
                continue
            finally:
                # `mapctl clear` empties a BAR RANGE, it does not delete the
                # session; a stale session would be silently re-used by the next arm.
                with contextlib.suppress(Exception):
                    shutil.rmtree(AM / "sessions" / name)
            rec[arm] = score(zp)
        hz = human_zip(sid)
        rec["HUMAN"] = score(hz) if hz else None
        rows_out.append(rec)
        got = {k: (rec.get(k) or {}).get("pulse_stability") for k in
               (*ARMS, "HUMAN")}
        print(f"  {sid}: " + "  ".join(
            f"{k}={v:.3f}" if isinstance(v, float) else f"{k}=--"
            for k, v in got.items()))

    print(f"\n{'metric':<20} " + "".join(f"{k:>10}" for k in (*ARMS, "HUMAN")))
    print("-" * 72)
    summary: dict[str, dict[str, float]] = {}
    for key in KEYS:
        cells = {}
        for k in (*ARMS, "HUMAN"):
            vals = [r[k][key] for r in rows_out
                    if r.get(k) and key in r[k] and r[k][key] == r[k][key]]
            if vals:
                cells[k] = statistics.median(vals)
        summary[key] = cells
        print(f"{key:<20} " + "".join(
            f"{cells[k]:>10.3f}" if k in cells else f"{'--':>10}"
            for k in (*ARMS, "HUMAN")))
    nn = {k: statistics.median([r[k]["n_notes"] for r in rows_out if r.get(k)])
          for k in (*ARMS, "HUMAN") if any(r.get(k) for r in rows_out)}
    print(f"{'n_notes (median)':<20} " + "".join(
        f"{nn[k]:>10.0f}" if k in nn else f"{'--':>10}" for k in (*ARMS, "HUMAN")))

    # ---- WHICH intervals, pooled over songs ----
    from collections import Counter
    pool = {k: Counter() for k in (*ARMS, "HUMAN")}
    for r in rows_out:
        for k in (*ARMS, "HUMAN"):
            if r.get(k):
                pool[k].update(r[k].get("_iois") or [])
    shown = sorted({v for k in pool for v, _ in pool[k].most_common(6)})
    print(f"\nIOI share (beats), pooled over {len(rows_out)} songs")
    print(f"{'ioi':<20} " + "".join(f"{k:>10}" for k in (*ARMS, "HUMAN")))
    print("-" * 72)
    for v in shown:
        print(f"{v:<20} " + "".join(
            f"{pool[k][v] / max(sum(pool[k].values()), 1):>10.3f}"
            for k in (*ARMS, "HUMAN")))
    print(f"{'distinct IOIs':<20} " + "".join(
        f"{len(pool[k]):>10d}" for k in (*ARMS, "HUMAN")))
    # ★A map whose intervals are spread over many near-duplicate values cannot hold
    # a pulse by construction, however it was selected.
    for k in (*ARMS, "HUMAN"):
        tot = max(sum(pool[k].values()), 1)
        top3 = sum(n for _, n in pool[k].most_common(3)) / tot
        print(f"  {k:<8} top-3 IOIs cover {top3:.3f} of intervals")

    # ---- verdict logic, stated before the numbers are read ----
    ps = summary.get("pulse_stability", {})
    print(f"\nsongs scored: {len(rows_out)}")
    if all(k in ps for k in ("DRUMS", "CARRIER", "UNION", "HUMAN")):
        # ★★TWO-SIDED, deliberately. `pulse_stability` has a human value, not a
        # human ceiling: 1.0 is a metronome. Scoring `human - arm` and taking the
        # max rewards an arm for overshooting, which is the exact one-sided read
        # that let `idiom_local` sit at the 98th percentile on 23/23 songs and
        # looked like a win. Distance is |arm - human| on every axis here.
        d = {k: abs(ps[k] - ps["HUMAN"]) for k in ARMS}
        near = min(d, key=d.get)
        side = {k: ("rigid" if ps[k] > ps["HUMAN"] else "loose") for k in ARMS}
        for k in ARMS:
            print(f"  {k:<8} {ps[k]:.3f}  |Δhuman| {d[k]:.3f}  ({side[k]})")
        print(f"  HUMAN    {ps['HUMAN']:.3f}")
        best_part = min(d["DRUMS"], d["CARRIER"])
        if best_part < 0.5 * d["UNION"]:
            print(f"⇒ THE MERGE COSTS PULSE: `{near}` alone sits {best_part:.3f} from "
                  f"the human against the union's {d['UNION']:.3f}. P0.5's fix (one "
                  f"stream defines the pulse per section) is aimed correctly.")
        elif best_part > 0.8 * d["UNION"]:
            print("⇒ THE MERGE IS NOT THE DEFECT: each stream is about as far from "
                  "the human alone. The defect is in event SELECTION, and picking a "
                  "carrier per section will NOT fix it.")
        else:
            print("⇒ PARTIAL: the merge costs some pulse but no single stream reaches "
                  "the human either. Both selection and merging contribute.")
        if side["DRUMS"] != side["CARRIER"]:
            print(f"★THE TWO STREAMS MISS ON OPPOSITE SIDES — drums {side['DRUMS']}, "
                  f"carrier {side['CARRIER']}, human between them. That is not 'pick "
                  f"one'; it says the {side['DRUMS'] == 'rigid' and 'drums' or 'carrier'}"
                  f" stream should DEFINE the grid and the other should be placed ON "
                  f"it, rather than contributing independent times.")
        print("⚠️A single stream also carries FEWER notes — compare n_notes before "
              "reading any pulse gain as free.")
    else:
        print("⚠️ not all arms scored — no verdict.")

    if a.json:
        a.json.parent.mkdir(parents=True, exist_ok=True)
        a.json.write_text(json.dumps({"songs": rows_out, "median": summary},
                                     indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
