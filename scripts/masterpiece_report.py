#!/usr/bin/env python
"""★ONE COMMAND FOR THE MASTERPIECE AXES — and the only correct way to compare arms.

    python scripts/masterpiece_report.py --arm tf_trim_ev03_rc05
    python scripts/masterpiece_report.py --arm mbb025 --vs tf_trim_ev03_rc05
    python scripts/masterpiece_report.py --arm v8_mbb025 --seeds s0,s1,s2

`suite_report.py` is the front door for one song: is this map on the beat, where
should I listen. This is the front door for the question behind Kyle's *"we need a
model that produces masterpieces"* — over the whole cohort, does the map answer the
song's STRUCTURE:

    M1  when the music comes back, does the pattern come back
    M2  is the map playing this bar's figure, and whose figure
    M3  is the emphasis spent where the music emphasises
    M4  does the map turn when the song turns   (⚠️fails its own control — shown
                                                  for continuity, never quoted)

Every axis is a CONTRAST — what the map does where the music says X, minus what it
does where the music says not-X — which is why a metronome and a random map score 0
on all of them, and why they are the first axes here that a degenerate strategy
cannot reach. Which ones may actually steer a lever is decided by
`audit_masterpiece.py` and read from its JSON; this report prints that verdict
beside every number so a diagnostic value is never mistaken for a target.

⚠️THE ONE RULE FOR READING THIS: quote the PAIRED delta, never two medians. Only
about half the eval songs ship a human Expert map, and comparing our median over
24 songs against a human median over 13 is the population error this project has
made more often than any other.
"""

from __future__ import annotations

import argparse
import glob
import json
import pathlib
import statistics as st
import sys

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

import eval_accent as m3  # noqa: E402
import eval_arrangement as m4  # noqa: E402
import eval_motif_rhyme as m1  # noqa: E402
import eval_rhythm_fidelity as m2  # noqa: E402
import song_structure as ss  # noqa: E402
from beatsaber_automapper.evaluation import alignment, scorecard  # noqa: E402
from calibrate_playfeel import load_expert_only  # noqa: E402

CACHE = REPO / "outputs" / "masterpiece"
AUDIT = REPO / "outputs" / "audit_masterpiece_2026-08-04.json"

REPORT_KEYS = [
    ("M1", "rhy_rhythm"), ("M1", "harm_rhythm"), ("M1", "timb_rhythm"),
    ("M1", "harm_place"),
    ("M2", "follow_mean"), ("M2", "follow_best"), ("M2", "follow_vocals"),
    ("M2", "follow_drums"), ("M2", "lead_persistence"),
    ("M3", "hands_x_downbeat"), ("M3", "hands_x_strength"), ("M3", "hands_x_coincid"),
    ("M3", "double_share"),
    ("M4", "arrange"),
]


def score_one(song: str, notes: list[tuple], B, A, stems, bnds) -> dict:
    out = {}
    out.update(m1.song_scores(notes, B, A) or {})
    out.update(m2.score_map(np.sort(np.array([n[0] for n in notes])), B, stems) or {})
    out.update(m3.score_map(notes, song, B) or {})
    if len(bnds) >= 4:
        out.update(m4.score_map(notes, B, bnds) or {})
    return out


def collect(arm: str, seed: str, rebuild: bool = False) -> list[dict]:
    CACHE.mkdir(parents=True, exist_ok=True)
    cf = CACHE / f"{arm}#{seed}.json"
    if cf.exists() and not rebuild:
        return json.loads(cf.read_text())
    files = sorted(glob.glob(str(REPO / f"outputs/eval_sweep_cache/{arm}#{seed}__*.zip"))) \
        or sorted(glob.glob(str(REPO / f"outputs/eval_sweep_cache/{arm}__*.zip")))
    rows = []
    for f in files:
        song = pathlib.Path(f).stem.split("__")[-1]
        L = scorecard._load_any(pathlib.Path(f))
        if not L:
            continue
        bm, bpm = L[0], float(L[1])
        t = np.asarray(alignment.note_times(bm, bpm), dtype=float)
        if len(t) < 100:
            continue
        B = ss.bars(song, bpm, ss.song_end(song, float(t.max())))
        if B is None or B.n < 24:
            continue
        A = ss.bar_audio_matrix(song, B)
        stems = m2.stem_onsets(song)
        if A is None or len(stems) < 3:
            continue
        nov = m4.novelty(A)
        bnds = m4.boundaries(nov) if nov is not None else []
        ours = score_one(song, m1.notes_xydc(bm, bpm), B, A, stems, bnds)
        H = load_expert_only(REPO / "data" / "raw" / f"{song}.zip")
        human = score_one(song, m1.notes_xydc(H[0], float(H[1])), B, A, stems, bnds) \
            if H else None
        rows.append({"song": song, "bars": B.n, "ours": ours, "human": human})
        print(f"    scored {song}")
    cf.write_text(json.dumps(rows, indent=1))
    return rows


def steer_verdicts() -> dict:
    if not AUDIT.exists():
        return {}
    d = json.loads(AUDIT.read_text())
    return {k: v.get("may_steer") for k, v in d.get("verdicts", {}).items()}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", default="tf_trim_ev03_rc05")
    ap.add_argument("--vs", default="", help="second arm; prints the PAIRED arm delta")
    ap.add_argument("--seeds", default="s0")
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--json", default="")
    a = ap.parse_args()

    seeds = a.seeds.split(",")
    print(f"collecting {a.arm} at seeds {seeds}")
    rows_by_seed = {s: collect(a.arm, s, a.rebuild) for s in seeds}
    rows = rows_by_seed[seeds[0]]
    verdicts = steer_verdicts()

    print(f"\n{'='*104}")
    print(f"MASTERPIECE REPORT — {a.arm} (seed {seeds[0]}), against the human map on the same songs")
    print(f"{'='*104}")
    print(f"{'ax':<4}{'metric':<20} {'n':>3} {'ours':>9} {'human':>9} "
          f"{'paired Δ':>10} {'Δ med':>9} {'resolv':>7}  steer?")
    summary = {}
    for ax, k in REPORT_KEYS:
        both = [r for r in rows if r.get("human")
                and r["ours"].get(k) is not None and r["human"].get(k) is not None]
        if len(both) < 5:
            continue
        o = st.median([r["ours"][k] for r in both])
        h = st.median([r["human"][k] for r in both])
        p = ss.paired_delta(rows, k)
        v = verdicts.get(k)
        mark = {True: "MAY STEER", False: "diagnostic"}.get(v, "unaudited")
        summary[k] = {"ours": round(o, 4), "human": round(h, 4), "paired": p,
                      "may_steer": v}
        print(f"{ax:<4}{k:<20} {len(both):>3d} {o:>+9.4f} {h:>+9.4f} "
              f"{p.get('delta', float('nan')):>+10.4f} "
              f"{p.get('delta_median', float('nan')):>+9.4f} "
              f"{('YES' if p.get('resolvable') else 'no'):>7}  {mark}")

    # ---- seed spread: is a difference between arms bigger than the seed noise?
    if len(seeds) > 1:
        print(f"\nSEED SPREAD over {len(seeds)} seeds (sd of the per-song median)")
        for ax, k in REPORT_KEYS:
            vals = []
            for s in seeds:
                v = [r["ours"][k] for r in rows_by_seed[s]
                     if r["ours"].get(k) is not None]
                if v:
                    vals.append(st.median(v))
            if len(vals) == len(seeds) and len(vals) > 1:
                print(f"  {k:<20} {np.mean(vals):+.4f} ± {np.std(vals, ddof=1):.4f}")

    # ---- arm vs arm, paired by song (the only valid arm comparison)
    if a.vs:
        print(f"\ncollecting {a.vs}")
        other = collect(a.vs, seeds[0], a.rebuild)
        by_song = {r["song"]: r["ours"] for r in other}
        merged = [{"a": r["ours"], "b": by_song[r["song"]]}
                  for r in rows if r["song"] in by_song]
        print(f"\n{'='*104}")
        print(f"ARM COMPARISON — {a.arm} minus {a.vs}, paired by song (n={len(merged)})")
        print("⚠️A difference here is only meaningful against the seed spread above; "
              "at one seed treat it as a screen.")
        print(f"{'='*104}")
        print(f"{'metric':<20} {'arm':>9} {'ref':>9} {'paired Δ':>10} {'resolv':>7}  steer?")
        for ax, k in REPORT_KEYS:
            d = ss.paired_delta(merged, k, a="a", b="b")
            if not d:
                continue
            va = [m["a"][k] for m in merged if m["a"].get(k) is not None]
            vb = [m["b"][k] for m in merged if m["b"].get(k) is not None]
            v = verdicts.get(k)
            mark = {True: "MAY STEER", False: "diagnostic"}.get(v, "unaudited")
            print(f"{k:<20} {st.median(va):>+9.4f} {st.median(vb):>+9.4f} "
                  f"{d['delta']:>+10.4f} "
                  f"{('YES' if d['resolvable'] else 'no'):>7}  {mark}")

    if a.json:
        pathlib.Path(a.json).write_text(json.dumps(
            {"arm": a.arm, "summary": summary}, indent=2))
        print(f"\nwrote {a.json}")


if __name__ == "__main__":
    main()
