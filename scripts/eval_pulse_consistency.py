#!/usr/bin/env python
"""W2 sharpened — do we play a steady beat CONSISTENTLY, or one in every two or three?

> *"It's on beat, but it's also an expert song and we shouldn't be afraid to play
> a simple beat that's medium tempo… we play like 1 out of 2/3 notes of an obvious
> slow beat. It just feels really empty for no reason."* — Kyle on Fallen Kingdom

**Why a new instrument is needed.** Everything measured on 2026-08-03/04 failed to
separate the map he called empty from the map he graded A+:

    distinct-nps / that song's own human map   Hunger 0.650   Fallen Kingdom 0.781
    response to k>=3 events                    Hunger 0.545   Fallen Kingdom 0.667
    sung phrases with a >1s hole               Hunger 0.500   Fallen Kingdom 0.538

On every one of those, Fallen Kingdom is EQUAL OR BETTER than the A+ map. So
"empty" is not overall density, not coincidence response, and not phrase holes.
Read his sentence again: *one out of two or three notes of an obvious beat*. That
is not a claim about totals — it is a claim about **consistency on a pulse**.

**Method.** Take the beat grid from the map's own bpm. A beat is "played by the
music" if any stem onset falls within `--tol` of it. Then, over runs where the
music plays consecutive beats:

    pulse_coverage   share of music-played beats that we answer with a note
    pulse_continuity P(we play beat n+1 | we played beat n), over consecutive
                     music-played beats -- the DIRECT form of "1 out of 2/3":
                     playing every beat gives ~1.0, playing alternate beats ~0.5,
                     one in three ~0.33

`pulse_continuity` is the one his sentence predicts. Coverage can be high while
continuity is low (scattered notes) and vice versa, so both are reported.

⚠️DIAGNOSTIC ONLY. Both 2026-08-03/04 metrics failed the control battery because a
metronome beats a human on them, and this one rewards regularity even more
directly — **assume it fails as a steering target and check before using it.**

Usage:
    python scripts/eval_pulse_consistency.py --gen 'outputs/eval_sweep_cache/arm#s*__*.zip'
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

from beatsaber_automapper.evaluation import alignment, scorecard  # noqa: E402
from calibrate_playfeel import load_expert_only  # noqa: E402

STEM_CACHE = REPO / "outputs" / "stem_onset_cache"


def stem_union(song_id: str) -> np.ndarray | None:
    f = STEM_CACHE / f"{song_id}.npz"
    if not f.exists():
        return None
    d = np.load(f, allow_pickle=True)
    if "onsets_union" in d.files:
        return np.sort(np.asarray(d["onsets_union"], dtype=float))
    return None


def pulse_metrics(notes: np.ndarray, onsets: np.ndarray, bpm: float,
                  tol: float, min_run: int) -> dict | None:
    if len(notes) < 100 or onsets is None or len(onsets) < 100:
        return None
    beat = 60.0 / bpm
    end = float(min(notes.max(), onsets.max()))
    beats = np.arange(0.0, end, beat)
    if len(beats) < 32:
        return None

    def near(arr, t):
        i = int(np.searchsorted(arr, t))
        c = [arr[j] for j in (i - 1, i) if 0 <= j < len(arr)]
        return bool(c) and min(abs(t - x) for x in c) <= tol

    music = np.array([near(onsets, t) for t in beats])
    ours = np.array([near(notes, t) for t in beats])

    # Only judge inside RUNS of consecutive music-played beats -- an "obvious
    # beat". Isolated hits are not a pulse and demanding coverage there would
    # penalise correct restraint.
    runs, i = [], 0
    while i < len(music):
        if music[i]:
            j = i
            while j < len(music) and music[j]:
                j += 1
            if j - i >= min_run:
                runs.append((i, j))
            i = j
        else:
            i += 1
    if not runs:
        return None

    cov_n = cov_d = 0
    cont_n = cont_d = 0
    for s, e in runs:
        seg = ours[s:e]
        cov_n += int(seg.sum())
        cov_d += len(seg)
        for k in range(len(seg) - 1):
            if seg[k]:
                cont_d += 1
                cont_n += int(seg[k + 1])
    if cov_d < 32 or cont_d < 16:
        return None
    return {"pulse_coverage": round(cov_n / cov_d, 4),
            "pulse_continuity": round(cont_n / cont_d, 4),
            "n_pulse_beats": cov_d}


def scan(paths, loader, label, tol, min_run):
    rows = []
    for p in paths:
        pp = pathlib.Path(p)
        on = stem_union(scorecard.song_id(pp))
        if on is None:
            continue
        try:
            L = loader(pp)
        except Exception:  # noqa: BLE001
            continue
        if not L:
            continue
        notes = np.sort(np.asarray(alignment.note_times(L[0], L[1]), dtype=float))
        r = pulse_metrics(notes, on, float(L[1]), tol, min_run)
        if r:
            r["song"] = scorecard.song_id(pp)
            rows.append(r)
    print(f"{label}: {len(rows)} maps scored")
    return rows


def report(rows, label):
    if not rows:
        return {}
    print(f"\n=== {label} (n={len(rows)}) ===")
    out = {"n": len(rows)}
    for k in ("pulse_coverage", "pulse_continuity"):
        v = [r[k] for r in rows]
        out[k] = {"median": round(st.median(v), 4),
                  "p10": round(float(np.percentile(v, 10)), 4),
                  "p90": round(float(np.percentile(v, 90)), 4)}
        print(f"  {k:17s} median {st.median(v):7.4f}   p10 {np.percentile(v,10):7.4f}"
              f"   p90 {np.percentile(v,90):7.4f}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gen", default="outputs/eval_sweep_cache/tf_trim_ev03_rc05#s*__*.zip")
    ap.add_argument("--human-n", type=int, default=150)
    ap.add_argument("--tol", type=float, default=0.060)
    ap.add_argument("--min-run", type=int, default=4,
                    help="consecutive music-played beats before it counts as a pulse")
    ap.add_argument("--songs", default=None, help="comma list to print per-song")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()

    cached = {p.stem for p in STEM_CACHE.glob("*.npz")}
    human = [p for p in sorted((REPO / "data" / "raw").glob("*.zip"))
             if p.stem in cached][:a.human_n]

    g = scan(sorted(glob.glob(a.gen)), scorecard._load_any, "ours", a.tol, a.min_run)
    h = scan(human, load_expert_only, "human", a.tol, a.min_run)
    out = {"ours": report(g, "OURS"), "human": report(h, "HUMAN (strict Expert)")}

    if a.songs:
        print("\n=== PER SONG (the two that matter) ===")
        print(f"{'song':10s}{'ours cov':>10s}{'ours cont':>11s}{'hum cov':>10s}{'hum cont':>10s}")
        for sid in a.songs.split(","):
            o = [r for r in g if r["song"] == sid]
            hh = [r for r in h if r["song"] == sid]
            if not o:
                continue
            oc = st.median([r["pulse_coverage"] for r in o])
            ot = st.median([r["pulse_continuity"] for r in o])
            hc = st.median([r["pulse_coverage"] for r in hh]) if hh else float("nan")
            ht = st.median([r["pulse_continuity"] for r in hh]) if hh else float("nan")
            print(f"{sid:10s}{oc:10.4f}{ot:11.4f}{hc:10.4f}{ht:10.4f}")

    if out["ours"] and out["human"]:
        print("\n=== READ ===")
        print("  Kyle's sentence predicts OUR pulse_continuity sits well below the")
        print("  human's: 'one out of two or three notes' is continuity ~0.33-0.50")
        print("  against a human near 1.0 on an obvious beat.")
        print("  ⚠️If Fallen Kingdom does NOT separate from Hunger here either, then")
        print("  four instruments have failed to find 'empty' and the honest report")
        print("  is that we cannot measure it yet -- ask him what he hears, do not")
        print("  keep inventing metrics.")

    if a.json:
        out["ours_rows"], out["human_rows"] = g, h
        pathlib.Path(a.json).write_text(json.dumps(out, indent=2))
        print(f"\nwrote {a.json}")


if __name__ == "__main__":
    main()
