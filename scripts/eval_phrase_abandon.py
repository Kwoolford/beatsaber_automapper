#!/usr/bin/env python
"""W4 — do we ABANDON vocal phrases part-way through?

> *"A few times the singer is still finishing a sentence and there's no notes."*
> — Kyle, 2026-08-03

Stated as a measurement. The `vocals` stem is in the seeded onset cache, so a
"phrase" can be defined without any new model: a run of vocal onsets separated by
less than `--gap` seconds, lasting at least `--min-len`. Then, per phrase, compare
how densely the map answers the **last third** against the **first two thirds**:

    tail_ratio = (notes per second in the final third)
                 -------------------------------------
                 (notes per second in the first two thirds)

    1.0  the map stays with the singer to the end of the line
    <1   the map drops out while the phrase is still going  <- Kyle's complaint
    >1   the map builds into the end of the line

The number is only meaningful against the human cohort: real mappers do thin out
at the end of a phrase sometimes, and a bar at 1.0 would be the `h_dist` mistake
of setting a target without looking. So both cohorts are always reported.

⚠️Like every metric from 2026-08-03, this is a DIAGNOSTIC until it clears
`scripts/audit_phase_metrics.py`-style controls. It reads note times only, so the
position-permuting controls will be blind to it by construction.

Usage:
    python scripts/eval_phrase_abandon.py --gen 'outputs/eval_sweep_cache/arm#s*__*.zip'
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


def vocal_phrases(song_id: str, gap: float, min_len: float) -> list[tuple[float, float]]:
    f = STEM_CACHE / f"{song_id}.npz"
    if not f.exists():
        return []
    d = np.load(f, allow_pickle=True)
    if "onsets_vocals" not in d.files:
        return []
    v = np.sort(np.asarray(d["onsets_vocals"], dtype=float))
    if len(v) < 20:
        return []
    out, s, prev = [], v[0], v[0]
    for t in v[1:]:
        if t - prev > gap:
            if prev - s >= min_len:
                out.append((s, prev))
            s = t
        prev = t
    if prev - s >= min_len:
        out.append((s, prev))
    return out


def phrase_silence(notes: np.ndarray, phrases) -> dict | None:
    """★THE PRIMARY METRIC — Kyle's sentence taken literally.

    `tail_ratio` below was the first attempt and it reported NO defect (ours and
    the human cohort both sit at a median of exactly 1.000). It was blunt: a
    *ratio* of densities cannot see a hole, because thinning from 4 notes to 2 and
    dropping to 0 for a second score similarly once averaged over a third of a
    phrase. Kyle did not say the map thins out. He said *"the singer is still
    finishing a sentence and there's no notes"* — that is **silence**, and this
    measures silence: the largest stretch inside a sung phrase containing no note
    at all, including the gaps at the phrase's own edges.

    Measured 2026-08-03 (gap 1.2 s, min_len 2.0 s):
        share_over_1s   ours 0.539  human 0.250   <- 2.2x
        med_hole        ours 1.071  human 0.698
        share_over_2s   ours 0.074  human 0.000
    """
    if len(notes) < 100 or len(phrases) < 4:
        return None
    holes = []
    for s, e in phrases:
        n = notes[(notes >= s) & (notes <= e)]
        if len(n) == 0:
            holes.append(e - s)
            continue
        pts = np.concatenate(([s], n, [e]))
        holes.append(float(np.max(np.diff(pts))))
    if len(holes) < 4:
        return None
    h = np.asarray(holes)
    return {"med_hole": round(float(np.median(h)), 4),
            "share_over_1s": round(float(np.mean(h > 1.0)), 4),
            "share_over_2s": round(float(np.mean(h > 2.0)), 4)}


def tail_ratio(notes: np.ndarray, phrases) -> dict | None:
    if len(notes) < 100 or len(phrases) < 4:
        return None
    ratios, abandoned = [], 0
    for s, e in phrases:
        dur = e - s
        if dur <= 0:
            continue
        cut = s + dur * (2.0 / 3.0)
        head = ((notes >= s) & (notes < cut)).sum() / (cut - s)
        tail = ((notes >= cut) & (notes <= e)).sum() / (e - cut)
        if head <= 0:
            continue           # we never engaged the phrase at all; not "abandoned"
        ratios.append(tail / head)
        if tail / head < 0.5:
            abandoned += 1
    if len(ratios) < 4:
        return None
    return {"tail_ratio": round(float(np.median(ratios)), 4),
            "abandon_rate": round(abandoned / len(ratios), 4),
            "n_phrases": len(ratios)}


def scan(paths, loader, label: str, gap: float, min_len: float) -> list[dict]:
    rows = []
    for p in paths:
        pp = pathlib.Path(p)
        ph = vocal_phrases(scorecard.song_id(pp), gap, min_len)
        if not ph:
            continue
        try:
            L = loader(pp)
        except Exception:  # noqa: BLE001
            continue
        if not L:
            continue
        notes = np.sort(np.asarray(alignment.note_times(L[0], L[1]), dtype=float))
        r = tail_ratio(notes, ph) or {}
        sil = phrase_silence(notes, ph)
        if sil:
            r.update(sil)
        if r and "share_over_1s" in r:
            r["song"] = scorecard.song_id(pp)
            rows.append(r)
    print(f"{label}: {len(rows)} maps scored")
    return rows


def report(rows, label) -> dict:
    if not rows:
        return {}
    print(f"\n=== {label} (n={len(rows)}) ===")
    out = {"n": len(rows)}
    for k in ("share_over_1s", "share_over_2s", "med_hole", "tail_ratio", "abandon_rate"):
        # Rows are heterogeneous on purpose: phrase_silence can succeed where
        # tail_ratio returns None (a map that never engages a phrase has no ratio
        # to take but definitely has a hole). Filter per key rather than assuming
        # rows[0] is representative.
        v = [r[k] for r in rows if k in r]
        if len(v) < 4:
            continue
        out[k] = {"median": round(st.median(v), 4),
                  "p10": round(float(np.percentile(v, 10)), 4),
                  "p90": round(float(np.percentile(v, 90)), 4)}
        print(f"  {k:14s} median {st.median(v):7.4f}   p10 {np.percentile(v,10):7.4f}"
              f"   p90 {np.percentile(v,90):7.4f}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gen", default="outputs/eval_sweep_cache/tf_trim_ev03_rc05#s*__*.zip")
    ap.add_argument("--human-n", type=int, default=200)
    ap.add_argument("--gap", type=float, default=1.2,
                    help="seconds of vocal silence that ends a phrase")
    ap.add_argument("--min-len", type=float, default=2.0)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()

    cached = {p.stem for p in STEM_CACHE.glob("*.npz")}
    human = [p for p in sorted((REPO / "data" / "raw").glob("*.zip"))
             if p.stem in cached][:a.human_n]

    g = scan(sorted(glob.glob(a.gen)), scorecard._load_any, "ours", a.gap, a.min_len)
    h = scan(human, load_expert_only, "human", a.gap, a.min_len)
    out = {"ours": report(g, "OURS"), "human": report(h, "HUMAN (strict Expert)")}

    if out["ours"] and out["human"]:
        o, hh = out["ours"], out["human"]
        print("\n=== READ ===")
        print(f"  share_over_1s ours {o['share_over_1s']['median']:.3f}"
              f"   human {hh['share_over_1s']['median']:.3f}   <- THE metric")
        print(f"  tail_ratio    ours {o['tail_ratio']['median']:.3f}"
              f"   human {hh['tail_ratio']['median']:.3f}   (blunt; reported for the record)")
        print("  share_over_1s is the fraction of SUNG phrases containing a >1s")
        print("  stretch with no notes at all -- Kyle's sentence taken literally.")
        print("  ⚠️tail_ratio said there was NO defect here (both cohorts 1.000). A")
        print("  ratio of densities cannot see a hole. Trust the silence metric.")

    if a.json:
        out["ours_rows"], out["human_rows"] = g, h
        pathlib.Path(a.json).write_text(json.dumps(out, indent=2))
        print(f"\nwrote {a.json}")


if __name__ == "__main__":
    main()
