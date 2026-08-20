#!/usr/bin/env python
"""Calibrate the per-map judge (`evaluation.mapjudge`) from the human corpus.

**Three disjoint slices, because the judge makes two different claims.**

    DIST     -> the empirical percentile distribution each metric is read against
    CALIB    -> the conformal threshold (how extreme a HUMAN map typically is)
    HELDOUT  -> the measured accept rate, on maps used for neither of the above

A map scored against a distribution it helped build has an inflated percentile, and
a threshold set on the same maps it is then tested on is circular -- this project has
recorded that failure twice (`h_dist`, and onset precision under `BEAT_GRID_PHASE`).
Three slices is the cheapest construction that has neither.

★**The held-out accept rate is a CHECK, not a result.** Conformal calibration
guarantees it lands near 1-alpha; if it does not, the splits are broken. The actual
evidence that the judge works is `scripts/audit_mapjudge.py` -- whether it REJECTS
degenerate maps.

Slices start at index 1000 of the canonical corpus ordering (sorted glob, then
`random.Random(0).shuffle`) because every existing reference in this repo -- flow,
rhythm, idiom, handrole, playfeel, and the idiom VOCABULARY -- is mined from indices
32..432 of that same ordering. Starting at 1000 keeps the judge disjoint from all of
them, so a map's own idioms cannot inflate its `idiom_coverage`.

Usage:
    python scripts/calibrate_mapjudge.py --n 1200
"""
from __future__ import annotations

import argparse
import json
import pathlib
import random
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from audit_eval_suite import _load_human  # noqa: E402
from beatsaber_automapper.evaluation import mapjudge as mj  # noqa: E402
from beatsaber_automapper.evaluation import scorecard  # noqa: E402

RAW = REPO / "data" / "raw"
# Existing references occupy 32..432 of the canonical ordering; the idiom vocab
# occupies 32..232. Anything at or past this index is clean.
CORPUS_OFFSET = 1000


def corpus(seed: int = 0) -> list[pathlib.Path]:
    raws = sorted(RAW.glob("*.zip"))
    random.Random(seed).shuffle(raws)
    return raws


ONSET_CACHE = REPO / "outputs" / "onset_cache"


def onsets_for(zp: pathlib.Path):
    """Cached audio onsets for a corpus map, or None.

    ★The alignment axis is the only one that loads the AUDIO, and it is the axis that
    caught the defect five others were blind to: a map can have human intervals, human
    hand roles, human flow and human difficulty *while sitting off the song's beat*.
    Without it `mapjudge` scores note attributes and their sequencing only, and is
    structurally unable to see anything music-relative -- which is D2, D3 and D4.
    """
    f = ONSET_CACHE / f"{zp.stem}.npz"
    if not f.exists():
        return None
    try:
        import numpy as _np
        return _np.load(f)["onsets"]
    except Exception:  # noqa: BLE001
        return None


def records_for(paths: list[pathlib.Path], want: int, label: str) -> list[dict]:
    """Metric records for up to `want` loadable maps, reporting what was skipped."""
    out, tried, skipped = [], 0, 0
    for zp in paths:
        if len(out) >= want:
            break
        tried += 1
        loaded = _load_human(zp)
        if loaded is None:
            skipped += 1
            continue
        notes, bpm = loaded
        # A map too short to have a rhythm cannot inform a rhythm percentile.
        if len(notes) < 50 or not (30.0 < bpm < 400.0):
            skipped += 1
            continue
        rec = mj.map_record(notes, bpm, onsets=onsets_for(zp))
        rec["_src"] = zp.name
        out.append(rec)
    n_audio = sum(1 for r in out if "onset_precision" in r)
    print(f"  {label}: {len(out)} maps ({skipped} skipped of {tried} tried), "
          f"{n_audio} with audio")
    return out


def _score_set(recs: list[dict], dists: dict, audio: bool) -> dict:
    """Conformal score quantiles for one scoring mode.

    ★**Two calibration sets, not one.** A map scored WITH the alignment axis is
    scored on 23 metrics; without it, 21. The conformal p-value compares a map's
    aggregate against calibration maps scored the same way, and a mean over 23
    metrics is simply not comparable to a mean over 21 -- reusing one set for both
    silently voids the guarantee that makes the verdict mean anything. So the
    audio set is calibrated on human maps that HAVE cached onsets, and the no-audio
    set on all of them.
    """
    ex = None if audio else {"alignment"}
    prov = {"distributions": dists,
            "calib_scores": {"mean": [], "topk": [], "max": []}}
    means, topks, maxes = [], [], []
    for r in recs:
        if audio and "onset_precision" not in r:
            continue
        res = mj.judge(r, prov, label=r.get("_src", "?"), exclude_axes=ex)
        if res.n_scored:
            means.append(res.s_mean)
            topks.append(res.s_topk)
            maxes.append(res.s_max)
    stage2 = {"distributions": dists,
              "calib_scores": {"mean": sorted(means), "topk": sorted(topks),
                               "max": sorted(maxes)}}
    pmins = []
    for r in recs:
        if audio and "onset_precision" not in r:
            continue
        res = mj.judge(r, stage2, label=r.get("_src", "?"), exclude_axes=ex)
        if res.n_scored and res.p_min == res.p_min:
            pmins.append(res.p_min)
    return {"mean": sorted(means), "topk": sorted(topks), "max": sorted(maxes),
            "pmin": sorted(pmins), "n": len(means)}


def build_reference(dist_recs: list[dict], calib_recs: list[dict]) -> dict:
    """Percentile distributions from DIST, conformal score quantiles from CALIB."""
    dists: dict[str, list[float]] = {}
    for name, _axis, _tail, _note in mj.CANDIDATES:
        vals = sorted(float(r[name]) for r in dist_recs
                      if name in r and r[name] == r[name])
        if len(vals) >= 50:
            dists[name] = vals
        else:
            print(f"  ⚠️ {name}: only {len(vals)} human values - NOT included")

    # Conformal scores, one set per scoring mode -- see `_score_set`.
    no_audio = _score_set(calib_recs, dists, audio=False)
    with_audio = _score_set(calib_recs, dists, audio=True)
    print(f"  calibration: {no_audio['n']} maps without the audio axis, "
          f"{with_audio['n']} with it")
    if with_audio["n"] < 100:
        print(f"  ⚠️ only {with_audio['n']} calibration maps carry cached onsets -- "
              f"the audio-mode p-value is coarse. Run "
              f"scripts/build_onset_cache.py over the CALIB span.")

    return {
        "_README": (
            "Per-map judge reference. distributions = sorted human values per metric "
            "(percentiles read against these); calib_scores = the extremeness scores "
            "of a DISJOINT human slice, which set the conformal p-value. Built by "
            "scripts/calibrate_mapjudge.py; audited by scripts/audit_mapjudge.py."
        ),
        "n_dist": len(dist_recs),
        "n_calib": len(calib_recs),
        "distributions": dists,
        "calib_scores": no_audio,
        "calib_scores_audio": with_audio,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=1200,
                    help="maps per slice (three slices are taken)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--alpha", type=float, default=0.10)
    ap.add_argument("--out", type=pathlib.Path, default=mj.QUANTILE_PATH)
    ap.add_argument("--heldout-json", type=pathlib.Path,
                    default=REPO / "outputs" / "mapjudge_heldout_records.json",
                    help="where to cache the held-out records for the audit")
    a = ap.parse_args()

    raws = corpus(a.seed)[CORPUS_OFFSET:]
    if len(raws) < 3 * a.n:
        print(f"⚠️ only {len(raws)} maps past offset {CORPUS_OFFSET}; "
              f"reducing n to {len(raws)//3}")
        a.n = len(raws) // 3

    # Take generously more paths than maps wanted, since some fail to load.
    span = int(a.n * 1.25) + 40
    print(f"corpus: {len(raws)} maps past index {CORPUS_OFFSET}; n={a.n} per slice")
    dist_recs = records_for(raws[0:span], a.n, "DIST   ")
    calib_recs = records_for(raws[span:2 * span], a.n, "CALIB  ")
    held_recs = records_for(raws[2 * span:3 * span], a.n, "HELDOUT")

    ref = build_reference(dist_recs, calib_recs)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(ref) + "\n")
    print(f"\nwrote {a.out}  ({len(ref['distributions'])} metrics, "
          f"{len(ref['calib_scores']['mean'])} calibration scores)")

    a.heldout_json.parent.mkdir(parents=True, exist_ok=True)
    a.heldout_json.write_text(json.dumps(held_recs) + "\n")
    print(f"wrote {a.heldout_json}  ({len(held_recs)} held-out records)")

    # The construction check: does a held-out human map pass at ~1-alpha?
    passed = unscored = 0
    for r in held_recs:
        res = mj.judge(r, ref, label=r.get("_src", "?"))
        v = res.verdict(a.alpha)
        if v == "UNSCORED":
            unscored += 1
        elif v == "PASS":
            passed += 1
    n = len(held_recs) - unscored
    if n:
        print(f"\nheld-out human accept rate: {passed}/{n} = {passed/n:.3f} "
              f"(expected ~{1-a.alpha:.2f} BY CONSTRUCTION - a check, not a result)")
    if unscored:
        print(f"  {unscored} unscored")
    print("\n★ The result is the CONTROL rejection rate: scripts/audit_mapjudge.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
