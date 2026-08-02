#!/usr/bin/env python
"""Calibrate the audio-alignment reference (eval suite v2, axis A8).

Same method as every other axis calibrator: median + MAD per metric over a human
cohort, written to `outputs/alignment_human_reference.json`. Two things are
specific to A8 and worth stating:

1. **A8 needs audio, so the cohort is limited to songs in the onset cache.** Run
   `scripts/build_onset_cache.py --from-raw N` first; without it this calibrates on
   the ~23 eval-songset songs only, which are also the songs A8 is used to judge.
2. **The bar is set from the human cohort's OWN gap, measured OUT OF SAMPLE.** The
   cohort is split in half: the reference comes from one half, the gap is measured
   on the other. Calibrating and testing on the same maps would report a gap near
   zero and produce a bar nothing could ever fail — the "more human than human"
   trap in a new costume. The printed recommendation is ~2x the held-out human gap,
   matching how A1/A2/A3/A7 bars were set.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import statistics
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from beatsaber_automapper.evaluation import alignment, scorecard  # noqa: E402

RAW = REPO / "data" / "raw"


def human_records(ids: list[str], verbose: bool = True) -> list[dict]:
    """Alignment metrics for the human Expert map of every cached song."""
    recs = []
    for sid in ids:
        zp = RAW / f"{sid}.zip"
        if not zp.exists():
            continue
        try:
            loaded = scorecard._load_any(zp)
        except Exception:  # noqa: BLE001
            loaded = None
        if not loaded:
            continue
        bm, bpm, onsets = loaded
        if onsets is None:
            continue
        rep = alignment.alignment_metrics(bm, bpm=bpm, onsets=onsets)
        m = rep.metrics
        if m.get("onset_precision") != m.get("onset_precision"):  # NaN
            continue
        m = dict(m)
        m["_song"] = sid
        recs.append(m)
        if verbose:
            print(f"  {sid:12s} notes={rep.n_notes:5d} onsets={rep.n_onsets:6d} "
                  f"prec={m['onset_precision']:.3f} mad={m['offset_mad_ms']:6.1f}ms "
                  f"lag={m['onset_lag_ms']:+7.1f}ms rec={m['onset_recall']:.3f}",
                  flush=True)
    return recs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--write", action="store_true",
                    help="write the reference file (otherwise dry-run)")
    a = ap.parse_args()

    ids = sorted(p.stem for p in scorecard.ONSET_CACHE.glob("*.npz"))
    if not ids:
        print("No cached onsets. Run: python scripts/build_onset_cache.py --from-raw 80")
        raise SystemExit(2)
    print(f"onset cache: {len(ids)} songs\n")
    recs = human_records(ids)
    print(f"\nhuman maps scored: {len(recs)}")
    if len(recs) < 10:
        print("FEWER THAN 10 HUMAN MAPS — the reference would be noise. Cache more "
              "songs with --from-raw before trusting anything below.")

    # Held-out gap: calibrate on half, score the other half as a cohort.
    half = len(recs) // 2
    ref_a = alignment.calibrate(recs[:half])
    if ref_a and half >= 5:
        cc = alignment.cohort_comparison(
            recs[half:], {k: (v["median"], v["mad"]) for k, v in ref_a.items()})
        s = cc["_summary"]
        gap, spread = s["alignment_gap"], s["min_spread"]
        print(f"\nHELD-OUT HUMAN COHORT ({len(recs) - half} maps vs a reference built "
              f"from the other {half}):")
        print(f"  alignment_gap = {gap:.3f}   min_spread = {spread:.3f}")
        for k in alignment.SEQUENCE_KEYS:
            if k in cc:
                print(f"    {k:18s} shift={cc[k]['shift']:+6.2f} "
                      f"spread={cc[k]['spread']:5.2f}")
        print(f"\n  => RECOMMENDED ALIGN_GAP_BAR = {2 * gap:.2f} "
              f"(2x the human cohort's own gap, same rule as A1/A2/A3/A7)")
    else:
        print("\nnot enough maps for a held-out gap estimate")

    ref = alignment.calibrate(recs)
    print(f"\n{'metric':18s}{'median':>10s}{'MAD':>10s}{'n':>6s}")
    for k, v in ref.items():
        print(f"{k:18s}{v['median']:10.3f}{v['mad']:10.3f}{v['n']:6d}")

    prec = [r["onset_precision"] for r in recs]
    print(f"\nonset_precision across human maps: min={min(prec):.3f} "
          f"median={statistics.median(prec):.3f} max={max(prec):.3f}")
    print("READ: the human median is the CEILING this metric can mean, not 1.0 — "
          "onset detection is imperfect.")

    if a.write:
        alignment.REFERENCE_PATH.write_text(json.dumps(ref, indent=2))
        print(f"\nwrote {alignment.REFERENCE_PATH}")
    else:
        print("\n(dry run — pass --write to save the reference)")


if __name__ == "__main__":
    main()
