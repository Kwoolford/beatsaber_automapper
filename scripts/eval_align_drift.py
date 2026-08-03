#!/usr/bin/env python
"""K1 — does onset alignment degrade toward the END of a song?

Kyle, playing the tempo-fix maps: *"notes playing about 5 seconds after the song
ends"*. A8 reports ONE precision per map, so drift *within* a song averages away —
a song-level metric cannot see a song-shaped defect, the same blind-spot shape as
the audio-blind suite itself.

**Calibration comes first.** Humans may drift too, and a bar set against a perfect
1.0 would repeat the `h_dist` "more human than human" failure that saturated the
old suite. So this script measures the HUMAN corpus and reports its distribution;
it deliberately does not define a pass/fail bar. Setting one is a separate
decision, made against these numbers.

Three sub-metrics, because "it gets worse at the end" could mean three different
defects and they want different fixes:

  drift_q1_q5   precision in the first fifth minus the last fifth, over
                equal-COUNT quintiles (equal-time bins would put almost no notes
                in a quiet intro and make the estimate noise). Positive = worse
                at the end. This is "gradual decay".

  drift_slope   least-squares slope of precision against quintile index. Same
                sign convention flipped (negative = decaying). Distinguishes a
                steady decay from a cliff in the final fifth.

  tail_after    notes placed AFTER the last detected onset — Kyle's literal
                complaint, and a different defect from decay: the music has
                stopped and we are still emitting. Reported as a count and as a
                share of notes, plus how far past the last onset they run.

Usage:
    python scripts/eval_align_drift.py --human --n 60
    python scripts/eval_align_drift.py --maps outputs/eval_sweep_cache/'arm#s0__*.zip'
"""

from __future__ import annotations

import argparse
import glob
import json
import pathlib
import statistics
import sys

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from beatsaber_automapper.evaluation import alignment, scorecard  # noqa: E402

NQ = 5  # quintiles


def drift_metrics(beatmap, *, bpm: float, onsets) -> dict | None:
    """Within-song alignment drift for one map. None if unscorable."""
    if onsets is None or len(onsets) == 0:
        return None
    times = alignment.note_times(beatmap, bpm)
    if len(times) < NQ * alignment.MIN_NOTES:
        return None
    ref = np.sort(np.asarray(onsets, dtype=np.float64))
    last_onset = float(ref[-1])

    # Equal-COUNT quintiles: every bin gets the same number of notes, so a quiet
    # section cannot masquerade as a precision collapse.
    prec = []
    for q, chunk in enumerate(np.array_split(np.asarray(times, dtype=np.float64), NQ)):
        if len(chunk) == 0:
            return None
        matched, _ = alignment.match_offsets(list(chunk), ref)
        prec.append(matched / len(chunk))

    idx = np.arange(NQ, dtype=np.float64)
    slope = float(np.polyfit(idx, np.asarray(prec), 1)[0])

    after = [t for t in times if t > last_onset]
    return {
        "prec_q": [round(p, 4) for p in prec],
        "drift_q1_q5": round(prec[0] - prec[-1], 4),
        "drift_slope": round(slope, 4),
        "tail_after_n": len(after),
        "tail_after_frac": round(len(after) / len(times), 5),
        "tail_after_secs": round(max(after) - last_onset, 3) if after else 0.0,
        "n_notes": len(times),
    }


def _score_zips(paths: list[pathlib.Path], label: str) -> list[dict]:
    rows = []
    for p in paths:
        try:
            loaded = scorecard._load_any(p)
        except Exception:  # noqa: BLE001
            continue
        if not loaded:
            continue
        bm, bpm, onsets = loaded
        r = drift_metrics(bm, bpm=bpm, onsets=onsets)
        if r:
            r["map"] = p.name
            rows.append(r)
    print(f"{label}: {len(rows)} maps scored")
    return rows


def _dist(rows: list[dict], key: str) -> dict:
    v = sorted(r[key] for r in rows if r.get(key) is not None)
    if not v:
        return {}
    med = statistics.median(v)
    return {
        "median": round(med, 4),
        "mad": round(statistics.median([abs(x - med) for x in v]), 4),
        "p10": round(float(np.percentile(v, 10)), 4),
        "p90": round(float(np.percentile(v, 90)), 4),
        "n": len(v),
    }


def report(rows: list[dict], label: str) -> dict:
    if not rows:
        print(f"\n{label}: nothing scored")
        return {}
    print(f"\n=== {label} (n={len(rows)}) ===")
    qs = np.array([r["prec_q"] for r in rows], dtype=float)
    print("precision by quintile (median across maps):")
    print("   " + "".join(f"q{i+1}{'':>7s}" for i in range(NQ)))
    print("   " + "".join(f"{np.median(qs[:, i]):<9.3f}" for i in range(NQ)))
    out = {"n_maps": len(rows), "prec_q_median": [round(float(np.median(qs[:, i])), 4)
                                                  for i in range(NQ)]}
    print(f"\n{'metric':16s}{'median':>10s}{'mad':>9s}{'p10':>9s}{'p90':>9s}")
    for k in ("drift_q1_q5", "drift_slope", "tail_after_n", "tail_after_frac",
              "tail_after_secs"):
        d = _dist(rows, k)
        out[k] = d
        if d:
            print(f"{k:16s}{d['median']:>10.4f}{d['mad']:>9.4f}"
                  f"{d['p10']:>9.4f}{d['p90']:>9.4f}")
    frac_with_tail = sum(1 for r in rows if r["tail_after_n"] > 0) / len(rows)
    out["share_maps_with_tail_notes"] = round(frac_with_tail, 4)
    print(f"\nmaps with ANY note after the last onset: {frac_with_tail:.1%}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--human", action="store_true",
                    help="score the human corpus in data/raw (the calibration run)")
    ap.add_argument("--n", type=int, default=60, help="max human maps")
    ap.add_argument("--maps", default=None,
                    help="glob of generated map zips to score instead/as well")
    ap.add_argument("--label", default="generated")
    ap.add_argument("--json", default=None, help="write results here")
    a = ap.parse_args()

    out: dict = {}
    rows_by_label: dict[str, list[dict]] = {}
    if a.human:
        zips = sorted((REPO / "data" / "raw").glob("*.zip"))[: a.n]
        rows_by_label["human"] = _score_zips(zips, "human")
        out["human"] = report(rows_by_label["human"], "HUMAN corpus")
    if a.maps:
        paths = [pathlib.Path(p) for p in sorted(glob.glob(a.maps))]
        rows_by_label[a.label] = _score_zips(paths, a.label)
        out[a.label] = report(rows_by_label[a.label], a.label.upper())
        worst = sorted(rows_by_label[a.label],
                       key=lambda r: r["drift_q1_q5"], reverse=True)[:5]
        print("\nworst-drifting maps (name them; the defect is per-song):")
        print(f"  {'map':28s}{'q1':>7s}{'q5':>7s}{'drift':>8s}{'tailN':>7s}{'tail_s':>8s}")
        for r in worst:
            print(f"  {r['map'][:28]:28s}{r['prec_q'][0]:>7.3f}{r['prec_q'][-1]:>7.3f}"
                  f"{r['drift_q1_q5']:>8.3f}{r['tail_after_n']:>7d}"
                  f"{r['tail_after_secs']:>8.2f}")
        out[a.label]["worst"] = [r["map"] for r in worst]

    if "human" in out and a.label in out and out["human"] and out[a.label]:
        h, g = out["human"], out[a.label]
        print("\n=== HUMAN vs " + a.label.upper() + " ===")
        print(f"{'metric':16s}{'human med':>11s}{'ours med':>10s}"
              f"{'human p90':>11s}{'ours p90':>10s}{'med inside':>12s}")
        for k in ("drift_q1_q5", "drift_slope", "tail_after_frac", "tail_after_secs"):
            if not (h.get(k) and g.get(k)):
                continue
            inside = h[k]["p10"] <= g[k]["median"] <= h[k]["p90"]
            print(f"{k:16s}{h[k]['median']:>11.4f}{g[k]['median']:>10.4f}"
                  f"{h[k]['p90']:>11.4f}{g[k]['p90']:>10.4f}"
                  f"{('yes' if inside else 'NO'):>12s}")

        # THE discriminating statistic. Measured 2026-08-02: our cohort MEDIAN
        # drift sits inside the human range, yet 1f8d6 falls 1.000 -> 0.571 and
        # runs 11 notes 4.43 s past the last onset. A median cannot see a defect
        # that lives in a SUBSET of songs -- the same blind-spot shape as A8
        # reporting one precision per song, one level up.
        exceed = {}
        for k in ("drift_q1_q5", "tail_after_frac"):
            if not (h.get(k) and rows_by_label.get(a.label)):
                continue
            bar = h[k]["p90"]
            over = [r for r in rows_by_label[a.label] if r.get(k) is not None
                    and r[k] > bar]
            exceed[k] = {"human_p90": bar, "n_over": len(over),
                         "share": round(len(over) / len(rows_by_label[a.label]), 4)}
            print(f"\nmaps above the human p90 for {k} ({bar:.4f}): "
                  f"{len(over)}/{len(rows_by_label[a.label])} "
                  f"({exceed[k]['share']:.1%}) — human, by definition, 10%")
        out["exceedance"] = exceed
        print("\nRead: the median is the WRONG summary here. Rank by the share of")
        print("maps past the human p90, and name them — the defect is per-song.")
        print("A bar belongs at the human distribution, NOT at zero drift:")
        print("humans drift too (q1 0.950 -> q5 0.920), and scoring against a")
        print("perfect 1.0 is how h_dist saturated.")

    if a.json:
        pathlib.Path(a.json).write_text(json.dumps(out, indent=2))
        print(f"\nwrote {a.json}")


if __name__ == "__main__":
    main()
