#!/usr/bin/env python
"""A4 — musical-role correctness. Does the map follow the section's lead instrument?

Kyle on 1f913: *"it doesn't seem to stick to one beat or one flow, it's kinda
trying to do the average of all of them."* On 1f333 at 3:05, where a guitar solo
enters: *"a good mapper 100% would have played notes to accentuate this
change... most if not all notes would have changed to be this guitar solo."*

**Why the first measurement was not believable.** It argmaxed over raw per-stem
onset COUNTS. Drums carry the most onsets in nearly every song, so both cohorts
read "drum-led" almost by construction and the null was a property of the
instrument, not of the maps. `docs/eval_suite_v2.md` planned this axis with
stems weighted by *salience*; it was never built.

**The fix for that bluntness**: a section's lead is the stem most active
*relative to its own song-wide baseline*, not the one with the most onsets. A
guitar that doubles its own activity leads a section even while drums still
out-count it — which is exactly the 3:05 case Kyle described.

Two metrics:

  role_follow       share of a section's notes landing on the LEAD stem's onsets.
                    High = the map plays the thing carrying the section.

  role_commitment   how concentrated a section's note-to-stem matches are on ONE
                    stem, as 1 - normalised entropy over stems. Kyle's "average
                    of all of them" is precisely LOW commitment, so this is the
                    metric his complaint predicts should separate us from humans.

Both are reported per cohort, against the human corpus. No bar is proposed here:
calibration is a separate decision, and a bar set before looking is how `h_dist`
saturated.

Usage:
    python scripts/eval_musical_role.py --maps 'outputs/eval_sweep_cache/arm#s0__*.zip'
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import pathlib
import statistics as st
import sys

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from beatsaber_automapper.evaluation import alignment, scorecard  # noqa: E402

STEM_CACHE = REPO / "outputs" / "stem_onset_cache"
SECTION_SEC = 8.0
TOL = 0.05


def stems_for(song_id: str) -> dict[str, np.ndarray] | None:
    f = STEM_CACHE / f"{song_id}.npz"
    if not f.exists():
        return None
    d = np.load(f, allow_pickle=True)
    out = {k[len("onsets_"):]: d[k] for k in d.files
           if k.startswith("onsets_") and k != "onsets_union"}
    return out or None


def role_metrics(beatmap, bpm: float, stems: dict[str, np.ndarray]) -> dict | None:
    """Per-map role_follow and role_commitment."""
    times = np.asarray(alignment.note_times(beatmap, bpm), dtype=np.float64)
    if len(times) < 100:
        return None
    names = sorted(stems)
    dur = float(max(times.max(), max((s.max() for s in stems.values() if len(s)),
                                     default=0.0)))
    if dur <= 0:
        return None
    edges = np.arange(0.0, dur + SECTION_SEC, SECTION_SEC)
    nsec = len(edges) - 1
    if nsec < 3:
        return None

    # Per-stem onset counts per section, and each stem's own song-wide mean.
    counts = {n: np.histogram(stems[n], bins=edges)[0].astype(float) for n in names}
    base = {n: (counts[n].mean() if counts[n].mean() > 0 else np.nan) for n in names}

    follows, commits = [], []
    for si in range(nsec):
        lo, hi = edges[si], edges[si + 1]
        sec_notes = times[(times >= lo) & (times < hi)]
        if len(sec_notes) < 5:
            continue
        # SALIENCE: activity relative to that stem's own baseline, so a stem
        # cannot lead merely by being busy everywhere.
        rel = {n: (counts[n][si] / base[n]) if base[n] == base[n] and base[n] > 0
               else 0.0 for n in names}
        lead = max(names, key=lambda n: rel[n])
        if rel[lead] <= 0:
            continue

        # WINNER-TAKE-ALL attribution. Counting every stem within TOL made the
        # metric blind: measured 2026-08-03, 68% of matched notes match MORE THAN
        # ONE stem (2.12 of 4 on average), because stems co-occur -- a note "on
        # the drums" is usually also on the bass and the other stem. Sharing the
        # credit therefore drove the entropy toward uniform for EVERY cohort and
        # the metric could not distinguish following one instrument from
        # following the mix. Each note now goes to its single NEAREST stem onset.
        dists = {}
        for n in names:
            s = np.sort(stems[n])
            if len(s) == 0:
                dists[n] = np.full(len(sec_notes), np.inf)
                continue
            idx = np.searchsorted(s, sec_notes).clip(1, len(s) - 1)
            dists[n] = np.minimum(np.abs(sec_notes - s[idx - 1]),
                                  np.abs(sec_notes - s[idx]))
        D = np.vstack([dists[n] for n in names])
        best = D.argmin(axis=0)
        ok = D.min(axis=0) <= TOL
        matched = {n: int(((best == i) & ok).sum()) for i, n in enumerate(names)}
        tot = sum(matched.values())
        if tot == 0:
            continue
        follows.append(matched[lead] / len(sec_notes))
        p = np.array([matched[n] / tot for n in names], dtype=float)
        p = p[p > 0]
        ent = -(p * np.log(p)).sum() / math.log(len(names)) if len(names) > 1 else 0.0
        commits.append(1.0 - ent)

    if len(follows) < 3:
        return None
    return {"role_follow": round(float(np.mean(follows)), 4),
            "role_commitment": round(float(np.mean(commits)), 4),
            "n_sections": len(follows)}


def _score(paths, label: str) -> list[dict]:
    rows = []
    for p in paths:
        sid = scorecard.song_id(pathlib.Path(p))
        stems = stems_for(sid)
        if not stems:
            continue
        try:
            L = scorecard._load_any(pathlib.Path(p))
        except Exception:  # noqa: BLE001
            continue
        if not L:
            continue
        r = role_metrics(L[0], L[1], stems)
        if r:
            r["map"] = pathlib.Path(p).name
            rows.append(r)
    print(f"{label}: {len(rows)} maps scored")
    return rows


def report(rows: list[dict], label: str) -> dict:
    if not rows:
        return {}
    out = {}
    print(f"\n=== {label} (n={len(rows)}) ===")
    for k in ("role_follow", "role_commitment"):
        v = [r[k] for r in rows]
        out[k] = {"median": round(st.median(v), 4),
                  "p10": round(float(np.percentile(v, 10)), 4),
                  "p90": round(float(np.percentile(v, 90)), 4)}
        print(f"  {k:18s} median {st.median(v):.4f}   "
              f"p10 {np.percentile(v, 10):.4f}   p90 {np.percentile(v, 90):.4f}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--maps", action="append", default=[],
                    help="glob of maps to score; repeatable")
    ap.add_argument("--label", action="append", default=[])
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    if not a.maps:
        sys.exit("pass at least one --maps glob")

    out = {}
    for i, g in enumerate(a.maps):
        lab = a.label[i] if i < len(a.label) else f"cohort{i}"
        out[lab] = report(_score(sorted(glob.glob(g)), lab), lab.upper())

    if len(out) >= 2:
        labs = list(out)
        print("\n=== COMPARISON ===")
        print(f"{'metric':20s}" + "".join(f"{l[:14]:>16s}" for l in labs))
        for k in ("role_follow", "role_commitment"):
            row = "".join(f"{out[l][k]['median']:>16.4f}" if out[l] else f"{'--':>16s}"
                          for l in labs)
            print(f"{k:20s}{row}")
        print("\nKyle's claim predicts OUR role_commitment sits BELOW the human one")
        print("('trying to do the average of all of them'). If it does not, the")
        print("claim is still not refuted -- but the burden moves to finding a")
        print("sharper instrument, because his ear has been ahead of the metrics")
        print("twice already.")

    if a.json:
        pathlib.Path(a.json).write_text(json.dumps(out, indent=2))
        print(f"\nwrote {a.json}")


if __name__ == "__main__":
    main()
