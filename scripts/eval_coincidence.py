#!/usr/bin/env python
"""W1 — the COINCIDENCE DETECTOR. Kyle's hypothesis, stated as a measurement.

> *"Maybe demucs should flag specific alignments when key instruments hit the
> same beat consistently and that could be a big flag for when a note should get
> placed."*  — Kyle, 2026-08-03

He described the failure concretely on SO TIRED ROCK @ 0:46: *"a booming guitar
plays 3 notes in sync with the booming bass"* and we map **nothing** — *"this
epic coordination of instruments colliding doesn't exist."*

That claim has two halves and they must be tested separately:

  (A) IS IT TRUE OF MUSIC?  Do human mappers place notes preferentially where
      several stems hit together, above their rate on a lone-stem onset?
      If humans do NOT respond to coincidence, the idea is wrong and no lever
      should be built on it.

  (B) IS IT A GAP FOR US?  Given (A), do WE respond less than humans do?
      Only if both hold is this the cause of his complaint.

**Method.** Per song, the seeded per-stem onset cache (`outputs/stem_onset_cache/`,
274 songs, DEMUCS_SEED=0) is clustered into EVENTS with a `--link` tolerance; an
event's coincidence order `k` is the number of DISTINCT stems present in it
(1..4). A map "responds" to an event if it has a note within `--tol` of it.

**The statistic that matters is the LIFT, not the hit rate.** A denser map hits
more events at every k, so raw hit rates are not comparable across cohorts:

    lift = P(respond | k >= 3) / P(respond | k == 1)

Lift is a within-map ratio, so a map's overall density cancels. AUROC of k as a
predictor of "this map put a note here" is reported alongside as a second
density-invariant view.

⚠️**This is a DIAGNOSTIC, not yet a scoring axis.** Before any number here is
allowed to steer the generator it must pass `scripts/audit_eval_suite.py` — the
standing rule in the /todo loop, and the rule `h_dist` was lost for breaking.

⚠️Not circular w.r.t. `BEAT_ONSET_EVIDENCE`: that lever consumes **librosa mix**
onsets, while this reads **Demucs per-stem** onsets. Related but not the same
signal. Say so in any write-up rather than assuming independence.

Usage:
    python scripts/eval_coincidence.py \
        --gen 'outputs/eval_sweep_cache/tf_trim_ev03_rc05#s*__*.zip' \
        --human-n 273 --json outputs/coincidence_2026-08-03.json
"""

from __future__ import annotations

import argparse
import collections
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
STEMS = ("bass", "drums", "other", "vocals")


def events_for(song_id: str, link: float) -> tuple[np.ndarray, np.ndarray] | None:
    """Cluster per-stem onsets into events. Returns (times, k) sorted by time.

    An event is a maximal run of onsets each within `link` of the previous one.
    `k` counts DISTINCT stems in the run, so a drum roll of five hits inside the
    window is still k=1 -- coincidence means different instruments, not many
    onsets.
    """
    f = STEM_CACHE / f"{song_id}.npz"
    if not f.exists():
        return None
    d = np.load(f, allow_pickle=True)
    pairs = []
    for s in STEMS:
        key = f"onsets_{s}"
        if key in d.files:
            pairs.extend((float(t), s) for t in d[key])
    if len(pairs) < 50:
        return None
    pairs.sort()

    times, ks = [], []
    cur_t, cur_s = [pairs[0][0]], {pairs[0][1]}
    for t, s in pairs[1:]:
        if t - cur_t[-1] <= link:
            cur_t.append(t)
            cur_s.add(s)
        else:
            times.append(float(np.mean(cur_t)))
            ks.append(len(cur_s))
            cur_t, cur_s = [t], {s}
    times.append(float(np.mean(cur_t)))
    ks.append(len(cur_s))
    return np.asarray(times), np.asarray(ks, dtype=int)


def _auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Rank-based AUROC, ties averaged. Density-invariant by construction."""
    pos, neg = int(labels.sum()), int((~labels.astype(bool)).sum())
    if pos == 0 or neg == 0:
        return float("nan")
    order = scores.argsort(kind="mergesort")
    ranks = np.empty(len(scores), dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1, dtype=float)
    # average ranks within tied score groups
    for v in np.unique(scores):
        m = scores == v
        ranks[m] = ranks[m].mean()
    return float((ranks[labels.astype(bool)].sum() - pos * (pos + 1) / 2) / (pos * neg))


def coincidence_metrics(beatmap, bpm: float, ev: tuple[np.ndarray, np.ndarray],
                        tol: float) -> dict | None:
    times, ks = ev
    notes = np.asarray(alignment.note_times(beatmap, bpm), dtype=np.float64)
    if len(notes) < 100 or len(times) < 100:
        return None
    notes.sort()
    idx = np.searchsorted(notes, times).clip(1, len(notes) - 1)
    dist = np.minimum(np.abs(times - notes[idx - 1]), np.abs(times - notes[idx]))
    hit = dist <= tol

    out: dict = {"n_events": int(len(times))}
    for k in (1, 2, 3, 4):
        m = ks == k
        out[f"p_hit_k{k}"] = round(float(hit[m].mean()), 4) if m.sum() >= 20 else None
        out[f"n_k{k}"] = int(m.sum())

    lo, hi = ks == 1, ks >= 3
    if lo.sum() < 20 or hi.sum() < 20 or hit[lo].mean() <= 0:
        return None
    out["lift"] = round(float(hit[hi].mean() / hit[lo].mean()), 4)
    out["auroc"] = round(_auroc(hit, ks.astype(float)), 4)
    out["hit_rate"] = round(float(hit.mean()), 4)
    return out


def scan(paths, loader, label: str, link: float, tol: float) -> list[dict]:
    rows = []
    for p in paths:
        pp = pathlib.Path(p)
        sid = scorecard.song_id(pp)
        ev = events_for(sid, link)
        if ev is None:
            continue
        try:
            L = loader(pp)
        except Exception:  # noqa: BLE001
            continue
        if not L:
            continue
        r = coincidence_metrics(L[0], L[1], ev, tol)
        if r:
            r["map"], r["song"] = pp.name, sid
            rows.append(r)
    print(f"{label}: {len(rows)} maps scored")
    return rows


def report(rows: list[dict], label: str) -> dict:
    if not rows:
        return {}
    print(f"\n=== {label} (n={len(rows)}) ===")
    out: dict = {"n": len(rows)}
    print("  response rate by coincidence order k (share of events carrying a note):")
    for k in (1, 2, 3, 4):
        v = [r[f"p_hit_k{k}"] for r in rows if r.get(f"p_hit_k{k}") is not None]
        if not v:
            print(f"    k={k}: (too few events)")
            continue
        out[f"p_hit_k{k}"] = round(st.median(v), 4)
        print(f"    k={k}  median {st.median(v):.4f}   (n maps {len(v)})")
    for k in ("lift", "auroc", "hit_rate"):
        v = [r[k] for r in rows]
        out[k] = {"median": round(st.median(v), 4),
                  "p10": round(float(np.percentile(v, 10)), 4),
                  "p90": round(float(np.percentile(v, 90)), 4)}
        print(f"  {k:10s} median {st.median(v):7.4f}   "
              f"p10 {np.percentile(v, 10):7.4f}   p90 {np.percentile(v, 90):7.4f}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gen", default="outputs/eval_sweep_cache/tf_trim_ev03_rc05#s*__*.zip",
                    help="glob of OUR maps")
    ap.add_argument("--human-n", type=int, default=273)
    ap.add_argument("--link", type=float, default=0.030,
                    help="seconds; onsets within this of each other are one event")
    ap.add_argument("--tol", type=float, default=0.050,
                    help="seconds; a note this close to an event counts as a response")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()

    cached = {p.stem for p in STEM_CACHE.glob("*.npz")}
    human = [p for p in sorted(REPO.glob("data/raw/*.zip"))
             if p.stem in cached][:a.human_n]

    out = {"params": {"link": a.link, "tol": a.tol}}
    g_rows = scan(sorted(glob.glob(a.gen)), scorecard._load_any, "ours", a.link, a.tol)
    # NEVER scorecard._load_any for the human cohort -- it prefers ExpertPlus.
    h_rows = scan(human, load_expert_only, "human", a.link, a.tol)
    out["ours"] = report(g_rows, "OURS")
    out["human"] = report(h_rows, "HUMAN (strict Expert)")

    if out["ours"] and out["human"]:
        o, h = out["ours"], out["human"]
        print("\n=== VERDICT LOGIC ===")
        print("(A) Is coincidence real in HUMAN maps?")
        print(f"    human lift = {h['lift']['median']:.3f}  (p10 {h['lift']['p10']:.3f})")
        print("    lift > 1.15 and p10 > 1.0  =>  humans DO respond to instrument")
        print("    coincidence; Kyle's hypothesis is sound and worth a lever.")
        print("    lift ~= 1.0  =>  the idea is WRONG; do not build on it.")
        print("\n(B) Do WE respond less than humans?")
        print(f"    ours  lift = {o['lift']['median']:.3f}   human {h['lift']['median']:.3f}")
        print(f"    ours auroc = {o['auroc']['median']:.3f}   human {h['auroc']['median']:.3f}")
        print("    ours materially below human on BOTH  =>  W1 has a measured")
        print("    cause and the decode lever (weight the budget by k) is next.")
        print("    ours >= human  =>  W1 is NOT a coincidence-blindness problem;")
        print("    the gap is elsewhere (probably Stage-1 instrument projection,")
        print("    i.e. Track B) -- report that and do NOT build the lever.")
        print("\n    Neither branch is a pass/fail: no bar is calibrated here, and")
        print("    a diagnostic must clear audit_eval_suite.py before it steers.")

    if a.json:
        out["ours_rows"], out["human_rows"] = g_rows, h_rows
        pathlib.Path(a.json).write_text(json.dumps(out, indent=2))
        print(f"\nwrote {a.json}")


if __name__ == "__main__":
    main()
