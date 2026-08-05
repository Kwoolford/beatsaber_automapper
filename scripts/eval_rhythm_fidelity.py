#!/usr/bin/env python
"""M2 — IS THE MAP PLAYING *THIS* BAR'S RHYTHM, OR JUST *A* RHYTHM?

Kyle, 2026-08-04: *"Mostly just syncing to rhythm more and making significantly
more intelligent and intentional placements of notes."*

A8 already asks whether each note sits on an onset, and we score ~0.92 on it. That
question is answerable one note at a time, and a map can answer it perfectly while
playing a generic stream: onsets are dense, so *some* onset is nearly always within
tolerance. What A8 cannot ask is whether the map reproduces the FIGURE — the
particular arrangement of hits that makes this bar this bar.

**Method.** Quantise both the map and each separated stem onto the bar's 16 slots.
Score the agreement with **Cohen's kappa** (chance-corrected, so a dense map cannot
win by covering everything). Then subtract the same map bar scored against OTHER
bars' stem patterns:

    follow = kappa(map bar i, stem bar i) - mean_j kappa(map bar i, stem bar j)

★**This is why the gain and not the level.** A map that plays a constant eighth
pattern gets a good kappa against any drum bar that is mostly eighths — and gets
exactly the same score against every other bar, so its gain is **0**. The gain is
positive only when the map tracks *what changed*. A metronome scores 0 by
construction, which is the property every earlier rhythm metric in this project
failed to have.

Also reported, because "intentional" is more than per-bar accuracy:

    lead_stem        which stem the map tracks in each bar (argmax gain)
    lead_persistence P(same stem next bar | this bar), minus the persistence
                     expected from the map's own marginal stem preferences.
                     A mapper commits to an instrument through a passage and
                     switches at a boundary; drifting between stems bar to bar is
                     the "doesn't stick to one flow" complaint (K5), measured
                     RHYTHMICALLY -- which is the reading A4 (`eval_musical_role`)
                     could not test, since it attributes single notes, not figures.

Usage:
    python scripts/eval_rhythm_fidelity.py --arm tf_trim_ev03_rc05
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

import song_structure as ss  # noqa: E402
from beatsaber_automapper.evaluation import alignment, scorecard  # noqa: E402
from calibrate_playfeel import load_expert_only  # noqa: E402

STEMS = ("drums", "bass", "other", "vocals")
SLOTS = ss.SLOTS_PER_BAR
MIN_MAP_NOTES = 3
MIN_STEM_ONSETS = 3
NULL_SAMPLE = 24


def stem_onsets(song_id: str) -> dict[str, np.ndarray]:
    f = REPO / "outputs" / "stem_onset_cache" / f"{song_id}.npz"
    if not f.exists():
        return {}
    d = np.load(f, allow_pickle=True)
    return {s: np.sort(np.asarray(d[f"onsets_{s}"], dtype=float))
            for s in STEMS if f"onsets_{s}" in d.files}


def quantise(times: np.ndarray, B: ss.Bars) -> np.ndarray:
    """(n_bars, SLOTS) binary occupancy of `times` on the bar grid."""
    M = np.zeros((B.n, SLOTS), dtype=float)
    dur = B.dur
    t0 = B.edges[0]
    for t in times:
        if t < t0 or t >= B.edges[-1]:
            continue
        bi = int((t - t0) // dur)
        if not (0 <= bi < B.n):
            continue
        frac = (t - (t0 + bi * dur)) / dur
        si = int(round(frac * SLOTS))
        if si >= SLOTS:
            bi, si = min(bi + 1, B.n - 1), 0
        M[bi, si] = 1.0
    return M


def kappa(a: np.ndarray, b: np.ndarray) -> float:
    po = float((a == b).mean())
    pa, pb = float(a.mean()), float(b.mean())
    pe = pa * pb + (1 - pa) * (1 - pb)
    return 0.0 if pe >= 1.0 else (po - pe) / (1 - pe)


def follow_scores(Mmap: np.ndarray, Mstem: np.ndarray,
                  rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Per-bar (gain, valid) of the map against one stem."""
    n = Mmap.shape[0]
    gain = np.full(n, np.nan)
    ok_map = Mmap.sum(axis=1) >= MIN_MAP_NOTES
    ok_stem = Mstem.sum(axis=1) >= MIN_STEM_ONSETS
    idx = np.where(ok_stem)[0]
    if len(idx) < 8:
        return gain, np.zeros(n, dtype=bool)
    for i in range(n):
        if not (ok_map[i] and ok_stem[i]):
            continue
        true = kappa(Mmap[i], Mstem[i])
        pool = idx[idx != i]
        if len(pool) > NULL_SAMPLE:
            pool = rng.choice(pool, NULL_SAMPLE, replace=False)
        null = float(np.mean([kappa(Mmap[i], Mstem[j]) for j in pool]))
        gain[i] = true - null
    return gain, np.isfinite(gain)


def score_map(times: np.ndarray, B: ss.Bars, stems: dict,
              seed: int = 0) -> dict | None:
    rng = np.random.default_rng(seed)
    Mmap = quantise(times, B)
    if Mmap.sum() < 60:
        return None
    per_stem, gains = {}, {}
    for s in STEMS:
        if s not in stems:
            continue
        g, ok = follow_scores(Mmap, quantise(stems[s], B), rng)
        if ok.sum() < 12:
            continue
        gains[s] = g
        per_stem[f"follow_{s}"] = round(float(np.nanmean(g)), 4)
    if not gains:
        return None

    # which stem does each bar track? (argmax over stems present in that bar)
    keys = list(gains)
    G = np.vstack([gains[k] for k in keys])
    valid = np.isfinite(G).all(axis=0)
    lead = np.full(G.shape[1], -1)
    lead[valid] = np.argmax(G[:, valid], axis=0)
    seq = lead[lead >= 0]
    persistence = expect = None
    if len(seq) > 12:
        # consecutive-in-time pairs only
        pairs = [(lead[i], lead[i + 1]) for i in range(len(lead) - 1)
                 if lead[i] >= 0 and lead[i + 1] >= 0]
        if len(pairs) > 10:
            persistence = float(np.mean([a == b for a, b in pairs]))
            c = collections.Counter(seq)
            tot = sum(c.values())
            expect = float(sum((v / tot) ** 2 for v in c.values()))
    out = dict(per_stem)
    out["follow_best"] = round(max(per_stem.values()), 4)
    out["follow_mean"] = round(float(np.mean(list(per_stem.values()))), 4)
    if persistence is not None:
        out["lead_persistence"] = round(persistence, 4)
        out["lead_persistence_gain"] = round(persistence - expect, 4)
        out["lead_stem"] = keys[int(collections.Counter(seq).most_common(1)[0][0])]
    return out


def paired(rows, key) -> dict:
    """Delegates to the shared estimator (mean + median delta, se, resolvability)."""
    return ss.paired_delta(rows, key)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", default="tf_trim_ev03_rc05")
    ap.add_argument("--seed", default="s0")
    ap.add_argument("--json", default="")
    a = ap.parse_args()

    files = sorted(glob.glob(str(REPO / f"outputs/eval_sweep_cache/{a.arm}#{a.seed}__*.zip"))) \
        or sorted(glob.glob(str(REPO / f"outputs/eval_sweep_cache/{a.arm}__*.zip")))
    if not files:
        print(f"no cached maps for {a.arm}")
        return

    rows = []
    for f in files:
        song = pathlib.Path(f).stem.split("__")[-1]
        stems = stem_onsets(song)
        if len(stems) < 3:
            continue
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
        ours = score_map(t, B, stems)
        H = load_expert_only(REPO / "data" / "raw" / f"{song}.zip")
        human = None
        if H:
            ht = np.asarray(alignment.note_times(H[0], float(H[1])), dtype=float)
            human = score_map(ht, B, stems)
        if ours is None:
            continue
        rows.append({"song": song, "bars": B.n, "ours": ours, "human": human})
        hs = f"human {human['follow_mean']:+.4f} ({human.get('lead_stem','-')})" if human else "no human"
        print(f"  {song:22s} ours {ours['follow_mean']:+.4f} "
              f"({ours.get('lead_stem','-'):6s})  {hs}")

    print(f"\n{'='*92}\nM2 RHYTHM FIDELITY — arm {a.arm}, {len(rows)} songs "
          f"(all columns over the PAIRED subset)\n{'='*92}")
    keys = ["follow_drums", "follow_bass", "follow_other", "follow_vocals",
            "follow_mean", "follow_best", "lead_persistence", "lead_persistence_gain"]
    print(f"{'metric':<24} {'n':>3} {'ours':>9} {'human':>9} {'paired Δ':>10} "
          f"{'Δ median':>10} {'resolvable':>11}")
    summary = {}
    for k in keys:
        both = [r for r in rows if r.get("human")
                and r["ours"].get(k) is not None and r["human"].get(k) is not None]
        if len(both) < 6:
            continue
        o = [r["ours"][k] for r in both]
        h = [r["human"][k] for r in both]
        p = paired(rows, k)
        summary[k] = {"n": len(both), "ours": round(st.median(o), 4),
                      "human": round(st.median(h), 4), "paired": p}
        print(f"{k:<24} {len(both):>3d} {st.median(o):>+9.4f} {st.median(h):>+9.4f} "
              f"{p.get('delta', float('nan')):>+10.4f} "
              f"{p.get('delta_median', float('nan')):>+10.4f} "
              f"{('YES' if p.get('resolvable') else 'no'):>11}")

    ol = collections.Counter(r["ours"].get("lead_stem") for r in rows if r["ours"].get("lead_stem"))
    hl = collections.Counter(r["human"].get("lead_stem") for r in rows
                             if r.get("human") and r["human"].get("lead_stem"))
    print(f"\nlead stem, ours:  {dict(ol)}")
    print(f"lead stem, human: {dict(hl)}")
    agree = [r for r in rows if r.get("human") and r["ours"].get("lead_stem")
             and r["human"].get("lead_stem")]
    if agree:
        same = sum(r["ours"]["lead_stem"] == r["human"]["lead_stem"] for r in agree)
        print(f"same lead stem as the human map: {same}/{len(agree)}")

    print("\nHOW TO READ: `follow_*` is a GAIN over the same map scored against OTHER")
    print("bars of the same stem, so a map playing a generic constant figure scores 0")
    print("however well that figure fits. Positive = the map tracks what CHANGED.")

    if a.json:
        pathlib.Path(a.json).write_text(json.dumps(
            {"arm": a.arm, "rows": rows, "summary": summary}, indent=2))
        print(f"\nwrote {a.json}")


if __name__ == "__main__":
    main()
