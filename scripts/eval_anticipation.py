#!/usr/bin/env python
"""M6 — DOES THE MAP KNOW WHAT IS COMING?

A mapper builds *into* a drop. The notes thicken a bar before the energy arrives,
so the player is already moving when it lands. A generator that allocates notes from
the loudness it currently sees can only ever answer the drop **after** it has
happened, and the difference is one of the clearest reasons a technically correct
map still feels flat. Kyle's K4 (*"build-ups under-respond"*) is this, one bar late.

**Method.** Per bar, take the map's note count and the audio's energy, both
**detrended inside a local window** (16 bars) so the song's overall arc cannot drive
the answer. Cross-correlate them at lags of −4 … +4 bars and report

    anticipation = corr(map, audio shifted so the MAP LEADS by one bar)
                 - corr(map, audio shifted so the MAP LAGS by one bar)

Positive = the map thickens before the music does. Negative = it reacts. A map whose
density is unrelated to the audio scores 0, and so does a metronome, because the
statistic is a difference between two lags of the same pair.

`peak_lag` (the lag with the highest correlation) is reported beside it, in bars:
0 = the map answers the bar it is in, −1 = it answers a bar early.

⚠️This is NOT a claim about causality in the model — our generator sees the whole
song. It is a claim about the map: leading is a compositional choice available to
any offline generator, and we do not currently make it.

Usage:
    python scripts/eval_anticipation.py --arm tf_trim_ev03_rc05
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

import song_structure as ss  # noqa: E402
from beatsaber_automapper.evaluation import alignment, scorecard  # noqa: E402
from calibrate_playfeel import load_expert_only  # noqa: E402
from eval_motif_rhyme import notes_xydc  # noqa: E402

WINDOW = 16          # bars, for local detrending
LAGS = range(-4, 5)


def bar_counts(notes: list[tuple], B: ss.Bars) -> np.ndarray:
    c = np.zeros(B.n)
    t0, dur = B.edges[0], B.dur
    for (t, *_rest) in notes:
        if t < t0 or t >= B.edges[-1]:
            continue
        bi = int((t - t0) // dur)
        if 0 <= bi < B.n:
            c[bi] += 1
    return c


def bar_energy(song: str, B: ss.Bars) -> np.ndarray | None:
    A = ss.audio_features(song)
    if A is None:
        return None
    t = np.asarray(A["times"], dtype=float)
    e = np.asarray(A["onset_env"], dtype=float)
    out = np.zeros(B.n)
    for i in range(B.n):
        lo = int(np.searchsorted(t, B.edges[i]))
        hi = int(np.searchsorted(t, B.edges[i + 1]))
        out[i] = float(e[lo:hi].mean()) if hi > lo else 0.0
    return out


def detrend(v: np.ndarray, w: int = WINDOW) -> np.ndarray:
    """Subtract a centred moving mean — the local-differencing rule these axes keep
    needing: two signals with slow structure correlate without meaning anything."""
    n = len(v)
    out = np.zeros(n)
    for i in range(n):
        lo, hi = max(0, i - w // 2), min(n, i + w // 2 + 1)
        out[i] = v[i] - v[lo:hi].mean()
    sd = out.std()
    return out / sd if sd > 1e-9 else out


def score_map(notes: list[tuple], B: ss.Bars, energy: np.ndarray) -> dict | None:
    c = bar_counts(notes, B)
    if c.sum() < 100 or B.n < 32:
        return None
    m, e = detrend(c), detrend(energy)
    prof = {}
    for L in LAGS:
        # positive L: compare the map at bar i with the audio at bar i+L, i.e. the
        # map LEADS the audio by L bars.
        if L >= 0:
            a, b = m[:B.n - L], e[L:]
        else:
            a, b = m[-L:], e[:B.n + L]
        if len(a) < 16 or a.std() < 1e-9 or b.std() < 1e-9:
            continue
        prof[L] = float(np.corrcoef(a, b)[0, 1])
    if 1 not in prof or -1 not in prof or 0 not in prof:
        return None
    peak = max(prof, key=lambda k: prof[k])
    return {"anticipation": round(prof[1] - prof[-1], 4),
            "corr_at_0": round(prof[0], 4),
            "peak_lag": peak,
            "peak_corr": round(prof[peak], 4),
            "profile": {str(k): round(v, 4) for k, v in sorted(prof.items())}}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", default="tf_trim_ev03_rc05")
    ap.add_argument("--seed", default="s0")
    ap.add_argument("--json", default="")
    a = ap.parse_args()

    files = sorted(glob.glob(str(REPO / f"outputs/eval_sweep_cache/{a.arm}#{a.seed}__*.zip"))) \
        or sorted(glob.glob(str(REPO / f"outputs/eval_sweep_cache/{a.arm}__*.zip")))
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
        if B is None:
            continue
        E = bar_energy(song, B)
        if E is None:
            continue
        ours = score_map(notes_xydc(bm, bpm), B, E)
        H = load_expert_only(REPO / "data" / "raw" / f"{song}.zip")
        human = score_map(notes_xydc(H[0], float(H[1])), B, E) if H else None
        if ours is None:
            continue
        rows.append({"song": song, "ours": ours, "human": human})
        print(f"  {song:22s} anticipation ours {ours['anticipation']:+.3f} "
              f"(peak lag {ours['peak_lag']:+d})"
              + (f"   human {human['anticipation']:+.3f} (peak {human['peak_lag']:+d})"
                 if human else "   (no human)"))

    print(f"\n{'='*88}\nM6 ANTICIPATION — arm {a.arm}, {len(rows)} songs (PAIRED subset)\n{'='*88}")
    print(f"{'metric':<18} {'n':>3} {'ours':>9} {'human':>9} {'paired Δ':>10} "
          f"{'Δ med':>9} {'resolvable':>11}")
    summary = {}
    for k in ("anticipation", "corr_at_0", "peak_corr", "peak_lag"):
        both = [r for r in rows if r.get("human")
                and r["ours"].get(k) is not None and r["human"].get(k) is not None]
        if len(both) < 5:
            continue
        o = st.median([r["ours"][k] for r in both])
        h = st.median([r["human"][k] for r in both])
        p = ss.paired_delta(rows, k)
        summary[k] = {"n": len(both), "ours": o, "human": h, "paired": p}
        print(f"{k:<18} {len(both):>3d} {o:>+9.4f} {h:>+9.4f} "
              f"{p.get('delta', float('nan')):>+10.4f} "
              f"{p.get('delta_median', float('nan')):>+9.4f} "
              f"{('YES' if p.get('resolvable') else 'no'):>11}")

    # the whole lag profile, averaged — the shape is the finding, not one number
    print("\nLAG PROFILE (mean corr; positive lag = the MAP LEADS the audio)")
    for who in ("ours", "human"):
        prof = {}
        for r in rows:
            d = r.get(who)
            if not d:
                continue
            for k, v in d["profile"].items():
                prof.setdefault(int(k), []).append(v)
        if prof:
            line = "  ".join(f"{k:+d}:{np.mean(v):+.3f}" for k, v in sorted(prof.items()))
            print(f"  {who:<6} {line}")

    if a.json:
        pathlib.Path(a.json).write_text(json.dumps(
            {"arm": a.arm, "rows": rows, "summary": summary}, indent=2))
        print(f"\nwrote {a.json}")


if __name__ == "__main__":
    main()
