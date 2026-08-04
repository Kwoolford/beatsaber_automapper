#!/usr/bin/env python
"""W1 control — is coincidence order `k` just a proxy for LOUDNESS?

`eval_coincidence.py` shows human mappers respond to multi-stem coincidences far
more than to lone-stem onsets (0.41 -> 0.85 as k goes 1 -> 4). Before any lever
is built on that, one alternative explanation has to be killed:

    **A loud downbeat has every instrument hitting it.** If k is merely a
    restatement of onset STRENGTH, then "flag instrument coincidences" adds
    nothing that a loudness prior does not already give -- and
    `BEAT_ONSET_EVIDENCE` is already a loudness-ish prior.

**Design.** For each song, take the same events and k, and sample librosa's
onset-strength envelope at each event time. Split events into within-song
strength DECILES, then compare P(note | k>=3) against P(note | k==1)
**inside each decile**, i.e. among events of comparable loudness.

  - If the contrast survives conditioning, k carries information loudness does
    not, and weighting the note budget by k is a real, new lever.
  - If it collapses to ~1.0, k is a loudness proxy. Say so and do not build it.

Reported as `lift_raw` (unconditioned, what the main script measures) beside
`lift_cond` (strength-conditioned). The DROP between them is the answer.

Usage:
    python scripts/eval_coincidence_control.py --n 60 --json outputs/coinc_control.json
"""

from __future__ import annotations

import argparse
import json
import pathlib
import statistics as st
import sys

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from beatsaber_automapper.evaluation import alignment  # noqa: E402
from calibrate_playfeel import load_expert_only  # noqa: E402
from eval_coincidence import events_for  # noqa: E402

N_DECILES = 10


def strength_at(audio: pathlib.Path, times: np.ndarray) -> np.ndarray | None:
    import librosa
    try:
        y, sr = librosa.load(str(audio), sr=22050, mono=True)
    except Exception:  # noqa: BLE001
        return None
    if len(y) < sr:
        return None
    env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=512)
    t = librosa.frames_to_time(np.arange(len(env)), sr=sr, hop_length=512)
    return np.interp(times, t, env)


def analyse(times, ks, hit, strength) -> dict | None:
    """lift_raw vs lift_cond (strength-decile-conditioned)."""
    lo, hi = ks == 1, ks >= 3
    if lo.sum() < 30 or hi.sum() < 30 or hit[lo].mean() <= 0:
        return None
    lift_raw = float(hit[hi].mean() / hit[lo].mean())

    # Within-song strength deciles, so "comparable loudness" is defined per song.
    edges = np.quantile(strength, np.linspace(0, 1, N_DECILES + 1))
    edges[-1] += 1e-9
    num, den, used = [], [], 0
    for i in range(N_DECILES):
        m = (strength >= edges[i]) & (strength < edges[i + 1])
        a, b = m & hi, m & lo
        if a.sum() < 8 or b.sum() < 8 or hit[b].mean() <= 0:
            continue
        num.append(float(hit[a].mean()))
        den.append(float(hit[b].mean()))
        used += 1
    if used < 4:
        return None
    # Pool the deciles rather than averaging ratios: a ratio of small counts is
    # unstable, and averaging ratios over-weights sparse deciles.
    lift_cond = float(np.mean(num) / np.mean(den))
    return {"lift_raw": round(lift_raw, 4), "lift_cond": round(lift_cond, 4),
            "deciles_used": used,
            "corr_k_strength": round(float(np.corrcoef(ks, strength)[0, 1]), 4)}


def main() -> None:
    import shutil
    import tempfile
    import zipfile

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--link", type=float, default=0.030)
    ap.add_argument("--tol", type=float, default=0.050)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()

    cached = {p.stem for p in (REPO / "outputs" / "stem_onset_cache").glob("*.npz")}
    zips = [p for p in sorted((REPO / "data" / "raw").glob("*.zip")) if p.stem in cached]

    rows = []
    tmp = pathlib.Path(tempfile.mkdtemp(prefix="coinc_ctrl_"))
    try:
        for zp in zips:
            if len(rows) >= a.n:
                break
            ev = events_for(zp.stem, a.link)
            if ev is None:
                continue
            L = load_expert_only(zp)
            if not L:
                continue
            times, ks = ev
            notes = np.asarray(alignment.note_times(L[0], L[1]), dtype=np.float64)
            if len(notes) < 100:
                continue
            try:
                with zipfile.ZipFile(zp) as zf:
                    an = next((n for n in zf.namelist()
                               if pathlib.Path(n).suffix.lower() in (".egg", ".ogg", ".wav")), None)
                    if an is None:
                        continue
                    dest = tmp / f"{zp.stem}{pathlib.Path(an).suffix.lower()}"
                    dest.write_bytes(zf.read(an))
            except Exception:  # noqa: BLE001
                continue
            s = strength_at(dest, times)
            dest.unlink(missing_ok=True)
            if s is None:
                continue
            notes.sort()
            idx = np.searchsorted(notes, times).clip(1, len(notes) - 1)
            dist = np.minimum(np.abs(times - notes[idx - 1]), np.abs(times - notes[idx]))
            r = analyse(times, ks, dist <= a.tol, s)
            if r:
                r["song"] = zp.stem
                rows.append(r)
                print(f"  {zp.stem}: raw {r['lift_raw']:.3f} -> cond {r['lift_cond']:.3f}"
                      f"   corr(k,strength) {r['corr_k_strength']:+.3f}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    if not rows:
        sys.exit("no songs scored")

    raw = [r["lift_raw"] for r in rows]
    cond = [r["lift_cond"] for r in rows]
    corr = [r["corr_k_strength"] for r in rows]
    print(f"\n=== HUMAN COHORT (n={len(rows)}) ===")
    print(f"  lift_raw        median {st.median(raw):.4f}   p10 {np.percentile(raw,10):.4f}"
          f"   p90 {np.percentile(raw,90):.4f}")
    print(f"  lift_cond       median {st.median(cond):.4f}   p10 {np.percentile(cond,10):.4f}"
          f"   p90 {np.percentile(cond,90):.4f}")
    print(f"  corr(k,strength) median {st.median(corr):+.4f}")
    retained = (st.median(cond) - 1.0) / max(st.median(raw) - 1.0, 1e-9)
    print(f"\n  RETAINED after conditioning on loudness: {retained*100:.0f}% of the raw lift")
    print("\n=== READ ===")
    print("  retained > 60%  =>  k is NOT a loudness proxy; instrument coincidence")
    print("                      is its own signal and the lever is worth building.")
    print("  retained < 25%  =>  k is mostly loudness; BEAT_ONSET_EVIDENCE already")
    print("                      captures it. Do not build a second loudness prior.")
    print("  in between      =>  partial; report as such and prefer the cheaper lever.")

    if a.json:
        pathlib.Path(a.json).write_text(json.dumps(
            {"n": len(rows), "lift_raw_median": st.median(raw),
             "lift_cond_median": st.median(cond), "retained": retained,
             "corr_k_strength_median": st.median(corr), "rows": rows}, indent=2))
        print(f"\nwrote {a.json}")


if __name__ == "__main__":
    main()
