#!/usr/bin/env python
"""W3 — is the difficulty in the RIGHT PLACES? Where does the map play hardest?

> *"Some parts of the song get really intense to play, even though they are not
> the main beat where you would expect the peak difficulty to be. Maybe we should
> revive some of the old work we did with assigning intensity to each part of the
> song like beat drop and what not and having higher nps for those sections."*

Kyle's claim is **not** that we peak too hard — it is that we peak in the **wrong
places**. Peak height alone cannot test that (ours 6.5 vs human 5.5 says nothing
about location), so this measures *what the music is doing where the map is
hardest*:

    peak_intensity   mean relative loudness of the windows in our TOP DECILE of
                     nps. High = we play hardest where the song is loudest.
    intensity_corr   Spearman(nps, intensity) across the whole song.
    peak_offset      |our loudest window - our densest window| in seconds.

Intensity is RMS **relative to the song's own peak** — Kyle's own framing
(*"maybe a sound compared to rest of song to easily draw intensity"*), and it
makes quiet songs comparable with loud ones.

**Both cohorts are scored on byte-identical audio.** The eval songset's .ogg files
were verified md5-identical to the audio inside the human zips (2026-08-02), so
ours and the human map are read against the same waveform and any difference is
the mapping, not the encode.

⚠️DIAGNOSTIC. Three metrics from 2026-08-03/04 have now failed the control battery
by rewarding regularity; check this one before it steers anything.

Usage:
    python scripts/eval_intensity_alloc.py --arm tf_trim_ev03_rc05
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

WIN, HOP = 4.0, 2.0


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    def rank(x):
        o = x.argsort(kind="mergesort")
        r = np.empty(len(x), dtype=float)
        r[o] = np.arange(len(x), dtype=float)
        for v in np.unique(x):
            m = x == v
            r[m] = r[m].mean()
        return r
    ra, rb = rank(a), rank(b)
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def intensity_curve(audio: pathlib.Path, centers: np.ndarray) -> np.ndarray | None:
    import librosa
    try:
        y, sr = librosa.load(str(audio), sr=22050, mono=True)
    except Exception:  # noqa: BLE001
        return None
    rms = librosa.feature.rms(y=y, hop_length=512)[0]
    t = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=512)
    v = np.interp(centers, t, rms)
    return v / max(float(v.max()), 1e-9)


def metrics(notes: np.ndarray, inten: np.ndarray, centers: np.ndarray) -> dict | None:
    if len(notes) < 100 or len(centers) < 20:
        return None
    nps = np.array([((notes >= c - WIN / 2) & (notes < c + WIN / 2)).sum() / WIN
                    for c in centers])
    if nps.max() <= 0:
        return None
    k = max(1, int(round(0.10 * len(nps))))
    top = np.argsort(nps)[-k:]
    return {"peak_intensity": round(float(inten[top].mean()), 4),
            "intensity_corr": round(_spearman(nps, inten), 4),
            "peak_offset": round(float(abs(centers[int(nps.argmax())]
                                           - centers[int(inten.argmax())])), 2),
            "peak_nps": round(float(nps.max()), 3)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", default="tf_trim_ev03_rc05")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()

    ours_rows, human_rows, per_song = [], [], []
    for og in sorted((REPO / "data" / "eval_songset").glob("*.ogg")):
        sid = og.stem
        gz = sorted(glob.glob(f"outputs/eval_sweep_cache/{a.arm}#s0__{sid}.zip"))
        hz = REPO / "data" / "raw" / f"{sid}.zip"
        if not gz or not hz.exists():
            continue
        G = scorecard._load_any(pathlib.Path(gz[0]))
        H = load_expert_only(hz)
        if not G or not H:
            continue
        go = np.sort(np.asarray(alignment.note_times(G[0], G[1]), dtype=float))
        ho = np.sort(np.asarray(alignment.note_times(H[0], float(H[1])), dtype=float))
        end = float(max(go.max(), ho.max()))
        centers = np.arange(WIN / 2, end - WIN / 2, HOP)
        inten = intensity_curve(og, centers)
        if inten is None:
            continue
        mo, mh = metrics(go, inten, centers), metrics(ho, inten, centers)
        if not mo or not mh:
            continue
        ours_rows.append(mo)
        human_rows.append(mh)
        per_song.append((sid, mo, mh))
        print(f"  {sid}: ours peak_int {mo['peak_intensity']:.3f} / "
              f"human {mh['peak_intensity']:.3f}")

    if not ours_rows:
        sys.exit("nothing scored")

    print(f"\n=== PAIRED, SAME AUDIO (n={len(ours_rows)} songs) ===")
    print(f"{'metric':18s}{'ours':>10s}{'human':>10s}{'paired delta':>15s}{'verdict':>14s}")
    out = {}
    for k in ("peak_intensity", "intensity_corr", "peak_offset", "peak_nps"):
        o = np.array([r[k] for r in ours_rows], dtype=float)
        h = np.array([r[k] for r in human_rows], dtype=float)
        m = ~(np.isnan(o) | np.isnan(h))
        d = o[m] - h[m]
        se = d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else float("nan")
        verdict = "resolvable" if abs(d.mean()) > 2 * se else "NO (noise)"
        out[k] = {"ours": round(float(np.median(o[m])), 4),
                  "human": round(float(np.median(h[m])), 4),
                  "delta": round(float(d.mean()), 4), "se": round(float(se), 4)}
        print(f"{k:18s}{np.median(o[m]):10.4f}{np.median(h[m]):10.4f}"
              f"{d.mean():+15.4f}{verdict:>14s}")

    print("\n=== READ ===")
    print("  Kyle's claim is about LOCATION, not height. It predicts our")
    print("  peak_intensity sits BELOW the human's -- we play hardest where the")
    print("  song is quieter than where they do -- and/or intensity_corr lower.")
    print("  If peak_intensity matches, our peaks are on the loud parts after all")
    print("  and the complaint is about something finer than section loudness")
    print("  (e.g. WHICH instrument peaks), which this cannot see.")

    if a.json:
        out["per_song"] = [{"song": s, "ours": o, "human": h} for s, o, h in per_song]
        pathlib.Path(a.json).write_text(json.dumps(out, indent=2))
        print(f"\nwrote {a.json}")


if __name__ == "__main__":
    main()
