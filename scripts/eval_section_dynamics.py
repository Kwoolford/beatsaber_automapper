#!/usr/bin/env python3
"""Section-transition dynamics eval — Track A-3 (2026-07-28).

Kyle's specific complaint on SO TIRED ROCK: at the ~15s drop, RMS energy
roughly DOUBLES (0.20 -> 0.78) but generated note density FALLS (5-7/s ->
4-6/s). The existing DoD metric (`eval_density_corr.py`, whole-song Spearman of
density vs a fixed onset reference) already passes ~0.40-0.8 on this
architecture (see the v7instr B-0 re-eval) -- so a global level correlation can
be fine while a *specific transition* still goes the wrong way. This script
measures TRANSITIONS directly: does note density rise when audio energy rises,
window to window?

Two views:
  1. `transition_corr` -- Spearman/Pearson of consecutive-window ENERGY DELTA
     vs DENSITY DELTA across the whole song (the general form of Kyle's
     complaint: do density changes track energy changes, not just levels).
  2. `biggest_jump` -- the single largest energy increase in the song (the
     "drop" in miniature) and whether density rose or fell there, which is
     exactly the observation Kyle made by ear.

CLI mirrors eval_density_corr.py: pass one --audio/--map pair, or --cache-arm
to score every song in outputs/eval_sweep_cache/ for one arm at once.
"""
from __future__ import annotations

import argparse
import json
import logging
import pathlib
import sys

import numpy as np

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

from eval_alignment import _load_generated_beatmap, _beat_to_seconds  # noqa: E402
from eval_density_corr import _bin_counts, _spearman, _pearson  # noqa: E402

log = logging.getLogger("eval_section_dynamics")


def _rms_windows(audio_path: pathlib.Path, duration: float, win_sec: float,
                  sr: int = 22050) -> np.ndarray:
    """RMS energy per uniform win_sec window, aligned with _bin_counts binning."""
    import librosa
    y, _ = librosa.load(str(audio_path), sr=sr, mono=True)
    hop = int(win_sec * sr)
    n_bins = max(1, int(np.ceil(duration / win_sec)))
    rms = np.zeros(n_bins, dtype=np.float64)
    for i in range(n_bins):
        seg = y[i * hop:(i + 1) * hop]
        rms[i] = float(np.sqrt(np.mean(seg ** 2))) if len(seg) else 0.0
    return rms


def transition_corr(energy: np.ndarray, density: np.ndarray) -> dict:
    n = min(len(energy), len(density))
    e, d = energy[:n], density[:n]
    de = np.diff(e)
    dd = np.diff(d)
    return {
        "n_transitions": int(len(de)),
        "delta_spearman": _spearman(de, dd) if len(de) > 1 else float("nan"),
        "delta_pearson": _pearson(de, dd) if len(de) > 1 else float("nan"),
    }


def biggest_jump(energy: np.ndarray, density: np.ndarray, win_sec: float) -> dict:
    n = min(len(energy), len(density))
    e, d = energy[:n], density[:n]
    de = np.diff(e)
    if len(de) == 0:
        return {}
    i = int(np.argmax(de))
    return {
        "window_index": i,
        "time_sec": round(i * win_sec, 1),
        "energy_before": round(float(e[i]), 3),
        "energy_after": round(float(e[i + 1]), 3),
        "density_before": round(float(d[i]), 3),
        "density_after": round(float(d[i + 1]), 3),
        "density_rose": bool(d[i + 1] > d[i]),
    }


def _score_one(audio: pathlib.Path, map_zip: pathlib.Path, win_sec: float,
               difficulty: str = "Expert") -> dict | None:
    notes, bpm = _load_generated_beatmap(map_zip, difficulty)
    gen_times = np.array(sorted(_beat_to_seconds(b, bpm) for b, _x, _c in notes),
                         dtype=np.float64)
    if len(gen_times) == 0:
        return None
    import librosa
    duration = float(librosa.get_duration(path=str(audio)))
    energy = _rms_windows(audio, duration, win_sec)
    density = _bin_counts(gen_times, duration, win_sec)
    tc = transition_corr(energy, density)
    bj = biggest_jump(energy, density, win_sec)
    return {"song": audio.stem, "map": map_zip.name, **tc, "biggest_jump": bj}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--audio", type=pathlib.Path)
    p.add_argument("--map", type=pathlib.Path, help="generated .zip or dir")
    p.add_argument("--cache-arm", help="score every outputs/eval_sweep_cache/<arm>__*.zip "
                                       "against data/eval_songset/")
    p.add_argument("--difficulty", default="Expert")
    p.add_argument("--window-sec", type=float, default=2.0)
    p.add_argument("--json", type=pathlib.Path, default=None)
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(name)s %(levelname)s: %(message)s")

    results = []
    if args.cache_arm:
        cache = REPO / "outputs" / "eval_sweep_cache"
        songset = REPO / "data" / "eval_songset"
        for zp in sorted(cache.glob(f"{args.cache_arm}__*.zip")):
            song_stem = zp.name[len(args.cache_arm) + 2:-4]
            audio = next((p for p in songset.glob(f"{song_stem}.*")
                         if p.suffix.lower() in (".ogg", ".mp3")), None)
            if audio is None:
                continue
            r = _score_one(audio, zp, args.window_sec, args.difficulty)
            if r:
                results.append(r)
                print(f"  [{r['song']}] delta_spearman={r['delta_spearman']:+.3f} "
                      f"biggest_jump@{r['biggest_jump'].get('time_sec')}s "
                      f"density {r['biggest_jump'].get('density_before')}"
                      f"->{r['biggest_jump'].get('density_after')} "
                      f"rose={r['biggest_jump'].get('density_rose')}")
    else:
        if not args.audio or not args.map:
            p.error("either --cache-arm, or both --audio and --map")
        r = _score_one(args.audio, args.map, args.window_sec, args.difficulty)
        if r:
            results.append(r)
            print(json.dumps(r, indent=2))

    if results:
        ds = [r["delta_spearman"] for r in results if r["delta_spearman"] == r["delta_spearman"]]
        rose = [r["biggest_jump"]["density_rose"] for r in results if r.get("biggest_jump")]
        print(f"\n=== {args.cache_arm or 'single'}: {len(results)} songs ===")
        print(f"  mean delta_spearman = {np.mean(ds):+.3f}")
        print(f"  biggest-jump density rose: {sum(rose)}/{len(rose)}")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(results, indent=2))
        log.info("wrote %s", args.json)


if __name__ == "__main__":
    main()
