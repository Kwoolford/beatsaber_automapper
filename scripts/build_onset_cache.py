#!/usr/bin/env python
"""Precompute audio onsets for the eval songset — the input axis A8 needs.

Axis A8 (`evaluation/alignment.py`) is the first axis that scores notes against
the MUSIC rather than against the declared BPM grid. It therefore needs the one
thing no other axis has ever needed: the audio. Running Demucs + onset detection
inside the scorecard would make every scoring run minutes long and would couple
the suite to GPU availability, so onsets are computed ONCE per song here and
cached to `outputs/onset_cache/<song_id>.npz`.

The cache is keyed by song id (`1f767`), which is what both a human map
(`data/raw/1f767.zip`) and a generated one (`outputs/eval_sweep_cache/
<arm>__1f767.zip`) resolve to — so both sides of every comparison are measured
against byte-identical onsets. That shared reference is the point: the whole
reason this gap survived for months is that the human control was never run on
the same footing as ours.

DETECTION PATH (do not change casually — the human baseline moves with it):
onsets are the UNION over Demucs stems, ~2378 events on 1f767. The mix-only path
gives far fewer and different onsets, and every number in TODO.md is against the
stem path.

`eval_alignment._separate_stems` GRACEFULLY DEGRADES to mix-only when Demucs is
unavailable, which would silently poison the cache with a different detection
path. This script refuses that fallback: it verifies the real 4-stem output and
errors out otherwise. Silent degradation is exactly the failure mode that hid the
alignment gap in the first place (`eval_alignment.py`'s loader returned 0 notes
for human zips and nobody noticed).

SOURCES. Every Beat Saber map zip ships its own audio (`song.egg`), so the human
reference for A8 is NOT limited to the 23 songs in `data/eval_songset` — any map in
`data/raw` can be cached with `--from-raw`. That matters: with only the songset,
A8's human reference would come from ~23 maps that are also the maps it is used to
judge, and the control battery could not score alignment on its usual random human
sample at all.

Usage:
  python scripts/build_onset_cache.py                 # all eval-songset songs
  python scripts/build_onset_cache.py --songs 1f767   # one song
  python scripts/build_onset_cache.py --from-raw 80   # 80 human maps (audio from zip)
  python scripts/build_onset_cache.py --force         # recompute existing
"""
from __future__ import annotations

import argparse
import pathlib
import sys
import time

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

SONGSET = REPO / "data" / "eval_songset"
CACHE_DIR = REPO / "outputs" / "onset_cache"
AUDIO_EXTS = (".ogg", ".mp3", ".wav", ".egg")

# Demucs stem names we expect; anything less means the fallback fired.
EXPECTED_STEMS = {"drums", "bass", "other", "vocals"}


def audio_paths() -> dict[str, pathlib.Path]:
    """song_id -> audio file, for every song in the eval songset."""
    out: dict[str, pathlib.Path] = {}
    for p in sorted(SONGSET.iterdir()):
        if p.suffix.lower() in AUDIO_EXTS:
            out[p.stem] = p
    return out

DEMUCS_SEED = 0


def compute_onsets(audio: pathlib.Path) -> tuple[np.ndarray, dict[str, int]]:
    """Union of per-stem librosa onsets. Raises if the mix-only fallback fired."""
    from eval_alignment import _detect_onsets_librosa, _separate_stems

    from beatsaber_automapper.data.stem_separator import DEMUCS_SR
    from beatsaber_automapper.generation.seeding import seed_everything

    # Demucs applies RANDOM shift augmentation and averages the results, so
    # unseeded it returns different stems every call: measured 2026-08-03,
    # two runs on the same file in the same session gave 3649 vs 3711 union
    # onsets, and bass alone varied 1160 vs 1258 (+8%). Seeded, two runs are
    # bit-identical. Without this the onset ground truth every A8 bar is
    # measured against is a random draw.
    seed_everything(DEMUCS_SEED)
    stems = _separate_stems(audio, DEMUCS_SR)
    if not EXPECTED_STEMS.issubset(stems.keys()):
        raise RuntimeError(
            f"Demucs fallback detected for {audio.name}: got stems {sorted(stems)}, "
            f"expected {sorted(EXPECTED_STEMS)}. The mix-only path yields a DIFFERENT "
            "onset set and would silently invalidate the human baseline. Fix Demucs "
            "rather than caching this."
        )
    per_stem: dict[str, int] = {}
    allon: list[float] = []
    for name, y in stems.items():
        on = _detect_onsets_librosa(np.asarray(y), DEMUCS_SR)
        per_stem[name] = int(len(on))
        allon.extend(on.tolist())
    union = np.array(sorted(set(np.round(allon, 4))), dtype=np.float64)
    return union, per_stem


def raw_audio_paths(n: int, seed: int, skip: int) -> dict[str, pathlib.Path]:
    """song_id -> extracted audio, for `n` human maps sampled from data/raw.

    The audio is extracted to a temp dir; only the onsets are kept. `skip` mirrors
    the other calibrators' hold-out convention so the reference and the maps being
    judged are not the same maps.
    """
    import random
    import shutil
    import tempfile
    import zipfile

    raw = REPO / "data" / "raw"
    zips = sorted(raw.glob("*.zip"))
    random.Random(seed).shuffle(zips)
    out: dict[str, pathlib.Path] = {}
    tmp = pathlib.Path(tempfile.mkdtemp(prefix="onsetcache_"))
    _TEMPDIRS.append(tmp)
    for zp in zips[skip:]:
        if len(out) >= n:
            break
        sid = zp.stem
        if (CACHE_DIR / f"{sid}.npz").exists():
            continue
        try:
            with zipfile.ZipFile(zp) as zf:
                names = zf.namelist()
                # An Expert difficulty is required — a map we cannot score is not
                # worth a Demucs pass.
                if not any(n_.lower().split("/")[-1] == "expertstandard.dat" for n_ in names):
                    continue
                audio = next((n_ for n_ in names
                              if pathlib.Path(n_).suffix.lower() in (".egg", ".ogg", ".wav")), None)
                if audio is None:
                    continue
                dest = tmp / f"{sid}{pathlib.Path(audio).suffix.lower()}"
                dest.write_bytes(zf.read(audio))
        except Exception:  # noqa: BLE001
            shutil.rmtree(tmp / f"{sid}", ignore_errors=True)
            continue
        out[sid] = dest
    return out


_TEMPDIRS: list[pathlib.Path] = []


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--songs", nargs="*", help="song ids (default: all in the songset)")
    ap.add_argument("--from-raw", type=int, default=0,
                    help="also cache N human maps from data/raw (audio from the zip)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip", type=int, default=32,
                    help="hold-out offset into the shuffled data/raw list")
    ap.add_argument("--force", action="store_true", help="recompute cached songs")
    a = ap.parse_args()

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    paths = audio_paths()
    ids = a.songs or sorted(paths)
    if a.from_raw:
        extra = raw_audio_paths(a.from_raw, a.seed, a.skip)
        paths.update(extra)
        ids = list(ids) + sorted(extra)

    print(f"onset cache: {CACHE_DIR}")
    print(f"{'song':28s}{'onsets':>8s}{'dur_s':>8s}{'per_s':>8s}{'secs':>7s}  stems")
    print("-" * 88)
    n_done = n_skip = n_fail = 0
    for sid in ids:
        audio = paths.get(sid)
        if audio is None:
            print(f"{sid:28s}  NO AUDIO IN SONGSET")
            n_fail += 1
            continue
        dest = CACHE_DIR / f"{sid}.npz"
        if dest.exists() and not a.force:
            d = np.load(dest, allow_pickle=False)
            print(f"{sid[:27]:28s}{len(d['onsets']):8d}{float(d['duration']):8.1f}"
                  f"{len(d['onsets']) / max(float(d['duration']), 1e-6):8.2f}"
                  f"{'--':>7s}  (cached)")
            n_skip += 1
            continue
        t0 = time.time()
        try:
            union, per_stem = compute_onsets(audio)
        except Exception as exc:  # noqa: BLE001
            print(f"{sid[:27]:28s}  FAILED: {exc}")
            n_fail += 1
            continue
        import librosa

        dur = float(librosa.get_duration(path=str(audio)))
        np.savez(dest, onsets=union, duration=np.float64(dur),
                 song_id=sid, audio=str(audio.name), method="demucs_stem_union")
        el = time.time() - t0
        stems_s = " ".join(f"{k}={v}" for k, v in sorted(per_stem.items()))
        print(f"{sid[:27]:28s}{len(union):8d}{dur:8.1f}{len(union) / max(dur, 1e-6):8.2f}"
              f"{el:7.1f}  {stems_s}")
        n_done += 1

    import shutil
    for d in _TEMPDIRS:
        shutil.rmtree(d, ignore_errors=True)

    print(f"\ncomputed {n_done}, cached-already {n_skip}, failed {n_fail}")
    if n_fail:
        print("FAILED songs have no onsets — A8 will not score them. Do not treat a")
        print("missing song as a pass; that is the silent-zero failure all over again.")


if __name__ == "__main__":
    main()
