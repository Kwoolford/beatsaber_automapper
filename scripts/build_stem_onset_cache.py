#!/usr/bin/env python
"""Cache PER-STEM onset times — the prerequisite for the unbuilt A4 axis (K5).

Kyle, on 1f913: *"it doesn't seem to stick to one beat or one flow, it's kinda
trying to do the average of all of them."* And on 1f333 at 3:05, where a guitar
solo enters: *"a good mapper 100% would have played notes to accentuate this
change... the lead hand would have played a lot of the solo."*

Measured once and it did not reproduce — but with a blunt instrument. The
existing check argmaxes over raw per-stem onset COUNTS, and drums carry the most
onsets in almost every song, so both cohorts read "drum-led" almost by
construction. `docs/eval_suite_v2.md` planned an A4 "musical-role correctness"
axis weighting stems by *salience* rather than count; it was never built. This
script provides what it needs.

**Deliberately a separate cache.** `outputs/onset_cache/` stores only the union
and every measurement in this project is made against it. Re-running that
builder to add per-stem keys would overwrite those files, and if Demucs is not
bit-reproducible the entire evening's numbers would shift silently underneath
the record. This writes to `outputs/stem_onset_cache/` and touches nothing else.

As a free by-product it reports whether the union it derives matches the cached
one — a direct check on whether Demucs *is* reproducible, which nobody has
verified and which every cached onset silently assumes.

Usage:
    python scripts/build_stem_onset_cache.py --songset
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

OUT = REPO / "outputs" / "stem_onset_cache"
UNION_CACHE = REPO / "outputs" / "onset_cache"
SONGSET = REPO / "data" / "eval_songset"

DEMUCS_SEED = 0


def compute_per_stem(audio: pathlib.Path) -> dict[str, np.ndarray]:
    """stem name -> onset times (s). Raises if Demucs degraded to mix-only."""
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
    if len(stems) < 3:
        raise RuntimeError(
            f"Demucs fallback for {audio.name}: got {sorted(stems)}. The mix-only "
            "path yields a different onset set; fix Demucs rather than caching it."
        )
    return {name: _detect_onsets_librosa(np.asarray(y), DEMUCS_SR)
            for name, y in stems.items()}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--songset", action="store_true",
                    help="process data/eval_songset (the 24 scored songs)")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    if not a.songset:
        sys.exit("nothing to do — pass --songset")

    OUT.mkdir(parents=True, exist_ok=True)
    songs = sorted([p for p in SONGSET.iterdir()
                    if p.suffix.lower() in (".ogg", ".mp3", ".wav")])
    print(f"{len(songs)} songs -> {OUT}\n")

    agree = disagree = 0
    for i, song in enumerate(songs, 1):
        dest = OUT / f"{song.stem}.npz"
        if dest.exists() and not a.force:
            print(f"  [{i}/{len(songs)}] {song.stem[:24]:24s} cached")
            continue
        t0 = time.time()
        try:
            per = compute_per_stem(song)
        except Exception as e:  # noqa: BLE001
            print(f"  [{i}/{len(songs)}] {song.stem[:24]:24s} FAILED: {e}")
            continue

        payload = {f"onsets_{k}": v for k, v in per.items()}
        allon: list[float] = []
        for v in per.values():
            allon.extend(np.asarray(v).tolist())
        union = np.array(sorted(set(np.round(allon, 4))), dtype=np.float64)
        payload["onsets_union"] = union
        payload["stems"] = np.array(sorted(per), dtype=object)
        np.savez(dest, **{k: v for k, v in payload.items() if k != "stems"},
                 stems=np.array(sorted(per)))

        # Free reproducibility check against the cache everything else uses.
        note = ""
        ref = UNION_CACHE / f"{song.stem}.npz"
        if ref.exists():
            old = np.load(ref)["onsets"]
            if len(old) == len(union) and np.allclose(old, union, atol=1e-3):
                agree += 1
                note = "  union MATCHES cache"
            else:
                disagree += 1
                note = (f"  ⚠ union DIFFERS from cache "
                        f"({len(old)} vs {len(union)} onsets)")
        counts = " ".join(f"{k}={len(v)}" for k, v in sorted(per.items()))
        print(f"  [{i}/{len(songs)}] {song.stem[:24]:24s} {counts} "
              f"({time.time()-t0:.0f}s){note}")

    print(f"\nreproducibility vs outputs/onset_cache: {agree} match, {disagree} differ")
    if disagree:
        print("⚠ Demucs is NOT bit-reproducible on this machine. Every cached onset")
        print("  set is therefore a one-off measurement, and any metric compared")
        print("  across a cache rebuild is comparing two different rulers.")
    elif agree:
        print("✓ Demucs reproduces its onsets exactly — cached onset sets are stable")
        print("  and safe to rebuild. Nobody had verified this before.")


if __name__ == "__main__":
    main()
