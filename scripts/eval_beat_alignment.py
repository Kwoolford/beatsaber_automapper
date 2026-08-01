#!/usr/bin/env python
"""A8 prototype — does the map land ON THE MUSIC? (the suite's blind spot)

Why this exists (2026-08-01). Kyle played `hl014_ds055` and `b1_e17_ds055`, both of
which PASS all five v2 axes, and said: "painfully obvious the notes are off beat.
The consistent beat of the song is not where the notes are played." He was right and
the suite could not see it: **not one of the five scorecard axes ever loads the
audio.** `rhythm.py` scores note times against the DECLARED BPM GRID, never against
the music, so a map can have a perfectly human interval distribution, human hand
roles and human flow while sitting off the song's actual beat. That is the whole
explanation for five different configurations "passing" while sharing an obvious
audible defect.

`scripts/eval_alignment.py` already measures onset alignment, but it was never made
an axis AND its map loader silently returns 0 notes for HUMAN map zips (it reads
n_notes=0 for data/raw/*.zip), which is precisely why nobody ever ran the human
control that would have exposed the gap. This script uses the loader that does work
on both (`scorecard._load_any`) so generated and human maps are measured the SAME
way — the control is the entire point, exactly as it was for the spread bars.

Reports precision/recall/F1 of generated note times against detected audio onsets,
plus the signed timing offset distribution (are we late, early, or just scattered?),
which is what distinguishes "wrong notes" from "right notes, constant lag".
"""
from __future__ import annotations

import argparse
import pathlib
import statistics
import sys

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from beatsaber_automapper.evaluation import scorecard  # noqa: E402


def _note_times(path: pathlib.Path) -> tuple[list[float], float]:
    """Note onset times in SECONDS, via the loader that works on human maps too."""
    loaded = scorecard._load_any(path)
    if not loaded:
        return [], 0.0
    bm, bpm = loaded
    spb = 60.0 / bpm if bpm > 0 else 0.5
    times = sorted({round(n.beat * spb, 4) for n in bm.color_notes})
    return times, bpm


def _onsets(audio: pathlib.Path, use_stems: bool) -> np.ndarray:
    from eval_alignment import _detect_onsets_librosa, _separate_stems

    if use_stems:
        # _separate_stems returns {stem_name: mono array} at a fixed sr
        from beatsaber_automapper.data.stem_separator import DEMUCS_SR

        stems = _separate_stems(audio, DEMUCS_SR)
        allon: list[float] = []
        for _name, y in stems.items():
            allon.extend(_detect_onsets_librosa(np.asarray(y), DEMUCS_SR).tolist())
        return np.array(sorted(set(np.round(allon, 4))))
    import librosa

    y, sr = librosa.load(str(audio), sr=None, mono=True)
    return _detect_onsets_librosa(y, sr)


def _match(gen: list[float], ref: np.ndarray, tol: float) -> dict:
    """Greedy nearest matching within `tol` seconds; also the signed offsets."""
    if len(gen) == 0 or len(ref) == 0:
        return {"tp": 0, "precision": 0.0, "recall": 0.0, "f1": 0.0, "offsets": []}
    ref_sorted = np.sort(ref)
    used = np.zeros(len(ref_sorted), bool)
    tp, offsets = 0, []
    for t in gen:
        i = int(np.searchsorted(ref_sorted, t))
        best, bestd = -1, tol + 1
        for j in (i - 1, i, i + 1):
            if 0 <= j < len(ref_sorted) and not used[j]:
                d = abs(ref_sorted[j] - t)
                if d < bestd:
                    best, bestd = j, d
        if best >= 0 and bestd <= tol:
            used[best] = True
            tp += 1
            offsets.append(float(t - ref_sorted[best]))
    p = tp / len(gen)
    r = tp / len(ref_sorted)
    return {"tp": tp, "precision": p, "recall": r,
            "f1": (2 * p * r / (p + r)) if (p + r) else 0.0, "offsets": offsets}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--audio", required=True)
    ap.add_argument("--maps", nargs="+", required=True,
                    help="label=path pairs, e.g. human=data/raw/1f8a3.zip")
    ap.add_argument("--tolerance-ms", type=float, default=50.0)
    ap.add_argument("--no-stems", action="store_true")
    a = ap.parse_args()

    tol = a.tolerance_ms / 1000.0
    ref = _onsets(pathlib.Path(a.audio), not a.no_stems)
    print(f"audio: {a.audio}\ndetected onsets: {len(ref)}  tolerance: "
          f"{a.tolerance_ms:.0f}ms\n")
    print(f"{'map':22s}{'notes':>7s}{'F1':>8s}{'prec':>8s}{'rec':>8s}"
          f"{'offset_ms':>11s}{'|off|_ms':>10s}")
    print("-" * 74)
    for spec in a.maps:
        label, _, p = spec.partition("=")
        if not p:
            label, p = pathlib.Path(spec).stem[:20], spec
        times, _bpm = _note_times(pathlib.Path(p))
        if not times:
            print(f"{label:22s}  COULD NOT LOAD NOTES")
            continue
        m = _match(times, ref, tol)
        offs = m["offsets"]
        med = statistics.median(offs) * 1000 if offs else float("nan")
        mad = (statistics.median([abs(o - statistics.median(offs)) for o in offs]) * 1000
               if offs else float("nan"))
        print(f"{label:22s}{len(times):7d}{m['f1']:8.3f}{m['precision']:8.3f}"
              f"{m['recall']:8.3f}{med:11.1f}{mad:10.1f}")
    print("\nREAD: precision = share of OUR notes that land on a detected onset — the"
          "\ndirect measure of \"is the note where the music is\". Read every row"
          "\nagainst the HUMAN row: detected onsets are imperfect, so the human map's"
          "\nscore is the ceiling this metric can mean, NOT 1.0.")


if __name__ == "__main__":
    main()
