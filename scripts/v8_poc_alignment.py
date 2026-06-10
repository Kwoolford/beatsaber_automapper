"""V8-0 PoC, part 2 — the decisive alignment gate, across in-dataset songs.

``v8_poc.py`` proved transcription *runs* and produces per-instrument structure
(validation c) on the held-out test song. But the test song has no human map, so it
can't answer the gate question that actually matters:

    Does the transcribed NoteEvent pool predict a HUMAN mapper's note times better
    than the signal V7 has to work with (a BPM-quantised grid + spectral-flux onsets)?

This script answers it on N randomly-sampled in-dataset songs (which DO have human
maps), reporting per song and aggregated:

  * ``cover_recall``  — fraction of human notes within ±tol of ANY candidate onset.
                        This is the candidate-pool ceiling: what Stage 1 can select
                        from. Higher = better representation.
  * ``f1``            — naive "place a note on every candidate" alignment F1 (the
                        literal "beats 0.41" metric; precision is expected to be low
                        because we keep ALL events — Stage 1's job is to thin them).
  * BPM-grid residual — fraction of human notes that do NOT fall on the 1/subdiv BPM
                        grid within ±tol. These are notes V7's representation CANNOT
                        place at all. If this is large, it is a hard argument for V8.

Compared candidate sets:
  * ``transcribed``   — basic-pitch (bass/vocals/other) ∪ multi-band drum onsets.
  * ``librosa_union`` — spectral-flux onsets on the drum + other stems (≈ what the
                        V7 eval treats as "real musical events").
  * ``bpm_grid``      — every 1/subdiv slot (what V7 Stage 1 is constrained to).

Usage:
    python scripts/v8_poc_alignment.py --n 12 --out outputs/v8_poc/alignment.json
"""

from __future__ import annotations

import argparse
import json
import logging
import pathlib
import random
import sys
import tempfile
import warnings
import zipfile

import numpy as np

warnings.filterwarnings("ignore")

REPO_ROOT = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("v8_align")

from v8_poc import (  # noqa: E402
    PITCHED_STEMS, DRUM_STEM, separate_stems, transcribe_pitched,
    transcribe_drums, librosa_onsets,
)
from eval_alignment import alignment_score  # noqa: E402

DIFF_PREFERENCE = ("ExpertPlus", "Expert", "Hard", "Normal", "Easy")


def _cover_recall(reference: np.ndarray, candidates: np.ndarray, tol: float) -> float:
    """Fraction of reference times within ``tol`` of any candidate (pool ceiling)."""
    if reference.size == 0:
        return 0.0
    if candidates.size == 0:
        return 0.0
    cand = np.sort(candidates)
    hits = 0
    for r in reference:
        idx = int(np.searchsorted(cand, r))
        near = []
        if idx < cand.size:
            near.append(cand[idx])
        if idx > 0:
            near.append(cand[idx - 1])
        if any(abs(c - r) <= tol for c in near):
            hits += 1
    return hits / reference.size


def _bpm_grid_times(bpm: float, duration: float, subdiv: int) -> np.ndarray:
    step = 60.0 / bpm / subdiv
    n = int(duration / step) + 1
    return np.arange(n) * step


def _human_times(map_dir: pathlib.Path, difficulty: str) -> tuple[np.ndarray, float]:
    from beatsaber_automapper.data.beatmap import parse_info_dat, parse_difficulty_dat

    info = parse_info_dat(next(map_dir.glob("[Ii]nfo.dat")))
    bpm = float(info.bpm)
    diff_path = None
    for f in sorted(map_dir.glob("*.dat")):
        if f.name.lower().startswith(difficulty.lower()):
            diff_path = f
            break
    if diff_path is None:
        return np.array([]), bpm
    bm = parse_difficulty_dat(diff_path)
    times = np.unique(np.round(
        np.array([n.beat * 60.0 / bpm for n in bm.color_notes], dtype=np.float64), 4
    ))
    return times, bpm


def _pick_difficulty(map_dir: pathlib.Path) -> str | None:
    names = [f.name for f in map_dir.glob("*.dat")]
    for pref in DIFF_PREFERENCE:
        for n in names:
            if n.lower().startswith(pref.lower()):
                return pref
    return None


def analyse_song(raw_zip: pathlib.Path, tol: float) -> dict | None:
    tmp = pathlib.Path(tempfile.mkdtemp(prefix="v8align_"))
    with zipfile.ZipFile(raw_zip) as zf:
        zf.extractall(tmp)
    egg = next(tmp.glob("*.egg"), None) or next(tmp.glob("*.ogg"), None)
    if egg is None:
        log.warning("%s: no audio", raw_zip.name)
        return None
    difficulty = _pick_difficulty(tmp)
    if difficulty is None:
        log.warning("%s: no difficulty .dat", raw_zip.name)
        return None
    human, bpm = _human_times(tmp, difficulty)
    if human.size < 50:
        log.warning("%s: too few human notes (%d)", raw_zip.name, human.size)
        return None

    stems, sr = separate_stems(egg)
    duration = max(len(v) for v in stems.values()) / sr

    # Transcribed pool.
    events = []
    for stem in PITCHED_STEMS:
        if stem in stems:
            tau = 0.10 if stem == "other" else 0.0
            merge = 40.0 if stem == "other" else 0.0
            events += transcribe_pitched(stems[stem], sr, stem, salience_tau=tau, chord_merge_ms=merge)
    if DRUM_STEM in stems:
        events += transcribe_drums(stems[DRUM_STEM], sr)
    trans = np.unique(np.round(np.array([e.onset_sec for e in events]), 4))

    # librosa union (drum + other).
    libro = np.unique(np.concatenate([
        librosa_onsets(stems.get("drums", stems["other"]), sr),
        librosa_onsets(stems.get("other", stems["drums"]), sr),
    ]))

    grid4 = _bpm_grid_times(bpm, duration, 4)

    def stats(cand):
        a = alignment_score(cand.tolist(), human.tolist(), tol)
        return {"n": int(cand.size), "cover_recall": round(_cover_recall(human, cand, tol), 4),
                "f1": round(a.f1, 4), "precision": round(a.precision, 4), "recall": round(a.recall, 4)}

    # BPM-grid residual: human notes NOT on the 1/4 grid (V7 can't place these).
    on_grid = _cover_recall(human, grid4, tol)

    res = {
        "song": raw_zip.stem, "bpm": bpm, "difficulty": difficulty,
        "duration": round(duration, 1), "n_human": int(human.size),
        "human_per_sec": round(human.size / duration, 2),
        "transcribed": stats(trans),
        "librosa_union": stats(libro),
        "bpm_grid_oncover": round(on_grid, 4),
        "bpm_grid_offgrid_residual": round(1.0 - on_grid, 4),
    }
    log.info("%s: human=%d (%.1f/s) | trans cover=%.3f f1=%.3f | libro cover=%.3f f1=%.3f | offgrid=%.3f",
             raw_zip.stem, human.size, human.size / duration,
             res["transcribed"]["cover_recall"], res["transcribed"]["f1"],
             res["librosa_union"]["cover_recall"], res["librosa_union"]["f1"],
             res["bpm_grid_offgrid_residual"])
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tolerance-ms", type=float, default=50.0)
    ap.add_argument("--out", type=pathlib.Path, default=REPO_ROOT / "outputs/v8_poc/alignment.json")
    ap.add_argument("--songs", nargs="*", help="explicit raw-zip stems to use")
    args = ap.parse_args()
    tol = args.tolerance_ms / 1000.0

    raw_dir = REPO_ROOT / "data/raw"
    if args.songs:
        zips = [raw_dir / f"{s}.zip" for s in args.songs]
    else:
        all_zips = sorted(raw_dir.glob("*.zip"))
        random.Random(args.seed).shuffle(all_zips)
        zips = all_zips[: args.n * 3]   # oversample; some get skipped

    results = []
    for z in zips:
        if len(results) >= args.n:
            break
        try:
            r = analyse_song(z, tol)
            if r is not None:
                results.append(r)
        except Exception as exc:
            log.warning("%s failed: %s", z.name, exc)

    if not results:
        log.error("No songs analysed.")
        return

    def agg(path):
        vals = [r[path[0]][path[1]] if len(path) == 2 else r[path[0]] for r in results]
        return round(float(np.mean(vals)), 4)

    summary = {
        "n_songs": len(results),
        "tolerance_ms": args.tolerance_ms,
        "mean_transcribed_cover_recall": agg(("transcribed", "cover_recall")),
        "mean_transcribed_f1": agg(("transcribed", "f1")),
        "mean_transcribed_precision": agg(("transcribed", "precision")),
        "mean_librosa_cover_recall": agg(("librosa_union", "cover_recall")),
        "mean_librosa_f1": agg(("librosa_union", "f1")),
        "mean_bpm_grid_offgrid_residual": agg(("bpm_grid_offgrid_residual",)),
        "baseline_v7_generated_f1": 0.41,
        "per_song": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2))

    print("\n" + "=" * 72)
    print(f"V8-0 alignment gate — {len(results)} in-dataset songs, ±{args.tolerance_ms:.0f} ms")
    print("=" * 72)
    print(f"  transcribed pool: cover_recall={summary['mean_transcribed_cover_recall']:.3f}  "
          f"f1={summary['mean_transcribed_f1']:.3f}  prec={summary['mean_transcribed_precision']:.3f}")
    print(f"  librosa union   : cover_recall={summary['mean_librosa_cover_recall']:.3f}  "
          f"f1={summary['mean_librosa_f1']:.3f}")
    print(f"  BPM-grid offgrid residual (human notes V7 CANNOT place): "
          f"{summary['mean_bpm_grid_offgrid_residual']:.3f}")
    print(f"  V7 generated-map baseline f1 (for reference): 0.41")
    cov = summary["mean_transcribed_cover_recall"]
    better = summary["mean_transcribed_cover_recall"] > summary["mean_librosa_cover_recall"]
    print("-" * 72)
    print(f"  (b) GATE: transcribed pool covers {cov:.0%} of human notes, "
          f"{'BEATS' if better else 'does NOT beat'} librosa; "
          f"off-grid residual {summary['mean_bpm_grid_offgrid_residual']:.0%}")
    print("=" * 72)
    log.info("Wrote %s", args.out)


if __name__ == "__main__":
    main()
