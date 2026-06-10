"""Evaluate how well a generated Beat Saber map aligns to real musical onsets.

The training-time ``val_f1_avg_tol`` only measures agreement with the *mapper's*
choices in the dataset. It doesn't tell us whether a freshly generated map
actually lands its notes on real audio events. This script answers that.

Pipeline:
  1. Load the audio.
  2. Detect ground-truth onsets with librosa on:
       - the drum stem (sharp percussive events)
       - the full mix      (vocal/melody attacks)
  3. Load the generated beatmap (.dat from an unzipped folder, or a .zip).
  4. Convert each note's beat-coordinate to seconds (audio time).
  5. Compute precision/recall/F1 with a ±tolerance window
     (default ±50 ms — MIR-standard onset eval).
  6. Break down by detected section (uses
     ``detect_sections_energy_percentile`` so it lines up with the inference
     thresholds).

The numbers tell you, per section:
  - precision = of the notes the model placed, what fraction land on an audio onset
  - recall    = of the real audio onsets, what fraction did the model place a note on

A "lots of random horizontal notes" complaint will surface here as low precision
in verse/intro/outro sections — notes exist with nothing musical underneath them.

Usage:
    python scripts/eval_alignment.py \\
        --audio data/test_songs/SO\\ TIRED\\ ROCK\\ -\\ NUEKI.mp3 \\
        --map outputs/v7_section_aware.zip \\
        --difficulty ExpertPlus
"""

from __future__ import annotations

import argparse
import json
import logging
import pathlib
import sys
import tempfile
import zipfile
from dataclasses import asdict, dataclass
from typing import Iterable

import numpy as np
import torch

REPO_ROOT = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


@dataclass(slots=True)
class AlignmentResult:
    """Precision/recall/F1 at a single tolerance window."""

    n_generated: int
    n_reference: int
    tp: int
    precision: float
    recall: float
    f1: float


def _greedy_match(
    generated: np.ndarray,
    reference: np.ndarray,
    tolerance_sec: float,
) -> int:
    """Count true positives by greedy nearest-match within tolerance.

    Each generated time matches at most one reference time, and each reference
    matches at most one generated. Both arrays must be sorted ascending.
    """
    if generated.size == 0 or reference.size == 0:
        return 0

    tp = 0
    ref_used = np.zeros(reference.shape[0], dtype=bool)
    for g in generated:
        # Binary-search the nearest unused reference within tolerance.
        idx = int(np.searchsorted(reference, g))
        candidates: list[int] = []
        if idx < reference.size:
            candidates.append(idx)
        if idx > 0:
            candidates.append(idx - 1)
        best = -1
        best_d = float("inf")
        for c in candidates:
            if ref_used[c]:
                continue
            d = abs(reference[c] - g)
            if d <= tolerance_sec and d < best_d:
                best = c
                best_d = d
        if best >= 0:
            ref_used[best] = True
            tp += 1
    return tp


def alignment_score(
    generated_times: Iterable[float],
    reference_times: Iterable[float],
    tolerance_sec: float = 0.05,
) -> AlignmentResult:
    """Compute precision/recall/F1 for generated notes vs. reference onsets."""
    gen = np.sort(np.asarray(list(generated_times), dtype=np.float64))
    ref = np.sort(np.asarray(list(reference_times), dtype=np.float64))
    tp = _greedy_match(gen, ref, tolerance_sec)
    n_gen = int(gen.size)
    n_ref = int(ref.size)
    precision = tp / n_gen if n_gen > 0 else 0.0
    recall = tp / n_ref if n_ref > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return AlignmentResult(
        n_generated=n_gen,
        n_reference=n_ref,
        tp=tp,
        precision=precision,
        recall=recall,
        f1=f1,
    )


def _detect_onsets_librosa(y: np.ndarray, sr: int, hop_length: int = 512) -> np.ndarray:
    """Detect onset times (in seconds) with librosa's spectral-flux detector."""
    import librosa

    onsets = librosa.onset.onset_detect(
        y=y,
        sr=sr,
        hop_length=hop_length,
        units="time",
        backtrack=True,
    )
    return np.asarray(onsets, dtype=np.float64)


def _load_audio_mono(path: pathlib.Path, target_sr: int = 44100) -> tuple[np.ndarray, int]:
    """Load an audio file as mono float32 numpy array."""
    from beatsaber_automapper.data.audio import load_audio

    waveform, sr = load_audio(path, target_sr=target_sr)
    y = waveform.squeeze().numpy().astype(np.float32)
    return y, sr


def _separate_stems(path: pathlib.Path, sr: int) -> dict[str, np.ndarray]:
    """Run Demucs to get the drum stem; returns dict of stem_name → mono array.

    If Demucs is unavailable or the path doesn't separate cleanly, returns
    the original mix under both ``drums`` and ``other`` keys (graceful degrade).
    """
    try:
        from beatsaber_automapper.data.stem_separator import separate as demucs_separate, DEMUCS_SR
        from beatsaber_automapper.data.audio import load_audio

        device = "cuda" if torch.cuda.is_available() else "cpu"
        waveform, src_sr = load_audio(path, target_sr=DEMUCS_SR)
        stems = demucs_separate(waveform, src_sr, device=device)
        out: dict[str, np.ndarray] = {}
        for name, stem in stems.items():
            arr = stem.detach().cpu().numpy().astype(np.float32)
            # Demucs returns [channels, samples] — collapse to mono.
            if arr.ndim == 2:
                arr = arr.mean(axis=0)
            elif arr.ndim == 3:
                arr = arr.mean(axis=(0, 1))
            # Demucs returns at DEMUCS_SR; resample to requested sr if needed.
            if DEMUCS_SR != sr:
                import librosa
                arr = librosa.resample(arr, orig_sr=DEMUCS_SR, target_sr=sr)
            out[name] = arr
        return out
    except Exception as exc:
        log.warning("Demucs unavailable (%s) — falling back to mix-only onsets", exc)
        y, _ = _load_audio_mono(path, target_sr=sr)
        return {"drums": y, "other": y}


def _load_generated_beatmap(map_path: pathlib.Path, difficulty: str) -> tuple[list[tuple[float, int, int]], float]:
    """Load a generated map. Returns ((beat, x, color)-tuples, bpm).

    ``color`` is 0/1 (red/blue) so we can break out per-hand stats later.
    """
    from beatsaber_automapper.data.beatmap import parse_info_dat, parse_difficulty_dat

    if map_path.suffix == ".zip":
        tmp = tempfile.mkdtemp(prefix="alignment_eval_")
        with zipfile.ZipFile(map_path) as zf:
            zf.extractall(tmp)
        map_dir = pathlib.Path(tmp)
    else:
        map_dir = map_path

    info_path = next(map_dir.glob("Info.dat"), None) or next(map_dir.glob("info.dat"), None)
    if info_path is None:
        raise FileNotFoundError(f"No Info.dat in {map_dir}")
    info = parse_info_dat(info_path)
    if info is None:
        raise RuntimeError(f"Failed to parse {info_path}")
    bpm = float(info.bpm)

    # Find a difficulty file matching the requested name (case-insensitive).
    diff_files = sorted(map_dir.glob("*.dat"))
    diff_path = None
    for f in diff_files:
        if f.name.lower().startswith(difficulty.lower()):
            diff_path = f
            break
    if diff_path is None:
        # Fall back to ExpertPlus → Expert → first non-Info.dat.
        for cand in ("ExpertPlus", "Expert", "Hard", "Normal", "Easy"):
            for f in diff_files:
                if cand.lower() in f.name.lower():
                    diff_path = f
                    break
            if diff_path is not None:
                break
    if diff_path is None:
        for f in diff_files:
            if "info" not in f.name.lower():
                diff_path = f
                break
    if diff_path is None:
        raise FileNotFoundError(f"No difficulty .dat in {map_dir}")

    log.info("Using difficulty file: %s", diff_path.name)

    beatmap = parse_difficulty_dat(diff_path)
    if beatmap is None:
        raise RuntimeError(f"Failed to parse {diff_path}")

    notes: list[tuple[float, int, int]] = [
        (n.beat, n.x, n.color) for n in beatmap.color_notes
    ]
    return notes, bpm


def _beat_to_seconds(beat: float, bpm: float) -> float:
    return beat * 60.0 / bpm


def _per_section_breakdown(
    note_times: np.ndarray,
    ref_times: np.ndarray,
    sections: list[tuple[str, float, float]],
    tolerance_sec: float,
) -> list[tuple[str, float, float, AlignmentResult]]:
    out: list[tuple[str, float, float, AlignmentResult]] = []
    for sec_type, s, e in sections:
        gen_sub = note_times[(note_times >= s) & (note_times < e)]
        ref_sub = ref_times[(ref_times >= s) & (ref_times < e)]
        res = alignment_score(gen_sub.tolist(), ref_sub.tolist(), tolerance_sec)
        out.append((sec_type, s, e, res))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate generated-map vs. audio-onset alignment")
    parser.add_argument("--audio", required=True, type=pathlib.Path, help="Source audio file")
    parser.add_argument("--map",   required=True, type=pathlib.Path,
                        help="Generated map (.zip or extracted folder)")
    parser.add_argument("--difficulty", default="ExpertPlus")
    parser.add_argument("--tolerance-ms", type=float, default=50.0,
                        help="Onset matching tolerance in milliseconds (default 50)")
    parser.add_argument("--target-sr", type=int, default=44100)
    parser.add_argument("--no-stems", action="store_true",
                        help="Skip Demucs; run onset detection on the mix only")
    parser.add_argument("--json", type=pathlib.Path, default=None,
                        help="Optional output JSON path for the full report")
    args = parser.parse_args()

    tol_sec = args.tolerance_ms / 1000.0

    log.info("Loading audio: %s", args.audio)
    y, sr = _load_audio_mono(args.audio, target_sr=args.target_sr)
    duration = len(y) / sr
    log.info("Duration %.1fs at %d Hz", duration, sr)

    log.info("Detecting reference onsets …")
    if args.no_stems:
        drum_onsets = _detect_onsets_librosa(y, sr)
        melody_onsets = drum_onsets.copy()
        log.info("Mix-only onsets: %d", drum_onsets.size)
    else:
        stems = _separate_stems(args.audio, sr=sr)
        drum_onsets = _detect_onsets_librosa(stems.get("drums", y), sr)
        melody_onsets = _detect_onsets_librosa(stems.get("other", y), sr)
        log.info("Drum onsets: %d  Melody onsets: %d", drum_onsets.size, melody_onsets.size)

    combined_ref = np.unique(np.concatenate([drum_onsets, melody_onsets]))

    log.info("Loading generated map: %s", args.map)
    notes, bpm = _load_generated_beatmap(args.map, args.difficulty)
    log.info("Loaded %d notes at BPM %.1f", len(notes), bpm)

    note_times_all = np.array(sorted(_beat_to_seconds(b, bpm) for b, _x, _c in notes), dtype=np.float64)
    red_times      = np.array(sorted(_beat_to_seconds(b, bpm) for b, _x, c in notes if c == 0), dtype=np.float64)
    blue_times     = np.array(sorted(_beat_to_seconds(b, bpm) for b, _x, c in notes if c == 1), dtype=np.float64)

    overall_combined = alignment_score(note_times_all.tolist(), combined_ref.tolist(), tol_sec)
    overall_drums    = alignment_score(note_times_all.tolist(), drum_onsets.tolist(), tol_sec)
    overall_melody   = alignment_score(note_times_all.tolist(), melody_onsets.tolist(), tol_sec)
    red_drums        = alignment_score(red_times.tolist(),      drum_onsets.tolist(), tol_sec)
    blue_drums       = alignment_score(blue_times.tolist(),     drum_onsets.tolist(), tol_sec)

    # Section breakdown (uses the same detector inference uses, so this report
    # mirrors what threshold band each note was generated under).
    from beatsaber_automapper.data.audio import detect_sections_energy_percentile

    sections = detect_sections_energy_percentile(torch.from_numpy(y).unsqueeze(0), sample_rate=sr)
    per_section = _per_section_breakdown(note_times_all, combined_ref, sections, tol_sec)

    # ---- Report ----
    def _fmt(r: AlignmentResult) -> str:
        return f"gen={r.n_generated:4d} ref={r.n_reference:4d} tp={r.tp:4d} P={r.precision:.3f} R={r.recall:.3f} F1={r.f1:.3f}"

    print()
    print(f"=== Alignment report  tolerance=±{args.tolerance_ms:.0f} ms  duration={duration:.1f}s ===")
    print(f"BPM={bpm:.1f}  notes={len(notes)}  drum_onsets={drum_onsets.size}  melody_onsets={melody_onsets.size}")
    print()
    print(f"All notes vs. drums  : {_fmt(overall_drums)}")
    print(f"All notes vs. melody : {_fmt(overall_melody)}")
    print(f"All notes vs. union  : {_fmt(overall_combined)}")
    print(f"Red  notes vs. drums : {_fmt(red_drums)}")
    print(f"Blue notes vs. drums : {_fmt(blue_drums)}")
    print()
    print("Per-section (vs. union of drum+melody onsets):")
    for sec_type, s, e, res in per_section:
        nps = res.n_generated / max(0.1, e - s)
        print(f"  {sec_type:<7s} {s:6.1f}s..{e:6.1f}s ({e - s:5.1f}s)  nps={nps:5.2f}  {_fmt(res)}")

    if args.json:
        out = {
            "audio": str(args.audio),
            "map": str(args.map),
            "difficulty": args.difficulty,
            "tolerance_ms": args.tolerance_ms,
            "duration_sec": duration,
            "bpm": bpm,
            "n_notes": len(notes),
            "n_drum_onsets": int(drum_onsets.size),
            "n_melody_onsets": int(melody_onsets.size),
            "overall_drums":    asdict(overall_drums),
            "overall_melody":   asdict(overall_melody),
            "overall_combined": asdict(overall_combined),
            "red_vs_drums":     asdict(red_drums),
            "blue_vs_drums":    asdict(blue_drums),
            "sections": [
                {
                    "type": t, "start_sec": s, "end_sec": e,
                    "nps": r.n_generated / max(0.1, e - s),
                    **asdict(r),
                }
                for (t, s, e, r) in per_section
            ],
        }
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(out, indent=2))
        log.info("Wrote %s", args.json)


if __name__ == "__main__":
    main()
