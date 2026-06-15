#!/usr/bin/env python3
"""Contour-follow eval — the TASK-3 (Stage-2 pitch-contour) DoD metric.

Question it answers: do a generated map's *swing directions* track the song's
melodic contour? TASK 3 feeds a per-slot pitch contour (lead_pitch / lead_dpitch
/ bass_pitch — cols 7:10 of instr_beat_features) into the Stage-2 layout encoder
so the decoder can bias swing DIRECTION to follow the line: when the lead rises
the swing should go up-ish, when it falls it should go down-ish. The North-Star
failure this targets is "diagonal swings for sport" that ignore the music.

Method: compute the per-slot lead Δpitch from the audio (same transcription pass
the model trains on), look up each note's Δpitch at its slot, and measure how
often the note's vertical swing component agrees with the sign of Δpitch:

    cut direction vertical component
        up / up-left / up-right       (0,4,5) -> +1
        down / down-left / down-right (1,6,7) -> -1
        left / right / dot            (2,3,8) ->  0  (no vertical info → skipped)

We only score notes where (a) |Δpitch| exceeds a deadband (a real melodic step,
not transcription jitter) and (b) the swing has a vertical component. The
contour-follow RATE is the fraction of those notes whose swing sign matches the
Δpitch sign. Baseline for an unconditioned model is ~0.5 (chance). DoD: arm A
(--use-contour) beats its no-contour control by a clear margin, ideally toward
≥0.60. Reuses the map/audio plumbing from eval_alignment / eval_density_corr.
"""
from __future__ import annotations

import argparse
import json
import logging
import pathlib
import sys

import numpy as np
import torch

REPO_ROOT = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from eval_alignment import _load_generated_beatmap  # same scripts/ dir

from beatsaber_automapper.data.audio import load_audio
from beatsaber_automapper.data.beat_grid import BEAT_SUBDIV
from beatsaber_automapper.data.instrument_features import (
    INSTR_FEATURE_NAMES,
    compute_instrument_features,
)
from beatsaber_automapper.data.beatmap import parse_difficulty_dat, parse_info_dat  # noqa: F401

log = logging.getLogger("eval_contour_follow")

_LEAD_DPITCH_COL = INSTR_FEATURE_NAMES.index("lead_dpitch")  # col 8

# Vertical component of each Beat Saber cut direction (0-8).
_VERT = {0: +1, 4: +1, 5: +1,    # up, up-left, up-right
         1: -1, 6: -1, 7: -1,    # down, down-left, down-right
         2: 0, 3: 0, 8: 0}       # left, right, dot → no vertical info


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--audio", type=pathlib.Path, required=True)
    p.add_argument("--map", type=pathlib.Path, required=True, help="generated .zip or dir")
    p.add_argument("--difficulty", default="Expert")
    p.add_argument("--deadband", type=float, default=0.05,
                   help="Min |lead_dpitch| (tanh-scaled) to count a note as a real "
                        "melodic step. Below this the contour is flat/jittery → skip.")
    p.add_argument("--device", default=None)
    p.add_argument("--json", type=pathlib.Path, default=None)
    p.add_argument("--label", default=None, help="tag echoed into the JSON/stdout")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(name)s %(levelname)s: %(message)s")

    # --- notes with direction + bpm ---
    notes, bpm = _load_generated_beatmap(args.map, args.difficulty)  # (beat,x,color)
    # _load_generated_beatmap drops y/direction; re-parse for the full note records.
    full_notes = _load_notes_with_direction(args.map, args.difficulty)
    log.info("Loaded %d notes (bpm=%.2f)", len(full_notes), bpm)

    # --- per-slot lead Δpitch from the audio (same transcription pass) ---
    from beatsaber_automapper.data.stem_separator import DEMUCS_SR
    waveform, src_sr = load_audio(args.audio, target_sr=DEMUCS_SR)
    duration_sec = waveform.shape[-1] / src_sr
    n_beats = duration_sec * bpm / 60.0
    n_slots = int(np.ceil(n_beats * BEAT_SUBDIV)) + 1
    log.info("Computing contour: dur=%.1fs n_slots=%d (Demucs→transcription) …",
             duration_sec, n_slots)
    instr = compute_instrument_features(
        waveform, src_sr, bpm, n_slots, subdiv=BEAT_SUBDIV, device=args.device,
    )                                              # [n_slots, 10]
    dpitch = instr[:, _LEAD_DPITCH_COL].float().numpy()

    # --- score ---
    n_scored = 0
    n_match = 0
    n_skip_flat = 0
    n_skip_horiz = 0
    for beat, _x, _y, _color, direction in full_notes:
        vert = _VERT.get(int(direction), 0)
        if vert == 0:
            n_skip_horiz += 1
            continue
        slot = int(round(beat * BEAT_SUBDIV))
        if slot < 0 or slot >= len(dpitch):
            continue
        dp = float(dpitch[slot])
        if abs(dp) < args.deadband:
            n_skip_flat += 1
            continue
        n_scored += 1
        if (dp > 0) == (vert > 0):
            n_match += 1

    rate = (n_match / n_scored) if n_scored else 0.0

    result = {
        "label": args.label,
        "audio": str(args.audio),
        "map": str(args.map),
        "difficulty": args.difficulty,
        "deadband": args.deadband,
        "n_notes": len(full_notes),
        "n_scored": n_scored,
        "n_skip_flat": n_skip_flat,
        "n_skip_horizontal": n_skip_horiz,
        "contour_follow_rate": rate,
        # Single-arm DoD is comparative (vs control); 0.5 = chance. We flag a
        # weak absolute bar here, but the real verdict is arm-vs-control delta.
        "above_chance": bool(rate > 0.5),
    }

    print(f"\n=== contour-follow  {args.label or args.map.name} ===")
    print(f"  notes={len(full_notes)}  scored={n_scored}  "
          f"(skip flat={n_skip_flat}, horiz/dot={n_skip_horiz})")
    print(f"  contour-follow rate = {rate:.4f}   (0.5 = chance; higher = swing "
          f"direction tracks the melody)")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(result, indent=2))
        log.info("wrote %s", args.json)


def _load_notes_with_direction(
    map_path: pathlib.Path, difficulty: str
) -> list[tuple[float, int, int, int, int]]:
    """Like eval_alignment._load_generated_beatmap but keeps y + direction.

    Returns (beat, x, y, color, direction) tuples.
    """
    import shutil
    import tempfile
    import zipfile

    from beatsaber_automapper.data.beatmap import parse_difficulty_dat

    tmp = None
    if map_path.suffix == ".zip":
        tmp = tempfile.mkdtemp(prefix="contour_eval_")
        with zipfile.ZipFile(map_path) as zf:
            zf.extractall(tmp)
        map_dir = pathlib.Path(tmp)
    else:
        map_dir = map_path

    diff_files = sorted(map_dir.glob("*.dat"))
    diff_path = None
    for f in diff_files:
        if f.name.lower().startswith(difficulty.lower()):
            diff_path = f
            break
    if diff_path is None:
        for cand in ("ExpertPlus", "Expert", "Hard", "Normal", "Easy"):
            for f in diff_files:
                if cand.lower() in f.name.lower():
                    diff_path = f
                    break
            if diff_path is not None:
                break
    if diff_path is None:
        if tmp is not None:
            shutil.rmtree(tmp, ignore_errors=True)
        raise FileNotFoundError(f"No difficulty .dat in {map_dir}")

    try:
        beatmap = parse_difficulty_dat(diff_path)
        if beatmap is None:
            raise RuntimeError(f"Failed to parse {diff_path}")
        return [(n.beat, n.x, n.y, n.color, n.direction) for n in beatmap.color_notes]
    finally:
        if tmp is not None:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
