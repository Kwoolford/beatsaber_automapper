"""Scoped V8 (TASK 2): cache per-instrument layering features to every .pt.

For each song in data/processed/:
  1. Load original audio from data/raw/{song_id}.zip
  2. Demucs separate → drums/bass/other/vocals
  3. Per-stem transcription (basic-pitch + multi-band drum onset) → NoteEvent stream
  4. Bin onsets to the SAME 1/4-note grid as drum_beat_features → [N_slots, INSTR_FEATURE_DIM]
  5. Store key ``instr_beat_features`` (fp16) back into the .pt (non-destructive)

Mirrors scripts/preprocess_v7.py. Grid length is taken from the existing
``drum_beat_features`` so the layering features line up row-for-row with the
MERT grid and the Stage-1 labels.

Usage:
    python scripts/preprocess_instruments.py [--limit K] [--force] [--song ID]
"""

from __future__ import annotations

import argparse
import logging
import os
import pathlib
import sys
import tempfile
import time
import zipfile

import torch

REPO_ROOT = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from beatsaber_automapper.data.audio import load_audio
from beatsaber_automapper.data.instrument_features import (
    INSTR_FEATURE_DIM,
    compute_instrument_features,
)
from beatsaber_automapper.data.stem_separator import DEMUCS_SR

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

DATA_DIR = REPO_ROOT / "data/processed"
RAW_DIR  = REPO_ROOT / "data/raw"
KEY = "instr_beat_features"


def _load_audio_from_zip(zip_path: pathlib.Path):
    audio_exts = {".mp3", ".ogg", ".wav", ".egg", ".flac"}
    try:
        with zipfile.ZipFile(zip_path) as zf:
            name = next((n for n in zf.namelist()
                         if pathlib.Path(n).suffix.lower() in audio_exts), None)
            if name is None:
                return None
            data = zf.read(name)
            ext = pathlib.Path(name).suffix.lower()
    except Exception as e:
        log.warning("Failed to open zip %s: %s", zip_path, e)
        return None
    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as f:
        f.write(data)
        tmp = f.name
    try:
        return load_audio(pathlib.Path(tmp), target_sr=DEMUCS_SR)
    except Exception as e:
        log.warning("Audio load failed for %s: %s", zip_path.name, e)
        return None
    finally:
        os.unlink(tmp)


def process_song(pt_path: pathlib.Path, device: str, force: bool = False) -> str:
    """Returns 'ok' | 'skip' | 'fail'."""
    song_id = pt_path.stem
    try:
        data = torch.load(pt_path, weights_only=False)
    except Exception as e:
        log.error("[%s] load .pt failed: %s", song_id, e)
        return "fail"

    if not force and KEY in data:
        return "skip"
    if "drum_beat_features" not in data:
        log.warning("[%s] no drum_beat_features (run preprocess_v7 first), skip.", song_id)
        return "fail"

    bpm = float(data.get("bpm", 0.0))
    if bpm <= 0:
        log.warning("[%s] bad bpm, skip.", song_id)
        return "fail"
    n_slots = int(data["drum_beat_features"].shape[0])

    zip_path = RAW_DIR / f"{song_id}.zip"
    if not zip_path.exists():
        log.warning("[%s] no raw zip, skip.", song_id)
        return "fail"
    loaded = _load_audio_from_zip(zip_path)
    if loaded is None:
        return "fail"
    wav, src_sr = loaded

    try:
        feats = compute_instrument_features(wav, src_sr, bpm, n_slots, device=device)
    except Exception as e:
        log.error("[%s] feature extraction failed: %s", song_id, e)
        return "fail"

    data[KEY] = feats.half()
    try:
        torch.save(data, pt_path)
    except Exception as e:
        log.error("[%s] save failed: %s", song_id, e)
        return "fail"

    nz = float((feats.abs().sum(dim=1) > 0).float().mean())
    log.info("[%s] done  instr=%s  nonzero_slots=%.2f", song_id, tuple(feats.shape), nz)
    return "ok"


def main() -> None:
    ap = argparse.ArgumentParser(description="Cache per-instrument layering features")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--song", type=str, default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    log.info("Device: %s  feature_dim=%d", args.device, INSTR_FEATURE_DIM)
    pt_files = sorted(DATA_DIR.glob("*.pt"))
    if args.song:
        pt_files = [DATA_DIR / f"{args.song}.pt"]
    if args.limit:
        pt_files = pt_files[:args.limit]
    total = len(pt_files)
    log.info("Processing %d songs …", total)

    # Warm up Demucs once.
    from beatsaber_automapper.data.stem_separator import _load_demucs
    _load_demucs(args.device)

    ok = skip = fail = 0
    t0 = time.time()
    for i, pt_path in enumerate(pt_files, 1):
        if not args.force and KEY in torch.load(pt_path, weights_only=False, mmap=True):
            skip += 1
            continue
        log.info("[%d/%d] %s", i, total, pt_path.stem)
        r = process_song(pt_path, device=args.device, force=args.force)
        ok += r == "ok"; skip += r == "skip"; fail += r == "fail"
        elapsed = time.time() - t0
        done = ok + fail
        rate = done / max(elapsed, 1)
        remain = (total - i) / max(rate, 1e-6)
        log.info("Progress: %d ok  %d skip  %d fail  ETA %.0f min", ok, skip, fail, remain / 60)

    log.info("Done. %d ok, %d skip, %d fail in %.1f min.", ok, skip, fail, (time.time() - t0) / 60)


if __name__ == "__main__":
    main()
