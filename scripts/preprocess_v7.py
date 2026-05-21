"""V7-1 preprocessing: Demucs + MERT features for all processed .pt files.

For each song in data/processed/:
  1. Load original audio from data/raw/{song_id}.zip
  2. Demucs separate → drums stem, other (melody) stem
  3. MERT-v1-95M encode each stem → [T_mert, 768] at 75 Hz
  4. Pool to 1/4-note beat grid → [N_slots, 768]
  5. Compute phrase fingerprints → [N_phrases, 768]
  6. Store new keys back into the .pt file (non-destructive)

Adds these keys to each .pt file:
  drum_beat_features  Tensor[N_slots, 768]   drum MERT pooled to 1/4-note grid
  mix_beat_features   Tensor[N_slots, 768]   melody stem MERT pooled to grid
  phrase_fingerprints Tensor[N_phrases, 768] mean MERT per 4-bar window
  phrase_boundaries   list[(start_slot, end_slot)]

Usage:
    python scripts/preprocess_v7.py [--workers N] [--limit K] [--force]

Options:
    --workers N  Number of parallel workers (default: 1; GPU forces 1)
    --limit K    Process only the first K songs (for testing)
    --force      Re-process songs that already have V7 features
    --song ID    Process only this song ID
"""

from __future__ import annotations

import argparse
import logging
import pathlib
import sys
import time
import zipfile

import torch

REPO_ROOT = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from beatsaber_automapper.data.mert_encoder import (
    extract_features,
    phrase_fingerprints,
    pool_to_beat_grid,
    BEAT_SUBDIV,
)
from beatsaber_automapper.data.stem_separator import separate, DEMUCS_SR
from beatsaber_automapper.data.audio import load_audio

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler()],
)
log = logging.getLogger(__name__)

DATA_DIR  = REPO_ROOT / "data/processed"
RAW_DIR   = REPO_ROOT / "data/raw"
FRAMES_PER_SEC = 44100 / 512  # mel frame rate (for duration estimation)
BEATS_PER_PHRASE = 16         # 4 bars at 4/4


def _extract_audio_from_zip(zip_path: pathlib.Path) -> bytes | None:
    """Return raw audio bytes from the first audio file inside a .zip."""
    audio_exts = {".mp3", ".ogg", ".wav", ".egg", ".flac"}
    try:
        with zipfile.ZipFile(zip_path) as zf:
            for name in zf.namelist():
                if pathlib.Path(name).suffix.lower() in audio_exts:
                    return zf.read(name), pathlib.Path(name).suffix.lower()
    except Exception as e:
        log.warning("Failed to open zip %s: %s", zip_path, e)
    return None, None


def _load_audio_from_zip(zip_path: pathlib.Path) -> tuple[torch.Tensor, int] | None:
    """Extract audio from zip and load into a waveform tensor."""
    import tempfile, os

    audio_bytes, ext = _extract_audio_from_zip(zip_path)
    if audio_bytes is None:
        return None

    # Write to temp file (some formats need seekable I/O)
    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as f:
        f.write(audio_bytes)
        tmp_path = f.name

    try:
        wav, sr = load_audio(pathlib.Path(tmp_path), target_sr=DEMUCS_SR)
        return wav, sr
    except Exception as e:
        log.warning("Audio load failed for %s: %s", zip_path.name, e)
        return None
    finally:
        os.unlink(tmp_path)


def process_song(
    pt_path: pathlib.Path,
    device: str,
    force: bool = False,
) -> bool:
    """Process one song: add V7 features to its .pt file.

    Returns True on success, False if skipped or failed.
    """
    song_id = pt_path.stem

    # Load existing .pt
    try:
        data = torch.load(pt_path, weights_only=False)
    except Exception as e:
        log.error("[%s] Failed to load .pt: %s", song_id, e)
        return False

    # Skip if already processed (unless --force)
    if not force and "drum_beat_features" in data:
        log.debug("[%s] Already has V7 features, skipping.", song_id)
        return True

    bpm = float(data.get("bpm", 0.0))
    if bpm <= 0:
        log.warning("[%s] Invalid BPM %.1f, skipping.", song_id, bpm)
        return False

    # Estimate song duration from mel spectrogram
    mel = data.get("mel_spectrogram")
    if mel is None:
        log.warning("[%s] No mel_spectrogram in .pt, skipping.", song_id)
        return False
    duration_secs = mel.shape[1] / FRAMES_PER_SEC
    total_beats   = duration_secs * bpm / 60.0

    # Load audio from raw zip
    zip_path = RAW_DIR / f"{song_id}.zip"
    if not zip_path.exists():
        log.warning("[%s] No raw zip found at %s, skipping.", song_id, zip_path)
        return False

    result = _load_audio_from_zip(zip_path)
    if result is None:
        log.warning("[%s] Could not load audio from zip.", song_id)
        return False
    wav, src_sr = result

    # Source separation
    try:
        stems = separate(wav, src_sr, device=device)
    except Exception as e:
        log.error("[%s] Demucs failed: %s", song_id, e)
        return False

    # MERT feature extraction for drum and melody stems
    try:
        drum_feats = extract_features(stems["drums"], DEMUCS_SR, device=device)
        mix_feats  = extract_features(stems["other"], DEMUCS_SR, device=device)
    except Exception as e:
        log.error("[%s] MERT failed: %s", song_id, e)
        return False

    # Pool to beat grid
    drum_beat = pool_to_beat_grid(drum_feats, bpm, total_beats, BEAT_SUBDIV)
    mix_beat  = pool_to_beat_grid(mix_feats,  bpm, total_beats, BEAT_SUBDIV)

    # Phrase fingerprints from melody stem (carries harmonic/melodic structure)
    fingerprints, boundaries = phrase_fingerprints(
        mix_beat, beats_per_phrase=BEATS_PER_PHRASE, subdiv=BEAT_SUBDIV,
    )

    # Write new keys back into the .pt file
    data["drum_beat_features"]  = drum_beat.half()   # fp16 to save space
    data["mix_beat_features"]   = mix_beat.half()
    data["phrase_fingerprints"] = fingerprints.half()
    data["phrase_boundaries"]   = boundaries          # list of (start, end) ints

    try:
        torch.save(data, pt_path)
    except Exception as e:
        log.error("[%s] Failed to save .pt: %s", song_id, e)
        return False

    log.info(
        "[%s] done  drum_beat=%s  mix_beat=%s  phrases=%d",
        song_id, tuple(drum_beat.shape), tuple(mix_beat.shape), len(fingerprints),
    )
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="V7-1: Demucs+MERT preprocessing")
    parser.add_argument("--workers", type=int, default=1,
                        help="Parallel workers (GPU separation forces 1)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Process at most this many songs")
    parser.add_argument("--force", action="store_true",
                        help="Re-process songs already having V7 features")
    parser.add_argument("--song", type=str, default=None,
                        help="Process only this song ID")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = args.device
    log.info("Device: %s", device)

    pt_files = sorted(DATA_DIR.glob("*.pt"))

    if args.song:
        pt_files = [DATA_DIR / f"{args.song}.pt"]
        if not pt_files[0].exists():
            log.error("Song %s not found in %s", args.song, DATA_DIR)
            sys.exit(1)

    if args.limit:
        pt_files = pt_files[:args.limit]

    total = len(pt_files)
    log.info("Processing %d songs …", total)

    # Warm up models before the loop (avoids cold-start on first song)
    log.info("Warming up Demucs + MERT …")
    from beatsaber_automapper.data.stem_separator import _load_demucs
    from beatsaber_automapper.data.mert_encoder import _load_mert
    _load_demucs(device)
    _load_mert(device)

    ok = skip = fail = 0
    t0 = time.time()

    for i, pt_path in enumerate(pt_files, 1):
        song_id = pt_path.stem
        if not args.force and "drum_beat_features" in torch.load(
            pt_path, weights_only=False, mmap=True
        ):
            skip += 1
            continue

        log.info("[%d/%d] %s", i, total, song_id)
        success = process_song(pt_path, device=device, force=args.force)
        if success:
            ok += 1
        else:
            fail += 1

        elapsed = time.time() - t0
        rate    = (ok + skip) / max(elapsed, 1)
        remain  = (total - i) / max(rate, 1e-6)
        log.info("Progress: %d ok  %d skip  %d fail  ETA %.0f min",
                 ok, skip, fail, remain / 60)

    log.info("Done. %d processed, %d skipped, %d failed in %.1f min.",
             ok, skip, fail, (time.time() - t0) / 60)


if __name__ == "__main__":
    main()
