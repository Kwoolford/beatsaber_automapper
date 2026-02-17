"""CLI: Preprocess downloaded maps into training tensors.

Usage:
    bsa-preprocess --input data/raw --output data/processed

Reads .zip map files, extracts audio + beatmap data, produces .pt files
containing mel spectrograms, onset labels, and token sequences.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import tempfile
import zipfile
from pathlib import Path

import torch
from tqdm import tqdm

from beatsaber_automapper.data.audio import (
    beat_to_frame,
    extract_mel_spectrogram,
    load_audio,
)
from beatsaber_automapper.data.beatmap import (
    parse_difficulty_dat_json,
    parse_info_dat_json,
)
from beatsaber_automapper.data.tokenizer import BeatmapTokenizer

logger = logging.getLogger(__name__)


def main() -> None:
    """Entry point for the bsa-preprocess CLI command."""
    parser = argparse.ArgumentParser(description="Preprocess Beat Saber maps into tensors")
    parser.add_argument("--input", type=Path, default=Path("data/raw"), help="Raw data directory")
    parser.add_argument(
        "--output", type=Path, default=Path("data/processed"), help="Output directory"
    )
    parser.add_argument("--sample-rate", type=int, default=44100, help="Target sample rate")
    parser.add_argument("--n-mels", type=int, default=80, help="Number of mel bands")
    parser.add_argument("--n-fft", type=int, default=1024, help="FFT window size")
    parser.add_argument("--hop-length", type=int, default=512, help="Hop length")
    parser.add_argument("--sigma", type=float, default=3.0, help="Gaussian smoothing sigma")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s: %(message)s"
    )

    preprocess_all(
        input_dir=args.input,
        output_dir=args.output,
        sample_rate=args.sample_rate,
        n_mels=args.n_mels,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        sigma=args.sigma,
    )


def preprocess_all(
    input_dir: Path | str,
    output_dir: Path | str,
    *,
    sample_rate: int = 44100,
    n_mels: int = 80,
    n_fft: int = 1024,
    hop_length: int = 512,
    sigma: float = 3.0,
) -> list[Path]:
    """Preprocess all map zips in input_dir into .pt files.

    Args:
        input_dir: Directory containing .zip map files.
        output_dir: Directory for output .pt files and splits.json.
        sample_rate: Target audio sample rate.
        n_mels: Number of mel frequency bands.
        n_fft: FFT window size.
        hop_length: Spectrogram hop length.
        sigma: Gaussian smoothing sigma for onset labels.

    Returns:
        List of paths to generated .pt files.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    zip_files = sorted(input_dir.glob("*.zip"))
    if not zip_files:
        logger.warning("No .zip files found in %s", input_dir)
        return []

    logger.info("Found %d zip files to process", len(zip_files))
    results: list[Path] = []

    for zip_path in tqdm(zip_files, desc="Preprocessing", unit="map"):
        try:
            pt_path = preprocess_single(
                zip_path,
                output_dir,
                sample_rate=sample_rate,
                n_mels=n_mels,
                n_fft=n_fft,
                hop_length=hop_length,
                sigma=sigma,
            )
            if pt_path is not None:
                results.append(pt_path)
        except Exception as e:
            logger.warning("Failed to process %s: %s", zip_path.name, e)

    # Generate train/val/test splits
    song_ids = [p.stem for p in results]
    if song_ids:
        _generate_splits(song_ids, output_dir)

    logger.info("Preprocessed %d/%d maps", len(results), len(zip_files))
    return results


def preprocess_single(
    zip_path: Path | str,
    output_dir: Path | str,
    *,
    sample_rate: int = 44100,
    n_mels: int = 80,
    n_fft: int = 1024,
    hop_length: int = 512,
    sigma: float = 3.0,
) -> Path | None:
    """Preprocess a single map zip into a .pt file.

    Args:
        zip_path: Path to the map .zip file.
        output_dir: Directory for output .pt file.
        sample_rate: Target audio sample rate.
        n_mels: Number of mel frequency bands.
        n_fft: FFT window size.
        hop_length: Spectrogram hop length.
        sigma: Gaussian smoothing sigma for onset labels.

    Returns:
        Path to the generated .pt file, or None on failure.
    """
    zip_path = Path(zip_path)
    output_dir = Path(output_dir)
    song_id = zip_path.stem

    # Skip if already processed
    pt_path = output_dir / f"{song_id}.pt"
    if pt_path.exists():
        logger.debug("Skipping %s (already processed)", song_id)
        return pt_path

    tokenizer = BeatmapTokenizer()

    with zipfile.ZipFile(zip_path, "r") as zf:
        # Find and parse Info.dat (case-insensitive)
        info_name = _find_file_in_zip(zf, "info.dat")
        if info_name is None:
            logger.warning("No Info.dat found in %s", zip_path.name)
            return None

        info_data = json.loads(zf.read(info_name).decode("utf-8"))
        info = parse_info_dat_json(info_data)
        if info is None:
            return None

        # Load audio via temp file (soundfile needs a real file path)
        audio_name = _find_file_in_zip(zf, info.song_filename)
        if audio_name is None:
            logger.warning("Audio file %s not found in %s", info.song_filename, zip_path.name)
            return None

        audio_bytes = zf.read(audio_name)
        suffix = Path(info.song_filename).suffix or ".ogg"

        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp_audio:
            tmp_audio.write(audio_bytes)
            tmp_audio_path = Path(tmp_audio.name)

        try:
            waveform, sr = load_audio(tmp_audio_path, target_sr=sample_rate)
        finally:
            tmp_audio_path.unlink(missing_ok=True)

        mel = extract_mel_spectrogram(
            waveform, sample_rate=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length
        )
        n_frames = mel.shape[1]

        # Process each difficulty
        difficulties: dict[str, dict] = {}
        for diff_info in info.difficulties:
            diff_name = _find_file_in_zip(zf, diff_info.filename)
            if diff_name is None:
                continue

            diff_data = json.loads(zf.read(diff_name).decode("utf-8"))
            beatmap = parse_difficulty_dat_json(diff_data)
            if beatmap is None:
                continue

            # Tokenize
            beat_tokens = tokenizer.encode_beatmap(beatmap)
            if not beat_tokens:
                continue

            # Convert beats to frames and build onset data
            onset_frames_list: list[int] = []
            token_sequences: list[list[int]] = []
            for beat in sorted(beat_tokens.keys()):
                frame = beat_to_frame(
                    beat, info.bpm, sample_rate=sr, hop_length=hop_length,
                    offset=info.song_time_offset,
                )
                if 0 <= frame < n_frames:
                    onset_frames_list.append(frame)
                    token_sequences.append(beat_tokens[beat])

            if not onset_frames_list:
                continue

            onset_frames = torch.tensor(onset_frames_list, dtype=torch.long)
            onset_labels = _compute_onset_labels(onset_frames, n_frames, sigma=sigma)

            difficulties[diff_info.difficulty] = {
                "onset_frames": onset_frames,
                "onset_labels": onset_labels,
                "token_sequences": token_sequences,
            }

    if not difficulties:
        logger.warning("No valid difficulties in %s", zip_path.name)
        return None

    # Save .pt
    torch.save(
        {
            "song_id": song_id,
            "bpm": info.bpm,
            "mel_spectrogram": mel,
            "difficulties": difficulties,
        },
        pt_path,
    )
    return pt_path


def _compute_onset_labels(
    onset_frames: torch.Tensor,
    n_frames: int,
    sigma: float = 3.0,
) -> torch.Tensor:
    """Compute Gaussian-smoothed onset labels.

    Args:
        onset_frames: 1D tensor of onset frame indices.
        n_frames: Total number of frames.
        sigma: Gaussian smoothing standard deviation in frames.

    Returns:
        Tensor of shape [n_frames] with smoothed onset labels in [0, 1].
    """
    labels = torch.zeros(n_frames)
    window = int(4 * sigma)  # ±4σ for efficiency

    for frame in onset_frames:
        f = int(frame.item())
        start = max(0, f - window)
        end = min(n_frames, f + window + 1)
        positions = torch.arange(start, end, dtype=torch.float32)
        gaussian = torch.exp(-0.5 * ((positions - f) / sigma) ** 2)
        labels[start:end] = torch.maximum(labels[start:end], gaussian)

    return labels


def _generate_splits(
    song_ids: list[str],
    output_dir: Path,
    train_ratio: float = 0.85,
    val_ratio: float = 0.10,
) -> None:
    """Generate deterministic train/val/test splits by song ID hash.

    Args:
        song_ids: List of song IDs.
        output_dir: Directory to save splits.json.
        train_ratio: Fraction for training set.
        val_ratio: Fraction for validation set.
    """
    train: list[str] = []
    val: list[str] = []
    test: list[str] = []

    for sid in sorted(song_ids):
        h = int(hashlib.md5(sid.encode()).hexdigest(), 16) % 100  # noqa: S324
        if h < int(train_ratio * 100):
            train.append(sid)
        elif h < int((train_ratio + val_ratio) * 100):
            val.append(sid)
        else:
            test.append(sid)

    splits = {"train": train, "val": val, "test": test}
    splits_path = output_dir / "splits.json"
    with open(splits_path, "w") as f:
        json.dump(splits, f, indent=2)

    logger.info(
        "Splits: train=%d, val=%d, test=%d",
        len(train),
        len(val),
        len(test),
    )


def _find_file_in_zip(zf: zipfile.ZipFile, target: str) -> str | None:
    """Find a file in a zip archive (case-insensitive).

    Args:
        zf: Open ZipFile.
        target: Filename to find (e.g. "Info.dat").

    Returns:
        Actual name in the zip, or None if not found.
    """
    target_lower = target.lower()
    for name in zf.namelist():
        # Match filename only (handles nested directories)
        basename = name.rsplit("/", 1)[-1] if "/" in name else name
        if basename.lower() == target_lower:
            return name
    return None


if __name__ == "__main__":
    main()
