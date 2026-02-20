"""CLI: Generate a Beat Saber level from an audio file.

Usage:
    python scripts/generate.py song.mp3
    python scripts/generate.py song.mp3 --difficulty Expert --output level.zip
    python scripts/generate.py song.mp3 --onset-ckpt checkpoints/onset.ckpt \
        --seq-ckpt checkpoints/seq.ckpt
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def main() -> None:
    """Entry point for the bsa-generate CLI command."""
    parser = argparse.ArgumentParser(
        description="Generate a Beat Saber level from an audio file",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("audio", type=Path, help="Input audio file (.mp3, .ogg, .wav)")
    parser.add_argument(
        "--difficulty",
        default="Expert",
        choices=["Easy", "Normal", "Hard", "Expert", "ExpertPlus"],
        help="Difficulty level to generate",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output .zip path (defaults to <audio_stem>.zip)",
    )
    parser.add_argument(
        "--onset-ckpt",
        type=Path,
        default=None,
        dest="onset_ckpt",
        help="Path to trained OnsetLitModule checkpoint (.ckpt)",
    )
    parser.add_argument(
        "--seq-ckpt",
        type=Path,
        default=None,
        dest="seq_ckpt",
        help="Path to trained SequenceLitModule checkpoint (.ckpt)",
    )
    parser.add_argument(
        "--lighting-ckpt",
        type=Path,
        default=None,
        dest="lighting_ckpt",
        help="Path to trained LightingLitModule checkpoint (.ckpt). Skipped if not provided.",
    )
    parser.add_argument(
        "--bpm",
        type=float,
        default=None,
        help="Song BPM. Auto-detected via librosa if not provided.",
    )
    parser.add_argument(
        "--genre",
        default="unknown",
        choices=[
            "unknown", "electronic", "rock", "pop", "anime",
            "hip-hop", "classical", "jazz", "country", "video-game", "other",
        ],
        help="Music genre for model conditioning.",
    )
    parser.add_argument(
        "--song-name",
        default=None,
        dest="song_name",
        help="Song title for Info.dat (defaults to audio filename stem)",
    )
    parser.add_argument(
        "--song-author",
        default="Unknown Artist",
        dest="song_author",
        help="Song artist name for Info.dat",
    )
    parser.add_argument(
        "--beam-size",
        type=int,
        default=8,
        dest="beam_size",
        help="Beam search width for note sequence generation",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature (>1 = more diverse)",
    )
    parser.add_argument(
        "--nucleus-sampling",
        action="store_true",
        dest="nucleus_sampling",
        help="Use nucleus sampling instead of beam search",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        dest="top_p",
        help="Top-p threshold for nucleus sampling",
    )
    parser.add_argument(
        "--onset-threshold",
        type=float,
        default=0.5,
        dest="onset_threshold",
        help="Onset detection probability threshold",
    )
    parser.add_argument(
        "--min-onset-distance",
        type=int,
        default=5,
        dest="min_onset_distance",
        help="Minimum frames between onsets",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Torch device (e.g. cuda, cpu). Auto-detected if not set.",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    )

    audio_path = Path(args.audio)
    if not audio_path.exists():
        parser.error(f"Audio file not found: {audio_path}")

    output_path = args.output or audio_path.with_suffix(".zip")

    from beatsaber_automapper.generation.generate import generate_level

    result = generate_level(
        audio_path=audio_path,
        output_path=output_path,
        difficulty=args.difficulty,
        onset_checkpoint=args.onset_ckpt,
        sequence_checkpoint=args.seq_ckpt,
        lighting_checkpoint=args.lighting_ckpt,
        onset_threshold=args.onset_threshold,
        min_onset_distance=args.min_onset_distance,
        beam_size=args.beam_size,
        temperature=args.temperature,
        use_sampling=args.nucleus_sampling,
        top_p=args.top_p,
        song_name=args.song_name,
        song_author=args.song_author,
        bpm=args.bpm,
        genre=args.genre,
        device=args.device,
    )

    print(f"Generated level: {result}")


if __name__ == "__main__":
    main()
