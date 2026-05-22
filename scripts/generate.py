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
        nargs="+",
        default=["Expert"],
        choices=["Easy", "Normal", "Hard", "Expert", "ExpertPlus"],
        help="Difficulty level(s) to generate (can specify multiple)",
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
        "--run-tag",
        default=None,
        dest="run_tag",
        help=(
            "Subdirectory tag for the output when --output is not specified. "
            "Map is written to data/generated/{run_tag}/{song_name}.zip. "
            "Example: --run-tag v5_joetastic_ep12"
        ),
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
        default=0.8,
        help="Sampling temperature (lower = less random, 0.8 recommended)",
    )
    parser.add_argument(
        "--nucleus-sampling",
        action="store_true",
        default=True,
        dest="nucleus_sampling",
        help="Use nucleus sampling (default: on)",
    )
    parser.add_argument(
        "--beam-search",
        action="store_false",
        dest="nucleus_sampling",
        help="Use beam search instead of nucleus sampling",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.85,
        dest="top_p",
        help="Top-p threshold for nucleus sampling (0.85 = tighter nucleus)",
    )
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=1.5,
        dest="repetition_penalty",
        help="Repetition penalty for nucleus sampling (1.5 = more variety)",
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
    parser.add_argument(
        "--v6",
        action="store_true",
        dest="use_v6",
        help="Use V6 swing-event generator (single grammar-constrained AR pass). "
             "Onset model is not needed in this mode.",
    )
    parser.add_argument(
        "--max-events",
        type=int,
        default=800,
        dest="max_events",
        help="V6 only: max swing events to generate (cap on song length).",
    )
    parser.add_argument(
        "--mapper-id",
        type=int,
        default=0,
        dest="mapper_id",
        help="V6 only: cohort/mapper conditioning index (0 = unknown).",
    )
    # V7 flags
    parser.add_argument(
        "--v7",
        action="store_true",
        dest="use_v7",
        help="Use V7 pipeline: Demucs + MERT + BeatClassifier + LayoutModel + PhraseIndex.",
    )
    parser.add_argument(
        "--beat-ckpt",
        type=Path,
        default=None,
        dest="beat_ckpt",
        help="V7 only: path to BeatLitModule checkpoint.",
    )
    parser.add_argument(
        "--layout-ckpt",
        type=Path,
        default=None,
        dest="layout_ckpt",
        help="V7 only: path to LayoutLitModule checkpoint.",
    )
    parser.add_argument(
        "--beat-threshold",
        type=float,
        default=0.4,
        dest="beat_threshold",
        help="V7 only: probability threshold for Stage 1 onset detection.",
    )
    parser.add_argument(
        "--phrase-similarity",
        type=float,
        default=0.85,
        dest="phrase_similarity",
        help="V7 only: cosine similarity threshold for PhraseIndex hard retrieval.",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    )

    audio_path = Path(args.audio)
    if not audio_path.exists():
        parser.error(f"Audio file not found: {audio_path}")

    if args.output:
        output_path = args.output
    elif args.run_tag:
        out_dir = Path("data/generated") / args.run_tag
        out_dir.mkdir(parents=True, exist_ok=True)
        output_path = out_dir / f"{audio_path.stem}.zip"
    else:
        output_path = audio_path.with_suffix(".zip")

    if args.use_v7:
        if len(args.difficulty) > 1:
            parser.error("--v7 mode generates one difficulty at a time")
        if args.beat_ckpt is None or args.layout_ckpt is None:
            parser.error("--v7 requires --beat-ckpt and --layout-ckpt")
        from beatsaber_automapper.generation.generate import generate_v7_level

        result = generate_v7_level(
            audio_path=audio_path,
            output_path=output_path,
            beat_checkpoint=args.beat_ckpt,
            layout_checkpoint=args.layout_ckpt,
            difficulty=args.difficulty[0],
            genre=args.genre,
            song_name=args.song_name,
            song_author=args.song_author,
            bpm=args.bpm,
            beat_threshold_left=args.beat_threshold,
            beat_threshold_right=args.beat_threshold,
            temperature=args.temperature,
            top_p=args.top_p,
            phrase_similarity=args.phrase_similarity,
            device=args.device,
        )
    elif args.use_v6:
        if len(args.difficulty) > 1:
            parser.error("--v6 mode generates one difficulty at a time")
        from beatsaber_automapper.generation.generate import generate_swing_level

        result = generate_swing_level(
            audio_path=audio_path,
            output_path=output_path,
            difficulty=args.difficulty[0],
            sequence_checkpoint=args.seq_ckpt,
            onset_checkpoint=args.onset_ckpt,
            temperature=args.temperature,
            top_p=args.top_p,
            max_events=args.max_events,
            song_name=args.song_name,
            song_author=args.song_author,
            bpm=args.bpm,
            genre=args.genre,
            mapper_id=args.mapper_id,
            device=args.device,
        )
    else:
        from beatsaber_automapper.generation.generate import generate_level

        result = generate_level(
            audio_path=audio_path,
            output_path=output_path,
            difficulties=args.difficulty,
            onset_checkpoint=args.onset_ckpt,
            sequence_checkpoint=args.seq_ckpt,
            onset_threshold=args.onset_threshold,
            min_onset_distance=args.min_onset_distance,
            beam_size=args.beam_size,
            temperature=args.temperature,
            use_sampling=args.nucleus_sampling,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            song_name=args.song_name,
            song_author=args.song_author,
            bpm=args.bpm,
            genre=args.genre,
            device=args.device,
        )

    print(f"Generated level: {result}")


if __name__ == "__main__":
    main()
