"""Evaluate model quality over time using a fixed reference song.

Runs the generation pipeline on a reference song, saves the output,
computes metrics, and appends to a history file for tracking improvement.

Usage:
    python scripts/evaluate_reference.py --audio data/reference/test_song.ogg
    python scripts/evaluate_reference.py --audio data/reference/test_song.ogg \
        --onset-ckpt outputs/.../onset.ckpt --seq-ckpt outputs/.../sequence.ckpt
    python scripts/evaluate_reference.py --plot  # show metrics over time
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import zipfile
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).parent.parent
REFERENCE_DIR = REPO_ROOT / "data" / "reference"
SNAPSHOTS_DIR = REFERENCE_DIR / "snapshots"
HISTORY_FILE = REFERENCE_DIR / "history.json"


def _find_best_checkpoint(stage: str) -> Path | None:
    """Find the best checkpoint for a stage in outputs/."""
    outputs_dir = REPO_ROOT / "outputs"
    if not outputs_dir.exists():
        return None

    best = None
    for ckpt in outputs_dir.rglob("*.ckpt"):
        if ckpt.stem.startswith(stage) or (
            ckpt.stem == "last"
            and any(s.stem.startswith(stage) for s in ckpt.parent.glob("*.ckpt"))
        ):
            if best is None or ckpt.stat().st_mtime > best.stat().st_mtime:
                best = ckpt
    return best


def _analyze_zip(zip_path: Path) -> dict:
    """Analyze a generated Beat Saber zip for quality metrics."""
    metrics = {
        "difficulties": {},
        "total_notes": 0,
        "total_bombs": 0,
        "total_walls": 0,
        "total_arcs": 0,
        "total_chains": 0,
        "total_light_events": 0,
    }

    with zipfile.ZipFile(zip_path) as zf:
        for name in zf.namelist():
            if not name.endswith("Standard.dat"):
                continue

            diff_name = name.replace("Standard.dat", "")
            data = json.loads(zf.read(name))

            notes = data.get("colorNotes", [])
            bombs = data.get("bombNotes", [])
            walls = data.get("obstacles", [])
            arcs = data.get("sliders", [])
            chains = data.get("burstSliders", [])
            lights = data.get("basicBeatmapEvents", [])

            # Grid coverage: which of the 12 cells (4x3) are used
            grid_cells = set()
            for n in notes:
                grid_cells.add((n.get("x", 0), n.get("y", 0)))

            # Direction distribution
            dir_counts = {}
            for n in notes:
                d = n.get("d", 8)
                dir_counts[d] = dir_counts.get(d, 0) + 1

            # Color balance
            red_count = sum(1 for n in notes if n.get("c", 0) == 0)
            blue_count = sum(1 for n in notes if n.get("c", 0) == 1)

            # Timing stats
            if notes:
                beats = [n.get("b", 0) for n in notes]
                duration_beats = max(beats) - min(beats) if len(beats) > 1 else 0
                intervals = sorted(
                    beats[i + 1] - beats[i]
                    for i in range(len(beats) - 1)
                    if beats[i + 1] > beats[i]
                )
                median_interval = (
                    intervals[len(intervals) // 2] if intervals else 0
                )
            else:
                duration_beats = 0
                median_interval = 0

            # Unique patterns (note type + position + direction combos)
            patterns = set()
            for n in notes:
                pat = (n.get("c", 0), n.get("x", 0), n.get("y", 0), n.get("d", 0))
                patterns.add(pat)

            diff_metrics = {
                "notes": len(notes),
                "bombs": len(bombs),
                "walls": len(walls),
                "arcs": len(arcs),
                "chains": len(chains),
                "light_events": len(lights),
                "grid_coverage": len(grid_cells),
                "grid_cells_used": sorted(list(grid_cells)),
                "unique_patterns": len(patterns),
                "direction_distribution": dir_counts,
                "red_notes": red_count,
                "blue_notes": blue_count,
                "color_balance": (
                    red_count / max(1, red_count + blue_count)
                ),
                "duration_beats": round(duration_beats, 2),
                "median_note_interval": round(median_interval, 4),
            }

            metrics["difficulties"][diff_name] = diff_metrics
            metrics["total_notes"] += len(notes)
            metrics["total_bombs"] += len(bombs)
            metrics["total_walls"] += len(walls)
            metrics["total_arcs"] += len(arcs)
            metrics["total_chains"] += len(chains)
            metrics["total_light_events"] += len(lights)

    return metrics


def evaluate(
    audio_path: Path,
    onset_ckpt: Path | None = None,
    seq_ckpt: Path | None = None,
    lighting_ckpt: Path | None = None,
    difficulty: str = "Expert",
) -> dict:
    """Run generation on reference song and compute metrics."""
    from beatsaber_automapper.generation.generate import generate_level

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    SNAPSHOTS_DIR.mkdir(parents=True, exist_ok=True)

    output_path = SNAPSHOTS_DIR / f"reference_{timestamp}.zip"

    # Auto-find checkpoints if not provided
    if onset_ckpt is None:
        onset_ckpt = _find_best_checkpoint("onset")
    if seq_ckpt is None:
        seq_ckpt = _find_best_checkpoint("sequence")
    if lighting_ckpt is None:
        lighting_ckpt = _find_best_checkpoint("lighting")

    ckpt_info = {
        "onset": str(onset_ckpt) if onset_ckpt else "random",
        "sequence": str(seq_ckpt) if seq_ckpt else "random",
        "lighting": str(lighting_ckpt) if lighting_ckpt else "random",
    }

    logger.info("Generating reference map...")
    logger.info("  Audio: %s", audio_path)
    logger.info("  Checkpoints: %s", ckpt_info)

    result = generate_level(
        audio_path=audio_path,
        output_path=output_path,
        difficulties=[difficulty],
        onset_checkpoint=onset_ckpt,
        sequence_checkpoint=seq_ckpt,
        lighting_checkpoint=lighting_ckpt,
        song_name="Reference Song",
        song_author="Evaluation",
    )

    logger.info("Generated: %s", result)

    # Analyze the generated zip
    zip_metrics = _analyze_zip(result)

    # Build history entry
    entry = {
        "timestamp": timestamp,
        "audio": str(audio_path),
        "checkpoints": ckpt_info,
        "difficulty": difficulty,
        "output_zip": str(result),
        "metrics": zip_metrics,
    }

    # Append to history
    history = []
    if HISTORY_FILE.exists():
        with open(HISTORY_FILE) as f:
            history = json.load(f)

    history.append(entry)

    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=2)

    return entry


def print_report(entry: dict) -> None:
    """Print a human-readable report of evaluation metrics."""
    m = entry["metrics"]
    print(f"\n{'=' * 60}")
    print(f"Reference Song Evaluation — {entry['timestamp']}")
    print(f"{'=' * 60}")
    print(f"Audio: {entry['audio']}")
    print("Checkpoints:")
    for stage, path in entry["checkpoints"].items():
        print(f"  {stage}: {Path(path).name if path != 'random' else 'random weights'}")

    print("\nTotals:")
    print(f"  Notes: {m['total_notes']}")
    print(f"  Bombs: {m['total_bombs']}")
    print(f"  Walls: {m['total_walls']}")
    print(f"  Arcs: {m['total_arcs']}")
    print(f"  Chains: {m['total_chains']}")
    print(f"  Light events: {m['total_light_events']}")

    for diff_name, dm in m["difficulties"].items():
        print(f"\n  [{diff_name}]")
        print(f"    Notes: {dm['notes']} (red={dm['red_notes']}, blue={dm['blue_notes']}, "
              f"balance={dm['color_balance']:.1%})")
        print(f"    Grid coverage: {dm['grid_coverage']}/12 cells")
        print(f"    Unique patterns: {dm['unique_patterns']}")
        print(f"    Duration: {dm['duration_beats']:.1f} beats")
        print(f"    Median note interval: {dm['median_note_interval']:.3f} beats")
        dirs = dm.get("direction_distribution", {})
        if dirs:
            dir_names = {
                "0": "up", "1": "down", "2": "left", "3": "right",
                "4": "upL", "5": "upR", "6": "dnL", "7": "dnR", "8": "any",
            }
            dir_str = ", ".join(
                f"{dir_names.get(str(k), k)}={v}" for k, v in sorted(dirs.items())
            )
            print(f"    Directions: {dir_str}")

    print(f"\nSnapshot saved: {entry['output_zip']}")
    print("Drag into ArcViewer to preview!")


def plot_history() -> None:
    """Plot metrics over time from history.json."""
    if not HISTORY_FILE.exists():
        print("No history.json found. Run an evaluation first.")
        return

    with open(HISTORY_FILE) as f:
        history = json.load(f)

    if len(history) < 2:
        print(f"Only {len(history)} evaluation(s) in history. Need at least 2 to plot.")
        for entry in history:
            print_report(entry)
        return

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed. Install with: uv pip install matplotlib")
        print("\nFalling back to text summary:")
        for entry in history:
            m = entry["metrics"]
            print(f"  {entry['timestamp']}: "
                  f"notes={m['total_notes']}, "
                  f"bombs={m['total_bombs']}, "
                  f"walls={m['total_walls']}, "
                  f"lights={m['total_light_events']}")
        return

    timestamps = [e["timestamp"] for e in history]
    notes = [e["metrics"]["total_notes"] for e in history]
    bombs = [e["metrics"]["total_bombs"] for e in history]
    walls = [e["metrics"]["total_walls"] for e in history]
    lights = [e["metrics"]["total_light_events"] for e in history]

    # Get per-difficulty metrics if available
    grid_coverages = []
    unique_patterns = []
    for e in history:
        diffs = e["metrics"]["difficulties"]
        if diffs:
            first_diff = next(iter(diffs.values()))
            grid_coverages.append(first_diff.get("grid_coverage", 0))
            unique_patterns.append(first_diff.get("unique_patterns", 0))
        else:
            grid_coverages.append(0)
            unique_patterns.append(0)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Reference Song — Quality Over Time", fontsize=14)

    x = range(len(timestamps))
    labels = [t[4:8] + "-" + t[8:10] + ":" + t[10:12] for t in timestamps]

    axes[0, 0].plot(x, notes, "b-o", label="Notes")
    axes[0, 0].plot(x, bombs, "r-s", label="Bombs")
    axes[0, 0].plot(x, walls, "g-^", label="Walls")
    axes[0, 0].set_title("Object Counts")
    axes[0, 0].legend()
    axes[0, 0].set_xticks(list(x))
    axes[0, 0].set_xticklabels(labels, rotation=45, fontsize=8)

    axes[0, 1].plot(x, lights, "m-o")
    axes[0, 1].set_title("Light Events")
    axes[0, 1].set_xticks(list(x))
    axes[0, 1].set_xticklabels(labels, rotation=45, fontsize=8)

    axes[1, 0].bar(x, grid_coverages, color="teal")
    axes[1, 0].set_title("Grid Coverage (out of 12)")
    axes[1, 0].set_ylim(0, 12)
    axes[1, 0].set_xticks(list(x))
    axes[1, 0].set_xticklabels(labels, rotation=45, fontsize=8)

    axes[1, 1].plot(x, unique_patterns, "k-o")
    axes[1, 1].set_title("Unique Note Patterns")
    axes[1, 1].set_xticks(list(x))
    axes[1, 1].set_xticklabels(labels, rotation=45, fontsize=8)

    plt.tight_layout()
    plot_path = REFERENCE_DIR / "quality_over_time.png"
    plt.savefig(plot_path, dpi=150)
    print(f"Plot saved: {plot_path}")
    plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate model quality using a reference song"
    )
    parser.add_argument(
        "--audio",
        type=Path,
        default=REFERENCE_DIR / "test_song.ogg",
        help="Path to reference audio file",
    )
    parser.add_argument("--onset-ckpt", type=Path, default=None)
    parser.add_argument("--seq-ckpt", type=Path, default=None)
    parser.add_argument("--lighting-ckpt", type=Path, default=None)
    parser.add_argument(
        "--difficulty", default="Expert", help="Difficulty to generate"
    )
    parser.add_argument(
        "--plot", action="store_true", help="Plot metrics history instead of evaluating"
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    )

    if args.plot:
        plot_history()
        return

    if not args.audio.exists():
        print(f"Reference audio not found: {args.audio}")
        print("\nTo set up reference evaluation:")
        print(f"  1. Copy your reference song to: {REFERENCE_DIR / 'test_song.ogg'}")
        print("  2. Run: python scripts/evaluate_reference.py")
        print("  3. After more training, run again to compare")
        print("  4. Run with --plot to see quality over time")
        sys.exit(1)

    entry = evaluate(
        audio_path=args.audio,
        onset_ckpt=args.onset_ckpt,
        seq_ckpt=args.seq_ckpt,
        lighting_ckpt=args.lighting_ckpt,
        difficulty=args.difficulty,
    )
    print_report(entry)


if __name__ == "__main__":
    main()
