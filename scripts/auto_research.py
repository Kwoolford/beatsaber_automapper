"""Auto-researcher CLI — run an experiment queue.

Usage:
    python scripts/auto_research.py experiments/queue/initial.yaml
    python scripts/auto_research.py experiments/queue/initial.yaml --resume
    python scripts/auto_research.py experiments/queue/initial.yaml --only <experiment_id>
    python scripts/auto_research.py experiments/queue/initial.yaml --dry-run

Each entry in the queue becomes one isolated run under experiments/runs/{id}/.
Results append to experiments/leaderboard.jsonl.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from beatsaber_automapper.research.runner import run_experiment  # noqa: E402
from beatsaber_automapper.research.spec import load_queue  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("auto_research")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("queue", type=Path, help="Path to queue YAML")
    p.add_argument("--project-root", type=Path, default=Path.cwd())
    p.add_argument("--experiments-root", type=Path, default=Path("experiments"))
    p.add_argument("--cohort-root", type=Path, default=Path("data/cohorts"))
    p.add_argument(
        "--test-audio",
        type=Path,
        default=Path("data/reference/so_tired_rock.mp3"),
    )
    p.add_argument(
        "--test-duration-sec",
        type=float,
        default=None,
        help="Duration of test audio in seconds. Auto-detected from the file header if omitted.",
    )
    p.add_argument(
        "--onset-ckpt",
        type=Path,
        default=Path(
            "outputs/beatsaber_automapper/version_0/checkpoints/"
            "onset-epoch=05-val_f1=0.732.ckpt"
        ),
        help=(
            "Shared onset checkpoint for all experiments. Onset detection is "
            "style-agnostic — no need to retrain per cohort."
        ),
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help="Skip experiments whose run_dir has status=done",
    )
    p.add_argument("--only", help="Only run this experiment_id or spec name")
    p.add_argument("--dry-run", action="store_true", help="Print planned runs; do nothing")
    args = p.parse_args()

    specs = load_queue(args.queue)
    leaderboard = args.experiments_root / "leaderboard.jsonl"
    args.experiments_root.mkdir(parents=True, exist_ok=True)

    logger.info("Loaded %d specs from %s", len(specs), args.queue)

    for spec in specs:
        eid = spec.experiment_id()
        if args.only and args.only not in (eid, spec.name):
            continue

        run_dir = args.experiments_root / "runs" / eid
        status_file = run_dir / "status.json"
        if args.resume and status_file.exists():
            try:
                import json

                status = json.loads(status_file.read_text(encoding="utf-8")).get("status")
                if status == "done":
                    logger.info("[%s] %s — already done, skipping", eid, spec.name)
                    continue
            except Exception:
                pass

        if args.dry_run:
            logger.info("[DRY] %s %s (%s)", eid, spec.name, spec.data_source)
            continue

        logger.info("=" * 60)
        logger.info("Running %s: %s (%s)", eid, spec.name, spec.data_source)
        logger.info("=" * 60)
        result = run_experiment(
            spec,
            project_root=args.project_root,
            experiments_root=args.experiments_root,
            test_audio=args.test_audio,
            test_duration_sec=args.test_duration_sec,
            onset_ckpt=args.onset_ckpt,
            cohort_root=args.cohort_root,
            leaderboard_path=leaderboard,
        )
        logger.info("[%s] status=%s", result.experiment_id, result.status)

    logger.info("Queue complete. Leaderboard at %s", leaderboard)
    return 0


if __name__ == "__main__":
    sys.exit(main())
