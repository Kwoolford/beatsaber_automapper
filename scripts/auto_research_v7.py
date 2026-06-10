"""V7 auto-researcher CLI — drive a queue of V7 layout experiments.

V6 used scripts/auto_research.py + research/runner.py (Hydra train.py).
V7 has separate stages (train_beats / train_layout) plus a v7-mode generate
+ alignment-eval step, so this script + research/runner_v7.py is the V7
analog. Each queue entry → one isolated run → one leaderboard row.

Usage:
    python scripts/auto_research_v7.py experiments/queue/v7_layout_ctx_ablation.yaml
    python scripts/auto_research_v7.py <queue> --resume
    python scripts/auto_research_v7.py <queue> --only <experiment_id|name>
    python scripts/auto_research_v7.py <queue> --dry-run
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from beatsaber_automapper.research.runner_v7 import run_v7_layout_experiment  # noqa: E402
from beatsaber_automapper.research.spec_v7 import load_v7_queue  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("auto_research_v7")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("queue", type=Path)
    p.add_argument("--project-root", type=Path, default=Path.cwd())
    p.add_argument("--experiments-root", type=Path, default=Path("experiments"))
    p.add_argument("--resume", action="store_true",
                   help="Skip experiments whose status.json already shows done.")
    p.add_argument("--only", help="Only run this experiment_id or name.")
    p.add_argument("--dry-run", action="store_true",
                   help="Print planned runs; do nothing.")
    args = p.parse_args()

    specs = load_v7_queue(args.queue)
    leaderboard = args.experiments_root / "leaderboard_v7.jsonl"
    args.experiments_root.mkdir(parents=True, exist_ok=True)

    logger.info("Loaded %d V7 specs from %s", len(specs), args.queue)

    for spec in specs:
        eid = spec.experiment_id()
        if args.only and args.only not in (eid, spec.name):
            continue

        run_dir = args.experiments_root / "runs" / eid
        status_file = run_dir / "status.json"
        if args.resume and status_file.exists():
            try:
                status = json.loads(status_file.read_text(encoding="utf-8")).get("status")
                if status == "done":
                    logger.info("[%s] %s — already done, skipping", eid, spec.name)
                    continue
            except Exception:
                pass

        if args.dry_run:
            logger.info("[DRY] %s %s ctx_len=%d max_song_phrases=%d", eid, spec.name,
                        spec.ctx_len, spec.max_song_phrases)
            continue

        logger.info("=" * 60)
        logger.info("Running %s: %s", eid, spec.name)
        logger.info("=" * 60)
        result = run_v7_layout_experiment(
            spec,
            project_root=args.project_root,
            experiments_root=args.experiments_root,
            leaderboard_path=leaderboard,
        )
        logger.info("[%s] status=%s", result.experiment_id, result.status)

    logger.info("Queue complete. V7 leaderboard at %s", leaderboard)
    return 0


if __name__ == "__main__":
    sys.exit(main())
