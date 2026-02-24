"""Chain all three training stages sequentially with full-GPU settings.

Runs onset -> sequence -> lighting training. Each stage uses full VRAM
with optimal batch sizes for the RTX 5090 (32 GB). Designed for unattended
overnight runs — all output is teed to logs/.

Usage:
    python scripts/run_training_pipeline.py
    python scripts/run_training_pipeline.py --skip-onset   # resume after onset
    python scripts/run_training_pipeline.py --stages sequence lighting
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
SPLITS_JSON = REPO_ROOT / "data" / "processed" / "splits.json"
LOGS_DIR = REPO_ROOT / "logs"

# Optimal batch sizes for RTX 5090 32GB, full VRAM (no gaming)
# Onset uses window_size=1024 (4x larger than old 256), so batch_size is reduced.
STAGE_SETTINGS: dict[str, dict[str, str]] = {
    "onset": {
        "data.dataset.batch_size": "32",
        "data.dataset.num_workers": "12",
    },
    "sequence": {
        "data.dataset.batch_size": "32",
        "data.dataset.num_workers": "8",
    },
    "lighting": {
        "data.dataset.batch_size": "48",
        "data.dataset.num_workers": "8",
    },
}


def run_stage(stage: str, max_epochs: int = 100) -> int:
    """Run a training stage, return exit code."""
    log_path = LOGS_DIR / f"train_{stage}_full.log"
    settings = STAGE_SETTINGS.get(stage, {})

    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "train.py"),
        f"stage={stage}",
        "data_dir=data/processed",
        f"max_epochs={max_epochs}",
        "accelerator=gpu",
    ]
    for key, val in settings.items():
        cmd.append(f"{key}={val}")

    print(f"\n{'='*60}")
    print(f"Starting {stage} training (max_epochs={max_epochs})")
    print(f"  batch_size={settings.get('data.dataset.batch_size', 'default')}")
    print(f"  num_workers={settings.get('data.dataset.num_workers', 'default')}")
    print(f"Log: {log_path}")
    print(f"{'='*60}", flush=True)

    with open(log_path, "w") as log_fh:
        proc = subprocess.Popen(
            cmd,
            cwd=REPO_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        for line in proc.stdout:  # type: ignore[union-attr]
            print(line, end="", flush=True)
            log_fh.write(line)
        proc.wait()

    return proc.returncode


def main() -> None:
    parser = argparse.ArgumentParser(description="Run full training pipeline")
    parser.add_argument(
        "--stages",
        nargs="+",
        default=["onset", "sequence", "lighting"],
        choices=["onset", "sequence", "lighting"],
        help="Stages to run (default: all three in order)",
    )
    parser.add_argument(
        "--max-epochs", type=int, default=100, help="Max epochs per stage"
    )
    parser.add_argument(
        "--skip-onset",
        action="store_true",
        help="Skip onset stage (shortcut for --stages sequence lighting)",
    )
    args = parser.parse_args()

    stages = args.stages
    if args.skip_onset:
        stages = [s for s in stages if s != "onset"]

    LOGS_DIR.mkdir(exist_ok=True)

    if not SPLITS_JSON.exists():
        print("ERROR: data/processed/splits.json not found.")
        print("Run preprocessing first:")
        print("  python scripts/preprocess.py --input data/raw --output data/processed")
        sys.exit(1)

    start_time = time.time()
    for stage in stages:
        stage_start = time.time()
        rc = run_stage(stage, max_epochs=args.max_epochs)
        elapsed = (time.time() - stage_start) / 3600
        if rc != 0:
            print(f"\nERROR: {stage} exited with code {rc} after {elapsed:.1f}h. Stopping.")
            sys.exit(rc)
        print(f"\n{stage} training finished successfully in {elapsed:.1f}h.")

    total_hours = (time.time() - start_time) / 3600
    print(f"\nAll stages complete in {total_hours:.1f}h! Check outputs/ for checkpoints.")


if __name__ == "__main__":
    main()
