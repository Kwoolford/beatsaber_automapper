"""Chain all three training stages after preprocessing completes.

Polls logs/preprocess_full.log until preprocessing is done (splits.json
written to data/processed/), then runs onset -> sequence -> lighting training
sequentially. All output is teed to logs/.

Usage:
    python scripts/run_training_pipeline.py
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
SPLITS_JSON = REPO_ROOT / "data" / "processed" / "splits.json"
PREPROCESS_LOG = REPO_ROOT / "logs" / "preprocess_full.log"
LOGS_DIR = REPO_ROOT / "logs"


def wait_for_preprocessing(poll_seconds: int = 30) -> None:
    """Block until data/processed/splits.json exists (preprocessing done)."""
    print(f"Waiting for preprocessing to finish (polling every {poll_seconds}s)...")
    while not SPLITS_JSON.exists():
        time.sleep(poll_seconds)
        if PREPROCESS_LOG.exists():
            last = PREPROCESS_LOG.read_text().splitlines()
            # Print last non-empty progress line
            for line in reversed(last):
                if "map/s" in line or "Preprocessed" in line:
                    print(f"  [{line.strip()[-80:]}]")
                    break
    print("Preprocessing complete!")


def run_stage(stage: str, max_epochs: int = 100) -> int:
    """Run a training stage, return exit code."""
    log_path = LOGS_DIR / f"train_{stage}_full.log"
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "train.py"),
        f"stage={stage}",
        "data_dir=data/processed",
        f"max_epochs={max_epochs}",
    ]
    print(f"\n{'='*60}")
    print(f"Starting {stage} training (max_epochs={max_epochs})")
    print(f"Log: {log_path}")
    print(f"{'='*60}")

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
    LOGS_DIR.mkdir(exist_ok=True)

    wait_for_preprocessing()

    for stage in ("onset", "sequence", "lighting"):
        rc = run_stage(stage, max_epochs=100)
        if rc != 0:
            print(f"\nERROR: {stage} training exited with code {rc}. Stopping pipeline.")
            sys.exit(rc)
        print(f"\n{stage} training finished successfully.")

    print("\nAll three stages complete! Check outputs/ for checkpoints.")


if __name__ == "__main__":
    main()
