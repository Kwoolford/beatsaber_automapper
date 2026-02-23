"""Launch a local TensorBoard dashboard for monitoring training runs.

Opens http://localhost:6006 showing live loss/metric curves for all
training stages (onset, sequence, lighting).

Usage:
    python scripts/dashboard.py                  # port 6006, opens browser
    python scripts/dashboard.py --port 6007      # alternate port
    python scripts/dashboard.py --no-browser     # don't auto-open browser
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
import webbrowser
from pathlib import Path

# ── project root is one level above this script ──────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = ROOT / "outputs"


def _latest_runs(base: Path) -> list[tuple[str, Path]]:
    """Return (label, event_dir) pairs for the most recent version of each stage."""
    runs: list[tuple[str, Path]] = []
    for stage_dir in sorted(base.iterdir()):
        if not stage_dir.is_dir():
            continue
        versions = sorted(stage_dir.glob("version_*/events.out.tfevents.*"))
        if versions:
            # Most-recently modified event file
            latest = max(versions, key=lambda p: p.stat().st_mtime)
            runs.append((stage_dir.name, latest.parent))
    return runs


def _print_status(base: Path) -> None:
    """Print a quick summary of the latest metrics from each stage."""
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError:
        return

    for stage_dir in sorted(base.iterdir()):
        if not stage_dir.is_dir():
            continue
        versions = sorted(
            stage_dir.glob("version_*"),
            key=lambda p: int(p.name.split("_")[1]),
        )
        if not versions:
            continue
        latest_ver = versions[-1]
        event_files = list(latest_ver.glob("events.out.tfevents.*"))
        if not event_files:
            continue

        ea = EventAccumulator(str(latest_ver))
        ea.Reload()
        scalars = ea.Tags().get("scalars", [])

        metrics: dict[str, float] = {}
        for tag in scalars:
            events = ea.Scalars(tag)
            if events:
                metrics[tag] = events[-1].value

        if not metrics:
            continue

        parts = [f"  [{stage_dir.name}/{latest_ver.name}]"]
        for key in ("train_loss", "val_loss", "val_f1", "epoch"):
            if key in metrics:
                parts.append(f"{key}={metrics[key]:.4f}")
        if "lr-AdamW" in metrics:
            parts.append(f"lr={metrics['lr-AdamW']:.2e}")
        print("  ".join(parts))


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch TensorBoard training dashboard")
    parser.add_argument("--port", type=int, default=6006, help="Port to serve on")
    parser.add_argument("--no-browser", action="store_true", help="Don't auto-open browser")
    parser.add_argument("--logdir", type=Path, default=OUTPUTS_DIR, help="TensorBoard logdir")
    args = parser.parse_args()

    logdir = args.logdir
    if not logdir.exists():
        print(f"No outputs directory found at {logdir}")
        print("Start a training run first: python scripts/train.py stage=onset")
        sys.exit(1)

    print("=" * 60)
    print("  Beat Saber Automapper — Training Dashboard")
    print("=" * 60)
    print(f"\nLogdir : {logdir}")
    print(f"URL    : http://localhost:{args.port}\n")

    print("Latest metrics:")
    _print_status(logdir)
    print()

    # Find tensorboard executable in the same venv
    tb_exe = Path(sys.executable).parent / "tensorboard"
    if not tb_exe.exists():
        tb_exe = Path(sys.executable).parent / "tensorboard.exe"
    if not tb_exe.exists():
        tb_exe = "tensorboard"  # fall back to PATH

    cmd = [
        str(tb_exe),
        "--logdir", str(logdir),
        "--port", str(args.port),
        "--reload_interval", "10",  # refresh every 10 s
        "--samples_per_plugin", "scalars=0",  # show all scalar steps
    ]

    print(f"Launching: {' '.join(cmd)}")
    print("Press Ctrl+C to stop.\n")

    proc = subprocess.Popen(cmd)

    if not args.no_browser:
        # Give TensorBoard a moment to start before opening the browser
        time.sleep(3)
        url = f"http://localhost:{args.port}"
        print(f"Opening {url} ...")
        webbrowser.open(url)

    try:
        proc.wait()
    except KeyboardInterrupt:
        proc.terminate()
        print("\nDashboard stopped.")


if __name__ == "__main__":
    main()
