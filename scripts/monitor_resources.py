"""Lightweight GPU/CPU/RAM sampler for long auto-researcher runs.

Writes one JSON line per sample to the output path so utilization curves can
be reviewed after an overnight sweep to decide if batch size / worker counts
should be tuned.

Usage:
    python scripts/monitor_resources.py \
        --output experiments/monitor.jsonl \
        --interval-sec 30

Intended to run in the background alongside ``scripts/auto_research.py``.
"""

from __future__ import annotations

import argparse
import json
import logging
import signal
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import psutil

logger = logging.getLogger("monitor")


_GPU_QUERY = (
    "utilization.gpu,utilization.memory,memory.used,memory.total,"
    "temperature.gpu,power.draw"
)


def _sample_gpu() -> list[dict[str, float]] | None:
    """Read per-GPU stats via nvidia-smi. Returns None if nvidia-smi fails."""
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                f"--query-gpu={_GPU_QUERY}",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
    except (subprocess.SubprocessError, FileNotFoundError):
        return None

    gpus = []
    for idx, line in enumerate(out.stdout.strip().splitlines()):
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 6:
            continue
        try:
            gpus.append(
                {
                    "idx": idx,
                    "util_pct": float(parts[0]),
                    "mem_util_pct": float(parts[1]),
                    "mem_used_mib": float(parts[2]),
                    "mem_total_mib": float(parts[3]),
                    "temp_c": float(parts[4]),
                    "power_w": float(parts[5]) if parts[5] not in ("N/A", "") else 0.0,
                }
            )
        except ValueError:
            continue
    return gpus


def _sample_host() -> dict[str, Any]:
    vm = psutil.virtual_memory()
    load1, load5, load15 = psutil.getloadavg() if hasattr(psutil, "getloadavg") else (0, 0, 0)
    return {
        "cpu_pct": psutil.cpu_percent(interval=None),
        "cpu_count": psutil.cpu_count(logical=True),
        "load_1m": load1,
        "load_5m": load5,
        "load_15m": load15,
        "ram_used_gib": vm.used / (1024**3),
        "ram_total_gib": vm.total / (1024**3),
        "ram_pct": vm.percent,
    }


class _Stopped:
    """Graceful-shutdown flag toggled by SIGINT/SIGTERM."""

    def __init__(self) -> None:
        self.flag = False

    def set(self, *_: Any) -> None:
        self.flag = True


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=Path("experiments/monitor.jsonl"))
    p.add_argument("--interval-sec", type=float, default=30.0)
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    stopped = _Stopped()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(sig, stopped.set)
        except (ValueError, OSError):
            pass

    # Prime psutil.cpu_percent so the first real sample isn't 0.
    psutil.cpu_percent(interval=None)

    logger.info("monitor started → %s (interval=%ss)", args.output, args.interval_sec)
    with args.output.open("a", encoding="utf-8") as fh:
        while not stopped.flag:
            row = {
                "t": datetime.now(UTC).isoformat(),
                "host": _sample_host(),
                "gpus": _sample_gpu() or [],
            }
            fh.write(json.dumps(row) + "\n")
            fh.flush()
            # Sleep in small chunks so Ctrl+C is responsive.
            slept = 0.0
            while slept < args.interval_sec and not stopped.flag:
                time.sleep(min(1.0, args.interval_sec - slept))
                slept += 1.0

    logger.info("monitor stopped")
    return 0


if __name__ == "__main__":
    sys.exit(main())
