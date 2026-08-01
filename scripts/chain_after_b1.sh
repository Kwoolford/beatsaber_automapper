#!/usr/bin/env bash
# Start the A6 hand-lead sweep the moment the B-1 scoring sweep frees the GPU.
#
# Hands-free sessions lose more time to an idle GPU between jobs than to any slow
# job, and the two sweeps are independent (different arms, and eval_sweep's own
# .sweep.lock serialises cache writes anyway, so this can never overlap them).
# Waits on the eval_sweep PROCESS rather than on the log, because a log line only
# says an arm finished, not that the script exited.
set -uo pipefail
cd /home/kyle/repos/beatsaber_automapper

echo "[chain] waiting for B-1 sweep to exit ($(date -Is))"
while pgrep -f "eval_sweep.py sweep --arms b1_e00" >/dev/null 2>&1; do
  sleep 60
done
echo "[chain] B-1 gone, waiting for its scorecard pass to finish too"
while pgrep -f "overnight_2026-07-30.sh" >/dev/null 2>&1; do
  sleep 60
done
echo "[chain] launching hand-lead sweep ($(date -Is))"
exec bash scripts/overnight_2026-08-01.sh
