#!/usr/bin/env bash
# Chain the A8 onset-cache build behind the running noise-floor sweep.
#
# NOTE ON THE GUARD: the 2026-08-01 retro found two chain scripts whose
# `pgrep -f <name>` guard matched their OWN command line and so never waited at
# all. This waits on an explicit PID passed in by the caller instead, which cannot
# match itself.
set -uo pipefail
cd /home/kyle/repos/beatsaber_automapper
source .venv/bin/activate
WAIT_PID="${1:?usage: chain_onset_cache.sh <pid-to-wait-for>}"
LOG=logs/overnight/onset_cache_2026-08-01.log
mkdir -p logs/overnight
exec > >(tee -a "$LOG") 2>&1
echo "=== waiting for pid $WAIT_PID (noise-floor sweep) $(date -Is) ==="
while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 20; done
echo "=== sweep done, building onset cache $(date -Is) ==="
python scripts/build_onset_cache.py
python scripts/build_onset_cache.py --from-raw 80
echo "=== CALIBRATE A8 $(date -Is) ==="
python scripts/calibrate_alignment.py --write
echo "=== COMPLETE $(date -Is) ==="
