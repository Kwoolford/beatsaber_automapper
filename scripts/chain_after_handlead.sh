#!/usr/bin/env bash
# Run the e12-push sweep once the hand-lead sweep frees the GPU.
# Same rationale as chain_after_b1.sh: an idle GPU between jobs is the main
# avoidable loss in a hands-free session. Waits on the process, not the log.
set -uo pipefail
cd /home/kyle/repos/beatsaber_automapper
echo "[chain2] waiting for the hand-lead sweep to start ($(date -Is))"
# don't race ahead of a sweep that has not begun yet
for _ in $(seq 1 240); do
  pgrep -f "overnight_2026-08-01.sh" >/dev/null 2>&1 && break
  sleep 30
done
echo "[chain2] hand-lead sweep seen, waiting for it to exit ($(date -Is))"
while pgrep -f "overnight_2026-08-01.sh" >/dev/null 2>&1; do sleep 60; done
echo "[chain2] launching e12-push sweep ($(date -Is))"
exec bash scripts/overnight_2026-08-01b.sh
