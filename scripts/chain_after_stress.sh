#!/usr/bin/env bash
# Run the noise-floor measurement once the stress sweep frees the GPU.
set -uo pipefail
cd /home/kyle/repos/beatsaber_automapper
echo "[chain3] waiting for the stress sweep to exit ($(date -Is))"
while pgrep -f "overnight_2026-08-01b.sh" >/dev/null 2>&1; do sleep 60; done
echo "[chain3] launching noise-floor sweep ($(date -Is))"
exec bash scripts/overnight_2026-08-01c.sh
