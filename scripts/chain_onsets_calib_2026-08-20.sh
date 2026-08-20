#!/usr/bin/env bash
# Wait for the running onset-cache job, then cache the CALIB and HELDOUT spans of
# the mapjudge corpus so the audio axis can be calibrated with its own, disjoint
# conformal set. The DIST span is already covered by the first run.
# ⚠️Explicit PID, never pgrep -f: pgrep inside a script matches its own command line
# and returns immediately (TODO landmine).
set -u
cd /home/kyle/repos/beatsaber_automapper
source .venv/bin/activate
while kill -0 7634 2>/dev/null; do sleep 30; done
echo "onset job $PID finished; caching CALIB span"
python scripts/build_onset_cache.py --skip 2415 --from-raw 400
echo "caching HELDOUT span"
python scripts/build_onset_cache.py --skip 3830 --from-raw 400
echo "ONSET_SPANS_COMPLETE"
