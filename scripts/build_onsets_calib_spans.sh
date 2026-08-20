#!/usr/bin/env bash
# Cache audio onsets over the CALIB and HELDOUT spans of the mapjudge corpus.
#
# WHY: `evaluation/mapjudge.py` currently scores note attributes only and is
# structurally blind to whether notes sit on the MUSIC (defects D2/D3/D4). The
# alignment axis needs onsets, and it needs its OWN conformal calibration set --
# a map scored on 23 metrics is not comparable to one scored on 21 -- so the
# CALIB and HELDOUT spans need coverage, not just DIST.
#
# STATE at 2026-08-20 close:  DIST 907/1415 cached,  CALIB 10/1415,  HELDOUT 6/1415.
# Run this, then re-run scripts/calibrate_mapjudge.py --n 1100 and the audio-mode
# p-value comes alive. Roughly 4 s/song on the 5090, so ~55 min for both spans.
#
# ⚠️This replaces chain_onsets_calib_2026-08-20.sh, which was written to wait on a
# PID and died the moment the wait finished: `set -u` plus an escaped `\$PID` in an
# echo put a LITERAL $PID into the script. The wait loop worked; the line after it
# killed the run, so the spans were never cached. A chained script must be tested by
# running it, not by reading it -- the failure was invisible until the log was read.
set -u
cd /home/kyle/repos/beatsaber_automapper
source .venv/bin/activate
echo "caching CALIB span"
python scripts/build_onset_cache.py --skip 2415 --from-raw 400
echo "caching HELDOUT span"
python scripts/build_onset_cache.py --skip 3830 --from-raw 400
echo "ONSET_SPANS_COMPLETE"
