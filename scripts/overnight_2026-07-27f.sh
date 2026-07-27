#!/usr/bin/env bash
# 2026-07-27 (part F) — HAND OFFSET: the unified fix for A2 (rhythm) + A6 (hand role).
#
# Found by dumping beat_probs next to the human note times on the same slot grid:
# our maps place a note on an odd 16th ZERO times in 679 slots; the human map
# puts 248 there and those are exactly the slots we miss. Cause is hand lockstep
# -- human hands are interleaved by a 16th 32% of the time, ours 0.2% -- and the
# union of two hands can only reach an odd 16th if they are offset. So the rhythm
# gap and the hand-role gap are ONE defect.
#
# The lever MOVES one hand by a 16th at shared slots rather than deleting it,
# which is what BEAT_HAND_ROLE did (costing 24% of the notes and hurting rhythm).
#
# SIGNIFICANCE BAR (measured noise floor): flow 0.03 | rhythm 0.08 | idiom 0.09 |
# handrole 0.29. Anything smaller is not a result.
#
# VERDICT: promote if rhythm AND handrole both improve past their floors with
# density_corr, parity and note count held. Watch for OVERSHOOT -- the 0.8 probe
# pushed odd-16ths to 0.409 against a human 0.238 and switch rate fell back.
set -uo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
ARMS="prod,ho03,ho05,ho07,ho05_best"
echo "=== sweep: $ARMS ==="
python scripts/eval_sweep.py sweep --arms "$ARMS" --true-bpm
echo
echo "=== consolidated 4-axis scorecard ==="
for A in prod ho03 ho05 ho07 ho05_best; do
  echo; python -m beatsaber_automapper.evaluation.scorecard \
    outputs/eval_sweep_cache/${A}__*.zip --label "$A" 2>&1 | grep -v "INFO\|WARNING"
done
echo; echo "COMPLETE — hand offset sweep"
