#!/usr/bin/env bash
# 2026-07-27 (part G) — spacing-aware HAND OFFSET: keep the rhythm win, recover flow.
#
# Part F established that BEAT_HAND_OFFSET essentially solves the rhythm axis
# (gap 2.37 -> 0.26, every sub-metric on the human value) and improves hand role,
# but costs flow. The flow loss is via `angle_change` (19.8 -> 23.1) and NOT via
# `travel` (5.73 -> 5.67, unchanged), so the travel penalty cannot fix it --
# confirmed by ho05_best, which only recovered flow 1.68 -> 1.31 and broke parity
# further (4 violations).
#
# Moving a note changes WHICH HAND plays WHEN, which shifts the wrist-rotation
# sequence. BEAT_HAND_OFFSET_SPACING=1 picks the neighbour that leaves the moving
# hand's own gaps more even, as the cheapest proxy for keeping that sequence
# smooth, instead of simply taking the higher-probability side.
#
# ARMS: ho03s / ho05s (spacing-aware at the two strengths that mattered) and
# ho03s_best (with the proven flow + crossover levers). Control = prod.
#
# SIGNIFICANCE BAR (measured noise floor): flow 0.03 | rhythm 0.08 | idiom 0.09 |
# handrole 0.29.
#
# VERDICT: promote if rhythm stays well under its 0.70 bar, hand role improves,
# flow returns to at least prod's 0.71, parity is 0, and density/note count hold.
# ho03 (non-spacing) remains the fallback candidate: rhythm 0.50, handrole 2.11,
# idiom unchanged, parity clean, flow 1.36.
set -uo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
ARMS="ho03s,ho05s,ho03s_best"
echo "=== sweep: $ARMS ==="
python scripts/eval_sweep.py sweep --arms "$ARMS" --true-bpm
echo
echo "=== consolidated 4-axis scorecard ==="
for A in prod ho03 ho03s ho05s ho03s_best; do
  echo; python -m beatsaber_automapper.evaluation.scorecard \
    outputs/eval_sweep_cache/${A}__*.zip --label "$A" 2>&1 | grep -v "INFO\|WARNING"
done
echo; echo "COMPLETE — spacing-aware hand offset"
