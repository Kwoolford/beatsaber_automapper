#!/usr/bin/env bash
# 2026-07-27 (part C) — the HAND-ROLE axis (A6) + promotion candidate.
#
# Chained behind part B. A6 is the discovery this whole run is organised around:
# found by READING a map next to its human counterpart (scripts/map_view.py), not
# by any statistic. Human mappers give ONE hand the lead within a passage while
# the other punctuates, then swap; our maps split every bar evenly, so they are
# balanced at every scale where human maps are balanced only GLOBALLY.
#
#   metric                  human   ours
#   local asymmetry         0.115   0.031
#   dominant-hand swap      0.461   0.269
#   handrole_gap            0.34    3.50   <- our worst axis, worse than RANDOM
#
# ARMS
#   hr05/hr075/hr10   BEAT_HAND_ROLE strength. Reassigns which hand plays each
#                     already-selected onset (times untouched), targeting the
#                     human reference. Strength 1.0 overshoots asymmetry on a
#                     single-song probe (0.241 vs 0.115), so sweep downward.
#                     NOTE: single-song probes are smoke tests only — the
#                     BEAT_HAND_INTERLEAVE lever looked good on one song and
#                     failed on all 24, because that song was half-tempo.
#   best              the two PROVEN levers alone: LAYOUT_TRAVEL_PENALTY=1
#                     (flow 0.81 -> 0.30 PASS) + COLOR_SEP_MODE=extreme
#                     (idiom 1.84 -> 0.30 PASS). No interleave (it made rhythm
#                     worse), no idiom bonus (weaker than xsep at the same job).
#   best_hr           best + hand role = candidate to pass all four axes.
#
# VERDICT LOGIC
#   Promote the arm with the most axes PASS and NONE regressed vs prod on the
#   consolidated scorecard, parity 0. Watch two specific traps:
#     - OVERSHOOT: a shift that flips sign and grows is over-correction (tp4 did
#       exactly this: flow 1.77 with spread 0.00, every map identical).
#     - DENSITY: hand-role de-doubles the map, which alone would delete ~38% of
#       the notes. The budget is inflated to compensate, but CHECK note counts and
#       density_corr against prod before promoting.
set -uo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate

for _ in $(seq 1 480); do
  grep -q "COMPLETE — rhythm + idiom lever sweep" logs/overnight/rhythm_idiom_2026-07-27.log 2>/dev/null && break
  sleep 30
done
echo "part B done; starting part C"

ARMS="hr05,hr075,hr10,best,best_hr"

echo "=============================================================="
echo "STEP 1 — sweep hand-role (A6) + the promotion candidate"
echo "=============================================================="
python scripts/eval_sweep.py sweep --arms "$ARMS"

echo
echo "=============================================================="
echo "STEP 2 — consolidated 4-axis scorecard per arm (the verdict)"
echo "=============================================================="
for ARM in prod best hr05 hr075 hr10 best_hr; do
  echo
  python -m beatsaber_automapper.evaluation.scorecard \
      outputs/eval_sweep_cache/${ARM}__*.zip --label "$ARM" 2>&1 | grep -v "INFO\|WARNING"
done

echo
echo "=============================================================="
echo "COMPLETE — hand-role sweep"
echo "=============================================================="
echo "Promote the arm with most axes PASS, none regressed, parity 0, and note"
echo "counts not collapsed vs prod. Then bake its env into the production"
echo "defaults and re-run the control battery to confirm separation holds."
