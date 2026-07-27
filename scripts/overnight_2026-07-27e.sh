#!/usr/bin/env bash
# 2026-07-27 (part E) — STAGE-1 IOI PRIOR: the actual rhythm experiment.
#
# Everything before this treated rhythm as a layout problem. It is not:
#   * rule_mapper.py scores rhythm 2.41 on OUR onsets and 0.25 on HUMAN onsets
#     with identical note-placement code, so rhythm is inherited entirely from
#     the onset layer.
#   * part D ruled out tempo detection: regenerating with the human-declared BPM
#     moves rhythm only 2.41 -> 2.37, and on the six songs that were actually
#     mis-detected it gets WORSE (1.96 -> 2.13), because correcting tempo removes
#     the artificial inflation beat-domain metrics get from tempo error.
#
# So this changes WHICH SLOTS Stage-1 selects. Within each density-allocated
# window, instead of taking the top-k by probability -- which just reproduces the
# audio's own periodicity, collapsing 92% of our intervals onto 1/8 -- the pick
# maximises model prob + lambda * human P(interval | previous interval), with the
# interval state carried across windows. The bigram is mined from 300 human maps
# (outputs/ioi_human_model.json) and is strongly diagonal: humans hold a
# subdivision for a run, then change gear.
#
# The window ALLOCATION is untouched, so the validated density_corr behaviour is
# preserved and only interval structure changes.
#
# ARMS: ioi05 / ioi1 / ioi2 / ioi4 (strength sweep) + ioi2_best (with the proven
# flow and crossover levers) + prod as control.
#
# SIGNIFICANCE BAR — the measured noise floor from two identical prod runs:
#   flow 0.03 | rhythm 0.08 | idiom 0.09 | handrole 0.29
# A difference smaller than its axis floor is NOT a result.
#
# VERDICT LOGIC
#   rhythm_gap drops well past 0.08 with density_corr, parity and note count held
#       => the rhythm gap really is onset selection and this is the fix; promote
#          the best strength and re-run the control battery.
#   rhythm improves but density_corr or note count regresses
#       => the prior is buying rhythm by breaking density; retune, do not promote.
#   rhythm does not move
#       => within-window selection is not enough, and the next step is the window
#          allocation itself or a Stage-1 retrain with a rhythm-aware objective.
set -uo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate

ARMS="prod,ioi05,ioi1,ioi2,ioi4,ioi2_best"

echo "=============================================================="
echo "STEP 1 — sweep the Stage-1 IOI prior (arms: $ARMS)"
echo "=============================================================="
python scripts/eval_sweep.py sweep --arms "$ARMS" --true-bpm

echo
echo "=============================================================="
echo "STEP 2 — consolidated 4-axis scorecard per arm"
echo "=============================================================="
for ARM in prod ioi05 ioi1 ioi2 ioi4 ioi2_best; do
  echo
  python -m beatsaber_automapper.evaluation.scorecard \
      outputs/eval_sweep_cache/${ARM}__*.zip --label "$ARM" 2>&1 | grep -v "INFO\|WARNING"
done

echo
echo "=============================================================="
echo "COMPLETE — Stage-1 IOI prior"
echo "=============================================================="
echo "Read rhythm_gap against the 0.08 noise floor, and check density_corr,"
echo "parity and note counts have not been traded away for it."
