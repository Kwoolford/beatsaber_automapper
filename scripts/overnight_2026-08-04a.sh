#!/usr/bin/env bash
# W2 round 2 — gamma x budget. Does lowering DENSITY_SELECT_GAMMA while raising
# the budget put the extra notes in BETTER places?
#
# Round 1 established two things, one of which killed the obvious plan:
#   * the marginal note is much worse than the average note (added notes are
#     31.8% k=0 vs our existing 9.5%, and 0.9% k>=3 vs 21.3%);
#   * ★ HUNGER, which Kyle graded A+, sits at 0.650 of its own human map's
#     density, while FALLEN KINGDOM, which he called "really empty", sits at
#     0.781 -- DENSER relative to its human. The density ratio is BACKWARDS from
#     his verdict, so "match the human note count" is refuted as a target.
#
# So this sweep is NOT about reaching a density. It asks the one question round 1
# left open: gamma 2.5 concentrates budget into LOUD windows, so does relaxing it
# move notes into the quiet windows that hold good onsets?
#
#   nb130_g15 / nb130_g10   more budget AND flatter allocation
#   g15                     flatter allocation at the SAME budget  <- the clean
#                           test of allocation alone, judge this one first
#
# ⚠️PRE-REGISTERED: gamma was raised to 2.5 on 2026-06-30 to buy density_corr and
# lowering it previously wrecked handrole (1.84 -> 2.70). Expect a cost. The
# question is whether the k-distribution of the ADDED notes improves -- judge with
# scripts/view_ab_diff.py, NOT the axis count, and NOT nps.
set -u
cd "$(dirname "$0")/.."
L=logs/overnight/gamma_budget_2026-08-04.log
mkdir -p logs/overnight
: > "$L"
echo "=== START $(date -Is) ===" >> "$L"
.venv/bin/python scripts/eval_sweep.py sweep \
    --arms tf_trim_ev03_rc05,g15,nb130_g15,nb130_g10 --seeds 3 >> "$L" 2>&1

echo "" >> "$L"; echo "##### WHERE DID THE NOTES GO? #####" >> "$L"
for arm in g15 nb130_g15 nb130_g10; do
  echo "--- $arm vs control, Fallen Kingdom ---" >> "$L"
  .venv/bin/python scripts/view_ab_diff.py --song 1f8d6 --no-plot \
     --a "outputs/eval_sweep_cache/tf_trim_ev03_rc05#s0__1f8d6.zip" \
     --b "outputs/eval_sweep_cache/${arm}#s0__1f8d6.zip" >> "$L" 2>&1
done
echo "=== COMPLETE $(date -Is) ===" >> "$L"
