#!/usr/bin/env bash
# 2026-07-27 (part B) — close the RHYTHM (A2) and IDIOM (A3) gaps.
#
# Chained after overnight_2026-07-27.sh (the flow-lever sweep): waits for that
# log to print its COMPLETE line, so only one GPU job runs at a time.
#
# The consolidated scorecard says current production FAILS all three axes:
#   flow 0.81 (bar 0.50) | rhythm 2.41 (bar 0.70) | idiom 1.84 (bar 1.00)
# A held-out human cohort passes all three (0.13 / 0.25 / 0.31), so the bars are
# reachable. Part A swept the flow levers; this sweeps the two bigger gaps.
#
# ARMS
#   il5/il7/il9      BEAT_HAND_INTERLEAVE — our two hands fire together on 85.6%
#                    of beats vs a human 17.5%, which is what makes the union
#                    rhythm metronomic. Probe: il0.5 fixed simultaneity (0.12)
#                    but not the rhythm; il0.9 fixed the rhythm (cond-entropy
#                    0.49, switch 12.2 vs human 0.54/13.7) but drove hands too
#                    far apart. The sweet spot should be between.
#   ib1/ib2/ib3      LAYOUT_IDIOM_BONUS — boost cut directions that complete a
#                    known human idiom. Probe at 2.0: coverage 0.759 -> 0.946
#                    (human 0.919), top50 essentially human, viol 0.
#   il7_tp1_xsep     rhythm + flow levers, do they fight?
#   combo            everything that looked good = candidate next production
#
# VERDICT LOGIC
#   Promote an arm only if the CONSOLIDATED SCORECARD improves: axis gaps drop
#   toward the bars WITHOUT any axis regressing, spread stays >= 0.35 (no mode
#   collapse), and viol stays 0. Watch for OVERSHOOT — a shift that flips sign
#   and grows is over-correction, not a fix. The whole reason we score against
#   the human distribution rather than a point target is to make that visible.
#   If an arm wins on one axis by breaking another, promote nothing and report.
set -uo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate

PART_A_LOG=logs/overnight/flow_levers_2026-07-27.log

echo "waiting for part A to finish ($PART_A_LOG) …"
for _ in $(seq 1 480); do            # up to ~4h
  if grep -q "COMPLETE — flow-lever sweep" "$PART_A_LOG" 2>/dev/null; then
    echo "part A complete; starting part B"
    break
  fi
  sleep 30
done

ARMS="il5,il7,il9,ib1,ib2,ib3,il7_tp1_xsep,combo"

echo "=============================================================="
echo "STEP 1 — sweep rhythm (A2) + idiom (A3) levers"
echo "arms: $ARMS"
echo "=============================================================="
python scripts/eval_sweep.py sweep --arms "$ARMS"

echo
echo "=============================================================="
echo "STEP 2 — consolidated scorecard per arm (the actual verdict)"
echo "=============================================================="
for ARM in prod il5 il7 il9 ib1 ib2 ib3 il7_tp1_xsep combo; do
  echo
  python -m beatsaber_automapper.evaluation.scorecard \
      outputs/eval_sweep_cache/${ARM}__*.zip --label "$ARM" 2>&1 \
    | grep -v "INFO\|WARNING"
done

echo
echo "=============================================================="
echo "COMPLETE — rhythm + idiom lever sweep"
echo "=============================================================="
echo "Promote the arm whose scorecard has the most axes PASS with none"
echo "regressed vs prod. Check shifts move toward 0 rather than flipping sign."
