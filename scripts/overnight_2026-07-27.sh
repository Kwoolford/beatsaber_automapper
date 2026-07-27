#!/usr/bin/env bash
# 2026-07-27 — close the two flow/ergonomics gaps that eval-suite v2 axis A1 exposed.
#
# A1 (evaluation/flow.py) measured two real defects the old scorecard was blind to:
#   travel     our hands move ~50% further per SECOND than human hands
#              (cohort shift +2.48 human-MADs; 6.0 vs human median 4.0)
#   crossover  0.000 vs a human median of 0.218 — enforce_color_separation moves
#              EVERY wrong-side note, so our maps never cross hands over at all
#
# Two new gated levers (both default OFF, prior behaviour unchanged):
#   LAYOUT_TRAVEL_PENALTY=S  decode-time penalty on placing a note far from the
#                            same hand's previous note, scaled by 1/slot-gap
#   COLOR_SEP_MODE=extreme   only relocate FULLY wrong-side notes (red col 3 /
#                            blue col 0), keeping the mild one-column crossovers
#                            the model chose. `off` = full ablation.
#
# ARMS (control = prod, already cached):
#   prod       current production (control)
#   tp1/2/4    travel penalty strength sweep
#   xsep_ext   crossover: keep mild crossovers
#   xsep_off   crossover: no separation at all (ablation — expect parity damage)
#   tp2_xsep   both levers, to check they compose rather than fight
#
# VERDICT LOGIC (printed at the end):
#   An arm WINS if it lowers `flow_gap` vs prod while HOLDING every guard:
#     viol == 0, density_corr pass-count not worse than prod, row_conc <= 0.60,
#     grid_coverage >= 0.85, and no metric overshooting PAST human (a |shift|
#     that flips sign and grows is over-correction, not a fix — the whole point
#     of scoring against the human distribution instead of a point target).
#   Smoke test showed tp2 overshoots travel BELOW human (3.13 vs human 4.0), so
#   the expected winner is tp1; tp4 is included to confirm the trend direction.
#   If the best arm only trades one axis for another, promote NOTHING and report.
set -uo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate

ARMS="prod,tp1,tp2,tp4,xsep_ext,xsep_off,tp2_xsep"

echo "=============================================================="
echo "STEP 1 — sweep flow levers over the 24-song set"
echo "arms: $ARMS"
echo "=============================================================="
python scripts/eval_sweep.py sweep --arms "$ARMS"

echo
echo "=============================================================="
echo "STEP 2 — control battery: do the winning maps still beat the"
echo "         degenerate controls? (a lever that games one axis"
echo "         must not break the suite's separation)"
echo "=============================================================="
python scripts/audit_eval_suite.py --n 12 --json outputs/eval_audit_2026-07-27_post.json 2>&1 \
  | grep -v "INFO\|WARNING"

echo
echo "=============================================================="
echo "COMPLETE — flow-lever sweep"
echo "=============================================================="
echo "Read the 'flow / ergonomics' table: the winning arm has the LOWEST"
echo "flow_gap with viol 0 and no guard regressed. Check each shift is"
echo "moving TOWARD 0, not flipping sign (overshoot past human)."
