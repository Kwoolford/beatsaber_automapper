#!/usr/bin/env bash
# K2 REACHABILITY — the lever Kyle's correction actually points at
#
# He rejected the global diagonal thin: diagonals "can be fun in fast passages",
# the objection is an outside-corner swing followed by a note that is hard to
# reach. "They should still be playable though that's the core problem not that
# they are diagonal."
#
# Measured (150 human Expert maps vs ours):
#   hard_rate (>=3 units within 0.3s)   ours 0.136   human 0.059   <- the defect
#   hard_given_diagonal                 ours 0.087   human 0.077   <- blameless
#   reach_p90                           ours 3.16    human 3.61    <- humans reach FURTHER
# Humans make bigger movements and give them TIME. The diagonal thin moved
# hard_rate by exactly 0.000.
#
# ⚠️ THE FAILURE MODE TO WATCH FOR, written before the run: this lever could
# "fix" hard_rate by shrinking every movement, which would trade playability for
# timidity and push reach_p90 further BELOW the human 3.61. The 1f333 probe at
# strength 0.7 already over-corrected (hard_rate 0.035 vs human 0.059) and pulled
# reach_p90 3.61 -> 3.16. So the arm to prefer is whichever lands hard_rate near
# 0.059 while keeping the MOST reach tail -- not whichever minimises hard_rate.
#
# Also watch flow and idiom: this moves note POSITIONS, which is more invasive
# than the diagonal rewrite (directions only). If flow degrades outside 2 sd the
# repositioning is doing harm.

set -uo pipefail
cd /home/kyle/repos/beatsaber_automapper
source .venv/bin/activate
LOG=logs/overnight/reach_2026-08-03.log
mkdir -p logs/overnight
exec > >(tee -a "$LOG") 2>&1

echo "=== K2 REACHABILITY (3 arms x 3 seeds; control cached) START $(date -Is) ==="
python scripts/eval_sweep.py sweep --arms tf_hl014_ds048_trim_ev03,tf_trim_ev03_rc05,tf_trim_ev03_rc07 --seeds 3
echo "=== SWEEP DONE $(date -Is) ==="

echo
echo "=== REACHABILITY METRICS PER ARM (3 seeds pooled) $(date -Is) ==="
for arm in tf_hl014_ds048_trim_ev03 tf_trim_ev03_rc05 tf_trim_ev03_rc07; do
  python scripts/eval_reachability.py --maps "outputs/eval_sweep_cache/${arm}#s*__*.zip" --label "$arm" 2>&1 | grep -E "===|reach_|hard_|corner_"
done
echo
echo "human (150 strict Expert): reach_p90 3.6056  hard_rate 0.0592  corner 0.1845"
echo
echo "=== READ ==="
echo "  Prefer the arm landing hard_rate near 0.059 while KEEPING the most"
echo "  reach_p90. Minimising hard_rate is not the goal -- humans reach further"
echo "  than we do, they just give it time."
echo "=== COMPLETE $(date -Is) ==="
