#!/usr/bin/env bash
# K2 — SPEED-CONDITIONED DIAGONALS: does the fix survive the corpus?
#
# THE DEFECT, measured on 200 strictly-Expert human maps against 24 of ours:
#
#   local nps    0-4     4-7    7-10     10+
#   human      0.355   0.346   0.301   0.236    <- backs off where it punishes
#   ours       0.466   0.476   0.536   0.631    <- leans in
#
# 2.7x the human diagonal share in the fastest passages. BEAT_SPEED_DIAG rewrites
# diagonals to the vertical they lean toward, but ONLY above a threshold, because
# Kyle wants broad diagonals kept in slow sections ("they get the player moving
# and feel like they are playing a grand orchestra") and objects to them only
# when fast ("difficult but possible, and not preferred").
#
# PROBE (1f333, seed 0, at 6:0.6): 10+ band 0.611 -> 0.354, slope +0.016 ->
# -0.017 (human -0.011), the two SLOW bands untouched, flow angle_change
# 17.9 -> 18.1, nps unchanged. One song -- hence this.
#
# ARMS: trim+ev03 (control, cached) vs +sd 6:0.6 vs +sd 6:1.0, 3 seeds each.
#
# WHAT TO WATCH, written before the run:
#
#   flow      THE risk. This rewrites cut directions after the fact, which is
#             precisely the kind of thing that breaks swing continuity. The probe
#             moved angle_change by 0.2 degrees, but one song proves nothing. If
#             flow degrades outside 2 sd, the lever is trading playability for a
#             statistic and should be rejected however good the diagonal numbers
#             look.
#   idiom     A3 measures the direction VOCABULARY. Pushing four directions into
#             two must move it; the question is whether it moves TOWARD human
#             (which leads with the vertical axis) or collapses variety. Watch
#             idiom_coverage, not just the gap.
#   handrole  parity repairs run after this; if handrole moves, the rewrite is
#             disturbing hand structure and not merely direction.
#
# ⚠️ The 2026-07-27 landmine in reverse: over-diversifying direction is what
# CAUSED K2, so a lever that reduces variety is directionally right -- but the
# same reasoning that made "more diversity = more human" false makes "less
# diversity = more human" false too. The target is the human LEVEL per speed
# band, not monotone reduction. 6:1.0 exists to show where overshoot begins.

set -uo pipefail
cd /home/kyle/repos/beatsaber_automapper
source .venv/bin/activate

LOG=logs/overnight/speeddiag_2026-08-03.log
mkdir -p logs/overnight
exec > >(tee -a "$LOG") 2>&1

echo "=== K2 SPEED DIAGONALS (3 arms x 3 seeds; control cached) START $(date -Is) ==="

ARMS="tf_hl014_ds048_trim_ev03,tf_trim_ev03_sd6,tf_trim_ev03_sd6f"
python scripts/eval_sweep.py sweep --arms "$ARMS" --seeds 3
echo "=== SWEEP DONE $(date -Is) ==="

echo
echo "=== K2 BANDS PER ARM (seed 0) $(date -Is) ==="
for arm in tf_hl014_ds048_trim_ev03 tf_trim_ev03_sd6 tf_trim_ev03_sd6f; do
  echo "--- $arm ---"
  python scripts/eval_diagonal_vs_speed.py \
    --maps "outputs/eval_sweep_cache/${arm}#s0__*.zip" --label "$arm" 2>&1 \
    | grep -E "0-4|diagonal share by|maps whose"
done
echo
echo "human (200 strict Expert):  0.355  0.346  0.301  0.236   slope -0.01141"

echo
echo "=== READ ==="
echo "  Promotable if: the 10+ band falls toward 0.236, the two SLOW bands stay"
echo "  put (Kyle wants those diagonals), and flow/idiom/handrole stay inside"
echo "  2 sd. Diagonals fixed at the cost of flow is NOT a win."
echo "=== COMPLETE $(date -Is) ==="
