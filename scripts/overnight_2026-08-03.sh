#!/usr/bin/env bash
# K1 TAIL TRIM — does removing post-music notes cost anything?
#
# BEAT_TRIM_TAIL cuts selected slots after (last librosa onset + grace). Probe on
# 1f8d6 at grace 0.5 s: tail notes 11 -> 2, tail seconds 4.43 -> 0.53, 494 -> 490
# notes. This run asks the only question a single-song probe cannot answer:
# **does it regress anything across the corpus?**
#
# 3 seeds per arm, because 1 seed decides nothing (P0). The control's three seeds
# are already cached from seedrepro_2026-08-02, so only the trim arm generates.
#
# WHAT TO EXPECT — write it down before reading the table:
#
#   alignment   should improve slightly or not move. The trim removes notes that
#               matched no onset, so precision can only go up -- but the effect
#               is small (18 slots out of ~800 on the worst song) and alignment's
#               unpaired sd is 0.113, so **a null here is the likely outcome and
#               is not a failure**. Look at the PAIRED row: alignment pairs 4.3x
#               tighter (sd 0.033), so it is the one axis where a small real
#               effect can actually be resolved.
#   nps         should fall a hair. Notes only ever get removed.
#   the rest    should NOT move. If flow/idiom/handrole/rhythm/playfeel shift
#               outside 2 sd, the trim is doing something beyond its remit and
#               wants investigating before it goes anywhere near a default.
#
# THIS RUN CANNOT SETTLE K1. Drift moved only 0.429 -> 0.378 on the probe against
# a human p90 of 0.145. The decay is the density-tracking defect (notes/s holding
# flat while onsets/s falls), which is a separate lever not built yet.

set -uo pipefail
cd /home/kyle/repos/beatsaber_automapper
source .venv/bin/activate

LOG=logs/overnight/trimtail_2026-08-03.log
mkdir -p logs/overnight
exec > >(tee -a "$LOG") 2>&1

echo "=== K1 TAIL TRIM (2 arms x 3 seeds; control cached) START $(date -Is) ==="

python scripts/eval_sweep.py sweep --arms tf_hl014_ds048,tf_hl014_ds048_trim --seeds 3
echo "=== SWEEP DONE $(date -Is) ==="

echo
echo "=== K1 SUB-METRICS: did the tail notes actually go? $(date -Is) ==="
for arm in tf_hl014_ds048 tf_hl014_ds048_trim; do
  echo "--- $arm (seed 0) ---"
  python scripts/eval_align_drift.py --human --n 60 \
    --maps "outputs/eval_sweep_cache/${arm}#s0__*.zip" --label "$arm" 2>&1 \
    | sed -n '/=== HUMAN vs/,$p'
done

echo
echo "=== READ ==="
echo "  tail_after_frac / tail_after_secs should collapse toward the human 0.0."
echo "  drift_q1_q5 should barely move -- that is the OTHER defect."
echo "  Any of flow/idiom/handrole/rhythm/playfeel moving outside 2 sd in the"
echo "  seed aggregate means the trim overreached; investigate before promoting."
echo "=== COMPLETE $(date -Is) ==="
