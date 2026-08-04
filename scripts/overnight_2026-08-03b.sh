#!/usr/bin/env bash
# W1a — IS THE OFFBEAT DEFECT IN STAGE-1's PROBABILITIES OR IN THE DECODE?
#
# halfbeat_rate says we put a note half a beat off a multi-instrument event 2.6x
# more than humans (0.245 vs 0.095). Two possible causes, and they need opposite
# fixes:
#
#   DECODE   Stage-1 prefers the event slot but selection/NMS takes the offbeat
#            => a decode lever can fix it, cheaply.
#   STAGE-1  the probability itself cannot tell the two apart
#            => no decode lever can fix it; this is Track B (the missing
#               instrument projection), the largest open item in the project.
#
# On SO TIRED ROCK alone the answer was STAGE-1: prob 0.773 on the event slot vs
# 0.763 a half-beat away, the event slot winning 49.2% of the time -- a coin
# flip -- while both sit 2.9x above a random slot. But 1f333 is a documented
# single-song probe trap, so this runs the whole eval songset before believing it.
set -u
cd "$(dirname "$0")/.."
L=logs/overnight/probsphase_2026-08-03.log
D=outputs/probs_phase_2026-08-03
mkdir -p "$D" logs/overnight
: > "$L"
echo "=== START $(date -Is) ===" >> "$L"

BEAT_CKPT="logs/beat_classifier/version_4/checkpoints/beat-epoch=11-val_f1_avg_tol=0.603.ckpt"
LAYOUT_CKPT="logs/layout_phrase/version_10/checkpoints/layout-epoch=09-val_token_acc=0.865.ckpt"

for f in data/eval_songset/*.ogg; do
  sid=$(basename "$f" .ogg)
  [ -f "$D/$sid.npz" ] && { echo "skip $sid (cached)" >> "$L"; continue; }
  echo "--- $sid ---" >> "$L"
  BEAT_PROBS_DUMP="$D/$sid.npz" .venv/bin/python scripts/generate.py "$f" \
      --v7 --beat-ckpt "$BEAT_CKPT" --layout-ckpt "$LAYOUT_CKPT" \
      --difficulty Expert --section-gate loud_only --seed 0 \
      --output "$D/$sid.zip" >> "$L" 2>&1
done

echo "" >> "$L"; echo "##### ANALYSIS #####" >> "$L"
.venv/bin/python scripts/eval_probs_phase.py --dumps "$D" >> "$L" 2>&1
echo "=== COMPLETE $(date -Is) ===" >> "$L"
