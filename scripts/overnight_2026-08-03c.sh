#!/usr/bin/env bash
# W1a / TRACK B — DOES THE INSTRUMENT-AWARE STAGE-1 SEPARATE THE BEAT FROM THE OFFBEAT?
#
# The promoted baseline uses beat `version_4`, which has only drum_proj + mix_proj
# and no instrument projection. Measured 2026-08-03: at multi-instrument events it
# prefers the correct slot over the one a half-beat away only 57.3% of the time,
# and corr(that win_rate, our halfbeat_rate) = -0.494 across 23 songs -- the songs
# where it cannot separate them are the songs where we play the offbeat.
#
# B-1 (`version_8`, epoch 12, --use-instr) is the honest instrument retrain that
# already exists. If Track B is the right answer for W1, ITS probabilities should
# separate the two slots better. This costs ~20 min of GPU and no retrain.
#
#   win_rate(v8) - win_rate(v4) clearly > 0  => Track B is validated FOR THIS DEFECT,
#                                              and pushing it further is justified.
#   ~ 0                                      => the instrument rebuild as built does
#                                              NOT fix phase discrimination. W1 needs
#                                              a different idea, and that is worth
#                                              knowing BEFORE committing a retrain.
#
# Paired by construction: same songs, same seed, same everything but the Stage-1
# checkpoint and --use-instr.
set -u
cd "$(dirname "$0")/.."
L=logs/overnight/probsphase_instr_2026-08-03.log
D=outputs/probs_phase_instr_2026-08-03
mkdir -p "$D" logs/overnight
: > "$L"
echo "=== START $(date -Is) ===" >> "$L"

BEAT_CKPT=$(ls logs/beat_classifier/version_8/checkpoints/beat-epoch=12-val_f1_avg_tol=*.ckpt | head -1)
LAYOUT_CKPT="logs/layout_phrase/version_10/checkpoints/layout-epoch=09-val_token_acc=0.865.ckpt"
echo "beat ckpt (B-1 instrument, epoch 12): $BEAT_CKPT" >> "$L"

for f in data/eval_songset/*.ogg; do
  sid=$(basename "$f" .ogg)
  [ -f "$D/$sid.npz" ] && continue
  echo "--- $sid ---" >> "$L"
  BEAT_PROBS_DUMP="$D/$sid.npz" .venv/bin/python scripts/generate.py "$f" \
      --v7 --beat-ckpt "$BEAT_CKPT" --layout-ckpt "$LAYOUT_CKPT" --use-instr \
      --difficulty Expert --section-gate loud_only --seed 0 \
      --output "$D/$sid.zip" >> "$L" 2>&1
done

echo "" >> "$L"; echo "##### B-1 INSTRUMENT MODEL #####" >> "$L"
.venv/bin/python scripts/eval_probs_phase.py --dumps "$D" >> "$L" 2>&1
echo "" >> "$L"; echo "##### version_4 CONTROL (same songs, already cached) #####" >> "$L"
.venv/bin/python scripts/eval_probs_phase.py --dumps outputs/probs_phase_2026-08-03 >> "$L" 2>&1
echo "=== COMPLETE $(date -Is) ===" >> "$L"
