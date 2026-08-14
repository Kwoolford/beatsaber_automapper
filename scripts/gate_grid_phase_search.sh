#!/usr/bin/env bash
# THE GATE MODE `1` NEVER GOT. Ten songs through the REAL generation path, scored
# against the suite's independent cached onsets, before any full run is queued.
#
# Chosen deliberately, not at random:
#   4 songs a shift rescues most     -> does the search FIND the gain?
#   3 songs mode `1` REGRESSED       -> does the do-no-harm gate hold?
#   3 songs that were already fine   -> is it a no-op where it should be?
set -u
cd "$(dirname "$0")/.." || exit 1
PY=.venv/bin/python
OUT=outputs/gate_gphase_search
mkdir -p "$OUT"
BC="logs/beat_classifier/version_4/checkpoints/beat-epoch=11-val_f1_avg_tol=0.603.ckpt"
LC="logs/layout_phrase/version_10/checkpoints/layout-epoch=09-val_token_acc=0.865.ckpt"
for s in 2c352 2e593 32c88 29a01 323af 2cdf2 236e7 1fccd 200bd 20288; do
  [ -f "$OUT/$s.zip" ] && continue
  BEAT_GRID_PHASE=search $PY scripts/generate.py "outputs/wide_cohort/audio/$s.ogg" \
    --v7 --beat-ckpt "$BC" --layout-ckpt "$LC" --difficulty Expert \
    --section-gate loud_only --song-name "$s" --seed 0 --output "$OUT/$s.zip" \
    > "$OUT/$s.log" 2>&1 || echo "  $s FAILED"
  grep -h "BEAT_GRID_PHASE" "$OUT/$s.log" | sed "s/^/  $s: /" | tail -1
done
