#!/usr/bin/env bash
# TASK 0 — evaluate the version_12 cohort-filtered retrain vs version_10.
# Generates v12 @ Expert + ExpertPlus and v10 @ Expert (v10 ExpertPlus already
# exists as outputs/v8_gatefix_loudonly.zip), all with --section-gate loud_only,
# then runs eval_alignment.py on each into outputs/task0/.
set -euo pipefail
cd /home/kyle/repos/beatsaber_automapper
source .venv/bin/activate

AUDIO="data/test_songs/SO TIRED ROCK - NUEKI.mp3"
BEAT="logs/beat_classifier/version_4/checkpoints/beat-epoch=11-val_f1_avg_tol=0.603.ckpt"
V12="logs/layout_phrase/version_12/checkpoints/layout-epoch=10-val_token_acc=0.863.ckpt"
V10="logs/layout_phrase/version_10/checkpoints/layout-epoch=09-val_token_acc=0.865.ckpt"
OUT=outputs/task0
mkdir -p "$OUT"

gen () {  # $1=ckpt $2=difficulty $3=tag
  echo "=== GENERATE $3 ($2) ==="
  python scripts/generate.py "$AUDIO" --v7 \
    --beat-ckpt "$BEAT" --layout-ckpt "$1" \
    --difficulty "$2" --section-gate loud_only \
    --output "$OUT/$3.zip"
}

evl () {  # $1=zip $2=difficulty $3=tag
  echo "=== EVAL $3 ($2) ==="
  python scripts/eval_alignment.py --audio "$AUDIO" \
    --map "$OUT/$1" --difficulty "$2" \
    --json "$OUT/align_$3.json" | tee "$OUT/report_$3.txt"
}

gen "$V12" ExpertPlus v12_ep
gen "$V12" Expert     v12_ex
gen "$V10" Expert     v10_ex

evl v12_ep.zip ExpertPlus v12_ep
evl v12_ex.zip Expert     v12_ex
evl v10_ex.zip Expert     v10_ex
# v10 ExpertPlus reuse the gate-fix loud_only map generated previously
cp outputs/v8_gatefix_loudonly.zip "$OUT/v10_ep.zip"
evl v10_ep.zip ExpertPlus v10_ep

echo "=== TASK0 DONE ==="
