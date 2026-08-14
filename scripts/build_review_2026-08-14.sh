#!/usr/bin/env bash
# Review set for BEAT_GRID_PHASE=search on Kyle's 4 standing songs.
# ⚠️1f767 (アリスブルー) is the C2 warning case — its HUMAN map wants the same -41 ms
# shift we do, i.e. an ONSET-DETECTOR offset rather than our grid. It is included on
# purpose: it is the song where "fitting the detector" would be audible if it is real.
set -u
cd "$(dirname "$0")/.." || exit 1
PY=.venv/bin/python
OUT=outputs/kyle_review_2026-08-14
mkdir -p "$OUT"
BC="logs/beat_classifier/version_4/checkpoints/beat-epoch=11-val_f1_avg_tol=0.603.ckpt"
LC="logs/layout_phrase/version_10/checkpoints/layout-epoch=09-val_token_acc=0.865.ckpt"
declare -A NAME=([1f333]=Hunger [1f8d6]=FallenKingdom [1f913]=DigitalLifeHacker [1f767]=AliceBlue)
for s in 1f333 1f8d6 1f913 1f767; do
  n=${NAME[$s]}
  for mode in BEFORE PHASE; do
    zp="$OUT/${n}_${mode}.zip"
    [ -f "$zp" ] && continue
    env_flag=""; [ "$mode" = "PHASE" ] && env_flag="search"
    BEAT_GRID_PHASE="${env_flag:-0}" $PY scripts/generate.py "data/eval_songset/$s.ogg" \
      --v7 --beat-ckpt "$BC" --layout-ckpt "$LC" --difficulty Expert \
      --section-gate loud_only --song-name "AUTO $n [$mode]" --seed 0 --output "$zp" \
      > "$OUT/${n}_${mode}.log" 2>&1 || echo "  $n $mode FAILED"
    grep -h "BEAT_GRID_PHASE" "$OUT/${n}_${mode}.log" 2>/dev/null \
      | sed "s/.*grid_phase INFO: /  $n $mode: /" | tail -1
  done
done
echo "=== REVIEW BUILD COMPLETE ==="
