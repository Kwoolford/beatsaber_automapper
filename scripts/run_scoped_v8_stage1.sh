#!/usr/bin/env bash
# Scoped V8 TASK 2 run: cache per-instrument layering features for all songs,
# then retrain Stage 1 with --use-instr. Best beat baseline was version_4
# (d=512 / 4-layer, val_f1_avg_tol=0.603) — match that and add the instr path.
set -euo pipefail
cd /home/kyle/repos/beatsaber_automapper
source .venv/bin/activate

LOG=logs/overnight
mkdir -p "$LOG"

echo "=== [$(date)] STEP 1: preprocess per-instrument features (all songs) ==="
python scripts/preprocess_instruments.py 2>&1 | tee "$LOG/instr_preprocess_2026-06-03.log"

echo "=== [$(date)] STEP 2: Stage-1 retrain with --use-instr ==="
python scripts/train_beats.py \
  --use-instr \
  --d-model 512 --n-layers 4 --n-heads 8 \
  --batch-size 64 --max-epochs 30 --patience 8 \
  --difficulties Expert ExpertPlus \
  --monitor val_f1_avg_tol \
  2>&1 | tee "$LOG/instr_stage1_train_2026-06-03.log"

echo "=== [$(date)] SCOPED-V8 STAGE1 RUN DONE ==="
