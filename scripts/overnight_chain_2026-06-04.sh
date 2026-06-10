#!/usr/bin/env bash
# Overnight chain (2026-06-04, post power-cut resume):
#   1. wait for the already-running per-instrument preprocess to finish
#   2. STEP2: Stage-1 retrain with --use-instr (matches version_4 d512/4L baseline)
#   3. eval: TASK-1 retrieval-key PoC at full cohort
# Each step gates the next; everything is logged for cold-start review.
set -uo pipefail
cd /home/kyle/repos/beatsaber_automapper
source .venv/bin/activate

LOG=logs/overnight
mkdir -p "$LOG"
SUMMARY="$LOG/chain_2026-06-04_summary.log"
echo "=== [$(date)] CHAIN START ===" | tee -a "$SUMMARY"

# ---- STEP 1: wait for the running preprocess to drain --------------------
echo "=== [$(date)] STEP 1: waiting for preprocess_instruments to finish ===" | tee -a "$SUMMARY"
while pgrep -f "preprocess_instruments.py" >/dev/null 2>&1; do
  sleep 60
done
COV=$(python - <<'PY'
import pathlib, torch
DATA=pathlib.Path("data/processed")
have=sum("instr_beat_features" in torch.load(p, weights_only=False, mmap=True)
         for p in DATA.glob("*.pt"))
tot=len(list(DATA.glob("*.pt")))
print(f"{have}/{tot}")
PY
)
echo "=== [$(date)] STEP 1 DONE — instr coverage $COV ===" | tee -a "$SUMMARY"

# ---- STEP 2: Stage-1 retrain with per-instrument features ----------------
echo "=== [$(date)] STEP 2: Stage-1 retrain --use-instr ===" | tee -a "$SUMMARY"
python scripts/train_beats.py \
  --use-instr \
  --d-model 512 --n-layers 4 --n-heads 8 \
  --batch-size 64 --max-epochs 30 --patience 8 \
  --difficulties Expert ExpertPlus \
  --monitor val_f1_avg_tol \
  2>&1 | tee "$LOG/instr_stage1_train_2026-06-04.log"
TRAIN_RC=${PIPESTATUS[0]}
echo "=== [$(date)] STEP 2 train exit=$TRAIN_RC ===" | tee -a "$SUMMARY"

# Record the best checkpoint (highest val_f1_avg_tol) from the newest version dir.
NEWVER=$(ls -d logs/beat_classifier/version_* 2>/dev/null | sort -V | tail -1)
echo "newest beat_classifier dir: $NEWVER" | tee -a "$SUMMARY"
ls "$NEWVER/checkpoints/" 2>/dev/null | tee -a "$SUMMARY"
BEST=$(ls "$NEWVER/checkpoints/" 2>/dev/null | grep -oE 'val_f1_avg_tol=[0-9.]+' | sort -t= -k2 -n | tail -1)
echo "BEST new $BEST  (baseline version_4 = val_f1_avg_tol=0.603)" | tee -a "$SUMMARY"

# ---- STEP 3: TASK-1 retrieval-key PoC at full cohort ---------------------
echo "=== [$(date)] STEP 3: TASK-1 retrieval-key PoC (--n 60 --difficulty Expert) ===" | tee -a "$SUMMARY"
python scripts/v8_poc_retrieval_key.py \
  --n 60 --difficulty Expert \
  --out outputs/v8_poc/retrieval_key_2026-06-04.json \
  2>&1 | tee "$LOG/task1_retrieval_key_2026-06-04.log"
echo "=== [$(date)] STEP 3 DONE ===" | tee -a "$SUMMARY"

echo "=== [$(date)] CHAIN COMPLETE ===" | tee -a "$SUMMARY"
