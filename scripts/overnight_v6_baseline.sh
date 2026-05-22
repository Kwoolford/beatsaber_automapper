#!/usr/bin/env bash
# Overnight V6 baseline: train clean swing model on data/processed, then auto-generate test map.
set -euo pipefail

cd /home/kyle/repos/beatsaber_automapper
source .venv/bin/activate

STAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR=logs/overnight_v6_${STAMP}
mkdir -p "${LOG_DIR}"

TRAIN_LOG="${LOG_DIR}/train.log"
GEN_LOG="${LOG_DIR}/generate.log"
SUMMARY="${LOG_DIR}/summary.txt"

echo "[$(date -Iseconds)] starting V6 baseline training" | tee -a "${SUMMARY}"

python scripts/train.py stage=sequence model/sequence=sequence_swing_small \
  dataset_format=swing max_epochs=30 max_samples_per_epoch=50000 \
  data.dataset.batch_size=32 data.dataset.num_workers=8 \
  limit_val_batches=100 model.sequence.phrase_energy_alpha=0.1 \
  early_stopping_patience=999 \
  2>&1 | tee -a "${TRAIN_LOG}"

TRAIN_RC=${PIPESTATUS[0]}
echo "[$(date -Iseconds)] training exited rc=${TRAIN_RC}" | tee -a "${SUMMARY}"

if [ "${TRAIN_RC}" -ne 0 ]; then
  echo "training failed; skipping generation" | tee -a "${SUMMARY}"
  exit "${TRAIN_RC}"
fi

# Pick the newest sequence ckpt (lowest val_loss is best, but newest works since EarlyStopping is disabled).
CKPT=$(ls -t outputs/beatsaber_automapper/sequence/version_*/checkpoints/sequence-epoch=*.ckpt 2>/dev/null | head -n1 || true)

if [ -z "${CKPT}" ]; then
  echo "no checkpoint found; cannot generate" | tee -a "${SUMMARY}"
  exit 2
fi

echo "[$(date -Iseconds)] generating with ckpt=${CKPT}" | tee -a "${SUMMARY}"

python scripts/generate.py "data/test_songs/SO TIRED ROCK - NUEKI.mp3" \
  --v6 --seq-ckpt "${CKPT}" --difficulty Expert --genre rock \
  --run-tag v6_baseline_${STAMP} --temperature 0.9 --top-p 0.9 \
  2>&1 | tee -a "${GEN_LOG}"

GEN_RC=${PIPESTATUS[0]}
echo "[$(date -Iseconds)] generation exited rc=${GEN_RC}" | tee -a "${SUMMARY}"
echo "ckpt: ${CKPT}" >> "${SUMMARY}"
exit "${GEN_RC}"
