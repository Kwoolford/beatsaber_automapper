#!/usr/bin/env bash
# Overnight NPS-fix sweep: investigate sparse-note / low-NPS failure mode.
#
# Three sequential training runs, each followed by generation on the test song.
# All runs use the stall-fixed generation code (max_events=2000, per_window_cap=128,
# current_beat sync on manual window advance).
#
# Run A — baseline bombs+dt fix:
#   bomb_hand_weight=0.3  (discourage HAND_NONE overgeneration)
#   dt_density_alpha=0.0  (no Δt density penalty — isolates bomb fix)
#
# Run B — dt density penalty moderate:
#   bomb_hand_weight=0.3
#   dt_density_alpha=0.5  (hinge penalty: P(Δt=0) > 20% at DT positions)
#
# Run C — dt density penalty strong:
#   bomb_hand_weight=0.3
#   dt_density_alpha=1.0  (stronger penalty)
#
# Budget: 3 × 30 epochs × ~4m22s ≈ 6.6h total.
# All runs write checkpoints to outputs/beatsaber_automapper/sequence/version_N/.
# Generation output lands in data/generated/nps_fix_{run}_{stamp}/.
set -euo pipefail

cd /home/kyle/repos/beatsaber_automapper
source .venv/bin/activate

STAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR=logs/overnight_nps_fix_${STAMP}
mkdir -p "${LOG_DIR}"

TEST_SONG="data/test_songs/SO TIRED ROCK - NUEKI.mp3"
BASE_TRAIN="python scripts/train.py stage=sequence model/sequence=sequence_swing_small \
  dataset_format=swing max_epochs=30 max_samples_per_epoch=50000 \
  data.dataset.batch_size=32 data.dataset.num_workers=8 \
  limit_val_batches=100 model.sequence.phrase_energy_alpha=0.1 \
  early_stopping_patience=999"

_run_experiment() {
  local tag="$1"
  local extra_args="$2"
  local train_log="${LOG_DIR}/train_${tag}.log"
  local gen_log="${LOG_DIR}/generate_${tag}.log"

  echo "======================================================"
  echo "[$(date -Iseconds)] RUN ${tag}: ${extra_args}"
  echo "======================================================"

  # Record the highest existing version before training so we can find the new one
  local prev_version
  prev_version=$(ls -d outputs/beatsaber_automapper/sequence/version_* 2>/dev/null \
    | sort -V | tail -n1 || echo "none")

  # Train
  # shellcheck disable=SC2086
  ${BASE_TRAIN} ${extra_args} 2>&1 | tee "${train_log}"
  local train_rc=${PIPESTATUS[0]}
  echo "[$(date -Iseconds)] [${tag}] training exited rc=${train_rc}"

  if [ "${train_rc}" -ne 0 ]; then
    echo "[${tag}] training FAILED; skipping generation"
    return "${train_rc}"
  fi

  # Find the checkpoint from the run we just completed (newest version dir)
  local ckpt
  ckpt=$(ls -t outputs/beatsaber_automapper/sequence/version_*/checkpoints/sequence-epoch=*.ckpt \
    2>/dev/null | head -n1 || true)

  if [ -z "${ckpt}" ]; then
    echo "[${tag}] no checkpoint found; skipping generation"
    return 2
  fi

  echo "[$(date -Iseconds)] [${tag}] generating with ckpt=${ckpt}"

  python scripts/generate.py "${TEST_SONG}" \
    --v6 --seq-ckpt "${ckpt}" --difficulty Expert --genre rock \
    --run-tag "nps_fix_${tag}_${STAMP}" --temperature 0.9 --top-p 0.9 \
    2>&1 | tee "${gen_log}"

  echo "[$(date -Iseconds)] [${tag}] generation done"
  echo "  ckpt: ${ckpt}" | tee -a "${LOG_DIR}/summary.txt"
}

# ---------------------------------------------------------------------------
# Run A: bomb weight fix only (isolate: does discouraging bombs help NPS?)
# ---------------------------------------------------------------------------
_run_experiment "A_bombfix" \
  "model.sequence.bomb_hand_weight=0.3"

# ---------------------------------------------------------------------------
# Run B: bomb fix + moderate dt density penalty
# ---------------------------------------------------------------------------
_run_experiment "B_dt05" \
  "model.sequence.bomb_hand_weight=0.3 model.sequence.dt_density_alpha=0.5"

# ---------------------------------------------------------------------------
# Run C: bomb fix + strong dt density penalty
# ---------------------------------------------------------------------------
_run_experiment "C_dt10" \
  "model.sequence.bomb_hand_weight=0.3 model.sequence.dt_density_alpha=1.0"

echo ""
echo "[$(date -Iseconds)] All overnight runs complete."
echo "Generated maps in data/generated/nps_fix_*_${STAMP}/"
echo "Training logs in ${LOG_DIR}/"
