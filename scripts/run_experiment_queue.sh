#!/usr/bin/env bash
# Experiment queue — runs each Stage 2 job in sequence after the Beat Classifier
# finishes. Designed to be launched once and left running for 2-3 days.
#
# Experiments:
#   Run 4 — Cross-phrase prefix (ctx_len=16), same arch as Run 2/3 (38.7M)
#   Run 5 — Cross-phrase prefix + Scheduled Sampling (ramp 0→0.3 over 20 epochs)
#
# Beat Classifier Run 5 is already running (PID 40268); this script waits for it.

set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate

LOGDIR=logs
BEAT_LOG="$LOGDIR/train_beats_v4.log"
BEAT_PID=40268

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOGDIR/experiment_queue.log"; }

# ── Wait for Beat Classifier ──────────────────────────────────────────────────
log "Waiting for Beat Classifier PID $BEAT_PID to finish…"
while kill -0 "$BEAT_PID" 2>/dev/null; do sleep 30; done
log "Beat Classifier done. Best checkpoint:"
grep "Best checkpoint:" "$BEAT_LOG" | tail -1 | tee -a "$LOGDIR/experiment_queue.log"

# ── Run 4: Cross-Phrase Prefix ────────────────────────────────────────────────
log "=== Stage 2 Run 4: cross-phrase prefix ctx_len=16 ==="
python scripts/train_layout.py \
  --max-epochs 60 \
  --batch-size 64 \
  --lr 2e-4 \
  --d-model 512 --n-heads 8 \
  --n-enc-layers 4 --n-dec-layers 6 --dim-feedforward 2048 \
  --patience 12 \
  --x-role-weight 1.0 \
  --ctx-len 16 \
  > "$LOGDIR/train_layout_v3.log" 2>&1

log "Run 4 done."
grep -E "Best val_token_acc|Best checkpoint" "$LOGDIR/train_layout_v3.log" | tail -2 \
  | tee -a "$LOGDIR/experiment_queue.log"

# ── Run 5: Cross-Phrase Prefix + Scheduled Sampling ──────────────────────────
log "=== Stage 2 Run 5: cross-phrase prefix + scheduled sampling (0→0.3) ==="
python scripts/train_layout.py \
  --max-epochs 60 \
  --batch-size 64 \
  --lr 2e-4 \
  --d-model 512 --n-heads 8 \
  --n-enc-layers 4 --n-dec-layers 6 --dim-feedforward 2048 \
  --patience 12 \
  --x-role-weight 1.0 \
  --ctx-len 16 \
  --sched-sampling-start 0.0 \
  --sched-sampling-end   0.3 \
  --sched-sampling-epochs 20 \
  > "$LOGDIR/train_layout_v4.log" 2>&1

log "Run 5 done."
grep -E "Best val_token_acc|Best checkpoint" "$LOGDIR/train_layout_v4.log" | tail -2 \
  | tee -a "$LOGDIR/experiment_queue.log"

log "All experiments complete."
