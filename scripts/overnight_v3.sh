#!/usr/bin/env bash
# Hardened overnight training pipeline v3
# Runs all 3 stages sequentially with auto-resume and health monitoring.
#
# Usage:
#   bash scripts/overnight_v3.sh                  # full run
#   bash scripts/overnight_v3.sh --smoke-test     # 5 epochs per stage
#   bash scripts/overnight_v3.sh --stage sequence # start from a specific stage
#
# Features:
#   - Auto-resumes from last.ckpt if available
#   - Per-stage error handling (no set -e)
#   - Heartbeat file for remote monitoring
#   - BELOW_NORMAL process priority (gaming-friendly)
#   - Configurable max_epochs and batch_size per stage

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR" || exit 1

# Parse arguments
SMOKE_TEST=false
START_STAGE="onset"
MAX_EPOCHS_ONSET=100
MAX_EPOCHS_SEQ=100
MAX_EPOCHS_LIGHT=100
BATCH_SIZE_ONSET=64
BATCH_SIZE_SEQ=48
BATCH_SIZE_LIGHT=256
PATIENCE=25

while [[ $# -gt 0 ]]; do
    case "$1" in
        --smoke-test)
            SMOKE_TEST=true
            MAX_EPOCHS_ONSET=5
            MAX_EPOCHS_SEQ=5
            MAX_EPOCHS_LIGHT=5
            PATIENCE=100  # don't early stop during smoke test
            shift
            ;;
        --stage)
            START_STAGE="$2"
            shift 2
            ;;
        --epochs)
            MAX_EPOCHS_ONSET="$2"
            MAX_EPOCHS_SEQ="$2"
            MAX_EPOCHS_LIGHT="$2"
            shift 2
            ;;
        *)
            echo "Unknown arg: $1"
            exit 1
            ;;
    esac
done

# Activate venv
if [ -f ".venv/Scripts/activate" ]; then
    source .venv/Scripts/activate
elif [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
fi

DATA_DIR="data/processed"
OUTPUT_DIR="outputs"
HEARTBEAT_FILE="$OUTPUT_DIR/heartbeat.json"
LOG_FILE="$OUTPUT_DIR/training_$(date +%Y%m%d_%H%M%S).log"

mkdir -p "$OUTPUT_DIR"

log() {
    local msg="[$(date '+%Y-%m-%d %H:%M:%S')] $*"
    echo "$msg" | tee -a "$LOG_FILE"
}

write_heartbeat() {
    local stage="$1" epoch="$2" metric="$3" status="$4"
    cat > "$HEARTBEAT_FILE" <<HEOF
{
  "timestamp": "$(date -Iseconds)",
  "stage": "$stage",
  "epoch": $epoch,
  "metric": "$metric",
  "status": "$status",
  "smoke_test": $SMOKE_TEST,
  "pid": $$
}
HEOF
}

find_last_checkpoint() {
    local stage="$1"
    # Search for last.ckpt in the stage's checkpoint directory
    local ckpt_dir="$OUTPUT_DIR/beatsaber_automapper"
    if [ -d "$ckpt_dir" ]; then
        # Find the latest version directory for this stage
        local latest_version
        latest_version=$(ls -d "$ckpt_dir"/version_* 2>/dev/null | sort -V | tail -1)
        if [ -n "$latest_version" ] && [ -f "$latest_version/checkpoints/last.ckpt" ]; then
            echo "$latest_version/checkpoints/last.ckpt"
            return 0
        fi
    fi
    return 1
}

run_stage() {
    local stage="$1"
    local max_epochs="$2"
    local batch_size="$3"

    log "=========================================="
    log "Starting stage: $stage (max_epochs=$max_epochs, batch=$batch_size)"
    log "=========================================="

    write_heartbeat "$stage" 0 "starting" "running"

    local resume_arg=""
    local ckpt
    ckpt=$(find_last_checkpoint "$stage")
    if [ -n "$ckpt" ]; then
        log "Found checkpoint to resume from: $ckpt"
        resume_arg="ckpt_path=$ckpt"
    fi

    python scripts/train.py \
        stage="$stage" \
        data_dir="$DATA_DIR" \
        output_dir="$OUTPUT_DIR" \
        max_epochs="$max_epochs" \
        data.dataset.batch_size="$batch_size" \
        data.dataset.num_workers=8 \
        early_stopping_patience="$PATIENCE" \
        low_priority=true \
        $resume_arg \
        2>&1 | tee -a "$LOG_FILE"

    local exit_code=${PIPESTATUS[0]}

    if [ $exit_code -eq 0 ]; then
        log "Stage $stage completed successfully"
        write_heartbeat "$stage" "$max_epochs" "done" "completed"
    else
        log "Stage $stage failed with exit code $exit_code"
        write_heartbeat "$stage" -1 "error" "failed"
    fi

    return $exit_code
}

# Main pipeline
log "============================================"
log "Overnight Training Pipeline v3"
log "Smoke test: $SMOKE_TEST"
log "Start stage: $START_STAGE"
log "Data dir: $DATA_DIR"
log "Output dir: $OUTPUT_DIR"
log "============================================"

stages_to_run=()
case "$START_STAGE" in
    onset)    stages_to_run=(onset sequence lighting) ;;
    sequence) stages_to_run=(sequence lighting) ;;
    lighting) stages_to_run=(lighting) ;;
    *)
        log "Unknown stage: $START_STAGE"
        exit 1
        ;;
esac

for stage in "${stages_to_run[@]}"; do
    case "$stage" in
        onset)    run_stage onset "$MAX_EPOCHS_ONSET" "$BATCH_SIZE_ONSET" ;;
        sequence) run_stage sequence "$MAX_EPOCHS_SEQ" "$BATCH_SIZE_SEQ" ;;
        lighting) run_stage lighting "$MAX_EPOCHS_LIGHT" "$BATCH_SIZE_LIGHT" ;;
    esac
    stage_result=$?

    if [ $stage_result -ne 0 ]; then
        log "Pipeline stopped at stage $stage due to error"
        # Continue to next stage anyway (non-fatal)
        log "Continuing to next stage despite error..."
    fi
done

log "============================================"
log "Pipeline complete!"
log "============================================"
write_heartbeat "done" 0 "all_stages_complete" "completed"
