#!/usr/bin/env bash
# Overnight training pipeline v3 — Smoke test #2 + production runs
# Lighting is now rule-based — only onset + sequence stages are trained.
#
# Usage:
#   bash scripts/overnight_v3.sh                  # full run
#   bash scripts/overnight_v3.sh --smoke-test     # reduced epochs
#   bash scripts/overnight_v3.sh --stage sequence # start from a specific stage
#
# Smoke test #2 changes from v2:
#   - No lighting stage (rule-based now)
#   - Sequence: LR=1e-4, warmup=2000, token_dropout=0.1, samples_per_epoch=1.5M
#   - Auto-generates baseline map after sequence training
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
MAX_EPOCHS_SEQ=50
BATCH_SIZE_ONSET=64
BATCH_SIZE_SEQ=192
PATIENCE=25
GENERATE_BASELINE=true

while [[ $# -gt 0 ]]; do
    case "$1" in
        --smoke-test)
            SMOKE_TEST=true
            MAX_EPOCHS_ONSET=10
            MAX_EPOCHS_SEQ=15
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
            shift 2
            ;;
        --patience)
            PATIENCE="$2"
            shift 2
            ;;
        --no-generate)
            GENERATE_BASELINE=false
            shift
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
    local ckpt_dir="$OUTPUT_DIR/beatsaber_automapper"
    if [ -d "$ckpt_dir" ]; then
        for vdir in $(ls -d "$ckpt_dir"/version_* 2>/dev/null | sort -V -r); do
            local ckpt_subdir="$vdir/checkpoints"
            if [ -d "$ckpt_subdir" ]; then
                if ls "$ckpt_subdir"/${stage}-*.ckpt 1>/dev/null 2>&1; then
                    if [ -f "$ckpt_subdir/last.ckpt" ]; then
                        echo "$ckpt_subdir/last.ckpt"
                        return 0
                    fi
                fi
            fi
        done
    fi
    return 1
}

find_best_checkpoint() {
    local stage="$1"
    local ckpt_dir="$OUTPUT_DIR/beatsaber_automapper"
    if [ -d "$ckpt_dir" ]; then
        for vdir in $(ls -d "$ckpt_dir"/version_* 2>/dev/null | sort -V -r); do
            local ckpt_subdir="$vdir/checkpoints"
            if [ -d "$ckpt_subdir" ]; then
                # Find best checkpoint (not last.ckpt) for the stage
                local best=$(ls "$ckpt_subdir"/${stage}-*.ckpt 2>/dev/null | grep -v last.ckpt | sort -V -r | head -1)
                if [ -n "$best" ]; then
                    echo "$best"
                    return 0
                fi
            fi
        done
    fi
    return 1
}

run_stage() {
    local stage="$1"
    local max_epochs="$2"
    local batch_size="$3"
    shift 3
    local extra_args="$*"

    log "=========================================="
    log "Starting stage: $stage (max_epochs=$max_epochs, batch=$batch_size)"
    if [ -n "$extra_args" ]; then
        log "Extra args: $extra_args"
    fi
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
        data.dataset.num_workers=12 \
        early_stopping_patience="$PATIENCE" \
        low_priority=true \
        $resume_arg \
        $extra_args \
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
log "Overnight Training Pipeline v3 (no lighting — rule-based)"
log "Smoke test: $SMOKE_TEST"
log "Start stage: $START_STAGE"
log "Data dir: $DATA_DIR"
log "Output dir: $OUTPUT_DIR"
log "============================================"

# Only onset + sequence stages (lighting is rule-based now)
stages_to_run=()
case "$START_STAGE" in
    onset)    stages_to_run=(onset sequence) ;;
    sequence) stages_to_run=(sequence) ;;
    *)
        log "Unknown stage: $START_STAGE (valid: onset, sequence)"
        exit 1
        ;;
esac

for stage in "${stages_to_run[@]}"; do
    case "$stage" in
        onset)
            run_stage onset "$MAX_EPOCHS_ONSET" "$BATCH_SIZE_ONSET"
            ;;
        sequence)
            # Smoke test #2 fixes: lower LR, more warmup, token dropout, more samples
            run_stage sequence "$MAX_EPOCHS_SEQ" "$BATCH_SIZE_SEQ" \
                "max_samples_per_epoch=1500000"
            ;;
    esac
    stage_result=$?

    if [ $stage_result -ne 0 ]; then
        log "Pipeline stopped at stage $stage due to error"
        log "Continuing to next stage despite error..."
    fi
done

# Auto-generate baseline map after training
if [ "$GENERATE_BASELINE" = true ]; then
    log "=========================================="
    log "Generating baseline map (so_tired_rock.mp3)"
    log "=========================================="

    ONSET_CKPT=$(find_best_checkpoint onset)
    SEQ_CKPT=$(find_best_checkpoint sequence)

    BASELINE_AUDIO="data/baseline/so_tired_rock.mp3"
    BASELINE_OUTPUT="data/generated/smoke_test_2_baseline.zip"
    mkdir -p "data/generated"

    if [ -f "$BASELINE_AUDIO" ] && [ -n "$ONSET_CKPT" ] && [ -n "$SEQ_CKPT" ]; then
        log "Onset checkpoint: $ONSET_CKPT"
        log "Sequence checkpoint: $SEQ_CKPT"

        python scripts/generate.py \
            "$BASELINE_AUDIO" \
            --output "$BASELINE_OUTPUT" \
            --onset-ckpt "$ONSET_CKPT" \
            --seq-ckpt "$SEQ_CKPT" \
            --difficulty Expert \
            --genre rock \
            2>&1 | tee -a "$LOG_FILE"

        if [ $? -eq 0 ]; then
            log "Baseline map generated: $BASELINE_OUTPUT"
        else
            log "Baseline generation failed"
        fi
    else
        log "Skipping baseline generation (missing audio or checkpoints)"
        [ ! -f "$BASELINE_AUDIO" ] && log "  Missing: $BASELINE_AUDIO"
        [ -z "$ONSET_CKPT" ] && log "  Missing onset checkpoint"
        [ -z "$SEQ_CKPT" ] && log "  Missing sequence checkpoint"
    fi
fi

log "============================================"
log "Pipeline complete!"
log "============================================"
write_heartbeat "done" 0 "all_stages_complete" "completed"
