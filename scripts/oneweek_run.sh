#!/usr/bin/env bash
# 1-Week Production Training Run
# Sequence model only (onset reuses smoke test #1, lighting is rule-based)
#
# Expected duration: ~5-7 days on RTX 5090
# Sequence: 100 epochs × ~1.5M samples/epoch × ~20 min/epoch = ~33 hours
#   (with patience=30, may stop earlier if converged)
# FULL BLAST: no low_priority, 12 workers, max GPU utilization
#
# Usage:
#   bash scripts/oneweek_run.sh
#   bash scripts/oneweek_run.sh --resume   # resume from last checkpoint
#
# Monitor:
#   tail -f outputs/training_*.log
#   cat outputs/heartbeat.json
#   tensorboard --logdir outputs/beatsaber_automapper

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR" || exit 1

# Parse arguments
RESUME=false
while [[ $# -gt 0 ]]; do
    case "$1" in
        --resume) RESUME=true; shift ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
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
LOG_FILE="$OUTPUT_DIR/training_1week_$(date +%Y%m%d_%H%M%S).log"

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
  "run_type": "1_week_production",
  "pid": $$
}
HEOF
}

find_best_checkpoint() {
    local stage="$1"
    local ckpt_dir="$OUTPUT_DIR/beatsaber_automapper"
    if [ -d "$ckpt_dir" ]; then
        for vdir in $(ls -d "$ckpt_dir"/version_* 2>/dev/null | sort -V -r); do
            local ckpt_subdir="$vdir/checkpoints"
            if [ -d "$ckpt_subdir" ]; then
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

# ============================================================
# PRE-FLIGHT CHECKS
# ============================================================
log "============================================"
log "1-WEEK PRODUCTION TRAINING RUN"
log "Started: $(date)"
log "============================================"

# Verify onset checkpoint exists
ONSET_CKPT=$(find_best_checkpoint onset)
if [ -z "$ONSET_CKPT" ]; then
    log "ERROR: No onset checkpoint found! Run onset training first."
    exit 1
fi
log "Using onset checkpoint: $ONSET_CKPT"

# Verify GPU
python -c "import torch; assert torch.cuda.is_available(), 'No GPU'; print(f'GPU: {torch.cuda.get_device_name()}, VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB')" 2>&1 | tee -a "$LOG_FILE"
if [ ${PIPESTATUS[0]} -ne 0 ]; then
    log "ERROR: GPU check failed!"
    exit 1
fi

# Verify data
TOTAL_FILES=$(find "$DATA_DIR" -maxdepth 1 -name "*.pt" 2>/dev/null | wc -l)
log "Training data: $TOTAL_FILES .pt files in $DATA_DIR"
if [ "$TOTAL_FILES" -lt 100 ]; then
    log "WARNING: Very few training files ($TOTAL_FILES). Expected 1000+."
fi

# ============================================================
# SEQUENCE TRAINING — THE MAIN EVENT
# ============================================================
log ""
log "=========================================="
log "Stage: Sequence Model (1-week production)"
log "  max_epochs: 100"
log "  patience: 30"
log "  batch_size: 192"
log "  learning_rate: 1e-4 (from sequence.yaml)"
log "  warmup_steps: 2000 (from sequence.yaml)"
log "  token_dropout: 0.1 (from sequence.yaml)"
log "  lr_min_ratio: 0.01 (from sequence.yaml)"
log "  max_samples_per_epoch: 1500000"
log "  gradient_clip_val: 1.0"
log "  low_priority: false (FULL BLAST)"
log "  num_workers: 8"
log "=========================================="

write_heartbeat "sequence" 0 "starting" "running"

RESUME_ARG=""
if [ "$RESUME" = true ]; then
    LAST_CKPT=$(find_last_checkpoint sequence)
    if [ -n "$LAST_CKPT" ]; then
        log "Resuming from: $LAST_CKPT"
        RESUME_ARG="ckpt_path=$LAST_CKPT"
    else
        log "No checkpoint to resume from — starting fresh"
    fi
fi

python scripts/train.py \
    stage=sequence \
    data_dir="$DATA_DIR" \
    output_dir="$OUTPUT_DIR" \
    max_epochs=100 \
    data.dataset.batch_size=192 \
    data.dataset.num_workers=8 \
    early_stopping_patience=30 \
    max_samples_per_epoch=1500000 \
    low_priority=false \
    $RESUME_ARG \
    2>&1 | tee -a "$LOG_FILE"

SEQ_EXIT=${PIPESTATUS[0]}

if [ $SEQ_EXIT -eq 0 ]; then
    log "Sequence training completed successfully!"
    write_heartbeat "sequence" 50 "done" "completed"
else
    log "Sequence training exited with code $SEQ_EXIT"
    write_heartbeat "sequence" -1 "error" "failed"
fi

# ============================================================
# POST-TRAINING: Generate baseline maps
# ============================================================
log ""
log "=========================================="
log "Generating baseline maps from best checkpoint"
log "=========================================="

SEQ_CKPT=$(find_best_checkpoint sequence)
BASELINE_AUDIO="data/reference/so_tired_rock.mp3"

if [ -f "$BASELINE_AUDIO" ] && [ -n "$SEQ_CKPT" ]; then
    log "Onset: $ONSET_CKPT"
    log "Sequence: $SEQ_CKPT"

    mkdir -p data/generated

    # Generate Expert
    python scripts/generate.py \
        "$BASELINE_AUDIO" \
        --output "data/generated/1week_expert.zip" \
        --onset-ckpt "$ONSET_CKPT" \
        --seq-ckpt "$SEQ_CKPT" \
        --difficulty Expert \
        --genre rock \
        2>&1 | tee -a "$LOG_FILE"

    # Generate ExpertPlus
    python scripts/generate.py \
        "$BASELINE_AUDIO" \
        --output "data/generated/1week_expertplus.zip" \
        --onset-ckpt "$ONSET_CKPT" \
        --seq-ckpt "$SEQ_CKPT" \
        --difficulty ExpertPlus \
        --genre rock \
        2>&1 | tee -a "$LOG_FILE"

    log "Baseline maps generated in data/generated/"
else
    log "Skipping baseline generation"
    [ ! -f "$BASELINE_AUDIO" ] && log "  Missing: $BASELINE_AUDIO"
    [ -z "$SEQ_CKPT" ] && log "  Missing sequence checkpoint"
fi

log ""
log "============================================"
log "1-WEEK RUN COMPLETE — $(date)"
log "============================================"
write_heartbeat "done" 0 "complete" "completed"
