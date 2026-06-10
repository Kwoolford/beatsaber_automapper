#!/usr/bin/env bash
# Overnight queue — 2026-05-26
#
# Goal: get user a side-by-side comparison + a new beat classifier checkpoint
# in the morning, after they raised "pause at the beat drop" + "random
# horizontal notes" + "do generated notes line up to music?" feedback.
#
# Order of operations:
#   1. Re-generate the existing test map using the new energy-percentile
#      section detector. Output → outputs/v7_energy_sections.zip
#   2. Run scripts/eval_alignment.py on BOTH the old (v7_section_aware.zip,
#      clustering detector) and new map. Saves JSON reports for comparison.
#   3. Train Beat Classifier Run 6 (struct-features path) — the run that
#      stalled earlier today. The new training produces version_6 in
#      logs/beat_classifier/.
#
# Inference does NOT yet pass struct_features to the beat classifier, so a
# post-Run-6 regeneration is intentionally skipped — that wiring is a clean
# follow-up the user can do tomorrow once they have the Run 6 numbers.

set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate

LOGDIR=logs
QUEUE_LOG="$LOGDIR/overnight_2026-05-26.log"
mkdir -p outputs/2026-05-26

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$QUEUE_LOG"; }

BEAT_CKPT="logs/beat_classifier/version_4/checkpoints/beat-epoch=11-val_f1_avg_tol=0.603.ckpt"
LAYOUT_CKPT="logs/layout_phrase/version_3/checkpoints/layout-epoch=10-val_token_acc=0.870.ckpt"
TEST_AUDIO="data/test_songs/SO TIRED ROCK - NUEKI.mp3"
DIFFICULTY="ExpertPlus"

log "=== Overnight queue start ==="
log "Beat ckpt:   $BEAT_CKPT"
log "Layout ckpt: $LAYOUT_CKPT"
log "Test audio:  $TEST_AUDIO"

# ── 1. Regenerate with energy-percentile section detector ─────────────────────
NEW_MAP="outputs/v7_energy_sections.zip"
log "=== Step 1: Regenerate map with energy-percentile section detector ==="
python scripts/generate.py \
  "$TEST_AUDIO" \
  --output "$NEW_MAP" \
  --difficulty "$DIFFICULTY" \
  --v7 \
  --beat-ckpt   "$BEAT_CKPT" \
  --layout-ckpt "$LAYOUT_CKPT" \
  > "$LOGDIR/overnight_generate.log" 2>&1
log "Generate done → $NEW_MAP"
grep -E "Sections|Stage 1|Generated|NPS" "$LOGDIR/overnight_generate.log" | tail -10 | tee -a "$QUEUE_LOG" || true

# ── 2. Alignment eval — old vs new section detector ───────────────────────────
log "=== Step 2: Alignment eval (old vs new section detector) ==="
python scripts/eval_alignment.py \
  --audio "$TEST_AUDIO" \
  --map outputs/v7_section_aware.zip \
  --difficulty "$DIFFICULTY" \
  --tolerance-ms 50 \
  --json outputs/2026-05-26/alignment_old_clustering.json \
  > "$LOGDIR/overnight_align_old.log" 2>&1
log "Eval (old clustering detector) → outputs/2026-05-26/alignment_old_clustering.json"
tail -25 "$LOGDIR/overnight_align_old.log" | tee -a "$QUEUE_LOG"

python scripts/eval_alignment.py \
  --audio "$TEST_AUDIO" \
  --map "$NEW_MAP" \
  --difficulty "$DIFFICULTY" \
  --tolerance-ms 50 \
  --json outputs/2026-05-26/alignment_new_energy.json \
  > "$LOGDIR/overnight_align_new.log" 2>&1
log "Eval (new energy detector) → outputs/2026-05-26/alignment_new_energy.json"
tail -25 "$LOGDIR/overnight_align_new.log" | tee -a "$QUEUE_LOG"

# ── 3. Beat Classifier Run 6 — struct features ────────────────────────────────
log "=== Step 3: Beat Classifier Run 6 (struct features) ==="
python scripts/train_beats.py \
  --max-epochs 25 \
  --batch-size 64 \
  --d-model 512 --n-heads 8 --n-layers 4 \
  --pos-weight 3.6 \
  --patience 8 \
  --difficulties Expert ExpertPlus \
  --tolerance-slots 1 \
  > "$LOGDIR/train_beats_v6.log" 2>&1
log "Run 6 done."
grep -E "Best checkpoint|Best val_f1" "$LOGDIR/train_beats_v6.log" | tail -2 | tee -a "$QUEUE_LOG"

log "=== Overnight queue complete ==="
