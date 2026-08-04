#!/usr/bin/env bash
# W1 task 1 -- Kyle's coincidence hypothesis, measured. CPU only, no GPU, no retrain.
# Arms:
#   ours      = tf_trim_ev03_rc05  (the PROMOTED baseline, 24 songs x 3 seeds)
#   ours_noev = tf_hl014_ds048     (pre BEAT_ONSET_EVIDENCE -- circularity control)
#   human     = 273 strict-Expert corpus maps that have a seeded stem cache
set -u
cd "$(dirname "$0")/.."
L=logs/overnight/coincidence_2026-08-03.log
: > "$L"
echo "=== START $(date -Is) ===" >> "$L"

echo "" >> "$L"; echo "##### ARM: promoted baseline vs human (273) #####" >> "$L"
.venv/bin/python scripts/eval_coincidence.py \
  --gen 'outputs/eval_sweep_cache/tf_trim_ev03_rc05#s*__*.zip' \
  --human-n 273 --json outputs/coincidence_2026-08-03.json >> "$L" 2>&1

echo "" >> "$L"; echo "##### CONTROL: pre-ev03 arm (is our lift an artifact of BEAT_ONSET_EVIDENCE?) #####" >> "$L"
.venv/bin/python scripts/eval_coincidence.py \
  --gen 'outputs/eval_sweep_cache/tf_hl014_ds048#s*__*.zip' \
  --human-n 40 --json outputs/coincidence_noev_2026-08-03.json >> "$L" 2>&1

echo "" >> "$L"; echo "=== COMPLETE $(date -Is) ===" >> "$L"
