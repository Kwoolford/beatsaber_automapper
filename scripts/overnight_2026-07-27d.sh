#!/usr/bin/env bash
# 2026-07-27 (part D) — how much of the RHYTHM gap is tempo detection?
#
# Every lever swept today leaves rhythm at ~2.4 (bar 0.70); the only thing that
# moves it is BEAT_HAND_ROLE, which makes it WORSE (4.05). Meanwhile:
#   * rule_mapper.py proved rhythm is inherited ENTIRELY from the onset layer
#     (2.41 on our onsets, 0.25 on human onsets, same placement code), so the
#     gap lives in Stage-1 selection, not in layout.
#   * 30% of the eval set generates at the WRONG TEMPO (16/23 correct vs the
#     human-declared BPM), and A2 measures intervals in the BEAT domain, so
#     tempo error contaminates precisely this axis. Mis-tempo maps even score
#     BETTER, so the contamination is not neutral.
#
# This run regenerates with --true-bpm (the human map's declared BPM) to remove
# tempo detection as a confound and get the first clean estimate of how much of
# the rhythm gap is OUR MAP QUALITY versus OUR TEMPO DETECTION.
#
# EVALUATION-ONLY: production has no human map to read a BPM from. A win here
# does not ship; it tells us where to spend the next effort.
#
# ARMS: prod and best, regenerated with --force --true-bpm into a separate cache
# so the existing (detected-BPM) results stay intact for comparison.
#
# VERDICT LOGIC
#   rhythm_gap drops a lot (say below ~1.5)  => a large share of the rhythm gap
#       is TEMPO DETECTION, and the next investment is a real tempo model rather
#       than more Stage-1 work.
#   rhythm_gap barely moves                  => tempo is a side issue and the gap
#       is genuinely Stage-1 onset selection => that is the retrain to scope.
#   Either way, report the split. This is the measurement that decides where the
#   next GPU night goes.
set -uo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate

for _ in $(seq 1 480); do
  grep -q "COMPLETE — hand-role sweep" logs/overnight/handrole_2026-07-27.log 2>/dev/null && break
  sleep 30
done
echo "part C done; starting part D"

CACHE=outputs/eval_sweep_cache
BACKUP=outputs/eval_sweep_cache_detectedbpm
mkdir -p "$BACKUP"
for A in prod best; do
  for f in "$CACHE"/${A}__*.zip; do
    [ -e "$f" ] && cp -n "$f" "$BACKUP/$(basename "$f")"
  done
done
echo "backed up detected-BPM maps for prod/best to $BACKUP"

echo "=============================================================="
echo "STEP 1 — regenerate prod + best with the human-declared BPM"
echo "=============================================================="
python scripts/eval_sweep.py sweep --arms prod,best --force --true-bpm

echo
echo "=============================================================="
echo "STEP 2 — scorecard with correct tempo"
echo "=============================================================="
for ARM in prod best; do
  echo
  python -m beatsaber_automapper.evaluation.scorecard \
      "$CACHE"/${ARM}__*.zip --label "$ARM (true BPM)" 2>&1 | grep -v "INFO\|WARNING"
done

echo
echo "=============================================================="
echo "STEP 3 — same arms at DETECTED BPM, for the direct comparison"
echo "=============================================================="
for ARM in prod best; do
  echo
  python -m beatsaber_automapper.evaluation.scorecard \
      "$BACKUP"/${ARM}__*.zip --label "$ARM (detected BPM)" 2>&1 | grep -v "INFO\|WARNING"
done

echo
echo "=============================================================="
echo "COMPLETE — true-BPM isolation"
echo "=============================================================="
echo "Compare rhythm_gap true-BPM vs detected-BPM. A large drop means the next"
echo "investment is a tempo model; a small one means it is Stage-1 onset"
echo "selection. NOTE the cache now holds TRUE-BPM maps for prod/best;"
echo "detected-BPM originals are in $BACKUP."
