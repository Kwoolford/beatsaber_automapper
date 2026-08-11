#!/usr/bin/env bash
# LAYOUT_TRAVEL_PENALTY=1 — the SECOND (and last) validated-but-unshipped lever.
#
# A sweep of PROGRESS.md for levers ticked ✅ in a results table and never promoted
# returns exactly two, both from the 2026-07-27 flow/idiom sweep:
#   COLOR_SEP_MODE=extreme      idiom 1.84 -> 0.30 PASS   ✅ re-validated today at n=149
#   LAYOUT_TRAVEL_PENALTY=1     flow  0.81 -> 0.30 PASS   <- this run
# It still defaults to 0.0 in `models/layout_model.py`.
#
# ⚠️THE REASON TO EXPECT LESS THIS TIME. That result is from BEFORE the tempo fix, the
# promotion, and BEAT_REACH. The defect it fixed — flow at 0.81 — no longer exists:
# today's control sits at flow **0.37 PASS**, and COLOR_SEP_MODE=extreme takes it to
# **0.23**. A lever that repaired a hole somebody else has since filled can easily be
# neutral now, or harmful by over-correction: the same sweep recorded
# `LAYOUT_TRAVEL_PENALTY=4` over-correcting to flow 1.77 with **spread 0.00 — every map
# identical**, which is the degenerate this project fears most.
#
# 🔴PRE-REGISTERED READING:
#   KEEP      flow improves on 0.37 without collapsing `spread` (bar 0.35) and nothing
#             else regresses ⇒ test it stacked on COLOR_SEP_MODE=extreme next.
#   REDUNDANT flow unchanged within noise ⇒ BEAT_REACH already did this job; record it as
#             superseded so nobody re-runs it a third time.
#   HARMFUL   spread falls toward 0.00, or flow over-corrects ⇒ dead, and say so loudly.
# ⚠️Watch `spread`, not just `gap`. A lever that makes every map identical scores a
# beautiful gap and is worthless — that is exactly how tp4 failed.
set -u
cd "$(dirname "$0")/.." || exit 1
LOG=logs/overnight/tp1_2026-08-11.log
mkdir -p logs/overnight outputs/me_2026-08-11
exec > >(tee -a "$LOG") 2>&1
PY=.venv/bin/python
echo "=== LAYOUT_TRAVEL_PENALTY=1 @ n=149 — $(date) ==="

$PY scripts/build_wide_cohort.py --n 150 --seed 0 --variant prod \
    --tag tp1 --env "LAYOUT_TRAVEL_PENALTY=1"

D=outputs/wide_cohort_prod_tp1
N=$(ls "$D"/*.zip 2>/dev/null | wc -l)
echo ""; echo "--- EVAL tp1 ($N maps) --- $(date +%H:%M)"
[ "$N" -lt 100 ] && { echo "SKIP: only $N maps"; exit 1; }
echo "  control : flow 0.37 (spread 0.66) rhythm 0.47 idiom 0.40 handrole 1.12 playfeel 0.59 alignment 0.62"
echo "  xsep    : flow 0.23 (spread 0.62) rhythm 0.47 idiom 0.52 handrole 1.12 playfeel 0.62 alignment 0.62"
$PY -m beatsaber_automapper.evaluation.scorecard "$D"/*.zip --label tp1
$PY scripts/eval_reachability.py --maps "$D/*.zip" --label tp1 \
    --maps outputs/wide_cohort/"*.zip" --label control \
    --json outputs/me_2026-08-11/reach_tp1.json
echo "COMPLETE $(date)"
