#!/usr/bin/env bash
# REPLICATE THE NIGHT'S HEADLINE AT AN INDEPENDENT SEED.
#
# `diag_full` closed ~45% of the rhy_rhythm gap and ~51% of harm_rhythm at n=149 — the
# first resolvable movement of a masterpiece axis in this project's history. It was
# measured at ONE SEED, and `masterpiece_report` prints its own warning about that:
# "at one seed treat it as a screen". The standing rule here is >=3 seeds, and this
# project has been burned twice by small-n effects (n=3 lied about idiom; n=13 inflated
# effect sizes 3-20x). A headline this size gets checked before it gets believed.
#
# The seed-1 prod control already exists (outputs/wide_cohort_prod_s1, 149 maps, same
# songs), so this is a fully PAIRED replication: same songs, same audio, seed 1 on both
# sides, differing in exactly the lever.
#
# 🔴READ IT AS: the measured seed-noise floor for these axes is ~+-0.004. The seed-0
# effects were +0.0423 (rhy_rhythm) and +0.0542 (harm_rhythm), i.e. ~10x that floor.
#   HOLDS      both stay well clear of +-0.004 and within ~30% of the seed-0 size ⇒ the
#              result is real and the write-up stands.
#   SHRINKS    effect drops toward the floor ⇒ seed 0 was lucky; downgrade the claim to
#              PARTLY CONFIRMED and say so in PROGRESS.md before anything is built on it.
set -u
cd "$(dirname "$0")/.." || exit 1
LOG=logs/overnight/me3_seed1_2026-08-11.log
mkdir -p logs/overnight outputs/me_2026-08-11
exec > >(tee -a "$LOG") 2>&1
PY=.venv/bin/python

echo "=== diag_full @ seed 1 — replication — $(date) ==="
while pgrep -f "overnight_2026-08-11.sh" > /dev/null; do sleep 60; done

$PY scripts/build_wide_cohort.py --n 150 --seed 1 --variant prod \
    --tag diag_full --env "BEAT_STRUCTURE_REUSE=diag_full:0.70:4:1.5:2.0:4"

D=outputs/wide_cohort_prod_s1_diag_full
N=$(ls "$D"/*.zip 2>/dev/null | wc -l)
echo ""; echo "--- EVAL diag_full@s1 ($N maps) vs the seed-1 control --- $(date +%H:%M)"
[ "$N" -lt 100 ] && { echo "SKIP: only $N maps"; exit 1; }

$PY scripts/masterpiece_report.py --arm diag_full_s1 --wide --wide-dir "$D" \
    --vs prod_s1 --vs-wide-dir outputs/wide_cohort_prod_s1 \
    --json outputs/me_2026-08-11/masterpiece_diag_full_s1.json
echo "  -- six-axis (seed-0 arm was flow 0.70 / idiom 1.75 / playfeel 0.88) --"
$PY -m beatsaber_automapper.evaluation.scorecard "$D"/*.zip --label diag_full_s1
echo "  -- seed-1 CONTROL six-axis, for the honest comparison --"
$PY -m beatsaber_automapper.evaluation.scorecard outputs/wide_cohort_prod_s1/*.zip --label control_s1

echo ""; echo "=== COMPARE TO SEED 0 ==="
echo "seed 0: rhy_rhythm +0.0423, harm_rhythm +0.0542, harm_place +0.0225 (floor +-0.004)"
echo "If seed 1 lands near those, the result is real. If it collapses toward the floor,"
echo "downgrade to PARTLY CONFIRMED in PROGRESS.md before building anything on it."
echo "COMPLETE $(date)"
