#!/usr/bin/env bash
# THE SHIPPABLE FORM OF M-E — dose-capped, and the question is what the cap costs.
#
# WHAT IS SETTLED. `diag_full` closes ~45-51% of the rhy_rhythm / harm_rhythm gap,
# REPLICATED at two seeds (+0.0423/+0.0415 and +0.0542/+0.0519). The axes are not being
# gamed: a fixed-lag periodic map scores rhy_rhythm 0.0125 / harm_rhythm 0.0007 against
# our 0.0924 / 0.0712 -- near zero, below even the control.
#
# WHAT IS BROKEN. At high copy share the map goes degenerate in a way NO axis reported:
# distinct bar patterns per bar 0.427 / 0.496 at ~71% share, against a human 0.951.
# ⚠️The cohort MEAN hid it perfectly (0.880 vs human 0.883) -- the project's own
# "a cohort median cannot see a subset-of-songs defect" trap.
#
# THE FIX UNDER TEST: `max_share` (default 0.20). On 60 songs it takes copy share from
# max 0.705 to max 0.205, and predicted diversity to mean 0.946 / min 0.835 -- human
# range. Stripes are kept strongest-first and whole, so the cap spends the budget on the
# most confident repeats.
#
# 🔴THE PRE-REGISTERED READING. Dose roughly halves (0.292 -> 0.149), so a gain that
# scales with dose should roughly halve too, to ~+0.021 / +0.026 -- still 5-6x the
# +-0.004 seed floor.
#   SHIP    structural gain stays resolvable AND no song's diversity falls below ~0.85
#           of its human ⇒ this is the config to put in front of Kyle, and the
#           uncapped arms are a diagnostic footnote.
#   THIN    gain collapses toward the floor ⇒ the gain was substantially the degenerate
#           after all. Say so plainly: the structural movement would then be mostly an
#           artifact of over-repetition, and M-E's headline needs rewriting.
set -u
cd "$(dirname "$0")/.." || exit 1
LOG=logs/overnight/me4_capped_2026-08-11.log
mkdir -p logs/overnight outputs/me_2026-08-11
exec > >(tee -a "$LOG") 2>&1
PY=.venv/bin/python
echo "=== M-E dose-capped — $(date) ==="
while pgrep -f "overnight_2026-08-11b.sh" > /dev/null; do sleep 60; done

$PY scripts/build_wide_cohort.py --n 150 --seed 0 --variant prod \
    --tag diag_capped --env "BEAT_STRUCTURE_REUSE=diag_full:0.70:4:1.5:2.0:4:0.20"

D=outputs/wide_cohort_prod_diag_capped
N=$(ls "$D"/*.zip 2>/dev/null | wc -l)
echo ""; echo "--- EVAL diag_capped ($N maps) --- $(date +%H:%M)"
[ "$N" -lt 100 ] && { echo "SKIP: only $N maps"; exit 1; }

$PY scripts/masterpiece_report.py --arm diag_capped --wide --wide-dir "$D" \
    --vs prod --vs-wide-dir outputs/wide_cohort \
    --json outputs/me_2026-08-11/masterpiece_diag_capped.json
echo "  -- six-axis (control flow 0.37 / idiom 0.40; uncapped arm 0.70 / 1.75) --"
$PY -m beatsaber_automapper.evaluation.scorecard "$D"/*.zip --label diag_capped
echo "  -- reachability (control hard_rate 0.0494, reach_median 2.8284) --"
$PY scripts/eval_reachability.py --maps "$D/*.zip" --label diag_capped \
    --maps outputs/wide_cohort/"*.zip" --label control \
    --json outputs/me_2026-08-11/reach_diag_capped.json

echo ""; echo "=== READ AGAINST ==="
echo "uncapped seed0: rhy_rhythm +0.0423  harm_rhythm +0.0542  harm_place +0.0225"
echo "uncapped seed1: rhy_rhythm +0.0415  harm_rhythm +0.0519  harm_place +0.0186"
echo "periodic degen: rhy_rhythm +0.0125  harm_rhythm +0.0007  (near zero = axes are honest)"
echo "seed floor +-0.004. Dose halved, so ~half the gain is the expected SHIP outcome."
echo "COMPLETE $(date)"
