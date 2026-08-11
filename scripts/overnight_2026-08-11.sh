#!/usr/bin/env bash
# M-E ROUND 2 — DECODE THE STRIPE, NOT THE BAR.
#
# WHAT ROUND 1 SETTLED. `me_z20` (per-bar copy) broke **flow 0.37 -> 0.75** and
# **idiom 0.40 -> 1.07** against the 149-song control, both across their bars, for a
# `harm_place` gain of +0.0008 against a 0.0200 gap. Every rhythm-side axis was
# identical to 4 dp, so the time-neutrality property held exactly and the damage is
# unambiguously in position/direction.
#
# WHY. Only **15.6 %** of copied bars continued the previous bar's copy — the lever
# shuffled ~29 bars per song in from two dozen places, and placement is not
# context-free. The cause is tie-breaking, not an absence of sections: when a chorus
# returns four times, adjacent bars pick different equally-good sources because each
# bar is decided alone. **That is C1 one level up.**
#
# THE FIX UNDER TEST. A repeated section is a DIAGONAL STRIPE in the self-similarity
# matrix. `plan_reuse_diagonal` decodes whole stripes, so contiguity is a property of
# the representation rather than a filter: copy share **0.297 -> 0.428** with
# contiguity **0.156 -> 0.648**, on 56/60 songs. It finds MORE repeats and they hang
# together.
#
# 🔴DoD, PRE-REGISTERED. The lever is only interesting if it buys `harm_place` WITHOUT
# the flow/idiom damage that killed round 1:
#   PASS   harm_place rises AND flow stays under its 0.50 bar AND idiom under 1.00
#          (control sits at 0.37 / 0.40, so there is real headroom) AND hard_rate does
#          not rise AND reach_median does not shrink the way round 1's did (2.83 ->
#          2.24 = "fixed it by making everything small").
#   DEAD   flow/idiom degrade like round 1 ⇒ the damage is inherent to copying
#          placement across contexts, not to the shuffle, and M-E's place mode is
#          finished. Say so plainly; do not tune it a third time.
# ⚠️`harm_place` remains a MANIPULATION CHECK, not evidence of quality — and Kyle,
# 2026-08-10: "the metrics still don't capture the full picture."
set -u
cd "$(dirname "$0")/.." || exit 1
LOG=logs/overnight/me2_2026-08-11.log
mkdir -p logs/overnight outputs/me_2026-08-11
exec > >(tee -a "$LOG") 2>&1
PY=.venv/bin/python

# ONE GPU JOB AT A TIME. Wait for round 1 to finish rather than competing with it.
echo "=== M-E round 2 — waiting for round 1 to finish — $(date) ==="
while pgrep -f "overnight_2026-08-10.sh" > /dev/null; do sleep 60; done
echo "round 1 clear at $(date)"

CTRL=outputs/wide_cohort
# ⚠️DESIGN FIX made before this burned any GPU. The first draft ran the diagonal
# planner at its natural settings, which copies MORE than round 1 did (share 0.428 vs
# 0.297). That changes TWO things at once -- contiguity and dose -- and the standing
# rule here is one lever at a time. A confounded arm cannot answer "was the shuffle the
# problem", which is the entire question round 2 exists to settle.
#   diag_dose  min_sim 0.70 -> share 0.292, contiguity 0.635
#              DOSE-MATCHED to me_z20 (share 0.297, contiguity 0.156). Same amount of
#              copying, 4x the contiguity ⇒ contiguity is the ONLY variable, and the
#              comparison against me_z20's flow 0.75 / idiom 1.07 is clean.
#   diag_full  same settings, but copies the bar's RHYTHM too. ★RUN FIRST -- it is now
#              the most valuable arm in the project. `me_full25` (full mode, per-bar)
#              produced the FIRST resolvable movement of a masterpiece axis this
#              project has ever had: rhy_rhythm +0.0175 and harm_rhythm +0.0280 against
#              a +-0.004 seed floor, and harm_place +0.0091 = 11x what copying
#              placement alone bought. It paid with idiom 0.40 -> 2.34. Round 1 showed
#              the copies are scattered across contexts; if that scattering is what
#              makes them idiomatically alien, contiguity should cut the bill while
#              keeping the structural gain. That is the whole experiment.
# (The earlier `diag_wide` share-0.428 arm is dropped -- a confounded ceiling probe is
#  worth less than the question above, and GPU time is the scarce thing.)
declare -A ARMS=(
  [diag_dose]="diag_place:0.70:4:1.5:2.0:4"
  [diag_full]="diag_full:0.70:4:1.5:2.0:4"
)
ORDER=(diag_full diag_dose)

for arm in "${ORDER[@]}"; do
  echo ""; echo "--- GENERATE $arm  (${ARMS[$arm]}) --- $(date +%H:%M)"
  $PY scripts/build_wide_cohort.py --n 150 --seed 0 --variant prod \
      --tag "$arm" --env "BEAT_STRUCTURE_REUSE=${ARMS[$arm]}"
done

for arm in "${ORDER[@]}"; do
  d="outputs/wide_cohort_prod_${arm}"
  n=$(ls "$d"/*.zip 2>/dev/null | wc -l)
  echo ""; echo "--- EVAL $arm ($n maps) --- $(date +%H:%M)"
  [ "$n" -lt 100 ] && { echo "  SKIP: only $n maps"; continue; }
  $PY scripts/masterpiece_report.py --arm "$arm" --wide --wide-dir "$d" \
      --vs prod --vs-wide-dir "$CTRL" \
      --json "outputs/me_2026-08-11/masterpiece_${arm}.json"
  echo "  -- six-axis suite (control: flow 0.37 / idiom 0.40 / playfeel 0.59) --"
  $PY -m beatsaber_automapper.evaluation.scorecard "$d"/*.zip --label "$arm"
  echo "  -- reachability (control: hard_rate 0.0494, reach_median 2.8284) --"
  $PY scripts/eval_reachability.py --maps "$d/*.zip" --label "$arm" \
      --maps "$CTRL/*.zip" --label control \
      --json "outputs/me_2026-08-11/reach_${arm}.json"
  echo "  -- did the copy survive postprocess? --"
  $PY scripts/check_reuse_survives.py --arm "$d" --min-z 2.0
done

echo ""; echo "=== VERDICT INPUTS — $(date) ==="
echo "control              : flow 0.37 PASS, idiom 0.40 PASS, hard_rate 0.0494"
echo "me_z25  (share 0.190): flow 0.61 FAIL, idiom 0.69 PASS"
echo "me_z20  (share 0.297): flow 0.75 FAIL, idiom 1.07 FAIL, harm_place +0.0008"
echo "full25  (share 0.190): flow 0.81 FAIL, idiom 2.34 FAIL, playfeel 1.03 FAIL"
echo ""
echo "full25  M-axes      : rhy_rhythm +0.0175, harm_rhythm +0.0280, harm_place +0.0091"
echo "                      (first resolvable masterpiece-axis movement in the project)"
echo ""
echo "diag_dose is DOSE-MATCHED to me_z20 -- compare against flow 0.75 / idiom 1.07."
echo "diag_full is THE arm: does contiguity keep full mode's structural gain while"
echo "cutting its idiom bill (2.34) back toward the 1.00 bar? Structural gain WITHOUT"
echo "the idiom blowup = the first real M-E result and worth Kyle's ear. Gain but idiom"
echo "still broken = the structural gap is buyable and we do not yet know how to afford"
echo "it -- say exactly that, and stop tuning placement copies."
echo "If a diag arm keeps flow < 0.50 and idiom < 1.00 while harm_place still rises,"
echo "the shuffle was the problem and this is worth Kyle's ear. If flow/idiom break"
echo "the same way, copying placement across contexts is the problem and place mode"
echo "is DONE -- write that down and stop tuning it."
echo "COMPLETE $(date)"
