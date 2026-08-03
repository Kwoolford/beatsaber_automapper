#!/usr/bin/env bash
# P0 — DOES SEEDING MAKE AN ARM'S VERDICT REPRODUCIBLE?
#
# THE PROBLEM. Five runs of a byte-identical configuration scored 4, 2, 1, 3 and
# 5 of the six axes. Nothing in the generation path was seeded, so "re-run the
# same arm" meant "draw a fresh map". Most single-run differences this project
# has ever reported sit inside that spread and are therefore unresolvable.
#
# THE FIX UNDER TEST (committed 7a5544d). generate.py --seed / BSA_SEED seeds
# torch (nucleus sampling + anti-repeat pick), python random (postprocess note
# deletion order + cut-direction reassignment) and numpy. Already verified at the
# map level on one song: seed 0 twice is byte-identical, seed 1 differs.
#
# WHAT THIS RUN ADDS. Map-level determinism does not by itself prove that a
# SCORE is reproducible or that ranking got easier. Two arms, three seeds each,
# all 23 songs:
#
#   tf_hl014_ds048   the current best arm and the promotion candidate
#   tf_hl014_ds055   its density neighbour -- deliberately a SMALL lever, because
#                    the small-effect regime is where this project keeps failing
#
# READING THE RESULT — three separate questions, do not conflate them:
#
# 1. REPRODUCIBILITY (the actual P0 DoD). The probe at the end regenerates three
#    songs at seed 0 and sha-compares them against the swept maps. Identical =>
#    an arm's score is now a function of its config, and re-running a sweep can
#    no longer change a verdict. Any difference => something else in the path is
#    non-deterministic (suspect CUDA kernel non-determinism) and P0 is NOT done.
#
# 2. ACROSS-SEED SPREAD. Expect this to be roughly UNCHANGED (~0.09 alignment).
#    Seeding does not make different seeds agree -- it makes each one repeatable.
#    Anyone reading a narrower sd here as "the fix worked" has misread it. The
#    win is that the seed is now a controlled variable rather than an unknown.
#
# 3. PAIRED vs UNPAIRED sd. The open question. Both arms start from the same RNG
#    state at a given seed, so their early decode draws coincide and part of the
#    seed effect should cancel. The pairing is PARTIAL -- the draw sequences
#    diverge as soon as the two configs make different numbers of decisions -- so
#    this is a measurement, not an assumption.
#       sd(paired) << sd(unpaired)  => rank levers with ~3 seeds instead of ~10;
#                                      this is the result that unblocks P0/P1.
#       sd(paired) ~= sd(unpaired)  => the density lever perturbs the decode too
#                                      early for pairing to help; fall back to
#                                      means over more seeds, and say so plainly.
#
# WHAT WOULD MAKE THIS RUN A FAILURE: the probe finds a difference (question 1).
# Everything else is informative either way.
#
# NOT tested here and still open: whether ds048 should be promoted. That is a
# call for Kyle, not for the scorecard -- he has already approved the sound by
# ear, and the suite scores this arm 4/6 at best.

set -uo pipefail
cd /home/kyle/repos/beatsaber_automapper
source .venv/bin/activate

LOG=logs/overnight/seedrepro_2026-08-02.log
mkdir -p logs/overnight
exec > >(tee -a "$LOG") 2>&1

echo "=== SEED REPRODUCIBILITY (2 arms x 3 seeds x 23 songs) START $(date -Is) ==="
echo "    ~62 s/map x 138 maps => expect ~2.4 h"

ARMS="tf_hl014_ds048,tf_hl014_ds055"
python scripts/eval_sweep.py sweep --arms "$ARMS" --seeds 3
echo "=== SWEEP DONE $(date -Is) ==="

echo
echo "=== PROBE: REGENERATE AT THE SAME SEED AND COMPARE BYTES $(date -Is) ==="
echo "This is the P0 DoD. Identical => a verdict is reproducible."
PROBE=outputs/seed_probe_2026-08-02
rm -rf "$PROBE"; mkdir -p "$PROBE"
CACHE=outputs/eval_sweep_cache
BEAT="logs/beat_classifier/version_4/checkpoints/beat-epoch=11-val_f1_avg_tol=0.603.ckpt"
LAYOUT="logs/layout_phrase/version_10/checkpoints/layout-epoch=09-val_token_acc=0.865.ckpt"

export DENSITY_SELECT=1 DENSITY_SELECT_GAMMA=2.5
export BEAT_DIFFICULTY_SCALE=0.48 BEAT_HAND_LEAD=0.14 BEAT_TEMPO_FIT=1

for song in 1f333 1f767 1f913; do
  src="$CACHE/tf_hl014_ds048#s0__${song}.zip"
  [ -f "$src" ] || { echo "  ! no swept map for $song -- skipping"; continue; }
  python scripts/generate.py "data/eval_songset/${song}.ogg" --v7 --difficulty Expert \
    --beat-ckpt "$BEAT" --layout-ckpt "$LAYOUT" \
    --section-gate loud_only --temperature 0.9 --top-p 0.97 --seed 0 \
    --output "$PROBE/${song}.zip" >/dev/null 2>&1
  a=$(unzip -p "$src" ExpertStandard.dat 2>/dev/null | sha256sum | cut -c1-16)
  b=$(unzip -p "$PROBE/${song}.zip" ExpertStandard.dat 2>/dev/null | sha256sum | cut -c1-16)
  if [ "$a" = "$b" ] && [ -n "$a" ]; then
    echo "  $song  REPRODUCED   $a"
  else
    echo "  $song  ** DIFFERS **  swept=$a probe=$b"
  fi
done

echo
echo "=== READ ==="
echo "  Probe all REPRODUCED  -> P0 DoD met: seeding makes a verdict repeatable."
echo "  Any DIFFERS           -> residual non-determinism; P0 stays open."
echo "  Then read the PAIRED table above: sd(paired) vs sd(unpaired) decides"
echo "  whether 3 seeds are enough to rank a small lever, or whether this"
echo "  project needs to stop ranking small levers at all."
echo "=== COMPLETE $(date -Is) ==="
