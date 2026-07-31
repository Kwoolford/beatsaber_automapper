#!/usr/bin/env bash
# B-1: score the instrument retrain (version_8) BY THE v2 SUITE, not by val_f1.
#
# Background: B-0 (2026-07-28) re-evaluated the shelved version_7 instrument ckpt
# and got a MIXED result -- density_corr (the thing instrument features were built
# for) improved +0.402 -> +0.453, but rhythm/idiom/handrole all got worse. That
# comparison was CONFOUNDED: version_7's only surviving checkpoints were epochs
# 0/2/7 (old hardcoded save_top_k=3 + early stopping on val_f1_avg_tol), so we
# compared an epoch-0 model against version_4's epoch-11. "Instrument features
# hurt quality" and "that model was undertrained" were not separable.
#
# version_8 fixes exactly that: 18 full epochs, --save-top-k -1, every epoch saved.
# Confirmed structurally distinct from version_4 -- it has an `instr_proj` head
# (512x10) where version_4 has only drum_proj + mix_proj, i.e. this is the model
# that can actually hear separate instruments (the 2026-07-27 representation gap).
#
# ARMS (14): b1_e{00,03,06,09,12,15,17} at prod density, and the same 7 at the
# Track A difficulty scale (_ds055), because prod density is a tier too dense to
# judge anything at and the density lever composes with everything.
# CONTROLS: `prod` (version_4, no instrument features) and `ds055` are already
# cached from prior sweeps; `v7instr` is the confounded B-0 arm.
#
# VERDICT LOGIC (printed at the end):
#   * best-by-suite b1 epoch beats `prod` on the 5-axis scorecard
#       => the instrument representation earns its retrain; B-0's regressions were
#          the undertraining. Next: fill in epochs around the optimum, then render
#          for Kyle.
#   * the whole epoch CURVE sits at/below `prod`
#       => B-0's regressions were the REPRESENTATION, not the training budget.
#          More epochs will not save it; Track B moves to B-2 (per-stem MERT).
#   * density_corr up but the other axes down at every epoch
#       => instrument features buy density-tracking at a cost elsewhere; that is a
#          real finding, not a failure -- log it and decide whether density_corr is
#          worth the trade only AFTER Kyle hears one.
# NOTE val_f1_avg_tol across version_8's epochs oscillates 0.562-0.599 with no
# trend. It is deliberately NOT the selection metric. Expect it to disagree with
# the suite -- that disagreement is the point (it has anti-correlated 3x now).

set -uo pipefail
cd /home/kyle/repos/beatsaber_automapper
source .venv/bin/activate

LOG=logs/overnight/b1_score_2026-07-30.log
mkdir -p logs/overnight
exec > >(tee -a "$LOG") 2>&1

echo "=== B-1 SUITE SCORING START $(date -Is) ==="

ARMS="b1_e00,b1_e03,b1_e06,b1_e09,b1_e12,b1_e15,b1_e17"
ARMS="$ARMS,b1_e00_ds055,b1_e03_ds055,b1_e06_ds055,b1_e09_ds055,b1_e12_ds055,b1_e15_ds055,b1_e17_ds055"

python scripts/eval_sweep.py sweep --arms "$ARMS"
echo "=== SWEEP DONE $(date -Is) ==="

# eval_sweep's own rhythm table silently prints nan (never wires rhythm records
# into its per-song dict) -- score with scorecard.py directly for a trustworthy
# 5-axis readout, one cohort per arm, plus the two controls.
echo "=== 5-AXIS SCORECARD (scorecard.py, the trustworthy path) ==="
for arm in prod ds055 v7instr $(echo "$ARMS" | tr ',' ' '); do
  zips=(outputs/eval_sweep_cache/${arm}__*.zip)
  if [ ! -e "${zips[0]}" ]; then
    echo "-- $arm: NO CACHED MAPS, skipping"
    continue
  fi
  echo "-- $arm (${#zips[@]} maps)"
  python -m beatsaber_automapper.evaluation.scorecard "${zips[@]}" --label "$arm" || true
done

echo "=== VERDICT $(date -Is) ==="
python - <<'PY'
import pathlib, re, subprocess, sys
log = pathlib.Path("logs/overnight/b1_score_2026-07-30.log").read_text(errors="replace")
print("""
READ THE SCORECARD TABLES ABOVE AGAINST THESE BARS:
  flow <= 0.50 | rhythm <= 0.70 | idiom <= 1.00 | handrole <= 2.00 | playfeel <= 1.00
  (a spread < 0.35 fails an axis even when its gap passes -- mode collapse)
BASELINES: prod 0.71/2.37/1.85/3.23/2.29 (fails all 5); ds055 0.30/0.36/0.52/1.92*/0.74 (4/5)
NOISE FLOOR: flow +-0.03 rhythm +-0.08 idiom +-0.09 handrole +-0.29 -- read every
delta against these, a change smaller than the floor is NOT a result.

VERDICT:
  (a) some b1 epoch beats prod on >=3 axes  -> instrument rep earns the retrain,
      fill in epochs around the optimum, render the winner for Kyle.
  (b) the whole curve sits at/below prod    -> B-0's regressions were the
      REPRESENTATION. Stop adding epochs; Track B goes to B-2 (per-stem MERT).
  (c) density_corr up, other axes down      -> real trade, not a failure. Log it,
      do not promote on density_corr alone.
COMPARE _ds055 arms against ds055 (NOT against prod) -- that pair isolates the
instrument representation at a difficulty tier that is actually judgeable.
""")
PY
echo "=== COMPLETE $(date -Is) ==="
