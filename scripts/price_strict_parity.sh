#!/usr/bin/env bash
# Does aligning idiomize's parity rule to fix_parity's help, hurt, or neither, on FULL builds?
# The shipped map's parity is identical either way -- fix_parity removes every reset regardless.
# What changes is WHO picks the direction: the vocabulary-aware sampler, or the blind repair.
# DoD: idiom_coverage does not fall, judge p does not fall, resets and violations stay at 0,
# and no songset map gains a red.
cd /home/kyle/repos/beatsaber_automapper
source .venv/bin/activate
OUT=outputs/strict_2026-09-13
mkdir -p $OUT
for sid in 1f913 1f8d6 1f767 1f333; do
  for seed in 1 2 3; do
    for arm in strict resets; do
      EXTRA=""
      [ "$arm" = "resets" ] && EXTRA="--allow-resets"
      python agent_mapper/autobuild.py data/eval_songset/$sid.ogg --pulse --lead-bias 0.2 \
          --lead-in --drop-orphan --carrier-bias 2.0 --seed $seed $EXTRA \
          --name "$arm s$seed $sid" --out $OUT/${arm}_s${seed}__$sid.zip >/dev/null 2>&1
      python agent_mapper/repeat.py $OUT/${arm}_s${seed}__$sid.zip \
          --out $OUT/${arm}_s${seed}__$sid.zip --song $sid >/dev/null 2>&1
      echo "built $sid seed=$seed $arm"
    done
  done
done
echo "STRICT_COMPLETE"
