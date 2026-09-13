#!/usr/bin/env bash
# SIX seeds, to RESOLVE idiom_coverage — the one axis standing between --map-memory and a default.
# At three seeds it read +0.017 / -0.058 / -0.016 / -0.068 across the songset, every one INSIDE
# 2sd. "Not resolvable" is not "unharmed", and it is the exact axis the palette flip forgot.
# 1f333 is the standing red; 1f8d6 carried the widest arm sd (0.102) of the four.
cd /home/kyle/repos/beatsaber_automapper
source .venv/bin/activate
OUT=outputs/memory_cov_2026-09-13
mkdir -p $OUT
for sid in 1f333 1f8d6; do
  for seed in 4 5 6 7 8 9; do
    for m in 0 4; do
      python agent_mapper/autobuild.py data/eval_songset/$sid.ogg --pulse --lead-bias 0.2 \
          --lead-in --drop-orphan --carrier-bias 2.0 --map-memory $m --seed $seed \
          --name "M$m s$seed $sid" --out $OUT/M${m}_s${seed}__$sid.zip >/dev/null 2>&1
      python agent_mapper/repeat.py $OUT/M${m}_s${seed}__$sid.zip \
          --out $OUT/M${m}_s${seed}__$sid.zip --song $sid >/dev/null 2>&1
      echo "built $sid seed=$seed m=$m"
    done
  done
done
echo "MEMORY_COV_COMPLETE"
