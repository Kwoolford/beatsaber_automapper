#!/usr/bin/env bash
# `repeat_p` has NEVER been reachable from a full build, so every map shipped used 0.55 blind.
# Isolated, 6 seeds: idiom_local 0.900/0.865/0.812/0.755 at p=0/.25/.55/.80 with block echo FLAT
# (0.454/0.450/0.441/0.436). Human median idiom_local is 0.867, which p=0.25 lands on.
# DoD: idiom_local rises toward the human median, echo does not fall, idiom_coverage and judge p
# do not fall, no songset map gains a red.
cd /home/kyle/repos/beatsaber_automapper
source .venv/bin/activate
OUT=outputs/repeatp0_2026-09-13
mkdir -p $OUT
for sid in 1f913 1f8d6 1f767 1f333; do
  for seed in 1 2 3; do
    for rp in 0.25 0.00; do
      python agent_mapper/autobuild.py data/eval_songset/$sid.ogg --pulse --lead-bias 0.2 \
          --lead-in --drop-orphan --carrier-bias 2.0 --seed $seed --repeat-p $rp \
          --name "rp$rp s$seed $sid" --out $OUT/rp${rp}_s${seed}__$sid.zip >/dev/null 2>&1
      python agent_mapper/repeat.py $OUT/rp${rp}_s${seed}__$sid.zip \
          --out $OUT/rp${rp}_s${seed}__$sid.zip --song $sid >/dev/null 2>&1
      echo "built $sid seed=$seed rp=$rp"
    done
  done
done
echo "REPEATP0_COMPLETE"
