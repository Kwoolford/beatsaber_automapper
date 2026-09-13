#!/usr/bin/env bash
# ★THREE SEEDS, because the one-seed sweep was NOT a dose response.
# 1f333 read idiom_coverage 0.838 -> 0.515 -> 0.840 -> 0.439 across map-memory 0/2/4/6.
# A knob whose effect reverses twice as the dose rises is either seed noise or a threshold
# effect, and one seed cannot tell those apart. 1f913 WAS monotone over the same arms
# (echo 0.507 -> 0.595, coverage 0.766 -> 0.853), which is exactly why the spread matters.
cd /home/kyle/repos/beatsaber_automapper
source .venv/bin/activate
OUT=outputs/memory_seeds_2026-09-13
mkdir -p $OUT
for sid in 1f913 1f333; do
  for seed in 1 2 3; do
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
echo "MEMORY_SEEDS_COMPLETE"
