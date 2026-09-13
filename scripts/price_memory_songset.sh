#!/usr/bin/env bash
# Does --map-memory hold on the two songset maps that ALREADY SHIP?
# A lever is not characterised by the song it was aimed at. 1f767 is the song whose human has
# the WIDEST vocabulary in the songset (5.90 bits, 99th percentile) and where we already beat
# his echo — if narrowing helps there too, the mechanism is not about matching him.
cd /home/kyle/repos/beatsaber_automapper
source .venv/bin/activate
OUT=outputs/memory_songset_2026-09-13
mkdir -p $OUT
for sid in 1f767 1f8d6; do
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
echo "MEMORY_SONGSET_COMPLETE"
