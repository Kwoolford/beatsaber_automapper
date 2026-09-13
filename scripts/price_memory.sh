#!/usr/bin/env bash
# What `--map-memory` costs and buys, on the FULL build and on every axis that matters.
# ★The DoD must include `idiom_coverage` — the axis the pass being changed was BUILT for.
# The palette sweep measured echo, vocabulary, top-10 and local variety, flipped a default,
# and a full build then showed coverage at the 1.7th human percentile (PROGRESS 2026-09-12i).
# ⚠️NOT comparable to that run's numbers: this build carries --pulse --lead-in --drop-orphan
# --carrier-bias, and its map-memory 0 control reads idiom_coverage 0.766 where the 09-12i
# palette 0 control read 0.992. Read the arms against THIS control, never across sessions.
cd /home/kyle/repos/beatsaber_automapper
source .venv/bin/activate
OUT=outputs/memory_2026-09-13
mkdir -p $OUT
for sid in 1f913 1f333; do
  for m in 0 2 4 6; do
    python agent_mapper/autobuild.py data/eval_songset/$sid.ogg --pulse --lead-bias 0.2 \
        --lead-in --drop-orphan --carrier-bias 2.0 --map-memory $m --seed 0 \
        --name "M$m $sid" --out $OUT/M${m}__$sid.zip >/dev/null 2>&1
    python agent_mapper/repeat.py $OUT/M${m}__$sid.zip --out $OUT/M${m}__$sid.zip \
        --song $sid >/dev/null 2>&1
    echo "built $sid m=$m"
  done
done
echo "MEMORY_SWEEP_COMPLETE"
