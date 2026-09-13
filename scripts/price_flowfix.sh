#!/usr/bin/env bash
# Price --lead-in --drop-orphan across 12 corpus songs, both arms, against each song's own human.
# DoD (TODO P0.6b): FLOW stays clear, idiom_jsd does not fall below its baseline percentile on any
# song, note count within +-5%.
cd /home/kyle/repos/beatsaber_automapper
source .venv/bin/activate
OUT=outputs/price_2026-09-12
for sid in 1f335 1f336 1f3d7 1f65d 1f7f1 1f8a3 1f8ce 1f9a0 1f9f0 1fa32 1fa48 1fb3f; do
  python agent_mapper/autobuild.py data/eval_songset/$sid.ogg --pulse --lead-bias 0.2 \
      --name "BASE $sid" --out $OUT/BASE__$sid.zip >/dev/null 2>&1
  python agent_mapper/autobuild.py data/eval_songset/$sid.ogg --pulse --lead-bias 0.2 \
      --lead-in --drop-orphan --name "FIX $sid" --out $OUT/FIX__$sid.zip >/dev/null 2>&1
  echo "built $sid"
done
echo "PRICE_BUILDS_COMPLETE"
