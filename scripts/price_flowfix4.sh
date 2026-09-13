#!/usr/bin/env bash
cd /home/kyle/repos/beatsaber_automapper
source .venv/bin/activate
OUT=outputs/price_2026-09-12
for sid in 1f335 1f336 1f3d7 1f65d 1f7f1 1f8a3 1f8ce 1f9a0 1f9f0 1fa32 1fa48 1fb3f 1f913 1f8d6 1f333 1f767; do
  python agent_mapper/autobuild.py data/eval_songset/$sid.ogg --pulse --lead-bias 0.2 \
      --lead-in --drop-orphan --name "F $sid" --out $OUT/FRAME__$sid.zip >/dev/null 2>&1
  echo "built $sid"
done
echo "FRAME_BUILDS_COMPLETE"
