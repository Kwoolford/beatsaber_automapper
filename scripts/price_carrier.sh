#!/usr/bin/env bash
cd /home/kyle/repos/beatsaber_automapper
source .venv/bin/activate
OUT=outputs/price_2026-09-12
for cb in 2.0 4.0; do
for sid in 1f335 1f336 1f3d7 1f65d 1f7f1 1f8ce 1f9f0 1fa32 1f913 1f333 1f767 1f8d6; do
  python agent_mapper/autobuild.py data/eval_songset/$sid.ogg --pulse --lead-bias 0.2 \
      --lead-in --drop-orphan --carrier-bias $cb --name "C$cb $sid" \
      --out $OUT/CB${cb}__$sid.zip >/dev/null 2>&1
done
echo "done cb=$cb"
done
echo "CARRIER_BUILDS_COMPLETE"
