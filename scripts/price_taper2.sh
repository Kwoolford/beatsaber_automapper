#!/usr/bin/env bash
cd /home/kyle/repos/beatsaber_automapper
source .venv/bin/activate
OUT=outputs/price_2026-09-12
for sid in 1f335 1f3d7 1f65d 1f333 1f767 1f913 1f8d6 1fa32; do
  python agent_mapper/autobuild.py data/eval_songset/$sid.ogg --pulse --lead-bias 0.2 \
      --lead-in --drop-orphan --carrier-bias 2.0 --taper 0.8 --name "T2 $sid" \
      --out $OUT/TP2__$sid.zip >/dev/null 2>&1
done
echo "TAPER2_COMPLETE"
