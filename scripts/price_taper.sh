#!/usr/bin/env bash
cd /home/kyle/repos/beatsaber_automapper
source .venv/bin/activate
OUT=outputs/price_2026-09-12
for tp in 0.5 0.8; do
for sid in 1f335 1f3d7 1f65d 1f333 1f767 1f913 1f8d6 1fa32; do
  python agent_mapper/autobuild.py data/eval_songset/$sid.ogg --pulse --lead-bias 0.2 \
      --lead-in --drop-orphan --carrier-bias 2.0 --taper $tp --name "T$tp $sid" \
      --out $OUT/TP${tp}__$sid.zip >/dev/null 2>&1
done
echo "done taper=$tp"
done
echo "TAPER_BUILDS_COMPLETE"
