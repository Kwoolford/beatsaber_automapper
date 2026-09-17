#!/usr/bin/env bash
# 2026-09-16: the same 23 builds with --no-phase-calibrate (raw per-song phase), for P1.0.
# Best build, seed 1, every eval song; read by scripts/exp_phase_sweep.py.
cd /home/kyle/repos/beatsaber_automapper
source .venv/bin/activate
OUT=outputs/phase16raw_2026-09-16
mkdir -p $OUT
for f in data/eval_songset/*.ogg; do
  sid=$(basename $f .ogg)
  python agent_mapper/autobuild.py $f --pulse --lead-bias 0.2 --lead-in --drop-orphan \
      --carrier-bias 2.0 --seed 1 --no-phase-calibrate --name "p16 $sid" --out $OUT/B__$sid.zip >/dev/null 2>&1
  python agent_mapper/repeat.py $OUT/B__$sid.zip --out $OUT/B__$sid.zip --song $sid >/dev/null 2>&1
  echo "built $sid"
done
echo "PHASE16RAW_COMPLETE"
