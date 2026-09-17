#!/usr/bin/env bash
# Does --palette still collapse idiom_coverage now that the parity leak is closed? (2026-09-16)
# 2026-09-12i put the palette FILTER at coverage 0.618 (1.7th pct) and the boost at 0.998 (97.5th).
# Both were measured BEFORE STRICT_PARITY (2026-09-13x) closed the vocabulary-blind repair that
# caused --map-memory's identical collapse, and before the boost got its median guard. Never
# re-measured since. ⚠️memory_cov2 is NOT the control (it predates REPEAT_P 0.55 -> 0.25); the
# control is a default rebuild against repeatp's rp0.25_s1 -- see eval_palette_strict.py.
cd /home/kyle/repos/beatsaber_automapper
source .venv/bin/activate
OUT=outputs/palette_strict_2026-09-16
mkdir -p $OUT
for sid in 1f333 1f8d6; do
  for seed in 4 5 6 7 8 9; do
    for p in 0 20; do
      python agent_mapper/autobuild.py data/eval_songset/$sid.ogg --pulse --lead-bias 0.2 \
          --lead-in --drop-orphan --carrier-bias 2.0 --palette $p --seed $seed \
          --name "M0 s$seed $sid" --out $OUT/P${p}_s${seed}__$sid.zip >/dev/null 2>&1
      python agent_mapper/repeat.py $OUT/P${p}_s${seed}__$sid.zip \
          --out $OUT/P${p}_s${seed}__$sid.zip --song $sid >/dev/null 2>&1
      echo "built $sid seed=$seed palette=$p"
    done
  done
done
echo "PALETTE_STRICT_COMPLETE"
