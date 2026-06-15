#!/usr/bin/env bash
# 2026-06-11 overnight: wait for the 400-map V7 cohort to finish generating, then
# train the 2b feel-discriminator at SCALE + run feature ablations to characterize
# WHAT drives the human-vs-V7 separation (smoke @ n=65 hit AUC 1.0 — confirm at scale
# and decompose by timing/spatial/direction so we know the reward isn't a degenerate
# fingerprint before building best-of-N on it).
#
# DoD: full-cohort held-out AUC(human vs V7) >= 0.75 (none arm). Read the ablation
# arms to see which feature group the reward leans on (a high-AUC dt arm = it reads
# our metronomic timing; high dir arm = our for-sport swing patterns; etc.).
set -u
cd /home/kyle/repos/beatsaber_automapper
source .venv/bin/activate
LOG=logs/overnight/feel_disc_scale_2026-06-11.log
GENLOG=logs/overnight/v7_cohort_gen_grow_2026-06-11.log
exec >>"$LOG" 2>&1

echo "=== $(date) waiting for cohort gen to finish (grep '[cohort] done=' in $GENLOG) ==="
for i in $(seq 1 240); do            # up to ~4h
  if grep -q "\[cohort\] done=" "$GENLOG" 2>/dev/null; then
    echo "cohort gen complete: $(grep '\[cohort\] done=' "$GENLOG" | tail -1)"
    break
  fi
  sleep 60
done

N=$(ls outputs/v7_cohort_2026-06-10/*.zip 2>/dev/null | wc -l)
echo "=== $(date) V7 cohort size = $N ; training feel-discriminator (scale + ablations) ==="

for ABL in none dt spatial dir; do
  echo "----- arm: ablate=$ABL -----"
  python scripts/feel_disc_poc.py --epochs 60 --ablate "$ABL" \
    --json "outputs/feel_disc_${ABL}_2026-06-11.json"
done

echo "=== $(date) DONE. Summary: ==="
python - <<'PY'
import json, glob, pathlib
rows=[]
for f in sorted(glob.glob("outputs/feel_disc_*_2026-06-11.json")):
    d=json.load(open(f))
    arm=pathlib.Path(f).stem.replace("feel_disc_","").replace("_2026-06-11","")
    rows.append((arm, d["best_val_auc"], d["n_v7"], d["n_pos"], d["val"]))
print(f"{'arm':10s} {'val_auc':>8s} {'n_v7':>6s} {'n_pos':>6s} {'n_val':>6s}")
for a,au,nv,npz,vv in rows:
    print(f"{a:10s} {au:8.4f} {nv:6d} {npz:6d} {vv:6d}")
none=[au for a,au,*_ in rows if a=='none']
if none:
    v = ("DoD MET: learned reward separates human vs V7 at scale." if none[0]>=0.75
         else "DoD MISS at scale: reconsider (add MERT audio conditioning).")
    print("\nVERDICT(none arm):", v)
PY
