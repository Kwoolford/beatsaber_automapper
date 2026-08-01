#!/usr/bin/env bash
# Push the e12 instrument candidate over the line.
#
# `b1_e12_ds055` is the first arm ever to PASS handrole at judgeable density (gap
# 1.72, spread 0.44) -- the axis that has been our worst since 2026-07-27, called
# "worse than random noise", and Kyle's stated priority. Its three remaining
# failures are all knife-edge:
#     flow     0.50  (bar 0.50 -- exactly on it)
#     idiom    spread 0.33 (bar 0.35, noise floor +-0.09)
#     playfeel 1.03  (bar 1.00)
# All three are density-sensitive, and e12 already emits fewer notes than the
# version_4 control (656 vs 800), so a smaller difficulty scale is the cheapest
# shot at converting all three without disturbing the handrole win.
#
# ARMS (4):
#   b1_e12_ds05, b1_e12_ds045  -- lower density on the winning checkpoint
#   b1_e12_ds055_hl014         -- + hand lead. e12 ALREADY passes handrole, so the
#                                 thing to watch is whether the extra asymmetry
#                                 buys back idiom SPREAD while handrole holds --
#                                 and whether role_asymmetry overshoots the human
#                                 0.115 (e12 is at 0.0631, the lever adds ~0.82x
#                                 of its setting).
#   b1_e15_ds055               -- guards against reading e12 as special when the
#                                 epoch curve is noisy; e15 was WORSE at prod
#                                 density, so it should be worse here too.
# CONTROLS: ds055 and b1_e12_ds055 are already cached.
#
# DoD: an arm that holds handrole (gap <= 2.00 AND spread >= 0.35) while clearing
# flow <= 0.50, idiom spread >= 0.35 and playfeel <= 1.00, at 0 parity violations.
# That would be the project's first 5/5 -> render it and send it to Kyle. It is
# NOT promoted to generate.py defaults on the scorecard alone; the 2026-07-27
# review found a lever that scored well on paper and was unplayable in practice.

set -uo pipefail
cd /home/kyle/repos/beatsaber_automapper
source .venv/bin/activate

LOG=logs/overnight/e12push_2026-08-01.log
mkdir -p logs/overnight
exec > >(tee -a "$LOG") 2>&1

echo "=== E12 PUSH START $(date -Is) ==="

ARMS="b1_e12_ds05,b1_e12_ds045,b1_e12_ds055_hl014,b1_e15_ds055"

python scripts/eval_sweep.py sweep --arms "$ARMS"
echo "=== SWEEP DONE $(date -Is) ==="

echo "=== 5-AXIS SCORECARD ==="
for arm in ds055 b1_e12_ds055 $(echo "$ARMS" | tr ',' ' '); do
  zips=(outputs/eval_sweep_cache/${arm}__*.zip)
  if [ ! -e "${zips[0]}" ]; then
    echo "-- $arm: NO CACHED MAPS, skipping"; continue
  fi
  echo "-- $arm (${#zips[@]} maps)"
  python -m beatsaber_automapper.evaluation.scorecard "${zips[@]}" --label "$arm" || true
done

echo "=== SUB-METRIC BREAKDOWN ==="
python scripts/eval_spread_breakdown.py \
  --arms "ds055,b1_e12_ds055,$ARMS" --human 24 || true

echo "=== DOUBLE SHARE / ASYMMETRY / SWAP (the control variable) ==="
python - <<'PY'
import pathlib, statistics, sys
REPO = pathlib.Path(".").resolve(); sys.path.insert(0, str(REPO / "src"))
from beatsaber_automapper.evaluation import scorecard, handrole
CACHE = REPO / "outputs" / "eval_sweep_cache"
arms = ["ds055", "b1_e12_ds055", "b1_e12_ds05", "b1_e12_ds045",
        "b1_e12_ds055_hl014", "b1_e15_ds055"]
print(f"{'arm':22s}{'dbl':>8s}{'asym':>9s}{'swap':>8s}{'notes':>8s}")
for a in arms:
    db, asy, sw, nn = [], [], [], []
    for p in sorted(CACHE.glob(f"{a}__*.zip")):
        try:
            r = scorecard._load_any(p)
        except Exception:
            r = None
        if not r:
            continue
        bm, _ = r
        byb = {}
        for x in bm.color_notes:
            byb.setdefault(round(x.beat * 16), set()).add(x.color)
        db.append(sum(2 for _b, c in byb.items() if len(c) > 1) / max(len(bm.color_notes), 1))
        m = handrole.handrole_metrics(bm).metrics
        if m.get("role_asymmetry") == m.get("role_asymmetry"):
            asy.append(m["role_asymmetry"]); sw.append(m["role_swap_rate"])
        nn.append(len(bm.color_notes))
    if not asy:
        print(f"{a:22s}  (no cached maps)"); continue
    print(f"{a:22s}{statistics.median(db):8.3f}{statistics.median(asy):9.4f}"
          f"{statistics.median(sw):8.3f}{statistics.median(nn):8.0f}")
print(f"{'human':22s}{0.231:8.3f}{0.115:9.4f}{0.461:8.3f}")
PY

echo "=== VERDICT $(date -Is) ==="
python - <<'PY'
print("""
BASELINES: ds055 0.30/0.36/0.52/1.92*/0.74 (4/5, * = handrole spread 0.27 FAILS)
           b1_e12_ds055 0.50/0.48/0.69*/1.72/1.03 (handrole PASSES, spread 0.44)
NOISE FLOOR: flow +-0.03 rhythm +-0.08 idiom +-0.09 handrole +-0.29.

 (a) an arm holds handrole (gap <=2.00 AND spread >=0.35) and clears flow, idiom
     spread and playfeel -> FIRST 5/5. Render it, SendUserFile it to Kyle, log it
     as awaiting ears, and do NOT touch generate.py defaults.
 (b) lower density fixes flow/playfeel but handrole falls back out
     -> density and hand-role are coupled through the double share; the fix is the
        hand-lead lever at ds05, not more density.
 (c) idiom spread stays < 0.35 everywhere
     -> idiom_jsd is the key that collapses (per eval_spread_breakdown.py). No
        density or hand lever has ever moved it. That becomes its own task, and it
        is the LAST axis blocking a 5/5 -- do not keep sweeping density at it.
 (d) b1_e15_ds055 matches or beats b1_e12_ds055
     -> the epoch curve is noisier than assumed; e12 is not special and the
        selection needs more epochs scored before trusting it.
""")
PY
echo "=== COMPLETE $(date -Is) ==="
