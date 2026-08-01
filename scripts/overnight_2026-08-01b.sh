#!/usr/bin/env bash
# STRESS-TEST THE PASSES. Rewritten 2026-08-01 (it originally pushed b1_e12, which
# the ds055 results superseded before it ever ran).
#
# Three arms now pass all 5 axes + parity:
#   hl014_ds055    flow 0.22 rhythm 0.19 idiom 0.44 handrole 1.04 playfeel 0.76
#   b1_e17_ds055   flow 0.38 rhythm 0.58 idiom 0.53 handrole 1.22 playfeel 0.85
#   b1_e15_ds055   flow 0.25 rhythm 0.44 idiom 0.36 handrole 1.82 playfeel 0.98
# hl014_ds055 beats b1_e17_ds055 on ALL FIVE and needs no retrained model -- it is
# version_4 (production) plus the difficulty scale plus BEAT_HAND_LEAD.
#
# THE REASON FOR THIS SWEEP IS DOUBT, NOT CONFIRMATION. hl014 passes while its
# neighbours hl010 and hl018 both FAIL. The mechanism explains that tidily --
# realised role_asymmetry is linear in the setting (0.0917 / 0.1197 / 0.1538 for
# 0.10 / 0.14 / 0.18) and hl014 is the arm that lands on the human 0.115 -- but a
# tidy explanation is not evidence, and a lever tuned until it clears five bars is
# exactly how the h_dist metric saturated (docs/eval_suite_v2.md). So:
#
# ARMS (6):
#   hl012_ds055, hl016_ds055  -- fill in the optimum. A PLATEAU across 0.12-0.16 is
#                                a real basin; if only 0.14 passes it is a knife
#                                edge fitted to the bars and must not be promoted.
#   hl014_seed1_ds055         -- same target, different lead/swap arrangement. A
#                                real effect survives re-seeding.
#   hl014_ds05                -- lower density on the winner; playfeel 0.76 has
#                                room and fewer notes is the other route toward the
#                                human double share (0.785 now vs 0.231).
#   b1_e17_ds055_hl014        -- do the two independent mechanisms COMPOSE? e17
#                                reaches asym 0.0706 by representation, hl014
#                                reaches 0.1197 by budget. Together they may
#                                overshoot the human 0.115 badly -- that is the
#                                point of measuring rather than assuming.
#   b1_e17_ds05               -- lower density on the other passing candidate.
# CONTROLS: ds055, hl014_ds055, b1_e17_ds055 already cached.
#
# DoD: hl014's pass survives BOTH re-seeding AND at least one neighbouring setting.
# If it does not, the honest report is "one arm passed and did not replicate",
# and the lever stays default-OFF.

set -uo pipefail
cd /home/kyle/repos/beatsaber_automapper
source .venv/bin/activate

LOG=logs/overnight/stress_2026-08-01.log
mkdir -p logs/overnight
exec > >(tee -a "$LOG") 2>&1

echo "=== PASS STRESS-TEST START $(date -Is) ==="

ARMS="hl012_ds055,hl016_ds055,hl014_seed1_ds055,hl014_ds05,b1_e17_ds055_hl014,b1_e17_ds05"

python scripts/eval_sweep.py sweep --arms "$ARMS"
echo "=== SWEEP DONE $(date -Is) ==="

echo "=== 5-AXIS SCORECARD ==="
for arm in ds055 hl014_ds055 b1_e17_ds055 $(echo "$ARMS" | tr ',' ' '); do
  zips=(outputs/eval_sweep_cache/${arm}__*.zip)
  if [ ! -e "${zips[0]}" ]; then echo "-- $arm: NO CACHED MAPS, skipping"; continue; fi
  echo "-- $arm (${#zips[@]} maps)"
  python -m beatsaber_automapper.evaluation.scorecard "${zips[@]}" --label "$arm" || true
done

echo "=== SUB-METRIC BREAKDOWN ==="
python scripts/eval_spread_breakdown.py \
  --arms "ds055,hl014_ds055,b1_e17_ds055,$ARMS" --human 24 || true

echo "=== CONTROL VARIABLE (doubles / asymmetry / swap / notes) ==="
python - <<'PY'
import pathlib, statistics, sys
REPO = pathlib.Path(".").resolve(); sys.path.insert(0, str(REPO / "src"))
from beatsaber_automapper.evaluation import scorecard, handrole
CACHE = REPO / "outputs" / "eval_sweep_cache"
arms = ["ds055", "hl010_ds055", "hl012_ds055", "hl014_ds055", "hl014_seed1_ds055",
        "hl016_ds055", "hl018_ds055", "hl014_ds05", "b1_e17_ds055",
        "b1_e17_ds055_hl014", "b1_e17_ds05"]
print(f"{'arm':24s}{'dbl':>8s}{'asym':>9s}{'swap':>8s}{'notes':>8s}")
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
        print(f"{a:24s}  (no cached maps)"); continue
    print(f"{a:24s}{statistics.median(db):8.3f}{statistics.median(asy):9.4f}"
          f"{statistics.median(sw):8.3f}{statistics.median(nn):8.0f}")
print(f"{'human':24s}{0.231:8.3f}{0.115:9.4f}{0.461:8.3f}")
PY

echo "=== VERDICT $(date -Is) ==="
python - <<'PY'
print("""
 (a) hl012 AND/OR hl016 pass, and hl014_seed1 passes -> the optimum is a real basin.
     hl014 becomes a genuine promotion candidate, still pending Kyle's ears.
 (b) only hl014 passes and hl014_seed1 FAILS -> the pass did not replicate under
     re-seeding. Report it as such, keep the lever default-OFF, and do not tune
     further toward the bars -- that is the h_dist saturation failure repeating.
 (c) hl014_seed1 passes but the neighbours fail -> real but narrow. Usable, but the
     setting is load-bearing and must be pinned with a test, not left to a default.
 (d) b1_e17_ds055_hl014 overshoots (role_asymmetry >> 0.115) and drops axes
     -> the two mechanisms do NOT compose; pick one. Expected if e17's 0.0706 and
        hl014's 0.1197 simply add.
 (e) anything reaches double share < 0.6 while still passing -> the biggest
     remaining structural gap (0.785 vs human 0.231) is finally moving; that is
     worth more than another axis-level tweak.
""")
PY
echo "=== COMPLETE $(date -Is) ==="
