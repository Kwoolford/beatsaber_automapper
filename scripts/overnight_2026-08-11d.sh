#!/usr/bin/env bash
# COLOR_SEP_MODE=extreme — an ALREADY-VALIDATED LEVER THAT WAS NEVER PROMOTED.
#
# HOW IT WAS FOUND (2026-08-11, sensitivity battery). `audit_sensitivity.py` showed the
# suite is BLIND to mirroring a map left-right (max |Δgap| 0.012 across all six axes).
# Chasing that: the per-map metric `crossover` detects it perfectly (0.000 -> 1.000) but
# is wired into NO axis. `flow.py` excludes it from the composite ON PURPOSE — it is
# order-independent and would dilute the shuffled-control — with the comment "still
# reported, as guards". **Nothing ever guarded it.**
#
# WHAT THAT EXPOSED, and it is categorical rather than a gap:
#   human maps  crossover median 0.183, p10 0.111, p90 0.271, and 0 of 150 maps have none
#   our maps    crossover 0.0000 on ALL 149 -- we never cross hands over, ever
# `enforce_color_separation` documents this itself: COLOR_SEP_MODE=full (the default)
# moves every wrong-side note, "which is why our maps measure crossover == 0.000".
#
# 🔴AND IT WAS ALREADY TESTED: PROGRESS.md (2026-07-27 lever sweep) records
# **`COLOR_SEP_MODE=extreme` -> idiom 1.84 -> 0.30 PASS**, ticked as a win. It is NOT in
# the promoted defaults and not in docs/BASELINE_2026-08-03.md. It fell through the
# cracks when the eight defaults were flipped on 2026-08-03.
# ⚠️That result predates the tempo fix, the promotion, and the wide cohort, so it is a
# LEAD, not evidence. This re-tests it against today's baseline at n=149.
#
# 🔴PRE-REGISTERED READING:
#   PROMOTE-WORTHY  crossover moves into the human band (~0.11-0.27) AND idiom does not
#                   regress AND flow/playfeel/alignment do not regress resolvably.
#                   Then it goes to Kyle's ear as a candidate default flip.
#   REJECT          crossover overshoots past the human p90, or flow degrades the way
#                   COLOR_SEP_MODE=off did in 2026-07 (flow 1.04).
# ⚠️A crossover bar must NEVER be set at zero: 0 of 150 human maps have zero crossovers,
# so "no crossovers" is the non-human state, and it is the state we ship today.
set -u
cd "$(dirname "$0")/.." || exit 1
LOG=logs/overnight/xsep_2026-08-11.log
mkdir -p logs/overnight outputs/me_2026-08-11
exec > >(tee -a "$LOG") 2>&1
PY=.venv/bin/python
echo "=== COLOR_SEP_MODE=extreme @ n=149 — $(date) ==="

$PY scripts/build_wide_cohort.py --n 150 --seed 0 --variant prod \
    --tag xsep --env "COLOR_SEP_MODE=extreme"

D=outputs/wide_cohort_prod_xsep
N=$(ls "$D"/*.zip 2>/dev/null | wc -l)
echo ""; echo "--- EVAL xsep ($N maps) --- $(date +%H:%M)"
[ "$N" -lt 100 ] && { echo "SKIP: only $N maps"; exit 1; }

echo "  -- six-axis (control: flow 0.37 rhythm 0.47 idiom 0.40 handrole 1.12 playfeel 0.59 alignment 0.62) --"
$PY -m beatsaber_automapper.evaluation.scorecard "$D"/*.zip --label xsep
echo "  -- reachability (control hard_rate 0.0494) --"
$PY scripts/eval_reachability.py --maps "$D/*.zip" --label xsep \
    --maps outputs/wide_cohort/"*.zip" --label control \
    --json outputs/me_2026-08-11/reach_xsep.json
echo "  -- crossover vs the human band --"
$PY - <<'PY2'
import sys, glob, pathlib, numpy as np
sys.path.insert(0,'src')
from beatsaber_automapper.evaluation import scorecard
def cross(bm):
    n=bm.color_notes
    if not n: return None
    w=sum(1 for x in n if (x.color==0 and x.x>=2) or (x.color==1 and x.x<=1))
    return w/len(n)
for d in ("wide_cohort","wide_cohort_prod_xsep"):
    v=[]
    for f in sorted(glob.glob(f"outputs/{d}/*.zip")):
        L=scorecard._load_any(pathlib.Path(f))
        if L:
            c=cross(L[0])
            if c is not None: v.append(c)
    v=np.array(v)
    print(f"  {d:28s} median {np.median(v):.4f}  p90 {np.percentile(v,90):.4f}  zero-maps {int((v==0).sum())}/{len(v)}")
print("  HUMAN (n=150)                median 0.1826  p10 0.1112  p90 0.2707  zero-maps 0/150")
PY2
echo "COMPLETE $(date)"
