#!/usr/bin/env bash
# THE CANDIDATE STACK — do the two independent wins interact?
#
# Two changes are in front of Kyle, found by different routes and validated separately:
#   COLOR_SEP_MODE=extreme               crossover 0.000 -> 0.112 (human 0.183), flow
#                                        0.37 -> 0.23, reach lands on human values
#   BEAT_STRUCTURE_REUSE=diag_full:...:0.20   ~45-51% of the structural gap, replicated
#                                        at 2 seeds, dose-capped to keep variety human
#
# Neither was tested WITH the other, and that is the config he would actually get if he
# likes both. They plausibly interact: xsep moves notes horizontally *after* the layout
# model chose them, and structure-reuse copies whole bars including their columns — so a
# copied bar may be re-separated differently from its source, weakening the copy.
#
# 🔴PRE-REGISTERED READING:
#   SHIP BOTH   the stack keeps xsep's flow gain (~0.23) AND structure-reuse's rhy_rhythm
#               / harm_rhythm gains (~+0.019 / +0.022 at the capped dose), with crossover
#               still inside the human band and no axis crossing a bar.
#   INTERACT    either gain is materially smaller than its solo run ⇒ say WHICH, and ship
#               only the one that survives; do not ship a stack whose parts fight.
# ⚠️Compare each against its own solo arm, not against the control — the question is
#   interaction, not effect.
set -u
cd "$(dirname "$0")/.." || exit 1
LOG=logs/overnight/stack_2026-08-11.log
mkdir -p logs/overnight outputs/me_2026-08-11
exec > >(tee -a "$LOG") 2>&1
PY=.venv/bin/python
echo "=== candidate stack: xsep + structure-reuse(capped) — $(date) ==="

$PY scripts/build_wide_cohort.py --n 150 --seed 0 --variant prod --tag stack \
    --env "COLOR_SEP_MODE=extreme" \
    --env "BEAT_STRUCTURE_REUSE=diag_full:0.70:4:1.5:2.0:4:0.20"

D=outputs/wide_cohort_prod_stack
N=$(ls "$D"/*.zip 2>/dev/null | wc -l)
echo ""; echo "--- EVAL stack ($N maps) --- $(date +%H:%M)"
[ "$N" -lt 100 ] && { echo "SKIP: only $N maps"; exit 1; }
echo "  control     flow 0.37 idiom 0.40 playfeel 0.59 | crossover 0.000 | rhy_rhythm +0.0536"
echo "  xsep solo   flow 0.23 idiom 0.52 playfeel 0.62 | crossover 0.112 | M-axes unchanged"
echo "  capped solo flow 0.55 idiom 1.21 playfeel 0.76 | crossover 0.000 | rhy_rhythm +0.0190 Δ"
$PY -m beatsaber_automapper.evaluation.scorecard "$D"/*.zip --label stack
$PY scripts/masterpiece_report.py --arm stack --wide --wide-dir "$D" \
    --vs prod --vs-wide-dir outputs/wide_cohort \
    --json outputs/me_2026-08-11/masterpiece_stack.json
echo "  -- crossover + bar-pattern diversity --"
$PY - <<'PY2'
import sys, glob, pathlib, numpy as np
sys.path.insert(0,'src'); sys.path.insert(0,'scripts')
import song_structure as ss, eval_motif_rhyme as m1
from beatsaber_automapper.evaluation import scorecard
def cross(bm):
    n=bm.color_notes
    w=sum(1 for x in n if (x.color==0 and x.x>=2) or (x.color==1 and x.x<=1))
    return w/len(n)
def div(bm,bpm,sid):
    B=ss.bars(sid,bpm,ss.song_end(sid))
    if B is None: return np.nan
    V=ss.map_bar_vectors(m1.notes_xydc(bm,bpm),B)
    pats=set(); n=0
    for i in range(B.n):
        if V["count"][i]<3: continue
        n+=1; pats.add(tuple(np.round(np.concatenate([V["rhythm"][i],V["place"][i].reshape(-1)]),3).tolist()))
    return len(pats)/n if n>=10 else np.nan
for d in ("wide_cohort","wide_cohort_prod_stack"):
    cs=[];dv=[]
    for f in sorted(glob.glob(f"outputs/{d}/*.zip")):
        L=scorecard._load_any(pathlib.Path(f))
        if not L: continue
        cs.append(cross(L[0])); dv.append(div(L[0],float(L[1]),pathlib.Path(f).stem))
    dv=np.array([x for x in dv if x==x])
    print(f"  {d:26s} crossover med {np.median(cs):.4f} | diversity mean {dv.mean():.3f} min {dv.min():.3f}")
print("  human: crossover 0.1826 (p10 0.1112 p90 0.2707) | diversity ~0.877")
PY2
echo "COMPLETE $(date)"
