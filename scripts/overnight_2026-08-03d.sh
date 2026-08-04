#!/usr/bin/env bash
# W2 + W7 — the two objections that ARE fixable in the decode today.
#
# W2 (Fallen Kingdom "really empty"): diagnosed 2026-08-03 as an ALLOCATION defect,
# not a representation one. Stage-1 scores the human's note slots at 0.797 vs
# 0.0032 for slots generally, and scores the 48 human notes we skip at 0.734 --
# the model points and the decode declines. Across 13 songs we emit 0.582 of the
# slots above prob 0.5 while humans take 0.854, and OUR fraction is nearly constant
# (spread 0.115) while the human's varies 4x more (0.435).
#
# W7: the map ends on an orphaned half-double 0.159 of the time vs the human 0.036.
#
# ARMS (all vs the promoted baseline as CONTROL, 3 seeds each):
#   tf_trim_ev03_rc05   control = exactly what Kyle graded A+
#   nb115 / nb130 / nb145   BEAT_NOTE_BUDGET, the user-facing "how many notes" dial
#   endres                  BEAT_END_RESOLVE=0.75
#
# ⚠️ READ THE BUDGET ARMS AS A PRICE, NOT A WIN. BEAT_DIFFICULTY_SCALE is 0.48
# precisely because Kyle called 6.18 nps "Expert+, not Expert", and W3 says parts
# of Hunger are ALREADY too intense. The question is what the axes charge per
# note, and whether there is a setting that closes the Fallen Kingdom gap without
# making Hunger worse -- a GLOBAL budget may well not exist, which is itself the
# result that justifies building the per-song version.
set -u
cd "$(dirname "$0")/.."
L=logs/overnight/budget_endres_2026-08-03.log
mkdir -p logs/overnight
: > "$L"
echo "=== START $(date -Is) ===" >> "$L"

.venv/bin/python scripts/eval_sweep.py \
    sweep --arms tf_trim_ev03_rc05,nb115,nb130,nb145,endres \
    --seeds 3 >> "$L" 2>&1

echo "" >> "$L"; echo "##### W7 ORPHANED-ENDING CHECK #####" >> "$L"
.venv/bin/python - >> "$L" 2>&1 <<'PY'
import pathlib, sys, glob, collections, statistics as st
sys.path.insert(0,'src')
from beatsaber_automapper.evaluation import scorecard
def orphan(bm,bpm):
    spb=60.0/bpm; ev=collections.defaultdict(set)
    for n in bm.color_notes: ev[round(n.beat*spb,4)].add(n.color)
    ts=sorted(ev)
    if len(ts)<6: return None
    fin=len(ev[ts[-1]])>=2
    prev=sum(1 for t in ts[-5:-1] if len(ev[t])>=2)/4.0
    return (not fin) and prev>=0.75
for arm in ("tf_trim_ev03_rc05","endres"):
    v=[]
    for p in sorted(glob.glob(f"outputs/eval_sweep_cache/{arm}#s*__*.zip")):
        L=scorecard._load_any(pathlib.Path(p))
        if not L: continue
        r=orphan(L[0],L[1])
        if r is not None: v.append(r)
    if v: print(f"{arm:22s} orphaned ending {sum(v)}/{len(v)} = {sum(v)/len(v):.4f}")
print("human = 0.036 (n=249). endres should land near it; the control near 0.159.")
PY

echo "" >> "$L"; echo "##### VERDICT #####" >> "$L"
.venv/bin/python - >> "$L" 2>&1 <<'PY'
print("""
HOW TO READ THIS NEXT SESSION

W7 / endres: DoD is orphaned-ending falling from ~0.159 toward the human 0.036
  while all six axes stay inside their noise floors (flow 0.20, rhythm 0.17,
  idiom 0.17, handrole 0.61, playfeel 0.10 at 2sd). It removes at most one note
  per map, so ANY resolvable axis move is suspicious and should be investigated
  before the lever is believed.

W2 / nb115-145: this is a PRICE CURVE, not a hunt for a winner.
  * playfeel is expected to WORSEN with budget -- nps is its sub-metric and 0.48
    was calibrated to Kyle's "Expert+, not Expert" complaint. A regression there
    is the cost being quoted, not a failure.
  * The informative question is whether ANY arm closes the Fallen Kingdom gap
    (ours 0.582 used/supply vs human 0.854) WITHOUT pushing Hunger's peak nps
    further past the human 5.5 -- check the two songs separately, never the
    cohort median, because a global dial that helps one and hurts the other is
    exactly the finding that justifies the PER-SONG budget.
  * If no global setting satisfies both, that is a RESULT: report it and build
    the per-song version (W2 task 2) instead of shipping a global bump.
  ⚠️ Nothing here is promotable without Kyle's ear -- he set the 0.48 by playing.
""")
PY
echo "=== COMPLETE $(date -Is) ===" >> "$L"
