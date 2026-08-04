#!/usr/bin/env bash
# S7 — pair the model that has the right PHASE with the prior that exploits it.
# version_4 inverts in our worst windows (on/off ratio 0.55); version_8 does not
# (1.74). Kyle's #1 complaint traces to that inversion, and BEAT_MAIN_BEAT_BONUS
# is capped at a third of the gap because a x1.25 prior cannot beat a 2x deficit.
# ⚠️v8's maps scored WORSE on the six axes in the B-1 sweep, so this is an
# experiment. Judge main-beat coverage AND the axes; a coverage win bought with an
# axis collapse is the BEAT_HAND_DEAL failure again.
set -u
cd "$(dirname "$0")/.."
L=logs/overnight/v8density_2026-08-04.log
mkdir -p logs/overnight
: > "$L"
echo "=== START $(date -Is) ===" >> "$L"
.venv/bin/python scripts/eval_sweep.py sweep \
    --arms tf_trim_ev03_rc05,mbb025,v8_nb120,v8_nb120_mbb025 --seeds 3 >> "$L" 2>&1
echo "" >> "$L"; echo "##### MAIN-BEAT COVERAGE #####" >> "$L"
.venv/bin/python - >> "$L" 2>&1 <<'PY'
import sys, pathlib, glob, statistics as st
import numpy as np
sys.path.insert(0,'src'); sys.path.insert(0,'scripts')
from beatsaber_automapper.evaluation import alignment, scorecard
from calibrate_playfeel import load_expert_only
from main_beat import find_main_beat, coverage
mbs={}
def mb_for(sid,bpm,end):
    if sid not in mbs: mbs[sid]=find_main_beat(sid,bpm,end)
    return mbs[sid]
print(f"{'arm':22s}{'covered':>10s}{'continuity':>12s}{'on_main':>10s}")
for arm in ("tf_trim_ev03_rc05","mbb025","v8_nb120","v8_nb120_mbb025"):
    C=[];T=[];O=[]
    for p in sorted(glob.glob(f"outputs/eval_sweep_cache/{arm}#s0__*.zip")):
        sid=scorecard.song_id(pathlib.Path(p)); L=scorecard._load_any(pathlib.Path(p))
        if not L: continue
        t=np.sort(np.array(alignment.note_times(L[0],L[1])))
        mb=mb_for(sid,float(L[1]),float(t.max()))
        if mb is None: continue
        c=coverage(t,mb)
        if c: C.append(c['main_covered']); T.append(c['main_continuity']); O.append(c['notes_on_main'])
    if C: print(f"{arm:22s}{st.median(C):10.3f}{st.median(T):12.3f}{st.median(O):10.3f}")
C=[];T=[];O=[]
for p in sorted(glob.glob('outputs/eval_sweep_cache/tf_trim_ev03_rc05#s0__*.zip')):
    sid=scorecard.song_id(pathlib.Path(p)); hz=pathlib.Path(f'data/raw/{sid}.zip')
    if not hz.exists(): continue
    H=load_expert_only(hz)
    if not H: continue
    t=np.sort(np.array(alignment.note_times(H[0],float(H[1]))))
    mb=mb_for(sid,float(H[1]),float(t.max()))
    if mb is None: continue
    c=coverage(t,mb)
    if c: C.append(c['main_covered']); T.append(c['main_continuity']); O.append(c['notes_on_main'])
print(f"{'HUMAN':22s}{st.median(C):10.3f}{st.median(T):12.3f}{st.median(O):10.3f}")
PY
echo "=== COMPLETE $(date -Is) ===" >> "$L"
