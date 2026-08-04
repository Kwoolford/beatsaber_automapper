#!/usr/bin/env bash
# P0 / Kyle's #1 complaint — BEAT_MAIN_BEAT_BONUS.
# "every couple main beat notes were mapped instead of most of the main beats...
#  it hits the main flow partially."
# Measured: main-beat coverage ours ~0.49 vs human ~0.70. Cause traced to DECODE:
# at the beats we skip, Stage-1's median probability is 0.591 vs 0.408 at a random
# slot -- the model knows, and probability-ranked selection lets a louder non-main
# onset take the window.
#
# ⚠️JUDGE THREE THINGS TOGETHER, NOT COVERAGE ALONE:
#   main_covered   should rise toward the human ~0.70
#   notes_on_main  MUST NOT run away -- the HUMAN sits at 0.517 and our base is
#                  already 0.648. Humans cover most of the main beat AND play
#                  plenty off it. Becoming main-beat-dominated is the metronome
#                  failure this project has hit twice (halfbeat_rate and
#                  share_over_1s both fell to it).
#   rhythm         the axis that killed BEAT_HAND_DEAL after every structural
#                  metric looked perfect. It is the real gate.
set -u
cd "$(dirname "$0")/.."
L=logs/overnight/mainbeat_2026-08-04.log
mkdir -p logs/overnight
: > "$L"
echo "=== START $(date -Is) ===" >> "$L"
.venv/bin/python scripts/eval_sweep.py sweep \
    --arms tf_trim_ev03_rc05,mbb015,mbb025,mbb050 --seeds 3 >> "$L" 2>&1

echo "" >> "$L"; echo "##### MAIN-BEAT COVERAGE vs the human #####" >> "$L"
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
print(f"{'arm':22s}{'main_covered':>14s}{'continuity':>12s}{'notes_on_main':>15s}")
for arm in ("tf_trim_ev03_rc05","mbb015","mbb025","mbb050"):
    C=[];T=[];O=[]
    for p in sorted(glob.glob(f"outputs/eval_sweep_cache/{arm}#s0__*.zip")):
        sid=scorecard.song_id(pathlib.Path(p))
        L=scorecard._load_any(pathlib.Path(p))
        if not L: continue
        t=np.sort(np.array(alignment.note_times(L[0],L[1])))
        mb=mb_for(sid,float(L[1]),float(t.max()))
        if mb is None: continue
        c=coverage(t,mb)
        if c: C.append(c['main_covered']); T.append(c['main_continuity']); O.append(c['notes_on_main'])
    if C: print(f"{arm:22s}{st.median(C):14.3f}{st.median(T):12.3f}{st.median(O):15.3f}")
C=[];T=[];O=[]
cached={p.stem for p in pathlib.Path('outputs/stem_onset_cache').glob('*.npz')}
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
print(f"{'HUMAN':22s}{st.median(C):14.3f}{st.median(T):12.3f}{st.median(O):15.3f}")
print("\nnotes_on_main running past the human is the METRONOME warning, not a win.")
PY
echo "=== COMPLETE $(date -Is) ===" >> "$L"
