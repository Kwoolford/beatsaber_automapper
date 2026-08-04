#!/usr/bin/env bash
# W7 round 2 — end the map where a human ends it.
# Kyle on the endres map: "The hunger one was noticibly better, but still a very
# small delay." Cause: humans end on a MULTI-INSTRUMENT hit (their last note sits
# on 3-4 stems); we end on a lone straggler. BEAT_TRIM_END_COINCIDENCE cuts at the
# last k>=3 stem coincidence instead of the last onset of any stem.
# Verified single-song: Hunger 272.07 -> 271.755 vs a human 271.755, exact.
# Guarded: Fallen Kingdom's soft vocal outro (28 human-mapped notes past the last
# coincidence) is detected and NOT cut.
set -u
cd "$(dirname "$0")/.."
L=logs/overnight/trimco_2026-08-04.log
mkdir -p logs/overnight
: > "$L"
echo "=== START $(date -Is) ===" >> "$L"
.venv/bin/python scripts/eval_sweep.py sweep \
    --arms tf_trim_ev03_rc05,endres,trimco3,trimco3_endres --seeds 3 >> "$L" 2>&1

echo "" >> "$L"; echo "##### ENDING PLACEMENT vs the human #####" >> "$L"
.venv/bin/python - >> "$L" 2>&1 <<'PY'
import sys, pathlib, glob, statistics as st
import numpy as np
sys.path.insert(0,'src'); sys.path.insert(0,'scripts')
from beatsaber_automapper.evaluation import alignment, scorecard
from calibrate_playfeel import load_expert_only
def carrier_end(sid):
    f=pathlib.Path(f'outputs/stem_onset_cache/{sid}.npz')
    if not f.exists(): return None
    d=np.load(f,allow_pickle=True)
    st_={s:np.sort(d[f'onsets_{s}']) for s in ('drums','bass') if f'onsets_{s}' in d.files}
    if not st_: return None
    return float(st_[max(st_,key=lambda s: len(st_[s]))].max())
for arm in ("tf_trim_ev03_rc05","endres","trimco3","trimco3_endres"):
    rel=[]
    for p in sorted(glob.glob(f"outputs/eval_sweep_cache/{arm}#s0__*.zip")):
        sid=scorecard.song_id(pathlib.Path(p)); ce=carrier_end(sid)
        hz=pathlib.Path(f'data/raw/{sid}.zip')
        if ce is None or not hz.exists(): continue
        L=scorecard._load_any(pathlib.Path(p)); H=load_expert_only(hz)
        if not L or not H: continue
        o=float(np.max(alignment.note_times(L[0],L[1])))-ce
        h=float(np.max(alignment.note_times(H[0],float(H[1]))))-ce
        rel.append(abs(o-h))
    if rel: print(f"{arm:18s} |ours-human| ending offset: median {st.median(rel):.3f}s  n={len(rel)}")
print("\nLower = the map ends closer to where the human ends it. Baseline is the")
print("number to beat; the human is 0.000 by definition.")
PY
echo "=== COMPLETE $(date -Is) ===" >> "$L"
