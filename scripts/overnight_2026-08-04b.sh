#!/usr/bin/env bash
# C5 / W3 — BEAT_HAND_DEAL. The project's known root cause, attacked directly.
#
# Stage-1's two hand channels correlate 0.985-0.993, so per-hand selection makes
# both hands pick the SAME slots. Hunger: 809 distinct note times vs the human's
# 1245; double share 0.6415 vs 0.1478. C5 says the fix must RAISE the distinct
# slot count, not redistribute -- which is why BEAT_HAND_INTERLEAVE (sent a hand
# to worse slots) and BEAT_HAND_ROLE (leaves times untouched) both failed.
#
# Single-song smoke test: distinct 809 -> 1193, doubles 0.6415 -> 0.1408, notes
# 1328 -> 1361. But:
#   ⚠️the double share landing on the human value is BY CONSTRUCTION -- the
#     parameter IS the target. It is not evidence. The UNTUNED number is distinct
#     times, and that is what to judge.
#   ⚠️1f333 is a documented single-song probe trap. Hence 24 songs x 3 seeds.
#   ⚠️role_asymmetry MUST be checked: BEAT_HAND_LEAD is a confirmed positive by
#     Kyle's ear and the deal takes over the L/R split it used to own.
#
# This is the biggest structural change since the tempo fit. Default OFF.
set -u
cd "$(dirname "$0")/.."
L=logs/overnight/handdeal_2026-08-04.log
mkdir -p logs/overnight
: > "$L"
echo "=== START $(date -Is) ===" >> "$L"
.venv/bin/python scripts/eval_sweep.py sweep \
    --arms tf_trim_ev03_rc05,deal10,deal14,deal20 --seeds 3 >> "$L" 2>&1

echo "" >> "$L"; echo "##### C5 STRUCTURE: distinct times + double share vs human #####" >> "$L"
.venv/bin/python - >> "$L" 2>&1 <<'PY'
import pathlib, sys, glob, collections, statistics as st
sys.path.insert(0,'src'); sys.path.insert(0,'scripts')
from beatsaber_automapper.evaluation import scorecard
from calibrate_playfeel import load_expert_only
def stat(bm,bpm):
    spb=60.0/bpm; ev=collections.defaultdict(set)
    for n in bm.color_notes: ev[round(n.beat*spb,4)].add(n.color)
    return len(ev), sum(1 for t in ev if len(ev[t])>=2)/len(ev)
for arm in ("tf_trim_ev03_rc05","deal10","deal14","deal20"):
    d=[];  s=[]
    for p in sorted(glob.glob(f"outputs/eval_sweep_cache/{arm}#s*__*.zip")):
        L=scorecard._load_any(pathlib.Path(p))
        if not L: continue
        a,b=stat(L[0],L[1]); d.append(a); s.append(b)
    if d: print(f"{arm:22s} distinct(median) {st.median(d):7.0f}   double_share {st.median(s):.4f}")
cached={p.stem for p in pathlib.Path('outputs/stem_onset_cache').glob('*.npz')}
hd=[];hs=[]
for zp in [p for p in sorted(pathlib.Path('data/raw').glob('*.zip')) if p.stem in cached][:150]:
    H=load_expert_only(zp)
    if not H: continue
    a,b=stat(H[0],float(H[1])); hd.append(a); hs.append(b)
print(f"{'HUMAN (n=%d)'%len(hd):22s} distinct(median) {st.median(hd):7.0f}   double_share {st.median(hs):.4f}")
print("\nJUDGE: distinct times is the UNTUNED number. double_share landing on the")
print("human value is by construction and proves nothing.")
PY
echo "=== COMPLETE $(date -Is) ===" >> "$L"
