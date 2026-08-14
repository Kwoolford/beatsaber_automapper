#!/usr/bin/env bash
# THE HALF-TEMPO SPEED CEILING at n=28+28 — does subdiv 8 lift it, and what does it cost?
#
# DEFECT (measured 2026-08-13): on the 28 wide-cohort songs detected at HALF the true
# tempo, our maps sit at EXACTLY 0.500x the human's ebpm_burst (p10 = 0.500, so >=90%
# are exactly at half). Mechanism: our minimum swing gap is one grid slot, and at half
# tempo that slot is 2x as long in real time. No decode lever can reach it.
#
# SCREEN (n=3): subdiv 8 takes the ratio 0.50 -> 1.00 on all three, with density moving
# TOWARD the human (209d2: 783 notes vs the human's 791) and precision barely moving.
#
# ⚠️THIS ARM USES AN ORACLE — the `half` label comes from comparing our bpm to the
# HUMAN's declared bpm, which production cannot see. That is deliberate and it is the
# point: measure the CEILING first, and only invest in octave DETECTION if the ceiling
# is worth having. Same structure as BEAT_BPM_ORACLE. It is NOT a shippable config.
#
# ★THE CONTROL ARM IS NOT OPTIONAL. subdiv 8 is applied to 28 `same`-tempo songs too,
# where the tempo is CORRECT and the lever should therefore do harm. On 20fc6 (a
# half-tempo song with no ceiling) it already pushed us to 2x the human's burst rate.
# Without this arm we would learn that the lever helps where we aimed it and nothing
# about what it does where we did not.
#
# 🔴PRE-REGISTERED READING:
#   WORTH DETECTING  the `half` group's ebpm ratio moves off 0.500 toward 1.000, its
#                    nps moves toward the human, and the `same` control is CLEARLY
#                    WORSE (that asymmetry is what makes octave detection worth
#                    building rather than just raising subdiv globally).
#   RAISE GLOBALLY   both groups improve => the grid was simply too coarse everywhere
#                    and detection is not the problem to solve. Would be a surprise.
#   DEAD             the `half` group does not reach ~1.000 at n=28, i.e. the n=3
#                    screen did not generalise.
# ⚠️Read PRECISION beside it: 2x the slots means going deeper down the probability
# ranking, which is exactly how BEAT_HAND_DEAL died (precision 0.919 -> 0.893).
set -u
cd "$(dirname "$0")/.." || exit 1
LOG=logs/overnight/subdiv_2026-08-14.log
mkdir -p logs/overnight outputs/subdiv_half outputs/subdiv_same
exec > >(tee -a "$LOG") 2>&1
PY=.venv/bin/python
BC="logs/beat_classifier/version_4/checkpoints/beat-epoch=11-val_f1_avg_tol=0.603.ckpt"
LC="logs/layout_phrase/version_10/checkpoints/layout-epoch=09-val_token_acc=0.865.ckpt"
echo "=== SUBDIV 8: half-tempo ceiling + same-tempo control — $(date) ==="

$PY - <<'PY' > /tmp/subdiv_songs.txt
import json
lab=json.load(open('outputs/true_bpm_wide_cohort_labels.json'))
half=[r['song'] for r in lab if r['label']=='half']
same=[r['song'] for r in lab if r['label']=='same'][:28]   # matched n for the control
for s in half: print('half', s)
for s in same: print('same', s)
PY

while read -r grp s; do
  out="outputs/subdiv_${grp}/${s}_sub8.zip"
  [ -f "$out" ] && continue
  [ -f "outputs/wide_cohort/audio/$s.ogg" ] || { echo "  $s: no audio"; continue; }
  BEAT_SUBDIV=8 $PY scripts/generate.py "outputs/wide_cohort/audio/$s.ogg" --v7 \
    --beat-ckpt "$BC" --layout-ckpt "$LC" --difficulty Expert --section-gate loud_only \
    --song-name "$s" --seed 0 --output "$out" > "outputs/subdiv_${grp}/${s}.log" 2>&1 \
    || echo "  $grp $s FAILED"
done < /tmp/subdiv_songs.txt

echo ""; echo "--- EVAL --- $(date +%H:%M)"
$PY - <<'PY'
import json, pathlib, statistics as st, sys
sys.path.insert(0,'src'); sys.path.insert(0,'scripts')
from beatsaber_automapper.evaluation import scorecard, flow
from beatsaber_automapper.evaluation.alignment import alignment_metrics
lab={r['song']:r['label'] for r in json.load(open('outputs/true_bpm_wide_cohort_labels.json'))}
def stats(grp):
    rows=[]
    for zp in sorted(pathlib.Path(f'outputs/subdiv_{grp}').glob('*_sub8.zip')):
        s=zp.stem.replace('_sub8','')
        c=pathlib.Path('outputs/wide_cohort')/f"{s}.zip"; h=pathlib.Path('data/raw')/f"{s}.zip"
        if not(c.exists() and h.exists()): continue
        on=scorecard.onsets_for(c)
        ra,rc,rh=scorecard._load_any(zp),scorecard._load_any(c),scorecard._load_any(h)
        if not(ra and rc and rh) or on is None or len(on)==0: continue
        def E(r): return flow.flow_metrics(r[0],bpm=r[1]).metrics['ebpm_burst']
        def P(r): return alignment_metrics(r[0],bpm=r[1],onsets=on).metrics['onset_precision']
        eh=E(rh)
        if not eh: continue
        rows.append({'song':s,'r4':E(rc)/eh,'r8':E(ra)/eh,'p4':P(rc),'p8':P(ra),
                     'n4':len(rc[0].color_notes),'n8':len(ra[0].color_notes),
                     'nh':len(rh[0].color_notes)})
    return rows
for grp,desc in (('half','TARGET — tempo is an octave error'),
                 ('same','CONTROL — tempo is CORRECT, lever should HARM')):
    rows=stats(grp)
    if not rows: print(f"\n{grp}: no maps"); continue
    print(f"\n=== {grp.upper()} ({desc}) — n={len(rows)} ===")
    print(f"  ebpm ratio vs human:  subdiv4 {st.median([r['r4'] for r in rows]):.3f}"
          f"  ->  subdiv8 {st.median([r['r8'] for r in rows]):.3f}   (1.000 = human)")
    print(f"  onset precision:      subdiv4 {st.median([r['p4'] for r in rows]):.4f}"
          f"  ->  subdiv8 {st.median([r['p8'] for r in rows]):.4f}")
    print(f"  notes / human notes:  subdiv4 {st.median([r['n4']/r['nh'] for r in rows]):.3f}"
          f"  ->  subdiv8 {st.median([r['n8']/r['nh'] for r in rows]):.3f}   (1.000 = human)")
    over=sum(1 for r in rows if r['r8']>1.25)
    print(f"  songs now >1.25x the human's burst rate: {over}/{len(rows)}")
    print(f"  ⚠️per-song MINIMUM ratio (the mean hides subsets): "
          f"{min(r['r8'] for r in rows):.3f}")
print("""
VERDICT LOGIC (pre-registered):
  WORTH DETECTING  half -> ~1.000 and nps toward human, AND the `same` control is
                   clearly worse. That asymmetry is what justifies building octave
                   detection instead of raising subdiv globally.
  RAISE GLOBALLY   both groups improve => the grid was too coarse everywhere.
  DEAD             half does not reach ~1.000 at n=28.""")
PY
echo ""; echo "=== COMPLETE $(date) ==="
