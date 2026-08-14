#!/usr/bin/env bash
# W4 — PHRASES ABANDONED MID-VOCAL. Does the density weighting cause it?
#
# THE DEFECT, reproduced at n=123 (originally measured on 13 songs):
#   share of sung phrases with a >1 s hole   ours 0.500  human 0.182   (2.75x)
#   share with a >2 s hole                   ours 0.071  human 0.000
#   paired: we abandon MORE on 109/123 songs, LESS on 6.
# ★It did NOT shrink at scale — unusually for this project, n=13 slightly
# UNDERSTATED it. This is the most robust defect currently on the list.
#
# THE HYPOTHESIS, and it is written in our own docstring:
#   `_density_aware_select` sets weight = (window-mean prob)^gamma with gamma 2.5,
#   "so loud/dense windows keep more notes and QUIET ONES THIN OUT". A sung phrase
#   over sparse backing IS a quiet window. `BEAT_ONSET_EVIDENCE` (0.3) multiplies a
#   second concentration factor on top, and is already known to concentrate notes
#   into dense windows (it degraded reachability that way on 2026-08-03).
#
# 🔴PRE-REGISTERED:
#   LEVER FOUND  share_over_1s falls materially toward the human's 0.182 as gamma
#                drops, AND the alignment/rhythm cost is bounded. Then W4 has its
#                first lever and the dose becomes the question.
#   CONFOUNDED   it falls but alignment/rhythm degrade in step => this is the same
#                "gain and damage are one dial" result as every other density knob,
#                and the honest report is the trade curve, not a win.
#   REFUTED      share_over_1s does not move => the density weighting is NOT what
#                abandons the phrase, and the cause is upstream in Stage-1 (which
#                would make W4 a Track B item, like W1).
# ⚠️Read the PAIRED per-song delta, not the cohort median — 109/123 is what makes
#   this defect solid, so the arm should be judged the same way.
set -u
cd "$(dirname "$0")/.." || exit 1
LOG=logs/overnight/w4_gamma_2026-08-14.log
mkdir -p logs/overnight outputs/w4
exec > >(tee -a "$LOG") 2>&1
PY=.venv/bin/python
BC="logs/beat_classifier/version_4/checkpoints/beat-epoch=11-val_f1_avg_tol=0.603.ckpt"
LC="logs/layout_phrase/version_10/checkpoints/layout-epoch=09-val_token_acc=0.865.ckpt"
echo "=== W4: does the density weighting abandon the phrase? — $(date) ==="

# Songs with enough vocal phrases to score, taken from the wide cohort.
SONGS=$($PY - <<'PY'
import pathlib, sys
sys.path.insert(0,'src'); sys.path.insert(0,'scripts')
from eval_phrase_abandon import vocal_phrases
out=[]
for zp in sorted(pathlib.Path('outputs/wide_cohort').glob('*.zip')):
    s=zp.stem
    if not (pathlib.Path('data/raw')/f"{s}.zip").exists(): continue
    ph=vocal_phrases(s,1.2,2.0)
    if ph and len(ph)>=8: out.append(s)
    if len(out)>=12: break
print(' '.join(out))
PY
)
echo "songs: $SONGS"

run_arm () {   # $1 = tag, rest = env assignments
  tag=$1; shift
  mkdir -p "outputs/w4/$tag"
  for s in $SONGS; do
    out="outputs/w4/$tag/$s.zip"
    [ -f "$out" ] && continue
    env "$@" $PY scripts/generate.py "outputs/wide_cohort/audio/$s.ogg" --v7 \
      --beat-ckpt "$BC" --layout-ckpt "$LC" --difficulty Expert \
      --section-gate loud_only --song-name "$s" --seed 0 --output "$out" \
      > "outputs/w4/$tag/$s.log" 2>&1 || echo "  $tag $s FAILED"
  done
  echo "  [$tag] $(ls outputs/w4/$tag/*.zip 2>/dev/null | wc -l) maps"
}
run_arm gamma15 DENSITY_SELECT_GAMMA=1.5
run_arm gamma10 DENSITY_SELECT_GAMMA=1.0
run_arm evid0   BEAT_ONSET_EVIDENCE=0.0

echo ""; echo "--- EVAL --- $(date +%H:%M)"
$PY - <<'PY'
import pathlib, sys, statistics as st
import numpy as np
sys.path.insert(0,'src'); sys.path.insert(0,'scripts')
from beatsaber_automapper.evaluation import scorecard
from beatsaber_automapper.evaluation.alignment import note_times, alignment_metrics
from eval_phrase_abandon import vocal_phrases, phrase_silence
arms={'baseline':pathlib.Path('outputs/wide_cohort')}
for t in ('gamma15','gamma10','evid0'):
    p=pathlib.Path(f'outputs/w4/{t}')
    if p.exists(): arms[t]=p
songs=[z.stem for z in sorted(pathlib.Path('outputs/w4/gamma15').glob('*.zip'))]
print(f"n={len(songs)} songs\n")
print(f"  {'arm':<12}{'share>1s':>10}{'med_hole':>10}{'precision':>11}{'notes':>8}"
      f"{'worse than human':>18}")
hum={}
for s in songs:
    rh=scorecard._load_any(pathlib.Path('data/raw')/f"{s}.zip")
    ph=vocal_phrases(s,1.2,2.0)
    hum[s]=(phrase_silence(np.array(sorted(note_times(rh[0],rh[1]))),ph),ph)
for tag,d in arms.items():
    sh,mh,pr,nn,worse=[],[],[],[],0
    for s in songs:
        zp=d/f"{s}.zip"
        if not zp.exists(): continue
        r=scorecard._load_any(zp)
        if not r: continue
        t=np.array(sorted(note_times(r[0],r[1])))
        a=phrase_silence(t,hum[s][1])
        if not a: continue
        sh.append(a['share_over_1s']); mh.append(a['med_hole'])
        nn.append(len(r[0].color_notes))
        on=scorecard.onsets_for(pathlib.Path('outputs/wide_cohort')/f"{s}.zip")
        pr.append(alignment_metrics(r[0],bpm=r[1],onsets=on).metrics['onset_precision'])
        if hum[s][0] and a['share_over_1s']>hum[s][0]['share_over_1s']: worse+=1
    print(f"  {tag:<12}{st.median(sh):>10.4f}{st.median(mh):>10.4f}"
          f"{st.median(pr):>11.4f}{st.median(nn):>8.0f}{worse:>14}/{len(sh)}")
hs=[hum[s][0]['share_over_1s'] for s in songs if hum[s][0]]
print(f"  {'HUMAN':<12}{st.median(hs):>10.4f}")
print("""
VERDICT (pre-registered):
  LEVER FOUND  share>1s falls toward the human AND precision/notes hold.
  CONFOUNDED   it falls but precision/notes fall in step -> report the trade curve.
  REFUTED      it does not move -> the cause is upstream in Stage-1 (Track B).""")
PY
echo ""; echo "=== COMPLETE $(date) ==="
