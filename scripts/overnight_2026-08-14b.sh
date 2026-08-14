#!/usr/bin/env bash
# BEAT_SUBDIV_AUTO at n=149 — the whole cohort, one arm, both groups at once.
#
# Unlike the oracle arm, this uses NO knowledge of which songs are half-tempo: the
# generator decides for itself from the fitted bpm (< 95 => double the subdivision).
# So this measures the DEPLOYABLE lever, not the ceiling.
#
# 🔴PRE-REGISTERED — the DoD is asymmetric and the false-positive side is the gate:
#   SHIP     the `half` group's ebpm ratio moves off 0.500 toward 1.000, AND **zero**
#            `same`-group songs regress resolvably on onset precision. The oracle arm
#            showed a false positive costs 0.127, so "a few regressions is fine" is
#            not available — the songs it would damage are ones that already work.
#   TUNE     the half group improves but some `same` songs are touched => the bpm<95
#            threshold is wrong on this cohort; report which songs and their fitted
#            bpm rather than sweeping the threshold to fit them.
#   DEAD     the half group does not move => the auto path is not firing where the
#            oracle arm said it should. Compare the fired set against the `half`
#            labels before concluding anything about the lever itself.
#
# ⚠️The threshold was chosen on the post-fit bpm of THIS cohort, so its false-positive
# count here is optimistic by construction. What the arm adds is whether the songs it
# fires on are the ones that benefit, and what it costs the rest.
set -u
cd "$(dirname "$0")/.." || exit 1
LOG=logs/overnight/subdiv_auto_2026-08-14.log
mkdir -p logs/overnight
exec > >(tee -a "$LOG") 2>&1
PY=.venv/bin/python
echo "=== BEAT_SUBDIV_AUTO @ n=149 — $(date) ==="

$PY scripts/build_wide_cohort.py --n 150 --seed 0 --variant prod --tag subauto \
    --env "BEAT_SUBDIV_AUTO=1"

D=outputs/wide_cohort_prod_subauto
N=$(ls "$D"/*.zip 2>/dev/null | wc -l)
echo ""; echo "--- EVAL ($N maps) --- $(date +%H:%M)"
[ "$N" -lt 100 ] && { echo "SKIP: only $N maps"; exit 1; }

$PY - <<'PY'
import json, pathlib, statistics as st, sys
sys.path.insert(0,'src'); sys.path.insert(0,'scripts')
from beatsaber_automapper.evaluation import scorecard, flow
from beatsaber_automapper.evaluation.alignment import alignment_metrics
lab={r['song']: r['label'] for r in json.load(open('outputs/true_bpm_wide_cohort_labels.json'))}
ARM=pathlib.Path('outputs/wide_cohort_prod_subauto'); CTRL=pathlib.Path('outputs/wide_cohort')
rows=[]
for zp in sorted(ARM.glob('*.zip')):
    s=zp.stem; c=CTRL/f"{s}.zip"; h=pathlib.Path('data/raw')/f"{s}.zip"
    if not(c.exists() and h.exists()): continue
    on=scorecard.onsets_for(c)
    if on is None or len(on)==0: continue
    ra,rc,rh=scorecard._load_any(zp),scorecard._load_any(c),scorecard._load_any(h)
    if not(ra and rc and rh): continue
    def E(r): return flow.flow_metrics(r[0],bpm=r[1]).metrics['ebpm_burst']
    def P(r): return alignment_metrics(r[0],bpm=r[1],onsets=on).metrics['onset_precision']
    eh=E(rh)
    if not eh: continue
    fired = len(ra[0].color_notes)!=len(rc[0].color_notes) or abs(ra[1]-rc[1])>1e-6
    rows.append({'song':s,'label':lab.get(s,'?'),'fired':fired,
                 'r_arm':E(ra)/eh,'r_ctl':E(rc)/eh,'p_arm':P(ra),'p_ctl':P(rc)})
fired=[r for r in rows if r['fired']]
print(f"\nn={len(rows)}   the generator chose subdiv 8 on {len(fired)} songs\n")
from collections import Counter
print("  it fired on:", dict(Counter(r['label'] for r in fired)))
print("  label mix overall:", dict(Counter(r['label'] for r in rows)))
for grp in ('half','same'):
    g=[r for r in rows if r['label']==grp]
    gf=[r for r in g if r['fired']]
    if not g: continue
    print(f"\n=== {grp.upper()} — n={len(g)}, fired on {len(gf)} ===")
    print(f"  ebpm ratio:  {st.median([r['r_ctl'] for r in g]):.3f} -> "
          f"{st.median([r['r_arm'] for r in g]):.3f}   (1.000 = human)")
    print(f"  precision:   {st.median([r['p_ctl'] for r in g]):.4f} -> "
          f"{st.median([r['p_arm'] for r in g]):.4f}")
    reg=[r for r in g if r['p_arm']-r['p_ctl'] < -0.02]
    print(f"  ★songs regressing >0.02 on precision: {len(reg)}")
    for r in reg[:8]:
        print(f"      {r['song']}  {r['p_ctl']:.3f} -> {r['p_arm']:.3f}")
print(f"""
VERDICT (pre-registered):
  SHIP  half ratio off 0.500 AND zero `same` songs regressing >0.02.
  TUNE  half improves but `same` songs are touched -> report which, and their
        fitted bpm; do not sweep the threshold to fit this cohort.
  DEAD  half does not move -> check the FIRED set against the labels first.""")
PY
echo ""; echo "=== COMPLETE $(date) ==="
