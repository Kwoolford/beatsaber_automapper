#!/usr/bin/env bash
# K1 DECAY — DOES ONSET-EVIDENCE WEIGHTING SURVIVE 24 SONGS AND 3 SEEDS?
#
# THE CAUSE (measured, BEAT_PROBS_DUMP). On 1f8d6's outro, windows with ZERO
# detected onsets carry Stage-1 wmean 0.28-0.42 -- as high as the body of the
# song -- so density-select allocates ~35 notes to a region with ~2 real onsets.
# wmean IS the defect, so no ceiling computed from wmean can fix it. These arms
# multiply the window weight by librosa's audio onset density instead.
#
# THE PROBE THAT MOTIVATED THIS (1f8d6, seed 0, on top of trim):
#   drift 0.378 -> 0.130 (human p90 0.145), q5 precision 0.622 -> 0.86,
#   overall precision 0.886 -> 0.930, notes 490 -> 501.
# ★ One song, one seed. That is the single-song probe trap the landmine list
#   warns about, and it has already caught two hypotheses in this project.
#
# ARMS: trim (control, cached) vs trim+ev0.5 vs trim+ev1.0, 3 seeds each.
#
# ⚠️ THE RISK IS DETECTOR-FITTING, and it is the main thing to read for.
# We weight by librosa-on-mix; A8 scores against a per-stem onset union. Those
# correlate, so part of any precision gain may be fitting the grader rather than
# the music -- the h_dist failure in a new costume. Three tells, in order:
#
#   1. PRECISION LANDING WELL ABOVE HUMAN (0.930 +- 0.032). Matching human is
#      the goal; sailing past it means we are grading ourselves. If ev1.0 lands
#      at ~0.96+, prefer ev0.5 or reject both.
#   2. THE OTHER FIVE AXES MOVING. Precision bought with worse rhythm, flow,
#      idiom, handrole or playfeel is a trade, not a win. All five should sit
#      inside 2 sd. Watch nps too -- this lever reshapes the budget, and it must
#      not quietly thin the map.
#   3. WHERE THE GAIN LANDS. It should concentrate on the songs the human
#      control says are OURS (1f336, 1f3d7, 1f767, 1f65d, 1f333) rather than on
#      1f8d6 and 1f8ce, whose HUMAN maps drift too (0.147, 0.208) and where part
#      of the drift belongs to the song or the detector. Gain concentrated on
#      those two is the signature of detector-fitting.
#
# A NULL IS A PERFECTLY GOOD OUTCOME and would be the fourth failed decode lever
# (C1 lists three). The difference this time is that the previous three were all
# functions of Stage-1's own probabilities and this one is not -- but that is a
# reason to try it, not evidence that it works.

set -uo pipefail
cd /home/kyle/repos/beatsaber_automapper
source .venv/bin/activate

LOG=logs/overnight/onsetevid_2026-08-03.log
mkdir -p logs/overnight
exec > >(tee -a "$LOG") 2>&1

echo "=== ONSET-EVIDENCE (3 arms x 3 seeds; control cached) START $(date -Is) ==="

ARMS="tf_hl014_ds048_trim,tf_hl014_ds048_trim_ev05,tf_hl014_ds048_trim_ev10"
python scripts/eval_sweep.py sweep --arms "$ARMS" --seeds 3
echo "=== SWEEP DONE $(date -Is) ==="

echo
echo "=== K1 DRIFT SUB-METRICS (all 3 seeds per arm) $(date -Is) ==="
for arm in tf_hl014_ds048_trim tf_hl014_ds048_trim_ev05 tf_hl014_ds048_trim_ev10; do
  for s in 0 1 2; do
    echo "--- $arm seed $s ---"
    python scripts/eval_align_drift.py --human --n 60 \
      --maps "outputs/eval_sweep_cache/${arm}#s${s}__*.zip" --label "$arm" 2>&1 \
      | grep -E "maps above the human p90|drift_q1_q5|tail_after_frac"
  done
done

echo
echo "=== PER-SONG: OURS-ALONE vs SHARED-WITH-HUMAN $(date -Is) ==="
echo "Detector-fitting shows up as gain concentrated on 1f8d6/1f8ce."
python - <<'PY'
import pathlib, sys, glob
sys.path.insert(0, "src"); sys.path.insert(0, "scripts")
import importlib.util
spec = importlib.util.spec_from_file_location("d", "scripts/eval_align_drift.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
from beatsaber_automapper.evaluation import scorecard
OURS = ["1f336", "1f3d7", "1f767", "1f65d", "1f333"]
SHARED = ["1f8d6", "1f8ce"]
arms = ["tf_hl014_ds048_trim", "tf_hl014_ds048_trim_ev05", "tf_hl014_ds048_trim_ev10"]
print(f"{'song':8s}" + "".join(f"{a.split('_')[-1]:>12s}" for a in arms))
for grp, name in ((OURS, "OURS ALONE"), (SHARED, "SHARED w/ human")):
    print(f"-- {name} --")
    for sid in grp:
        cells = []
        for a in arms:
            ds = []
            for s in (0, 1, 2):
                g = glob.glob(f"outputs/eval_sweep_cache/{a}#s{s}__{sid}.zip")
                if not g:
                    continue
                L = scorecard._load_any(pathlib.Path(g[0]))
                if not L:
                    continue
                r = m.drift_metrics(L[0], bpm=L[1], onsets=L[2])
                if r:
                    ds.append(r["drift_q1_q5"])
            cells.append(f"{sum(ds)/len(ds):+.3f}" if ds else "--")
        print(f"{sid:8s}" + "".join(f"{c:>12s}" for c in cells))
print("\n(mean drift over 3 seeds; human p90 = 0.145)")
PY

echo
echo "=== READ ==="
echo "  Best case: drift exceedance falls toward 10%, precision approaches but"
echo "  does NOT sail past human 0.930, other five axes flat, gain concentrated"
echo "  on the ours-alone songs. Then it is a promotion candidate."
echo "  Precision >> 0.930, or gain concentrated on 1f8d6/1f8ce, means we are"
echo "  fitting the onset detector -- reject and say so."
echo "=== COMPLETE $(date -Is) ==="
