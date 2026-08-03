#!/usr/bin/env bash
# ONSET EVIDENCE AT 5 SEEDS — resolve what n=3 could not, and price the trade
#
# The 3-seed run said: rhythm and idiom improve RESOLVABLY at beta 0.5 and 1.0,
# alignment (0.381 -> 0.221 -> 0.183) and precision (0.912 -> 0.923 -> 0.927) are
# consistent on every seed but NOT resolvable by the project's own 2 sd rule, and
# playfeel degrades MONOTONICALLY in beta (0.671 -> 0.781 -> 1.039).
#
# Two jobs here:
#
#   1. RESOLVE ALIGNMENT AND PRECISION. n=3 leaves pooled sd 0.113 on alignment,
#      so a 0.160 delta cannot clear 2 sd. n=5 shrinks the standard error by
#      ~1.3x and, more usefully, makes the sd estimate itself trustworthy -- at
#      n=3 the sd is nearly as uncertain as the mean.
#
#   2. PRICE THE PLAYFEEL TRADE. beta 0.3 adds a third point to a curve that so
#      far has three: 0.0 -> 0.5 -> 1.0. If playfeel is flat at 0.3 while rhythm
#      and idiom keep most of their gain, 0.3 is the promotable setting. If
#      playfeel already moves at 0.3, the trade is intrinsic and the lever should
#      be judged as a trade, not as a free win.
#
# ★ THE HONEST DEFAULT IS STILL "NOT PROMOTED". Two levers now look good on the
# suite (BEAT_TRIM_TAIL, BEAT_ONSET_EVIDENCE) and the suite has been wrong about
# "ready" twice and right zero times. The decisive test is Kyle playing them.
#
# ⚠️ playfeel is the axis Kyle's ear has most directly endorsed -- "when there is
# a slow spot we let the player breathe... we no longer have the monotony flood
# of notes". A lever that improves five axes while degrading THAT one is exactly
# the kind of trade the scorecard is bad at judging. Treat a playfeel regression
# as expensive regardless of what the other five do.

set -uo pipefail
cd /home/kyle/repos/beatsaber_automapper
source .venv/bin/activate

LOG=logs/overnight/onsetevid5_2026-08-03.log
mkdir -p logs/overnight
exec > >(tee -a "$LOG") 2>&1

echo "=== ONSET EVIDENCE @ 5 SEEDS START $(date -Is) ==="

ARMS="tf_hl014_ds048_trim,tf_hl014_ds048_trim_ev03,tf_hl014_ds048_trim_ev05"
python scripts/eval_sweep.py sweep --arms "$ARMS" --seeds 5
echo "=== SWEEP DONE $(date -Is) ==="

echo
echo "=== PLAYFEEL SUB-METRICS: what exactly degrades? $(date -Is) ==="
python - <<'PY'
import pathlib, sys, glob, statistics as st
sys.path.insert(0, "src")
from beatsaber_automapper.evaluation import playfeel, scorecard
arms = ["tf_hl014_ds048_trim", "tf_hl014_ds048_trim_ev03", "tf_hl014_ds048_trim_ev05"]
keys = None
print(f"{'metric':22s}" + "".join(f"{a.split('_')[-1]:>12s}" for a in arms))
acc = {}
for a in arms:
    vals = {}
    for s in range(5):
        for zp in glob.glob(f"outputs/eval_sweep_cache/{a}#s{s}__*.zip"):
            try:
                L = scorecard._load_any(pathlib.Path(zp))
            except Exception:
                continue
            if not L:
                continue
            m = playfeel.playfeel_metrics(L[0], bpm=L[1]).metrics
            for k, v in m.items():
                if v == v:
                    vals.setdefault(k, []).append(v)
    acc[a] = vals
    keys = keys or list(vals)
for k in keys or []:
    row = "".join(f"{st.median(acc[a][k]):>12.3f}" if acc[a].get(k) else f"{'--':>12s}"
                  for a in arms)
    print(f"{k:22s}{row}")
print("\nHuman refs: nps 3.91, diagonal share 0.370, double share 0.231.")
print("If the regression is DIAGONAL SHARE or DOUBLES rising, that is K2/C5")
print("territory and the lever is pushing on a defect we already track.")
PY

echo
echo "=== READ ==="
echo "  Promotable if: rhythm/idiom keep their gain, alignment and precision"
echo "  now clear 2 sd, and playfeel is FLAT at the chosen beta."
echo "  If playfeel moves at every beta, report it as a TRADE and let Kyle's"
echo "  ear decide -- he has endorsed the density pacing this would touch."
echo "=== COMPLETE $(date -Is) ==="
