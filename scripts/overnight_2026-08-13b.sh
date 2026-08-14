#!/usr/bin/env bash
# BEAT_GRID_PHASE=search at n=149 — find the grid offset instead of predicting it.
#
# WHY THERE IS A SECOND ARM. Mode `1` (apply the FITTED phase) was refuted tonight:
# subset 39 -> 37, alignment gap 0.62 -> 1.32, because corr(applied, wanted) fell
# from the +0.367 validated offline to +0.065 in production — the offline test used
# CACHED onsets while generate.py fits from Demucs stems. The defect is real; the
# ESTIMATOR was the problem.
#
# WHAT CHANGED. The diagnostic's "oracle" shift was never oracular — it maximised
# match rate against the cached STEM ONSETS, not against the human map, and
# generate.py already computes stem onsets. So SEARCH for the shift instead of
# predicting it, and refuse to move unless the search finds a real gain.
#
# ✅THE 10-SONG GATE THAT MODE `1` NEVER GOT (scored on the suite's INDEPENDENT
# cached onsets, which the search never sees):
#   4 big movers        hit the ORACLE EXACTLY  (0.900 / 0.877 / 0.926 / 0.956)
#   3 mode-1 regressions fixed or neutral       (323af -0.159 -> +0.022)
#   3 already fine      untouched, +0.000       (do-no-harm gate declined the shift)
#   zero regressions across all ten.
#
# 🔴PRE-REGISTERED READING — the SUBSET is the statistic, not the median:
#   SHIP     songs >0.10 below human fall from 39 toward the oracle's ~26, no axis
#            regresses resolvably, and THE DETECTOR CHECK BELOW COMES BACK CLEAN.
#   PARTIAL  the subset shrinks materially but stalls short of 26 => the residual is
#            the 15 songs no shift helps; that is a SELECTION defect, not phase.
#   PIVOT    it does not shrink => the gate did not generalise from 10 songs to 149,
#            and the 10 were chosen to be favourable. Say so plainly.
#
# ⚠️⚠️THE DETECTOR CHECK IS A GATE, NOT A FOOTNOTE. This optimises our map against
# OUR OWN onset detector, so it can fit that detector's systematic offset — the
# h_dist failure, and exactly what C2 warns about on 1f767 (where the HUMAN map
# wants the same -41 ms we do). If the songs we shift are also songs whose human map
# wants the same shift, we are tuning the detector and calling it quality. The
# diagnostic said only 1 of 39 is such a case; that must hold at n=149.
set -u
cd "$(dirname "$0")/.." || exit 1
LOG=logs/overnight/gphase_search_2026-08-13.log
mkdir -p logs/overnight
exec > >(tee -a "$LOG") 2>&1
PY=.venv/bin/python
echo "=== BEAT_GRID_PHASE=search @ n=149 — $(date) ==="

$PY scripts/build_wide_cohort.py --n 150 --seed 0 --variant prod --tag gsearch \
    --env "BEAT_GRID_PHASE=search"

D=outputs/wide_cohort_prod_gsearch
N=$(ls "$D"/*.zip 2>/dev/null | wc -l)
echo ""; echo "--- EVAL gsearch ($N maps) --- $(date +%H:%M)"
[ "$N" -lt 100 ] && { echo "SKIP: only $N maps generated"; exit 1; }

$PY - <<'PY'
import pathlib, statistics as st, sys
import numpy as np
sys.path.insert(0, "src"); sys.path.insert(0, "scripts")
from beatsaber_automapper.evaluation import scorecard
from beatsaber_automapper.evaluation.alignment import alignment_metrics, note_times

CTRL = pathlib.Path("outputs/wide_cohort")
ARM  = pathlib.Path("outputs/wide_cohort_prod_gsearch")

def load(p):
    r = scorecard._load_any(p)
    return r if r else (None, None, None)

rows = []
for zp in sorted(ARM.glob("*.zip")):
    song = zp.stem
    ctrl, hum = CTRL / f"{song}.zip", pathlib.Path("data/raw") / f"{song}.zip"
    if not ctrl.exists() or not hum.exists():
        continue
    # ⚠️SAME onsets on every side; load_expert_only returns a 2-tuple and the human
    # side is silently onset-less unless they are passed explicitly.
    on = scorecard.onsets_for(ctrl)
    if on is None or len(on) == 0:
        continue
    ab, abpm, _ = load(zp); cb, cbpm, _ = load(ctrl); hb, hbpm, _ = load(hum)
    if ab is None or cb is None or hb is None:
        continue
    ma = alignment_metrics(ab, bpm=abpm, onsets=on).metrics
    mc = alignment_metrics(cb, bpm=cbpm, onsets=on).metrics
    mh = alignment_metrics(hb, bpm=hbpm, onsets=on).metrics
    if ma["onset_precision"] != ma["onset_precision"]:
        continue
    ta, tc = note_times(ab, abpm), note_times(cb, cbpm)
    shift = (float(np.median(np.array(ta) - np.array(tc))) * 1000.0
             if len(ta) == len(tc) else float("nan"))
    # What shift does the HUMAN map want, and does it gain from it? If the human
    # gains too, the offset is in the DETECTOR and we are fitting it.
    hb2 = load(hum)[0]
    best_h, base_h = mh["onset_precision"], mh["onset_precision"]
    for d in np.arange(-120.0, 120.1, 5.0) / 1000.0:
        v = alignment_metrics(hb2, bpm=hbpm, onsets=np.asarray(on) - d).metrics
        best_h = max(best_h, v["onset_precision"])
    rows.append({"song": song, "arm": ma["onset_precision"], "ctrl": mc["onset_precision"],
                 "hum": base_h, "hum_best": best_h, "shift": shift,
                 "mad_a": ma["offset_mad_ms"], "mad_c": mc["offset_mad_ms"]})

n = len(rows)
print(f"\npaired on {n} songs\n")
arm = [r["arm"] for r in rows]; ctl = [r["ctrl"] for r in rows]; hum = [r["hum"] for r in rows]
print(f"  {'':<12}{'median':>9}{'vs human':>10}")
for lbl, v in (("control", ctl), ("search", arm), ("human", hum)):
    print(f"  {lbl:<12}{st.median(v):>9.4f}"
          f"{st.median([a - b for a, b in zip(v, hum)]):>+10.4f}")
print(f"\n  paired delta arm-control (median)  "
      f"{st.median([a - b for a, b in zip(arm, ctl)]):+.4f}")

bad_c = sum(1 for a, h in zip(ctl, hum) if a - h < -0.10)
bad_a = sum(1 for a, h in zip(arm, hum) if a - h < -0.10)
print(f"\n  ★songs >0.10 BELOW human:  control {bad_c}  ->  search {bad_a}   "
      f"(oracle predicted ~26)")
win  = sum(1 for r in rows if r["arm"] - r["ctrl"] > 0.02)
lose = sum(1 for r in rows if r["arm"] - r["ctrl"] < -0.02)
moved = [r for r in rows if abs(r["shift"]) > 1.0]
print(f"  songs moved >0.02:         better {win}   worse {lose}")
print(f"  songs the search shifted:  {len(moved)} / {n}  "
      f"(median |shift| {st.median([abs(r['shift']) for r in moved]):.1f} ms)"
      if moved else "  songs the search shifted: 0")
print(f"  median scatter (mad ms):   control {st.median([r['mad_c'] for r in rows]):.1f}"
      f"  ->  search {st.median([r['mad_a'] for r in rows]):.1f}")

# ⚠️THE DETECTOR CHECK. Of the songs we shifted, how many have a HUMAN map that
# also gains materially from a shift? Those are songs where the offset is in the
# onset detector and "improving" them is fitting it.
det = [r for r in moved if r["hum_best"] - r["hum"] > 0.02]
print(f"\n  ⚠️DETECTOR CHECK — of the {len(moved)} songs we shifted, "
      f"{len(det)} have a human map that ALSO gains >0.02 from a shift.")
print("     Those are detector-offset songs; the rest are genuinely our grid.")
if moved:
    print(f"     share {len(det) / len(moved):.1%}  "
          f"(the diagnostic said ~1 of 39 — a large share here means we are "
          f"fitting our own detector and must stop)")

print("\n  biggest movers:")
for r in sorted(rows, key=lambda r: -(r["arm"] - r["ctrl"]))[:8]:
    print(f"    {r['song']:<8} {r['ctrl']:.3f} -> {r['arm']:.3f} "
          f"({r['arm']-r['ctrl']:+.3f})  shift {r['shift']:+6.1f} ms  human {r['hum']:.3f}")
print("  biggest regressions:")
for r in sorted(rows, key=lambda r: (r["arm"] - r["ctrl"]))[:5]:
    print(f"    {r['song']:<8} {r['ctrl']:.3f} -> {r['arm']:.3f} "
          f"({r['arm']-r['ctrl']:+.3f})  shift {r['shift']:+6.1f} ms  human {r['hum']:.3f}")
PY

echo ""; echo "--- POSITIONAL AXES: a rigid translation must NOT move these ---"
echo "    (a tie to 2 dp is the PASS here; movement is a bug signal)"
$PY -m beatsaber_automapper.evaluation.scorecard outputs/wide_cohort/*.zip 2>&1 \
    | sed -n '3,14p' | sed 's/^/  control  /'
$PY -m beatsaber_automapper.evaluation.scorecard outputs/wide_cohort_prod_gsearch/*.zip 2>&1 \
    | sed -n '3,14p' | sed 's/^/  gsearch  /'

echo ""; echo "=== COMPLETE $(date) ==="
