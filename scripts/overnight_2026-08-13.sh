#!/usr/bin/env bash
# BEAT_GRID_PHASE at n=149 — does consuming the phase we already estimate fix the
# baseline's alignment failure?
#
# THE DEFECT. Restoring A8 exposed that the promoted baseline fails alignment at
# n=149 (ours 0.8914 vs human 0.9492 paired). It is BIMODAL and song-driven
# (corr(s0,s1) = +0.981), and nothing checked predicted which songs — bpm, nps,
# density ratio, onset density all null. Phase was never among them, and
# generate.py computes it, logs it, and throws it away.
#
# WHAT IS ALREADY MEASURED (2026-08-13, CPU only, no generation):
#   * a global shift recovers +0.0428 on the 39 failing songs vs a +0.0174 floor
#   * 20 of the 39 gain from a shift their HUMAN map does not want => our grid
#   * the fitted phase PREDICTS that shift: median |err| 15.2 ms vs 39.1 chance,
#     and corr +0.757 on the 12 songs a shift rescues most
#   * smoke test, 2c352: precision 0.4562 -> 0.8969 (human 0.9569), scatter
#     23.1 -> 6.9 ms (human 7.0), note count unchanged
#
# 🔴PRE-REGISTERED READING — and the SUBSET is the statistic, not the mean. The
# cohort median is expected to move only ~+0.003 even if this works perfectly,
# because it is a fix for a quarter of the cohort. Reading the median here is the
# exact trap this project walked into twice on 2026-08-11.
#
#   SHIP      songs >0.10 below human fall from 39 toward the ~26 the oracle
#             predicts, AND no other axis moves resolvably.
#   PARTIAL   the subset shrinks but by materially less than the oracle's 13
#             songs => the ESTIMATOR is the limit, not the idea; the next move is
#             a better phase estimate, not a different lever.
#   PIVOT     the subset does not shrink => the note-only translation does not
#             transfer from the diagnostic to generation, and the difference is
#             the thing to explain before anything else is built.
#
# ⚠️A RIGID TRANSLATION MUST NOT MOVE A POSITIONAL AXIS. Adding a constant to every
# note's beat leaves every interval and every position identical, so flow / rhythm
# / idiom / handrole are expected to tie to 3+ decimals. Here a TIE IS THE PASS —
# and any axis that moves resolvably is a BUG SIGNAL, not a cost. (The project's
# own rule, inverted: "a tie to 3+ decimals is a construction, not a result".)
set -u
cd "$(dirname "$0")/.." || exit 1
LOG=logs/overnight/grid_phase_2026-08-13.log
mkdir -p logs/overnight
exec > >(tee -a "$LOG") 2>&1
PY=.venv/bin/python
echo "=== BEAT_GRID_PHASE @ n=149 — $(date) ==="

$PY scripts/build_wide_cohort.py --n 150 --seed 0 --variant prod --tag gphase \
    --env "BEAT_GRID_PHASE=1"

D=outputs/wide_cohort_prod_gphase
N=$(ls "$D"/*.zip 2>/dev/null | wc -l)
echo ""; echo "--- EVAL gphase ($N maps) --- $(date +%H:%M)"
[ "$N" -lt 100 ] && { echo "SKIP: only $N maps generated"; exit 1; }

$PY - <<'PY'
import pathlib, statistics as st, sys
sys.path.insert(0, "src"); sys.path.insert(0, "scripts")
from beatsaber_automapper.evaluation import scorecard
from beatsaber_automapper.evaluation.alignment import alignment_metrics

CTRL = pathlib.Path("outputs/wide_cohort")
ARM  = pathlib.Path("outputs/wide_cohort_prod_gphase")

def prec(p, onsets):
    r = scorecard._load_any(p)
    if r is None:
        return None
    bm, bpm, _ = r
    m = alignment_metrics(bm, bpm=bpm, onsets=onsets).metrics
    v = m.get("onset_precision")
    return (v, m.get("offset_mad_ms"), len(bm.color_notes)) if v == v else None

rows = []
for zp in sorted(ARM.glob("*.zip")):
    song = zp.stem
    ctrl, hum = CTRL / f"{song}.zip", pathlib.Path("data/raw") / f"{song}.zip"
    if not ctrl.exists() or not hum.exists():
        continue
    # ⚠️SAME onsets on every side — load_expert_only returns a 2-tuple and the
    # human side is silently onset-less unless they are passed explicitly. That
    # mistake made the first run of this analysis return 0 scorable songs.
    on = scorecard.onsets_for(ctrl)
    if on is None or len(on) == 0:
        continue
    a, b, h = prec(zp, on), prec(ctrl, on), prec(hum, on)
    if a and b and h:
        rows.append((song, a, b, h))

n = len(rows)
print(f"\npaired on {n} songs\n")
arm  = [r[1][0] for r in rows]
ctl  = [r[2][0] for r in rows]
hum  = [r[3][0] for r in rows]
print(f"  {'':<18}{'median':>9}{'vs human':>10}")
for lbl, v in (("control", ctl), ("BEAT_GRID_PHASE", arm), ("human", hum)):
    d = st.median([x - y for x, y in zip(v, hum)])
    print(f"  {lbl:<18}{st.median(v):>9.4f}{d:>+10.4f}")
print(f"\n  paired delta arm-control (median)  "
      f"{st.median([a - b for a, b in zip(arm, ctl)]):+.4f}")

# ★THE STATISTIC THAT MATTERS. The mean cannot see a subset defect; this project
# has been caught by that twice on two different instruments.
bad_c = sum(1 for a, h in zip(ctl, hum) if a - h < -0.10)
bad_a = sum(1 for a, h in zip(arm, hum) if a - h < -0.10)
print(f"\n  songs >0.10 BELOW human:   control {bad_c}  ->  gphase {bad_a}   "
      f"(oracle predicted ~26)")
win  = sum(1 for a, b in zip(arm, ctl) if a - b > 0.02)
lose = sum(1 for a, b in zip(arm, ctl) if a - b < -0.02)
print(f"  songs moved >0.02:         better {win}   worse {lose}")
print(f"  median scatter (mad ms):   control {st.median([r[2][1] for r in rows]):.1f}"
      f"  ->  gphase {st.median([r[1][1] for r in rows]):.1f}"
      f"   (human {st.median([r[3][1] for r in rows]):.1f})")
dn = [r[1][2] - r[2][2] for r in rows]
print(f"  note-count change:         median {st.median(dn):+.0f}, "
      f"min {min(dn):+d}, max {max(dn):+d}   (a translation should be ~0)")

print("\n  biggest movers:")
mv = sorted(rows, key=lambda r: -(r[1][0] - r[2][0]))[:8]
for song, a, b, h in mv:
    print(f"    {song:<8} {b[0]:.3f} -> {a[0]:.3f}  ({a[0]-b[0]:+.3f})   human {h[0]:.3f}")
worst = sorted(rows, key=lambda r: (r[1][0] - r[2][0]))[:5]
print("  biggest regressions:")
for song, a, b, h in worst:
    print(f"    {song:<8} {b[0]:.3f} -> {a[0]:.3f}  ({a[0]-b[0]:+.3f})   human {h[0]:.3f}")

print(f"""
VERDICT LOGIC (pre-registered):
  SHIP     bad-song count {bad_c} -> {bad_a}, approaching the oracle's ~26, and the
           positional axes below tie to 3 decimals.
  PARTIAL  it shrinks but well short of 26 => the phase ESTIMATOR is the limit,
           not the idea. Next move is a better estimate, not another lever.
  PIVOT    it does not shrink => the note-only translation did not transfer from
           the diagnostic to generation. Explain that gap before building on.""")
PY

echo ""; echo "--- POSITIONAL AXES: a rigid translation must NOT move these ---"
echo "    (a tie to 3+ decimals is the PASS here; movement is a bug signal)"
$PY -m beatsaber_automapper.evaluation.scorecard outputs/wide_cohort/*.zip 2>&1 \
    | sed -n '3,14p' | sed 's/^/  control  /'
$PY -m beatsaber_automapper.evaluation.scorecard outputs/wide_cohort_prod_gphase/*.zip 2>&1 \
    | sed -n '3,14p' | sed 's/^/  gphase   /'

echo ""; echo "=== COMPLETE $(date) ==="
