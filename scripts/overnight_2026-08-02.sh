#!/usr/bin/env bash
# IS OUR TIMING SCATTER THE MODEL'S, OR DID WE MANUFACTURE IT IN POST?
#
# Axis A8 (2026-08-02) scored notes against the AUDIO for the first time. Against a
# 98-map human reference our arms sit at alignment_gap 5.0-7.2 where the bar is
# 0.39 and a HELD-OUT HUMAN COHORT scores 0.20. For scale, on the same battery:
#   metronome 2.37   timing_jitter 1.91   timing_random 6.74
# Our production maps score WORSE than a metronome, and `b1_e17_ds055` (7.21)
# scores worse than a map whose note times were REPLACED WITH RANDOM ONES (6.74).
#
# Two keys drive it: onset_precision 0.73-0.77 (human 0.930) and offset_mad 17.4ms
# (human 10.3ms). This sweep attacks the second, because there is a mechanical
# suspect with an arithmetic prediction attached:
#
# `_quantize_to_beat_grid` snaps every Stage-1 onset to a 1/8 beat grid. Spacing is
# 60/bpm/8; displacement is uniform on +-half of that, so predicted MAD = 1875/bpm
# ms -- 11.7ms at 160bpm, where we MEASURED 11.7ms on 1f767. The offset histogram
# agrees: human offsets are a unimodal peak on the onset, ours are FLAT across the
# whole +-50ms window, which is what a grid does and not what timing does. Stage-1's
# own frames are 11.6ms apart, so this function discards timing the model already
# had. The grid is also built from the DETECTED bpm, which is exact on 1 of 21 eval
# songs (median error 0.74%; four songs land at 2/3 of the true tempo), so it slides
# against the music as the song plays.
#
# ARMS halve the displacement bound in turn: 23.2 -> 11.6 -> 5.8 -> 0 ms.
#   ds055        (cached control, subdiv 8 = current default)
#   q16_ds055    subdiv 16
#   q32_ds055    subdiv 32
#   q0_ds055     snapping OFF, frame resolution only
#   q16_hl014_ds055 / q0_hl014_ds055 -- the same on the best-known config
#
# VERDICT LOGIC
#   offset_mad_ms falls with the bound  -> the scatter was OURS, made in post, and
#       this is a decode-time fix for half of the defect Kyle hears. Promote the
#       finest subdivision that does not regress another axis by more than its
#       MEASURED noise floor (2sd: flow 0.20, rhythm 0.17, idiom 0.17, handrole
#       0.61, playfeel 0.10 -- from the 5-seed floor run, 2026-08-01).
#   offset_mad_ms flat across arms      -> the scatter is the MODEL's. This lever is
#       dead, say so, and alignment work moves to Stage 1 (a retrain, not a knob).
#   precision does not move either way  -> EXPECTED. Snapping by <=23ms cannot push
#       a note outside a 50ms tolerance, so it cannot explain the precision gap.
#       Precision is a SELECTION defect (we put notes where no onset exists) and
#       needs its own experiment; do not read a flat precision as this sweep failing.
#
# WATCH: q0 puts notes off the 1/16 grid, and human maps are 94-99% ON it. Expect
# A2 rhythm's offgrid guard to push back. That trade-off is the point of pricing
# q16 and q32 alongside q0 rather than just switching snapping off.

set -uo pipefail
cd /home/kyle/repos/beatsaber_automapper
source .venv/bin/activate

LOG=logs/overnight/quantgrid_2026-08-02.log
mkdir -p logs/overnight
exec > >(tee -a "$LOG") 2>&1

echo "=== BEAT-GRID QUANTISATION SWEEP START $(date -Is) ==="

ARMS="q16_ds055,q32_ds055,q0_ds055,q16_hl014_ds055,q0_hl014_ds055"
python scripts/eval_sweep.py sweep --arms "$ARMS"
echo "=== SWEEP DONE $(date -Is) ==="

echo "=== SIX-AXIS SCORECARDS (A8 included) ==="
for arm in ds055 hl014_ds055 $(echo "$ARMS" | tr ',' ' '); do
  zips=(outputs/eval_sweep_cache/${arm}__*.zip)
  if [ ! -e "${zips[0]}" ]; then echo "-- $arm: NO CACHED MAPS, skipping"; continue; fi
  echo "-- $arm (${#zips[@]} maps)"
  python -m beatsaber_automapper.evaluation.scorecard "${zips[@]}" --label "$arm" || true
done

echo "=== ALIGNMENT VERDICT $(date -Is) ==="
python - <<'PY'
import pathlib, statistics, sys
REPO = pathlib.Path(".").resolve(); sys.path.insert(0, str(REPO / "src"))
from beatsaber_automapper.evaluation import alignment, scorecard
CACHE = REPO / "outputs" / "eval_sweep_cache"

# displacement bound in ms at 140bpm (the songset's rough median) for each arm
BOUND = {"ds055": 26.8, "q16_ds055": 13.4, "q32_ds055": 6.7, "q0_ds055": 0.0,
         "hl014_ds055": 26.8, "q16_hl014_ds055": 13.4, "q0_hl014_ds055": 0.0}
ARMS = ["ds055", "q16_ds055", "q32_ds055", "q0_ds055",
        "hl014_ds055", "q16_hl014_ds055", "q0_hl014_ds055"]
HUMAN_MAD, HUMAN_PREC = 10.35, 0.930

rows = {}
for arm in ARMS:
    zips = sorted(CACHE.glob(f"{arm}__*.zip"))
    if len(zips) < 20:
        print(f"  {arm}: only {len(zips)} maps, skipped"); continue
    loaded = []
    for p in zips:
        try:
            r = scorecard._load_any(p)
        except Exception:
            r = None
        if r:
            loaded.append(r)
    res = scorecard.score_cohort(loaded, arm)
    recs = res["records"]
    mad = [r["offset_mad_ms"] for r in recs if r.get("offset_mad_ms") == r.get("offset_mad_ms")]
    prc = [r["onset_precision"] for r in recs if r.get("onset_precision") == r.get("onset_precision")]
    rows[arm] = {"axes": {a.name: (a.gap, a.min_spread, a.passed) for a in res["axes"]},
                 "mad": statistics.median(mad) if mad else float("nan"),
                 "prec": statistics.median(prc) if prc else float("nan"),
                 "viol": res["total_viol"], "n_pass": sum(1 for a in res["axes"] if a.passed)}

print(f"\n{'arm':20s}{'bound_ms':>10s}{'mad_ms':>9s}{'prec':>8s}"
      f"{'align':>8s}{'rhythm':>8s}{'flow':>7s}{'idiom':>7s}{'hrole':>7s}{'pfeel':>7s}{'pass':>6s}")
print("-" * 104)
for arm in ARMS:
    if arm not in rows: continue
    r = rows[arm]; ax = r["axes"]
    g = lambda k: ax[k][0] if k in ax else float("nan")
    print(f"{arm:20s}{BOUND.get(arm, float('nan')):10.1f}{r['mad']:9.1f}{r['prec']:8.3f}"
          f"{g('alignment'):8.2f}{g('rhythm'):8.2f}{g('flow'):7.2f}{g('idiom'):7.2f}"
          f"{g('handrole'):7.2f}{g('playfeel'):7.2f}{r['n_pass']:5d}/6")
print(f"\nhuman reference: offset_mad {HUMAN_MAD:.1f}ms, onset_precision {HUMAN_PREC:.3f}")

print("\n--- DOES SCATTER TRACK THE QUANTISATION BOUND? ---")
fam = [a for a in ["ds055", "q16_ds055", "q32_ds055", "q0_ds055"] if a in rows]
if len(fam) >= 3:
    mads = [rows[a]["mad"] for a in fam]
    drop = mads[0] - min(mads)
    monotone = all(mads[i] >= mads[i + 1] - 0.5 for i in range(len(mads) - 1))
    for a in fam:
        print(f"  {a:16s} bound {BOUND[a]:5.1f}ms -> measured MAD {rows[a]['mad']:5.1f}ms")
    print(f"\n  MAD falls {drop:.1f}ms from the 1/8 default to the best arm"
          f"; monotone with the bound: {monotone}")
    if drop >= 3.0 and monotone:
        print("  => THE SCATTER WAS OURS. Quantisation manufactured it; this is a")
        print("     decode-time fix. Promote the finest subdiv that costs no other")
        print("     axis more than its 2sd noise floor.")
    elif drop >= 3.0:
        print("  => scatter improves but NOT monotonically — something else moves with")
        print("     the subdivision (density-curve interaction?). Investigate before")
        print("     promoting; a non-monotone response is not a mechanism.")
    else:
        print("  => THE SCATTER IS THE MODEL'S. Lever dead. Record the negative and")
        print("     move alignment work to Stage 1; do not tune this knob further.")

print("\n--- REGRESSION CHECK (2sd measured noise floors, 5-seed run 2026-08-01) ---")
FLOOR2SD = {"flow": 0.20, "rhythm": 0.17, "idiom": 0.17, "handrole": 0.61, "playfeel": 0.10}
for base, arms in [("ds055", ["q16_ds055", "q32_ds055", "q0_ds055"]),
                   ("hl014_ds055", ["q16_hl014_ds055", "q0_hl014_ds055"])]:
    if base not in rows: continue
    for arm in arms:
        if arm not in rows: continue
        bad = []
        for ax, f in FLOOR2SD.items():
            d = rows[arm]["axes"].get(ax, (float("nan"),))[0] - rows[base]["axes"].get(ax, (float("nan"),))[0]
            if d > f:
                bad.append(f"{ax} +{d:.2f} (>{f})")
        print(f"  {arm:20s} vs {base:16s} "
              f"{'REGRESSES: ' + ', '.join(bad) if bad else 'no axis regresses beyond noise'}")
PY

echo "=== COMPLETE $(date -Is) ==="
