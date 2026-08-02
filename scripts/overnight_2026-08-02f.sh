#!/usr/bin/env bash
# GIVE THE ALIGNMENT AXIS A MEASURED NOISE FLOOR, AND TEST THE BEST ARM 5 WAYS.
#
# `tf_hl014_ds048` scores alignment 0.40 against a 0.39 bar. Calling that a FAIL
# asserts a precision the suite has never demonstrated: **there is no measured
# noise floor for A8.** The 5-seed floor run (2026-08-01) predates the axis
# entirely. Assuming a floor is exactly how the handrole floor came to be ~3x
# understated, and how "b1_e17 beats b1_e15" was concluded from a difference that
# turned out to be noise.
#
# Four more seeds of the identical config -> five samples, which gives:
#   1. the first measured per-axis sd for ALIGNMENT,
#   2. whether the 4/6 is stable or another seed lottery (2 of 5 identical seeds
#      passed 5/5 last time this project called something a pass),
#   3. the re-seed precondition every verdict script here now demands before a
#      promotion is even discussed.
#
# CONTEXT — why this is the right job rather than another lever. Three decode
# knobs have now failed to move onset_precision off ~0.90:
#     density  tf_ds045/048/052/055 span 3.63-4.42 nps -> precision 0.895-0.904
#     gamma    2.5/1.5/1.0 -> 0.902/0.907/0.898 (non-monotone, inside noise)
#     prob floor  no effect at any quantile (per-window top-k already skips weak slots)
# A probs-dump replay predicted gamma would buy +0.007 monotonically; the real
# decode did not reproduce the direction, so the replay does not transfer -- NMS,
# thresholds and the section gate sit between them. In the replay a min-distance
# of 2-3 slots alone costs 0.948 -> 0.923-0.931, which is most of the gap between
# the replay's ceiling and what the pipeline achieves.
#
# So the residual is NOT reachable with the knobs we have, and the honest next
# question is not "which knob" but "is 0.40 even different from 0.39".
#
# VERDICT LOGIC
#   sd is small and all 5 seeds sit near 0.40 -> the arm genuinely misses the bar
#       by a hair. Report it as a near-miss with a measured floor, and take the
#       remaining work to the model rather than the decode.
#   sd is large (say >0.05) -> 0.40 vs 0.39 was never a distinction, and any
#       ranking this session made among arms within ~2sd needs re-reading.
#   seeds straddle the bar -> a lottery again. Say so, do not pick the winner,
#       and do not promote: picking the passing seed is fitting the bars.

set -uo pipefail
cd /home/kyle/repos/beatsaber_automapper
source .venv/bin/activate

LOG=logs/overnight/alignseeds_2026-08-02.log
mkdir -p logs/overnight
exec > >(tee -a "$LOG") 2>&1

echo "=== ALIGNMENT NOISE FLOOR / SEED STABILITY START $(date -Is) ==="

ARMS="tf_hl014_ds048_s1,tf_hl014_ds048_s2,tf_hl014_ds048_s3,tf_hl014_ds048_s4"
python scripts/eval_sweep.py sweep --arms "$ARMS"
echo "=== SWEEP DONE $(date -Is) ==="

echo "=== PER-SEED SCORECARDS ==="
for arm in tf_hl014_ds048 $(echo "$ARMS" | tr ',' ' '); do
  zips=(outputs/eval_sweep_cache/${arm}__*.zip)
  if [ ! -e "${zips[0]}" ]; then echo "-- $arm: NO CACHED MAPS, skipping"; continue; fi
  echo "-- $arm (${#zips[@]} maps)"
  python -m beatsaber_automapper.evaluation.scorecard "${zips[@]}" --label "$arm" || true
done

echo "=== MEASURED NOISE FLOOR, INCLUDING ALIGNMENT $(date -Is) ==="
python - <<'PY'
import pathlib, statistics, sys
REPO = pathlib.Path(".").resolve(); sys.path.insert(0, str(REPO / "src"))
from beatsaber_automapper.evaluation import alignment, scorecard
CACHE = REPO / "outputs" / "eval_sweep_cache"
ARMS = ["tf_hl014_ds048"] + [f"tf_hl014_ds048_s{i}" for i in (1, 2, 3, 4)]
# measured 2026-08-01 on 5 identical seeds; alignment had no entry because the
# axis did not exist yet
PRIOR_SD = {"flow": 0.099, "rhythm": 0.087, "idiom": 0.084, "handrole": 0.303,
            "playfeel": 0.048, "alignment": None}
BARS = {"flow": 0.50, "rhythm": 0.70, "idiom": 1.00, "handrole": 2.00,
        "playfeel": 1.00, "alignment": scorecard.ALIGN_GAP_BAR}

rows = {}
for arm in ARMS:
    zips = sorted(CACHE.glob(f"{arm}__*.zip"))
    if len(zips) < 20:
        print(f"  {arm}: only {len(zips)} maps, skipped"); continue
    loaded, prec = [], []
    for p in zips:
        try:
            r = scorecard._load_any(p)
        except Exception:
            continue
        if not r:
            continue
        loaded.append(r)
        if r[2] is not None:
            prec.append(alignment.alignment_metrics(
                r[0], bpm=r[1], onsets=r[2]).metrics["onset_precision"])
    res = scorecard.score_cohort(loaded, arm)
    rows[arm] = {"axes": {a.name: (a.gap, a.min_spread, a.passed) for a in res["axes"]},
                 "npass": sum(1 for a in res["axes"] if a.passed),
                 "viol": res["total_viol"],
                 "prec": statistics.median([x for x in prec if x == x])}

if len(rows) < 3:
    print("  not enough seeds finished to estimate a floor"); raise SystemExit(0)

print(f"\n{'seed':22s}{'prec':>8s}{'align':>8s}{'rhythm':>8s}{'flow':>7s}{'idiom':>7s}"
      f"{'hrole':>7s}{'pfeel':>7s}{'pass':>7s}")
print("-" * 82)
for arm in ARMS:
    if arm not in rows:
        continue
    r = rows[arm]
    g = lambda k: r["axes"][k][0] if k in r["axes"] else float("nan")
    print(f"{arm:22s}{r['prec']:8.3f}{g('alignment'):8.2f}{g('rhythm'):8.2f}"
          f"{g('flow'):7.2f}{g('idiom'):7.2f}{g('handrole'):7.2f}{g('playfeel'):7.2f}"
          f"{r['npass']:6d}/6")

print(f"\n{'axis':12s}{'mean':>9s}{'sd':>8s}{'min':>8s}{'max':>8s}{'bar':>8s}"
      f"{'prior sd':>10s}  verdict")
print("-" * 84)
for ax in ["alignment", "flow", "rhythm", "idiom", "handrole", "playfeel"]:
    vals = [rows[a]["axes"][ax][0] for a in rows if ax in rows[a]["axes"]]
    vals = [v for v in vals if v == v]
    if len(vals) < 3:
        continue
    sd = statistics.stdev(vals)
    prior = PRIOR_SD.get(ax)
    bar = BARS[ax]
    if prior is None:
        verdict = f"FIRST MEASUREMENT — 2sd = {2 * sd:.2f}"
    elif sd > prior:
        verdict = f"wider than 2026-08-01 ({prior:.3f})"
    else:
        verdict = "consistent with 2026-08-01"
    ps = f"{prior:.3f}" if prior is not None else "--"
    print(f"{ax:12s}{statistics.fmean(vals):9.3f}{sd:8.3f}{min(vals):8.3f}"
          f"{max(vals):8.3f}{bar:8.2f}{ps:>10s}  {verdict}")

av = [rows[a]["axes"]["alignment"][0] for a in rows if "alignment" in rows[a]["axes"]]
av = [v for v in av if v == v]
print("\n--- IS 0.40 DIFFERENT FROM THE 0.39 BAR? ---")
if len(av) >= 3:
    sd = statistics.stdev(av)
    mean = statistics.fmean(av)
    print(f"  alignment across {len(av)} identical seeds: mean {mean:.3f}, sd {sd:.3f}, "
          f"range {min(av):.3f}-{max(av):.3f}")
    print(f"  bar {scorecard.ALIGN_GAP_BAR:.2f}; distance from mean to bar = "
          f"{abs(mean - scorecard.ALIGN_GAP_BAR):.3f} = {abs(mean - scorecard.ALIGN_GAP_BAR) / sd if sd else float('inf'):.1f} sd")
    if sd and abs(mean - scorecard.ALIGN_GAP_BAR) < 2 * sd:
        print("  => NOT DISTINGUISHABLE FROM THE BAR. Report it as 'at the bar, within")
        print("     noise', never as a pass or a fail. Do not pick the passing seed.")
    else:
        print("  => genuinely distinct from the bar at this sample size.")
n_pass = sum(1 for a in rows if rows[a]["npass"] == 6 and rows[a]["viol"] == 0)
print(f"\n*** {n_pass}/{len(rows)} identical seeds pass all six axes ***")
print("    Whatever this says, the next step is the same: Kyle PLAYS one. His ear")
print("    found the defect five axes could not see; the suite has now been wrong")
print("    about 'ready' twice, and it has never been right about it once.")
PY

echo "=== COMPLETE $(date -Is) ==="
