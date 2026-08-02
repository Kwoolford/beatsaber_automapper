#!/usr/bin/env bash
# THE LAST 0.0067 OF PRECISION — is it the budget ALLOCATION?
#
# After the tempo fix the alignment axis sits at 0.40-0.59 against a 0.39 bar, and
# the entire remainder is onset_precision: 0.902 vs a human 0.930, with timing
# scatter (10.2ms) already BETTER than human. Passing needs precision >= 0.9087,
# so we are short by 0.0067.
#
# Two candidate causes were tested, and both behaved OPPOSITE to intuition:
#
#   1. Total density does not move precision. tf_ds045/048/052/055 span 3.63-4.42
#      nps and precision stays 0.895/0.902/0.904/0.902. This was logged as an
#      explicit prediction beforehand and falsified.
#   2. Stage-1 probability DOES know where the music is: AUROC 0.755 against "this
#      slot is on a detected onset", top decile 0.986 precise against a 0.687 base
#      rate. So it is not simply a representation gap, and the Track B thesis is
#      not what this particular number is about.
#
# Replaying selection policies over a BEAT_PROBS_DUMP at a fixed budget locates it
# in the budget ALLOCATION, again in the unintuitive direction:
#
#     global top-k by probability     0.948
#     per-window gamma = 1.0          0.944
#     per-window gamma = 2.5 (ship)   0.937
#     per-window gamma = 4.0          0.919
#     per-window gamma = 8.0          0.894
#
# High gamma concentrates the budget into loud windows, forcing notes deeper down
# those windows' ranking while starving quiet windows holding a few excellent
# onsets. A probability floor changes nothing (0.937 at every quantile) because
# per-window top-k already skips the weak slots inside a window.
#
# ARMS: gamma 1.0 and 1.5 at the re-tuned density, plus 1.5 on the hand-lead arm.
# Controls tf_ds048 and tf_hl014_ds048 are cached.
#
# VERDICT LOGIC
#   alignment passes and density_corr holds -> promote the pair (BEAT_TEMPO_FIT +
#       the lower gamma), then RE-SEED 5x before believing it and get Kyle to play
#       it. This would be the first arm in the project to pass the axis that
#       measures his actual complaint.
#   alignment passes but density_corr collapses -> a REAL TENSION between "notes
#       land on the music" and "density follows the music". Report it as a tension;
#       gamma was raised to 2.5 on 2026-06-30 specifically to buy density_corr
#       (+0.53, 5/6 songs), so this is trading a won axis for another. Do not
#       quietly pick a side.
#   alignment does not move -> the dump replay does not transfer to the real
#       decode (NMS, thresholds and the section gate all sit between them). Say so;
#       the replay was a prediction, not a measurement of the shipped path.

set -uo pipefail
cd /home/kyle/repos/beatsaber_automapper
source .venv/bin/activate

LOG=logs/overnight/gamma_2026-08-02.log
mkdir -p logs/overnight
exec > >(tee -a "$LOG") 2>&1

echo "=== GAMMA / ALLOCATION SWEEP START $(date -Is) ==="

ARMS="tf_g1_ds048,tf_g15_ds048,tf_hl014_g15_ds048"
python scripts/eval_sweep.py sweep --arms "$ARMS"
echo "=== SWEEP DONE $(date -Is) ==="

echo "=== SIX-AXIS SCORECARDS ==="
for arm in tf_ds048 tf_g1_ds048 tf_g15_ds048 tf_hl014_ds048 tf_hl014_g15_ds048; do
  zips=(outputs/eval_sweep_cache/${arm}__*.zip)
  if [ ! -e "${zips[0]}" ]; then echo "-- $arm: NO CACHED MAPS, skipping"; continue; fi
  echo "-- $arm (${#zips[@]} maps)"
  python -m beatsaber_automapper.evaluation.scorecard "${zips[@]}" --label "$arm" || true
done

echo "=== ALLOCATION VERDICT $(date -Is) ==="
python - <<'PY'
import pathlib, statistics, sys
REPO = pathlib.Path(".").resolve(); sys.path.insert(0, str(REPO / "src"))
from beatsaber_automapper.evaluation import alignment, playfeel, scorecard
CACHE = REPO / "outputs" / "eval_sweep_cache"
ARMS = ["tf_ds048", "tf_g15_ds048", "tf_g1_ds048",
        "tf_hl014_ds048", "tf_hl014_g15_ds048"]
GAMMA = {"tf_ds048": 2.5, "tf_g15_ds048": 1.5, "tf_g1_ds048": 1.0,
         "tf_hl014_ds048": 2.5, "tf_hl014_g15_ds048": 1.5}
# predicted by replaying policies over a probs dump, BEFORE this sweep ran
PREDICTED = {2.5: 0.937, 1.5: 0.941, 1.0: 0.944}

rows = {}
for arm in ARMS:
    zips = sorted(CACHE.glob(f"{arm}__*.zip"))
    if len(zips) < 20:
        continue
    loaded, prec, nps = [], [], []
    for p in zips:
        try:
            r = scorecard._load_any(p)
        except Exception:
            continue
        if not r:
            continue
        loaded.append(r)
        nps.append(playfeel.playfeel_metrics(r[0], bpm=r[1]).metrics["nps"])
        if r[2] is not None:
            prec.append(alignment.alignment_metrics(
                r[0], bpm=r[1], onsets=r[2]).metrics["onset_precision"])
    res = scorecard.score_cohort(loaded, arm)
    rows[arm] = {"axes": {a.name: (a.gap, a.passed) for a in res["axes"]},
                 "npass": sum(1 for a in res["axes"] if a.passed),
                 "viol": res["total_viol"],
                 "prec": statistics.median([x for x in prec if x == x]),
                 "nps": statistics.median([x for x in nps if x == x])}

print(f"\n{'arm':22s}{'gamma':>7s}{'nps':>7s}{'prec':>8s}{'align':>8s}{'rhythm':>8s}"
      f"{'flow':>7s}{'idiom':>7s}{'hrole':>7s}{'pfeel':>7s}{'pass':>7s}")
print("-" * 96)
for arm in ARMS:
    if arm not in rows:
        print(f"{arm:22s}  not scored"); continue
    r = rows[arm]
    g = lambda k: r["axes"][k][0] if k in r["axes"] else float("nan")
    print(f"{arm:22s}{GAMMA[arm]:7.1f}{r['nps']:7.2f}{r['prec']:8.3f}{g('alignment'):8.2f}"
          f"{g('rhythm'):8.2f}{g('flow'):7.2f}{g('idiom'):7.2f}{g('handrole'):7.2f}"
          f"{g('playfeel'):7.2f}{r['npass']:6d}/6")
print("\nhuman: precision 0.930, alignment bar 0.39 (human cohort 0.20)")

print("\n--- PREDICTED (probs-dump replay) vs MEASURED (real decode) ---")
print(f"{'gamma':>7s}{'predicted':>11s}{'measured':>10s}{'delta':>8s}")
for arm in ["tf_ds048", "tf_g15_ds048", "tf_g1_ds048"]:
    if arm not in rows:
        continue
    g = GAMMA[arm]
    pred = PREDICTED.get(g, float("nan"))
    print(f"{g:7.1f}{pred:11.3f}{rows[arm]['prec']:10.3f}{rows[arm]['prec'] - pred:+8.3f}")
print("  The replay ignores NMS, thresholds and the section gate. If the measured")
print("  ORDERING matches the predicted one, the mechanism transfers even when the")
print("  absolute numbers do not — that is what to read here.")

print("\n--- VERDICT ---")
passed = [a for a in rows if rows[a]["axes"].get("alignment", (9, False))[1]]
if passed:
    print(f"  ALIGNMENT PASSES on: {', '.join(passed)}")
    for a in passed:
        print(f"    {a}: {rows[a]['npass']}/6, viol {rows[a]['viol']}")
    print("  ** Still do not promote on one seed. ** 2 of 5 identical seeds passed")
    print("  5/5 last time this project called something a pass. Re-seed 5x, then")
    print("  have Kyle PLAY it -- his ear is what found this defect, and the suite")
    print("  has now been wrong about 'ready' twice.")
else:
    print("  Alignment still fails everywhere. Best:")
    for a in sorted(rows, key=lambda x: rows[x]["axes"].get("alignment", (9,))[0])[:3]:
        print(f"    {a:22s} align {rows[a]['axes']['alignment'][0]:.2f} "
              f"prec {rows[a]['prec']:.3f}  ({rows[a]['npass']}/6)")
    print("  If gamma moved precision in the predicted DIRECTION but not far enough,")
    print("  the remaining gap is the model's ceiling on this axis (top-decile")
    print("  precision was 0.986, so the headroom above the shipped policy is only")
    print("  ~+0.011) and further decode tuning is not the answer.")
PY

echo "=== COMPLETE $(date -Is) ==="
