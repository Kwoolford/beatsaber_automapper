#!/usr/bin/env bash
# RE-TUNE DENSITY ON THE CORRECTED GRID.
#
# BEAT_TEMPO_FIT works (overnight_2026-08-02c.sh):
#     arm            prec   mad_ms   align   flow  idiom  pfeel   pass
#     ds055         0.756     17.4    5.41   0.30   0.52   0.74    4/6
#     obpm_ds055    0.887     10.7    0.80   0.54   0.75   1.38    1/6   (oracle)
#     tf_ds055      0.902     10.2    0.49   0.64   0.77   1.02    2/6
#     human         0.930     10.3    0.20                         bar 0.39
#
# Alignment improves 11x and the scatter lands ON the human value. But the arm
# went 4/6 -> 2/6, because a corrected tempo changes how many 1/4-beat slots exist
# per second and EVERY density lever here was fitted against the wrong grid.
# Measured, not guessed: obpm_ds055 nps 4.42 vs human 3.909 (+0.88 human MADs),
# so the scale landing on the human median is 0.55 * 3.909/4.42 ~= 0.486.
#
# ARMS bracket that: tf_ds045 / tf_ds048 / tf_ds052, plus tf_hl014_ds048 on the
# best-known config. Controls tf_ds055 and ds055 are cached.
#
# VERDICT LOGIC
#   an arm reaches 5/6 or 6/6  -> the tempo fix PLUS its re-tune is the new best
#       configuration in the project's history, and the first one aimed at the
#       defect Kyle reports. Do not promote on one seed: the measured noise floor
#       (5 identical seeds, 2026-08-01) is flow 0.099 / rhythm 0.087 / idiom 0.084
#       / handrole 0.303 / playfeel 0.048, and only 2 of 5 identical seeds passed
#       5/5 last time. Re-seed the winner BEFORE believing it.
#   playfeel closes but flow/idiom do not -> density was not the whole story; the
#       corrected grid changed the note SPACING too, not just the count.
#   nothing improves -> the regression is not density. Look at flow travel first
#       (0.30 -> 0.64 is well beyond its 0.20 two-sigma floor).
#
# ★ ALIGNMENT IS STILL A FAIL AT 0.49 (bar 0.39, human 0.20). Do not let a good
# playfeel number distract from that -- the remaining gap is grid PHASE plus slot
# selection, and is documented in TODO.md §8 with the human control that separates
# the two.

set -uo pipefail
cd /home/kyle/repos/beatsaber_automapper
source .venv/bin/activate

LOG=logs/overnight/density_retune_2026-08-02.log
mkdir -p logs/overnight
exec > >(tee -a "$LOG") 2>&1

echo "=== DENSITY RE-TUNE ON THE CORRECTED GRID $(date -Is) ==="

ARMS="tf_ds045,tf_ds048,tf_ds052,tf_hl014_ds048"
python scripts/eval_sweep.py sweep --arms "$ARMS"
echo "=== SWEEP DONE $(date -Is) ==="

echo "=== SIX-AXIS SCORECARDS ==="
for arm in ds055 tf_ds055 $(echo "$ARMS" | tr ',' ' ') hl014_ds055 tf_hl014_ds055; do
  zips=(outputs/eval_sweep_cache/${arm}__*.zip)
  if [ ! -e "${zips[0]}" ]; then echo "-- $arm: NO CACHED MAPS, skipping"; continue; fi
  echo "-- $arm (${#zips[@]} maps)"
  python -m beatsaber_automapper.evaluation.scorecard "${zips[@]}" --label "$arm" || true
done

echo "=== RE-TUNE VERDICT $(date -Is) ==="
python - <<'PY'
import json, pathlib, statistics, sys
REPO = pathlib.Path(".").resolve(); sys.path.insert(0, str(REPO / "src"))
from beatsaber_automapper.evaluation import alignment, playfeel, scorecard
CACHE = REPO / "outputs" / "eval_sweep_cache"
ref = json.loads((REPO / "outputs" / "playfeel_human_reference.json").read_text())
HNPS = ref["nps"]["median"]
ARMS = ["ds055", "tf_ds055", "tf_ds045", "tf_ds048", "tf_ds052",
        "hl014_ds055", "tf_hl014_ds055", "tf_hl014_ds048"]
# 2sd of the MEASURED noise floor (5 identical seeds, 2026-08-01)
FLOOR2SD = {"flow": 0.20, "rhythm": 0.17, "idiom": 0.17, "handrole": 0.61,
            "playfeel": 0.10, "alignment": None}

rows = {}
for arm in ARMS:
    zips = sorted(CACHE.glob(f"{arm}__*.zip"))
    if len(zips) < 20:
        continue
    loaded, per = [], []
    for p in zips:
        try:
            r = scorecard._load_any(p)
        except Exception:
            continue
        if not r:
            continue
        loaded.append(r)
        per.append(playfeel.playfeel_metrics(r[0], bpm=r[1]).metrics)
        if r[2] is not None:
            m = alignment.alignment_metrics(r[0], bpm=r[1], onsets=r[2]).metrics
            per[-1]["_prec"] = m["onset_precision"]
    res = scorecard.score_cohort(loaded, arm)
    nps = [d["nps"] for d in per if d.get("nps") == d.get("nps")]
    prc = [d["_prec"] for d in per if d.get("_prec") == d.get("_prec")]
    rows[arm] = {"axes": {a.name: (a.gap, a.passed) for a in res["axes"]},
                 "npass": sum(1 for a in res["axes"] if a.passed),
                 "viol": res["total_viol"],
                 "nps": statistics.median(nps) if nps else float("nan"),
                 "prec": statistics.median(prc) if prc else float("nan")}

print(f"\n{'arm':20s}{'nps':>7s}{'prec':>8s}{'align':>8s}{'rhythm':>8s}{'flow':>7s}"
      f"{'idiom':>7s}{'hrole':>7s}{'pfeel':>7s}{'viol':>6s}{'pass':>7s}")
print("-" * 92)
for arm in ARMS:
    if arm not in rows:
        print(f"{arm:20s}  not scored"); continue
    r = rows[arm]
    g = lambda k: r["axes"][k][0] if k in r["axes"] else float("nan")
    print(f"{arm:20s}{r['nps']:7.2f}{r['prec']:8.3f}{g('alignment'):8.2f}"
          f"{g('rhythm'):8.2f}{g('flow'):7.2f}{g('idiom'):7.2f}{g('handrole'):7.2f}"
          f"{g('playfeel'):7.2f}{r['viol'] if r['viol'] is not None else -1:6d}"
          f"{r['npass']:6d}/6")
print(f"\nhuman: nps {HNPS:.2f}, precision 0.930, alignment bar 0.39 (human 0.20)")

best = [a for a in rows if rows[a]["npass"] >= 5]
print("\n--- VERDICT ---")
if best:
    b = max(best, key=lambda a: rows[a]["npass"])
    print(f"  BEST: {b} at {rows[b]['npass']}/6")
    print("  ** DO NOT PROMOTE ON THIS RUN. ** The measured noise floor (5 identical")
    print("  seeds) is flow 0.099 / rhythm 0.087 / idiom 0.084 / handrole 0.303 /")
    print("  playfeel 0.048, and only 2 of 5 identical seeds passed 5/5 last time.")
    print("  Re-seed this arm 5x first; a pass within noise of the bar is a lottery")
    print("  ticket. Then have Kyle PLAY it -- the suite has been wrong before, and")
    print("  it is his ear that found the defect this whole session chased.")
else:
    print("  No arm reaches 5/6.")
    for arm in [a for a in rows if a.startswith("tf_ds")]:
        r = rows[arm]
        fails = [f"{k} {v[0]:.2f}" for k, v in r["axes"].items() if not v[1]]
        print(f"  {arm:16s} nps {r['nps']:.2f} (human {HNPS:.2f})  fails: {', '.join(fails)}")
    print("\n  If playfeel closed but flow/idiom did not, the corrected grid changed")
    print("  note SPACING and not just note COUNT -- a different problem from density.")

print("\n--- REGRESSION vs the un-retuned tempo-fit arm ---")
for base, arms in [("tf_ds055", ["tf_ds045", "tf_ds048", "tf_ds052"]),
                   ("tf_hl014_ds055", ["tf_hl014_ds048"])]:
    if base not in rows:
        continue
    for arm in arms:
        if arm not in rows:
            continue
        bad = []
        for ax, f in FLOOR2SD.items():
            if f is None or ax not in rows[arm]["axes"]:
                continue
            d = rows[arm]["axes"][ax][0] - rows[base]["axes"][ax][0]
            if d > f:
                bad.append(f"{ax} +{d:.2f} (>{f})")
        print(f"  {arm:18s} vs {base:16s} "
              f"{'REGRESSES: ' + ', '.join(bad) if bad else 'nothing beyond noise'}")
PY

echo "=== COMPLETE $(date -Is) ==="
