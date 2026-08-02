#!/usr/bin/env bash
# HOW MUCH OF THE ORACLE'S GAIN DOES A REAL TEMPO ESTIMATOR REACH?
#
# The oracle sweep (overnight_2026-08-02b.sh) settled attribution across all 24
# songs. Handing the generator the human-declared tempo:
#
#     arm                prec   mad_ms   align_gap
#     ds055             0.756     17.4        5.41
#     obpm_ds055        0.887     10.7        0.80
#     human             0.930     10.35       0.20
#
# Alignment improves 6.8x and the timing SCATTER lands essentially on the human
# value. The defect Kyle hears is the tempo the note grid is built on.
#
# `BEAT_TEMPO_FIT=1` is the shippable version: it fits tempo AND phase to the
# per-stem onsets (the same onsets A8 scores against) and reads no human map. On
# the 23 eval songs it recovers the human-declared bpm exactly on 21, where the
# current detector manages 1. Smoke-tested on 1f767: 161.50 -> 159.997 (R=0.344),
# against an oracle value of 160.0.
#
# ARMS: tf_ds055, tf_hl014_ds055, tf_prod -- each against BOTH its plain control
# (ds055/hl014_ds055/prod, cached) and its oracle ceiling (obpm_*, cached).
#
# VERDICT LOGIC
#   tf_* lands near obpm_*  -> the estimator captures the available gain and
#       BEAT_TEMPO_FIT should become the default. This is the first change in the
#       project's history aimed at the defect Kyle actually reports.
#   tf_* lands between      -> partial. Read the per-song table: the 2 known
#       metrical-level ties (1fbda, 1fbfb) should be the losses. If OTHER songs
#       lose, the fitter is unstable on real audio and needs the R gate tightened.
#   tf_* lands near ds055   -> the fitter does not survive contact with the
#       generation path. Check the log for "fit UNTRUSTED" lines first.
#
# ★ EXPECT THE OTHER FIVE AXES TO REGRESS, AND DO NOT READ THAT AS FAILURE.
# The oracle arms dropped from 4/6 and 5/6 to 1/6 and 0/6: playfeel 0.74 -> 1.38,
# flow 0.30 -> 0.54. The cause is mechanical -- a corrected tempo changes how many
# 1/4-beat slots exist per second, so note counts move (1f333: 838 -> 1509), and
# EVERY density/flow lever in this repo was tuned against the wrong grid.
# Re-tuning BEAT_DIFFICULTY_SCALE on the corrected grid is the next job, not a
# reason to reject the fix. A map that is on the beat and too dense is a tuning
# problem; a map that is off the beat is the problem we have been failing to see.

set -uo pipefail
cd /home/kyle/repos/beatsaber_automapper
source .venv/bin/activate

LOG=logs/overnight/tempofit_2026-08-02.log
mkdir -p logs/overnight
exec > >(tee -a "$LOG") 2>&1

echo "=== TEMPO-FIT SWEEP START $(date -Is) ==="

ARMS="tf_ds055,tf_hl014_ds055,tf_prod"
python scripts/eval_sweep.py sweep --arms "$ARMS"
echo "=== SWEEP DONE $(date -Is) ==="

echo "=== HOW OFTEN DID THE FITTER FIRE, AND TO WHAT? ==="
grep -h "BEAT_TEMPO_FIT" "$LOG" | tail -80 || true

echo "=== SIX-AXIS SCORECARDS ==="
for arm in ds055 obpm_ds055 tf_ds055 hl014_ds055 obpm_hl014_ds055 tf_hl014_ds055 \
           prod obpm_prod tf_prod; do
  zips=(outputs/eval_sweep_cache/${arm}__*.zip)
  if [ ! -e "${zips[0]}" ]; then echo "-- $arm: NO CACHED MAPS, skipping"; continue; fi
  echo "-- $arm (${#zips[@]} maps)"
  python -m beatsaber_automapper.evaluation.scorecard "${zips[@]}" --label "$arm" || true
done

echo "=== TEMPO-FIT VERDICT $(date -Is) ==="
python - <<'PY'
import json, pathlib, statistics, sys
REPO = pathlib.Path(".").resolve(); sys.path.insert(0, str(REPO / "src"))
from beatsaber_automapper.evaluation import alignment, scorecard
CACHE = REPO / "outputs" / "eval_sweep_cache"
TRUE = json.loads((REPO / "outputs" / "true_bpm_eval_songset.json").read_text())
HUMAN_PREC, HUMAN_MAD, HUMAN_GAP = 0.930, 10.35, 0.20
FAMILIES = [("ds055", "obpm_ds055", "tf_ds055"),
            ("hl014_ds055", "obpm_hl014_ds055", "tf_hl014_ds055"),
            ("prod", "obpm_prod", "tf_prod")]

def load(arm):
    per, loaded = {}, []
    for p in sorted(CACHE.glob(f"{arm}__*.zip")):
        try:
            r = scorecard._load_any(p)
        except Exception:
            continue
        if not r:
            continue
        loaded.append(r)
        if r[2] is not None:
            per[p.stem.split("__")[-1]] = (
                alignment.alignment_metrics(r[0], bpm=r[1], onsets=r[2]).metrics, r[1])
    if len(loaded) < 20:
        return None
    res = scorecard.score_cohort(loaded, arm)
    return {"per": per, "axes": {a.name: (a.gap, a.passed) for a in res["axes"]},
            "npass": sum(1 for a in res["axes"] if a.passed), "viol": res["total_viol"]}

rows = {}
for fam in FAMILIES:
    for arm in fam:
        if arm not in rows:
            d = load(arm)
            if d:
                rows[arm] = d

print(f"\n{'arm':22s}{'prec':>8s}{'mad_ms':>9s}{'align':>8s}{'rhythm':>8s}"
      f"{'flow':>7s}{'idiom':>7s}{'hrole':>7s}{'pfeel':>7s}{'pass':>7s}")
print("-" * 96)
for fam in FAMILIES:
    for arm in fam:
        if arm not in rows:
            print(f"{arm:22s}  not scored"); continue
        d = rows[arm]
        prec = statistics.median(m["onset_precision"] for m, _ in d["per"].values())
        mad = statistics.median(m["offset_mad_ms"] for m, _ in d["per"].values())
        g = lambda k: d["axes"][k][0] if k in d["axes"] else float("nan")
        print(f"{arm:22s}{prec:8.3f}{mad:9.1f}{g('alignment'):8.2f}{g('rhythm'):8.2f}"
              f"{g('flow'):7.2f}{g('idiom'):7.2f}{g('handrole'):7.2f}{g('playfeel'):7.2f}"
              f"{d['npass']:6d}/6")
    print()
print(f"human reference: precision {HUMAN_PREC:.3f}, scatter {HUMAN_MAD:.1f}ms, "
      f"alignment_gap {HUMAN_GAP:.2f} (bar 0.39)")

print("\n--- HOW MUCH OF THE ORACLE CEILING DID THE REAL ESTIMATOR REACH? ---")
for base, orc, tf in FAMILIES:
    if not all(a in rows for a in (base, orc, tf)):
        continue
    def pr(a):
        return statistics.median(m["onset_precision"] for m, _ in rows[a]["per"].values())
    b, o, t = pr(base), pr(orc), pr(tf)
    frac = (t - b) / (o - b) if abs(o - b) > 1e-9 else float("nan")
    print(f"  {base:16s} {b:.3f} -> oracle {o:.3f} -> fitted {t:.3f}   "
          f"captured {frac * 100:5.1f}% of the available gain")

print("\n--- PER-SONG: WHICH SONGS DID THE FITTER GET WRONG? ---")
print("(the two known metrical-level ties are 1fbda and 1fbfb; anything else is new)")
base, orc, tf = FAMILIES[0]
if all(a in rows for a in (base, orc, tf)):
    print(f"{'song':10s}{'true':>8s}{'tf_bpm':>9s}{'err%':>8s}"
          f"{'prec_base':>11s}{'prec_orc':>10s}{'prec_tf':>9s}")
    for sid in sorted(rows[tf]["per"]):
        tb = TRUE.get(sid)
        mt, bt = rows[tf]["per"][sid]
        mb = rows[base]["per"].get(sid, ({}, 0))[0]
        mo = rows[orc]["per"].get(sid, ({}, 0))[0]
        err = (bt - tb) / tb * 100 if tb else float("nan")
        flag = "  <-- WRONG LEVEL" if tb and abs(err) > 5 else ""
        print(f"{sid[:9]:10s}{tb or 0:8.1f}{bt:9.2f}{err:+8.2f}"
              f"{mb.get('onset_precision', float('nan')):11.3f}"
              f"{mo.get('onset_precision', float('nan')):10.3f}"
              f"{mt.get('onset_precision', float('nan')):9.3f}{flag}")

print("\n--- VERDICT ---")
if all(a in rows for a in FAMILIES[0]):
    base, orc, tf = FAMILIES[0]
    pr = lambda a: statistics.median(m["onset_precision"] for m, _ in rows[a]["per"].values())
    b, o, t = pr(base), pr(orc), pr(tf)
    frac = (t - b) / (o - b) if abs(o - b) > 1e-9 else 0.0
    ga = rows[tf]["axes"].get("alignment", (float("nan"),))[0]
    print(f"  alignment_gap {rows[base]['axes']['alignment'][0]:.2f} -> {ga:.2f} "
          f"(oracle {rows[orc]['axes']['alignment'][0]:.2f}, bar 0.39, human 0.20)")
    if frac >= 0.75:
        print("  => THE FIX WORKS WITHOUT AN ORACLE. Make BEAT_TEMPO_FIT the default,")
        print("     then RE-TUNE the density/flow levers on the corrected grid — they")
        print("     were all fitted against a grid that was wrong on 20 of 21 songs.")
    elif frac >= 0.4:
        print("  => partial capture. Check the per-song table: if the losses are the")
        print("     two known level ties, tighten the level choice; if they are new")
        print("     songs, the fitter is unstable on real audio.")
    else:
        print("  => the estimator does not survive the generation path. Grep the log")
        print("     for 'fit UNTRUSTED' before concluding anything about the method.")
PY

echo "=== COMPLETE $(date -Is) ==="
