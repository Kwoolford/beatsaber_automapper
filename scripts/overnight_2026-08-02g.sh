#!/usr/bin/env bash
# CAN A RHYTHMICALLY COHERENT SELECTION THIN WITHOUT KILLING THE PULSE?
# Run at 3 SEEDS, because tonight proved 1 seed decides nothing.
#
# The finding this tests. Re-tuning density to the human note rate costs rhythm,
# and the sub-metrics say exactly why (shift in human MADs):
#
#     arm         nps   pulse_stability  ioi_cond_entropy   gap
#     tf_ds055   4.42        -0.06            +0.47        0.25
#     tf_ds048   3.88        -0.66            +1.20        0.64
#     tf_ds045   3.63        -1.11            +1.61        1.06
#
# Thinning by probability keeps confident notes wherever they fall, which breaks
# the runs that make a rhythm legible. Humans at 3.9 nps have a pulse; we at
# 3.9 nps (thinned from 4.4) do not.
#
# `BEAT_IOI_PRIOR=1.0` switches selection from per-window top-k to _ioi_dp_select,
# sampling from softmax(log p + lam * log P(interval | previous)) over the human
# interval bigram. Built 2026-07-27 for this exact defect, default-off ever since,
# and never judged on a correct tempo grid -- which until today did not exist.
#
# ★ WHY THREE SEEDS. Measured tonight on 5 identical configs: alignment sd 0.092,
# flow 0.116, handrole 0.317, and the six-axis PASS COUNT ranged 1 to 5. A
# single-run comparison against the 5-seed tf_hl014_ds048 baseline would be
# precisely the unresolvable difference this session spent the night documenting.
# The baseline already has 5 seeds; this gives the treatment 3.
#
# VERDICT LOGIC (compare MEANS, and state the sd alongside every number)
#   rhythm mean improves by more than 2sd AND playfeel holds -> coherent thinning
#       is the answer to the density/rhythm tension, and it is the first lever this
#       project has that fixes a defect without trading another one away.
#   rhythm improves but PRECISION drops -> sampling costs alignment (it is
#       deliberately not greedy). That is a real trade to report, not a knob to
#       pick a side on: precision has been pinned at 0.898-0.905 through every
#       other lever tried tonight, so a drop here is the first thing that HAS
#       moved it, and is informative even if unwanted.
#   nothing moves beyond noise -> the IOI prior does not survive a correct grid
#       either. Record the negative and stop: with tonight's floors, the honest
#       next step is fixing the variance, not finding another lever.

set -uo pipefail
cd /home/kyle/repos/beatsaber_automapper
source .venv/bin/activate

LOG=logs/overnight/ioiprior_2026-08-02.log
mkdir -p logs/overnight
exec > >(tee -a "$LOG") 2>&1

echo "=== IOI-PRIOR (3 seeds) START $(date -Is) ==="

ARMS="tf_hl014_ioi1_ds048,tf_hl014_ioi1_ds048_s1,tf_hl014_ioi1_ds048_s2"
python scripts/eval_sweep.py sweep --arms "$ARMS"
echo "=== SWEEP DONE $(date -Is) ==="

echo "=== VERDICT: MEANS AND SPREADS, NOT SINGLE RUNS $(date -Is) ==="
python - <<'PY'
import pathlib, statistics, sys
REPO = pathlib.Path(".").resolve(); sys.path.insert(0, str(REPO / "src"))
from beatsaber_automapper.evaluation import alignment, playfeel, scorecard
CACHE = REPO / "outputs" / "eval_sweep_cache"
BASE = ["tf_hl014_ds048"] + [f"tf_hl014_ds048_s{i}" for i in (1, 2, 3, 4)]
TREAT = ["tf_hl014_ioi1_ds048"] + [f"tf_hl014_ioi1_ds048_s{i}" for i in (1, 2)]
AXES = ["alignment", "rhythm", "flow", "idiom", "handrole", "playfeel"]

def score(arm):
    zips = sorted(CACHE.glob(f"{arm}__*.zip"))
    if len(zips) < 20:
        return None
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
    return {"axes": {a.name: a.gap for a in res["axes"]},
            "npass": sum(1 for a in res["axes"] if a.passed),
            "viol": res["total_viol"],
            "prec": statistics.median([x for x in prec if x == x]),
            "nps": statistics.median([x for x in nps if x == x])}

groups = {}
for name, arms in (("baseline (top-k)", BASE), ("IOI prior", TREAT)):
    rows = [r for r in (score(a) for a in arms) if r]
    if rows:
        groups[name] = rows
    print(f"{name}: {len(rows)} seeds scored")

if len(groups) < 2:
    print("\nnot enough seeds on both sides to compare"); raise SystemExit(0)

def ms(rows, key):
    v = [r["axes"][key] for r in rows if key in r["axes"] and r["axes"][key] == r["axes"][key]]
    return (statistics.fmean(v), statistics.stdev(v) if len(v) > 1 else 0.0, len(v))

print(f"\n{'axis':12s}" + "".join(f"{g:>26s}" for g in groups) + f"{'delta':>10s}{'resolvable?':>14s}")
print("-" * 90)
for ax in AXES:
    cells, means, sds = "", [], []
    for g, rows in groups.items():
        m, sd, n = ms(rows, ax)
        means.append(m); sds.append(sd)
        cells += f"{m:>18.3f} ±{sd:<6.3f}"
    d = means[1] - means[0]
    pooled = max((sds[0] ** 2 + sds[1] ** 2) ** 0.5, 1e-9)
    res = "yes" if abs(d) > 2 * pooled else "NO (noise)"
    print(f"{ax:12s}{cells}{d:+10.3f}{res:>14s}")

for label, key in (("precision", "prec"), ("nps", "nps"), ("axes passed", "npass")):
    vals = [[r[key] for r in rows] for rows in groups.values()]
    print(f"\n{label:12s}" + "".join(
        f"{statistics.fmean(v):>18.3f} ±{(statistics.stdev(v) if len(v) > 1 else 0.0):<6.3f}"
        for v in vals))

print("\n--- READ ---")
print("  Only differences marked 'yes' are real. Everything else is this pipeline's")
print("  seed variance, which is the binding constraint on the method until it is")
print("  fixed. And whatever this table says, the decisive test is unchanged:")
print("  Kyle plays outputs/kyle_review_2026-08-02/. His ear found the defect that")
print("  five axes could not see.")
PY

echo "=== COMPLETE $(date -Is) ==="
