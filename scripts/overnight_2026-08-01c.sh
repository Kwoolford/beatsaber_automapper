#!/usr/bin/env bash
# MEASURE THE NOISE FLOOR INSTEAD OF ASSUMING IT.
#
# Seeds 0 and 1 of an IDENTICAL configuration (hl014_ds055) scored handrole_gap
# 1.04 and 0.26 -- a spread of 0.78 against the documented floor of +-0.29. The
# floor is wrong by roughly 3x on the axis this project cares most about, and that
# silently invalidates fine-grained rankings: b1_e17_ds055 (1.22) vs b1_e15_ds055
# (1.82) differ by 0.60 and are therefore NOT distinguishable, even though "e17 is
# the best epoch" was concluded from exactly that comparison.
#
# Cause is mechanical: handrole_gap averages |shift| over role_asymmetry AND
# role_swap_rate, and the lead arrangement (a seed) moves swap rate a lot --
# seed0 landed 0.345, seed1 landed 0.479 (human 0.461) for the same setting.
#
# This runs 3 more seeds of the same config, giving 5 samples of an identical
# configuration -> a real per-axis standard deviation. Everything else in the suite
# is compared against that number, so it is worth 3 arms of GPU time.
#
# DoD: a per-axis sd over 5 identical-config seeds, written into TODO.md and
# docs/eval_suite_v2.md as the REPLACEMENT for the assumed floor. Any axis whose sd
# exceeds its current documented floor invalidates every difference smaller than
# ~2sd reported today -- say so explicitly rather than quietly re-baselining.

set -uo pipefail
cd /home/kyle/repos/beatsaber_automapper
source .venv/bin/activate

LOG=logs/overnight/noisefloor_2026-08-01.log
mkdir -p logs/overnight
exec > >(tee -a "$LOG") 2>&1

echo "=== NOISE FLOOR MEASUREMENT START $(date -Is) ==="

ARMS="hl014_seed2_ds055,hl014_seed3_ds055,hl014_seed4_ds055"

python scripts/eval_sweep.py sweep --arms "$ARMS"
echo "=== SWEEP DONE $(date -Is) ==="

echo "=== PER-SEED SCORECARDS ==="
for arm in hl014_ds055 hl014_seed1_ds055 $(echo "$ARMS" | tr ',' ' '); do
  zips=(outputs/eval_sweep_cache/${arm}__*.zip)
  if [ ! -e "${zips[0]}" ]; then echo "-- $arm: NO CACHED MAPS, skipping"; continue; fi
  echo "-- $arm (${#zips[@]} maps)"
  python -m beatsaber_automapper.evaluation.scorecard "${zips[@]}" --label "$arm" || true
done

echo "=== EMPIRICAL PER-AXIS NOISE FLOOR (5 identical-config seeds) ==="
python - <<'PY'
import pathlib, statistics, subprocess, sys, re
REPO = pathlib.Path(".").resolve(); sys.path.insert(0, str(REPO / "src"))
from beatsaber_automapper.evaluation import scorecard
CACHE = REPO / "outputs" / "eval_sweep_cache"
arms = ["hl014_ds055", "hl014_seed1_ds055", "hl014_seed2_ds055",
        "hl014_seed3_ds055", "hl014_seed4_ds055"]
DOCUMENTED = {"flow": 0.03, "rhythm": 0.08, "idiom": 0.09, "handrole": 0.29,
              "playfeel": None}

rows = {}
for a in arms:
    zips = sorted(CACHE.glob(f"{a}__*.zip"))
    if len(zips) < 24:
        print(f"  {a}: only {len(zips)} maps, skipped"); continue
    loaded = []
    for p in zips:
        try:
            r = scorecard._load_any(p)
        except Exception:
            r = None
        if r:
            loaded.append(r)
    res = scorecard.score_cohort(loaded, a)
    rows[a] = {ax.name: (ax.gap, ax.min_spread) for ax in res["axes"]}

if len(rows) < 3:
    print("  not enough seeds finished to estimate a floor"); raise SystemExit(0)

print(f"\n{'axis':10s}{'mean gap':>10s}{'sd':>8s}{'min':>8s}{'max':>8s}"
      f"{'range':>8s}{'documented':>12s}  verdict")
print("-" * 74)
for ax in ["flow", "rhythm", "idiom", "handrole", "playfeel"]:
    vals = [rows[a][ax][0] for a in rows if ax in rows[a]]
    vals = [v for v in vals if v == v]
    if len(vals) < 3:
        continue
    sd = statistics.stdev(vals)
    doc = DOCUMENTED.get(ax)
    if doc is None:
        verdict = "no documented floor"
    elif sd > doc:
        verdict = f"UNDERSTATED {sd/doc:.1f}x -- re-read every delta below {2*sd:.2f}"
    else:
        verdict = "documented floor holds"
    docs = f"{doc:.2f}" if doc is not None else "--"
    print(f"{ax:10s}{statistics.fmean(vals):10.3f}{sd:8.3f}{min(vals):8.3f}"
          f"{max(vals):8.3f}{max(vals)-min(vals):8.3f}{docs:>12s}  {verdict}")

print("\nSPREADS (same 5 seeds) -- the spread bar is 0.35 and several arms today")
print("failed by 0.02, so spread noise matters as much as gap noise:")
print(f"{'axis':10s}{'mean':>10s}{'sd':>8s}{'min':>8s}{'max':>8s}")
for ax in ["flow", "rhythm", "idiom", "handrole", "playfeel"]:
    vals = [rows[a][ax][1] for a in rows if ax in rows[a]]
    vals = [v for v in vals if v == v]
    if len(vals) < 3:
        continue
    print(f"{ax:10s}{statistics.fmean(vals):10.3f}{statistics.stdev(vals):8.3f}"
          f"{min(vals):8.3f}{max(vals):8.3f}")

n_pass = sum(1 for a in rows
             if all(rows[a][ax][0] <= b and rows[a][ax][1] >= 0.35
                    for ax, b in [("flow", 0.50), ("rhythm", 0.70), ("idiom", 1.00),
                                  ("handrole", 2.00), ("playfeel", 1.00)]))
print(f"\n*** {n_pass}/{len(rows)} identical-config seeds achieve a full 5/5 PASS ***")
print("    5/5 -> the configuration passes reliably, not by luck.")
print("    1-2/5 -> hl014's pass is a SEED LOTTERY. Report it that way; do not")
print("             promote, and do not pick the winning seed -- that is fitting")
print("             the bars, the same failure as h_dist saturation.")
PY

echo "=== COMPLETE $(date -Is) ==="
