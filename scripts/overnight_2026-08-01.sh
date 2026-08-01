#!/usr/bin/env bash
# A6 / HAND LEAD: close the last failing axis without deleting notes.
#
# Background (2026-08-01, scripts/eval_spread_breakdown.py). scorecard.py reports
# only `min_spread` -- the MINIMUM spread over an axis's sequence keys -- so it
# never says which sub-metric collapsed. Decomposing it showed the handrole failure
# is ONE key: `role_asymmetry`, human 0.115 vs ours 0.026-0.046, cohort spread 0.27
# against a 0.35 bar. The human cohort passes every axis (min_spread 0.71-1.06), so
# the bar is sound and this is a real collapse, not a metric artifact.
#
# Upstream of it: we put both hands on the same slot 84-94% of the time against a
# human 23%. With both hands on nearly every slot a per-window lead is
# arithmetically impossible, which is why the old BEAT_HAND_ROLE had to DELETE ~24%
# of the notes to manufacture asymmetry (and wrecked rhythm doing it).
#
# BEAT_HAND_LEAD biases each hand's per-window budget SHARE while holding its TOTAL
# fixed -- balanced globally, lopsided locally, note count preserved. Smoke test on
# 1f8a3: OFF 0.0429 (= the ds055 baseline), hl=0.30 -> 0.247, notes 678 -> 687.
# Realised asymmetry is ~0.82x the requested share, hence the 0.10-0.25 grid.
#
# ARMS (6, all on ds055 -- prod density is a tier too dense to judge at):
#   hl010/hl014/hl018/hl025   -- the asymmetry grid, hl014 predicted to land on 0.115
#   hl014_sw07                -- + swap 0.70, because the smoke test dropped
#                                role_swap_rate 0.436 -> 0.282 (human 0.461) and
#                                handrole_gap averages |shift| over BOTH keys
#   hl014_ar_xy               -- does the lead compose with the direction lever?
# CONTROLS: `ds055` and `ar_xy_ds055` are already cached from prior sweeps.
#
# DoD (from TODO.md):
#   role_asymmetry >= 0.08, handrole spread >= 0.35 with gap still <= 2.00, and no
#   axis ds055 already passes regressing beyond its noise floor
#   (flow +-0.03 / rhythm +-0.08 / idiom +-0.09 / handrole +-0.29). Parity
#   violations MUST stay 0 -- that is what killed the hand-offset lever above ho03.

set -uo pipefail
cd /home/kyle/repos/beatsaber_automapper
source .venv/bin/activate

LOG=logs/overnight/handlead_2026-08-01.log
mkdir -p logs/overnight
exec > >(tee -a "$LOG") 2>&1

echo "=== A6 HAND-LEAD SWEEP START $(date -Is) ==="

ARMS="hl010_ds055,hl014_ds055,hl018_ds055,hl025_ds055,hl014_sw07_ds055,hl014_ar_xy_ds055"

python scripts/eval_sweep.py sweep --arms "$ARMS"
echo "=== SWEEP DONE $(date -Is) ==="

# eval_sweep's own rhythm table silently prints nan (never wires rhythm records
# into its per-song dict) -- scorecard.py is the trustworthy 5-axis path.
echo "=== 5-AXIS SCORECARD ==="
for arm in ds055 ar_xy_ds055 $(echo "$ARMS" | tr ',' ' '); do
  zips=(outputs/eval_sweep_cache/${arm}__*.zip)
  if [ ! -e "${zips[0]}" ]; then
    echo "-- $arm: NO CACHED MAPS, skipping"
    continue
  fi
  echo "-- $arm (${#zips[@]} maps)"
  python -m beatsaber_automapper.evaluation.scorecard "${zips[@]}" --label "$arm" || true
done

# The axis verdict alone cannot say WHY an arm moved. Print the per-sub-metric
# decomposition too, so a handrole change is attributable to role_asymmetry rather
# than assumed to be.
echo "=== SUB-METRIC BREAKDOWN (which key actually moved) ==="
python scripts/eval_spread_breakdown.py \
  --arms "ds055,ar_xy_ds055,$ARMS" --human 24 || true

echo "=== RAW role_asymmetry + NOTE COUNT PER ARM ==="
python - <<'PY'
import pathlib, statistics, sys
REPO = pathlib.Path(".").resolve(); sys.path.insert(0, str(REPO / "src"))
from beatsaber_automapper.evaluation import scorecard, handrole
CACHE = REPO / "outputs" / "eval_sweep_cache"
arms = ["ds055", "ar_xy_ds055", "hl010_ds055", "hl014_ds055", "hl018_ds055",
        "hl025_ds055", "hl014_sw07_ds055", "hl014_ar_xy_ds055"]
print(f"{'arm':22s}{'asym':>8s}{'swap':>8s}{'notes':>9s}{'dbl':>8s}  (human 0.115 / 0.461 / - / 0.231)")
for a in arms:
    asy, swp, nn, db = [], [], [], []
    for p in sorted(CACHE.glob(f"{a}__*.zip")):
        try:
            r = scorecard._load_any(p)
        except Exception:
            r = None
        if not r:
            continue
        bm, _ = r
        m = handrole.handrole_metrics(bm).metrics
        if m.get("role_asymmetry") == m.get("role_asymmetry"):
            asy.append(m["role_asymmetry"]); swp.append(m["role_swap_rate"])
        nn.append(len(bm.color_notes))
        byb = {}
        for x in bm.color_notes:
            byb.setdefault(round(x.beat * 16), set()).add(x.color)
        db.append(sum(2 for _b, c in byb.items() if len(c) > 1) / max(len(bm.color_notes), 1))
    if not asy:
        print(f"{a:22s}  (no cached maps)"); continue
    print(f"{a:22s}{statistics.median(asy):8.4f}{statistics.median(swp):8.3f}"
          f"{statistics.median(nn):9.0f}{statistics.median(db):8.3f}")
PY

echo "=== VERDICT $(date -Is) ==="
python - <<'PY'
print("""
DoD: role_asymmetry >= 0.08 AND handrole spread >= 0.35 AND handrole gap <= 2.00,
with NO axis ds055 already passes regressing past its noise floor
(flow +-0.03 / rhythm +-0.08 / idiom +-0.09 / handrole +-0.29), parity viol = 0.

READ IT LIKE THIS:
 (a) an arm clears handrole (gap AND spread) and holds flow/rhythm/idiom/playfeel
     -> FIRST 5/5 CANDIDATE. Render it and send to Kyle. Do NOT promote to
        generate.py defaults on the scorecard alone -- the 2026-07-27 review found
        a lever that scored well on paper and was unplayable in practice.
 (b) role_asymmetry hits the human 0.115 but handrole SPREAD stays < 0.35
     -> the lever sets a CONSTANT asymmetry, so every song gets the same lead
        pattern. Fix is to modulate the target per song (e.g. scale it by that
        song's own energy variance) rather than to push the value harder.
 (c) asymmetry improves but rhythm/idiom regress past the noise floor
     -> the budget shift is stealing notes from where they belong musically.
        Look at the double-share column before concluding anything: if it barely
        moved, the lead is cosmetic and the real lever is still the double rate.
 (d) role_swap_rate craters while asymmetry rises (watch hl014 vs hl014_sw07)
     -> lead blocks outlive handrole.py's 8-beat window; BEAT_HAND_LEAD_SWAP is
        the handle, and sw07 should beat plain hl014 on handrole_gap.
""")
PY
echo "=== COMPLETE $(date -Is) ==="
