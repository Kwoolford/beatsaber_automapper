#!/usr/bin/env bash
# M-E — STRUCTURE-CONDITIONED DECODE, three arms against the existing control.
#
# THE QUESTION. Every lever this project has tried changed how a slot is scored, or
# how well. None moved a masterpiece axis, and C1 now says why from six directions:
# every slot is decided on its own, so no per-slot evidence can make bar 74 relate to
# bar 42. M-E copies a decision instead of computing a better one — when the AUDIO
# says this bar is a return of an earlier one, reuse the map already generated there.
#
# ARMS (control = outputs/wide_cohort, the 149-song prod cohort, seed 0, already built)
#   me_z20     place:0.6:4:1.5:2.0   copy position+direction, looser distinctiveness
#   me_z25     place:0.6:4:1.5:2.5   copy position+direction, stricter
#   me_full25  full:0.6:4:1.5:2.5    also copy the bar's RHYTHM (the risky arm)
# All 149 songs, same audio, same seed as the control ⇒ paired, differing in one thing.
#
# ★WHY THE `place` ARMS ARE UNUSUALLY CLEAN. They move no note in time and add or
# remove none, verified end-to-end on 1fccd (566 notes, times byte-identical, 25.4% of
# notes re-placed). So alignment, rhythm (A2), density, nps and onset precision CANNOT
# move — not "did not move", cannot. Anything that does move is position or direction.
#
# 🔴THE VERDICT LOGIC, PRE-REGISTERED BEFORE THE FIRST MAP IS GENERATED:
#   `harm_place` rising is a MANIPULATION CHECK, not a win. This lever copies placement
#   on musical repeats and that axis scores placement reuse on musical repeats; citing
#   it as quality would be fitting the metric. It answers "did the lever fire", nothing
#   more. What would make this worth Kyle's ear:
#     PASS   harm_place rises AND the six-axis suite is unmoved AND hard_rate does not
#            rise (reachability — a lever can pass every axis and still carry a defect
#            no axis measures; BEAT_ONSET_EVIDENCE did exactly that) AND follow_* does
#            not fall (a copied bar is only right if the repeat really is a repeat).
#     PIVOT  harm_place rises but flow/idiom/playfeel or hard_rate degrade resolvably
#            ⇒ the copy is fighting the local music; try stricter z or shorter spans.
#     DEAD   harm_place does not move at n=149 ⇒ postprocess is eating the copy
#            (check the parity-fix rewrite rate) or the repeats are not where we think.
#   ⚠️And Kyle, 2026-08-10: "the metrics still don't capture the full picture." A green
#   table here buys a listening session, not a promotion.
set -u
cd "$(dirname "$0")/.." || exit 1
LOG=logs/overnight/me_2026-08-10.log
mkdir -p logs/overnight outputs/me_2026-08-10
exec > >(tee -a "$LOG") 2>&1
PY=.venv/bin/python
echo "=== M-E structure-conditioned decode — $(date) ==="

CTRL=outputs/wide_cohort
declare -A ARMS=(
  [me_z20]="place:0.6:4:1.5:2.0"
  [me_z25]="place:0.6:4:1.5:2.5"
  [me_full25]="full:0.6:4:1.5:2.5"
)
ORDER=(me_z20 me_z25 me_full25)

for arm in "${ORDER[@]}"; do
  spec="${ARMS[$arm]}"
  echo ""; echo "--- GENERATE $arm  (BEAT_STRUCTURE_REUSE=$spec) --- $(date +%H:%M)"
  $PY scripts/build_wide_cohort.py --n 150 --seed 0 --variant prod \
      --tag "$arm" --env "BEAT_STRUCTURE_REUSE=$spec"
done

for arm in "${ORDER[@]}"; do
  d="outputs/wide_cohort_prod_${arm}"
  n=$(ls "$d"/*.zip 2>/dev/null | wc -l)
  echo ""; echo "--- EVAL $arm ($n maps) --- $(date +%H:%M)"
  [ "$n" -lt 100 ] && { echo "  SKIP: only $n maps"; continue; }

  $PY scripts/masterpiece_report.py --arm "$arm" --wide --wide-dir "$d" \
      --vs prod --vs-wide-dir "$CTRL" \
      --json "outputs/me_2026-08-10/masterpiece_${arm}.json"

  echo "  -- six-axis suite --"
  $PY -m beatsaber_automapper.evaluation.scorecard "$d"/*.zip --label "$arm"

  echo "  -- reachability guard (K2) --"
  $PY scripts/eval_reachability.py --maps "$d/*.zip" --label "$arm" \
      --maps "$CTRL/*.zip" --label control \
      --json "outputs/me_2026-08-10/reach_${arm}.json"
done

echo ""; echo "--- CONTROL six-axis (for the comparison above) --- $(date +%H:%M)"
$PY -m beatsaber_automapper.evaluation.scorecard "$CTRL"/*.zip --label control

echo ""; echo "=== SUMMARY — $(date) ==="
$PY - <<'PY'
import json, pathlib, glob
OUT = pathlib.Path("outputs/me_2026-08-10")
KEY = ["harm_place", "rhy_rhythm", "harm_rhythm", "follow_vocals", "follow_mean",
       "timb_rhythm", "double_share"]
print("\nPAIRED vs the prod control, n=149. Read `harm_place` as a manipulation")
print("check (did the lever fire), NOT as evidence the map got better.\n")
for f in sorted(OUT.glob("masterpiece_*.json")):
    arm = f.stem.replace("masterpiece_", "")
    try:
        d = json.loads(f.read_text())
    except Exception as e:
        print(f"{arm}: unreadable ({e})"); continue
    print(f"--- {arm} ---")
    print(json.dumps(d, indent=1)[:2000])
for f in sorted(OUT.glob("reach_*.json")):
    print(f"\n--- reachability {f.stem} ---")
    print(f.read_text()[:1200])
print("""
HOW TO READ THIS NEXT SESSION
  1. Did harm_place move at all? No  -> DEAD: postprocess is eating the copy, or the
     detected repeats are not where the music actually returns. Check the generator
     log line "N/M bars are musical repeats" before touching the metric.
  2. It moved -> check the six-axis table and hard_rate ABOVE for a resolvable
     regression. The place arms cannot have moved alignment/rhythm/nps/precision; if
     one of those DID move, the time-neutrality property is broken and everything
     here is void -- that is the first thing to verify, not the last.
  3. Nothing regressed -> build the review set for Kyle (BEFORE/AFTER on 1f767 /
     1f913 / 1f333 / 1f8d6, his standing four) and put the structure PNG beside it.
     His ear decides. The axes do not: M-F showed they rank Fallen Kingdom 2nd best
     and Hunger 5th worst, which is the exact reverse of his verdicts.
""")
PY
echo "COMPLETE $(date)"
