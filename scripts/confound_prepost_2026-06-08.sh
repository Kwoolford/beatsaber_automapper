#!/usr/bin/env bash
# TASK-3 CONFOUND TEST (2026-06-08)
# ---------------------------------------------------------------------------
# The 06-07 A/B came back NULL end-to-end (contour−control delta < +0.05 at
# both difficulties; ExpertPlus contour even HURT). The runner's verdict says:
# before killing TASK 3, rule out the postprocess parity-fix confound — it
# rewrites ~48% of swing directions and can erase the model's contour choices.
#
# This re-runs ONLY generate + contour-follow, dumping the PRE-postprocess
# beatmap via BS_PREPOST_OUT, and scores contour-follow on that raw stream.
# No training. Arms reuse the already-trained ckpts from the 06-07 run.
#
# Verdict logic:
#   pre-post delta A−B >= +0.05 at BOTH diffs  → parity-fix ERASED it.
#       TASK 3 is real; the fix is to make postprocess contour-aware. RESURRECT.
#   pre-post delta < +0.05 (or contour still <= chance)  → model never learned
#       it. Stage-2 direction is a subjectivity ceiling. KILL TASK 3 → TASK 5.
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate

AUDIO="data/test_songs/SO TIRED ROCK - NUEKI.mp3"
BEAT_BASE="logs/beat_classifier/version_4/checkpoints/beat-epoch=11-val_f1_avg_tol=0.603.ckpt"
LAYOUT_A="logs/layout_phrase/version_13/checkpoints/last.ckpt"   # contour
LAYOUT_B="logs/layout_phrase/version_14/checkpoints/last.ckpt"   # control
OUT="outputs/2026-06-07/prepost"
mkdir -p "$OUT"

run_one () {
  local tag="$1" layout="$2" diff="$3" cflag="$4"
  local post="$OUT/${tag}_post.zip"
  local pre="$OUT/${tag}_pre.zip"
  echo "=== [$(date)] GEN $tag (diff=$diff $cflag) + PREPOST dump ==="
  BS_PREPOST_OUT="$pre" python scripts/generate.py "$AUDIO" --v7 \
      --beat-ckpt "$BEAT_BASE" --layout-ckpt "$layout" \
      --difficulty "$diff" --section-gate loud_only --no-use-instr "$cflag" \
      --output "$post" \
    || { echo "!! GEN $tag FAILED"; return 1; }

  echo "=== [$(date)] CONTOUR-FOLLOW $tag (PRE-postprocess) ==="
  python scripts/eval_contour_follow.py --audio "$AUDIO" --map "$pre" \
      --difficulty "$diff" --label "${tag}_PRE" \
      --json "$OUT/contour_${tag}_pre.json" || echo "!! contour PRE $tag FAILED"
}

run_one "A_contour_ex" "$LAYOUT_A" "Expert"     --use-contour
run_one "A_contour_ep" "$LAYOUT_A" "ExpertPlus" --use-contour
run_one "B_control_ex" "$LAYOUT_B" "Expert"     --no-use-contour
run_one "B_control_ep" "$LAYOUT_B" "ExpertPlus" --no-use-contour

echo "=== [$(date)] PRE-POSTPROCESS SUMMARY ==="
python - <<'PY'
import json, pathlib
OUT = pathlib.Path("outputs/2026-06-07/prepost")
def rate(tag):
    d = json.load(open(OUT / f"contour_{tag}_pre.json"))
    return d["contour_follow_rate"], d["n_scored"]
rows = {}
for tag in ("A_contour_ex","B_control_ex","A_contour_ep","B_control_ep"):
    r,n = rate(tag); rows[tag]=(r,n)
    print(f"  {tag:16s} PRE contour={r:.4f}  n={n}")
ax,_ = rows["A_contour_ex"]; bx,_ = rows["B_control_ex"]
ap,_ = rows["A_contour_ep"]; bp,_ = rows["B_control_ep"]
dx, dp = ax-bx, ap-bp
print()
print(f"Expert      PRE  contour A={ax:.4f}  control B={bx:.4f}  delta={dx:+.4f}")
print(f"ExpertPlus  PRE  contour A={ap:.4f}  control B={bp:.4f}  delta={dp:+.4f}")
print()
if dx >= 0.05 and dp >= 0.05:
    print("VERDICT: CONFOUND CONFIRMED. Pre-postprocess delta >= +0.05 at BOTH "
          "difficulties → the parity-fix was ERASING the contour signal. TASK 3 "
          "is real; resurrect it and make postprocess contour-aware (preserve "
          "swing sign where it doesn't violate parity).")
else:
    print("VERDICT: NO CONFOUND. Even pre-postprocess the model does not follow "
          "the lead contour above the control by >= +0.05. The signal was never "
          "learned, not erased. KILL TASK 3 → Stage-2 swing direction is a "
          "subjectivity ceiling. Next live item: TASK 5 (sparse long-range) or "
          "accept the current pipeline.")
PY
echo "=== [$(date)] CONFOUND TEST COMPLETE ==="
