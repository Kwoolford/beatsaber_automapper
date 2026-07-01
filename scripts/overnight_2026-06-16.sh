#!/usr/bin/env bash
# P1-4 — best-of-N=16 rerank PoC (Phase-2 kickoff), 2026-06-16.
#
# ARMS: 16 stochastic V7 draws of the SAME song (production config: beat version_4 +
#       layout version_10, section_gate=loud_only, temp 0.8 / top_p 0.85 defaults).
# CONTROL: the no-rerank pick = first candidate by filename (what you'd ship blind).
# SELECTION: best_of_n_poc.py = swing-sim hard filter -> z(feel) - lambda*z(monotony).
# DELIVERABLE: outputs/bon_2026-06-16/{winner.png, control.png} for Kyle to ArcViewer.
#
# This is UNGATED (Phase-2 kickoff). The "verdict" is whether selection is real:
# finite feel+monotony spread across N, and a winner that dominates the control on
# both. Kyle stays the final judge (milestone re-anchor).
set -u
cd "$(dirname "$0")/.." || exit 1
source .venv/bin/activate

SONG="data/test_songs/SO TIRED ROCK - NUEKI.mp3"
OUTDIR="outputs/bon_2026-06-16"
LOG="logs/overnight/bon_2026-06-16.log"
BEAT_CKPT="logs/beat_classifier/version_4/checkpoints/beat-epoch=11-val_f1_avg_tol=0.603.ckpt"
LAYOUT_CKPT="logs/layout_phrase/version_10/checkpoints/layout-epoch=09-val_token_acc=0.865.ckpt"
N=16
DIFF="Expert"

mkdir -p "$OUTDIR" "$(dirname "$LOG")"
exec > >(tee -a "$LOG") 2>&1
echo "=================================================================="
echo "P1-4 best-of-N=$N  $(date)"
echo "song: $SONG"
echo "=================================================================="

ok=0
for i in $(seq 1 "$N"); do
  out="$OUTDIR/cand_$(printf '%02d' "$i").zip"
  echo "--- [gen $i/$N] -> $out ---"
  t0=$(date +%s)
  if python scripts/generate.py "$SONG" --v7 --difficulty "$DIFF" \
        --beat-ckpt "$BEAT_CKPT" --layout-ckpt "$LAYOUT_CKPT" \
        --section-gate loud_only --output "$out"; then
    ok=$((ok + 1))
    echo "    ok ($(($(date +%s) - t0))s)  [$ok ok so far]"
  else
    echo "    !! generate failed for draw $i (continuing)"
  fi
done
echo "=== generation done: $ok/$N succeeded ==="

if [ "$ok" -lt 2 ]; then
  echo "!! fewer than 2 candidates generated; cannot rerank. ABORT."
  exit 2
fi

echo "=== rerank (swing-sim filter -> z(feel) - lambda*z(monotony)) ==="
python scripts/best_of_n_poc.py \
  --maps "$OUTDIR/cand_*.zip" --difficulty "$DIFF" \
  --ckpt outputs/feel_disc_ep1_2026-06-15.pt \
  --out-dir "$OUTDIR" --lambda 1.0

echo ""
echo "=== VERDICT LOGIC (read next session) ==="
python - "$OUTDIR/bon_summary.json" <<'PY'
import json, sys
s = json.load(open(sys.argv[1]))
fs, ms = s["feel_spread"], s["monotony_spread"]
w, c = s["winner_rec"], s["control_rec"]
moved = s["rerank_moved"]
dom = (w["feel_logit"] >= c["feel_logit"]) and (w["monotony"] <= c["monotony"])
print(f"N={s['n_candidates']}  feel_spread={fs}  monotony_spread={ms}")
print(f"winner  feel={w['feel_logit']} monotony={w['monotony']} viol={w['swing_violations']}")
print(f"control feel={c['feel_logit']} monotony={c['monotony']} viol={c['swing_violations']}")
print(f"rerank_moved_off_control={moved}   winner_dominates_control(feel↑ & monotony↓)={dom}")
print()
if fs < 1e-3 and ms < 1e-3:
    print("AMBER: no spread across N — every draw scores identically. The reranker has")
    print("nothing to act on; best-of-N is a no-op for this song. Next: widen sampling")
    print("(temperature/top_p) or add phrase-level resampling so candidates actually differ.")
elif moved and dom:
    print("GREEN (PoC): selection is REAL — winner != control and dominates it on BOTH")
    print("feel(↑) and monotony(↓). Deliverable: ArcView winner.png vs control.png. If Kyle")
    print("agrees the winner feels less monotonous, Phase-2 best-of-N selection is validated;")
    print("scale to phrase-level resampling + a held-out song set.")
else:
    print("AMBER: there IS spread but the winner does not cleanly dominate the control on")
    print("both axes (feel/monotony may disagree). Inspect the ranking in bon_summary.json")
    print("and tune lambda; still ArcView winner.png vs control.png for Kyle's read.")
PY

echo ""
echo "=== COMPLETE $(date) ==="
