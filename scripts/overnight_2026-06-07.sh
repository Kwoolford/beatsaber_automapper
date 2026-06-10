#!/usr/bin/env bash
#
# Overnight 2026-06-07 — TASK-3 Stage-2 pitch-contour DoD.
#
# The last live build item. TASK 2 (per-instrument density into Stage-1) came
# back NULL on inference (2026-06-06): instr features raise density variation but
# generated-vs-human density never clears Spearman 0.41. TASK 3 attacks a
# different axis — not WHEN notes land but WHICH WAY they swing. We feed the
# per-slot pitch contour (lead_pitch / lead_dpitch / bass_pitch, cols 7:10 of the
# already-cached instr_beat_features — NO new preprocess) into the Stage-2 layout
# ENCODER so the decoder can make swing DIRECTION follow the melodic line:
# ascending → up-ish, descending → down-ish. Targets the North-Star "diagonal
# swings for sport" complaint.
#
# Design (single variable = contour; test song SO TIRED ROCK, the only one in
# data/test_songs/):
#   Arm A — train Stage-2 with --use-contour          (version_10 config + contour)
#   Arm B — train Stage-2 WITHOUT contour, same recipe (the control)
# Both: --ctx-len 16 (version_10 config), default d384/3enc/4dec, song-mem 150.
# Generate from each arm's last.ckpt (avoids val_token_acc selection bias — that
# metric anti-correlates with structure quality), production beat (version_4),
# section_gate=loud_only, at Expert + ExpertPlus.
#
# DoD metric: scripts/eval_contour_follow.py — fraction of vertical-swing notes
# whose swing sign matches the lead Δpitch sign at that slot (0.5 = chance).
# Also density-corr + alignment as REGRESSION GUARDS (contour must not wreck WHEN).
#
# Verdict logic for next session:
#   A contour-follow > B by a clear margin (>=+0.05) at BOTH difficulties, AND
#     alignment F1 / density-corr not materially worse than B
#       → TASK 3 DoD MET. Make --use-contour the default Stage-2; ArcViewer check.
#   A ~= B (delta < 0.05) or A regresses alignment
#       → TASK 3 null. Contour conditioning doesn't change swing direction;
#         Stage-2 spatial choices are a subjectivity ceiling like the per-slot
#         metrics. Next: TASK 5 (sparse long-range) or accept current pipeline.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
source .venv/bin/activate

AUDIO="data/test_songs/SO TIRED ROCK - NUEKI.mp3"
BEAT_BASE="logs/beat_classifier/version_4/checkpoints/beat-epoch=11-val_f1_avg_tol=0.603.ckpt"

OUT="outputs/2026-06-07"
LOG="logs/overnight/task3_contour_dod_2026-06-07.log"
mkdir -p "$OUT" "$(dirname "$LOG")"
exec > >(tee -a "$LOG") 2>&1

echo "=== [$(date)] TASK-3 contour DoD START ==="

# --- newest layout_phrase/version_* dir (the one a train just created) ---
newest_layout_dir () { ls -dt logs/layout_phrase/version_*/ 2>/dev/null | head -1; }

train_arm () {
  # $1 = tag, rest = extra train flags. All progress goes to stderr (still tee'd
  # to the log via the exec redirect); ONLY the resulting version dir is printed
  # to stdout so `dir="$(train_arm ...)"` captures the path and nothing else.
  local tag="$1"; shift
  echo "=== [$(date)] TRAIN $tag  $* ===" >&2
  python scripts/train_layout.py \
      --difficulties Expert ExpertPlus \
      --ctx-len 16 \
      --max-epochs 30 --patience 8 \
      "$@" >&2 \
    || { echo "!! TRAIN $tag FAILED" >&2; return 1; }
  local dir; dir="$(newest_layout_dir)"
  echo "=== [$(date)] TRAIN $tag done → $dir ===" >&2
  echo "$dir"
}

# ---- Arm A: contour ----
DIR_A="$(train_arm A_contour --use-contour | tail -1)"
LAYOUT_A="${DIR_A%/}/checkpoints/last.ckpt"
echo "LAYOUT_A=$LAYOUT_A"

# ---- Arm B: control (no contour, identical recipe) ----
DIR_B="$(train_arm B_control | tail -1)"
LAYOUT_B="${DIR_B%/}/checkpoints/last.ckpt"
echo "LAYOUT_B=$LAYOUT_B"

# ---- generate + eval one (arm, layout-ckpt, difficulty, contour-flag) ----
run_eval () {
  local tag="$1" layout="$2" diff="$3" cflag="$4"
  local zip="$OUT/${tag}.zip"
  echo "=== [$(date)] GEN $tag (diff=$diff $cflag) ==="
  python scripts/generate.py "$AUDIO" --v7 \
      --beat-ckpt "$BEAT_BASE" --layout-ckpt "$layout" \
      --difficulty "$diff" --section-gate loud_only --no-use-instr "$cflag" \
      --output "$zip" \
    || { echo "!! GEN $tag FAILED"; return 1; }

  echo "=== [$(date)] CONTOUR-FOLLOW $tag (DoD) ==="
  python scripts/eval_contour_follow.py --audio "$AUDIO" --map "$zip" \
      --difficulty "$diff" --label "$tag" \
      --json "$OUT/contour_${tag}.json" || echo "!! contour-follow $tag FAILED"

  echo "=== [$(date)] DENSITY-CORR $tag (regression guard) ==="
  python scripts/eval_density_corr.py --audio "$AUDIO" --map "$zip" \
      --difficulty "$diff" --label "$tag" \
      --json "$OUT/density_${tag}.json" || echo "!! density-corr $tag FAILED"

  echo "=== [$(date)] ALIGNMENT $tag (regression guard) ==="
  python scripts/eval_alignment.py --audio "$AUDIO" --map "$zip" \
      --difficulty "$diff" --tolerance-ms 50 \
      --json "$OUT/align_${tag}.json" || echo "!! alignment $tag FAILED"
}

run_eval "A_contour_ex" "$LAYOUT_A" "Expert"     --use-contour
run_eval "A_contour_ep" "$LAYOUT_A" "ExpertPlus" --use-contour
run_eval "B_control_ex" "$LAYOUT_B" "Expert"     --no-use-contour
run_eval "B_control_ep" "$LAYOUT_B" "ExpertPlus" --no-use-contour

echo
echo "=== [$(date)] SUMMARY ==="
python - <<'PY'
import json, glob
def load(pat):
    out = {}
    for f in sorted(glob.glob(pat)):
        d = json.load(open(f)); out[d["label"]] = d
    return out
cf  = load("outputs/2026-06-07/contour_*.json")
den = load("outputs/2026-06-07/density_*.json")
al_files = {}
for f in sorted(glob.glob("outputs/2026-06-07/align_*.json")):
    lbl = f.split("align_")[1].rsplit(".json",1)[0]
    try: al_files[lbl] = json.load(open(f))
    except Exception: pass

print(f"{'arm':<16} {'contour_follow':>14} {'n_scored':>9} {'density_spear':>14} {'gen_cv':>7}")
for lbl in ["A_contour_ex","B_control_ex","A_contour_ep","B_control_ep"]:
    c = cf.get(lbl, {}); d = den.get(lbl, {})
    print(f"{lbl:<16} {c.get('contour_follow_rate',float('nan')):>14.4f} "
          f"{c.get('n_scored',0):>9d} {d.get('spearman',float('nan')):>14.4f} "
          f"{d.get('gen_density_cv',float('nan')):>7.3f}")

def cfr(lbl): return cf.get(lbl, {}).get("contour_follow_rate")
print()
for diff, a_lbl, b_lbl in [("Expert","A_contour_ex","B_control_ex"),
                           ("ExpertPlus","A_contour_ep","B_control_ep")]:
    a, b = cfr(a_lbl), cfr(b_lbl)
    if a is not None and b is not None:
        print(f"{diff:<11} contour A={a:.4f}  control B={b:.4f}  delta={a-b:+.4f}")

ex_a, ex_b = cfr("A_contour_ex"), cfr("B_control_ex")
ep_a, ep_b = cfr("A_contour_ep"), cfr("B_control_ep")
deltas = [x for x in [(ex_a-ex_b) if (ex_a is not None and ex_b is not None) else None,
                      (ep_a-ep_b) if (ep_a is not None and ep_b is not None) else None]
          if x is not None]
print()
if deltas and all(d >= 0.05 for d in deltas):
    print("VERDICT: TASK-3 DoD MET — contour raises swing-direction follow >= +0.05 at "
          "all difficulties. Make --use-contour the default Stage-2 + ArcViewer check. "
          "(Confirm alignment/density not regressed vs B above.)")
else:
    print("VERDICT: TASK-3 NULL on the END-TO-END map (delta < 0.05). BEFORE killing "
          "TASK 3, rule out the CONFOUND: postprocess parity-fix rewrites ~48% of swing "
          "directions for playability and may clobber the model's contour choices. "
          "Re-run the contour-follow eval on the PRE-postprocess token stream to "
          "disambiguate 'model didn't learn it' from 'parity-fix erased it'. If null "
          "holds pre-postprocess too → Stage-2 direction is a subjectivity ceiling; "
          "next is TASK 5 (sparse long-range) or accept the current pipeline.")
PY
echo "=== [$(date)] TASK-3 contour DoD COMPLETE ==="
