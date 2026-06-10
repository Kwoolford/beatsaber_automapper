#!/usr/bin/env bash
#
# Overnight 2026-06-05 — TASK-2 inference DoD.
#
# The question scoped-V8 TASK 2 still hasn't answered: with the Stage-1 section
# gate OFF, does feeding per-instrument layering features (version_7 --use-instr
# checkpoint) make the GENERATED note density track the song's real musical
# density? Every gate-fixed map so far is a flat ~8 NPS metronome (see
# outputs/task0/report_v10_ex.txt). val_f1 said the instr features were a wash,
# but val_f1 is per-slot binary acc — the WRONG yardstick. This is the right one.
#
# Design (test song: SO TIRED ROCK, the only one in data/test_songs/):
#   A — instr   + gate OFF   (the hypothesis: learned density tracks structure)
#   B — baseline+ gate OFF   (control: same gate, no instr → isolates instr's effect)
#   C — instr   + gate loud_only (instr + the silent-drop safety net combined)
#  each at Expert and ExpertPlus.
# Metric: scripts/eval_density_corr.py — Spearman(gen density, ref onset density)
# over uniform 2 s windows. DoD: instr arm Spearman >= 0.41 AND > control.
# Also eval_alignment.py for the per-section F1 the leaderboard tracks.
#
# Verdict logic for next session:
#   instr beats control AND >=0.41  → wire instr into the default inference path,
#                                     retire _SECTION_THRESHOLDS (TASK 2 DoD MET).
#   instr ~= control or < 0.41      → TASK 2 is dead on inference too; pivot to
#                                     TASK 3 (Stage-2 contour) as the live build.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
source .venv/bin/activate

AUDIO="data/test_songs/SO TIRED ROCK - NUEKI.mp3"
BEAT_INSTR="logs/beat_classifier/version_7/checkpoints/beat-epoch=00-val_f1_avg_tol=0.600.ckpt"
BEAT_BASE="logs/beat_classifier/version_4/checkpoints/beat-epoch=11-val_f1_avg_tol=0.603.ckpt"
LAYOUT="logs/layout_phrase/version_10/checkpoints/layout-epoch=09-val_token_acc=0.865.ckpt"

OUT="outputs/2026-06-05"
LOG="logs/overnight/task2_infer_dod_2026-06-05.log"
mkdir -p "$OUT" "$(dirname "$LOG")"
exec > >(tee -a "$LOG") 2>&1

echo "=== [$(date)] TASK-2 inference DoD START ==="

# arm  beat-ckpt   extra-flags                       difficulty
run_arm () {
  local tag="$1" beat="$2" diff="$3"; shift 3
  local extra=("$@")
  local zip="$OUT/${tag}.zip"
  echo "=== [$(date)] GEN $tag (diff=$diff) ${extra[*]} ==="
  python scripts/generate.py "$AUDIO" --v7 \
      --beat-ckpt "$beat" --layout-ckpt "$LAYOUT" \
      --difficulty "$diff" --section-gate off \
      --output "$zip" "${extra[@]}" \
    || { echo "!! GEN $tag FAILED"; return 1; }

  echo "=== [$(date)] DENSITY-CORR $tag ==="
  python scripts/eval_density_corr.py --audio "$AUDIO" --map "$zip" \
      --difficulty "$diff" --label "$tag" \
      --json "$OUT/density_${tag}.json" || echo "!! density-corr $tag FAILED"

  echo "=== [$(date)] ALIGNMENT $tag ==="
  python scripts/eval_alignment.py --audio "$AUDIO" --map "$zip" \
      --difficulty "$diff" --tolerance-ms 50 \
      --json "$OUT/align_${tag}.json" || echo "!! alignment $tag FAILED"
}

# Arm A: instr, gate off
run_arm "A_instr_off_ex" "$BEAT_INSTR" "Expert"     --use-instr
run_arm "A_instr_off_ep" "$BEAT_INSTR" "ExpertPlus" --use-instr

# Arm B: baseline (no instr), gate off — the control
run_arm "B_base_off_ex"  "$BEAT_BASE"  "Expert"     --no-use-instr
run_arm "B_base_off_ep"  "$BEAT_BASE"  "ExpertPlus" --no-use-instr

# Arm C: instr + loud_only safety gate (Expert only)
echo "=== [$(date)] GEN C_instr_loud_ex (Expert) ==="
python scripts/generate.py "$AUDIO" --v7 \
    --beat-ckpt "$BEAT_INSTR" --layout-ckpt "$LAYOUT" \
    --difficulty Expert --section-gate loud_only --use-instr \
    --output "$OUT/C_instr_loud_ex.zip" \
  && python scripts/eval_density_corr.py --audio "$AUDIO" \
       --map "$OUT/C_instr_loud_ex.zip" --difficulty Expert \
       --label "C_instr_loud_ex" --json "$OUT/density_C_instr_loud_ex.json" \
  || echo "!! arm C FAILED"

echo
echo "=== [$(date)] SUMMARY (Spearman density-corr) ==="
python - <<'PY'
import json, glob, os
rows = []
for f in sorted(glob.glob("outputs/2026-06-05/density_*.json")):
    d = json.load(open(f))
    rows.append((d["label"], d["spearman"], d["pearson"],
                 d["gen_density_cv"], d["dod_pass"]))
print(f"{'arm':<20} {'spearman':>9} {'pearson':>8} {'gen_cv':>7} {'DoD':>5}")
for lbl, sp, pe, cv, ok in rows:
    print(f"{lbl:<20} {sp:>9.4f} {pe:>8.4f} {cv:>7.3f} {'PASS' if ok else 'fail':>5}")
# verdict
def get(sub):
    return next((sp for lbl, sp, *_ in rows if sub in lbl), None)
a, b = get("A_instr_off_ex"), get("B_base_off_ex")
if a is not None and b is not None:
    print(f"\nEXPERT  instr={a:.4f}  control={b:.4f}  delta={a-b:+.4f}")
    if a >= 0.41 and a > b:
        print("VERDICT: TASK-2 DoD MET on inference — instr tracks structure > control.")
    else:
        print("VERDICT: TASK-2 inference null — pivot to TASK 3 (Stage-2 contour).")
PY
echo "=== [$(date)] TASK-2 inference DoD COMPLETE ==="
