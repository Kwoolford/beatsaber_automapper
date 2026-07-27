#!/usr/bin/env bash
# 2026-07-26 — LATE-SONG-COLLAPSE: population-scale validation.
#
# WHY: session 2026-07-23 built scripts/eval_late_window.py and found the late-song
# collapse complaint does NOT reproduce (mean late_gap -0.024 over 6 songs). The one
# open caveat was sample size: "the 6-song eval set may not include a song Kyle
# actually saw collapse on." This run kills that caveat WITHOUT needing Kyle to
# remember a specific song — it widens the set to 24 songs and reports the
# per-song outlier distribution, not just the mean.
#
# ALSO NEW this session: eval_late_window.py now compares against the actual HUMAN
# map from data/raw (human_gap), not only the librosa/Demucs audio-onset reference.
# That is the complaint's direct form: did the human mapper keep the tail busy while
# we thinned out?
#
# ARMS: prod (current production: temp 0.9/top_p 0.97, anti-repeat W1/S2 baked in,
#        section_gate loud_only, density-select) — the config Kyle would ship.
#       noar is NOT re-run here; the 07-23 regression already covered the
#        anti-repeat ablation and it is orthogonal to tail behaviour.
#
# VERDICT LOGIC (printed at the end):
#   DoD MET  => mean late_gap <= 0.03 AND mean human_gap <= 0.03 AND
#               <=10% of songs have late_gap > 0.10
#               ⇒ late-song collapse is CLOSED at population scale; the complaint
#                 was fixed by section_gate=loud_only + density-select-y2.5.
#                 Next session = ship-it / step-back fork for Kyle.
#   NOT MET  => the outlier list names the songs that DO collapse. Those become the
#               diagnosis set: compare Stage-1 beat_probs vs Stage-2 layout context
#               drift in the tail on those songs specifically.
set -uo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate

N_SONGS=24
LOG_DIR=logs/overnight
mkdir -p "$LOG_DIR"

echo "=============================================================="
echo "STEP 1 — expand eval songset to ${N_SONGS} songs (incremental)"
echo "=============================================================="
python scripts/eval_sweep.py build-songset --n "$N_SONGS"

echo
echo "=============================================================="
echo "STEP 2 — generate prod maps for the full songset (cached songs skipped)"
echo "=============================================================="
python scripts/eval_sweep.py sweep --arms prod

echo
echo "=============================================================="
echo "STEP 3 — late-window verdict, final 20% and final 10% tails"
echo "=============================================================="
for TAIL in 0.20 0.10; do
  echo
  echo "-------- tail = ${TAIL} --------"
  python scripts/eval_late_window.py --arm prod --tail "$TAIL" --worst 8 2>&1 \
    | grep -v "INFO\|WARNING"
done

echo
echo "=============================================================="
echo "COMPLETE — late-window population validation"
echo "=============================================================="
echo "Read the verdict lines above. 'population DoD ... MET' at BOTH tails"
echo "=> late-song collapse CLOSED; report the ship-it/step-back fork to Kyle."
echo "Otherwise the named outlier songs are the diagnosis set for next session."
