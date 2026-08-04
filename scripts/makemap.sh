#!/usr/bin/env bash
# Make a playable Beat Saber map from any audio file, using the PROMOTED
# 2026-08-03 baseline defaults (docs/BASELINE_2026-08-03.md).
#
#   scripts/makemap.sh <audio> [song name] [output.zip]
#
# Examples
#   scripts/makemap.sh ~/Music/track.mp3
#   scripts/makemap.sh "data/test_songs/SO TIRED ROCK - NUEKI.mp3" "So Tired Rock"
#   scripts/makemap.sh ~/Music/track.mp3 "My Song" ~/maps/mysong.zip
#
# Then put it in the headset:
#   scripts/deploy_maps.py <output.zip>
#
# All eight promoted levers are ON by default -- no env vars needed. Override any
# of them the usual way, e.g.  BEAT_DIFFICULTY_SCALE=0.55 scripts/makemap.sh song.mp3
# Seed defaults to 0 so a given song+config always reproduces the same map; pass
# BSA_SEED=1 for a different draw.
set -euo pipefail
cd "$(dirname "$0")/.."

AUDIO="${1:?usage: scripts/makemap.sh <audio> [song name] [output.zip]}"
NAME="${2:-$(basename "${AUDIO%.*}")}"
OUT="${3:-outputs/manual/$(basename "${AUDIO%.*}").zip}"

BEAT_CKPT="logs/beat_classifier/version_4/checkpoints/beat-epoch=11-val_f1_avg_tol=0.603.ckpt"
LAYOUT_CKPT="logs/layout_phrase/version_10/checkpoints/layout-epoch=09-val_token_acc=0.865.ckpt"

mkdir -p "$(dirname "$OUT")"
echo "song   : $NAME"
echo "audio  : $AUDIO"
echo "output : $OUT"
echo

.venv/bin/python scripts/generate.py "$AUDIO" \
    --v7 \
    --beat-ckpt   "$BEAT_CKPT" \
    --layout-ckpt "$LAYOUT_CKPT" \
    --difficulty Expert \
    --section-gate loud_only \
    --song-name "$NAME" \
    --seed "${BSA_SEED:-0}" \
    --output "$OUT"

echo
echo "done -> $OUT"
echo "install it with:  scripts/deploy_maps.py \"$OUT\""
