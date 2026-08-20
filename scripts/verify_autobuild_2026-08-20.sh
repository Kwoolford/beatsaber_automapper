#!/usr/bin/env bash
# Reproduce the 23/23 pass rate from a CLEAN run with the committed defaults
# (two-sided judge + idiomize REPEAT_P=0.55). The figure quoted mid-session came
# from re-dressing maps that had already been dressed at repeat_p=0, which is a
# second pass and not the shipped path.
set -u
cd /home/kyle/repos/beatsaber_automapper
source .venv/bin/activate
OUT=/mnt/giga_speed/claude_tmp/claude-1000/-home-kyle/9e5c785e-f25a-4d83-a66a-c190cb0e731d/scratchpad/ab_clean
mkdir -p "$OUT"
for f in data/eval_songset/*.ogg; do
  s=$(basename "$f" .ogg)
  timeout 1800 python agent_mapper/autobuild.py "$f" --name "cl_$s" \
      --out "$OUT/$s.zip" --json "$OUT/$s.json" > "$OUT/$s.log" 2>&1
  echo "$s exit=$?"
done
echo "VERIFY_COMPLETE"
