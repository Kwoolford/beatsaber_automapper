#!/usr/bin/env bash
# Autobuild every eval-songset song end to end, then report the pass rate.
# The DoD for "create any map from any song": the loop must COMPLETE on every song
# (a crash is a failure of the framework) and the resulting maps must be judged
# honestly -- a low pass rate is a finding about the builder, not a reason to
# loosen the judge.
set -u
cd /home/kyle/repos/beatsaber_automapper
source .venv/bin/activate
OUT=/mnt/giga_speed/claude_tmp/claude-1000/-home-kyle/9e5c785e-f25a-4d83-a66a-c190cb0e731d/scratchpad/ab
mkdir -p "$OUT"
for f in data/eval_songset/*.ogg; do
  s=$(basename "$f" .ogg)
  timeout 1800 python agent_mapper/autobuild.py "$f" --name "ab_$s" \
      --out "$OUT/$s.zip" --json "$OUT/$s.json" > "$OUT/$s.log" 2>&1
  echo "$s exit=$?"
done
echo "AUTOBUILD_COMPLETE"
