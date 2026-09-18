#!/bin/bash
# Rebuild the sotired session from spec.txt and export the zip. Usage: build.sh [bars-to-read]
set -e
cd /home/kyle/repos/beatsaber_automapper && source .venv/bin/activate
D=agent_mapper/sessions/sotired
A="data/test_songs/SO TIRED ROCK - NUEKI.mp3"
python $D/compose.py $D/spec.txt $D/composed.txt
python agent_mapper/mapctl.py init "$A" --name sotired --fresh > /dev/null
python agent_mapper/mapctl.py add sotired --from $D/composed.txt | tail -1
python agent_mapper/mapctl.py check sotired | tail -6
python agent_mapper/mapctl.py export sotired --out outputs/sotired/sotired_notes.zip | tail -1
