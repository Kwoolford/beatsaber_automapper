#!/usr/bin/env bash
# A complete map of Hunger, planned off brief.py's 8-bar timeline.
# ★ACCENTS FIRST, then fill: in a dense alternating pass both hands are always busy,
#  so a double can never be placed. Sparse accent pass -> doubles land -> fill around.
set -e
M="python agent_mapper/mapctl.py"; N=${1:-hunger}
$M auto $N --bars 1-216   --follow drums --every 4 --lead L --doubles      # the accents
$M auto $N --bars 1-8     --follow bass   --every 2 --lead L               # intro: sparse
$M auto $N --bars 9-16    --follow drums  --every 2 --lead R
$M auto $N --bars 17-32   --follow drums  --every 2 --lead L --wide        # drums heavy
$M auto $N --bars 33-56   --follow vocals --lead L --wide                  # VERSE: the voice
$M auto $N --bars 33-56   --follow drums  --every 3 --lead R
$M auto $N --bars 57-72   --follow drums  --lead L --wide                  # CHORUS
$M auto $N --bars 57-72   --follow vocals --every 2 --lead R
$M auto $N --bars 73-88   --follow drums  --every 2 --lead R --wide
$M auto $N --bars 89-112  --follow vocals --lead L --wide                  # VERSE 2
$M auto $N --bars 89-112  --follow drums  --every 3 --lead R
$M auto $N --bars 113-128 --follow drums  --lead L --wide                  # CHORUS 2
$M auto $N --bars 113-128 --follow vocals --every 2 --lead R
$M auto $N --bars 129-136 --follow bass   --every 2 --lead L               # BREAKDOWN
$M auto $N --bars 137-160 --follow drums  --every 2 --lead R --wide
$M auto $N --bars 161-168 --follow other  --every 2 --lead L               # QUIET
$M auto $N --bars 169-192 --follow drums  --every 2 --lead L --wide
$M auto $N --bars 193-208 --follow drums  --lead R --wide                  # FINAL CHORUS
$M auto $N --bars 193-208 --follow vocals --every 2 --lead L
$M auto $N --bars 209-216 --follow bass   --every 2 --lead R               # outro
