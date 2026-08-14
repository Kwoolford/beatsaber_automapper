#!/usr/bin/env python
"""Choose `BEAT_SUBDIV` for a song from its detected tempo. Prints `4` or `8`.

**What this is for.** On songs our detector calls at HALF the true tempo, our maps are
capped at exactly 0.500× the human's burst rate, because the minimum swing gap is one
grid slot and at half tempo that slot is twice as long in real time. `BEAT_SUBDIV=8`
lifts that ceiling exactly (0.500 → 1.000, n=28) — and **wrecks correctly-detected
songs** (1.000 → 2.000, onset precision −0.127). So it must only fire where the tempo
really is an octave low.

★**The detector is a threshold on the detected bpm itself**, which beat three
hand-designed statistics and a cross-validated tempogram classifier (AUC 0.973 vs
0.922; separations 0.848 vs 0.724, 0.350, 0.114).

🔴🔴**BUT MEASURED ON THE RAW BPM THIS SCRIPT CAN SEE, THE TRADE IS NOT GOOD ENOUGH —
DO NOT SHIP THIS AS A PRE-PASS.** n=133:

| threshold | catches | false positives | *(on post-fit bpm)* |
|---|---|---|---|
| 95 | 15/28 | **5** | *0* |
| 100 | 20/28 | **9** | *2* |
| 110 | 26/28 | 18 | — |

The largest zero-false-positive threshold catches **1 of 28**. At T=100 the trade is
20 songs gaining the ceiling against **9 working songs losing 0.127 precision**, and
the harm lands on songs that were fine.

★**WHY, AND WHAT TO BUILD INSTEAD**: on the **post-`BEAT_TEMPO_FIT`** bpm the groups
nearly separate (`same` floor rises 77.1 → 96.0) and T=95 is free. **The tempo fit is
what makes this detector work.** So the subdivision should be chosen *after* the fit,
inside the generator — which is feasible, since the subdivision is first used at
`pool_to_beat_grid`, already downstream of the fit. The obstacle is only that
`BEAT_SUBDIV` is read at import time (so `beat_grid` and `mert_encoder` cannot
disagree); the value would have to be threaded through as a parameter instead.
**This file is kept as the measurement that establishes that, not as a shipping path.**

⚠️**This is a heuristic about OUR DETECTOR, not a metrical analysis.** `librosa`'s
`start_bpm=120` prior means octave errors land *low*, which is the whole reason a flat
threshold works. It will misfire on genuinely slow music. **Do not call it octave
detection.**

🔴🔴**CALIBRATE ON THE RAW DETECTED BPM — NOT the bpm in a generated map.** The first
version of this threshold was swept over the bpm written into our generated maps,
which is **post-`BEAT_TEMPO_FIT`**; this script can only see the **raw** `detect_bpm`
output, before the fit. They disagree enough to flip songs: `21836` is correct-tempo
but raw-detects at 79.5 bpm, a false positive the post-fit calibration never showed.
★**That is the same "validated on a different input than production" error that
refuted `BEAT_GRID_PHASE=1` earlier the same night** — caught here *before* it cost a
run, by checking the lever's own output against the songs it was calibrated on.
The thresholds quoted below are swept over **raw** bpm for exactly this reason.

★**Why a separate script**: `BEAT_SUBDIV` is read from the environment at import time,
deliberately, so that `beat_grid` and `mert_encoder` cannot disagree. The choice
therefore has to be made *before* the generator is imported — so the caller runs this,
then sets the variable. Keeping it out-of-process also keeps the decision auditable:
the chosen subdivision is visible in the command, not buried in a log line.

Usage:
    SUB=$(python scripts/pick_subdiv.py song.ogg)
    BEAT_SUBDIV=$SUB python scripts/generate.py song.ogg ...
"""

from __future__ import annotations

import argparse
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

# Measured on the 149-song wide cohort. 100 is the conservative pick: 71% of the
# available ceilings at 2 false positives in 105. 95 is the free one (0 false
# positives, 54% of the ceilings) if a regression would be unacceptable.
DEFAULT_THRESHOLD_BPM = 100.0


def pick(audio: pathlib.Path, threshold: float = DEFAULT_THRESHOLD_BPM,
         verbose: bool = False) -> int:
    from beatsaber_automapper.data.audio import detect_bpm, load_audio

    wav, sr = load_audio(str(audio))
    bpm = float(detect_bpm(wav, sr))
    sub = 8 if 0 < bpm < threshold else 4
    if verbose:
        print(f"detected {bpm:.2f} bpm  threshold {threshold:.0f}  -> subdiv {sub}",
              file=sys.stderr)
    return sub


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("audio", type=pathlib.Path)
    ap.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD_BPM)
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args()
    if not a.audio.exists():
        print(f"no such audio: {a.audio}", file=sys.stderr)
        return 2
    try:
        print(pick(a.audio, a.threshold, a.verbose))
    except Exception as exc:  # noqa: BLE001
        # A failure here must fall back to the SAFE value, not stop generation — but
        # say so, because a silent fallback would look like a song that was assessed.
        print(f"pick_subdiv failed ({exc}) — falling back to 4", file=sys.stderr)
        print(4)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
