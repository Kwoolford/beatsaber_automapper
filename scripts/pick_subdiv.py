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
0.922; separations 0.848 vs 0.724, 0.350, 0.114). On the 149-song wide cohort the
groups barely overlap — `half` tops out at 117.5 bpm, `same` bottoms out at 96.0:

| threshold | catches | false positives |
|---|---|---|
| 95 | 15/28 | **0** |
| **100 (default)** | 20/28 | 2 of 105 |
| 120 | 28/28 | 16 of 105 |

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
