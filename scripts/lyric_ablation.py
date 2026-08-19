#!/usr/bin/env python
"""ISOLATE THE LYRIC CHANGE — which of the two shipped changes actually did the work?

On 2026-08-18 the missing-lyrics defect ("I'm not seeing all words from the song") was
fixed by shipping **two** changes at once: `medium -> large-v3` AND `vad_filter ON ->
OFF`. Only the second is evidenced. At fixed VAD=ON the model upgrade moved coverage
**0.927 -> 0.918**, i.e. slightly *worse*, so `large-v3` is on the default for an
untested reason ("a bigger model must know more words") — and that reason is about word
IDENTITY, which coverage cannot see.

This runs the missing cell of the 2x2 (`medium` + vad OFF) so the two levers can be read
apart, and prints the agreement between the two vad-OFF arms as a second view: if
`medium` and `large-v3` transcribe the same words at the same times, the upgrade is
buying nothing on this song and the cheaper model should be the default.

**sung-coverage** = share of *pitched vocal onsets* (melody.py, the notes we would
actually place) that have a transcribed word over them. It measures whether the singing
is PRESENT in the transcript. ⚠️It does **not** measure whether the words are right.

Usage:
    python scripts/lyric_ablation.py data/eval_songset/1f8d6.ogg
"""

from __future__ import annotations

import argparse
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "agent_mapper"))

import lyrics as L      # noqa: E402
import melody as M      # noqa: E402

# A word "covers" an onset if the onset falls inside the word's span, padded. The pad is
# not a free knob: whisper's word ends are tight on the consonant, so a sung vowel that
# rings past the token would read as uncovered with pad 0.
PAD = 0.15

ARMS = [
    # (label, model, vad)
    ("medium,  vad ON  (old default)", "medium",   True),
    ("medium,  vad OFF (the control)", "medium",   False),
    ("large-v3, vad ON",               "large-v3", True),
    ("large-v3, vad OFF (shipped)",    "large-v3", False),
]


def sung_coverage(words: list[dict], onsets: list[dict], pad: float = PAD) -> float:
    if not onsets:
        return float("nan")
    spans = [(w["t"] - pad, w["end"] + pad) for w in words]
    spans.sort()
    hit = 0
    for ev in onsets:
        t = ev["t"]
        if any(a <= t <= b for a, b in spans):
            hit += 1
    return hit / len(onsets)


def _key(model: str, vad: bool, song: str) -> str:
    return f"_abl_{song}_{model.replace('-', '')}_{'vadon' if vad else 'vadoff'}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("audio", type=pathlib.Path)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    song = a.audio.stem
    vox = M.analyse(a.audio)["stems"]["vocals"]
    print(f"{song}: {len(vox)} pitched vocal onsets to cover\n")

    rows, transcripts = [], {}
    for label, model, vad in ARMS:
        d = L.transcribe(a.audio, model_name=model, force=a.force, vad=vad,
                         temperature=0.0, cache_key=_key(model, vad, song))
        cov = sung_coverage(d["words"], vox)
        rows.append((label, len(d["words"]), cov))
        transcripts[(model, vad)] = d
        print(f"  {label:<32} words {len(d['words']):>4}   sung-coverage {cov:.3f}")

    print("\n| config | words | sung-coverage |\n|---|---|---|")
    for label, n, cov in rows:
        print(f"| {label} | {n} | {cov:.3f} |")

    # Second view: do the two vad-OFF arms actually SAY different things?
    m_off = transcripts[("medium", False)]["words"]
    l_off = transcripts[("large-v3", False)]["words"]
    same = sum(1 for w in l_off
               if any(abs(v["t"] - w["t"]) <= 0.30
                      and v["word"].strip().lower().strip(".,!?") ==
                          w["word"].strip().lower().strip(".,!?")
                      for v in m_off))
    print(f"\nvad-OFF arms agree on {same}/{len(l_off)} large-v3 words "
          f"({same / max(len(l_off), 1):.3f}) at |dt| <= 0.30 s")
    print("\nVERDICT LOGIC: if `medium, vad OFF` reaches large-v3's coverage, the model "
          "upgrade bought nothing MEASURABLE and the only evidenced lever is VAD. That "
          "does not license reverting large-v3 on quality grounds -- word IDENTITY is "
          "still unmeasured -- it licenses saying so out loud in the default.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
