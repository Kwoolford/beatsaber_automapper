#!/usr/bin/env python
"""POLYPHONY for the LEAD lane — the chords a salience peak cannot see.

★**The gap this closes.** `melody.py` gives **one pitch per onset**: for `vocals` and
`bass` that is right (they are monophonic), but the `other` stem is guitars, keys and
pads, and one salience peak per frame is a *summary* of a chord, not the chord. The
notesheet has been drawing that summary as if it were the line, and V1's own docstring
flags it as "a salience peak; see the confidence note".

**Measured on the four standing songs: median polyphony 2.0, and 56-70 % of the song
has two or more notes sounding.** So the LEAD lane was structurally missing about half
its content everywhere — not on the hard songs, on all of them.

## Is basic-pitch's extra content real, or is it noise?
Ground truth does not exist here, so the check is **key coherence**: a random pitch set
sits near 0.58 in-key, and adding noise to a real transcription must drag it *down*.

| song | our LEAD in-key | basic-pitch in-key | verdict |
|---|---|---|---|
| Fallen Kingdom | 0.930 | **0.996** | basic-pitch better |
| アリスブルー | 0.910 | **0.993** | basic-pitch better |
| Digital Life Hacker | 0.952 | 0.936 | a tie |
| Hunger | **0.870** | 0.647 | ★**our tracker better** |

⇒**This is NOT a blanket swap, and `USE_BP` defaults to the sanity check rather than to
"yes".** ⚠️And the Hunger row is genuinely ambiguous: it is metal with distorted guitars,
so the in-key proxy **cannot distinguish "basic-pitch is wrong" from "the song is
chromatic"** — the proxy assumes diatonic music. It is used only to *refuse* the swap
where the evidence for it is absent, which is the safe direction of that ambiguity.

Cost is negligible: **1.6-2.4 s a song** through the ONNX backend (the TF wheel does not
build on Python 3.12; onnxruntime is already installed and is all this needs).

Usage:
    python agent_mapper/chords.py data/eval_songset/1f8d6.ogg
    python agent_mapper/chords.py <audio> --stem other --force
"""

from __future__ import annotations

import argparse
import collections
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

REPO = pathlib.Path(__file__).resolve().parents[1]
CACHE = REPO / "outputs" / "chords_cache"

# Krumhansl major profile, only ever used for the in-key SANITY CHECK above — never to
# claim a key. `melody.key_of` is the module that answers that question.
_MAJ = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
_IN_KEY = {0, 2, 4, 5, 7, 9, 11}
RANDOM_IN_KEY = 0.583      # 7 of 12 pitch classes: what "no better than noise" looks like


def in_key_share(pitches) -> float:
    """Share of notes inside the best-fitting major key. Higher = more coherent."""
    ps = [int(round(float(p))) % 12 for p in pitches]
    if not ps:
        return float("nan")
    pc = collections.Counter(ps)
    best = max(range(12), key=lambda k: sum(pc[(k + i) % 12] * _MAJ[i] for i in range(12)))
    return sum(pc[(best + i) % 12] for i in _IN_KEY) / len(ps)


def transcribe(audio: pathlib.Path, stem: str = "other", force: bool = False) -> dict:
    """Polyphonic note events for one stem: [{t, end, midi, amp}], cached."""
    CACHE.mkdir(parents=True, exist_ok=True)
    f = CACHE / f"{audio.stem}.{stem}.json"
    if f.exists() and not force:
        return json.loads(f.read_text())

    import soundfile as sf
    from stemcache import stems, SR

    from basic_pitch import ICASSP_2022_MODEL_PATH
    from basic_pitch.inference import predict

    y = np.asarray(stems(audio)[stem], dtype="float32")
    tmp = CACHE / f"{audio.stem}.{stem}.wav"
    sf.write(str(tmp), y, SR)
    try:
        _, _, events = predict(str(tmp), ICASSP_2022_MODEL_PATH)
    finally:
        tmp.unlink(missing_ok=True)

    notes = [{"t": round(float(e[0]), 3), "end": round(float(e[1]), 3),
              "midi": int(round(float(e[2]))), "amp": round(float(e[3]), 3)}
             for e in sorted(events, key=lambda e: e[0])]
    out = {"song": audio.stem, "stem": stem, "n": len(notes),
           "in_key": round(in_key_share([n["midi"] for n in notes]), 3),
           "notes": notes}
    f.write_text(json.dumps(out))
    return out


def polyphony(notes: list[dict], dur: float, step: float = 0.2) -> dict:
    """How many notes sound at once — the quantity a single salience peak destroys."""
    if not notes:
        return {"median": 0.0, "p90": 0.0, "share_chord": 0.0}
    st = np.array([n["t"] for n in notes])
    en = np.array([n["end"] for n in notes])
    grid = np.arange(0.0, max(dur, step), step)
    k = np.array([int(np.sum((st <= t) & (en >= t))) for t in grid])
    return {"median": float(np.median(k)), "p90": float(np.percentile(k, 90)),
            "share_chord": float((k >= 2).mean())}


def better_than_ours(audio: pathlib.Path, stem: str = "other",
                     force: bool = False) -> tuple[bool, dict]:
    """★The gate. Adopt polyphony for this song ONLY where it is better supported.

    Refusing on a tie is deliberate: our tracker is what the endorsed page already
    draws, and swapping the picture Kyle has seen needs positive evidence, not parity.
    """
    import melody as _mel

    bp = transcribe(audio, stem, force)
    ours = _mel.analyse(audio)["stems"].get(stem, [])
    ours_k = in_key_share([e["midi"] for e in ours])
    info = {"bp_in_key": bp["in_key"], "ours_in_key": round(ours_k, 3),
            "bp_notes": bp["n"], "our_notes": len(ours),
            "random_floor": RANDOM_IN_KEY}
    ok = bool(bp["in_key"] > ours_k + 0.02 and bp["in_key"] > RANDOM_IN_KEY)
    info["verdict"] = ("polyphony adopted" if ok else
                       "kept our salience peak — basic-pitch is not better here")
    return ok, info


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("audio", type=pathlib.Path)
    ap.add_argument("--stem", default="other")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    import brief as _brief
    dur = _brief.analyse(a.audio)["dur"]
    bp = transcribe(a.audio, a.stem, a.force)
    p = polyphony(bp["notes"], dur)
    ok, info = better_than_ours(a.audio, a.stem, False)
    print(f"{a.audio.stem} · {a.stem}: {bp['n']} note events")
    print(f"  polyphony median {p['median']:.1f}, p90 {p['p90']:.1f}, "
          f"{p['share_chord']:.0%} of the song has 2+ notes sounding")
    print(f"  in-key  basic-pitch {info['bp_in_key']:.3f}  ours {info['ours_in_key']:.3f}"
          f"  (noise floor {RANDOM_IN_KEY})")
    print(f"  ⇒ {info['verdict']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
