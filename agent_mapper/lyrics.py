#!/usr/bin/env python
"""Timestamped lyrics from a song's VOCALS stem.

**Why this exists.** Kyle's request: *"if you had a longitudinal view with notes by
breakdown and importantly with when lyrics are said, you could create some amazing
maps."* He is pointing at the thing our model provably cannot do — `follow_vocals`
is ours 0.020 vs human 0.149 (7×) and we abandon sung phrases at 2.75× the human
rate, both because Stage-1 has no instrument projection and cannot hear the singer.

An agent given the words and their times has none of that handicap.

★**Transcribe the SEPARATED VOCALS, not the mix.** Demucs already gives us a clean
vocal stem; running Whisper on the full mix transcribes over drums and guitar and
produces both worse text and worse timings. We pay for the separation anyway.

⚠️**Whisper is transcribing SINGING, and it is not reliable at it.** Coverage — is
there a word wherever there is singing — is now good (0.967 of pitched vocal onsets on
1f8d6). **Accuracy of the words themselves is a different question and is NOT
established**: a self-consistency check (the same section letter should transcribe the
same way twice) scored 0.187 for the old config and 0.198 for the new one over only 3
pairs, which is **not resolvable** and settles nothing. Treat the words as a landmark
for *where you are in the song*, not as a quotation.

⚠️**Word timings are approximate.** Whisper's per-word timestamps come from
cross-attention alignment, not from onset detection — they are good to roughly a
syllable, not to the 50 ms the alignment axis uses. Use them to know *what is being
sung and roughly when*, and use the vocal STEM ONSETS (`onsets_vocals`) when you need
the exact instant to place a note on. They answer different questions and the brief
shows both.

Cache: `outputs/lyrics_cache/<song>.json`. Transcription is slow; nothing here
re-transcribes a song that already has a cache entry unless `--force`.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]
CACHE = REPO / "outputs" / "lyrics_cache"
DEFAULT_MODEL = "large-v3"


def _vocals_stem(audio: pathlib.Path, out_dir: pathlib.Path) -> pathlib.Path:
    """Separate and return the vocals stem, caching the wav next to the lyrics."""
    import numpy as np
    import soundfile as sf
    import torch

    sys.path.insert(0, str(REPO / "src"))
    from beatsaber_automapper.data.audio import load_audio
    from beatsaber_automapper.data.stem_separator import separate, DEMUCS_SR

    wav_path = out_dir / f"{audio.stem}.vocals.wav"
    if wav_path.exists():
        return wav_path
    y, sr = load_audio(str(audio))
    t = torch.as_tensor(y, dtype=torch.float32)
    if t.ndim == 1:
        t = t.unsqueeze(0)
    stems = separate(t, sr)
    v = stems["vocals"].detach().cpu().numpy()
    if v.ndim > 1:
        v = v.mean(axis=tuple(range(v.ndim - 1)))
    out_dir.mkdir(parents=True, exist_ok=True)
    sf.write(str(wav_path), np.asarray(v, dtype="float32"), DEMUCS_SR)
    return wav_path


# Whisper's stock filler on non-speech audio. It emits these confidently-formatted
# phrases when there is nothing to transcribe, and they are indistinguishable from real
# lyrics downstream -- `brief.py` will happily print one as the song's structural
# backbone. Caught 2026-08-21 on an instrumental that returned "Thank you for watching"
# three times at lang p=0.253 and had it presented as a 3x lyric repeat.
_FILLER = (
    "thank you for watching", "thanks for watching", "thank you.", "thank you",
    "please subscribe", "subscribe to my channel", "see you next time",
    "you", "bye.", "bye", ".", "♪", "music", "outro", "[music]",
)


def hallucination_risk(d: dict) -> tuple[bool, str]:
    """(suspect, why) -- is this transcript Whisper talking to itself?

    Three signals, any one of which is enough to distrust the words:
      * low language confidence -- the model is not sure it heard a language at all
      * filler text -- the stock phrases it emits over non-speech
      * a handful of words spread over minutes -- real singing is denser than this
    ⚠️This flags a transcript as UNTRUSTWORTHY; it does not delete it. An instrumental
    with a sung hook is a real case, and the agent should see the words and the doubt.
    """
    lp = float(d.get("language_probability") or 0.0)
    words = d.get("words") or []
    lines = d.get("lines") or []
    texts = [str(r.get("text", "")).strip().lower() for r in lines]
    n_filler = sum(1 for x in texts if x.rstrip(".!?") in
                   {f.rstrip(".!?") for f in _FILLER})
    why = []
    if lp and lp < 0.5:
        why.append(f"language confidence {lp:.2f} < 0.50")
    if texts and n_filler >= max(1, len(texts) // 2):
        why.append(f"{n_filler}/{len(texts)} lines are Whisper's stock filler")
    dur = float(d.get("duration") or 0.0)
    if dur > 30 and len(words) and len(words) / (dur / 60.0) < 20:
        why.append(f"{len(words)} words over {dur/60:.1f} min is far below sung density")
    return (bool(why), "; ".join(why))


def transcribe(audio: pathlib.Path, model_name: str = DEFAULT_MODEL,
               force: bool = False, language: str | None = None,
               vad: bool = False, temperature: float = 0.0,
               cache_key: str | None = None) -> dict:
    """Return {'words': [{t, end, word}], 'lines': [{t, end, text}], ...}.

    ⚠️`vad` / `temperature` / `cache_key` exist for the ABLATION only
    (`scripts/lyric_ablation.py`); their defaults ARE the shipped production path.
    The cache is keyed by song alone, so an arm that changes a setting **must** pass
    its own `cache_key` or it silently reads back another arm's transcription.
    """
    CACHE.mkdir(parents=True, exist_ok=True)
    cache_f = CACHE / f"{cache_key or audio.stem}.json"
    if cache_f.exists() and not force:
        return json.loads(cache_f.read_text())

    from faster_whisper import WhisperModel

    wav = _vocals_stem(audio, CACHE)
    # float16 on the 5090; the model is small next to Demucs so this is not the
    # expensive part of the pipeline.
    model = WhisperModel(model_name, device="cuda", compute_type="float16")
    # ⚠️**`vad_filter=True` EATS SUNG LYRICS.** Silero's VAD is tuned for speech and
    # discards sustained singing as non-speech, which is exactly the defect Kyle
    # reported ("I'm not seeing all words from the song"). Measured on 1f8d6 against
    # the pitched vocal onsets we already detect: VAD on covers **0.927** of the
    # singing, VAD off **0.967**, and word count goes 303 -> 430 (+42 %).
    #
    # ⚠️**`temperature=0` is not a quality knob here, it is a REPRODUCIBILITY one.**
    # faster-whisper's default is a temperature *fallback* list, so two identical runs
    # returned 391 and 387 words. Measured: temperature=0 gives byte-identical output
    # across runs; the fallback does not. A transcription that changes between runs
    # silently moves every lyric on the page.
    segments, info = model.transcribe(str(wav), word_timestamps=True,
                                      language=language, vad_filter=vad,
                                      temperature=temperature)
    words, lines = [], []
    for seg in segments:
        txt = (seg.text or "").strip()
        if txt:
            lines.append({"t": round(seg.start, 3), "end": round(seg.end, 3),
                          "text": txt})
        for w in (seg.words or []):
            tok = (w.word or "").strip()
            if tok:
                words.append({"t": round(w.start, 3), "end": round(w.end, 3),
                              "word": tok})
    out = {"song": audio.stem, "model": model_name,
           "vad": bool(vad), "temperature": temperature,
           "language": getattr(info, "language", None),
           "language_probability": round(getattr(info, "language_probability", 0.0), 3),
           "n_words": len(words), "n_lines": len(lines),
           "words": words, "lines": lines}
    cache_f.write_text(json.dumps(out, ensure_ascii=False, indent=1))
    return out


def load(song: str) -> dict | None:
    """Cached lyrics for a song id, or None. Never transcribes."""
    f = CACHE / f"{song}.json"
    if not f.exists():
        return None
    try:
        return json.loads(f.read_text())
    except Exception:  # noqa: BLE001
        return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("audio", type=pathlib.Path)
    ap.add_argument("--model", default=DEFAULT_MODEL,
                    help="whisper size: tiny/base/small/medium/large-v3")
    ap.add_argument("--language", default=None, help="force a language code")
    ap.add_argument("--force", action="store_true", help="re-transcribe")
    ap.add_argument("--lines", action="store_true", help="print lines, not words")
    a = ap.parse_args()
    if not a.audio.exists():
        print(f"no such audio: {a.audio}", file=sys.stderr)
        return 2
    d = transcribe(a.audio, a.model, a.force, a.language)
    print(f"{d['song']}: {d['n_words']} words, {d['n_lines']} lines, "
          f"lang={d['language']} p={d['language_probability']}")
    for r in (d["lines"] if a.lines else d["words"])[:40]:
        k = "text" if a.lines else "word"
        print(f"  {r['t']:8.2f}s  {r[k]}")
    if not d["words"]:
        print("  ⚠️no words found — instrumental, or the vocal stem is empty. "
              "That is a real answer, not a failure; the brief will say so.")
    else:
        suspect, why = hallucination_risk(d)
        if suspect:
            print(f"\n  🔴 DO NOT TRUST THESE WORDS — {why}.")
            print("  Whisper invents text on non-speech audio. Treat this song as an "
                  "INSTRUMENTAL:\n  ignore the lyric line and the lyric-repeat map in "
                  "`brief.py`, and plan from density instead.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
