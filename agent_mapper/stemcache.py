#!/usr/bin/env python
"""Shared on-disk stem cache — separate a song ONCE, analyse it many ways.

Every perception tool here (`brief`, `melody`, `percussion`, `structure`) needs the
same four Demucs stems. Before this existed each one re-separated the song, which
cost ~40 s of GPU per tool per song and made building a new perception axis feel
expensive enough to not do it. It is now ~40 s once and free afterwards.

Stored as **mono float16 at 22 050 Hz**, which is the right resolution for every
question we ask of the stems and is 8× smaller than the 44.1 kHz stereo Demucs
returns. 22 050 Hz gives a Nyquist of 11 kHz — above every fundamental (a hi-hat's
*energy* extends higher, but its onset and band ratio do not need those octaves).
Use `separate_full()` if a future tool genuinely needs stereo at 44.1 kHz.

    from stemcache import stems, SR
    s = stems(pathlib.Path("data/eval_songset/1f333.ogg"))   # dict[str, np.ndarray]
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

CACHE = REPO / "outputs" / "stem_cache"
SR = 22050
STEMS = ("drums", "bass", "other", "vocals")

# ★The 6-source model, added 2026-08-20 for `events.py`. Kyle: *"these electric songs
# have LOTS of different note types."* With 4 stems every synth, lead, pluck, pad and
# FX in a track collapses into one bucket called `other`, so the perception layer
# literally cannot tell two different instruments apart in the half of the music that
# is not drums, bass or voice. `htdemucs_6s` splits `guitar` and `piano` out of that
# bucket. It does NOT split synths from each other -- that is what the per-stem timbre
# clustering in `events.py` is for -- but it is a real widening for free.
# ⚠️Cached SEPARATELY (`stem_cache_6s/`): the 6-source model is a different network and
# its `other` is not the 4-source `other`. Mixing the two caches would silently change
# what every existing number in this repo was measured on.
STEMS6 = ("drums", "bass", "other", "vocals", "guitar", "piano")
CACHE6 = REPO / "outputs" / "stem_cache_6s"


def _separate(audio: pathlib.Path, model: str | None = None) -> dict[str, np.ndarray]:
    import torch
    import torchaudio
    from beatsaber_automapper.data.audio import load_audio
    from beatsaber_automapper.data.stem_separator import separate, DEMUCS_SR

    y, sr = load_audio(str(audio))
    t = torch.as_tensor(y, dtype=torch.float32)
    if t.ndim == 1:
        t = t.unsqueeze(0)
    raw = _separate6(t, sr) if model == "htdemucs_6s" else separate(t, sr)

    out = {}
    for name, arr in raw.items():
        a = arr.detach().cpu().float()
        if a.ndim > 1:
            a = a.mean(dim=0)                       # stereo -> mono
        if DEMUCS_SR != SR:
            a = torchaudio.functional.resample(a, DEMUCS_SR, SR)
        out[name] = a.numpy().astype("float16")
    return out


def _separate6(wav, sr):
    """Run `htdemucs_6s` directly -- the shared separator is pinned to the 4-source model."""
    import torch
    from demucs.apply import apply_model
    from demucs.pretrained import get_model
    from beatsaber_automapper.data.stem_separator import DEMUCS_SR
    import torchaudio

    model = get_model("htdemucs_6s")
    model.eval()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(dev)
    if sr != DEMUCS_SR:
        wav = torchaudio.functional.resample(wav, sr, DEMUCS_SR)
    if wav.shape[0] == 1:
        wav = wav.repeat(2, 1)
    ref = wav.mean(0)
    wav = (wav - ref.mean()) / (ref.std() + 1e-8)
    with torch.no_grad():
        out = apply_model(model, wav[None].to(dev), split=True, overlap=0.25,
                          progress=False)[0]
    out = out * (ref.std() + 1e-8) + ref.mean()
    return {name: out[i].cpu() for i, name in enumerate(model.sources)}


def stems6(audio: pathlib.Path, force: bool = False) -> dict[str, np.ndarray]:
    """The SIX stems (adds `guitar` and `piano`) as mono float32 at `SR`.

    Separate cache from `stems()` on purpose -- see the STEMS6 note above.
    """
    CACHE6.mkdir(parents=True, exist_ok=True)
    f = CACHE6 / f"{audio.stem}.npz"
    if f.exists() and not force:
        d = np.load(f)
        if all(s in d for s in STEMS6):
            return {s: d[s].astype("float32") for s in STEMS6}
    out = _separate(audio, model="htdemucs_6s")
    np.savez(f, **out)
    return {s: out[s].astype("float32") for s in STEMS6}


def stems(audio: pathlib.Path, force: bool = False) -> dict[str, np.ndarray]:
    """The four stems as mono float32 arrays at `SR`. Separates once, then caches.

    Returned as float32 even though the cache is float16: every downstream librosa
    call wants float32, and converting here means no caller has to remember.
    """
    CACHE.mkdir(parents=True, exist_ok=True)
    f = CACHE / f"{audio.stem}.npz"
    if f.exists() and not force:
        d = np.load(f)
        if all(s in d for s in STEMS):
            return {s: d[s].astype("float32") for s in STEMS}
    out = _separate(audio)
    np.savez(f, **out)
    return {s: out[s].astype("float32") for s in STEMS}


def mix(audio: pathlib.Path) -> np.ndarray:
    """The full mix, mono at `SR` — for structure/energy work that wants everything."""
    import torchaudio
    import torch
    from beatsaber_automapper.data.audio import load_audio

    y, sr = load_audio(str(audio))
    t = torch.as_tensor(y, dtype=torch.float32)
    if t.ndim > 1:
        t = t.mean(dim=0)
    if sr != SR:
        t = torchaudio.functional.resample(t, sr, SR)
    return t.numpy().astype("float32")


if __name__ == "__main__":
    import time
    for p in sys.argv[1:]:
        a = pathlib.Path(p)
        t0 = time.time()
        s = stems(a)
        print(f"{a.stem}: {', '.join(f'{k} {len(v)/SR:.1f}s' for k, v in s.items())}"
              f"   ({time.time()-t0:.1f}s)")
