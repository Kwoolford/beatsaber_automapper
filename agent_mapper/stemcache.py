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


def _separate(audio: pathlib.Path) -> dict[str, np.ndarray]:
    import torch
    import torchaudio
    from beatsaber_automapper.data.audio import load_audio
    from beatsaber_automapper.data.stem_separator import separate, DEMUCS_SR

    y, sr = load_audio(str(audio))
    t = torch.as_tensor(y, dtype=torch.float32)
    if t.ndim == 1:
        t = t.unsqueeze(0)
    raw = separate(t, sr)

    out = {}
    for name, arr in raw.items():
        a = arr.detach().cpu().float()
        if a.ndim > 1:
            a = a.mean(dim=0)                       # stereo -> mono
        if DEMUCS_SR != SR:
            a = torchaudio.functional.resample(a, DEMUCS_SR, SR)
        out[name] = a.numpy().astype("float16")
    return out


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
