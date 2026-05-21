"""Demucs htdemucs audio source separation helper for V7 preprocessing.

Separates an audio waveform into [drums, bass, other, vocals] stems using
the pretrained htdemucs model. Returns each stem as a [2, T] stereo tensor
at the model's native 44100 Hz sample rate.

The model is loaded lazily and cached for the process lifetime.
"""

from __future__ import annotations

import logging
from functools import lru_cache

import torch

logger = logging.getLogger(__name__)

DEMUCS_MODEL = "htdemucs"
DEMUCS_SR    = 44100
DEMUCS_SOURCES = ["drums", "bass", "other", "vocals"]


@lru_cache(maxsize=1)
def _load_demucs(device: str = "cuda") -> tuple:
    """Load and cache htdemucs model. Called once per process."""
    from demucs.pretrained import get_model
    logger.info("Loading Demucs %s …", DEMUCS_MODEL)
    model = get_model(DEMUCS_MODEL)
    model = model.to(device)
    model.eval()
    logger.info("Demucs loaded on %s. sources=%s", device, model.sources)
    return model


def separate(
    waveform: torch.Tensor,
    src_sr: int,
    device: str = "cuda",
) -> dict[str, torch.Tensor]:
    """Separate a waveform into source stems.

    Args:
        waveform: Audio [C, T] at src_sr Hz. Mono is duplicated to stereo.
        src_sr:   Source sample rate.
        device:   Torch device string.

    Returns:
        Dict mapping stem name → [2, T] tensor at DEMUCS_SR (44100 Hz).
    """
    import torchaudio
    from demucs.apply import apply_model

    model = _load_demucs(device)

    wav = waveform.float()
    # Resample to 44100 if needed
    if src_sr != DEMUCS_SR:
        wav = torchaudio.functional.resample(wav, src_sr, DEMUCS_SR)
    # Ensure stereo [2, T]
    if wav.ndim == 1:
        wav = wav.unsqueeze(0).repeat(2, 1)
    elif wav.shape[0] == 1:
        wav = wav.repeat(2, 1)

    wav = wav.to(device)
    with torch.no_grad():
        # apply_model expects [B, C, T] → returns [B, n_sources, C, T]
        sources = apply_model(model, wav.unsqueeze(0), device=device, progress=False)

    sources = sources.squeeze(0).cpu()  # [n_sources, 2, T]
    return {name: sources[i] for i, name in enumerate(model.sources)}
