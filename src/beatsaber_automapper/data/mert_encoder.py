"""MERT-v1-95M feature extractor for V7 preprocessing.

Loads the frozen pretrained model once and provides:
  - extract_features(waveform, sr) → [T_frames, 768] at 75 Hz
  - pool_to_beat_grid(features, bpm, total_beats) → [N_slots, 768]
  - phrase_fingerprints(beat_features, beats_per_phrase) → [N_phrases, 768]

The MERT model is loaded lazily on first call and cached for the process lifetime.
Expected to run with the model frozen (no gradients).
"""

from __future__ import annotations

import logging
import os
from functools import lru_cache
from typing import TYPE_CHECKING

import torch
import torchaudio

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

MERT_MODEL_ID = "m-a-p/MERT-v1-95M"
MERT_SAMPLE_RATE = 24000
MERT_HZ = 75          # output frame rate (frames per second)
# 1/4-note slots per beat. Overridable by env so the WHOLE pipeline moves together —
# `beat_grid.BEAT_SUBDIV` reads the same variable and the two must never disagree
# (generate.py imports it from BOTH modules, in three different places).
#
# ★WHY IT IS A KNOB AT ALL (2026-08-13): on the 28 wide-cohort songs detected at HALF
# the true tempo, our maps are capped at EXACTLY 0.500x the human's burst rate
# (p10 = 0.500 — >=90% sit exactly at half), because our minimum swing gap is one grid
# slot and at half tempo that slot is twice as long in real time. Doubling the
# subdivision there restores the TRAINING-TIME slot exactly: 174 bpm at subdiv 4 is
# 86.2 ms, and 87 bpm at subdiv 8 is also 86.2 ms. Stage-1 is a transformer over the
# slot axis, so the length is free and no retrain is implied.
# ⚠️It does NOT fix beat-keyed windows — a 16-beat phrase is still 2x too long at half
# tempo. Already true today; not made worse.
# ⚠️Raising this on a CORRECTLY detected song hands the model a finer grid than it was
# ever trained on. Only use it where the tempo is believed to be an octave error.
BEAT_SUBDIV = int(os.environ.get("BEAT_SUBDIV", "4"))
_BEAT_SUBDIV_DEFAULT = BEAT_SUBDIV


def set_beat_subdiv(n: int) -> int:
    """Set the grid subdivision for the current process. Returns the value set.

    ★**This module is the single source of truth.** `beat_grid` re-exports it and
    `generate.py` reads it from here, because the value has to be chosen *after*
    `BEAT_TEMPO_FIT` — the post-fit bpm separates half-tempo songs from correct ones
    far better than anything available before the fit (0 false positives at bpm<95 vs
    5 on the raw detection), and `fit_tempo` is where the octave correction actually
    happens.

    ⚠️**Mutating a module global is process state.** Generation normally runs one song
    per process (`build_wide_cohort.py` spawns a fresh `generate.py` per map), but any
    caller that generates twice in one process MUST call `reset_beat_subdiv()` first
    or the second song inherits the first song's grid. `generate_v7_level` does this.
    """
    n = int(n)
    if n < 1 or n > 16:
        raise ValueError(f"BEAT_SUBDIV must be in 1..16, got {n}")
    global BEAT_SUBDIV
    BEAT_SUBDIV = n
    return BEAT_SUBDIV


def reset_beat_subdiv() -> int:
    """Restore the process default (the env value, or 4)."""
    global BEAT_SUBDIV
    BEAT_SUBDIV = _BEAT_SUBDIV_DEFAULT
    return BEAT_SUBDIV


@lru_cache(maxsize=1)
def _load_mert(device: str = "cuda") -> tuple:
    """Load and cache MERT processor + model. Called once per process."""
    from transformers import Wav2Vec2FeatureExtractor, AutoModel

    logger.info("Loading MERT processor from %s …", MERT_MODEL_ID)
    processor = Wav2Vec2FeatureExtractor.from_pretrained(
        MERT_MODEL_ID, trust_remote_code=True,
    )
    logger.info("Loading MERT model (frozen) …")
    model = AutoModel.from_pretrained(MERT_MODEL_ID, trust_remote_code=True)
    model = model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    logger.info("MERT loaded on %s.", device)
    return processor, model


_CHUNK_SECS = 30   # max audio seconds per MERT forward pass (avoids OOM on long songs)


def extract_features(
    waveform: torch.Tensor,
    src_sr: int,
    device: str = "cuda",
    layer: int = -1,
) -> torch.Tensor:
    """Encode a waveform with MERT-v1-95M.

    Args:
        waveform: Audio tensor [C, T] or [T] at src_sr Hz.
        src_sr:   Source sample rate.
        device:   Torch device string.
        layer:    Hidden layer index to return (-1 = final layer).

    Returns:
        Feature tensor [T_mert, 768] at MERT_HZ (75 Hz).
    """
    processor, model = _load_mert(device)

    # Mono + resample to 24 kHz
    wav = waveform.float()
    if wav.ndim == 2:
        wav = wav.mean(0)
    if src_sr != MERT_SAMPLE_RATE:
        wav = torchaudio.functional.resample(wav, src_sr, MERT_SAMPLE_RATE)

    chunk_samples = _CHUNK_SECS * MERT_SAMPLE_RATE
    total_samples = wav.shape[0]

    if total_samples <= chunk_samples:
        return _mert_forward(wav, processor, model, device, layer)

    # Long audio: process in non-overlapping chunks and concatenate
    chunks = []
    for start in range(0, total_samples, chunk_samples):
        chunk = wav[start : start + chunk_samples]
        chunks.append(_mert_forward(chunk, processor, model, device, layer))
    return torch.cat(chunks, dim=0)


def _mert_forward(
    wav: torch.Tensor,
    processor,
    model,
    device: str,
    layer: int,
) -> torch.Tensor:
    wav_np = wav.cpu().numpy()
    inputs = processor(wav_np, sampling_rate=MERT_SAMPLE_RATE,
                       return_tensors="pt", padding=True)
    input_values = inputs["input_values"].to(device)

    with torch.no_grad():
        outputs = model(input_values, output_hidden_states=True)

    hidden = outputs.hidden_states[layer]   # [1, T, 768]
    return hidden.squeeze(0).cpu()          # [T, 768]


def pool_to_beat_grid(
    mert_features: torch.Tensor,
    bpm: float,
    total_beats: float,
    subdiv: int = BEAT_SUBDIV,
) -> torch.Tensor:
    """Mean-pool MERT frame features to a 1/subdiv-note beat grid.

    Args:
        mert_features: [T_mert, D] at MERT_HZ.
        bpm:           Song tempo in beats per minute.
        total_beats:   Total song length in beats.
        subdiv:        Beat subdivisions per beat (4 = 1/4 note).

    Returns:
        [N_slots, D] where N_slots = int(total_beats * subdiv).
    """
    frames_per_slot = MERT_HZ * 60.0 / bpm / subdiv
    n_slots = int(total_beats * subdiv)
    T, D = mert_features.shape

    grid = torch.zeros(n_slots, D)
    for slot in range(n_slots):
        start = int(slot * frames_per_slot)
        end   = min(T, int((slot + 1) * frames_per_slot))
        if end > start:
            grid[slot] = mert_features[start:end].mean(0)
        elif start < T:
            grid[slot] = mert_features[start]
    return grid


def phrase_fingerprints(
    beat_features: torch.Tensor,
    beats_per_phrase: int = 16,
    subdiv: int = BEAT_SUBDIV,
) -> tuple[torch.Tensor, list[tuple[int, int]]]:
    """Compute mean MERT embedding per phrase window.

    Args:
        beat_features:    [N_slots, D] — already pooled to beat grid.
        beats_per_phrase: Phrase window size in beats (16 = 4 bars at 4/4).
        subdiv:           Slots per beat (must match pool_to_beat_grid call).

    Returns:
        (fingerprints [N_phrases, D], boundaries [(start_slot, end_slot), …])
    """
    slots_per_phrase = beats_per_phrase * subdiv
    N = beat_features.shape[0]
    fingerprints = []
    boundaries   = []
    start = 0
    while start < N:
        end = min(N, start + slots_per_phrase)
        fp  = beat_features[start:end].mean(0)  # [D]
        fingerprints.append(fp)
        boundaries.append((start, end))
        start += slots_per_phrase

    return torch.stack(fingerprints), boundaries
