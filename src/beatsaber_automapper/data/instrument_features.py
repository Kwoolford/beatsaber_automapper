"""Scoped V8 (TASK 2): per-instrument event-density features per beat-grid slot.

Pipeline:
    audio -> Demucs (drums/bass/other/vocals)
          -> drums : multi-band librosa onset      -> kick/snare/hat events
          -> bass  : basic-pitch                   -> pitched events
          -> vocals: basic-pitch                   -> pitched events
          -> other : basic-pitch (salience gate + chord-merge) -> lead events
          -> merged NoteEvent stream
          -> events_to_slot_features() -> [N_slots, INSTR_FEATURE_DIM]

The slot grid matches ``mert_encoder.pool_to_beat_grid`` / ``beat_grid`` exactly
(1/BEAT_SUBDIV-note slots anchored to a single global BPM), so the layering
features line up row-for-row with ``drum_beat_features`` / ``mix_beat_features``
and the Stage-1 binary labels.

Design note (user feedback 2026-06-03): pass the FULL per-instrument layering
vector — drums *and* bass/synth/vocal activity plus the lead/bass pitch contour —
and let the model weight it per genre. EDM structure rides on bass/synth layering,
rock on drums; we don't pre-pick a stem. The V8-0 structure test measured drum
density Spearman r=0.41 vs human note density (> the section detector's 0.27), so
these features are a better density/structure signal than the hand-tuned section
gate that produced the silent-drop bug.

The transcription core is shared with ``scripts/v8_poc.py`` (the V8-0 gate PoC);
heavy deps (Demucs, basic-pitch) are lazy-imported so the pure feature math and
the package import stay cheap and unit-testable without a GPU.
"""

from __future__ import annotations

import dataclasses
import logging
import os
import tempfile

import numpy as np
import torch

from beatsaber_automapper.data.beat_grid import BEAT_SUBDIV

logger = logging.getLogger(__name__)

# Pitched stems get basic-pitch; the drum stem gets multi-band onset detection.
PITCHED_STEMS = ("bass", "vocals", "other")
DRUM_STEM = "drums"

# Per-slot feature layout. Order is fixed — preprocessing writes it, the dataset
# reads it, so any change here is a breaking change to the cached .pt key.
STEM_CHANNELS = ("kick", "snare", "hat", "bass", "vocals", "lead")
INSTR_FEATURE_NAMES = (
    *(f"{s}_density" for s in STEM_CHANNELS),  # 0..5  salience-weighted activity
    "n_active_stems",                          # 6     # stems active in slot / len(STEM_CHANNELS)
    "lead_pitch",                              # 7     normalized lead MIDI (0 if none)
    "lead_dpitch",                             # 8     tanh-scaled semitone step vs prev lead
    "bass_pitch",                              # 9     normalized bass MIDI (0 if none)
)
INSTR_FEATURE_DIM = len(INSTR_FEATURE_NAMES)  # 10

# Piano MIDI range for pitch normalization (A0=21 .. C8=108, 88 keys).
_MIDI_LO, _MIDI_HI = 21, 108


@dataclasses.dataclass
class NoteEvent:
    """A single transcribed musical event in continuous time (not a grid slot)."""

    onset_sec: float
    dur_sec: float
    pitch: int | None        # MIDI pitch; None for unpitched drum hits
    stem: str                # one of STEM_CHANNELS
    salience: float          # transcription confidence x amplitude, roughly [0, 1]


def _norm_pitch(p: int) -> float:
    return float(np.clip((p - _MIDI_LO) / (_MIDI_HI - _MIDI_LO), 0.0, 1.0))


# ---------------------------------------------------------------------------
# Pure feature math (unit-testable, no models)
# ---------------------------------------------------------------------------
def events_to_slot_features(
    events: list[NoteEvent],
    bpm: float,
    n_slots: int,
    subdiv: int = BEAT_SUBDIV,
) -> torch.Tensor:
    """Bin a NoteEvent stream into per-slot layering features.

    Density channels use mass-preserving linear interpolation across the two
    nearest slots, so an onset a little off the grid still lands its energy
    (robust to small BPM/phase jitter, the V8-0 Layer-2 concern). Pitch channels
    snap to the nearest slot (pitch is not additive) and keep the highest-salience
    event when several map to one slot.

    Args:
        events:  transcribed NoteEvents (continuous-time).
        bpm:     song tempo.
        n_slots: number of grid slots (match ``drum_beat_features.shape[0]``).
        subdiv:  slots per beat (4 = 1/4 note).

    Returns:
        Tensor[n_slots, INSTR_FEATURE_DIM] float32.
    """
    feats = torch.zeros(n_slots, INSTR_FEATURE_DIM, dtype=torch.float32)
    if n_slots <= 0 or bpm <= 0:
        return feats

    stem_col = {s: i for i, s in enumerate(STEM_CHANNELS)}
    slots_per_sec = bpm / 60.0 * subdiv
    presence = np.zeros((n_slots, len(STEM_CHANNELS)), dtype=bool)

    # Track best (highest-salience) lead/bass pitch per slot for the contour channels.
    lead_best = {}  # slot -> (salience, pitch)
    bass_best = {}

    for ev in events:
        col = stem_col.get(ev.stem)
        if col is None:
            continue
        slot_f = ev.onset_sec * slots_per_sec
        s0 = int(np.floor(slot_f))
        frac = slot_f - s0
        sal = float(ev.salience)
        # Linear-interpolated density into s0 and s0+1.
        for s, w in ((s0, 1.0 - frac), (s0 + 1, frac)):
            if 0 <= s < n_slots and w > 0:
                feats[s, col] += sal * w
                presence[s, col] = True
        # Pitch channels snap to nearest slot.
        snap = int(round(slot_f))
        if 0 <= snap < n_slots and ev.pitch is not None:
            if ev.stem == "lead":
                if snap not in lead_best or sal > lead_best[snap][0]:
                    lead_best[snap] = (sal, ev.pitch)
            elif ev.stem == "bass":
                if snap not in bass_best or sal > bass_best[snap][0]:
                    bass_best[snap] = (sal, ev.pitch)

    # n_active_stems (normalized).
    feats[:, 6] = torch.from_numpy(presence.sum(axis=1).astype(np.float32)) / len(STEM_CHANNELS)

    # Lead pitch + signed step contour (Δ vs the previous lead onset, in onset order).
    prev_lead = None
    for snap in sorted(lead_best):
        _, pitch = lead_best[snap]
        feats[snap, 7] = _norm_pitch(pitch)
        if prev_lead is not None:
            feats[snap, 8] = float(np.tanh((pitch - prev_lead) / 12.0))
        prev_lead = pitch

    for snap, (_, pitch) in bass_best.items():
        feats[snap, 9] = _norm_pitch(pitch)

    return feats


# ---------------------------------------------------------------------------
# Transcription (lazy heavy imports)
# ---------------------------------------------------------------------------
def transcribe_pitched(
    y: np.ndarray,
    sr: int,
    stem_name: str,
    salience_tau: float = 0.0,
    chord_merge_ms: float = 0.0,
) -> list[NoteEvent]:
    """basic-pitch transcription of one pitched stem (bass/vocals/other)."""
    import soundfile as sf
    from basic_pitch.inference import predict
    from basic_pitch import ICASSP_2022_MODEL_PATH

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
        tmp_path = tf.name
    try:
        sf.write(tmp_path, y, sr)
        _, _, note_events = predict(tmp_path, ICASSP_2022_MODEL_PATH)
    finally:
        os.unlink(tmp_path)

    out_stem = "lead" if stem_name == "other" else stem_name
    events: list[NoteEvent] = [
        NoteEvent(float(start), float(end) - float(start), int(pitch), out_stem, float(amp))
        for start, end, pitch, amp, _bends in note_events
    ]
    events.sort(key=lambda e: e.onset_sec)

    if salience_tau > 0.0 and events:
        thr = salience_tau * max(e.salience for e in events)
        events = [e for e in events if e.salience >= thr]
    if chord_merge_ms > 0.0 and events:
        events = _chord_merge(events, chord_merge_ms / 1000.0)
    return events


def _chord_merge(events: list[NoteEvent], window_sec: float) -> list[NoteEvent]:
    """Collapse onsets within ``window_sec`` into the single highest-salience event."""
    merged: list[NoteEvent] = []
    cluster: list[NoteEvent] = []
    for ev in events:  # onset-sorted
        if cluster and ev.onset_sec - cluster[0].onset_sec <= window_sec:
            cluster.append(ev)
        else:
            if cluster:
                merged.append(max(cluster, key=lambda e: e.salience))
            cluster = [ev]
    if cluster:
        merged.append(max(cluster, key=lambda e: e.salience))
    return merged


def transcribe_drums(y: np.ndarray, sr: int) -> list[NoteEvent]:
    """Multi-band onset detection on the drum stem -> kick/snare/hat events."""
    import librosa
    from scipy.signal import butter, sosfiltfilt

    bands = {
        "kick":  ("low", 0.0, 150.0),
        "snare": ("band", 150.0, 2000.0),
        "hat":   ("high", 6000.0, 0.0),
    }
    events: list[NoteEvent] = []
    nyq = sr / 2.0
    for name, (kind, lo, hi) in bands.items():
        if kind == "low":
            sos = butter(4, hi / nyq, btype="low", output="sos")
        elif kind == "high":
            sos = butter(4, lo / nyq, btype="high", output="sos")
        else:
            sos = butter(4, [lo / nyq, hi / nyq], btype="band", output="sos")
        yb = sosfiltfilt(sos, y).astype(np.float32)
        env = librosa.onset.onset_strength(y=yb, sr=sr, hop_length=512)
        onsets = librosa.onset.onset_detect(
            onset_envelope=env, sr=sr, hop_length=512, units="time", backtrack=True
        )
        frames = np.clip(librosa.time_to_frames(onsets, sr=sr, hop_length=512), 0, len(env) - 1)
        env_max = float(env.max()) if env.size else 1.0
        for t, fr in zip(onsets, frames):
            events.append(NoteEvent(float(t), 0.0, None, name, float(env[fr]) / (env_max + 1e-9)))
    events.sort(key=lambda e: e.onset_sec)
    return events


def transcribe_stems(
    stems: dict[str, np.ndarray],
    sr: int,
    salience_tau: float = 0.10,
    chord_merge_ms: float = 40.0,
) -> list[NoteEvent]:
    """Run the full per-stem transcription over a Demucs stem dict."""
    events: list[NoteEvent] = []
    for stem in PITCHED_STEMS:
        if stem not in stems:
            continue
        tau = salience_tau if stem == "other" else 0.0
        merge = chord_merge_ms if stem == "other" else 0.0
        events.extend(transcribe_pitched(stems[stem], sr, stem, salience_tau=tau, chord_merge_ms=merge))
    if DRUM_STEM in stems:
        events.extend(transcribe_drums(stems[DRUM_STEM], sr))
    events.sort(key=lambda e: e.onset_sec)
    return events


def separate_to_stems(waveform: torch.Tensor, src_sr: int, device: str | None = None) -> tuple[dict[str, np.ndarray], int]:
    """Demucs-separate a waveform into mono float32 stems. Returns (stems, sr)."""
    from beatsaber_automapper.data.stem_separator import separate, DEMUCS_SR

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    stems = separate(waveform, src_sr, device=device)
    out: dict[str, np.ndarray] = {}
    for name, stem in stems.items():
        arr = stem.detach().cpu().numpy().astype(np.float32)
        if arr.ndim == 2:        # [channels, samples] -> mono
            arr = arr.mean(axis=0)
        out[name] = arr
    return out, DEMUCS_SR


def compute_instrument_features(
    waveform: torch.Tensor,
    src_sr: int,
    bpm: float,
    n_slots: int,
    subdiv: int = BEAT_SUBDIV,
    device: str | None = None,
) -> torch.Tensor:
    """End-to-end: waveform -> stems -> transcription -> [n_slots, INSTR_FEATURE_DIM]."""
    stems, sr = separate_to_stems(waveform, src_sr, device=device)
    events = transcribe_stems(stems, sr)
    return events_to_slot_features(events, bpm, n_slots, subdiv=subdiv)
