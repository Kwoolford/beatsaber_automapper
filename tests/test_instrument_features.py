"""Tests for scoped-V8 per-instrument layering features (TASK 2).

Covers the pure slot-binning math (no models/GPU) and the wiring through the
BeatClassifier model + BeatLitModule.
"""

from __future__ import annotations

import torch

from beatsaber_automapper.data.instrument_features import (
    INSTR_FEATURE_DIM,
    INSTR_FEATURE_NAMES,
    NoteEvent,
    STEM_CHANNELS,
    events_to_slot_features,
)


def test_feature_dim_matches_names():
    assert INSTR_FEATURE_DIM == len(INSTR_FEATURE_NAMES) == 10
    # First 6 channels are the per-stem densities, in STEM_CHANNELS order.
    assert INSTR_FEATURE_NAMES[:6] == tuple(f"{s}_density" for s in STEM_CHANNELS)


def test_empty_events_all_zero():
    feats = events_to_slot_features([], bpm=120.0, n_slots=32)
    assert feats.shape == (32, INSTR_FEATURE_DIM)
    assert torch.count_nonzero(feats) == 0


def test_single_kick_on_grid():
    # bpm=120, subdiv=4 -> 8 slots/sec. Onset at 1.0s -> exactly slot 8.
    ev = NoteEvent(onset_sec=1.0, dur_sec=0.0, pitch=None, stem="kick", salience=0.7)
    feats = events_to_slot_features([ev], bpm=120.0, n_slots=32)
    kick_col = STEM_CHANNELS.index("kick")
    assert torch.isclose(feats[8, kick_col], torch.tensor(0.7), atol=1e-5)
    # No mass anywhere else in the kick column.
    assert torch.isclose(feats[:, kick_col].sum(), torch.tensor(0.7), atol=1e-5)
    # n_active_stems = 1/6 at slot 8.
    assert torch.isclose(feats[8, 6], torch.tensor(1.0 / 6), atol=1e-5)


def test_offgrid_onset_mass_preserved():
    # Onset at 1.0625s -> slot_f = 8.5 -> split half/half into slots 8 and 9.
    ev = NoteEvent(onset_sec=8.5 / 8.0, dur_sec=0.0, pitch=None, stem="snare", salience=1.0)
    feats = events_to_slot_features([ev], bpm=120.0, n_slots=32)
    col = STEM_CHANNELS.index("snare")
    assert torch.isclose(feats[8, col], torch.tensor(0.5), atol=1e-4)
    assert torch.isclose(feats[9, col], torch.tensor(0.5), atol=1e-4)
    # Total mass conserved == salience.
    assert torch.isclose(feats[:, col].sum(), torch.tensor(1.0), atol=1e-4)


def test_lead_pitch_and_contour():
    # Two lead onsets, ascending pitch: slot 0 (A4=69) then slot 8 (A5=81, +12 semis).
    evs = [
        NoteEvent(0.0, 0.1, 69, "lead", 0.9),
        NoteEvent(1.0, 0.1, 81, "lead", 0.9),
    ]
    feats = events_to_slot_features(evs, bpm=120.0, n_slots=32)
    # lead_pitch channel (idx 7) is normalized and nonzero where lead events land.
    assert feats[0, 7] > 0
    assert feats[8, 7] > feats[0, 7]          # higher pitch -> higher normalized value
    # lead_dpitch (idx 8): first event has no predecessor (0); second is +1 semitone/oct up.
    assert torch.isclose(feats[0, 8], torch.tensor(0.0), atol=1e-6)
    assert feats[8, 8] > 0                      # ascending -> positive tanh step


def test_higher_salience_wins_pitch_slot():
    # Two lead events in the same slot; the higher-salience pitch is kept.
    evs = [
        NoteEvent(0.0, 0.1, 60, "lead", 0.2),
        NoteEvent(0.0, 0.1, 72, "lead", 0.9),
    ]
    feats = events_to_slot_features(evs, bpm=120.0, n_slots=8)
    from beatsaber_automapper.data.instrument_features import _norm_pitch
    assert torch.isclose(feats[0, 7], torch.tensor(_norm_pitch(72)), atol=1e-5)


def test_bass_pitch_channel():
    ev = NoteEvent(0.0, 0.2, 40, "bass", 0.8)
    feats = events_to_slot_features([ev], bpm=120.0, n_slots=8)
    assert feats[0, 9] > 0                      # bass_pitch
    assert feats[0, STEM_CHANNELS.index("bass")] > 0  # bass_density


def test_unknown_stem_ignored():
    ev = NoteEvent(0.0, 0.0, None, "tambourine", 1.0)
    feats = events_to_slot_features([ev], bpm=120.0, n_slots=8)
    assert torch.count_nonzero(feats) == 0


# ---------------------------------------------------------------------------
# Model / module wiring
# ---------------------------------------------------------------------------
def test_beat_classifier_accepts_instr_features():
    from beatsaber_automapper.models.beat_classifier import BeatClassifier

    model = BeatClassifier(instr_dim=INSTR_FEATURE_DIM, d_model=32, n_heads=2, n_layers=1)
    B, W = 2, 16
    drum = torch.randn(B, W, 768)
    mix = torch.randn(B, W, 768)
    struct = torch.randn(B, W, 8)
    instr = torch.randn(B, W, INSTR_FEATURE_DIM)
    out = model(drum, mix, difficulty=3, slot_offset=0, struct_features=struct, instr_features=instr)
    assert out.shape == (B, W, 2)
    # instr path actually contributes: zeroing it changes the output.
    out_zero = model(drum, mix, difficulty=3, slot_offset=0, struct_features=struct,
                     instr_features=torch.zeros_like(instr))
    assert not torch.allclose(out, out_zero)


def test_beat_classifier_instr_optional():
    # instr_dim=0 -> no instr_proj; passing instr_features is simply ignored.
    from beatsaber_automapper.models.beat_classifier import BeatClassifier

    model = BeatClassifier(instr_dim=0, d_model=32, n_heads=2, n_layers=1)
    assert model.instr_proj is None
    out = model(torch.randn(1, 8, 768), instr_features=torch.randn(1, 8, INSTR_FEATURE_DIM))
    assert out.shape == (1, 8, 2)


def test_beat_module_training_step_with_instr():
    from beatsaber_automapper.training.beat_module import BeatLitModule

    module = BeatLitModule(instr_dim=INSTR_FEATURE_DIM, d_model=32, n_heads=2, n_layers=1)
    B, W = 2, 16
    batch = {
        "drum_features":   torch.randn(B, W, 768),
        "mix_features":    torch.randn(B, W, 768),
        "struct_features": torch.randn(B, W, 8),
        "instr_features":  torch.randn(B, W, INSTR_FEATURE_DIM),
        "left_labels":     torch.randint(0, 2, (B, W)),
        "right_labels":    torch.randint(0, 2, (B, W)),
        "slot_offset":     torch.zeros(B, dtype=torch.long),
        "difficulty":      torch.full((B,), 3, dtype=torch.long),
    }
    loss = module.training_step(batch, 0)
    assert loss.requires_grad and torch.isfinite(loss)
