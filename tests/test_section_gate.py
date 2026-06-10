"""Tests for the Stage-1 section-gate threshold logic (V8-0 silent-drop fix).

`_build_section_threshold_vector` decides how detected sections modulate the
Stage-1 onset threshold. The safety-critical invariant is that the default
``loud_only`` gate can NEVER raise the threshold above the base — so no
mislabeled section can silence a real drop (the user's headline complaint).
"""

from __future__ import annotations

import torch

from beatsaber_automapper.generation.generate import (
    _build_section_threshold_vector,
    _SECTION_THRESHOLDS,
)

# 120 BPM, subdiv 4 -> 8 slots/sec. A 4-second song = 32 slots.
BPS = 120.0 / 60.0
SUBDIV = 4
N_SLOTS = 32
# A drop mislabeled as "intro" over the first 2 s — the exact silent-drop setup.
SECTIONS = [("intro", 0.0, 2.0), ("drop", 2.0, 4.0)]
BASE = 0.50


def _vec(gate: str):
    return _build_section_threshold_vector(
        SECTIONS, N_SLOTS, BASE, BASE, BPS, SUBDIV, gate
    )


def test_loud_only_never_exceeds_base() -> None:
    thr_L, thr_R = _vec("loud_only")
    assert torch.all(thr_L <= BASE + 1e-6)
    assert torch.all(thr_R <= BASE + 1e-6)


def test_loud_only_keeps_mislabeled_intro_unsilenced() -> None:
    # The "intro" region (slots 0..15) must stay at base (0.50), NOT 0.68.
    thr_L, _ = _vec("loud_only")
    intro_region = thr_L[:16]
    assert torch.allclose(intro_region, torch.full_like(intro_region, BASE))


def test_loud_only_still_densifies_drop() -> None:
    # The "drop" region keeps its low 0.38 threshold (denser than base).
    thr_L, _ = _vec("loud_only")
    drop_region = thr_L[16:]
    assert torch.allclose(drop_region, torch.full_like(drop_region, _SECTION_THRESHOLDS["drop"]))


def test_off_is_flat_base_everywhere() -> None:
    thr_L, thr_R = _vec("off")
    assert torch.allclose(thr_L, torch.full_like(thr_L, BASE))
    assert torch.allclose(thr_R, torch.full_like(thr_R, BASE))


def test_legacy_reproduces_silencing_intro() -> None:
    # Legacy is the OLD behavior: intro raised to 0.68 (this is the bug we keep
    # only for A/B comparison).
    thr_L, _ = _vec("legacy")
    intro_region = thr_L[:16]
    assert torch.allclose(intro_region, torch.full_like(intro_region, _SECTION_THRESHOLDS["intro"]))


def test_invalid_gate_raises() -> None:
    try:
        _vec("nonsense")
    except ValueError:
        return
    raise AssertionError("expected ValueError for invalid section_gate")
