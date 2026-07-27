"""Tests for the hand-role metrics (eval suite v2, axis A6).

A6 is the axis behind the project's largest measured defect (prod 3.50 vs human
0.34, worse than a uniformly random map), so its behaviour is pinned here.
"""
from __future__ import annotations

import pytest

from beatsaber_automapper.data.beatmap import ColorNote
from beatsaber_automapper.evaluation import handrole


class _BM:
    def __init__(self, notes):
        self.color_notes = notes
        self.bomb_notes = []


def _lockstep(n=200, step=0.5):
    """Both hands on every beat — our generator's actual failure mode."""
    out = []
    for i in range(n):
        b = i * step
        out.append(ColorNote(beat=b, x=0, y=1, color=0, direction=1))
        out.append(ColorNote(beat=b, x=3, y=1, color=1, direction=1))
    return out


def _lead_swap(n=200, step=0.5, window=8.0, share=0.5575):
    """One hand leads each 2-bar window by a HUMAN-sized margin, and swaps.

    `share` defaults to the value implied by the measured human reference:
    role_asymmetry 0.115 means the lead hand takes ~55.75% of a window, not 75%.
    Over-shooting asymmetry is its own failure — a map where one hand takes three
    notes in four is further from human than a perfectly balanced one — which is
    exactly what the BEAT_HAND_ROLE lever did at strength 1.0.
    """
    out = []
    n_lead = 0
    for i in range(n):
        b = i * step
        lead = int(b // window) % 2
        take_lead = (n_lead / i) < share if i else True
        n_lead += take_lead
        color = lead if take_lead else 1 - lead
        x = 0 if color == 0 else 3
        out.append(ColorNote(beat=b, x=x, y=1, color=color, direction=1))
    return out


def test_lockstep_has_no_local_asymmetry():
    """Both hands on every beat is perfectly balanced in every window.

    This is the exact shape of our generated maps, and the reason the pre-existing
    whole-map `flow.handedness` metric could not see the defect: global balance is
    identical to a human map's.
    """
    m = handrole.handrole_metrics(_BM(_lockstep())).metrics
    assert m["role_asymmetry"] == pytest.approx(0.0, abs=1e-9)


def test_lead_and_swap_produces_human_like_structure():
    m = handrole.handrole_metrics(_BM(_lead_swap())).metrics
    # a human-sized lead: clearly lopsided, but nowhere near one-sided
    assert 0.05 < m["role_asymmetry"] < 0.30
    assert m["role_swap_rate"] > 0.5          # and the lead changes between windows


def test_over_asymmetry_is_penalised_like_under_asymmetry():
    """One hand taking three notes in four is NOT more human than balance.

    Pinning this because it is the trap BEAT_HAND_ROLE fell into at strength 1.0
    (asymmetry 0.241 against a human 0.115): "give one hand the lead" is a
    distribution to match, not a direction to maximise.
    """
    ref = {k: (0.115, 0.025) for k in handrole.KEYS}
    ref["role_swap_rate"] = (0.461, 0.062)
    ref["role_run_len"] = (1.364, 0.085)
    human_like = [handrole.handrole_metrics(_BM(_lead_swap())).metrics for _ in range(5)]
    extreme = [handrole.handrole_metrics(_BM(_lead_swap(share=0.9))).metrics
               for _ in range(5)]
    g_human = handrole.cohort_comparison(human_like, ref)["_summary"]["handrole_gap"]
    g_extreme = handrole.cohort_comparison(extreme, ref)["_summary"]["handrole_gap"]
    assert g_extreme > g_human


def test_single_hand_is_lopsided_but_never_swaps():
    """The obvious failure of "let one hand lead": if it ALWAYS leads, the map is
    lopsided rather than human, which is what role_swap_rate guards."""
    notes = [ColorNote(beat=i * 0.5, x=0, y=1, color=0, direction=1)
             for i in range(200)]
    m = handrole.handrole_metrics(_BM(notes)).metrics
    assert m["role_asymmetry"] == pytest.approx(1.0)
    assert m["role_swap_rate"] == pytest.approx(0.0)


def test_run_len_is_one_when_hands_fire_together():
    """Notes on the same beat are ordered L-then-R, so a lockstep map has run
    length ~1.0 by construction. This is why role_run_len is a GUARD and not a
    composite driver — it largely restates the A2 simultaneity finding."""
    m = handrole.handrole_metrics(_BM(_lockstep())).metrics
    assert m["role_run_len"] == pytest.approx(1.0, abs=1e-9)
    assert "role_run_len" not in handrole.SEQUENCE_KEYS


def test_short_map_yields_nan_not_a_fake_score():
    notes = [ColorNote(beat=i * 0.5, x=0, y=1, color=i % 2, direction=1)
             for i in range(10)]
    m = handrole.handrole_metrics(_BM(notes)).metrics
    assert all(v != v for v in m.values())


def test_cohort_comparison_separates_lockstep_from_lead_swap():
    """The end-to-end property A6 exists for: a lockstep cohort must score worse
    than a role-divided one against the same human reference."""
    ref = {k: (0.115, 0.025) for k in handrole.KEYS}
    ref["role_swap_rate"] = (0.461, 0.062)
    ref["role_run_len"] = (1.364, 0.085)

    lock = [handrole.handrole_metrics(_BM(_lockstep())).metrics for _ in range(5)]
    role = [handrole.handrole_metrics(_BM(_lead_swap())).metrics for _ in range(5)]
    g_lock = handrole.cohort_comparison(lock, ref)["_summary"]["handrole_gap"]
    g_role = handrole.cohort_comparison(role, ref)["_summary"]["handrole_gap"]
    assert g_lock > g_role
