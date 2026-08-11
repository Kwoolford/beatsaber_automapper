"""M-E structure-conditioned decode — the properties the lever CLAIMS, asserted.

The claims that matter for reading tonight's arms are structural, not statistical:
`place` mode must be provably time-neutral (otherwise every "nothing regressed" number
is unearned), and the lever must be a no-op when unset (otherwise the control arm is
not a control).
"""

import sys
import pathlib

import numpy as np
import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from beatsaber_automapper.data.beatmap import ColorNote, DifficultyBeatmap  # noqa: E402
from beatsaber_automapper.generation import structure_reuse as sr  # noqa: E402

BPM = 120.0
SPB = 60.0 / BPM
BAR_S = 2.0


def _ssm(n, period=8, hi=1.0, lo=0.1):
    M = np.full((n, n), lo)
    for i in range(n):
        for j in range(n):
            if i % period == j % period:
                M[i, j] = hi
    return M


def _scene(n=16):
    edges = np.arange(n + 1) * BAR_S
    S = {"harm": _ssm(n), "rhy": _ssm(n), "timb": _ssm(n), "energy": np.ones(n)}
    return S, edges


def _notes(bars_and_place, slots=(0, 4, 8)):
    out = []
    for bi, (x, y, d) in bars_and_place:
        for s in slots:
            t = bi * BAR_S + s * (BAR_S / 16)
            out.append(ColorNote(beat=t / SPB, x=x, y=y, color=0, direction=d))
    return out


def test_plan_finds_the_periodic_repeat():
    S, edges = _scene()
    plan = sr.plan_reuse(S, edges, min_sim=0.6, min_lag=4, min_z=0.0)
    assert plan.source, "an exactly periodic song must produce copies"
    assert all(t > s for t, s in plan.source.items()), "a bar may only copy the PAST"
    assert all((t - s) >= 4 for t, s in plan.source.items()), "min_lag must hold"


def test_root_resolution_collapses_a_chain():
    """Three returns of one section must all point at the ORIGINAL, not at each other.

    This is what keeps a repeated chorus reading as sharp discrete squares in the
    structure panel instead of a chain of drifting copies.
    """
    S, edges = _scene(n=32)
    plan = sr.plan_reuse(S, edges, min_sim=0.6, min_lag=4, min_z=0.0)
    for tgt, src in plan.source.items():
        assert src not in plan.source, f"bar {tgt} copies {src}, which is itself a copy"


def test_min_lag_blocks_the_neighbour_confound():
    """Adjacent bars are similar for reasons that are not musical form."""
    n = 16
    edges = np.arange(n + 1) * BAR_S
    M = np.full((n, n), 0.05)
    for i in range(n):
        for j in range(n):
            if abs(i - j) <= 2:                     # local autocorrelation only
                M[i, j] = 0.95
    S = {"harm": M, "rhy": M, "timb": M, "energy": np.ones(n)}
    plan = sr.plan_reuse(S, edges, min_sim=0.6, min_lag=4, min_z=0.0)
    assert plan.n_copied == 0, "a purely local similarity must not count as a repeat"


def test_distinctiveness_rejects_the_uniform_song():
    """A bar that resembles every earlier bar equally resembles none in particular.

    Guards the failure the first smoke test found: a bare threshold flagged 76-88 % of
    bars on real songs, which is the uniform-blob defect, not the fix for it.
    """
    n = 20
    edges = np.arange(n + 1) * BAR_S
    M = np.full((n, n), 0.9)                        # everything matches everything
    S = {"harm": M, "rhy": M, "timb": M, "energy": np.ones(n)}
    plan = sr.plan_reuse(S, edges, min_sim=0.6, min_lag=4, min_z=2.5)
    assert plan.n_copied == 0
    # ...and with the guard off, the same song is flagged wholesale — i.e. the test
    # above is testing the guard, not an accident of the fixture.
    assert sr.plan_reuse(S, edges, min_sim=0.6, min_lag=4, min_z=0.0).n_copied > 0


def test_views_must_agree():
    """Same chords, different groove = a different section to a mapper."""
    n = 16
    edges = np.arange(n + 1) * BAR_S
    S = {"harm": _ssm(n, hi=0.95), "rhy": np.full((n, n), 0.1),
         "timb": _ssm(n), "energy": np.ones(n)}
    plan = sr.plan_reuse(S, edges, min_sim=0.6, min_lag=4, min_z=0.0)
    assert plan.n_copied == 0, "harmony alone must not carry a copy"


def test_energy_guard_blocks_a_loudness_mismatch():
    S, edges = _scene()
    S["energy"] = np.array([0.02] * 8 + [1.0] * 8)      # quiet intro, loud return
    plan = sr.plan_reuse(S, edges, min_sim=0.6, min_lag=4, min_z=0.0, energy_tol=1.5)
    assert plan.n_copied == 0


def test_place_mode_copies_placement_and_moves_no_note_in_time():
    """★THE LOAD-BEARING PROPERTY. `place` cannot regress a time-domain axis.

    Alignment, rhythm (A2), density, nps and onset precision are identical to the
    control BY CONSTRUCTION. If this test fails, every "nothing regressed" claim about
    the `place` arm is unearned.
    """
    S, edges = _scene()
    plan = sr.plan_reuse(S, edges, min_sim=0.6, min_lag=4, min_z=0.0)
    notes = _notes([(0, (1, 0, 1)), (8, (3, 2, 5))])
    before = [n.beat for n in notes]
    bm = DifficultyBeatmap(version="3.0", color_notes=list(notes))

    stats = sr.apply_reuse(bm, plan, BPM, mode="place")

    assert stats["notes_added"] == 0 and stats["notes_removed"] == 0
    assert [n.beat for n in bm.color_notes] == before, "place mode moved a note in time"
    assert len(bm.color_notes) == len(before)
    tgt = [n for n in bm.color_notes if n.beat * SPB >= 16.0]
    assert tgt and all((n.x, n.y, n.direction) == (1, 0, 1) for n in tgt)


def test_place_mode_is_idempotent():
    S, edges = _scene()
    plan = sr.plan_reuse(S, edges, min_sim=0.6, min_lag=4, min_z=0.0)
    bm = DifficultyBeatmap(version="3.0",
                           color_notes=_notes([(0, (1, 0, 1)), (8, (3, 2, 5))]))
    sr.apply_reuse(bm, plan, BPM, mode="place")
    snap = [(n.beat, n.x, n.y, n.direction) for n in bm.color_notes]
    again = sr.apply_reuse(bm, plan, BPM, mode="place")
    assert again["notes_changed"] == 0
    assert [(n.beat, n.x, n.y, n.direction) for n in bm.color_notes] == snap


def test_full_mode_replaces_the_bar_rhythm():
    """`full` is the arm that CAN move rhythm — and is expected to cost precision."""
    S, edges = _scene()
    plan = sr.plan_reuse(S, edges, min_sim=0.6, min_lag=4, min_z=0.0)
    notes = _notes([(0, (1, 0, 1))], slots=(0, 2, 4, 6, 8))
    notes += _notes([(8, (3, 2, 5))], slots=(1,))
    bm = DifficultyBeatmap(version="3.0", color_notes=list(notes))
    sr.apply_reuse(bm, plan, BPM, mode="full")
    tgt = sorted(n.beat * SPB - 16.0 for n in bm.color_notes if n.beat * SPB >= 16.0)
    assert len(tgt) == 5, "the target bar must take the source bar's note COUNT"
    assert tgt == pytest.approx([0.0, 0.25, 0.5, 0.75, 1.0], abs=1e-6)


def test_empty_plan_is_a_no_op():
    edges = np.arange(9) * BAR_S
    plan = sr.ReusePlan(edges=edges, source={}, sim={}, n_bars=8)
    bm = DifficultyBeatmap(version="3.0", color_notes=_notes([(0, (1, 0, 1))]))
    snap = [(n.beat, n.x, n.y, n.direction) for n in bm.color_notes]
    stats = sr.apply_reuse(bm, plan, BPM, mode="place")
    assert stats["notes_changed"] == 0
    assert [(n.beat, n.x, n.y, n.direction) for n in bm.color_notes] == snap


def test_lever_is_off_unless_the_env_var_is_set(monkeypatch):
    """The control arm is only a control if an unset variable means untouched."""
    monkeypatch.delenv("BEAT_STRUCTURE_REUSE", raising=False)
    bm = DifficultyBeatmap(version="3.0", color_notes=_notes([(0, (1, 0, 1))]))
    assert sr.maybe_apply(bm, None, 44100, {}, BPM, 60.0) is None


def test_unparseable_spec_is_a_warning_not_a_crash(monkeypatch):
    monkeypatch.setenv("BEAT_STRUCTURE_REUSE", "place:not-a-number")
    bm = DifficultyBeatmap(version="3.0", color_notes=_notes([(0, (1, 0, 1))]))
    assert sr.maybe_apply(bm, None, 44100, {}, BPM, 60.0) is None
