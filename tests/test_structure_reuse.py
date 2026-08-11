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


def test_min_run_keeps_only_contiguous_sections():
    """The fix the first arm's failure named: copy a SECTION, not scattered bars.

    me_z20 broke flow (0.37 -> 0.75) and idiom (0.40 -> 1.07) because only 15.6 % of
    its copies continued the previous bar's copy — it shuffled bars in from all over
    the song rather than reusing a passage.
    """
    n = 24
    edges = np.arange(n + 1) * BAR_S
    M = np.full((n, n), 0.05)
    # bars 12..15 are a contiguous return of 0..3 — a real section repeat
    for k in range(4):
        M[12 + k, k] = M[k, 12 + k] = 0.95
    # bar 20 matches bar 6 alone — an isolated coincidence, not a section
    M[20, 6] = M[6, 20] = 0.95
    for i in range(n):
        M[i, i] = 1.0
    S = {"harm": M, "rhy": M, "timb": M, "energy": np.ones(n)}

    loose = sr.plan_reuse(S, edges, min_sim=0.6, min_lag=4, min_z=2.0, min_run=1)
    assert 20 in loose.source, "fixture: the isolated match must be found at min_run=1"

    strict = sr.plan_reuse(S, edges, min_sim=0.6, min_lag=4, min_z=2.0, min_run=3)
    assert 20 not in strict.source, "an isolated bar copy must be dropped"
    assert strict.source, "the contiguous section must survive"
    for t in strict.source:
        prev, nxt = t - 1, t + 1
        assert (prev in strict.source and strict.source[prev] == strict.source[t] - 1) \
            or (nxt in strict.source and strict.source[nxt] == strict.source[t] + 1), \
            f"bar {t} survived without a neighbour advancing with it"


def test_min_run_default_preserves_old_behaviour():
    S, edges = _scene()
    a = sr.plan_reuse(S, edges, min_sim=0.6, min_lag=4, min_z=0.0)
    b = sr.plan_reuse(S, edges, min_sim=0.6, min_lag=4, min_z=0.0, min_run=1)
    assert a.source == b.source


def test_diagonal_planner_recovers_a_contiguous_section():
    """A repeated section is a diagonal stripe — decode the stripe, not each bar.

    The per-bar planner shipped as me_z20 and broke flow/idiom because it chose each
    bar's source independently and only 15.6 % of copies continued the previous one.
    """
    n = 32
    edges = np.arange(n + 1) * BAR_S
    M = np.full((n, n), 0.05)
    for k in range(8):                       # bars 16..23 return bars 0..7
        M[16 + k, k] = M[k, 16 + k] = 0.95
    for i in range(n):
        M[i, i] = 1.0
    S = {"harm": M, "rhy": M, "timb": M, "energy": np.ones(n)}

    plan = sr.plan_reuse_diagonal(S, edges, min_sim=0.6, min_lag=4, min_run=4)
    assert plan.source, "the stripe must be found"
    for t, s in plan.source.items():
        assert t - s == 16, "every bar of one section must share a single lag"
    contiguous = sum(1 for t in plan.source
                     if t - 1 in plan.source and plan.source[t - 1] == plan.source[t] - 1)
    assert contiguous >= len(plan.source) - 1, "the copy must advance bar for bar"


def test_diagonal_planner_ignores_an_isolated_coincidence():
    n = 32
    edges = np.arange(n + 1) * BAR_S
    M = np.full((n, n), 0.05)
    M[20, 6] = M[6, 20] = 0.99                # one bar, no section around it
    for i in range(n):
        M[i, i] = 1.0
    S = {"harm": M, "rhy": M, "timb": M, "energy": np.ones(n)}
    plan = sr.plan_reuse_diagonal(S, edges, min_sim=0.6, min_lag=4, min_run=4)
    assert plan.n_copied == 0


def test_diagonal_planner_never_claims_a_bar_twice():
    n = 40
    edges = np.arange(n + 1) * BAR_S
    M = np.full((n, n), 0.05)
    for k in range(8):
        M[16 + k, k] = M[k, 16 + k] = 0.95
        M[32 + k if 32 + k < n else n - 1, k] = 0.90
    for i in range(n):
        M[i, i] = 1.0
    S = {"harm": M, "rhy": M, "timb": M, "energy": np.ones(n)}
    plan = sr.plan_reuse_diagonal(S, edges, min_sim=0.6, min_lag=4, min_run=4)
    assert len(plan.source) == len(set(plan.source)), "a bar was assigned twice"
    assert all(s < t for t, s in plan.source.items()), "a bar may only copy the PAST"


def test_diag_prefix_selects_diagonal_planner_and_keeps_the_mode(monkeypatch):
    """`diag_full` must mean diagonal planning AND rhythm copying, not one or the other.

    Checked because it is the spec the most valuable arm runs under, and a silent
    fallback to per-bar planning would make that arm a repeat of the one it is meant to
    improve on — which no number in the report would reveal.
    """
    seen = {}
    real_diag = sr.plan_reuse_diagonal
    real_apply = sr.apply_reuse

    def spy_diag(*a, **k):
        seen["planner"] = "diagonal"
        return real_diag(*a, **k)

    def spy_apply(bm, plan, bpm, mode="place"):
        seen["mode"] = mode
        return real_apply(bm, plan, bpm, mode=mode)

    monkeypatch.setattr(sr, "plan_reuse_diagonal", spy_diag)
    monkeypatch.setattr(sr, "apply_reuse", spy_apply)
    monkeypatch.setenv("BEAT_STRUCTURE_REUSE", "diag_full:0.70:4:1.5:2.0:4")

    n = 32
    edges = np.arange(n + 1) * BAR_S
    M = np.full((n, n), 0.05)
    for k in range(8):
        M[16 + k, k] = M[k, 16 + k] = 0.95
    S = {"harm": M, "rhy": M, "timb": M, "energy": np.ones(n)}
    monkeypatch.setattr(sr, "bar_edges", lambda *a, **k: edges)
    monkeypatch.setattr(sr, "audio_bar_ssm", lambda *a, **k: S)

    bm = DifficultyBeatmap(version="3.0", color_notes=_notes([(0, (1, 0, 1))]))
    sr.maybe_apply(bm, np.zeros(1000, dtype="float32"), 44100, {}, BPM, 60.0)

    assert seen.get("planner") == "diagonal", "diag_ prefix did not select the stripe planner"
    assert seen.get("mode") == "full", "the mode after the diag_ prefix was lost"
