"""The masterpiece axes (M1–M3) — the properties they are only useful if they have.

These axes are contrasts, and the whole argument for them is that a degenerate map
scores **0 by construction** rather than by luck. That argument is worth exactly as
much as a test of it, so the load-bearing properties are pinned here:

  * a map with no relation to the audio scores ~0, not something small-but-positive;
  * a map that ANSWERS the audio's repeat structure scores clearly positive;
  * the rhythm similarity is chance-corrected, so two bars that merely share a note
    COUNT score 0 — with cosine they did not, and our maps beat the humans on Hunger
    because of it;
  * the contrast is computed inside bar-distance strata, so a map whose similarity
    decays with time cannot look like a map that answers repeats;
  * `paired_delta` refuses to compare across song sets.

They run on synthetic maps and a synthetic audio SSM: no audio, no checkpoints, no
cache — so a failure here is a real regression in the estimator rather than a
missing artefact.
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np
import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import song_structure as ss  # noqa: E402
from eval_rhythm_fidelity import kappa, quantise  # noqa: E402


def make_bars(n: int = 64, dur: float = 2.0) -> ss.Bars:
    edges = np.arange(n + 1) * dur
    return ss.Bars(edges=edges, period=dur / 4, ratio=1.0, beats_per_bar=4,
                   confidence="high", f1=1.0)


def notes_from_pattern(B: ss.Bars, patterns, slots: int = ss.SLOTS_PER_BAR):
    """patterns[i] = iterable of slot indices played in bar i."""
    out = []
    for i, slots_played in enumerate(patterns):
        for s in slots_played:
            t = B.edges[i] + (s / slots) * B.dur
            out.append((float(t), s % 4, (s // 4) % 3, s % 9, s % 2))
    return sorted(out)


def block_ssm(n: int, block: int) -> np.ndarray:
    """Audio SSM for a song made of alternating sections of `block` bars."""
    lab = (np.arange(n) // block) % 2
    return np.where(lab[:, None] == lab[None, :], 1.0, -1.0)


# --------------------------------------------------------------- kappa

def test_kappa_is_chance_corrected_not_a_density_match():
    """Two bars with the SAME number of notes in DIFFERENT slots must score ~0.

    This is the property cosine lacked. `DENSITY_SELECT` makes our note count track
    loudness, so similar-sounding bars hold similar counts; under cosine that read
    as "the same pattern" and our maps out-scored the humans.
    """
    a = np.zeros(16)
    b = np.zeros(16)
    a[[0, 4, 8, 12]] = 1
    b[[1, 5, 9, 13]] = 1
    assert abs(kappa(a, b)) < 0.35          # not credited for matching density
    assert kappa(a, a) == pytest.approx(1.0)


def test_kappa_rewards_identical_patterns_only():
    a = np.zeros(16)
    a[[0, 3, 6, 10]] = 1
    assert kappa(a, a) > kappa(a, np.roll(a, 1))


# --------------------------------------------------- the contrast estimator

def test_map_answering_the_structure_scores_positive():
    """A map that plays pattern P in section A and Q in section B, on a song whose
    sections alternate, must score clearly positive."""
    n, block = 64, 8
    B = make_bars(n)
    P, Q = [0, 4, 8, 12], [0, 2, 3, 7, 11]
    pats = [P if (i // block) % 2 == 0 else Q for i in range(n)]
    V = ss.map_bar_vectors(notes_from_pattern(B, pats), B)
    S = ss.bar_map_similarity(V)
    got = ss.stratified_contrast(block_ssm(n, block), S["rhythm"])
    assert got["contrast"] > 0.75          # measured 1.022


def test_map_ignoring_the_structure_scores_about_zero():
    """The same song, but the map's pattern is chosen at random per bar."""
    rng = np.random.default_rng(0)
    n, block = 64, 8
    B = make_bars(n)
    pats = [sorted(rng.choice(16, 5, replace=False).tolist()) for _ in range(n)]
    V = ss.map_bar_vectors(notes_from_pattern(B, pats), B)
    S = ss.bar_map_similarity(V)
    got = ss.stratified_contrast(block_ssm(n, block), S["rhythm"])
    assert abs(got["contrast"]) < 0.05     # measured -0.004


def test_constant_map_cannot_score(monkeypatch):
    """★The metronome property. A map that is identical everywhere is equally
    similar to itself everywhere, so no contrast exists to find — whatever the
    song does. Every level metric this project built failed exactly here."""
    n, block = 64, 8
    B = make_bars(n)
    pats = [[0, 4, 8, 12]] * n
    V = ss.map_bar_vectors(notes_from_pattern(B, pats), B)
    S = ss.bar_map_similarity(V)
    got = ss.stratified_contrast(block_ssm(n, block), S["rhythm"])
    assert abs(got["contrast"]) < 1e-6


def test_stratification_removes_a_proximity_confound():
    """A map that simply drifts — each bar resembling its neighbours and nothing
    else — must not be credited on a song whose repeats are FAR apart.

    Without the bar-distance strata this is the map that scores highest, because
    near bars are both more audio-similar and more map-similar for reasons that
    have nothing to do with intent.
    """
    n, block = 64, 8
    B = make_bars(n)
    rng = np.random.default_rng(1)
    pats, cur = [], [0, 4, 8, 12]
    for _ in range(n):                      # slow random walk of the pattern
        cur = sorted({(s + int(rng.integers(-1, 2))) % 16 for s in cur})
        pats.append(list(cur))
    V = ss.map_bar_vectors(notes_from_pattern(B, pats), B)
    S = ss.bar_map_similarity(V)
    # audio: repeats at a LONG lag only (sections alternate every `block` bars)
    got = ss.stratified_contrast(block_ssm(n, block), S["rhythm"])
    assert got["contrast"] < 0.10          # measured -0.009: the strata remove it fully


# ------------------------------------------------------------ quantisation

def test_quantise_puts_a_note_in_the_bar_it_belongs_to():
    B = make_bars(8)
    times = np.array([0.0, B.dur * 0.5, B.dur * 1.25])
    M = quantise(times, B)
    assert M[0, 0] == 1
    assert M[0, ss.SLOTS_PER_BAR // 2] == 1
    assert M[1, ss.SLOTS_PER_BAR // 4] == 1
    assert M.sum() == 3


# ------------------------------------------------------------ paired_delta

def test_paired_delta_uses_only_songs_present_on_both_sides():
    rows = [{"ours": {"m": 0.2}, "human": {"m": 0.5}},
            {"ours": {"m": 0.3}, "human": {"m": 0.6}},
            {"ours": {"m": 0.1}, "human": None},          # no human map
            {"ours": {"m": 0.4}, "human": {"m": 0.7}},
            {"ours": {"m": 0.2}, "human": {"m": 0.5}},
            {"ours": {"m": 0.3}, "human": {"m": 0.6}},
            {"ours": {"m": 0.2}, "human": {"m": 0.5}}]
    d = ss.paired_delta(rows, "m")
    assert d["n"] == 6                       # the unpaired song is dropped
    assert d["delta"] == pytest.approx(-0.3, abs=1e-6)
    assert d["sign_consistent"] is True


def test_paired_delta_reports_median_beside_mean():
    """One outlier must not be able to speak for the cohort unnoticed — on
    `hands_x_downbeat` the mean said −0.387 and the median −0.111."""
    rows = [{"ours": {"m": 0.0}, "human": {"m": 0.1}} for _ in range(6)]
    rows.append({"ours": {"m": 0.0}, "human": {"m": 9.0}})
    d = ss.paired_delta(rows, "m")
    assert d["delta_median"] == pytest.approx(-0.1)
    assert d["delta"] < -1.0


def test_paired_delta_refuses_a_tiny_sample():
    rows = [{"ours": {"m": 0.1}, "human": {"m": 0.2}} for _ in range(3)]
    assert ss.paired_delta(rows, "m") == {}
