"""BREATHING (2026-09-03a): playing through the rest the human leaves.

Found by catching a map with a CLEAN verdict page doing it -- `LOOP__1f333` played 37 events
across seven bars where the human rests. Kyle's own words are the spec: "when there is a slow
spot we let the player breathe."
"""
from __future__ import annotations

import importlib.util
import pathlib

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[1]


def _load(name):
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


Q = _load("queries")
TQ = importlib.util.module_from_spec(
    importlib.util.spec_from_file_location("tq", pathlib.Path(__file__).with_name("test_queries.py")))
TQ.__spec__.loader.exec_module(TQ)
PER, _arrs, _map = TQ.PER, TQ._arrs, TQ._map


def _bar(b, beat=0):
    """slot index of bar b (1-based), on its beat."""
    return (b - 1) * PER + beat * 4


def test_fires_when_he_rests_and_we_play_through_it():
    a = _arrs(12)
    # he plays bars 1-4 and 9-12; bars 5-8 are his rest
    him = [(_bar(b, k), "L") for b in (1, 2, 3, 4, 9, 10, 11, 12) for k in range(4)]
    ours = him + [(_bar(b, k), "R") for b in (5, 6, 7, 8) for k in range(4)]
    a["human"], a["map"] = _map(len(a["bar"]), him), _map(len(a["bar"]), ours)
    hits = Q.q_breathing(a)
    assert [h[0] for h in hits] == ["BREATHING"]
    assert hits[0][2] == 5, "the address is the first bar of HIS rest"
    assert "rests 4 bar(s)" in hits[0][3] and "16 events" in hits[0][3]


def test_silent_when_we_rest_with_him():
    a = _arrs(12)
    him = [(_bar(b, k), "L") for b in (1, 2, 3, 4, 9, 10, 11, 12) for k in range(4)]
    a["human"] = a["map"] = _map(len(a["bar"]), him)
    assert Q.q_breathing(a) == []


def test_a_couple_of_notes_tailing_into_his_rest_is_a_phrase_end_not_a_defect():
    a = _arrs(12)
    him = [(_bar(b, k), "L") for b in (1, 2, 3, 4, 9, 10, 11, 12) for k in range(4)]
    ours = him + [(_bar(5, 0), "R"), (_bar(5, 1), "R")]      # 2 events in a 4-bar rest
    a["human"], a["map"] = _map(len(a["bar"]), him), _map(len(a["bar"]), ours)
    assert Q.q_breathing(a) == [], "under min_events / per_bar -- must stay silent"


def test_a_one_bar_gap_is_not_a_rest():
    a = _arrs(12)
    him = [(_bar(b, k), "L") for b in (1, 2, 3, 5, 6, 7, 8, 9) for k in range(4)]  # bar 4 only
    ours = him + [(_bar(4, k), "R") for k in range(4)]
    a["human"], a["map"] = _map(len(a["bar"]), him), _map(len(a["bar"]), ours)
    assert Q.q_breathing(a) == [], "one bar is a gap between phrases, not a breath"


def test_reads_only_inside_his_mapped_span():
    """Notes before his first note or after his last are a different amount of song
    (q_events' job), not a failure to breathe."""
    a = _arrs(12)
    him = [(_bar(b, k), "L") for b in (5, 6, 7, 8) for k in range(4)]
    ours = him + [(_bar(b, k), "R") for b in (1, 2, 3, 11, 12) for k in range(4)]
    a["human"], a["map"] = _map(len(a["bar"]), him), _map(len(a["bar"]), ours)
    assert Q.q_breathing(a) == []


def test_silent_without_a_human_map():
    a = _arrs(8)
    a["map"] = _map(len(a["bar"]), [(_bar(b, k), "R") for b in range(1, 9) for k in range(4)])
    a.pop("human")
    assert Q.q_breathing(a) == []


def test_it_is_in_q_all_and_carries_its_code():
    assert Q.q_breathing in Q.QUERIES
    assert "BREATHING" in Q.q_all.codes
