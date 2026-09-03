"""2026-09-02j: `double` used to mirror the note already there, which is a reset ~94 % of the
time (49 doubles drew 46 resets on 1f333) because the two hands are not swinging in step.
The added hand's arrow must come from ITS OWN flow, and fall back to a dot, which cannot re-cock.
"""
from __future__ import annotations

import importlib.util
import json
import pathlib
import zipfile

REPO = pathlib.Path(__file__).resolve().parents[1]


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


M_ = _load("mapedit", REPO / "agent_mapper" / "mapedit.py")

DOWN, UP = 1, 0


def _zip(tmp_path, notes) -> pathlib.Path:
    p = tmp_path / "m.zip"
    with zipfile.ZipFile(p, "w") as zf:
        zf.writestr("Info.dat", json.dumps({"_version": "2.0.0", "_beatsPerMinute": 120}))
        zf.writestr("ExpertStandard.dat", json.dumps({"version": "3.3.0", "colorNotes": notes}))
    return p


def _n(b, c, d, x=1, y=0):
    return {"b": b, "x": x, "y": y, "c": c, "d": d, "a": 0}


def test_double_takes_the_arrow_from_the_added_hands_own_flow(tmp_path):
    # L swings DOWN at 0 and DOWN again at 2. The lone R note at beat 1 is DOWN and
    # H_MIRROR[DOWN] is DOWN, so the old mirror gave L three DOWNs in a row. The opposite
    # arrow alternates instead -- and here it even resolves the reset L already had.
    m = M_.Map(_zip(tmp_path, [_n(0, 0, DOWN), _n(1, 1, DOWN), _n(2, 0, DOWN)]))
    before = sum(1 for _, c, _, _ in M_.reset_swings(m) if c == 0)
    msg = M_.op_double(m, ["1.2.0"], 4)

    added = [n for n in m.notes if n["c"] == 0 and n["b"] == 1]
    assert len(added) == 1, "the other hand was added exactly once"
    assert added[0]["d"] == UP, "alternate with the added hand's own previous arrow"
    assert sum(1 for _, c, _, _ in M_.reset_swings(m) if c == 0) <= before
    assert "added L" in msg


def test_double_falls_back_to_a_dot_when_both_neighbours_box_the_hand_in(tmp_path):
    # DOWN before and UP after: the mirror repeats the DOWN, the opposite repeats the UP,
    # so no arrow is free and the dot is the only note that cannot re-cock.
    m = M_.Map(_zip(tmp_path, [_n(0, 0, DOWN), _n(1, 1, DOWN), _n(2, 0, UP)]))
    M_.op_double(m, ["1.2.0"], 4)
    added = [n for n in m.notes if n["c"] == 0 and n["b"] == 1][0]
    assert added["d"] == 8, "a dot has no direction to re-cock"
    assert not M_.reset_swings(m), "and it leaves the map with no reset at all"


def test_double_still_refuses_a_slot_that_is_not_a_single_note(tmp_path):
    m = M_.Map(_zip(tmp_path, [_n(0, 0, DOWN), _n(0, 1, DOWN)]))
    try:
        M_.op_double(m, ["1.1.0"], 4)
    except M_.EditError as e:
        assert "exactly one note" in str(e)
    else:
        raise AssertionError("doubling an existing double must refuse")
