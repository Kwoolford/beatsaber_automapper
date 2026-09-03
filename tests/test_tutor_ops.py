"""P4 (2026-09-02): the two section ops the 1f767 loop scripted by hand, now in tutor.py —
`--copy a-b` (his cells replace ours) and `--thin a-b` (ours survive only beside his; odd-16th
survivors move onto his slot; a second claim on one slot+hand is a delete)."""
from __future__ import annotations

import importlib.util
import pathlib

REPO = pathlib.Path(__file__).resolve().parents[1]


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


T_ = _load("tutor", REPO / "scripts" / "tutor.py")
TQ = _load("test_queries", pathlib.Path(__file__).with_name("test_queries.py"))
PER, _arrs, _map, alt8 = TQ.PER, TQ._arrs, TQ._map, TQ.alt8


def test_copy_deletes_ours_and_places_his():
    a = _arrs(4)
    T = len(a["bar"])
    a["human"] = _map(T, [(0, "D"), (8, "L")])            # 1.1.0 double, 1.3.0 L
    a["map"] = _map(T, [(2, "R"), (8, "R")])              # 1.1.2 R, 1.3.0 R
    ops = T_.emit_copy(a, 1, 1)
    assert ops == ["place 1.1.0 L 0,0 •", "place 1.1.0 R 2,0 •",
                   "delete 1.1.2", "delete 1.3.0", "place 1.3.0 L 0,0 •"]


def test_thin_keeps_beside_his_moves_odd_16ths_and_deletes_the_rest():
    a = _arrs(4)
    T = len(a["bar"])
    a["human"] = _map(T, [(0, "L"), (8, "L"), (12, "D")])     # 1.1.0 · 1.3.0 · 1.4.0
    a["map"] = _map(T, [(0, "L"), (4, "R"), (7, "R"), (9, "R"), (11, "L"), (13, "L")])
    ops = T_.emit_thin(a, 1, 1)
    assert "delete 1.2.0" in ops                          # nothing of his within a slot
    assert "move 1.2.3 R 1.3.0 2,0" in ops                 # odd 16th beside his 1.3.0 -> his slot, free cell
    assert "delete 1.3.1 R" in ops                         # second R claim on 1.3.0 -> delete, not collide
    assert "move 1.3.3 L 1.4.0 0,0" in ops                 # lands in HIS L cell at 1.4.0
    assert "delete 1.4.1 L" in ops                         # our L already moved onto 1.4.0
    assert not any(o.startswith("delete 1.1.0") for o in ops)   # on his slot: kept


def test_fill_places_his_slots_we_skipped_with_our_parity():
    a = _arrs(4)
    T = len(a["bar"])
    # his: 1.1.0 L · 1.2.0 R · 1.3.0 L · 1.4.0 R ; ours: 1.1.0 L↓ only (see _map's arrow)
    a["human"] = _map(T, [(0, "L"), (4, "R"), (8, "L"), (12, "R")])
    a["map"] = _map(T, [(0, "L")])
    ops = T_.emit_fill(a, 1, 1)
    assert not any(o.startswith("place 1.1.0") for o in ops)          # answered already
    assert any(o.startswith("place 1.2.0 R ") for o in ops)
    assert any(o.startswith("place 1.3.0 L ") for o in ops)
    assert any(o.startswith("place 1.4.0 R ") for o in ops)
    # our L at 1.1.0 is directional -> the fill at 1.3.0 L is its vertical opposite
    ours = [c for c in T_._cells(a["map"][0]) if c[0] == "L"][0][3]
    filled = next(o for o in ops if o.startswith("place 1.3.0 L ")).split()[-1]
    if ours in T_._DOWN:
        assert filled == "↑"
    elif ours in T_._UP:
        assert filled == "↓"
    else:
        assert filled == "•"
