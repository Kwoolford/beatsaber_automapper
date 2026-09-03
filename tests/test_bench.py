"""The bench (P2, 2026-09-02): the label file is well-formed, every path it names
exists, and the scoring rules do what the README says."""
from __future__ import annotations

import importlib.util
import json
import pathlib

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
LABELS = REPO / "docs" / "eval_references" / "labelled_maps.json"

_spec = importlib.util.spec_from_file_location("bench", REPO / "scripts" / "bench.py")
bench = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bench)

CODES = {"D1", "D2", "D3", "D4", "D5", "D6", "EMPTY", "FLOW", "ELEMENTS",
         "HANDROLE", "BREATHING", "ONBEAT_MAIN"}


def test_labels_well_formed():
    rows = json.loads(LABELS.read_text())["rows"]
    assert len(rows) >= 8
    ids = [r["id"] for r in rows]
    assert len(ids) == len(set(ids))
    for r in rows:
        assert r["label"] in ("GOOD", "PREFERRED", "DEFECT", "CLEAN", "UNLABELLED")
        assert r["strength"] in ("strong", "weak")
        assert set(r["codes"]) <= CODES and set(r["must_not_flag"]) <= CODES
        assert not (set(r["codes"]) & set(r["must_not_flag"]))
        if r["bars"]:
            assert r["bars_from"] in ("kyle", "agent-read")


@pytest.mark.skipif(not (REPO / "data" / "raw").exists(), reason="no corpus checkout")
def test_every_labelled_path_exists():
    for r in json.loads(LABELS.read_text())["rows"]:
        assert (REPO / r["map"]).exists(), r["id"]
        for p in r.get("also", []):
            assert (REPO / p).exists(), f"{r['id']} also {p}"


def test_same_notes_pair_present():
    rows = {r["id"]: r for r in json.loads(LABELS.read_text())["rows"]}
    assert rows["1f333-aplus"]["song"] == rows["1f333-before"]["song"] == "1f333"
    assert rows["1f333-aplus"]["must_not_flag"] == rows["1f333-before"]["must_not_flag"]


def _row(**kw):
    base = {"id": "x", "label": "DEFECT", "codes": ["FLOW"], "must_not_flag": [],
            "strength": "strong", "bars": None}
    base.update(kw)
    return base


def test_score_row_rules():
    hit = ("FLOW", 43.0, 34, "why")
    assert bench.score_row(_row(), [hit])[0] == "HIT"
    assert bench.score_row(_row(), [])[0] == "MISS"
    assert bench.score_row(_row(), [("EMPTY", 1.0, 2, "")])[0] == "MISS"
    # Labelled bars: a fire outside them is a hit elsewhere, not a hit.
    assert bench.score_row(_row(bars="33-36"), [hit])[0] == "HIT"
    assert bench.score_row(_row(bars="50-52"), [hit])[0] == "HIT-elsewhere"
    # Negatives.
    clean = _row(label="CLEAN", codes=[], must_not_flag=["HANDROLE"])
    assert bench.score_row(clean, [])[0] == "CLEAN"
    assert bench.score_row(clean, [("EMPTY", 1.0, 2, "")])[0] == "FALSE"
    assert bench.score_row(clean, [("HANDROLE", 1.0, 2, "")])[0] == "VIOLATION"
    good = _row(label="GOOD", codes=[], must_not_flag=["FLOW"])
    assert bench.score_row(good, [("EMPTY", 1.0, 2, "")])[0] == "fires"
    assert bench.score_row(good, [hit])[0] == "VIOLATION"
    assert bench.score_row(_row(label="UNLABELLED", codes=[]), [hit])[0] == "n/a"
