"""The judge's hard gates (P0, 2026-09-02): parity, the undiluted alignment floor,
and the requested-density gate. These sit BEFORE the pooled conformal p-value, and
each one exists because pooling was measured blind to the defect it catches.
"""
from __future__ import annotations

import math

import pytest

from beatsaber_automapper.evaluation import mapjudge as mj


def _reference(n: int = 200) -> dict:
    """A synthetic reference: every metric uniform on [0, 1], so a value of 0.5 is
    the human median and the pooled gate has nothing to object to."""
    grid = [i / (n - 1) for i in range(n)]
    dists = {name: list(grid) for name, _ax, _t, _n in mj.CANDIDATES}
    calib = {"mean": [0.5 + 0.4 * g for g in grid], "topk": [0.6 + 0.4 * g for g in grid],
             "max": [0.7 + 0.3 * g for g in grid], "pmin": list(grid)}
    return {"distributions": dists, "calib_scores": calib, "calib_scores_audio": calib,
            "align_floor": {"metric": "onset_precision", "min": 0.10, "q": 0.10}}


def _record(**over) -> dict:
    rec = {name: 0.5 for name, _ax, _t, _n in mj.CANDIDATES}
    rec.update({"viol": 0, "n_notes": 500})
    rec.update(over)
    return rec


def test_median_map_passes():
    res = mj.judge(_record(), _reference(), align_floor=True)
    assert res.verdict() == "PASS"
    assert res.why_fail() == []
    assert not math.isnan(res.align_value) and res.align_floor == pytest.approx(0.10)


def test_alignment_floor_fails_an_off_music_map_the_pool_accepts():
    # Every other metric is dead on the human median: the pooled p-value cannot see
    # this map's problem, which is exactly the 65 %-of-offbeat blindness.
    res = mj.judge(_record(onset_precision=0.05), _reference(), align_floor=True)
    assert res.align_fail
    assert res.verdict() == "FAIL"
    assert any("off the music" in w for w in res.why_fail())
    assert "OFF THE MUSIC" in mj.report(res)


def test_alignment_floor_is_reversible():
    rec = _record(onset_precision=0.05)
    assert mj.judge(rec, _reference(), align_floor=False).verdict() == "PASS"
    # The env switch, read when align_floor is None.
    import os
    old = os.environ.get(mj.ALIGN_FLOOR_ENV)
    try:
        os.environ[mj.ALIGN_FLOOR_ENV] = "0"
        assert mj.judge(rec, _reference()).verdict() == "PASS"
        os.environ[mj.ALIGN_FLOOR_ENV] = "1"
        assert mj.judge(rec, _reference()).verdict() == "FAIL"
    finally:
        if old is None:
            os.environ.pop(mj.ALIGN_FLOOR_ENV, None)
        else:
            os.environ[mj.ALIGN_FLOOR_ENV] = old


def test_floor_needs_the_alignment_axis():
    ref = _reference()
    rec = {k: v for k, v in _record().items() if k not in ("onset_precision", "offset_mad_ms")}
    res = mj.judge(rec, ref, align_floor=True)
    assert math.isnan(res.align_value) and not res.align_fail
    assert not res.scored_audio


def test_requested_density_gate():
    ref = _reference()
    # 4.0 nps asked for, 4.4 delivered: inside ±15 %.
    ok = mj.judge(_record(nps=4.4), ref, nps_request=4.0)
    assert ok.verdict() == "PASS" and not ok.nps_fail
    # 3.0 asked for, 4.4 delivered: a miss, and the report says by how much.
    miss = mj.judge(_record(nps=4.4), ref, nps_request=3.0)
    assert miss.nps_fail and miss.verdict() == "FAIL"
    assert any("requested 3.00" in w for w in miss.why_fail())
    # nps/peak_nps leave the pooled score when a request is given.
    assert miss.n_scored == ok.n_scored == len(mj.judge(_record(), ref).metrics) - 2


def test_parity_still_comes_first():
    res = mj.judge(_record(viol=3, onset_precision=0.05), _reference(), align_floor=True)
    assert res.verdict() == "FAIL"
    assert res.why_fail()[0].startswith("3 parity")
