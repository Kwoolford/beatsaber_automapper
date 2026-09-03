"""P4 (2026-09-02): the verdict page — reds are addresses with a tool, SHIP? follows the reds,
and the colours follow the 10 % tolerance Kyle's "vast majority is A+" set."""
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


V = _load("verdict")
_tq = importlib.util.spec_from_file_location("test_queries", pathlib.Path(__file__).with_name("test_queries.py"))
TQ = importlib.util.module_from_spec(_tq)
_tq.loader.exec_module(TQ)
PER, _arrs, _map, alt8 = TQ.PER, TQ._arrs, TQ._map, TQ.alt8


def _npz(tmp_path, arrs) -> pathlib.Path:
    f = tmp_path / "x_1f8d6.npz"
    np.savez(f, **arrs)
    return f


def test_verdict_red_names_bars_and_tool(tmp_path):
    a = _arrs(16)
    T = len(a["bar"])
    a["human"] = _map(T, alt8(T))                                   # 8 events / bar
    a["map"] = _map(T, [(s, "D") for s in range(0, 12 * PER, 4)] + alt8(T, 12 * PER))
    v = V.verdict(_npz(tmp_path, a), song="1f8d6", with_bench=False)
    by = {ln["code"]: ln for ln in v["lines"]}
    assert by["EMPTY"]["state"] == "🔴" and by["EMPTY"]["spans"].startswith("1-12")
    assert by["D1"]["state"] == "🔴"                                # any hit of D1 is red
    assert by["D3"]["state"] == "✅" and by["FLOW"]["state"] == "✅"
    assert v["ship"] == "NO" and v["reds"] >= 2
    page = V.render(v)
    assert "SHIP? NO" in page and "--bars 1-12" in page and "fix:" in page
    assert "EMPTY (bars 1-12" in page.splitlines()[-1]


def test_verdict_yellow_ships(tmp_path):
    a = _arrs(100)
    T = len(a["bar"])
    a["human"] = _map(T, alt8(T))
    ours = alt8(T)
    # jitter on 2 bars of 100: under the 10 % tolerance -> yellow, ships
    ours = [(s, h) for s, h in ours if not (10 * PER <= s < 12 * PER)]
    ours += [(s, "L") for s in range(10 * PER, 12 * PER, 8)] + \
            [(s + 3, "R") for s in range(10 * PER, 12 * PER, 8)]
    a["map"] = _map(T, ours)
    v = V.verdict(_npz(tmp_path, a), song="1f8d6", with_bench=False)
    by = {ln["code"]: ln for ln in v["lines"]}
    assert by["FLOW"]["state"] == "🟡" and by["FLOW"]["share"] < 0.10
    assert v["ship"] == "YES" and "SHIP? YES" in V.render(v)


def test_verdict_without_human_is_grey(tmp_path):
    a = _arrs(8)
    T = len(a["bar"])
    a["map"] = _map(T, alt8(T))
    a.pop("human")
    v = V.verdict(_npz(tmp_path, a), song=None, with_bench=False)
    by = {ln["code"]: ln for ln in v["lines"]}
    assert by["EMPTY"]["state"] == "⚪" and by["D4"]["state"] == "⚪"
    assert by["FLOW"]["state"] == "✅"                              # onsets are the reference
    assert "could not be asked" in V.render(v)
