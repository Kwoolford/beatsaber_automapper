"""P3 (2026-09-02): queries over the arrays answer with an ADDRESS, and read against the
same song's human map (or the song) rather than an absolute norm."""
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
bench = _load("bench")

SUB, PER = 4, 16
SONG_NAMES = ["kit_kick", "kit_snare", "kit_hat", "kit_crash", "bass_midi", "lead_midi",
              "vox_midi", "bass_sus", "lead_sus", "vox_sus", "gtr", "pno", "energy", "onset",
              "main"]
MAP_NAMES = ([f"c{i}_{k}" for i in range(12) for k in ("color", "dir")] + ["bombs"]
             + [f"wall_lane{i}" for i in range(4)] + ["arc_L", "arc_R", "chain_L", "chain_R",
                                                      "align_ms", "trv_L", "rot_L", "trv_R", "rot_R"])


def _map(T: int, events) -> np.ndarray:
    """events: [(slot, 'L'|'R'|'D')] -> map array with notes in cell 0 (L) / cell 2 (R)."""
    arr = np.zeros((T, len(MAP_NAMES)))
    for s, h in events:
        if h in "LD":
            arr[s, 0] = 1
        if h in "RD":
            arr[s, 4] = 2      # cell 2 (x=2, y=0)
    return arr


def _arrs(n_bars: int = 16, energy=None) -> dict:
    T = n_bars * PER
    song = np.zeros((T, len(SONG_NAMES)))
    song[:, SONG_NAMES.index("energy")] = 0.5 if energy is None else energy
    song[::2, SONG_NAMES.index("onset")] = 1.0        # onsets on the 8th grid
    return dict(t_sec=np.arange(T) * 0.1, beat=np.arange(T) / SUB,
                bar=(np.arange(T) // PER + 1), section=np.array(["A"] * T),
                lyric=np.array([""] * T, dtype="<U16"), song=song,
                song_names=np.array(SONG_NAMES), map=_map(T, []),
                map_names=np.array(MAP_NAMES), human=_map(T, []), sub=SUB, bpm=120.0, offset=0.0)


def alt8(T, s0=0, s1=None, hand_seq="LR"):
    s1 = T if s1 is None else s1
    return [(s, hand_seq[(s // 2) % 2]) for s in range(s0, s1, 2)]


def test_events_empty_d6_doubles_and_overdense():
    a = _arrs(16)
    T = len(a["bar"])
    a["human"] = _map(T, alt8(T))                                   # 8 events / bar
    # ours: doubles on every beat in bars 1-12 (4 events/bar, 100 % doubles), same as human after
    a["map"] = _map(T, [(s, "D") for s in range(0, 12 * PER, 4)] + alt8(T, 12 * PER))
    hits = Q.q_events(a)
    codes = [h[0] for h in hits]
    assert "EMPTY" in codes and "D6" in codes and "D1" in codes     # median ratio 0.5: very slow
    e = [h for h in hits if h[0] == "EMPTY"]
    assert e[0][2] == 1 and e[0][4] == 12 and "doubles" in e[0][3]
    assert "60%" in [h for h in hits if h[0] == "D6"][0][3]
    # the other direction: 2x the human's events is D6 too
    b = _arrs(8)
    T = len(b["bar"])
    b["human"] = _map(T, [(s, "L") for s in range(0, T, 4)])       # 4/bar
    b["map"] = _map(T, [(s, "LR"[(s // 2) % 2]) for s in range(0, T, 2)])   # 8/bar
    hits = Q.q_events(b)
    assert [h[0] for h in hits] == ["D6"] and "over-dense" in hits[0][3]
    # no human -> silent, by contract
    b.pop("human")
    assert Q.q_events(b) == []


def test_flow_reads_against_the_reference():
    a = _arrs(8)
    T = len(a["bar"])
    # jitter: notes on the "e" from silence, interleaved with on-grid notes
    jitter = [(s, "L") for s in range(0, T, 8)] + [(s + 3, "R") for s in range(0, T, 8)]
    a["map"] = _map(T, jitter)
    a["human"] = _map(T, alt8(T))
    hits = Q.q_flow(a)
    assert hits and hits[0][0] == "FLOW" and hits[0][2] == 1
    # the human jitters the same way (a 195-bpm chart): not FLOW
    a["human"] = _map(T, jitter)
    assert Q.q_flow(a) == []
    # a shifted grid: everything on odd 16ths, human on the 8th grid -> D2, never FLOW
    a["map"] = _map(T, [(s + 1, "LR"[(s // 2) % 2]) for s in range(0, T, 2)])
    a["human"] = _map(T, alt8(T))
    hits = Q.q_flow(a)
    assert {h[0] for h in hits} == {"D2"}


def test_vocals_unanswered_vs_human():
    a = _arrs(8)
    T = len(a["bar"])
    main = a["song"][:, SONG_NAMES.index("main")]
    main[::4] = 1                                    # vox main on every beat
    a["lyric"][::4] = "la"
    a["human"] = _map(T, [(s, "L") for s in range(0, T, 4)])
    a["map"] = _map(T, [(s, "L") for s in range(0, T, 8)])   # answers half
    hits = Q.q_vocals(a)
    assert hits and hits[0][0] == "D4" and "la" in hits[0][3]
    a["map"] = a["human"].copy()
    assert Q.q_vocals(a) == []


def test_drops_late_vs_human():
    n = 12
    E = np.r_[np.full(6 * PER, 0.3), np.full(6 * PER, 0.8)]      # jump at bar 7
    a = _arrs(n, energy=E)
    T = len(a["bar"])
    a["human"] = _map(T, [(s, "L") for s in range(0, 6 * PER, 8)] + alt8(T, 6 * PER))
    # ours: same before, the chorus arrives 2 beats late and half as dense
    a["map"] = _map(T, [(s, "L") for s in range(0, 6 * PER, 8)]
                    + [(s, "R") for s in range(6 * PER + 8, T, 4)])
    hits = Q.q_drops(a)
    assert hits and hits[0][0] == "D3" and hits[0][2] == 7 and "2.00 beats" in hits[0][3]
    a["map"] = a["human"].copy()
    assert Q.q_drops(a) == []


def test_elements_walls_only_when_the_human_has_them():
    a = _arrs(8)
    T = len(a["bar"])
    a["human"] = _map(T, alt8(T))
    a["map"] = _map(T, alt8(T))
    assert Q.q_elements(a) == []
    for k in range(6):                                   # six walls in lane 0
        a["human"][k * 16 + 4: k * 16 + 8, MAP_NAMES.index("wall_lane0")] = 1
    hits = Q.q_elements(a)
    assert len(hits) == 1 and hits[0][0] == "ELEMENTS" and hits[0][2] == 1
    a["map"][4:8, MAP_NAMES.index("wall_lane0")] = 1     # one wall of ours: silent
    assert Q.q_elements(a) == []


def test_bench_tolerance_and_claims():
    row = dict(id="x", label="GOOD", codes=[], must_not_flag=["FLOW"], strength="strong",
               bars=None, tolerance=0.1)
    fires = [("FLOW", 0.0, 10, "why", 11)]              # 2 of 100 bars
    assert bench.score_row(row, fires, None, 100)[0] == "tolerated"
    assert bench.score_row(row, fires + [("FLOW", 0.0, 20, "why", 40)], None, 100)[0] == "VIOLATION"
    assert bench.score_row(dict(row, tolerance=0), fires, None, 100)[0] == "VIOLATION"
    d = dict(id="d", label="DEFECT", codes=["FLOW"], must_not_flag=[], strength="strong", bars="9-12")
    assert bench.score_row(d, [("EMPTY", 0.0, 9, "w", 9)], {"EMPTY"}, 100)[0] == "n/a"
    assert bench.score_row(d, [("EMPTY", 0.0, 9, "w", 9)], {"FLOW", "EMPTY"}, 100)[0] == "MISS"
    assert bench.score_row(d, [("FLOW", 0.0, 10, "w", 11)], {"FLOW"}, 100)[0] == "HIT"
    assert bench.coverage([("FLOW", 0, 3, "w", 5), ("FLOW", 0, 5, "w", 6)], {"FLOW"}, 10) == 0.4
