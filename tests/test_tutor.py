"""Study mode (P2b, 2026-09-02): situations come from the SONG columns only, patterns
name what a map does there, and `same_way` is the DoD's yardstick."""
from __future__ import annotations

import importlib.util
import pathlib

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("tutor", REPO / "scripts" / "tutor.py")
tutor = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tutor)

SUB, PER = 4, 16
SONG_NAMES = ["kit_kick", "kit_snare", "kit_hat", "kit_crash", "bass_midi", "lead_midi",
              "vox_midi", "bass_sus", "lead_sus", "vox_sus", "gtr", "pno", "energy", "onset",
              "main"]


def _arrs(n_bars: int = 12) -> dict:
    T = n_bars * PER
    song = np.zeros((T, len(SONG_NAMES)))
    section = np.array(["A"] * (6 * PER) + ["B"] * (6 * PER))
    # energy: quiet for 6 bars, loud after
    song[:, SONG_NAMES.index("energy")] = np.r_[np.full(6 * PER, 0.2), np.full(6 * PER, 0.8)]
    # vocals enter at bar 7 beat 1, with a lyric
    lyric = np.array([""] * T, dtype="<U16")
    song[6 * PER:, SONG_NAMES.index("vox_midi")] = 60
    lyric[6 * PER] = "hello"
    # drums present from bar 4 (kick every beat)
    song[3 * PER::SUB, SONG_NAMES.index("kit_kick")] = 1
    return dict(t_sec=np.arange(T) * 0.1, beat=np.arange(T) / SUB,
                bar=(np.arange(T) // PER + 1), section=section, lyric=lyric,
                song=song, song_names=np.array(SONG_NAMES),
                map=np.zeros((T, 38)), sub=SUB)


def _map(T: int, events: list[tuple[int, str]]) -> np.ndarray:
    arr = np.zeros((T, 38))
    for s, h in events:
        if h in "LD":
            arr[s, 0] = 1
        if h in "RD":
            arr[s, 2] = 2
    return arr


def test_situations_come_from_the_song():
    arrs = _arrs()
    sits = tutor.find_situations(arrs)
    by_bar = {s["bar"]: s for s in sits}
    assert 7 in by_bar and by_bar[7]["kind"] == "section"
    kinds = " ".join(by_bar[7]["kinds"])
    assert "section A→B" in kinds and "E jump" in kinds and "vox enters 'hello'" in kinds
    assert 4 in by_bar and by_bar[4]["kind"] == "drums-in"
    # an empty map changes nothing: situations ignore the map columns
    arrs["map"] = _map(len(arrs["bar"]), [(0, "D")])
    assert [s["bar"] for s in tutor.find_situations(arrs)] == [s["bar"] for s in sits]


def test_pattern_words():
    T = 12 * PER
    bar = np.arange(T) // PER + 1
    stream = _map(T, [(6 * PER + i, "LR"[i % 2]) for i in range(0, 32)])
    assert tutor.pattern(stream, bar, 7, SUB)["word"] == "stream"
    alt8 = _map(T, [(6 * PER + i, "LR"[(i // 2) % 2]) for i in range(0, 32, 2)])
    p = tutor.pattern(alt8, bar, 7, SUB)
    assert p["word"] == "alt-8ths" and p["ev_bar"] == 8.0 and p["first"] == 0.0
    dbl = _map(T, [(6 * PER + i, "D") for i in range(0, 32, 4)])
    p = tutor.pattern(dbl, bar, 7, SUB)
    assert p["word"] == "doubles" and p["dbl_pct"] == 100.0
    assert tutor.pattern(_map(T, []), bar, 7, SUB)["word"] == "rest"
    late = _map(T, [(6 * PER + 8, "L"), (6 * PER + 12, "R"), (6 * PER + 16, "L")])
    assert tutor.pattern(late, bar, 7, SUB)["first"] == 2.0   # beats after the bar line
    assert tutor.pattern(late, bar, 7, SUB)["pre"] == "rest"


def test_same_way():
    t = dict(word="4ths", ev_bar=4.5, first=0.0)
    assert tutor.same_way(t, dict(word="4ths", ev_bar=5.5, first=0.5))
    assert not tutor.same_way(t, dict(word="alt-4ths", ev_bar=4.5, first=0.0))
    assert not tutor.same_way(t, dict(word="4ths", ev_bar=7.0, first=0.0))
    assert not tutor.same_way(t, dict(word="4ths", ev_bar=4.5, first=1.5))
    assert not tutor.same_way(t, dict(word="4ths", ev_bar=4.5, first=None))
