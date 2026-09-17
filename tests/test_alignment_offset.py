"""The judge scores notes on the GAME's clock: songTimeOffset + beat * spb (2026-09-16j)."""
import json
import zipfile

from beatsaber_automapper.evaluation import alignment


class _N:
    def __init__(self, beat):
        self.beat = beat


class _BM:
    def __init__(self, beats, off=None):
        self.color_notes = [_N(b) for b in beats]
        if off is not None:
            self.song_time_offset = off


def test_note_times_apply_song_time_offset():
    assert alignment.note_times(_BM([0.0, 1.0]), bpm=60.0) == [0.0, 1.0]
    assert alignment.note_times(_BM([0.0, 1.0], off=-0.05), bpm=60.0) == [-0.05, 0.95]


def test_zip_song_time_offset(tmp_path):
    z = tmp_path / "m.zip"
    with zipfile.ZipFile(z, "w") as zf:
        zf.writestr("Info.dat", json.dumps({"_beatsPerMinute": 120, "_songTimeOffset": -0.034}))
    assert alignment.zip_song_time_offset(z) == -0.034
    with zipfile.ZipFile(tmp_path / "n.zip", "w") as zf:
        zf.writestr("Info.dat", json.dumps({"_beatsPerMinute": 120}))
    assert alignment.zip_song_time_offset(tmp_path / "n.zip") == 0.0
