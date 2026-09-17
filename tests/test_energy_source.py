"""A song's energy must come from its AUDIO, wherever the song lives (2026-09-17l).

`score.py` used to find audio only in `data/eval_songset` / `data/test_songs` and silently fell
back to event loudness; `answer.py` read only the energy cache that `verdict.py` writes after it.
"""
import json
import pathlib
import sys
import zipfile

import numpy as np

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "agent_mapper"))
sys.path.insert(0, str(REPO))

import answer as AN  # noqa: E402
import score as S  # noqa: E402


def _tone_ogg(path: pathlib.Path, sr=22050, secs=16.0, bpm=120.0):
    """Quiet for 8 s, loud for 8 s: one energy jump at bar 5 (120 bpm)."""
    import soundfile as sf
    t = np.arange(int(sr * secs)) / sr
    y = 0.02 * np.sin(2 * np.pi * 220 * t)
    y[t >= 8.0] *= 30
    sf.write(path, y.astype(np.float32), sr, format="OGG", subtype="VORBIS")


def _map_zip(path: pathlib.Path, audio: pathlib.Path, notes):
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("Info.dat", json.dumps({"_version": "2.1.0", "_beatsPerMinute": 120.0,
                                            "_songTimeOffset": 0.0, "_songFilename": "song.ogg"}))
        zf.writestr("ExpertStandard.dat", json.dumps({"version": "3.3.0", "colorNotes": notes}))
        zf.write(audio, "song.ogg")


def test_resolve_song_takes_audio_from_the_map_zip(tmp_path, monkeypatch):
    monkeypatch.setattr(S, "OUT", tmp_path / "outputs")
    a = tmp_path / "a.ogg"
    _tone_ogg(a)
    z = tmp_path / "X__zz999.zip"
    _map_zip(z, a, [])
    sid, audio, how = S.resolve_song("zz999", z)
    assert sid == "zz999" and audio is not None and audio.exists(), how
    assert "map zip" in how


def test_answer_computes_energy_without_a_cache(tmp_path, monkeypatch):
    from agent_mapper import score as S2          # the module object answer.py imports
    monkeypatch.setattr(S2, "SCORE_CACHE", tmp_path / "score_cache")
    monkeypatch.setattr(S, "SCORE_CACHE", tmp_path / "score_cache")
    a = tmp_path / "a.ogg"
    _tone_ogg(a)
    # bar 5 starts at beat 16 (8 s); our first note there is 2 beats late
    notes = [{"b": float(b), "x": 1, "y": 0, "c": 0, "d": 1} for b in (0, 4, 8, 12)]
    notes.append({"b": 18.0, "x": 1, "y": 0, "c": 0, "d": 0})
    em = AN._energy_per_bar("zz998", 120.0, 5, audio=a)
    assert em is not None and em[4] - em[3] >= AN.JUMP
    out, moved = AN.answer(notes, 120.0, "zz998", audio=a)
    assert moved == 1 and out[-1]["b"] < 18.0
