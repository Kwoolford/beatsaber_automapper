"""Pins the `bass` transcription added to `agent_mapper/melody.py` on 2026-08-17.

Two things are worth a test here rather than a comment:

1. **`bass` is in `MELODIC` and routes to pYIN, not to the salience tracker.** The
   salience path exists for polyphonic `other` and returns a plausible-looking wrong
   answer on a monophonic stem, so a silent mis-route would not crash — it would just
   make the bass line quietly worse. Asserted on the source of truth (`_PYIN_RANGE`).

2. **`subharmonic_share` actually fires.** It returns 0.000 on every real song measured
   so far, and a guard that has only ever returned zero is indistinguishable from one
   that cannot fire (this repo has shipped exactly that mistake before — an
   "implausible phase" bound that `wrap_to_slot` made unreachable). The synthetic case
   below forces it to fire, so the zeros on real songs are a *measurement*.
"""

import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "agent_mapper"))

import melody as M  # noqa: E402

SR = 22050


def _tone(midi: float, dur: float = 1.0, sr: int = SR) -> np.ndarray:
    """A tone at `midi` with a couple of harmonics, so it looks like an instrument."""
    import librosa

    f = float(librosa.midi_to_hz(midi))
    t = np.arange(int(dur * sr)) / sr
    y = np.sin(2 * np.pi * f * t)
    y += 0.4 * np.sin(2 * np.pi * 2 * f * t)
    y += 0.2 * np.sin(2 * np.pi * 3 * f * t)
    return (y / np.abs(y).max()).astype("float32")


def test_bass_is_melodic_and_tracked_monophonically():
    assert "bass" in M.MELODIC
    assert "bass" in M._PYIN_RANGE, "bass must route to pYIN, not the salience tracker"
    lo, hi = M._PYIN_RANGE["bass"]
    assert (lo, hi) == ("C1", "C4")


def test_subharmonic_share_is_zero_when_the_pitch_is_the_fundamental():
    y = _tone(36)                                   # C2
    ev = [{"t": 0.1, "midi": 36}, {"t": 0.5, "midi": 36}]
    assert M.subharmonic_share(y, SR, ev) == 0.0


def test_subharmonic_share_fires_when_the_pitch_is_an_octave_too_high():
    """The failure mode it exists to catch: energy sits an octave below the label."""
    y = _tone(36)                                   # the tone really is C2 ...
    ev = [{"t": 0.1, "midi": 48}, {"t": 0.5, "midi": 48}]   # ... but we labelled it C3
    assert M.subharmonic_share(y, SR, ev) == 1.0


def test_subharmonic_share_is_empty_safe():
    assert M.subharmonic_share(np.zeros(SR, dtype="float32"), SR, []) == 0.0


def test_stale_cache_without_bass_is_rejected(tmp_path, monkeypatch):
    """A melody cache written before bass existed must not be returned as-is."""
    import json

    monkeypatch.setattr(M, "CACHE", tmp_path)
    audio = tmp_path / "song.ogg"
    stale = {"stems": {"vocals": [], "other": []}, "meta": {}}
    (tmp_path / "song.json").write_text(json.dumps(stale))

    # It must not short-circuit; with no real audio behind it, it has to try to
    # recompute and fail on the missing file rather than hand back the stale dict.
    with pytest.raises(Exception):
        M.analyse(audio)

    fresh = {"stems": {s: [] for s in M.MELODIC}, "meta": {}}
    (tmp_path / "song.json").write_text(json.dumps(fresh))
    assert M.analyse(audio) == fresh
