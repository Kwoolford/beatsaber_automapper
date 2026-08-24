"""`--snap-onsets` on a song that is not in the corpus.

The snap was songset-only for one reason: `outputs/onset_cache` is keyed by corpus
song id. The detector itself takes a plain audio path, so the generalisation is a
cache-key change -- but the guards around that key are load-bearing:

  * a non-corpus song must NEVER write `<song_id>.npz`. That file is the fixed point
    every alignment number in TODO.md is measured against; a collision would move the
    human baseline silently.
  * a corpus song must keep using its CACHED entry rather than recomputing. Measured
    2026-08-24: a fresh run differs from the stored cache by up to one librosa
    analysis frame (p75 11.6 ms = 512/44100), so recomputing would quietly shift the
    reference.
  * `compute=False` must not launch Demucs. A surprise 60-second GPU pass inside what
    looks like a cache lookup is the kind of cost that hides for months.
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np
import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from agent_mapper import refonsets as RO  # noqa: E402


@pytest.fixture()
def fake_cache(tmp_path, monkeypatch):
    """Redirect the onset cache and clear the per-process memos."""
    monkeypatch.setattr(RO, "CACHE", tmp_path)
    RO._ONSET_MEMO.clear()
    RO._KEY_MEMO.clear()
    yield tmp_path
    RO._ONSET_MEMO.clear()
    RO._KEY_MEMO.clear()


def _write(cache: pathlib.Path, key: str, onsets) -> None:
    np.savez(cache / f"{key}.npz", onsets=np.asarray(onsets, dtype=float))


def test_audio_key_is_prefixed_so_it_cannot_collide_with_a_song_id(tmp_path):
    aud = tmp_path / "whatever.ogg"
    aud.write_bytes(b"not really audio, but hashable")
    key = RO.audio_key(aud)
    assert key.startswith("audio_"), key
    # A corpus id is a bare hex stem like `1f767`; the prefix guarantees no overlap.
    assert not key[len("audio_"):].startswith("audio_")


def test_audio_key_follows_content_not_filename(tmp_path):
    a, b = tmp_path / "one.ogg", tmp_path / "two.ogg"
    a.write_bytes(b"same bytes")
    b.write_bytes(b"same bytes")
    c = tmp_path / "three.ogg"
    c.write_bytes(b"different bytes")
    assert RO.audio_key(a) == RO.audio_key(b)
    assert RO.audio_key(a) != RO.audio_key(c)


def test_corpus_entry_wins_over_the_computed_one(fake_cache, tmp_path):
    """A songset song keeps its cached reference even when audio is available."""
    aud = tmp_path / "song.ogg"
    aud.write_bytes(b"audio bytes")
    _write(fake_cache, "1f767", [1.0, 2.0, 3.0])
    _write(fake_cache, RO.audio_key(aud), [9.0, 9.5])

    got = RO.reference_onsets("1f767", audio=aud, compute=True)
    assert list(got) == [1.0, 2.0, 3.0]


def test_non_corpus_song_uses_the_content_keyed_entry(fake_cache, tmp_path):
    aud = tmp_path / "song.ogg"
    aud.write_bytes(b"audio bytes")
    _write(fake_cache, RO.audio_key(aud), [4.0, 5.0])

    assert list(RO.reference_onsets("unknown_song", audio=aud, compute=True)) == [4.0, 5.0]


def test_compute_false_never_runs_the_detector(fake_cache, tmp_path, monkeypatch):
    aud = tmp_path / "song.ogg"
    aud.write_bytes(b"audio bytes")

    def boom(*_a, **_k):  # pragma: no cover - must not be reached
        raise AssertionError("compute=False must not run Demucs")

    monkeypatch.setattr(RO, "compute_reference_onsets", boom)
    assert RO.reference_onsets("unknown_song", audio=aud, compute=False) is None


def test_missing_everything_leaves_times_unchanged(fake_cache, tmp_path):
    """A snap with no reference is a no-op, not a crash and not a filter."""
    times = [1.0, 2.0, 3.5]
    out, moved, n_in = RO.snap(times, "unknown_song")
    assert out == times and moved == 0 and n_in == 3


def test_snap_moves_events_onto_the_reference(fake_cache, tmp_path):
    aud = tmp_path / "song.ogg"
    aud.write_bytes(b"audio bytes")
    _write(fake_cache, RO.audio_key(aud), [1.000, 2.000])

    # 1.03 is inside the 60 ms window; 2.50 is far from anything and must survive.
    out, moved, n_in = RO.snap([1.03, 2.50], "unknown_song", audio=aud, compute=True)
    assert moved == 1 and n_in == 2
    assert out == [1.0, 2.5]


def test_snap_dedups_when_two_events_land_on_one_onset(fake_cache, tmp_path):
    aud = tmp_path / "song.ogg"
    aud.write_bytes(b"audio bytes")
    _write(fake_cache, RO.audio_key(aud), [1.000])

    out, moved, n_in = RO.snap([0.98, 1.02], "unknown_song", audio=aud, compute=True)
    # Both moved, and they collapse -- the landmine that made "moved 818/745" print.
    assert moved == 2 and n_in == 2 and out == [1.0]
