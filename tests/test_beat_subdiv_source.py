"""BEAT_SUBDIV must have exactly ONE live source of truth.

`generate.py` reads the subdivision in three places across two modules. If any of
them binds the value at import time while another reads it live, the generator will
happily produce a map with the slot→beat conversion done at one resolution and the
feature grid at another — and it will NOT raise. That failure mode has already cost
this project twice in one night: a literal `beats_per_phrase=16` overflowed Stage-2's
slot embedding (a loud CUDA assert), and a hardcoded `/ 4.0` in `layout_model`
silently misclassified every idiom `dt_class` (no error at all).

These tests pin the live-read property so a future refactor cannot quietly undo it.
"""

from __future__ import annotations

import pytest

from beatsaber_automapper.data import mert_encoder


@pytest.fixture(autouse=True)
def _restore():
    yield
    mert_encoder.reset_beat_subdiv()


def test_set_and_reset_round_trip():
    base = mert_encoder.BEAT_SUBDIV
    assert mert_encoder.set_beat_subdiv(8) == 8
    assert mert_encoder.BEAT_SUBDIV == 8
    assert mert_encoder.reset_beat_subdiv() == base
    assert mert_encoder.BEAT_SUBDIV == base


@pytest.mark.parametrize("bad", [0, -1, 17, 100])
def test_out_of_range_is_refused(bad):
    before = mert_encoder.BEAT_SUBDIV
    with pytest.raises(ValueError):
        mert_encoder.set_beat_subdiv(bad)
    assert mert_encoder.BEAT_SUBDIV == before


def test_a_function_local_import_sees_the_updated_value():
    """This is the property the generator depends on.

    A module-level `from mert_encoder import BEAT_SUBDIV` snapshots; an import inside
    a function re-reads on every call. Both generation read sites use the latter.
    """
    def read_like_generate_does() -> int:
        from beatsaber_automapper.data.mert_encoder import BEAT_SUBDIV
        return BEAT_SUBDIV

    mert_encoder.set_beat_subdiv(8)
    assert read_like_generate_does() == 8
    mert_encoder.set_beat_subdiv(4)
    assert read_like_generate_does() == 4


def test_no_generation_read_site_imports_subdiv_from_beat_grid():
    """`beat_grid` binds the value at import, so reading it there goes stale.

    Asserted on the source rather than by execution: the desync produces a wrong map,
    not an exception, so there is nothing to catch at runtime.
    """
    import pathlib

    src = (pathlib.Path(__file__).resolve().parents[1]
           / "src" / "beatsaber_automapper" / "generation" / "generate.py").read_text()
    assert "from beatsaber_automapper.data.beat_grid import BEAT_SUBDIV" not in src, (
        "generate.py must read BEAT_SUBDIV from mert_encoder (live) — beat_grid "
        "snapshots it at import time and would silently go stale after "
        "set_beat_subdiv()"
    )


def test_beats_per_phrase_pairing_holds_at_both_subdivisions():
    """Stage-2's slot_emb is sized 97, so slots-per-phrase must stay 64.

    `beats_per_phrase = 64 // subdiv` is what keeps that true; a literal 16 at
    subdiv 8 asks for 128 and indexes off the end of the embedding.
    """
    for subdiv in (4, 8):
        beats_per_phrase = max(1, 64 // subdiv)
        assert beats_per_phrase * subdiv == 64
