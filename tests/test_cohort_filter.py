"""Tests for the LayoutPhraseDataset cohort NPS filter (V8-0 orthogonal data fix).

The filter drops whole (song, difficulty) pairs whose overall notes-per-second is
outside ``[min_nps, max_nps]`` so the model is not trained on for-sport ExpertPlus
density (or on near-empty maps).
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from beatsaber_automapper.data.beatmap import ColorNote, DifficultyBeatmap
from beatsaber_automapper.data.layout_dataset import LayoutPhraseDataset
from beatsaber_automapper.data.swing_tokenizer import SwingEventTokenizer

_TOK = SwingEventTokenizer()
# Frames-per-slot bookkeeping does not matter for these tests; we control NPS purely
# through note spacing in beats and the bpm.
BEAT_SUBDIV = 4
PHRASE_SLOTS = 64  # 16 beats * 4


def _swing_tokens_at_nps(nps: float, bpm: float, duration_sec: float) -> list[int]:
    """Build a swing stream with ~nps notes/sec evenly spaced over duration_sec."""
    n_notes = max(2, int(round(nps * duration_sec)))
    beats_per_sec = bpm / 60.0
    total_beats = duration_sec * beats_per_sec
    step = total_beats / n_notes
    notes = [
        ColorNote(beat=i * step, x=i % 4, y=i % 3, color=i % 2, direction=i % 9)
        for i in range(n_notes)
    ]
    bm = DifficultyBeatmap(
        version="3.3.0", color_notes=notes, bomb_notes=[], obstacles=[],
        sliders=[], burst_sliders=[],
    )
    return _TOK.encode_beatmap(bm)


def _write_song(tmp: Path, song_id: str, diff_nps: dict[str, float],
                bpm: float = 120.0, duration_sec: float = 60.0) -> None:
    n_slots = int(duration_sec * (bpm / 60.0) * BEAT_SUBDIV)
    # One phrase boundary list spanning the song in PHRASE_SLOTS windows.
    boundaries = [(s, min(s + PHRASE_SLOTS, n_slots))
                  for s in range(0, n_slots, PHRASE_SLOTS)]
    diffs = {
        name: {"swing_tokens": _swing_tokens_at_nps(nps, bpm, duration_sec)}
        for name, nps in diff_nps.items()
    }
    data = {
        "song_id": song_id,
        "bpm": bpm,
        "mix_beat_features": torch.zeros(n_slots, 768, dtype=torch.float16),
        "drum_beat_features": torch.zeros(n_slots, 768, dtype=torch.float16),
        "phrase_boundaries": boundaries,
        "phrase_fingerprints": torch.zeros(len(boundaries), 768, dtype=torch.float16),
        "difficulties": diffs,
        "mod_requirements": {"category": "vanilla", "genre": "electronic"},
    }
    torch.save(data, tmp / f"{song_id}.pt")


def _make_dataset(tmp: Path, **kw) -> LayoutPhraseDataset:
    (tmp / "splits.json").write_text(json.dumps({"train": ["lo", "mid", "hi"]}))
    return LayoutPhraseDataset(
        data_dir=tmp, split="train", difficulties=["Expert"],
        max_phrase_slots=PHRASE_SLOTS, **kw,
    )


def test_no_band_keeps_all_songs(tmp_path: Path) -> None:
    _write_song(tmp_path, "lo", {"Expert": 2.0})
    _write_song(tmp_path, "mid", {"Expert": 6.0})
    _write_song(tmp_path, "hi", {"Expert": 12.0})
    ds = _make_dataset(tmp_path)
    songs = {p.stem for (p, *_rest) in ds.samples}
    assert songs == {"lo", "mid", "hi"}


def test_max_nps_excludes_for_sport_density(tmp_path: Path) -> None:
    _write_song(tmp_path, "lo", {"Expert": 2.0})
    _write_song(tmp_path, "mid", {"Expert": 6.0})
    _write_song(tmp_path, "hi", {"Expert": 12.0})
    ds = _make_dataset(tmp_path, max_nps=8.0)
    songs = {p.stem for (p, *_rest) in ds.samples}
    assert "hi" not in songs           # 12 NPS dropped
    assert "mid" in songs and "lo" in songs


def test_nps_band_keeps_only_middle(tmp_path: Path) -> None:
    _write_song(tmp_path, "lo", {"Expert": 2.0})
    _write_song(tmp_path, "mid", {"Expert": 6.0})
    _write_song(tmp_path, "hi", {"Expert": 12.0})
    ds = _make_dataset(tmp_path, min_nps=4.0, max_nps=8.0)
    songs = {p.stem for (p, *_rest) in ds.samples}
    assert songs == {"mid"}            # only 6 NPS survives [4,8]
