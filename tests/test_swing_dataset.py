"""Tests for SwingSequenceDataset (V6 Stage 2 dataset)."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest
import torch

from beatsaber_automapper.data.beatmap import (
    BombNote,
    BurstSlider,
    ColorNote,
    DifficultyBeatmap,
    Slider,
)
from beatsaber_automapper.data.dataset import (
    SwingSequenceDataset,
    swing_collate_fn,
)
from beatsaber_automapper.data.swing_tokenizer import (
    BOS,
    EOS,
    PAD,
    VOCAB_SIZE,
    SwingEventTokenizer,
)


# ---------------------------------------------------------------------------
# Helpers: build synthetic V6 .pt fixtures
# ---------------------------------------------------------------------------

_TOK = SwingEventTokenizer()


def _make_swing_tokens(n_notes: int = 40, start_beat: float = 0.0) -> list[int]:
    """Synthetic swing stream: alternating L/R notes over n_notes beats."""
    notes = [
        ColorNote(
            beat=start_beat + i * 0.5,
            x=i % 4,
            y=i % 3,
            color=i % 2,
            direction=i % 9,
        )
        for i in range(n_notes)
    ]
    bm = DifficultyBeatmap(
        version="3.3.0",
        color_notes=notes,
        bomb_notes=[],
        obstacles=[],
        sliders=[],
        burst_sliders=[],
    )
    return _TOK.encode_beatmap(bm)


def _make_v6_pt(
    tmp_dir: Path,
    song_id: str = "test001",
    n_notes: int = 40,
    bpm: float = 120.0,
    difficulties: list[str] | None = None,
) -> Path:
    """Write a minimal V6-format .pt file to tmp_dir."""
    if difficulties is None:
        difficulties = ["Expert"]

    n_frames = 4096
    n_mels = 80

    diffs = {}
    for diff in difficulties:
        swing_toks = _make_swing_tokens(n_notes)
        diffs[diff] = {
            "onset_frames": torch.arange(n_notes, dtype=torch.long) * 16,
            "onset_labels": torch.zeros(n_frames),
            "token_sequences": [[1]] * n_notes,  # dummy V5 tokens
            "swing_tokens": swing_toks,
        }

    data = {
        "song_id": song_id,
        "bpm": bpm,
        "mel_spectrogram": torch.randn(n_mels, n_frames),
        "structure_features": torch.randn(8, n_frames),
        "difficulties": diffs,
        "mod_requirements": {"category": "vanilla", "genre": "electronic"},
    }
    pt_path = tmp_dir / f"{song_id}.pt"
    torch.save(data, pt_path)
    return pt_path


def _write_splits(tmp_dir: Path, song_ids: list[str]) -> None:
    splits = {"train": song_ids, "val": [], "test": []}
    (tmp_dir / "splits.json").write_text(json.dumps(splits))


# ---------------------------------------------------------------------------
# Dataset initialisation
# ---------------------------------------------------------------------------


def test_dataset_empty_dir() -> None:
    with tempfile.TemporaryDirectory() as d:
        ds = SwingSequenceDataset(Path(d), window_events=8, window_hop=4)
        assert len(ds) == 0


def test_dataset_skips_v5_only_file() -> None:
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        # Write a .pt file WITHOUT swing_tokens
        data = {
            "song_id": "v5only",
            "bpm": 120.0,
            "mel_spectrogram": torch.zeros(80, 512),
            "difficulties": {"Expert": {"onset_frames": torch.tensor([0]), "swing_tokens": []}},
            "mod_requirements": {"category": "vanilla", "genre": "electronic"},
        }
        torch.save(data, tmp / "v5only.pt")
        ds = SwingSequenceDataset(tmp, window_events=8, window_hop=4)
        assert len(ds) == 0


def test_dataset_indexes_v6_file() -> None:
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        _make_v6_pt(tmp, n_notes=40)
        ds = SwingSequenceDataset(tmp, window_events=8, window_hop=8)
        # 40 notes ≈ 40 events; with window=8 hop=8, should get ~5 windows
        assert len(ds) > 0


def test_dataset_respects_split_filter() -> None:
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        _make_v6_pt(tmp, song_id="song_a", n_notes=40)
        _make_v6_pt(tmp, song_id="song_b", n_notes=40)
        _write_splits(tmp, ["song_a"])  # only song_a in train
        ds = SwingSequenceDataset(tmp, split="train", window_events=8, window_hop=8)
        # Only song_a should be indexed
        paths = {s[0].stem for s in ds.samples}
        assert paths == {"song_a"}


def test_dataset_respects_difficulty_filter() -> None:
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        _make_v6_pt(tmp, n_notes=40, difficulties=["Expert", "ExpertPlus"])
        ds_expert = SwingSequenceDataset(
            tmp, window_events=8, window_hop=8, difficulties=["Expert"]
        )
        ds_all = SwingSequenceDataset(tmp, window_events=8, window_hop=8)
        # Expert-only should have fewer samples than all-difficulties
        assert len(ds_expert) < len(ds_all)


def test_dataset_excludes_modded_category() -> None:
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        swing_toks = _make_swing_tokens(40)
        data = {
            "song_id": "noodle",
            "bpm": 120.0,
            "mel_spectrogram": torch.zeros(80, 512),
            "difficulties": {"Expert": {"swing_tokens": swing_toks}},
            "mod_requirements": {"category": "noodle", "genre": "electronic"},
        }
        torch.save(data, tmp / "noodle.pt")
        ds = SwingSequenceDataset(
            tmp, window_events=8, window_hop=8, exclude_categories=["noodle"]
        )
        assert len(ds) == 0


# ---------------------------------------------------------------------------
# Sample shape and content
# ---------------------------------------------------------------------------


def test_getitem_returns_expected_keys() -> None:
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        _make_v6_pt(tmp, n_notes=40)
        ds = SwingSequenceDataset(
            tmp, window_events=8, window_hop=8, context_frames=64, phrase_frames=128
        )
        sample = ds[0]
        for key in ("tokens", "token_length", "saber_state", "mel", "phrase_mel",
                    "structure", "difficulty", "genre", "mapper_id"):
            assert key in sample, f"missing key: {key}"


def test_getitem_token_shapes() -> None:
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        _make_v6_pt(tmp, n_notes=40)
        ds = SwingSequenceDataset(
            tmp, window_events=8, window_hop=8, max_swing_len=128
        )
        sample = ds[0]
        assert sample["tokens"].shape == (128,)
        assert sample["tokens"].dtype == torch.long


def test_getitem_token_length_le_max() -> None:
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        _make_v6_pt(tmp, n_notes=40)
        ds = SwingSequenceDataset(
            tmp, window_events=8, window_hop=8, max_swing_len=128
        )
        sample = ds[0]
        assert int(sample["token_length"].item()) <= 128


def test_getitem_saber_state_shape() -> None:
    """V6: saber state is per-token, shape [max_swing_len, 12]."""
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        _make_v6_pt(tmp, n_notes=40)
        max_swing_len = 128
        ds = SwingSequenceDataset(
            tmp, window_events=8, window_hop=8, max_swing_len=max_swing_len,
        )
        sample = ds[0]
        assert sample["saber_state"].shape == (max_swing_len, 12)
        assert sample["saber_state"].dtype == torch.float32


def test_getitem_saber_state_finite() -> None:
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        _make_v6_pt(tmp, n_notes=40)
        ds = SwingSequenceDataset(tmp, window_events=8, window_hop=8)
        sample = ds[0]
        assert torch.isfinite(sample["saber_state"]).all()


def test_getitem_mel_shape() -> None:
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        _make_v6_pt(tmp, n_notes=40)
        ds = SwingSequenceDataset(
            tmp, window_events=8, window_hop=8, context_frames=64, phrase_frames=256
        )
        sample = ds[0]
        assert sample["mel"].shape == (80, 64)
        assert sample["phrase_mel"].shape == (80, 256)


def test_getitem_structure_shape() -> None:
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        _make_v6_pt(tmp, n_notes=40)
        ds = SwingSequenceDataset(tmp, window_events=8, window_hop=8, context_frames=64)
        sample = ds[0]
        assert sample["structure"].shape == (8, 64)


def test_getitem_tokens_in_vocab() -> None:
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        _make_v6_pt(tmp, n_notes=40)
        ds = SwingSequenceDataset(tmp, window_events=8, window_hop=8, max_swing_len=128)
        sample = ds[0]
        tokens = sample["tokens"]
        assert (tokens >= 0).all()
        assert (tokens < VOCAB_SIZE).all()


def test_getitem_starts_with_bos_or_pad() -> None:
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        _make_v6_pt(tmp, n_notes=40)
        ds = SwingSequenceDataset(tmp, window_events=8, window_hop=8)
        sample = ds[0]
        first = int(sample["tokens"][0].item())
        assert first in (BOS, PAD)


def test_getitem_pad_after_eos() -> None:
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        _make_v6_pt(tmp, n_notes=40)
        ds = SwingSequenceDataset(tmp, window_events=8, window_hop=8, max_swing_len=128)
        sample = ds[0]
        tokens = sample["tokens"].tolist()
        if EOS in tokens:
            eos_pos = tokens.index(EOS)
            assert all(t == PAD for t in tokens[eos_pos + 1:])


def test_getitem_mapper_id() -> None:
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        _make_v6_pt(tmp, n_notes=40)
        ds = SwingSequenceDataset(tmp, window_events=8, window_hop=8, mapper_id=7)
        sample = ds[0]
        assert int(sample["mapper_id"].item()) == 7


def test_getitem_difficulty_range() -> None:
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        _make_v6_pt(tmp, n_notes=40, difficulties=["Expert"])
        ds = SwingSequenceDataset(tmp, window_events=8, window_hop=8)
        sample = ds[0]
        diff = int(sample["difficulty"].item())
        assert 0 <= diff <= 4


# ---------------------------------------------------------------------------
# Round-trip: tokens from batch → valid DifficultyBeatmap
# ---------------------------------------------------------------------------


def test_round_trip_from_batch() -> None:
    """DoD 2.4: decode batch tokens back to a valid DifficultyBeatmap."""
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        _make_v6_pt(tmp, n_notes=40)
        ds = SwingSequenceDataset(tmp, window_events=16, window_hop=8, max_swing_len=256)
        sample = ds[0]
        tok_len = int(sample["token_length"].item())
        tokens = sample["tokens"][:tok_len].tolist()
        bm = _TOK.decode_beatmap(tokens)
        # Decoded beatmap should have at least some notes
        assert isinstance(bm, DifficultyBeatmap)
        total_events = (
            len(bm.color_notes) + len(bm.sliders) * 2
            + len(bm.burst_sliders) * 2 + len(bm.bomb_notes)
        )
        assert total_events > 0


def test_round_trip_with_arcs_and_chains() -> None:
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        # Build a custom beatmap with arcs + chains
        notes = [ColorNote(beat=float(i), x=i % 4, y=0, color=i % 2, direction=i % 9)
                 for i in range(10)]
        sliders = [Slider(color=0, beat=10.0, x=1, y=0, direction=0, mu=1.0,
                          tail_beat=12.0, tail_x=1, tail_y=2, tail_direction=1, tail_mu=1.0)]
        chains = [BurstSlider(color=1, beat=14.0, x=2, y=1, direction=3,
                              tail_beat=14.5, tail_x=3, tail_y=1, slice_count=4, squish=0.5)]
        bm = DifficultyBeatmap(
            version="3.3.0",
            color_notes=notes,
            bomb_notes=[],
            obstacles=[],
            sliders=sliders,
            burst_sliders=chains,
        )
        swing_toks = _TOK.encode_beatmap(bm)

        data = {
            "song_id": "mixed",
            "bpm": 120.0,
            "mel_spectrogram": torch.zeros(80, 2048),
            "difficulties": {"Expert": {"swing_tokens": swing_toks}},
            "mod_requirements": {"category": "vanilla", "genre": "rock"},
        }
        torch.save(data, tmp / "mixed.pt")

        # 10 notes + 2 arc events + 2 chain events = 14 events total
        ds = SwingSequenceDataset(tmp, window_events=8, window_hop=4, max_swing_len=256)
        assert len(ds) > 0
        sample = ds[0]
        tok_len = int(sample["token_length"].item())
        out_bm = _TOK.decode_beatmap(sample["tokens"][:tok_len].tolist())
        assert isinstance(out_bm, DifficultyBeatmap)


# ---------------------------------------------------------------------------
# Collate function
# ---------------------------------------------------------------------------


def test_swing_collate_batch() -> None:
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        _make_v6_pt(tmp, n_notes=40)
        ds = SwingSequenceDataset(
            tmp, window_events=8, window_hop=8,
            context_frames=64, phrase_frames=128, max_swing_len=128,
        )
        assert len(ds) >= 2
        batch = swing_collate_fn([ds[0], ds[1]])
        assert batch["tokens"].shape == (2, 128)
        # V6: saber_state is per-token, matching max_swing_len
        assert batch["saber_state"].shape == (2, 128, 12)
        assert batch["mel"].shape == (2, 80, 64)
        assert batch["phrase_mel"].shape == (2, 80, 128)
        assert batch["mapper_id"].shape == (2,)


def test_swing_collate_single() -> None:
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        _make_v6_pt(tmp, n_notes=40)
        ds = SwingSequenceDataset(tmp, window_events=8, window_hop=8, max_swing_len=128)
        batch = swing_collate_fn([ds[0]])
        assert batch["tokens"].shape == (1, 128)


# ---------------------------------------------------------------------------
# Mirror augmentation
# ---------------------------------------------------------------------------


def test_mirror_augment_produces_different_tokens() -> None:
    """With mirror augment enabled, at least some samples should differ from plain."""
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        _make_v6_pt(tmp, n_notes=40)
        ds_plain = SwingSequenceDataset(
            tmp, window_events=8, window_hop=8, mirror_augment=False
        )
        ds_mirror = SwingSequenceDataset(
            tmp, window_events=8, window_hop=8, mirror_augment=True
        )
        # Draw 20 samples and check at least one differs in HAND tokens
        import random as rnd
        rnd.seed(42)
        diffs_found = 0
        for i in range(min(20, len(ds_plain))):
            t_plain = ds_plain[i]["tokens"]
            # Sample mirror version multiple times (50% flip prob)
            for _ in range(6):
                t_mirror = ds_mirror[i]["tokens"]
                if not torch.equal(t_plain, t_mirror):
                    diffs_found += 1
                    break
        assert diffs_found > 0, "mirror augment produced no differences across 20 samples"


# ---------------------------------------------------------------------------
# Conversion utility smoke test
# ---------------------------------------------------------------------------


def test_convert_v5_to_v6() -> None:
    """Test the V5→V6 conversion utility on a synthetic V5 .pt file."""
    from scripts.convert_to_swing import convert_file
    from beatsaber_automapper.data.tokenizer import BeatmapTokenizer

    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)

        # Build a V5-format .pt file (chord tokens per onset, no swing_tokens)
        notes_v5 = [ColorNote(beat=float(i) * 0.5, x=i % 4, y=0, color=i % 2, direction=i % 9)
                    for i in range(20)]
        v5_tok = BeatmapTokenizer()
        beat_tokens = {n.beat: v5_tok.encode_beatmap(
            DifficultyBeatmap(
                version="3.3.0",
                color_notes=[n], bomb_notes=[], obstacles=[],
                sliders=[], burst_sliders=[],
            )
        )[n.beat] for n in notes_v5}

        bpm = 120.0
        fps = 44100 / 512
        fpb = fps * 60.0 / bpm

        onset_frames = torch.tensor([int(b * fpb) for b in sorted(beat_tokens.keys())],
                                    dtype=torch.long)
        token_seqs = [beat_tokens[b] for b in sorted(beat_tokens.keys())]

        data = {
            "song_id": "v5song",
            "bpm": bpm,
            "mel_spectrogram": torch.zeros(80, 512),
            "difficulties": {
                "Expert": {
                    "onset_frames": onset_frames,
                    "token_sequences": token_seqs,
                    "onset_labels": torch.zeros(512),
                }
            },
            "mod_requirements": {"category": "vanilla", "genre": "rock"},
        }
        pt_path = tmp / "v5song.pt"
        torch.save(data, pt_path)

        swing_tok = SwingEventTokenizer()
        stats = convert_file(pt_path, swing_tok, v5_tok)

        assert stats["converted"] == 1
        assert stats["failed"] == 0

        # Reload and verify swing_tokens present
        out = torch.load(pt_path, weights_only=False)
        assert "swing_tokens" in out["difficulties"]["Expert"]
        swing_toks = out["difficulties"]["Expert"]["swing_tokens"]
        assert len(swing_toks) > 2  # not just BOS EOS
        assert swing_toks[0] == BOS
        assert swing_toks[-1] == EOS


def test_convert_skips_existing_by_default() -> None:
    """convert_file should skip diffs that already have swing_tokens."""
    from scripts.convert_to_swing import convert_file
    from beatsaber_automapper.data.tokenizer import BeatmapTokenizer

    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        existing_swing = _make_swing_tokens(10)
        data = {
            "song_id": "already",
            "bpm": 120.0,
            "mel_spectrogram": torch.zeros(80, 256),
            "difficulties": {
                "Expert": {
                    "onset_frames": torch.tensor([0]),
                    "token_sequences": [[EOS]],
                    "swing_tokens": existing_swing,
                }
            },
            "mod_requirements": {"category": "vanilla", "genre": "rock"},
        }
        pt_path = tmp / "already.pt"
        torch.save(data, pt_path)

        swing_tok = SwingEventTokenizer()
        v5_tok = BeatmapTokenizer()
        stats = convert_file(pt_path, swing_tok, v5_tok)

        assert stats["skipped"] == 1
        assert stats["converted"] == 0
