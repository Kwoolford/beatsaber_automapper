"""Tests for V6 grammar-constrained swing-event decoder."""

from __future__ import annotations

import torch
import pytest

from beatsaber_automapper.data.swing_tokenizer import (
    ARC_HEAD,
    ARC_TAIL,
    BOMB,
    BOS,
    CHAIN_HEAD,
    CHAIN_TAIL,
    DT_BASE,
    EOS,
    HAND_LEFT,
    HAND_NONE,
    HAND_RIGHT,
    KIND_BASE,
    NOTE,
    PAD,
    VOCAB_SIZE,
    X_BASE,
    Y_BASE,
    DIR_BASE,
    ANGLE_BASE,
    MU_BASE,
    SLICE_BASE,
    SQUISH_BASE,
)
from beatsaber_automapper.generation.beam_search_v6 import (
    _GrammarState,
    _Phase,
    _build_mask,
    _nucleus_sample,
    _record_field,
    _transition,
    nucleus_sampling_v6,
)
from beatsaber_automapper.models.sequence_model import SequenceModel


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def small_model():
    m = SequenceModel(
        vocab_size=VOCAB_SIZE,
        d_model=64,
        nhead=4,
        num_layers=1,
        dim_feedforward=128,
        num_difficulties=5,
        num_genres=11,
        dropout=0.0,
    )
    m.eval()
    return m


@pytest.fixture
def audio_features():
    return torch.randn(1, 8, 64)


@pytest.fixture
def diff():
    return torch.tensor([3])


@pytest.fixture
def genre():
    return torch.tensor([0])


# ---------------------------------------------------------------------------
# Grammar state machine unit tests
# ---------------------------------------------------------------------------


class TestGrammarMask:
    def test_expect_hand_allows_hands_and_eos(self):
        mask = _build_mask(_Phase.EXPECT_HAND, HAND_LEFT, NOTE, VOCAB_SIZE)
        assert mask[HAND_LEFT].item() == 0.0
        assert mask[HAND_RIGHT].item() == 0.0
        assert mask[HAND_NONE].item() == 0.0
        assert mask[EOS].item() == 0.0
        assert mask[NOTE].item() == float("-inf")
        assert mask[DT_BASE].item() == float("-inf")

    def test_expect_dt_allows_only_dt_range(self):
        mask = _build_mask(_Phase.EXPECT_DT, HAND_LEFT, NOTE, VOCAB_SIZE)
        for i in range(32):
            assert mask[DT_BASE + i].item() == 0.0
        assert mask[HAND_LEFT].item() == float("-inf")
        assert mask[NOTE].item() == float("-inf")

    def test_expect_kind_none_hand_only_bomb(self):
        mask = _build_mask(_Phase.EXPECT_KIND, HAND_NONE, NOTE, VOCAB_SIZE)
        assert mask[BOMB].item() == 0.0
        assert mask[NOTE].item() == float("-inf")
        assert mask[ARC_HEAD].item() == float("-inf")

    def test_expect_kind_swing_hands_no_bomb(self):
        mask = _build_mask(_Phase.EXPECT_KIND, HAND_LEFT, NOTE, VOCAB_SIZE)
        assert mask[BOMB].item() == float("-inf")
        assert mask[NOTE].item() == 0.0
        assert mask[ARC_HEAD].item() == 0.0
        assert mask[CHAIN_TAIL].item() == 0.0

    def test_expect_x_allows_only_x_range(self):
        mask = _build_mask(_Phase.EXPECT_X, HAND_LEFT, NOTE, VOCAB_SIZE)
        for i in range(4):
            assert mask[X_BASE + i].item() == 0.0
        assert mask[Y_BASE].item() == float("-inf")
        assert mask[NOTE].item() == float("-inf")

    def test_expect_field_d_note_uses_angle_range(self):
        mask = _build_mask(_Phase.EXPECT_FIELD_D, HAND_LEFT, NOTE, VOCAB_SIZE)
        for i in range(7):
            assert mask[ANGLE_BASE + i].item() == 0.0
        assert mask[MU_BASE].item() == float("-inf")

    def test_expect_field_d_arc_uses_mu_range(self):
        mask = _build_mask(_Phase.EXPECT_FIELD_D, HAND_LEFT, ARC_HEAD, VOCAB_SIZE)
        for i in range(9):
            assert mask[MU_BASE + i].item() == 0.0
        assert mask[ANGLE_BASE].item() == float("-inf")

    def test_expect_field_d_chain_head_uses_slice_range(self):
        mask = _build_mask(_Phase.EXPECT_FIELD_D, HAND_LEFT, CHAIN_HEAD, VOCAB_SIZE)
        for i in range(31):
            assert mask[SLICE_BASE + i].item() == 0.0
        assert mask[ANGLE_BASE].item() == float("-inf")

    def test_expect_squish_allows_squish_range(self):
        mask = _build_mask(_Phase.EXPECT_SQUISH, HAND_LEFT, CHAIN_TAIL, VOCAB_SIZE)
        for i in range(11):
            assert mask[SQUISH_BASE + i].item() == 0.0
        assert mask[DIR_BASE].item() == float("-inf")


class TestGrammarTransitions:
    def test_hand_to_dt(self):
        s = _GrammarState()
        assert s.phase == _Phase.EXPECT_HAND
        _transition(s, HAND_LEFT)
        assert s.phase == _Phase.EXPECT_DT
        assert s.current_hand == HAND_LEFT

    def test_eos_terminates(self):
        s = _GrammarState()
        _transition(s, EOS)
        assert s.phase == _Phase.DONE

    def test_full_note_event_cycle(self):
        s = _GrammarState()
        _transition(s, HAND_LEFT)           # → EXPECT_DT
        _transition(s, DT_BASE + 4)         # → EXPECT_KIND
        _transition(s, NOTE)                # → EXPECT_X
        _transition(s, X_BASE + 1)          # → EXPECT_Y
        _transition(s, Y_BASE + 0)          # → EXPECT_DIR
        _transition(s, DIR_BASE + 1)        # → EXPECT_FIELD_D
        _record_field(s, DIR_BASE + 1, _Phase.EXPECT_DIR)
        _transition(s, ANGLE_BASE + 3)      # → EXPECT_HAND (event done)
        assert s.phase == _Phase.EXPECT_HAND

    def test_bomb_event_cycle(self):
        s = _GrammarState()
        _transition(s, HAND_NONE)           # → EXPECT_DT
        _transition(s, DT_BASE + 0)         # → EXPECT_KIND
        _transition(s, BOMB)                # → EXPECT_X
        _transition(s, X_BASE + 2)          # → EXPECT_Y
        _transition(s, Y_BASE + 1)          # → EXPECT_HAND (5-token done)
        assert s.phase == _Phase.EXPECT_HAND

    def test_chain_tail_event_cycle(self):
        s = _GrammarState()
        _transition(s, HAND_RIGHT)          # → EXPECT_DT
        _transition(s, DT_BASE + 8)         # → EXPECT_KIND
        _transition(s, CHAIN_TAIL)          # → EXPECT_X
        _transition(s, X_BASE + 3)          # → EXPECT_Y
        _transition(s, Y_BASE + 2)          # → EXPECT_SQUISH
        assert s.phase == _Phase.EXPECT_SQUISH
        _transition(s, SQUISH_BASE + 5)     # → EXPECT_HAND (6-token done)
        assert s.phase == _Phase.EXPECT_HAND

    def test_arc_head_event_cycle(self):
        s = _GrammarState()
        _transition(s, HAND_LEFT)
        _transition(s, DT_BASE + 2)
        _transition(s, ARC_HEAD)
        _transition(s, X_BASE + 0)
        _transition(s, Y_BASE + 0)
        assert s.phase == _Phase.EXPECT_DIR
        _transition(s, DIR_BASE + 0)
        assert s.phase == _Phase.EXPECT_FIELD_D
        _transition(s, MU_BASE + 4)
        assert s.phase == _Phase.EXPECT_HAND


class TestSaberStateUpdate:
    def test_saber_position_updates_after_note(self):
        s = _GrammarState()
        _transition(s, HAND_LEFT)
        _transition(s, DT_BASE + 0)
        _transition(s, NOTE)
        _record_field(s, X_BASE + 2, _Phase.EXPECT_X)
        _transition(s, X_BASE + 2)
        _record_field(s, Y_BASE + 1, _Phase.EXPECT_Y)
        _transition(s, Y_BASE + 1)
        _record_field(s, DIR_BASE + 1, _Phase.EXPECT_DIR)
        _transition(s, DIR_BASE + 1)     # → EXPECT_FIELD_D
        _transition(s, ANGLE_BASE + 3)   # event done → EXPECT_HAND, saber updated
        # L_x should be 2/3 ≈ 0.667
        assert abs(s.saber[0] - 2 / 3) < 0.01
        # L_y should be 1/2 = 0.5
        assert abs(s.saber[1] - 0.5) < 0.01

    def test_right_hand_not_affected_by_left_event(self):
        s = _GrammarState()
        # Do a left-hand event using record+transition pairs
        _transition(s, HAND_LEFT)
        _transition(s, DT_BASE + 0)
        _transition(s, NOTE)
        _record_field(s, X_BASE + 1, _Phase.EXPECT_X)
        _transition(s, X_BASE + 1)
        _record_field(s, Y_BASE + 0, _Phase.EXPECT_Y)
        _transition(s, Y_BASE + 0)
        _record_field(s, DIR_BASE + 1, _Phase.EXPECT_DIR)
        _transition(s, DIR_BASE + 1)
        _transition(s, ANGLE_BASE + 3)
        # Right hand saber state should be unchanged (still 0)
        assert s.saber[6] == 0.0   # R_x
        assert s.saber[7] == 0.0   # R_y


# ---------------------------------------------------------------------------
# Nucleus sample
# ---------------------------------------------------------------------------


def test_nucleus_sample_valid_token():
    logits = torch.zeros(VOCAB_SIZE)
    tok = _nucleus_sample(logits, temperature=1.0, top_p=0.9)
    assert 0 <= tok < VOCAB_SIZE


def test_nucleus_sample_respects_neg_inf():
    logits = torch.full((VOCAB_SIZE,), float("-inf"))
    logits[5] = 0.0   # only token 5 is allowed
    tok = _nucleus_sample(logits, temperature=1.0, top_p=0.95)
    assert tok == 5


# ---------------------------------------------------------------------------
# End-to-end nucleus sampling
# ---------------------------------------------------------------------------


class TestNucleusSamplingV6:
    def test_returns_list(self, small_model, audio_features, diff, genre):
        tokens = nucleus_sampling_v6(
            small_model, audio_features, diff, genre,
            max_events=8, max_tokens=128, temperature=1.0, top_p=0.95,
        )
        assert isinstance(tokens, list)

    def test_no_bos_eos_in_output(self, small_model, audio_features, diff, genre):
        tokens = nucleus_sampling_v6(
            small_model, audio_features, diff, genre,
            max_events=8, max_tokens=64,
        )
        assert BOS not in tokens
        assert EOS not in tokens
        assert PAD not in tokens

    def test_all_tokens_in_vocab(self, small_model, audio_features, diff, genre):
        tokens = nucleus_sampling_v6(
            small_model, audio_features, diff, genre,
            max_events=8, max_tokens=128,
        )
        for t in tokens:
            assert 0 <= t < VOCAB_SIZE, f"token {t} out of vocab range"

    def test_grammar_valid_token_sequence(self, small_model, audio_features, diff, genre):
        """Verify that the generated token sequence satisfies the V6 grammar."""
        tokens = nucleus_sampling_v6(
            small_model, audio_features, diff, genre,
            max_events=16, max_tokens=256, temperature=1.0, top_p=1.0,
        )
        if not tokens:
            return  # empty is acceptable (all EOS)

        # Replay through grammar state machine; all tokens must pass mask
        state = _GrammarState()
        for tok in tokens:
            if state.phase == _Phase.DONE:
                break
            mask = _build_mask(state.phase, state.current_hand, state.current_kind, VOCAB_SIZE)
            assert mask[tok].item() == 0.0, (
                f"Token {tok} blocked by grammar mask in phase {state.phase}"
            )
            _record_field(state, tok, state.phase)
            _transition(state, tok)

    def test_decodes_to_valid_beatmap(self, small_model, audio_features, diff, genre):
        """Round-trip: generated tokens decode to a DifficultyBeatmap."""
        from beatsaber_automapper.data.beatmap import DifficultyBeatmap
        from beatsaber_automapper.data.swing_tokenizer import SwingEventTokenizer

        tokens = nucleus_sampling_v6(
            small_model, audio_features, diff, genre,
            max_events=16, max_tokens=256,
        )
        bm = SwingEventTokenizer().decode_beatmap(tokens)
        assert isinstance(bm, DifficultyBeatmap)

    def test_max_events_respected(self, small_model, audio_features, diff, genre):
        from beatsaber_automapper.data.swing_tokenizer import SwingEventTokenizer

        tokens = nucleus_sampling_v6(
            small_model, audio_features, diff, genre,
            max_events=4, max_tokens=512,
        )
        # Count complete events (HAND tokens in output)
        event_count = sum(1 for t in tokens if t in (HAND_LEFT, HAND_RIGHT, HAND_NONE))
        assert event_count <= 4


# ---------------------------------------------------------------------------
# generate_swing_level smoke test
# ---------------------------------------------------------------------------


def test_generate_swing_level_creates_zip(tmp_path):
    """End-to-end V6 level generation smoke test."""
    import wave
    import zipfile
    from beatsaber_automapper.generation.generate import generate_swing_level

    wav = tmp_path / "song.wav"
    n_samples = int(3.0 * 44100)  # needs ≥ 2s for detect_sections
    with wave.open(str(wav), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(44100)
        wf.writeframes(b"\x00\x00" * n_samples)

    out = tmp_path / "level.zip"
    result = generate_swing_level(
        audio_path=wav,
        output_path=out,
        difficulty="Expert",
        bpm=120.0,
        max_events=16,
        context_frames=64,
        phrase_frames=128,
        device="cpu",
    )
    assert result == out
    assert out.exists()
    assert zipfile.is_zipfile(out)
