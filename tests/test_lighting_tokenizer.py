"""Tests for LightingTokenizer — encode/decode round-trip and vocabulary constants."""

from __future__ import annotations

import pytest

from beatsaber_automapper.data.beatmap import BasicEvent, ColorBoostEvent, DifficultyBeatmap
from beatsaber_automapper.data.tokenizer import (
    LIGHT_BASIC,
    LIGHT_BOOST,
    LIGHT_BOS,
    LIGHT_BRIGHT_COUNT,
    LIGHT_BRIGHT_OFFSET,
    LIGHT_EOS,
    LIGHT_ET_COUNT,
    LIGHT_ET_OFFSET,
    LIGHT_ONOFF_OFFSET,
    LIGHT_PAD,
    LIGHT_SEP,
    LIGHT_VAL_OFFSET,
    LIGHT_VOCAB_SIZE,
    LightingTokenizer,
)


@pytest.fixture
def tokenizer() -> LightingTokenizer:
    return LightingTokenizer()


@pytest.fixture
def beatmap_with_lights() -> DifficultyBeatmap:
    return DifficultyBeatmap(
        version="3.3.0",
        basic_events=[
            BasicEvent(beat=1.0, event_type=0, value=1, float_value=1.0),
            BasicEvent(beat=2.0, event_type=4, value=3, float_value=0.5),
            BasicEvent(beat=2.0, event_type=1, value=0, float_value=0.0),  # second event same beat
        ],
        color_boost_events=[
            ColorBoostEvent(beat=1.0, boost=True),
            ColorBoostEvent(beat=3.0, boost=False),
        ],
    )


# ---------------------------------------------------------------------------
# Vocabulary constants
# ---------------------------------------------------------------------------


class TestVocabConstants:
    def test_special_tokens_unique(self):
        assert len({LIGHT_PAD, LIGHT_EOS, LIGHT_SEP, LIGHT_BOS}) == 4

    def test_special_tokens_low_ids(self):
        assert LIGHT_PAD == 0
        assert LIGHT_EOS == 1
        assert LIGHT_SEP == 2
        assert LIGHT_BOS == 3

    def test_event_type_tokens(self):
        assert LIGHT_BASIC == 4
        assert LIGHT_BOOST == 5

    def test_et_range(self):
        assert LIGHT_ET_OFFSET == 6
        assert LIGHT_ET_COUNT == 15
        assert LIGHT_ET_OFFSET + LIGHT_ET_COUNT - 1 == 20

    def test_val_range(self):
        assert LIGHT_VAL_OFFSET == 21

    def test_bright_range(self):
        assert LIGHT_BRIGHT_OFFSET == 29
        assert LIGHT_BRIGHT_COUNT == 4

    def test_vocab_size(self):
        assert LIGHT_VOCAB_SIZE == 35


# ---------------------------------------------------------------------------
# encode_lighting
# ---------------------------------------------------------------------------


class TestEncodeLighting:
    def test_returns_dict(self, tokenizer, beatmap_with_lights):
        result = tokenizer.encode_lighting(beatmap_with_lights)
        assert isinstance(result, dict)

    def test_beat_keys(self, tokenizer, beatmap_with_lights):
        result = tokenizer.encode_lighting(beatmap_with_lights)
        # Beats 1.0, 2.0, 3.0 from events; 1.0 also has boost
        assert 1.0 in result
        assert 2.0 in result
        assert 3.0 in result

    def test_each_sequence_ends_with_eos(self, tokenizer, beatmap_with_lights):
        result = tokenizer.encode_lighting(beatmap_with_lights)
        for tokens in result.values():
            assert tokens[-1] == LIGHT_EOS

    def test_basic_event_token_structure(self, tokenizer):
        bm = DifficultyBeatmap(
            version="3.3.0",
            basic_events=[BasicEvent(beat=1.0, event_type=0, value=1, float_value=1.0)],
        )
        result = tokenizer.encode_lighting(bm)
        tokens = result[1.0]
        # Should be [LIGHT_BASIC, ET, VAL, BRIGHT, EOS]
        assert tokens[0] == LIGHT_BASIC
        assert tokens[1] == LIGHT_ET_OFFSET + 0  # et=0
        assert tokens[2] == LIGHT_VAL_OFFSET + 1  # val=1
        # BRIGHT bin for 1.0 should be the last bin (index 3)
        assert tokens[3] == LIGHT_BRIGHT_OFFSET + 3
        assert tokens[4] == LIGHT_EOS

    def test_boost_event_token_structure(self, tokenizer):
        bm = DifficultyBeatmap(
            version="3.3.0",
            color_boost_events=[ColorBoostEvent(beat=1.0, boost=True)],
        )
        result = tokenizer.encode_lighting(bm)
        tokens = result[1.0]
        # Should be [LIGHT_BOOST, ONOFF(1), EOS]
        assert tokens[0] == LIGHT_BOOST
        assert tokens[1] == LIGHT_ONOFF_OFFSET + 1  # boost=True -> 1
        assert tokens[2] == LIGHT_EOS

    def test_multiple_events_same_beat_have_sep(self, tokenizer, beatmap_with_lights):
        result = tokenizer.encode_lighting(beatmap_with_lights)
        tokens = result[2.0]  # two basic events at beat 2.0
        assert LIGHT_SEP in tokens

    def test_empty_beatmap_returns_empty_dict(self, tokenizer):
        bm = DifficultyBeatmap(version="3.3.0")
        result = tokenizer.encode_lighting(bm)
        assert result == {}

    def test_et_clamped_to_range(self, tokenizer):
        bm = DifficultyBeatmap(
            version="3.3.0",
            basic_events=[BasicEvent(beat=1.0, event_type=99, value=0, float_value=0.0)],
        )
        result = tokenizer.encode_lighting(bm)
        tokens = result[1.0]
        et_token = tokens[1]
        # Should be clamped to max valid ET
        assert LIGHT_ET_OFFSET <= et_token < LIGHT_ET_OFFSET + LIGHT_ET_COUNT


# ---------------------------------------------------------------------------
# decode_lighting
# ---------------------------------------------------------------------------


class TestDecodeLighting:
    def test_returns_tuple_of_lists(self, tokenizer):
        basic, boost = tokenizer.decode_lighting({})
        assert isinstance(basic, list)
        assert isinstance(boost, list)

    def test_empty_input(self, tokenizer):
        basic, boost = tokenizer.decode_lighting({})
        assert basic == []
        assert boost == []

    def test_decode_basic_event(self, tokenizer):
        tokens = [LIGHT_BASIC, LIGHT_ET_OFFSET + 0, LIGHT_VAL_OFFSET + 1,
                  LIGHT_BRIGHT_OFFSET + 3, LIGHT_EOS]
        basic, boost = tokenizer.decode_lighting({1.0: tokens})
        assert len(basic) == 1
        assert basic[0].beat == 1.0
        assert basic[0].event_type == 0
        assert basic[0].value == 1
        assert basic[0].float_value == pytest.approx(1.0, abs=0.01)

    def test_decode_boost_event(self, tokenizer):
        tokens = [LIGHT_BOOST, LIGHT_ONOFF_OFFSET + 1, LIGHT_EOS]
        basic, boost = tokenizer.decode_lighting({2.0: tokens})
        assert len(boost) == 1
        assert boost[0].beat == 2.0
        assert boost[0].boost is True

    def test_decode_boost_off(self, tokenizer):
        tokens = [LIGHT_BOOST, LIGHT_ONOFF_OFFSET + 0, LIGHT_EOS]
        basic, boost = tokenizer.decode_lighting({1.0: tokens})
        assert boost[0].boost is False

    def test_truncated_tokens_dont_crash(self, tokenizer):
        # LIGHT_BASIC without enough following tokens
        tokens = [LIGHT_BASIC, LIGHT_ET_OFFSET + 0, LIGHT_EOS]  # only 2 attr instead of 3
        basic, boost = tokenizer.decode_lighting({1.0: tokens})
        # Should not raise, just skip the malformed event
        assert basic == []


# ---------------------------------------------------------------------------
# Round-trip
# ---------------------------------------------------------------------------


class TestRoundTrip:
    def test_basic_event_round_trip(self, tokenizer):
        bm = DifficultyBeatmap(
            version="3.3.0",
            basic_events=[
                BasicEvent(beat=1.0, event_type=4, value=3, float_value=1.0),
            ],
        )
        encoded = tokenizer.encode_lighting(bm)
        basic, boost = tokenizer.decode_lighting(encoded)
        assert len(basic) == 1
        assert basic[0].beat == 1.0
        assert basic[0].event_type == 4
        assert basic[0].value == 3
        # float_value will be approximately 1.0 after quantization
        assert basic[0].float_value == pytest.approx(1.0, abs=0.4)

    def test_boost_event_round_trip(self, tokenizer):
        bm = DifficultyBeatmap(
            version="3.3.0",
            color_boost_events=[
                ColorBoostEvent(beat=2.0, boost=True),
            ],
        )
        encoded = tokenizer.encode_lighting(bm)
        basic, boost = tokenizer.decode_lighting(encoded)
        assert len(boost) == 1
        assert boost[0].beat == 2.0
        assert boost[0].boost is True

    def test_multi_event_round_trip(self, tokenizer, beatmap_with_lights):
        encoded = tokenizer.encode_lighting(beatmap_with_lights)
        basic, boost = tokenizer.decode_lighting(encoded)
        # 3 basic events total
        assert len(basic) == 3
        # 2 boost events total
        assert len(boost) == 2
