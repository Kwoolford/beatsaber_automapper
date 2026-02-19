"""Tests for beam search and nucleus sampling decoding."""

import pytest
import torch

from beatsaber_automapper.data.tokenizer import BOS, EOS, VOCAB_SIZE
from beatsaber_automapper.generation.beam_search import (
    beam_search_decode,
    nucleus_sampling_decode,
)
from beatsaber_automapper.models.sequence_model import SequenceModel


@pytest.fixture
def model():
    m = SequenceModel(
        vocab_size=VOCAB_SIZE,
        d_model=64,
        nhead=4,
        num_layers=1,
        dim_feedforward=128,
        num_difficulties=5,
        dropout=0.0,
    )
    m.eval()
    return m


@pytest.fixture
def audio_features():
    return torch.randn(1, 10, 64)


@pytest.fixture
def difficulty():
    return torch.tensor([2])


class TestBeamSearch:
    """Test beam search decoding."""

    def test_returns_list_of_ints(self, model, audio_features, difficulty):
        result = beam_search_decode(model, audio_features, difficulty, beam_size=2, max_length=10)
        assert isinstance(result, list)
        assert all(isinstance(t, int) for t in result)

    def test_no_bos_in_output(self, model, audio_features, difficulty):
        result = beam_search_decode(model, audio_features, difficulty, beam_size=2, max_length=10)
        assert BOS not in result

    def test_no_eos_in_output(self, model, audio_features, difficulty):
        result = beam_search_decode(model, audio_features, difficulty, beam_size=2, max_length=10)
        assert EOS not in result

    def test_max_length_respected(self, model, audio_features, difficulty):
        result = beam_search_decode(model, audio_features, difficulty, beam_size=2, max_length=5)
        assert len(result) <= 5

    def test_beam_size_1_deterministic(self, model, audio_features, difficulty):
        """Beam size 1 should be equivalent to greedy decoding (deterministic)."""
        torch.manual_seed(42)
        result1 = beam_search_decode(model, audio_features, difficulty, beam_size=1, max_length=8)
        torch.manual_seed(42)
        result2 = beam_search_decode(model, audio_features, difficulty, beam_size=1, max_length=8)
        assert result1 == result2


class TestNucleusSampling:
    """Test nucleus sampling decoding."""

    def test_returns_list_of_ints(self, model, audio_features, difficulty):
        torch.manual_seed(42)
        result = nucleus_sampling_decode(
            model, audio_features, difficulty, max_length=10, top_p=0.9
        )
        assert isinstance(result, list)
        assert all(isinstance(t, int) for t in result)

    def test_no_bos_in_output(self, model, audio_features, difficulty):
        torch.manual_seed(42)
        result = nucleus_sampling_decode(
            model, audio_features, difficulty, max_length=10, top_p=0.9
        )
        assert BOS not in result

    def test_max_length_respected(self, model, audio_features, difficulty):
        torch.manual_seed(42)
        result = nucleus_sampling_decode(model, audio_features, difficulty, max_length=5, top_p=0.9)
        assert len(result) <= 5

    def test_no_eos_in_output(self, model, audio_features, difficulty):
        torch.manual_seed(42)
        result = nucleus_sampling_decode(
            model, audio_features, difficulty, max_length=10, top_p=0.9
        )
        assert EOS not in result
