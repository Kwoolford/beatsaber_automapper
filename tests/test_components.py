"""Tests for shared model components."""

from __future__ import annotations

import torch

from beatsaber_automapper.models.components import SinusoidalPositionalEncoding


def test_positional_encoding_shape() -> None:
    """Positional encoding output should match input shape."""
    pe = SinusoidalPositionalEncoding(d_model=64, max_len=500, dropout=0.0)
    x = torch.zeros(2, 100, 64)
    out = pe(x)
    assert out.shape == (2, 100, 64)


def test_positional_encoding_not_all_zeros() -> None:
    """Positional encoding should add non-zero values to zero input."""
    pe = SinusoidalPositionalEncoding(d_model=64, max_len=500, dropout=0.0)
    x = torch.zeros(1, 50, 64)
    out = pe(x)
    assert out.abs().sum() > 0


def test_positional_encoding_varies_by_position() -> None:
    """Different positions should get different encodings."""
    pe = SinusoidalPositionalEncoding(d_model=64, max_len=500, dropout=0.0)
    x = torch.zeros(1, 10, 64)
    out = pe(x)
    # Position 0 and position 5 should differ
    assert not torch.allclose(out[0, 0], out[0, 5])
