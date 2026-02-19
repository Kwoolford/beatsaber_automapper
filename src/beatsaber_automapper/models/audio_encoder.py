"""Shared audio encoder: CNN frontend + Transformer encoder.

Processes mel spectrograms into contextualized audio frame embeddings
used by all three pipeline stages. Trained end-to-end on Beat Saber data
to capture low-level rhythmic features (transients, beat subdivisions).

Architecture:
    Mel spectrogram [B, n_mels, T] -> 2D CNN (4 layers) -> Transformer encoder
    (6 layers, 8 heads, d_model=512) -> frame embeddings [B, T, d_model]

The CNN strides only on the frequency axis (stride=(2,1)), preserving the
time dimension for per-frame onset prediction.
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn

from beatsaber_automapper.models.components import SinusoidalPositionalEncoding

logger = logging.getLogger(__name__)


class AudioEncoder(nn.Module):
    """CNN + Transformer encoder for audio feature extraction.

    Args:
        n_mels: Number of mel frequency bands in input spectrogram.
        d_model: Transformer model dimension.
        nhead: Number of attention heads.
        num_layers: Number of transformer encoder layers.
        dim_feedforward: Feed-forward network dimension.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        n_mels: int = 80,
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if n_mels % 16 != 0:
            raise ValueError(f"n_mels must be divisible by 16, got {n_mels}")

        self.n_mels = n_mels
        self.d_model = d_model

        # CNN frontend: 4 conv layers, stride=(2,1) reduces freq by 16x, keeps time
        channels = [1, 32, 64, 128, 256]
        cnn_layers: list[nn.Module] = []
        for i in range(4):
            cnn_layers.extend(
                [
                    nn.Conv2d(
                        channels[i],
                        channels[i + 1],
                        kernel_size=3,
                        stride=(2, 1),
                        padding=(1, 1),
                    ),
                    nn.BatchNorm2d(channels[i + 1]),
                    nn.GELU(),
                ]
            )
        self.cnn = nn.Sequential(*cnn_layers)

        # Project flattened CNN output to d_model
        freq_out = n_mels // 16
        self.proj = nn.Linear(channels[-1] * freq_out, d_model)

        # Positional encoding
        self.pos_enc = SinusoidalPositionalEncoding(d_model=d_model, dropout=dropout)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """Encode mel spectrogram into frame embeddings.

        Args:
            mel: Mel spectrogram tensor [B, n_mels, T].

        Returns:
            Frame embeddings [B, T, d_model].
        """
        # Add channel dim: [B, n_mels, T] -> [B, 1, n_mels, T]
        x = mel.unsqueeze(1)

        # CNN: [B, 1, n_mels, T] -> [B, 256, n_mels//16, T]
        x = self.cnn(x)

        # Reshape: [B, 256, freq_out, T] -> [B, T, 256*freq_out]
        b, c, f, t = x.shape
        x = x.permute(0, 3, 1, 2).reshape(b, t, c * f)

        # Project to d_model: [B, T, d_model]
        x = self.proj(x)

        # Positional encoding + Transformer
        x = self.pos_enc(x)
        x = self.transformer(x)

        return x
