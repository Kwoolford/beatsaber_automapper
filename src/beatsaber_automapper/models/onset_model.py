"""Stage 1: Onset / beat prediction model.

Binary classification per audio frame — predicts whether a note should
appear at each time step. Uses a hybrid TCN + Transformer architecture:

- TCN (Temporal Convolutional Network): 4 blocks with dilated convolutions
  for efficient local temporal pattern detection. Large receptive fields
  via exponentially increasing dilation factors (1,2,4,8,16,32).
- 2-layer Transformer encoder on top for global context (verse/chorus/drop).

This replaces the original Transformer-only onset model, following
proven approaches from madmom, BeatNet, and InfernoSaber.

Training: Binary cross-entropy with logits (sigmoid applied at inference).
Inference: Peak picking with configurable threshold.
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn
import torch.utils.checkpoint

logger = logging.getLogger(__name__)


class TemporalConvBlock(nn.Module):
    """Single TCN residual block with dilated causal convolution.

    Architecture: Conv1d (dilated) -> BatchNorm -> GELU -> Dropout
                  -> Conv1d (1x1) -> BatchNorm -> GELU -> Dropout
                  + residual connection (with optional projection)

    Args:
        channels: Number of input and output channels.
        kernel_size: Convolution kernel size.
        dilation: Dilation factor for the first conv.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        # Causal padding: pad only on the left so output doesn't see future
        self.pad = (kernel_size - 1) * dilation

        self.conv1 = nn.Conv1d(
            channels, channels, kernel_size,
            dilation=dilation, padding=0,  # manual padding for causal
        )
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=1)
        self.bn2 = nn.BatchNorm1d(channels)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with residual connection.

        Args:
            x: Input tensor [B, C, T].

        Returns:
            Output tensor [B, C, T] (same shape).
        """
        residual = x

        # Causal padding: pad left only
        out = nn.functional.pad(x, (self.pad, 0))
        out = self.conv1(out)
        out = self.bn1(out)
        out = self.activation(out)
        out = self.dropout(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.activation(out)
        out = self.dropout(out)

        return out + residual


class TCNEncoder(nn.Module):
    """Temporal Convolutional Network encoder.

    Stacks multiple TemporalConvBlocks with exponentially increasing
    dilation factors for large receptive fields.

    Args:
        input_dim: Input feature dimension (from audio encoder d_model).
        channels: Number of TCN channels.
        num_blocks: Number of TCN residual blocks.
        kernel_size: Convolution kernel size.
        dilations: List of dilation factors. If None, uses powers of 2.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        input_dim: int = 512,
        channels: int = 128,
        num_blocks: int = 6,
        kernel_size: int = 3,
        dilations: list[int] | None = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        if dilations is None:
            dilations = [2**i for i in range(num_blocks)]

        # Project from d_model to TCN channels
        self.input_proj = nn.Conv1d(input_dim, channels, kernel_size=1)

        # Stack of TCN blocks
        self.blocks = nn.ModuleList([
            TemporalConvBlock(channels, kernel_size, dilation=d, dropout=dropout)
            for d in dilations
        ])

        # Project back to d_model
        self.output_proj = nn.Conv1d(channels, input_dim, kernel_size=1)

        # Receptive field: sum of (kernel_size - 1) * dilation for all blocks
        self.receptive_field = sum((kernel_size - 1) * d for d in dilations) + 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through TCN blocks.

        Args:
            x: Input tensor [B, T, D] (batch-first, time, features).

        Returns:
            Output tensor [B, T, D] (same shape as input).
        """
        # Transpose to [B, D, T] for Conv1d
        x = x.transpose(1, 2)

        x = self.input_proj(x)

        for block in self.blocks:
            x = block(x)

        x = self.output_proj(x)

        # Transpose back to [B, T, D]
        return x.transpose(1, 2)


class OnsetModel(nn.Module):
    """Hybrid TCN + Transformer onset prediction head for Stage 1.

    The TCN captures local temporal patterns (beat subdivisions, rhythmic
    patterns) with large receptive fields via dilated convolutions. The
    Transformer on top adds global context (verse/chorus awareness).

    This keeps the same interface as the original Transformer-only model
    for drop-in compatibility with OnsetLitModule and generate.py.

    Args:
        d_model: Input embedding dimension (from audio encoder).
        nhead: Number of attention heads for the Transformer layers.
        num_layers: Number of Transformer encoder layers on top of TCN.
        num_difficulties: Number of difficulty levels (5: Easy-ExpertPlus).
        num_genres: Number of genre classes.
        dropout: Dropout rate.
        tcn_channels: Number of channels in the TCN blocks.
        tcn_num_blocks: Number of TCN residual blocks.
        tcn_kernel_size: Kernel size for TCN convolutions.
        tcn_dilations: Explicit dilation factors. None = powers of 2.
        conditioning_dropout: Dropout probability for difficulty/genre embeddings
            during training (enables Classifier-Free Guidance at inference).
    """

    def __init__(
        self,
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 2,
        num_difficulties: int = 5,
        num_genres: int = 11,
        dropout: float = 0.1,
        use_checkpoint: bool = False,
        # TCN parameters
        tcn_channels: int = 128,
        tcn_num_blocks: int = 6,
        tcn_kernel_size: int = 3,
        tcn_dilations: list[int] | None = None,
        # Conditioning dropout for CFG
        conditioning_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.use_checkpoint = use_checkpoint
        self.conditioning_dropout = conditioning_dropout

        # Difficulty conditioning: learned embedding added to every frame
        self.difficulty_emb = nn.Embedding(num_difficulties, d_model)

        # Genre conditioning: learned embedding added to every frame
        self.genre_emb = nn.Embedding(num_genres, d_model)

        # TCN for local temporal pattern detection
        self.tcn = TCNEncoder(
            input_dim=d_model,
            channels=tcn_channels,
            num_blocks=tcn_num_blocks,
            kernel_size=tcn_kernel_size,
            dilations=tcn_dilations,
            dropout=dropout,
        )

        # Small transformer for global context on top of TCN features
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output head: LayerNorm -> Linear -> squeeze to [B, T]
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 1),
        )

    def forward(
        self,
        audio_features: torch.Tensor,
        difficulty: torch.Tensor,
        genre: torch.Tensor,
    ) -> torch.Tensor:
        """Predict onset logits per frame.

        Args:
            audio_features: Audio encoder output [B, T, d_model].
            difficulty: Difficulty index per sample [B].
            genre: Genre index per sample [B].

        Returns:
            Raw logits [B, T] (apply sigmoid for probabilities).
        """
        # Add difficulty and genre embeddings to every frame
        diff_emb = self.difficulty_emb(difficulty)  # [B, d_model]
        genre_emb = self.genre_emb(genre)           # [B, d_model]

        # Conditioning dropout: zero out embeddings with probability p during training
        # This enables Classifier-Free Guidance at inference
        if self.training and self.conditioning_dropout > 0:
            mask = torch.rand(diff_emb.shape[0], 1, device=diff_emb.device)
            drop_mask = (mask < self.conditioning_dropout).float()
            diff_emb = diff_emb * (1 - drop_mask)
            genre_emb = genre_emb * (1 - drop_mask)

        x = audio_features + diff_emb.unsqueeze(1) + genre_emb.unsqueeze(1)  # [B, T, d_model]

        # TCN for local temporal patterns
        x = self.tcn(x)

        # Transformer for global context
        if self.use_checkpoint:
            for layer in self.transformer.layers:
                x = torch.utils.checkpoint.checkpoint(layer, x, use_reentrant=False)
            if self.transformer.norm is not None:
                x = self.transformer.norm(x)
        else:
            x = self.transformer(x)

        # Output head
        logits = self.head(x).squeeze(-1)  # [B, T]
        return logits
