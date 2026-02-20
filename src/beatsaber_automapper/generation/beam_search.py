"""Beam search and nucleus sampling for Stage 2 note sequence generation.

Provides two decoding strategies:
    - beam_search_decode: Deterministic beam search with length normalization
    - nucleus_sampling_decode: Top-p sampling for creative diversity
"""

from __future__ import annotations

import logging

import torch
from torch.nn import functional as F  # noqa: N812

from beatsaber_automapper.data.tokenizer import BOS, EOS

logger = logging.getLogger(__name__)


def beam_search_decode(
    model: object,
    audio_features: torch.Tensor,
    difficulty: torch.Tensor,
    genre: torch.Tensor,
    beam_size: int = 8,
    max_length: int = 64,
    temperature: float = 1.0,
) -> list[int]:
    """Run beam search decoding on the sequence model.

    Maintains beam_size hypotheses, expands by top-k tokens at each step,
    and scores by length-normalized log probability.

    Args:
        model: The SequenceModel instance (must have decode_step method).
        audio_features: Encoded audio frame embeddings [1, T, d_model].
        difficulty: Difficulty index tensor [1].
        genre: Genre index tensor [1].
        beam_size: Number of beams to maintain.
        max_length: Maximum output token sequence length.
        temperature: Temperature for logit scaling (>1 = more diverse).

    Returns:
        Best token sequence from beam search (without BOS).
    """
    device = audio_features.device

    # Each beam: (log_prob, token_list)
    beams: list[tuple[float, list[int]]] = [(0.0, [BOS])]
    completed: list[tuple[float, list[int]]] = []

    for _ in range(max_length):
        candidates: list[tuple[float, list[int]]] = []

        for log_prob, tokens in beams:
            token_tensor = torch.tensor([tokens], dtype=torch.long, device=device)
            logits = model.decode_step(token_tensor, audio_features, difficulty, genre)  # [1, V]
            logits = logits.squeeze(0) / temperature  # [V]
            log_probs = F.log_softmax(logits, dim=-1)

            # Take top beam_size tokens
            topk_log_probs, topk_indices = log_probs.topk(beam_size)

            for k in range(beam_size):
                new_token = topk_indices[k].item()
                new_log_prob = log_prob + topk_log_probs[k].item()
                new_tokens = tokens + [new_token]

                if new_token == EOS:
                    # Length-normalized score (exclude BOS from length)
                    seq_len = len(new_tokens) - 1  # exclude BOS
                    normalized_score = new_log_prob / max(1, seq_len)
                    completed.append((normalized_score, new_tokens))
                else:
                    candidates.append((new_log_prob, new_tokens))

        if not candidates:
            break

        # Prune to beam_size best candidates
        candidates.sort(key=lambda x: x[0], reverse=True)
        beams = candidates[:beam_size]

        # Early stop if we have enough completed
        if len(completed) >= beam_size:
            break

    # If no completed beams, use best incomplete
    if not completed:
        completed = [(lp / max(1, len(t) - 1), t) for lp, t in beams]

    # Return best sequence, stripping BOS
    completed.sort(key=lambda x: x[0], reverse=True)
    best_tokens = completed[0][1]

    # Strip BOS and EOS
    result = [t for t in best_tokens if t != BOS and t != EOS]
    return result


def nucleus_sampling_decode(
    model: object,
    audio_features: torch.Tensor,
    difficulty: torch.Tensor,
    genre: torch.Tensor,
    max_length: int = 64,
    temperature: float = 1.0,
    top_p: float = 0.9,
) -> list[int]:
    """Run nucleus (top-p) sampling on the sequence model.

    Samples from the smallest set of tokens whose cumulative probability
    exceeds top_p. Better diversity for creative generation.

    Args:
        model: The SequenceModel instance (must have decode_step method).
        audio_features: Encoded audio frame embeddings [1, T, d_model].
        difficulty: Difficulty index tensor [1].
        genre: Genre index tensor [1].
        max_length: Maximum output token sequence length.
        temperature: Temperature for logit scaling.
        top_p: Cumulative probability threshold for nucleus filtering.

    Returns:
        Sampled token sequence (without BOS or EOS).
    """
    device = audio_features.device
    tokens = [BOS]

    for _ in range(max_length):
        token_tensor = torch.tensor([tokens], dtype=torch.long, device=device)
        logits = model.decode_step(token_tensor, audio_features, difficulty, genre)  # [1, V]
        logits = logits.squeeze(0) / temperature  # [V]

        # Sort by descending probability
        probs = F.softmax(logits, dim=-1)
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

        # Find cutoff: keep tokens with cumulative prob <= top_p (plus one more)
        mask = cumulative_probs - sorted_probs > top_p
        sorted_probs[mask] = 0.0

        # Re-normalize
        sorted_probs = sorted_probs / sorted_probs.sum()

        # Sample
        idx = torch.multinomial(sorted_probs, num_samples=1).item()
        next_token = sorted_indices[idx].item()

        if next_token == EOS:
            break

        tokens.append(next_token)

    # Strip BOS
    return tokens[1:]
