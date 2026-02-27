"""Beam search and nucleus sampling for Stage 2 note sequence generation.

Provides two decoding strategies:
    - beam_search_decode: Deterministic beam search with length normalization
    - nucleus_sampling_decode: Top-p sampling for creative diversity

Both support KV caching for ~10x faster inference when available.
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
    prev_tokens: torch.Tensor | None = None,
    min_length: int = 3,
) -> list[int]:
    """Run beam search decoding on the sequence model.

    Maintains beam_size hypotheses, expands by top-k tokens at each step,
    and scores by length-normalized log probability.

    Uses KV caching when available (model.decode_step_cached) for 10x
    faster inference. Falls back to non-cached decode_step otherwise.

    Args:
        model: The SequenceModel instance (must have decode_step method).
        audio_features: Encoded audio frame embeddings [1, T, d_model].
        difficulty: Difficulty index tensor [1].
        genre: Genre index tensor [1].
        beam_size: Number of beams to maintain.
        max_length: Maximum output token sequence length.
        temperature: Temperature for logit scaling (>1 = more diverse).
        prev_tokens: Optional previous onset tokens [1, K, S] for inter-onset context.
        min_length: Minimum tokens before EOS is allowed (prevents empty sequences).

    Returns:
        Best token sequence from beam search (without BOS).
    """
    device = audio_features.device
    use_cache = hasattr(model, "decode_step_cached") and hasattr(model, "new_caches")

    if use_cache:
        return _beam_search_cached(
            model, audio_features, difficulty, genre,
            beam_size, max_length, temperature,
            prev_tokens=prev_tokens, min_length=min_length,
        )

    # Fallback: non-cached beam search (original implementation)
    beams: list[tuple[float, list[int]]] = [(0.0, [BOS])]
    completed: list[tuple[float, list[int]]] = []

    for step_idx in range(max_length):
        candidates: list[tuple[float, list[int]]] = []

        for log_prob, tokens in beams:
            token_tensor = torch.tensor([tokens], dtype=torch.long, device=device)
            logits = model.decode_step(
                token_tensor, audio_features, difficulty, genre, prev_tokens=prev_tokens,
            )
            logits = logits.squeeze(0) / temperature

            # Suppress EOS before min_length
            if step_idx < min_length:
                logits[EOS] = float("-inf")

            log_probs = F.log_softmax(logits, dim=-1)

            topk_log_probs, topk_indices = log_probs.topk(beam_size)

            for k in range(beam_size):
                new_token = topk_indices[k].item()
                new_log_prob = log_prob + topk_log_probs[k].item()
                new_tokens = tokens + [new_token]

                if new_token == EOS:
                    seq_len = len(new_tokens) - 1
                    normalized_score = new_log_prob / max(1, seq_len)
                    completed.append((normalized_score, new_tokens))
                else:
                    candidates.append((new_log_prob, new_tokens))

        if not candidates:
            break

        candidates.sort(key=lambda x: x[0], reverse=True)
        beams = candidates[:beam_size]

        if len(completed) >= beam_size:
            break

    if not completed:
        completed = [(lp / max(1, len(t) - 1), t) for lp, t in beams]

    completed.sort(key=lambda x: x[0], reverse=True)
    best_tokens = completed[0][1]
    return [t for t in best_tokens if t != BOS and t != EOS]


def _beam_search_cached(
    model: object,
    audio_features: torch.Tensor,
    difficulty: torch.Tensor,
    genre: torch.Tensor,
    beam_size: int,
    max_length: int,
    temperature: float,
    prev_tokens: torch.Tensor | None = None,
    min_length: int = 3,
) -> list[int]:
    """KV-cached beam search implementation.

    Each beam maintains its own set of KV caches. When beams are pruned,
    their caches are preserved. This avoids recomputing attention from
    scratch at each step.
    """
    device = audio_features.device

    # Each beam: (log_prob, token_list, layer_caches)
    initial_caches = model.new_caches()
    beams: list[tuple[float, list[int], list]] = [(0.0, [BOS], initial_caches)]
    completed: list[tuple[float, list[int]]] = []

    # Process BOS token first to populate caches
    bos_tensor = torch.tensor([[BOS]], dtype=torch.long, device=device)
    logits = model.decode_step_cached(
        bos_tensor, audio_features, difficulty, genre,
        initial_caches, step=0, prev_tokens=prev_tokens,
    )
    logits = logits.squeeze(0) / temperature

    # Suppress EOS at step 0 (before min_length)
    if min_length > 0:
        logits[EOS] = float("-inf")

    log_probs = F.log_softmax(logits, dim=-1)

    # Expand BOS into beam_size initial beams
    topk_log_probs, topk_indices = log_probs.topk(beam_size)
    beams = []
    for k in range(beam_size):
        new_token = topk_indices[k].item()
        new_log_prob = topk_log_probs[k].item()

        if new_token == EOS:
            completed.append((new_log_prob, [BOS, new_token]))
            continue

        # Deep copy caches for this beam
        beam_caches = _clone_caches(initial_caches)
        beams.append((new_log_prob, [BOS, new_token], beam_caches))

    # Continue from step 1 onward
    for step in range(1, max_length):
        if not beams:
            break

        candidates: list[tuple[float, list[int], list]] = []

        for log_prob, tokens, caches in beams:
            last_token = torch.tensor([[tokens[-1]]], dtype=torch.long, device=device)
            logits = model.decode_step_cached(
                last_token, audio_features, difficulty, genre,
                caches, step=step, prev_tokens=prev_tokens,
            )
            logits = logits.squeeze(0) / temperature

            # Suppress EOS before min_length
            if step < min_length:
                logits[EOS] = float("-inf")

            step_log_probs = F.log_softmax(logits, dim=-1)

            topk_lp, topk_idx = step_log_probs.topk(beam_size)

            for k in range(beam_size):
                new_token = topk_idx[k].item()
                new_log_prob = log_prob + topk_lp[k].item()
                new_tokens = tokens + [new_token]

                if new_token == EOS:
                    seq_len = len(new_tokens) - 1
                    normalized_score = new_log_prob / max(1, seq_len)
                    completed.append((normalized_score, new_tokens))
                else:
                    # Reuse cache for the first candidate (k==0), clone for others
                    if k == 0:
                        candidates.append((new_log_prob, new_tokens, caches))
                    else:
                        candidates.append((new_log_prob, new_tokens, _clone_caches(caches)))

        if not candidates:
            break

        candidates.sort(key=lambda x: x[0], reverse=True)
        beams = candidates[:beam_size]

        if len(completed) >= beam_size:
            break

    if not completed:
        completed = [(lp / max(1, len(t) - 1), t) for lp, t, _caches in beams]

    completed.sort(key=lambda x: x[0], reverse=True)
    best_tokens = completed[0][1]
    return [t for t in best_tokens if t != BOS and t != EOS]


def _clone_caches(caches: list) -> list:
    """Deep clone KV caches for beam forking."""
    from beatsaber_automapper.models.components import KVCache, LayerCaches

    cloned = []
    for lc in caches:
        new_self = KVCache(
            k=lc.self_attn.k.clone() if lc.self_attn.k is not None else None,
            v=lc.self_attn.v.clone() if lc.self_attn.v is not None else None,
        )
        new_cross = KVCache(
            k=lc.cross_attn.k.clone() if lc.cross_attn.k is not None else None,
            v=lc.cross_attn.v.clone() if lc.cross_attn.v is not None else None,
        )
        cloned.append(LayerCaches(self_attn=new_self, cross_attn=new_cross))
    return cloned


def nucleus_sampling_decode(
    model: object,
    audio_features: torch.Tensor,
    difficulty: torch.Tensor,
    genre: torch.Tensor,
    max_length: int = 64,
    temperature: float = 1.0,
    top_p: float = 0.9,
    prev_tokens: torch.Tensor | None = None,
    min_length: int = 3,
) -> list[int]:
    """Run nucleus (top-p) sampling on the sequence model.

    Uses KV caching when available for faster inference.

    Args:
        model: The SequenceModel instance (must have decode_step method).
        audio_features: Encoded audio frame embeddings [1, T, d_model].
        difficulty: Difficulty index tensor [1].
        genre: Genre index tensor [1].
        max_length: Maximum output token sequence length.
        temperature: Temperature for logit scaling.
        top_p: Cumulative probability threshold for nucleus filtering.
        prev_tokens: Optional previous onset tokens [1, K, S] for inter-onset context.
        min_length: Minimum tokens before EOS is allowed.

    Returns:
        Sampled token sequence (without BOS or EOS).
    """
    device = audio_features.device
    use_cache = hasattr(model, "decode_step_cached") and hasattr(model, "new_caches")

    if use_cache:
        return _nucleus_cached(
            model, audio_features, difficulty, genre,
            max_length, temperature, top_p,
            prev_tokens=prev_tokens, min_length=min_length,
        )

    # Fallback: non-cached nucleus sampling
    tokens = [BOS]
    for step in range(max_length):
        token_tensor = torch.tensor([tokens], dtype=torch.long, device=device)
        logits = model.decode_step(
            token_tensor, audio_features, difficulty, genre, prev_tokens=prev_tokens,
        )
        logits = logits.squeeze(0) / temperature

        # Suppress EOS before min_length
        if step < min_length:
            logits[EOS] = float("-inf")

        probs = F.softmax(logits, dim=-1)
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

        mask = cumulative_probs - sorted_probs > top_p
        sorted_probs[mask] = 0.0
        sorted_probs = sorted_probs / sorted_probs.sum()

        idx = torch.multinomial(sorted_probs, num_samples=1).item()
        next_token = sorted_indices[idx].item()

        if next_token == EOS:
            break
        tokens.append(next_token)

    return tokens[1:]


def _nucleus_cached(
    model: object,
    audio_features: torch.Tensor,
    difficulty: torch.Tensor,
    genre: torch.Tensor,
    max_length: int,
    temperature: float,
    top_p: float,
    prev_tokens: torch.Tensor | None = None,
    min_length: int = 3,
) -> list[int]:
    """KV-cached nucleus sampling."""
    device = audio_features.device
    caches = model.new_caches()
    tokens = [BOS]

    for step in range(max_length):
        token_tensor = torch.tensor([[tokens[-1]]], dtype=torch.long, device=device)
        logits = model.decode_step_cached(
            token_tensor, audio_features, difficulty, genre,
            caches, step=step, prev_tokens=prev_tokens,
        )
        logits = logits.squeeze(0) / temperature

        # Suppress EOS before min_length
        if step < min_length:
            logits[EOS] = float("-inf")

        probs = F.softmax(logits, dim=-1)
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

        mask = cumulative_probs - sorted_probs > top_p
        sorted_probs[mask] = 0.0
        sorted_probs = sorted_probs / sorted_probs.sum()

        idx = torch.multinomial(sorted_probs, num_samples=1).item()
        next_token = sorted_indices[idx].item()

        if next_token == EOS:
            break
        tokens.append(next_token)

    return tokens[1:]
